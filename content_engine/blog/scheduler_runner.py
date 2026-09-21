"""Bounded, foreground-only runner for one SahilBlog pipeline attempt.

The scheduler layer owns only local coordination: a non-blocking single-flight
lock, an explicit deadline handed to a child seam, and durable JSON records.
It never schedules work or calls delivery providers itself.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

MAX_NEW_DRAFTS = 1
MAX_IMAGES = 3
DEFAULT_TIMEOUT_SECONDS = 20 * 60

ChildRunner = Callable[..., dict[str, Any]]


def terminal_json(result: dict[str, Any]) -> str:
    """Return one stable, compact JSON terminal line."""
    return json.dumps(result, sort_keys=True, separators=(",", ":"), default=str)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(terminal_json(payload) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_bounds(max_new_drafts: int, max_images: int) -> None:
    if not 0 <= max_new_drafts <= MAX_NEW_DRAFTS:
        raise ValueError(f"max_new_drafts must be between 0 and {MAX_NEW_DRAFTS}")
    if not 0 <= max_images <= MAX_IMAGES:
        raise ValueError(f"max_images must be between 0 and {MAX_IMAGES}")


DANGEROUS_MODES = frozenset({"publish", "approval", "commit", "push", "delivery"})


def run_production_child(
    *,
    stream: str,
    repo: Path | str,
    deadline: float,
    max_new_drafts: int,
    max_images: int,
    dry_run: bool = False,
    **modes: Any,
) -> dict[str, Any]:
    """Run the only production child seam: bounded, uncommitted staging.

    The explicit import prevents this runner from acquiring any publisher or
    delivery dependency.  Unknown mode switches are rejected rather than being
    forwarded to the pipeline.
    """
    forbidden = sorted(name for name, enabled in modes.items() if enabled and name in DANGEROUS_MODES)
    if forbidden:
        raise ValueError(f"mode(s) not permitted by stage-only runner: {', '.join(forbidden)}")
    if modes:
        raise ValueError(f"unknown stage-only mode(s): {', '.join(sorted(modes))}")
    if max_new_drafts != MAX_NEW_DRAFTS or max_images != MAX_IMAGES:
        raise ValueError("production child requires exactly one draft and three images")
    if time.monotonic() >= deadline:
        raise TimeoutError("deadline exceeded before pipeline started")

    from blog.blog_pipeline import run_stage_draft_only

    payload = run_stage_draft_only(
        stream=stream,
        repo=str(repo),
        max_new_drafts=MAX_NEW_DRAFTS,
        max_images=MAX_IMAGES,
        dry_run=dry_run,
    )
    return {
        "operation": "stage_draft_only",
        "stage_only": True,
        "max_new_drafts": MAX_NEW_DRAFTS,
        "max_images": MAX_IMAGES,
        "pipeline": payload,
    }


def run_once(
    *,
    state_root: Path | str,
    child_runner: ChildRunner,
    max_new_drafts: int = MAX_NEW_DRAFTS,
    max_images: int = MAX_IMAGES,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run one bounded child attempt and always leave terminal local state.

    ``child_runner`` receives ``deadline`` (from ``time.monotonic``) and both
    validated caps. It must return JSON-compatible data. The runner never
    retries or schedules: a caller chooses whether to invoke another attempt.
    """
    _validate_bounds(max_new_drafts, max_images)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    root = Path(state_root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "runner.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "locked", "run_id": None}

        run_id = uuid.uuid4().hex
        started_at = time.time()
        deadline = time.monotonic() + timeout_seconds
        state_path = root / "state.json"
        record_path = root / "runs" / f"{run_id}.json"
        _atomic_json(state_path, {"active_run_id": run_id, "last_run_id": None})
        _atomic_json(record_path, {"run_id": run_id, "status": "running", "started_at": started_at})

        result: dict[str, Any]
        try:
            payload = child_runner(
                deadline=deadline,
                max_new_drafts=max_new_drafts,
                max_images=max_images,
            )
            if time.monotonic() > deadline:
                raise TimeoutError("child returned after deadline")
            result = {"status": "completed", "run_id": run_id, "result": payload}
        except TimeoutError as exc:
            result = {"status": "timed_out", "run_id": run_id, "error": str(exc)}
        except Exception as exc:  # terminal state is more useful than a daemon traceback
            result = {"status": "failed", "run_id": run_id, "error": str(exc)}
        finally:
            # A crashed/timed-out child must never leave a permanent active run.
            _atomic_json(state_path, {"active_run_id": None, "last_run_id": run_id})

        _atomic_json(record_path, {**result, "started_at": started_at, "finished_at": time.time()})
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return result


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Run one bounded, stage-only SahilBlog attempt")
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True, help="Existing SahilBlog worktree")
    parser.add_argument("--stream", default="ai", choices=["ai", "pm", "builder"])
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    execution = parser.add_mutually_exclusive_group(required=True)
    execution.add_argument(
        "--controlled-dry-run",
        action="store_true",
        help="Validate state/repository paths and refuse all provider execution",
    )
    execution.add_argument(
        "--operator-execute-providers",
        action="store_true",
        help="Operator-only: execute the bounded stage-only provider seam",
    )
    args = parser.parse_args()
    if not args.repo.is_dir():
        parser.error(f"--repo must be an existing directory: {args.repo}")

    result = run_once(
        state_root=args.state_root,
        child_runner=lambda **kwargs: run_production_child(
            stream=args.stream,
            repo=args.repo,
            dry_run=args.controlled_dry_run,
            **kwargs,
        ),
        # The scheduler CLI has no override switches: these are hard bounds.
        max_new_drafts=MAX_NEW_DRAFTS,
        max_images=MAX_IMAGES,
        timeout_seconds=args.timeout_seconds,
    )
    print(terminal_json(result))
    return 0 if result["status"] in {"completed", "locked"} else 1


if __name__ == "__main__":
    sys.exit(_cli())
