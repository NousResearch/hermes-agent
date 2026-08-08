#!/usr/bin/env python3
"""Sequential runner for guarded Codex implementation slices."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


_SLICE_ID_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")
_CONTINUE_POLICIES = {"stop-on-review-needed", "continue-on-passed"}
_DIRTY_BASELINE_POLICIES = {"require-clean", "allow-listed-owned", "fail-on-overlap"}
_VERIFY_CMD_IDS = {"none", "diff-check"}
_STOP_STATUSES = {"passed", "review_needed", "takeover_candidate", "blocked_by_allowlist", "failed", "unusable"}
_MAX_STRING = 4000
_MAX_LIST = 200
_MAX_DICT_KEYS = 80
_MAX_IMPL_GUARD_STDOUT_BYTES = 200_000


def _bounded(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= _MAX_STRING else value[: _MAX_STRING - 20] + "...[truncated]"
    if isinstance(value, list):
        items = [_bounded(item) for item in value[:_MAX_LIST]]
        if len(value) > _MAX_LIST:
            items.append(f"...[{len(value) - _MAX_LIST} more]")
        return items
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _MAX_DICT_KEYS:
                out["...[truncated_keys]"] = len(value) - _MAX_DICT_KEYS
                break
            out[str(key)] = _bounded(item)
        return out
    return value


def _emit(result: dict[str, Any]) -> int:
    print(json.dumps(_bounded(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") == "completed" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run guarded Codex implementation slices from a JSON plan.")
    parser.add_argument("--plan-file")
    parser.add_argument("--impl-guard-file", default=str(Path(__file__).with_name("codex_impl_guard.py")))
    parser.add_argument("--raw-dir")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser


def _git_root(repo: Path) -> Path:
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise ValueError("repo_not_git_repository")
    return Path(proc.stdout.strip()).resolve()


def _strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _path_parts(path: str) -> list[str]:
    return [part for part in path.replace("\\", "/").split("/") if part]


def _valid_repo_relative_pattern(pattern: str) -> bool:
    if not pattern or Path(pattern).is_absolute():
        return False
    parts = _path_parts(pattern)
    if not parts or any(part == ".." for part in parts):
        return False
    return not pattern.startswith("../")


def _invalid_allowlist_patterns(files: list[str], globs: list[str]) -> list[str]:
    return [pattern for pattern in [*files, *globs] if not _valid_repo_relative_pattern(pattern)]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_prompt_file(path: Path, *, repo: Path, plan_dir: Path) -> tuple[Path | None, str | None]:
    resolved = path.resolve(strict=False)
    if resolved.is_symlink():
        return None, "prompt_file_symlink"
    if not resolved.is_file():
        return None, "prompt_file_not_found"
    if not (_is_relative_to(resolved, repo) or _is_relative_to(resolved, plan_dir)):
        return None, "prompt_file_outside_allowed_roots"
    if resolved.stat().st_size > 200_000:
        return None, "prompt_file_too_large"
    return resolved, None


def _safe_raw_dir(path: Path, *, repo: Path) -> tuple[Path | None, str | None]:
    resolved = path.resolve(strict=False)
    if _is_relative_to(resolved, repo):
        return None, "unsafe_raw_dir"
    if resolved.is_symlink() or (resolved.exists() and not resolved.is_dir()):
        return None, "unsafe_raw_dir"
    parent = resolved.parent.resolve(strict=False)
    if _is_relative_to(parent, repo):
        return None, "unsafe_raw_dir"
    return resolved, None


def _load_plan(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "missing_plan"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(loaded, dict):
        return None, "plan_not_object"
    return loaded, None


def _validate_plan(plan: dict[str, Any], *, plan_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    repo_value = plan.get("repo")
    if not isinstance(repo_value, str) or not repo_value:
        return None, "invalid_repo"
    repo = Path(repo_value).expanduser().resolve()
    if not repo.exists() or not repo.is_dir():
        return None, "repo_not_found"
    try:
        repo = _git_root(repo)
    except ValueError as exc:
        return None, str(exc)

    continue_policy = plan.get("continue_policy", "stop-on-review-needed")
    if continue_policy not in _CONTINUE_POLICIES:
        return None, "invalid_continue_policy"
    dirty_baseline_policy = plan.get("dirty_baseline_policy", "require-clean")
    if dirty_baseline_policy not in _DIRTY_BASELINE_POLICIES:
        return None, "invalid_dirty_baseline_policy"

    slices = plan.get("slices")
    if not isinstance(slices, list) or not slices:
        return None, "empty_slices"

    seen: set[str] = set()
    normalized_slices: list[dict[str, Any]] = []
    for item in slices:
        if not isinstance(item, dict):
            return None, "invalid_slice"
        slice_id = item.get("id")
        if not isinstance(slice_id, str) or not _SLICE_ID_RE.fullmatch(slice_id):
            return None, "invalid_slice_id"
        if slice_id in seen:
            return None, "duplicate_slice_id"
        seen.add(slice_id)

        prompt_file_value = item.get("prompt_file")
        if not isinstance(prompt_file_value, str) or not prompt_file_value:
            return None, "invalid_prompt_file"
        prompt_file = Path(prompt_file_value).expanduser()
        if not prompt_file.is_absolute():
            prompt_file = plan_dir / prompt_file
        prompt_file, prompt_error = _safe_prompt_file(prompt_file, repo=repo, plan_dir=plan_dir)
        if prompt_error:
            return None, prompt_error
        assert prompt_file is not None

        allowed_files = item.get("allowed_files", [])
        allowed_globs = item.get("allowed_globs", [])
        verify_cmd_ids = item.get("verify_cmd_ids", [])
        if not _strings(allowed_files):
            return None, "invalid_allowed_files"
        if not _strings(allowed_globs):
            return None, "invalid_allowed_globs"
        invalid_allowlist = _invalid_allowlist_patterns(allowed_files, allowed_globs)
        if invalid_allowlist:
            return None, "invalid_allowlist"
        if not _strings(verify_cmd_ids):
            return None, "invalid_verify_cmd_ids"
        if any(verify_id not in _VERIFY_CMD_IDS for verify_id in verify_cmd_ids):
            return None, "unsupported_verify_cmd_id"

        slice_dirty_policy = item.get("dirty_baseline_policy", dirty_baseline_policy)
        if slice_dirty_policy not in _DIRTY_BASELINE_POLICIES:
            return None, "invalid_dirty_baseline_policy"

        normalized_slices.append({
            "id": slice_id,
            "prompt_file": str(prompt_file),
            "allowed_files": list(allowed_files),
            "allowed_globs": list(allowed_globs),
            "verify_cmd_ids": list(verify_cmd_ids),
            "dirty_baseline_policy": slice_dirty_policy,
        })

    return {
        "repo": str(repo),
        "continue_policy": continue_policy,
        "dirty_baseline_policy": dirty_baseline_policy,
        "slices": normalized_slices,
    }, None


def _guard_argv(
    impl_guard_file: Path,
    repo: str,
    item: dict[str, Any],
    *,
    raw_dir: Path | None,
    timeout_seconds: float,
) -> list[str]:
    argv = [
        sys.executable,
        str(impl_guard_file),
        "--workdir",
        repo,
        "--prompt-file",
        item["prompt_file"],
        "--dirty-baseline-policy",
        item["dirty_baseline_policy"],
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    if raw_dir is not None:
        argv.extend(["--raw-log", str(raw_dir / f"{item['id']}.raw.log")])
        argv.extend(["--final-file", str(raw_dir / f"{item['id']}.final.json")])
    for allowed_file in item["allowed_files"]:
        argv.extend(["--allowed-file", allowed_file])
    for allowed_glob in item["allowed_globs"]:
        argv.extend(["--allowed-glob", allowed_glob])
    for verify_id in item["verify_cmd_ids"]:
        argv.extend(["--verify-cmd-id", verify_id])
    return argv


def _parse_guard_stdout(stdout: str) -> dict[str, Any]:
    try:
        loaded = json.loads(stdout)
    except json.JSONDecodeError:
        return {"status": "unusable", "reason": "impl_guard_invalid_json"}
    if not isinstance(loaded, dict):
        return {"status": "unusable", "reason": "impl_guard_non_object"}
    return loaded


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _terminate_process_group(proc: subprocess.Popen[Any]) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


def _run_impl_guard(argv: list[str], *, cwd: str, timeout_seconds: float) -> tuple[dict[str, Any], int | None]:
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=(os.name == "posix"),
    )
    selector = selectors.DefaultSelector()
    stdout_chunks: list[bytes] = []
    stdout_size = 0
    stderr_size = 0
    assert proc.stdout is not None
    assert proc.stderr is not None
    selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
    selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
    deadline = time.monotonic() + max(1.0, timeout_seconds + 10.0)
    try:
        while selector.get_map():
            if time.monotonic() > deadline:
                _terminate_process_group(proc)
                return {"status": "unusable", "reason": "impl_guard_timeout"}, None
            events = selector.select(timeout=0.05)
            if not events and proc.poll() is not None:
                # Give pipes one more drain cycle.
                events = selector.select(timeout=0)
                if not events:
                    break
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 8192)
                if not chunk:
                    try:
                        selector.unregister(key.fileobj)
                    except Exception:
                        pass
                    continue
                if key.data == "stdout":
                    stdout_size += len(chunk)
                    if stdout_size > _MAX_IMPL_GUARD_STDOUT_BYTES:
                        _terminate_process_group(proc)
                        return {
                            "status": "unusable",
                            "reason": "impl_guard_stdout_limit_exceeded",
                            "stdout_bytes_previewed": _MAX_IMPL_GUARD_STDOUT_BYTES,
                        }, proc.returncode
                    stdout_chunks.append(chunk)
                else:
                    stderr_size += len(chunk)
                    if stderr_size > _MAX_IMPL_GUARD_STDOUT_BYTES:
                        _terminate_process_group(proc)
                        return {
                            "status": "unusable",
                            "reason": "impl_guard_stderr_limit_exceeded",
                            "stderr_bytes_previewed": _MAX_IMPL_GUARD_STDOUT_BYTES,
                        }, proc.returncode
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            _terminate_process_group(proc)
            return {"status": "unusable", "reason": "impl_guard_timeout"}, None
    finally:
        try:
            selector.close()
        except Exception:
            pass
    stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
    return _parse_guard_stdout(stdout), proc.returncode


def _slice_result(slice_id: str, guard_result: dict[str, Any], returncode: int | None) -> dict[str, Any]:
    return {
        "id": slice_id,
        "status": guard_result.get("status", "unusable"),
        "reason": guard_result.get("reason", "missing_reason"),
        "returncode": returncode,
        "raw_log_path": guard_result.get("raw_log_path"),
        "final_file": guard_result.get("final_file"),
        "impl_guard_result": guard_result,
    }


def _should_continue(policy: str, guard_result: dict[str, Any]) -> bool:
    if policy != "continue-on-passed":
        return False
    if guard_result.get("status") != "passed":
        return False
    verification = guard_result.get("verification")
    if not isinstance(verification, list) or not verification:
        return False
    return all(isinstance(item, dict) and item.get("status") == "passed" for item in verification)


def run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.plan_file:
        return _emit({
            "status": "unusable",
            "reason": "missing_plan",
            "repo": None,
            "continue_policy": "stop-on-review-needed",
            "completed_slices": [],
            "stopped_slice": None,
            "slice_results": [],
            "recommended_next_action": "Provide --plan-file pointing to a JSON stage plan.",
        })

    plan_path = Path(args.plan_file).expanduser()
    plan, load_error = _load_plan(plan_path)
    if load_error:
        return _emit({
            "status": "unusable",
            "reason": load_error,
            "repo": None,
            "continue_policy": "stop-on-review-needed",
            "completed_slices": [],
            "stopped_slice": None,
            "slice_results": [],
            "recommended_next_action": "Fix the stage plan before running slices.",
        })
    assert plan is not None
    normalized, validation_error = _validate_plan(plan, plan_dir=plan_path.resolve().parent)
    if validation_error:
        return _emit({
            "status": "unusable",
            "reason": validation_error,
            "repo": plan.get("repo"),
            "continue_policy": plan.get("continue_policy", "stop-on-review-needed"),
            "completed_slices": [],
            "stopped_slice": None,
            "slice_results": [],
            "recommended_next_action": "Fix the stage plan before running slices.",
        })
    assert normalized is not None

    impl_guard_file = Path(args.impl_guard_file).resolve()
    if not impl_guard_file.is_file():
        return _emit({
            "status": "unusable",
            "reason": "impl_guard_file_missing",
            "repo": normalized["repo"],
            "continue_policy": normalized["continue_policy"],
            "completed_slices": [],
            "stopped_slice": None,
            "slice_results": [],
            "recommended_next_action": "Fix --impl-guard-file before running slices.",
        })

    impl_guard_digest = _file_digest(impl_guard_file)

    repo = normalized["repo"]
    repo_path = Path(repo).resolve()
    raw_dir = None
    if args.raw_dir:
        raw_dir, raw_error = _safe_raw_dir(Path(args.raw_dir).expanduser(), repo=repo_path)
        if raw_error:
            return _emit({
                "status": "unusable",
                "reason": raw_error,
                "repo": repo,
                "continue_policy": normalized["continue_policy"],
                "completed_slices": [],
                "stopped_slice": None,
                "slice_results": [],
                "recommended_next_action": "Choose a --raw-dir outside the repo.",
            })
        assert raw_dir is not None
        raw_dir.mkdir(parents=True, exist_ok=True)

    continue_policy = normalized["continue_policy"]
    completed_slices: list[str] = []
    slice_results: list[dict[str, Any]] = []
    stopped_slice: str | None = None
    reason = "all_slices_completed"
    status = "completed"

    for item in normalized["slices"]:
        guard_result, returncode = _run_impl_guard(
            _guard_argv(
                impl_guard_file,
                repo,
                item,
                raw_dir=raw_dir,
                timeout_seconds=args.timeout_seconds,
            ),
            cwd=repo,
            timeout_seconds=args.timeout_seconds,
        )
        if _file_digest(impl_guard_file) != impl_guard_digest:
            guard_result = {
                **guard_result,
                "status": "unusable",
                "reason": "impl_guard_file_changed",
                "impl_guard_reported_status": guard_result.get("status"),
            }
        elif returncode not in (0, None) and guard_result.get("status") in {"passed", "review_needed"}:
            guard_result = {
                **guard_result,
                "status": "unusable",
                "reason": "impl_guard_nonzero_exit",
                "impl_guard_reported_status": guard_result.get("status"),
            }
        result = _slice_result(item["id"], guard_result, returncode)
        slice_results.append(result)
        slice_status = str(result["status"])
        if _should_continue(continue_policy, guard_result):
            completed_slices.append(item["id"])
            continue
        stopped_slice = item["id"]
        status = "stopped"
        reason = f"slice_{slice_status}" if slice_status in _STOP_STATUSES else "unknown_slice_status"
        break

    return _emit({
        "status": status,
        "reason": reason,
        "repo": repo,
        "continue_policy": continue_policy,
        "completed_slices": completed_slices,
        "stopped_slice": stopped_slice,
        "slice_results": slice_results,
        "recommended_next_action": "Run read-only review for stopped slice." if reason == "slice_review_needed" else "Inspect stopped slice before continuing." if status == "stopped" else "All requested slices completed.",
    })


if __name__ == "__main__":
    raise SystemExit(run())
