#!/usr/bin/env python3
"""Run the durable Ares profile panel with bounded, replayable receipts.

This is intentionally a user-local skill support script, not a core Hermes
model tool. It owns orchestration mechanics only; profile outputs remain
advisory until the controller verifies the receipt and current source state.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any


PROFILES = (
    "public",
    "explorer",
    "job-scout",
    "longmemeval-bench",
    "statistician",
    "ml-evaluation-researcher",
    "cognitive-scientist",
    "psychometrician",
    "inbox-manager",
)

DEFAULT_PROFILE_TIMEOUT_SECONDS = 180.0
DEFAULT_PANEL_TIMEOUT_SECONDS = 600.0


class ProcessRegistry:
    """Thread-safe ownership of live profile process groups."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen[bytes]] = {}

    def add(self, profile: str, process: subprocess.Popen[bytes]) -> None:
        with self._lock:
            self._processes[profile] = process

    def discard(self, profile: str, process: subprocess.Popen[bytes]) -> None:
        with self._lock:
            if self._processes.get(profile) is process:
                self._processes.pop(profile, None)

    def snapshot(self) -> dict[str, subprocess.Popen[bytes]]:
        with self._lock:
            return dict(self._processes)


def atomic_write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Replace a receipt atomically so interruption leaves valid JSON behind."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def ordered_results(
    results: dict[str, dict[str, Any]],
    profiles: tuple[str, ...] = PROFILES,
) -> list[dict[str, Any]]:
    """Project available results into the admitted canonical profile order."""
    return [results[profile] for profile in profiles if profile in results]


def select_profiles(raw: str | None, full_panel: bool) -> tuple[str, ...]:
    """Resolve an explicit minimal specialist set in canonical order."""
    if full_panel:
        return PROFILES
    requested = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if not requested:
        raise ValueError("select --profiles or explicitly request --full-panel")
    if len(requested) != len(set(requested)):
        raise ValueError("profiles must not contain duplicates")
    unknown = sorted(set(requested) - set(PROFILES))
    if unknown:
        raise ValueError("unknown profiles: " + ", ".join(unknown))
    requested_set = set(requested)
    return tuple(profile for profile in PROFILES if profile in requested_set)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


MAX_CAPTURE_BYTES = 512 * 1024
_SECRET_ENV_MARKERS = (
    "API_KEY",
    "ACCESS_TOKEN",
    "AUTH_TOKEN",
    "CLIENT_SECRET",
    "COOKIE",
    "CREDENTIAL",
    "PASSWORD",
    "PRIVATE_KEY",
    "REFRESH_TOKEN",
    "SECRET",
    "SESSION_TOKEN",
)
_TOKEN_PATTERNS = (
    re.compile(r"(?i)\b(Bearer)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\b(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{8,})\b"),
    re.compile(
        r"(?im)(\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|password|secret|authorization|cookie)\b\s*[:=]\s*)([^\s,;]+)"
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def secret_environment_values(environment: dict[str, str]) -> tuple[str, ...]:
    """Return secret-like environment values without recording their names or contents."""
    values = {
        value
        for name, value in environment.items()
        if value and len(value) >= 8 and any(marker in name.upper() for marker in _SECRET_ENV_MARKERS)
    }
    return tuple(sorted(values, key=len, reverse=True))


def redact_text(text: str, secret_values: tuple[str, ...] = ()) -> str:
    """Redact known environment secrets and common token-shaped output."""
    redacted = text
    for value in secret_values:
        redacted = redacted.replace(value, "[REDACTED]")
    redacted = _TOKEN_PATTERNS[0].sub(r"\1 [REDACTED]", redacted)
    redacted = _TOKEN_PATTERNS[1].sub("[REDACTED]", redacted)
    redacted = _TOKEN_PATTERNS[2].sub(r"\1[REDACTED]", redacted)
    return redacted


def write_capture(path: Path, raw: bytes, secret_values: tuple[str, ...]) -> tuple[int, str, bool]:
    """Write bounded, UTF-8, redacted output and return size, digest, truncation."""
    text = raw.decode("utf-8", errors="replace")
    encoded = redact_text(text, secret_values).encode("utf-8")
    truncated = len(encoded) > MAX_CAPTURE_BYTES
    if truncated:
        marker = b"\n[profile output truncated at 512 KiB]\n"
        encoded = encoded[: MAX_CAPTURE_BYTES - len(marker)] + marker
    path.write_bytes(encoded)
    return len(encoded), hashlib.sha256(encoded).hexdigest(), truncated


def runtime_revision(runtime: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(runtime), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def terminate_process_groups(processes: list[subprocess.Popen[bytes]]) -> None:
    """TERM every group, then KILL survivors after one shared grace period."""
    live = [process for process in processes if process.poll() is None]
    for process in live:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + 5.0
    for process in live:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in live:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
    for process in live:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def archive_automation_session(
    *, runtime: Path, profile_home: Path, session_id: str
) -> dict[str, str]:
    """Archive one known panel session even when its child was terminated.

    A timed-out one-shot is killed before its own finally block can reach the
    opt-in archive hook. The controller owns the exact generated session id, so
    it can make the same reversible projection change after child exit without
    broad title/source matching.
    """
    program = (
        "import json\n"
        "from hermes_state import SessionDB\n"
        "db = SessionDB()\n"
        "try:\n"
        "    found = db.get_session(__import__('os').environ['PANEL_SESSION_ID']) is not None\n"
        "    archived = bool(db.set_session_archived(__import__('os').environ['PANEL_SESSION_ID'], True)) if found else False\n"
        "    print(json.dumps({'found': found, 'archived': archived}))\n"
        "finally:\n"
        "    db.close()\n"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "HERMES_HOME": str(profile_home),
            "PANEL_SESSION_ID": session_id,
            "PYTHONPATH": str(runtime),
        }
    )
    environment.pop("HERMES_SESSION_SOURCE", None)
    try:
        completed = subprocess.run(
            [str(runtime / ".venv" / "bin" / "python"), "-c", program],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            return {"outcome": "failed", "detail": "archive helper exited nonzero"}
        payload = json.loads(completed.stdout)
        if payload.get("archived") is True:
            return {"outcome": "archived", "detail": "exact panel session archived"}
        if payload.get("found") is False:
            return {"outcome": "not_created", "detail": "panel session row was not created"}
        return {"outcome": "failed", "detail": "archive helper did not confirm archival"}
    except Exception as exc:
        return {"outcome": "failed", "detail": type(exc).__name__}


def run_one(
    *,
    profile: str,
    brief: str,
    runtime: Path,
    workspace: Path,
    receipt_dir: Path,
    timeout: float,
    registry: ProcessRegistry | None = None,
    panel_cancelled: threading.Event | None = None,
) -> dict[str, Any]:
    profile_home = Path.home() / ".ares" / "profiles" / profile
    session_id = f"profile-panel-{receipt_dir.name}-{profile}"
    profile_dir = receipt_dir / "profiles"
    stdout_path = profile_dir / f"{profile}.stdout.txt"
    stderr_path = profile_dir / f"{profile}.stderr.txt"
    python = runtime / ".venv" / "bin" / "python"
    command = [
        str(python),
        "-m",
        "hermes_cli.main",
        "--in",
        str(workspace),
        "--reasoning",
        "low",
        "-z",
        brief,
    ]
    environment = os.environ.copy()
    # A controller launched from Hermes TUI inherits this renderer-routing
    # hint. A panel child must remain a real `-z` oneshot so it reaches the
    # archival hook instead of entering TUI session creation and polluting the
    # specialist's visible Sessions projection.
    environment.pop("HERMES_TUI", None)
    environment.update(
        {
            "HERMES_HOME": str(profile_home),
            # A panel child is a headless one-shot, not a continuation of the
            # controller's visible TUI/desktop conversation. This explicit
            # source prevents inherited session context from stamping rows as
            # visible interactive TUI sessions before archival runs.
            "HERMES_SESSION_SOURCE": "cli",
            "HERMES_ONESHOT_SESSION_ID": session_id,
            "HERMES_ONESHOT_ARCHIVE_SESSION": "1",
            "ARES_MANAGED_RUNTIME": "1",
            "PYTHONPATH": str(runtime),
        }
    )
    started_at = utc_now()
    started = time.monotonic()
    timed_out = False
    return_code: int | None = None
    profile_dir.mkdir(parents=True, exist_ok=True)
    secret_values = secret_environment_values(environment)
    if panel_cancelled is not None and panel_cancelled.is_set():
        raise concurrent.futures.CancelledError("panel deadline reached before profile start")
    process = subprocess.Popen(
        command,
        cwd=workspace,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    if registry is not None:
        registry.add(profile, process)
    try:
        if panel_cancelled is not None and panel_cancelled.is_set():
            timed_out = True
            terminate_process_group(process)
        try:
            stdout_raw, stderr_raw = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(process)
            stdout_raw, stderr_raw = process.communicate()
    finally:
        if registry is not None:
            registry.discard(profile, process)
    return_code = process.returncode
    archive = archive_automation_session(
        runtime=runtime,
        profile_home=profile_home,
        session_id=session_id,
    )
    stdout_bytes, stdout_sha256, stdout_truncated = write_capture(
        stdout_path, stdout_raw or b"", secret_values
    )
    stderr_bytes, stderr_sha256, stderr_truncated = write_capture(
        stderr_path, stderr_raw or b"", secret_values
    )
    ended_at = utc_now()
    panel_deadline = panel_cancelled is not None and panel_cancelled.is_set()
    outcome = "timed_out" if timed_out or panel_deadline else ("returned" if return_code == 0 else "failed")
    return {
        "profile": profile,
        "profile_home": str(profile_home),
        "session_id": session_id,
        "session_archive_requested": True,
        "session_archive_outcome": archive["outcome"],
        "session_archive_detail": archive["detail"],
        "runtime": str(runtime),
        "command": [*command[:-1], "<brief>"],
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": round(time.monotonic() - started, 3),
        "exit_code": return_code,
        "outcome": outcome,
        "stdout_path": str(stdout_path.relative_to(receipt_dir)),
        "stderr_path": str(stderr_path.relative_to(receipt_dir)),
        "stdout_bytes": stdout_bytes,
        "stderr_bytes": stderr_bytes,
        "stdout_sha256": stdout_sha256,
        "stderr_sha256": stderr_sha256,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "output_policy": "redacted_utf8_bounded_v1",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an explicit relevance-gated Ares specialist set with bounded receipt capture."
    )
    parser.add_argument("--brief", required=True, help="Identical self-contained brief for every selected profile")
    routing = parser.add_mutually_exclusive_group(required=True)
    routing.add_argument(
        "--profiles",
        help="Comma-separated canonical specialists selected by relevance",
    )
    routing.add_argument(
        "--full-panel",
        action="store_true",
        help="Explicitly escalate to all eight specialists",
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path.home() / ".ares" / "runtime" / "current",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Receipt directory; defaults to ~/.ares/profile-collaboration/receipts/<run>",
    )
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_PROFILE_TIMEOUT_SECONDS,
        help="Per-profile timeout in seconds (default: 180)",
    )
    parser.add_argument(
        "--panel-timeout",
        type=float,
        default=DEFAULT_PANEL_TIMEOUT_SECONDS,
        help="Independent wall-clock deadline for the complete panel (default: 600)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    args = parse_args()
    try:
        profiles = select_profiles(args.profiles, args.full_panel)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    runtime = args.runtime.expanduser().resolve()
    workspace = args.workspace.expanduser().resolve()
    if not runtime.is_dir():
        raise SystemExit(f"runtime does not exist: {runtime}")
    if not workspace.is_dir():
        raise SystemExit(f"workspace does not exist: {workspace}")
    python = runtime / ".venv" / "bin" / "python"
    if not python.is_file():
        raise SystemExit(f"runtime Python does not exist: {python}")
    if not args.brief.strip():
        raise SystemExit("brief must not be empty")
    if args.max_workers < 1 or args.max_workers > len(PROFILES):
        raise SystemExit(f"max-workers must be between 1 and {len(PROFILES)}")
    worker_count = min(args.max_workers, len(profiles))
    if args.timeout <= 0:
        raise SystemExit("timeout must be positive")
    if args.panel_timeout <= 0:
        raise SystemExit("panel-timeout must be positive")

    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    receipt_dir = (
        args.out.expanduser().resolve()
        if args.out
        else Path.home() / ".ares" / "profile-collaboration" / "receipts" / run_name
    )
    receipt_dir.mkdir(parents=True, exist_ok=False)
    (receipt_dir / "profiles").mkdir()
    brief_digest = hashlib.sha256(args.brief.encode("utf-8")).hexdigest()
    manifest: dict[str, Any] = {
        "schema": "AresProfilePanelReceiptV1",
        "run_id": run_name,
        "started_at": utc_now(),
        "runtime": str(runtime),
        "runtime_revision": runtime_revision(runtime),
        "workspace": str(workspace),
        "required_profiles": list(profiles),
        "routing_mode": "full_panel" if args.full_panel else "relevance_gated",
        "brief_sha256": brief_digest,
        "max_workers": worker_count,
        "timeout_seconds": args.timeout,
        "panel_timeout_seconds": args.panel_timeout,
        "scope": "read-only profile consultation; no profile mutation authorized",
        "session_projection_policy": "archive_automation_oneshot_v1",
        "output_policy": {
            "mode": "redacted_utf8_bounded_v1",
            "max_bytes_per_stream": MAX_CAPTURE_BYTES,
            "secrets_are_not_recorded": True,
        },
        "dry_run": args.dry_run,
        "results": [],
        "execution_complete": False,
        "controller_verified": False,
    }
    if args.dry_run:
        manifest["planned_commands"] = [
            {
                "profile": profile,
                "profile_home": str(Path.home() / ".ares" / "profiles" / profile),
                "command": [str(python), "-m", "hermes_cli.main", "--in", str(workspace), "--reasoning", "low", "-z", "<brief>"],
            }
            for profile in profiles
        ]
        manifest["execution_complete"] = True
        manifest["ended_at"] = utc_now()
        atomic_write_manifest(receipt_dir / "panel.json", manifest)
        print(receipt_dir)
        return 0

    panel_path = receipt_dir / "panel.json"
    atomic_write_manifest(panel_path, manifest)
    registry = ProcessRegistry()
    panel_cancelled = threading.Event()
    results: dict[str, dict[str, Any]] = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=worker_count)
    futures = {
        executor.submit(
            run_one,
            profile=profile,
            brief=args.brief,
            runtime=runtime,
            workspace=workspace,
            receipt_dir=receipt_dir,
            timeout=args.timeout,
            registry=registry,
            panel_cancelled=panel_cancelled,
        ): profile
        for profile in profiles
    }
    pending = set(futures)
    deadline = time.monotonic() + args.panel_timeout
    panel_timed_out = False
    try:
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                panel_timed_out = True
                break
            done, pending = concurrent.futures.wait(
                pending,
                timeout=remaining,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                profile = futures[future]
                try:
                    results[profile] = future.result()
                except concurrent.futures.CancelledError:
                    results[profile] = {"profile": profile, "outcome": "cancelled"}
                except Exception as exc:  # Preserve one profile failure without hiding others.
                    results[profile] = {
                        "profile": profile,
                        "outcome": "controller_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                manifest["results"] = ordered_results(results, profiles)
                atomic_write_manifest(panel_path, manifest)

        if panel_timed_out:
            panel_cancelled.set()
            active_profiles = set(registry.snapshot())
            terminate_process_groups(list(registry.snapshot().values()))
            for future in pending:
                future.cancel()
            # Capture workers which finished as a direct result of termination.
            done_after_cancel, _ = concurrent.futures.wait(pending, timeout=0.25)
            for future in done_after_cancel:
                profile = futures[future]
                if profile in results:
                    continue
                try:
                    results[profile] = future.result()
                except concurrent.futures.CancelledError:
                    pass
                except Exception as exc:
                    results[profile] = {
                        "profile": profile,
                        "outcome": "controller_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
            for profile in profiles:
                if profile not in results:
                    results[profile] = {
                        "profile": profile,
                        "outcome": "timed_out" if profile in active_profiles else "cancelled",
                        "exit_code": None,
                        "error": "panel wall-clock deadline reached",
                    }
            manifest["panel_timed_out"] = True
            manifest["results"] = ordered_results(results, profiles)
            manifest["ended_at"] = utc_now()
            atomic_write_manifest(panel_path, manifest)
    finally:
        # Crucially, a panel deadline must not enter ThreadPoolExecutor's
        # context-manager shutdown path, which waits for every queued task.
        executor.shutdown(wait=not panel_timed_out, cancel_futures=panel_timed_out)

    ordered = ordered_results(results, profiles)
    manifest["results"] = ordered
    manifest["ended_at"] = utc_now()
    manifest["execution_complete"] = len(ordered) == len(profiles) and all(
        result.get("outcome") == "returned" and result.get("exit_code") == 0
        for result in ordered
    )
    atomic_write_manifest(panel_path, manifest)
    print(receipt_dir)
    return 0 if manifest["execution_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
