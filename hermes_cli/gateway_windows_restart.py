"""Windows Gateway Transactional Restart Coordinator.

Orchestrates safe, verifiable gateway restarts from three entry points:
1. Chat platform /restart command
2. Agent terminal ``hermes gateway restart``
3. External PowerShell ``hermes gateway restart``

The coordinator writes an intent, spawns a detached worker, and lets the
current gateway drain and exit.  The worker waits for the old gateway to
die, verifies the port is free, starts a new gateway, and confirms it came
up — all without inheriting ``_HERMES_GATEWAY=1``.

This module does NOT call ``hermes gateway restart`` recursively.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional


def _assert_windows() -> None:
    if sys.platform != "win32":
        raise RuntimeError("gateway_windows_restart is Windows-only")


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight_check(
    *,
    profile: str = "default",
    hermes_home: str | None = None,
    target_pid: int = 0,
) -> tuple[bool, str]:
    """Run preflight checks before stopping the old gateway.

    Returns (ok, detail).  If ok is False, the old gateway must NOT be stopped.
    """
    _assert_windows()
    errors: list[str] = []

    # 1. Python / pythonw available
    python_exe = sys.executable
    if not python_exe or not Path(python_exe).exists():
        errors.append(f"Python executable not found: {python_exe}")
    try:
        from hermes_cli.gateway_windows import _derive_venv_pythonw
        pythonw = _derive_venv_pythonw(python_exe)
        if not pythonw or not Path(pythonw).exists():
            errors.append("pythonw.exe not found — detached restart will fail")
    except Exception as e:
        errors.append(f"pythonw resolution failed: {e}")

    # 2. Worker module importable
    try:
        import hermes_cli.gateway_windows_restart_worker  # noqa: F401
    except ImportError as e:
        errors.append(f"Worker module not importable: {e}")

    # 3. HERMES_HOME readable
    from hermes_cli.config import get_hermes_home
    home = hermes_home or str(Path(get_hermes_home()).resolve())
    if not Path(home).is_dir():
        errors.append(f"HERMES_HOME not readable: {home}")

    # 4. Profile config exists
    profile_dir = Path(home) / "profiles" / profile if profile != "default" else Path(home)
    if not profile_dir.is_dir():
        errors.append(f"Profile directory not found: {profile_dir}")

    # 5. Task name resolvable
    try:
        from hermes_cli.gateway_windows import get_task_name
        task_name = get_task_name()
        if not task_name:
            errors.append("Task name resolved to empty")
    except Exception as e:
        errors.append(f"Task name resolution failed: {e}")

    # 6. Restart base directory writable
    try:
        from hermes_cli.gateway_restart_state import _get_restart_base
        base = _get_restart_base()
        test_file = base / ".preflight-test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
    except (OSError, Exception) as e:
        errors.append(f"Restart directory not writable: {e}")

    # 7. Logs directory writable
    try:
        from hermes_cli.gateway_restart_state import _get_logs_dir
        logs_dir = _get_logs_dir()
        test_file = logs_dir / ".preflight-test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
    except (OSError, Exception) as e:
        errors.append(f"Logs directory not writable: {e}")

    if errors:
        return False, "; ".join(errors)
    return True, "preflight_ok"


# ---------------------------------------------------------------------------
# Schedule restart handoff
# ---------------------------------------------------------------------------

def schedule_restart_handoff(
    *,
    origin: str = "external-cli",
    profile: str = "default",
    wait: bool = True,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Schedule a transactional restart.

    Correct ordering: acquire lock → write intent → spawn worker →
    worker claims lease → hand off active.lock ownership to worker.

    Returns a result dict with keys:
    - request_id: str
    - scheduled: bool
    - detail: str
    - completed: bool (only if wait=True)
    - old_pid: int
    - new_pid: int (only if completed)
    - launcher: str (only if completed)
    """
    _assert_windows()
    # P1-3: Light GC of expired request directories
    try:
        from hermes_cli.gateway_restart_state import gc_expired_request_dirs
        gc_expired_request_dirs(profile=profile)
    except Exception:
        pass  # GC failure must not block restart
    from hermes_cli.gateway_restart_state import (
        RestartLock,
        append_restart_log,
        cleanup_intent,
        create_intent,
        write_status,
    )
    from gateway.status import get_running_pid

    # Resolve current gateway PID
    old_pid = get_running_pid() or 0

    # Get task name
    try:
        from hermes_cli.gateway_windows import get_task_name
        task_name = get_task_name()
    except Exception:
        task_name = ""

    # Preflight
    ok, detail = preflight_check(profile=profile, target_pid=old_pid)
    if not ok:
        append_restart_log(
            request_id="", profile=profile, old_pid=old_pid,
            origin=origin, state="failed", error=f"preflight: {detail}",
        )
        return {
            "request_id": "",
            "scheduled": False,
            "detail": f"Preflight failed: {detail}",
        }

    # Generate ONE request_id for the entire transaction
    request_id = str(uuid.uuid4())

    # Acquire lock FIRST (P1-1: record worker_pid for claim timeout recovery)
    lock = RestartLock(profile)
    try:
        # We don't know worker_pid yet — acquire without it, update after spawn
        if not lock.try_acquire(request_id):
            append_restart_log(
                request_id=request_id, profile=profile, old_pid=old_pid,
                origin=origin, state="failed", error="lock contention",
            )
            return {
                "request_id": request_id,
                "scheduled": False,
                "detail": "Another restart is already in progress",
            }

        # Lock acquired — initialize transaction
        try:
            intent = create_intent(
                request_id=request_id,
                profile=profile,
                target_pid=old_pid,
                task_name=task_name,
                origin=origin,
            )

            write_status(profile, "scheduled", request_id=request_id, old_pid=old_pid)
            append_restart_log(
                request_id=request_id, profile=profile, old_pid=old_pid,
                origin=origin, state="scheduled",
            )

            # Spawn detached worker
            worker_pid = _spawn_worker(intent, profile, request_id)
        except Exception as e:
            lock.release()
            cleanup_intent(profile, request_id)
            append_restart_log(
                request_id=request_id, profile=profile, old_pid=old_pid,
                origin=origin, state="initialization_failed", error=str(e),
            )
            return {
                "request_id": request_id,
                "scheduled": False,
                "completed": False,
                "detail": f"Failed to initialize restart transaction: {e}",
            }

        # P0-3: Update lock with worker_pid and claim_deadline for recovery
        _spawned_ok = False
        for _attempt in range(3):
            if lock.mark_worker_spawned(worker_pid, time.time() + 30):
                _spawned_ok = True
                break
            time.sleep(0.5)
        if not _spawned_ok:
            append_restart_log(
                request_id=request_id, profile=profile, old_pid=old_pid,
                origin=origin, state="failed",
                reason="mark_worker_spawned_failed_after_retries",
            )
            lock.release()
            return {
                "request_id": request_id,
                "scheduled": False,
                "detail": "Failed to record worker spawn — cannot enable stale recovery",
            }

        # Wait for worker to claim lease
        claimed = _wait_for_worker_claim(profile, request_id, timeout_s=10.0)
        handoff_succeeded = False
        if claimed:
            # P0-1 + P0-2: Hand off active.lock to Worker instead of releasing
            lease_data = _read_lease_data(profile, request_id)
            if lease_data:
                handoff_ok = lock.handoff_active_lock(
                    request_id,
                    coordinator_owner_token=lock.owner_token,
                    worker_pid=lease_data.get("worker_pid", 0),
                    lease_owner_token=lease_data.get("owner_token", ""),
                )
                if handoff_ok:
                    # Worker now owns active.lock — Coordinator must NOT release
                    handoff_succeeded = True
                else:
                    # Handoff failed — release to avoid permanent lock
                    lock.release()
                    append_restart_log(
                        request_id=request_id, profile=profile, old_pid=old_pid,
                        origin=origin, state="failed",
                        reason="handoff_active_lock_failed",
                    )
            else:
                # Lease disappeared (Worker crashed?) — release active.lock
                lock.release()
                append_restart_log(
                    request_id=request_id, profile=profile, old_pid=old_pid,
                    origin=origin, state="failed",
                    reason="lease_data_missing_for_handoff",
                )
        else:
            # Worker failed to claim — lock stays (P1-1 recovery will handle it)
            append_restart_log(
                request_id=request_id, profile=profile, old_pid=old_pid,
                origin=origin, state="scheduled",
                reason="worker_claim_timeout",
            )

        result: dict[str, Any] = {
            "request_id": request_id,
            "scheduled": handoff_succeeded,
            "detail": f"Restart scheduled (worker PID: {worker_pid})",
            "old_pid": old_pid,
        }
        if not handoff_succeeded:
            result["detail"] = "Restart handoff failed — worker may not have started"

        # If external CLI with wait, poll for completion
        if wait:
            completed, final_state = _wait_for_completion(
                profile, timeout_s, request_id)
            result["completed"] = completed
            final_status = _read_final_status(profile, request_id)
            if final_status:
                result["new_pid"] = final_status.get("new_pid", 0)
                result["launcher"] = final_status.get("launcher", "")
                if final_status.get("state") == "failed":
                    result["detail"] = f"Restart failed: {final_status.get('error', 'unknown')}"
                elif final_status.get("state") == "completed":
                    result["detail"] = "Restart completed successfully"
                else:
                    result["completed"] = False
                    result["detail"] = f"Restart timed out in state: {final_state}"
            else:
                result["detail"] = "Restart timed out waiting for completion"

        return result
    finally:
        lock.close()


def _wait_for_worker_claim(
    profile: str,
    request_id: str,
    timeout_s: float = 10.0,
) -> bool:
    """Poll the lease file until the worker claims it."""
    from hermes_cli.gateway_restart_state import lease_json_path
    lp = lease_json_path(profile, request_id)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if lp.exists():
            return True
        time.sleep(0.5)
    return False


def _read_lease_data(profile: str, request_id: str) -> Optional[dict[str, Any]]:
    """Read the lease file data for handoff verification."""
    from hermes_cli.gateway_restart_state import lease_json_path
    lp = lease_json_path(profile, request_id)
    if not lp.exists():
        return None
    try:
        data = json.loads(lp.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _wait_for_completion(
    profile: str,
    timeout_s: float,
    request_id: str = "",
) -> tuple[bool, str]:
    """Poll status file until completion or timeout.

    Returns (completed, last_seen_state).
    P1-2: Only returns completed=True when state == "completed".
    """
    from hermes_cli.gateway_restart_state import read_status

    deadline = time.monotonic() + timeout_s
    last_state = ""
    while time.monotonic() < deadline:
        status = read_status(profile, request_id)
        if status:
            last_state = status.get("state", "")
            if last_state == "completed":
                return True, last_state
            if last_state == "failed":
                return False, last_state
        time.sleep(1.0)
    return False, last_state


def _read_final_status(profile: str, request_id: str) -> Optional[dict[str, Any]]:
    from hermes_cli.gateway_restart_state import read_status
    return read_status(profile, request_id)


# ---------------------------------------------------------------------------
# Worker spawning
# ---------------------------------------------------------------------------

def _spawn_worker(intent: dict[str, Any], profile: str, request_id: str) -> int:
    """Spawn the restart worker as a fully detached process.

    Returns the worker PID.  The worker does NOT inherit _HERMES_GATEWAY=1.
    Worker reads intent from the per-request directory (no CLI arg exposure).
    """
    import subprocess
    from hermes_cli.gateway_windows import _derive_venv_pythonw

    python_exe = sys.executable
    pythonw = _derive_venv_pythonw(python_exe) or python_exe

    # Worker reads intent from per-request directory via request_id
    worker_module = "hermes_cli.gateway_windows_restart_worker"
    argv = [pythonw, "-m", worker_module,
            "--profile", profile,
            "--request-id", request_id]

    # Clean environment: remove _HERMES_GATEWAY
    env = os.environ.copy()
    env.pop("_HERMES_GATEWAY", None)
    env["HERMES_GATEWAY_RESTART_WORKER"] = "1"

    # C4: Set PYTHONPATH so the worker loads hermes_cli from the same
    # checkout as the coordinator.  Without this, an editable-install
    # .pth file in site-packages causes the worker to load from the
    # production checkout instead of the cwd-derived candidate.
    _this_src = str(Path(__file__).resolve().parent.parent)
    _existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _this_src + (os.pathsep + _existing_pp if _existing_pp else "")

    # Working directory
    from hermes_cli.config import get_hermes_home
    cwd = str(Path(get_hermes_home()).resolve())

    # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    # + CREATE_BREAKAWAY_FROM_JOB
    flags = 0x00000008 | 0x00000200 | 0x08000000 | 0x01000000

    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            creationflags=flags,
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        # Retry without CREATE_BREAKAWAY_FROM_JOB
        flags_no_breakaway = flags & ~0x01000000
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            creationflags=flags_no_breakaway,
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return proc.pid
