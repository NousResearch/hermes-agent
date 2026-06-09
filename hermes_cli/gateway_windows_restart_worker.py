"""Gateway restart worker — detached process that performs the actual restart.

This module runs as ``pythonw.exe -m hermes_cli.gateway_windows_restart_worker``
with a clean environment (no ``_HERMES_GATEWAY``).  It:

1. Reads and validates the intent from the per-request directory.
2. Atomically claims the lease (O_EXCL).
3. Waits for the old gateway PID to exit.
4. Waits for the listening port to be released.
5. Starts a new gateway (Scheduled Task /Run or direct spawn).
6. Verifies the new gateway came up.
7. Writes status and JSONL log.

It does NOT call ``hermes gateway restart`` recursively.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


def main() -> None:
    """Worker entry point.

    P0-8: Top-level exception handling ensures that ANY unhandled error
    results in a ``failed`` status and resource cleanup.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Gateway restart worker")
    parser.add_argument("--profile", default="default", help="Hermes profile")
    parser.add_argument("--request-id", required=True, help="Transaction request_id")
    args = parser.parse_args()

    profile = args.profile
    request_id = args.request_id

    # Validate path components BEFORE any file operations
    from hermes_cli.gateway_restart_state import (
        _validate_profile, _validate_request_id,
    )
    try:
        _validate_profile(profile)
        _validate_request_id(request_id)
    except ValueError as exc:
        _log_error("invalid_args", str(exc),
                   profile=profile, request_id=request_id)
        sys.exit(1)

    # Read intent from per-request directory
    from hermes_cli.gateway_restart_state import read_intent
    intent = read_intent(profile, request_id)
    if not intent:
        _log_error("missing_intent", "Intent not found or invalid",
                   profile=profile, request_id=request_id)
        sys.exit(1)

    old_pid = intent.get("target_pid", 0)
    origin = intent.get("origin", "worker")
    nonce = intent.get("nonce", "")
    hermes_home = intent.get("hermes_home", "")
    task_name = intent.get("task_name", "")

    try:
        _run_restart_transaction(intent, profile, request_id, nonce,
                                 old_pid, origin, hermes_home, task_name)
    except Exception as exc:
        from hermes_cli.gateway_restart_state import append_restart_log
        append_restart_log(
            request_id=request_id, profile=profile, old_pid=old_pid,
            origin=origin, state="failed", error=f"unhandled: {exc}",
        )
        sys.exit(1)
    # NOTE: No blanket finally:cleanup_intent here.
    # - Winner cleanup: _run_restart_transaction inner finally
    # - Invalid-request cleanup: _fail_closed() does its own cleanup
    # - Loser (SystemExit from claim_lease failure): must NOT cleanup
    #   because the winner is actively using the same request directory.


def _wait_for_handoff(profile: str, request_id: str,
                      expected_owner_token: str,
                      timeout: float = 15.0) -> bool:
    from hermes_cli.gateway_restart_state import lock_path
    lp = lock_path(profile)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not lp.exists():
            return False
        try:
            data = json.loads(lp.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if (data.get("request_id") == request_id
                        and data.get("owner_token") == expected_owner_token
                        and data.get("phase") == "running"):
                    return True
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(0.5)
    return False


def _run_restart_transaction(
    intent: dict[str, Any],
    profile: str,
    request_id: str,
    nonce: str,
    old_pid: int,
    origin: str,
    hermes_home: str,
    task_name: str,
) -> None:
    """Execute the full restart transaction.

    P0-4: From claim_lease() success onwards, ALL logic is in one try/finally.
    """
    from hermes_cli.gateway_restart_state import (
        RestartLock,
        append_restart_log,
        read_intent,
        validate_intent_nonce,
        write_status,
        _pid_exists,
    )

    # Validate intent on disk matches what we read
    disk_intent = read_intent(profile, request_id)
    if not disk_intent:
        _fail_closed(profile, request_id, old_pid, origin, "missing_intent",
                     "Intent file missing or unreadable", nonce=nonce)
    if disk_intent["request_id"] != request_id:
        _fail_closed(profile, request_id, old_pid, origin, "request_id_mismatch",
                     f"Expected {request_id}, got {disk_intent['request_id']}", nonce=nonce)
    if not validate_intent_nonce(disk_intent, nonce):
        _fail_closed(profile, request_id, old_pid, origin, "nonce_mismatch",
                     "Nonce validation failed", nonce=nonce)
    if disk_intent["profile"] != profile:
        _fail_closed(profile, request_id, old_pid, origin, "profile_mismatch",
                     f"Expected profile {profile}, got {disk_intent['profile']}", nonce=nonce)
    if disk_intent["target_pid"] != old_pid:
        _fail_closed(profile, request_id, old_pid, origin, "target_pid_mismatch",
                     f"Expected PID {old_pid}, got {disk_intent['target_pid']}", nonce=nonce)
    if disk_intent.get("schema_version") != 1:
        _fail_closed(profile, request_id, old_pid, origin, "schema_version_unsupported",
                     f"Unsupported schema_version: {disk_intent.get('schema_version')}", nonce=nonce)
    disk_expires = disk_intent.get("expires_at", 0)
    if isinstance(disk_expires, (int, float)) and time.time() > disk_expires:
        _fail_closed(profile, request_id, old_pid, origin, "expired_intent",
                     f"Disk intent expired at {disk_expires}", nonce=nonce)

    # B1: task_name must be a non-empty string — schtasks /End, readiness
    # probe, and /Run all depend on it.  Empty = coordinator bug or corrupt
    # intent → fail closed, do NOT proceed with drain/start.
    if not isinstance(task_name, str) or not task_name.strip():
        _fail_closed(profile, request_id, old_pid, origin, "empty_task_name",
                     f"task_name is {task_name!r} — cannot drain or restart "
                     "without a valid Scheduled Task name", nonce=nonce)

    # P0-1 + P0-3: Atomic lease claim via O_EXCL + intent state transition
    lock = RestartLock(profile)
    lease_owned = False
    if not lock.claim_lease(request_id, nonce, expected_state="scheduled"):
        # P0-2: Lease loser must NOT write status or cleanup intent.
        # Only log and exit.
        append_restart_log(
            request_id=request_id, profile=profile, old_pid=old_pid,
            origin=origin, state="loser_exit", reason="lease_claim_failed",
        )
        sys.exit(1)
    lease_owned = True

    try:
        # P0-2: Wait for Coordinator handoff acknowledgement
        if not _wait_for_handoff(profile, request_id, lock.owner_token, timeout=15.0):
            write_status(profile, "failed", request_id=request_id,
                         error="handoff_timeout")
            append_restart_log(
                request_id=request_id, profile=profile, old_pid=old_pid,
                origin=origin, state="failed",
                error="handoff_timeout",
            )
            return  # exit cleanly, finally will cleanup

        # C1: Always (re)set HERMES_HOME from the intent — the worker must
        # not rely on inherited environment to locate config / profiles.
        if hermes_home:
            os.environ["HERMES_HOME"] = hermes_home
        # C2: Profile env — default clears stale HERMES_PROFILE inheritance;
        # non-default explicitly sets.
        if profile and profile != "default":
            os.environ["HERMES_PROFILE"] = profile
        else:
            os.environ.pop("HERMES_PROFILE", None)

        write_status(profile, "preflight_ok", request_id=request_id, old_pid=old_pid)
        append_restart_log(
            request_id=request_id, profile=profile, old_pid=old_pid,
            origin=origin, state="preflight_ok",
        )

        # --- Phase 1: Drain and stop old gateway ---
        _drain_and_stop(profile, request_id, old_pid, origin, task_name=task_name)

        # P0-6: Old PID MUST be dead before starting a new gateway
        if old_pid > 0 and _pid_exists(old_pid):
            raise RuntimeError(
                f"Old gateway PID {old_pid} is still alive after drain/stop/terminate/force-kill. "
                "Cannot safely start a new gateway."
            )

        # --- Phase 2: Wait for port release ---
        port = _detect_gateway_port()
        if port > 0:
            write_status(profile, "waiting_port_release", request_id=request_id, port=port)
            _wait_for_port_release(profile, request_id, old_pid, origin, port)

        # --- Phase 2.5: Task Scheduler registration & readiness ---
        # Tri-state probe: True=registered, False=not installed, None=ambiguous
        task_registered: bool | None = None
        if task_name:
            write_status(profile, "waiting_task_ready", request_id=request_id,
                         detail="probing registration")
            task_registered = _probe_task_registration(task_name)
            append_restart_log(
                request_id=request_id, profile=profile, old_pid=old_pid,
                origin=origin, state="probing",
                reason=f"registered={task_registered}",
            )

            if task_registered is None:
                # Ambiguous — fail closed, do not /Run, do not direct spawn
                raise RuntimeError(
                    f"Scheduled Task '{task_name}' registration probe failed "
                    "(ambiguous).  Will NOT execute /Run or direct spawn."
                )

            if task_registered:
                # Must wait for READY(3) before /Run
                task_ready = _wait_for_task_ready(
                    task_name,
                    profile, request_id, old_pid, origin,
                    timeout=30.0,
                )
                if not task_ready:
                    raise RuntimeError(
                        f"Scheduled Task '{task_name}' did not reach READY state "
                        "within 30s.  Will NOT execute /Run to avoid "
                        "MultipleInstancesPolicy=IgnoreNew suppression."
                    )
            # else: task_registered == False → skip readiness, allow direct spawn

        # --- Phase 3: Start new gateway ---
        new_pid, launcher = _start_new_gateway(
            profile, request_id, old_pid, origin, task_name, task_registered,
            hermes_home=hermes_home,
        )

        # --- Phase 4: Verify ---
        _verify_new_gateway(profile, request_id, old_pid, new_pid, origin, launcher)
    except Exception as exc:
        # P0-1: Winner must write terminal failed status on exception
        if lease_owned:
            from hermes_cli.gateway_restart_state import (
                write_status as _ws,
                append_restart_log as _arl,
            )
            try:
                _ws(profile, "failed", request_id=request_id,
                    error=str(exc))
                _arl(request_id=request_id, profile=profile,
                     old_pid=old_pid, origin=origin, state="failed",
                     error=str(exc))
            except Exception:
                pass
        raise
    finally:
        # P1-2: Order matters — release active.lock first, then lease, then intent
        # All operations are owner-scoped.
        try:
            lock.release()  # release active.lock (handed off)
        except Exception:
            pass
        if lease_owned:
            from hermes_cli.gateway_restart_state import (
                release_lease as _release_lease,
                sanitize_intent as _sanitize_intent,
            )
            _sanitize_intent(profile, request_id,
                            expected_nonce=nonce,
                            owner_token=lock.owner_token,
                            worker_pid=os.getpid())
            _release_lease(profile, request_id,
                          owner_token=lock.owner_token,
                          worker_pid=os.getpid())
        lock.close()


# ---------------------------------------------------------------------------
# Phase 1: Drain and stop
# ---------------------------------------------------------------------------

def _drain_and_stop(
    profile: str,
    request_id: str,
    old_pid: int,
    origin: str,
    task_name: str = "",
) -> None:
    """Drain and stop the old gateway using existing infrastructure.

    C2: ``task_name`` is passed from the intent — never derived from
    inherited environment at runtime.
    """
    from hermes_cli.gateway_restart_state import (
        append_restart_log,
        write_status,
        _pid_exists,
    )

    if old_pid <= 0:
        return

    # Step 1: Write planned-stop marker
    write_status(profile, "draining", request_id=request_id, old_pid=old_pid)
    append_restart_log(
        request_id=request_id, profile=profile, old_pid=old_pid,
        origin=origin, state="draining",
    )

    try:
        from gateway.status import write_planned_stop_marker
        write_planned_stop_marker(old_pid)
    except Exception:
        pass  # Best-effort

    # Step 2: Wait for agent drain (brief)
    _pid_wait(old_pid, timeout=5.0)
    if not _pid_exists(old_pid):
        append_restart_log(
            request_id=request_id, profile=profile, old_pid=old_pid,
            origin=origin, state="stopped", reason="drain_exit",
        )
        return

    # Step 3: schtasks /End
    write_status(profile, "stopping", request_id=request_id, old_pid=old_pid)
    append_restart_log(
        request_id=request_id, profile=profile, old_pid=old_pid,
        origin=origin, state="stopping",
    )

    try:
        from hermes_cli.gateway_windows import _exec_schtasks
        code = -1
        if task_name:
            code, out, err = _exec_schtasks(["/End", "/TN", task_name])
        if code == 0:
            append_restart_log(
                request_id=request_id, profile=profile, old_pid=old_pid,
                origin=origin, state="stopping", reason="schtasks_end_ok",
            )
    except Exception:
        pass

    # Step 4: Wait for PID exit
    write_status(profile, "waiting_pid_exit", request_id=request_id, old_pid=old_pid)
    if _pid_wait(old_pid, timeout=10.0):
        return

    # Step 5: taskkill /T (graceful)
    append_restart_log(
        request_id=request_id, profile=profile, old_pid=old_pid,
        origin=origin, state="waiting_pid_exit", reason="escalating_taskkill",
    )
    try:
        from gateway.status import terminate_pid
        terminate_pid(old_pid, force=False)
    except Exception:
        pass

    if _pid_wait(old_pid, timeout=5.0):
        return

    # Step 6: taskkill /T /F (force)
    append_restart_log(
        request_id=request_id, profile=profile, old_pid=old_pid,
        origin=origin, state="waiting_pid_exit", reason="escalating_taskkill_force",
    )
    try:
        from gateway.status import terminate_pid
        terminate_pid(old_pid, force=True)
    except Exception:
        pass

    _pid_wait(old_pid, timeout=5.0)

    if _pid_exists(old_pid):
        append_restart_log(
            request_id=request_id, profile=profile, old_pid=old_pid,
            origin=origin, state="waiting_pid_exit",
            error=f"PID {old_pid} still alive after force kill",
        )

    return


# ---------------------------------------------------------------------------
# Phase 2: Port release
# ---------------------------------------------------------------------------

def _detect_gateway_port() -> int:
    """Detect the gateway's listening port."""
    try:
        from hermes_cli.config import get_config
        config = get_config()
        platforms = config.get("platforms", {})
        api_cfg = platforms.get("api_server", {})
        port = api_cfg.get("port", 0)
        if port:
            return int(port)
    except Exception:
        pass

    for candidate_port in (8080, 8081, 8443):
        if _is_port_in_use(candidate_port):
            pids = _get_pids_on_port(candidate_port)
            for pid in pids:
                if _is_hermes_gateway_pid(pid):
                    return candidate_port
    return 0


try:
    import psutil as _psutil_mod
except ImportError:
    _psutil_mod = None  # type: ignore[assignment]


def _is_port_in_use(port: int) -> bool:
    """Check if a TCP port is in use."""
    if _psutil_mod is not None:
        try:
            for conn in _psutil_mod.net_connections(kind="inet"):
                if conn.laddr.port == port and conn.status == "LISTEN":
                    return True
            return False
        except _psutil_mod.AccessDenied:
            pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", port))
            return False
    except OSError:
        return True


def _get_pids_on_port(port: int) -> list[int]:
    """Get PIDs listening on a port."""
    if _psutil_mod is not None:
        try:
            pids = []
            for conn in _psutil_mod.net_connections(kind="inet"):
                if conn.laddr.port == port and conn.status == "LISTEN" and conn.pid:
                    pids.append(conn.pid)
            return pids
        except _psutil_mod.AccessDenied:
            pass
    return []


def _argv_looks_like_gateway(argv: list[str]) -> bool:
    """Token-level check: does *argv* represent a running Hermes gateway?

    Recognises the patterns that the real gateway uses:

    * ``pythonw -m hermes_cli.main gateway run``
    * ``pythonw -m hermes_cli.main -p work gateway run``
    * ``pythonw -m hermes_cli.main --profile work gateway run``
    * ``python hermes_cli/main.py gateway run``
    * ``hermes gateway run``
    * Direct script: ``.../gateway/run.py``

    Profile flags (``-p``/``--profile``) and other flags may appear anywhere
    between the module specifier and ``gateway``.
    """
    if not argv:
        return False

    # Pattern: direct gateway/run.py invocation
    for tok in argv:
        norm = tok.replace("\\", "/")
        if norm.endswith("/gateway/run.py") or norm == "gateway/run.py":
            return True

    # Token-level scanning for ``-m hermes_cli.main ... gateway ...`` or
    # ``hermes ... gateway ...``.
    tokens = argv[:]  # shallow copy
    i = 0
    found_module = False

    while i < len(tokens):
        tok = tokens[i]

        # Pattern A: ``-m hermes_cli.main`` or ``-m hermes_cli/main.py``
        if tok == "-m" and i + 1 < len(tokens):
            nxt = tokens[i + 1].replace("\\", "/")
            if nxt in ("hermes_cli.main", "hermes_cli/main.py"):
                found_module = True
                i += 2
                continue

        # Pattern B: ``hermes`` or ``hermes.exe`` CLI entry point
        # Accept both bare name and full path (e.g. C:\hermes\hermes.exe)
        _base = os.path.basename(tok).lower()
        if _base in ("hermes", "hermes.exe") or tok.lower() in ("hermes", "hermes.exe"):
            found_module = True
            i += 1
            continue

        i += 1

    if not found_module:
        return False

    # Now scan for ``gateway`` followed (not necessarily immediately) by
    # ``run``.  Skip flag-with-value pairs (``-p X``, ``--profile X``)
    # and standalone flags.
    j = 0
    while j < len(tokens):
        if tokens[j] == "gateway":
            # Look forward for ``run`` — skip flags and their values
            k = j + 1
            while k < len(tokens):
                t = tokens[k]
                if t in ("-p", "--profile", "-o", "--output"):
                    k += 2  # skip flag + value
                    continue
                if t.startswith("-"):
                    k += 1  # skip standalone flag
                    continue
                if t == "run":
                    return True
                break
            # ``gateway`` alone (no ``run`` found) — still a gateway process
            # (e.g. ``gateway status`` or just ``gateway``)
            return True
        j += 1

    return False


def _is_hermes_gateway_pid(pid: int) -> bool:
    """Check if a PID is a Hermes Gateway process."""
    if _psutil_mod is None:
        return False
    try:
        proc = _psutil_mod.Process(pid)
        argv = proc.cmdline()
        return _argv_looks_like_gateway(argv)
    except (_psutil_mod.NoSuchProcess, _psutil_mod.AccessDenied):
        pass
    return False


def _wait_for_port_release(
    profile: str,
    request_id: str,
    old_pid: int,
    origin: str,
    port: int,
    timeout: float = 30.0,
) -> None:
    """Wait for the port to be released — with closed-loop verification."""
    from hermes_cli.gateway_restart_state import append_restart_log

    # Step 1: Wait for natural release
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_port_in_use(port):
            append_restart_log(
                request_id=request_id, profile=profile, old_pid=old_pid,
                origin=origin, state="waiting_port_release",
                port=port, reason="port_released",
            )
            return
        time.sleep(1.0)

    # Step 2: Port still occupied — query who owns it
    pids = _get_pids_on_port(port)

    if not pids:
        append_restart_log(
            request_id=request_id, profile=profile, old_pid=old_pid,
            origin=origin, state="failed", port=port,
            error=f"Port {port} still occupied but no PIDs returned from query",
        )
        raise RuntimeError(
            f"Port {port} is still in use but no listening PIDs could be identified. "
            "Cannot safely proceed."
        )

    # Step 3: Identify each listener
    for pid in pids:
        if pid == os.getpid():
            continue
        if _is_ancestor(pid):
            continue

        # P1-5: Only terminate if PID matches the old gateway PID
        if pid == old_pid:
            append_restart_log(
                request_id=request_id, profile=profile, old_pid=old_pid,
                origin=origin, state="waiting_port_release",
                port=port, listener_pid=pid,
                reason="cleaning_old_gateway_listener",
            )
            try:
                from gateway.status import terminate_pid
                terminate_pid(pid, force=True)
            except Exception:
                pass
        else:
            append_restart_log(
                request_id=request_id, profile=profile, old_pid=old_pid,
                origin=origin, state="failed",
                port=port, listener_pid=pid,
                error=f"Port {port} occupied by PID {pid} (not old gateway {old_pid})",
            )
            raise RuntimeError(
                f"Port {port} is occupied by PID {pid} (not the old gateway {old_pid}). "
                "Cannot safely restart.  Will NOT kill."
            )

    # Step 4: Re-verify port release after killing Hermes listeners
    verify_deadline = time.monotonic() + 10.0
    while time.monotonic() < verify_deadline:
        if not _is_port_in_use(port):
            append_restart_log(
                request_id=request_id, profile=profile, old_pid=old_pid,
                origin=origin, state="waiting_port_release",
                port=port, reason="port_released_after_cleanup",
            )
            return
        time.sleep(0.5)

    # Step 5: Port STILL occupied — fail closed
    append_restart_log(
        request_id=request_id, profile=profile, old_pid=old_pid,
        origin=origin, state="failed", port=port,
        error=f"Port {port} still occupied after terminating Hermes listeners",
    )
    raise RuntimeError(
        f"Port {port} is still occupied after terminating Hermes Gateway listeners. "
        "Cannot safely start a new gateway."
    )


# ---------------------------------------------------------------------------
# Phase 2.5: Wait for Task Scheduler state convergence
# ---------------------------------------------------------------------------

# Task Scheduler COM state constants (locale-independent)
_TASK_STATE_UNKNOWN = 0
_TASK_STATE_DISABLED = 1
_TASK_STATE_QUEUED = 2
_TASK_STATE_READY = 3
_TASK_STATE_RUNNING = 4

_TASK_STATE_NAMES = {
    0: "UNKNOWN", 1: "DISABLED", 2: "QUEUED", 3: "READY", 4: "RUNNING",
}


def _probe_task_registration(task_name: str) -> bool | None:
    """Tri-state probe: is the Scheduled Task registered?

    Uses Task Scheduler COM API directly.  Does NOT rely on schtasks.exe
    exit codes (which conflate "not found" with "timeout"/"access denied").

    Returns:
        True  = task definitely exists (GetTask succeeds, state queryable)
        False = task definitely does not exist (FILE_NOT_FOUND HRESULT)
        None  = query failed / ambiguous — caller must fail closed
    """
    import subprocess
    try:
        # GetTask throws for non-existent tasks; catch the HRESULT.
        # 0x80070002 = HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND)
        # 0x80070003 = HRESULT_FROM_WIN32(ERROR_PATH_NOT_FOUND)
        ps_cmd = (
            f'try {{ '
            f'$svc = New-Object -ComObject "Schedule.Service"; '
            f'$svc.Connect(); '
            f'$null = $svc.GetFolder("\\").GetTask("\\{task_name}"); '
            f'"EXISTS" '
            f'}} catch {{ '
            f'$hr = [System.Runtime.InteropServices.Marshal]::GetHRForException($_.Exception); '
            f'"HRESULT:$hr" '
            f'}}'
        )
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=10,
            creationflags=0x08000000,  # CREATE_NO_WINDOW
        )
        output = (proc.stdout or "").strip()

        if proc.returncode != 0 or not output:
            return None         # PowerShell itself failed

        if output == "EXISTS":
            return True         # GetTask succeeded → task exists

        if output.startswith("HRESULT:"):
            try:
                hr = int(output.split(":")[1]) & 0xFFFFFFFF  # signed → unsigned
                if hr == 0x80070002:
                    return False    # FILE_NOT_FOUND → definitely absent
                if hr == 0x80070003:
                    return None     # PATH_NOT_FOUND → ambiguous, fail closed
            except (ValueError, IndexError):
                pass
            return None         # Other HRESULT → ambiguous

        return None             # Unexpected output → ambiguous
    except subprocess.TimeoutExpired:
        return None             # Timeout → ambiguous
    except Exception:
        return None             # Any exception → ambiguous


def _query_task_state_com(task_name: str) -> int:
    """Query Scheduled Task state via COM API (locale-independent).

    Returns numeric state: 0=UNKNOWN, 1=DISABLED, 2=QUEUED, 3=READY, 4=RUNNING.
    Returns -1 on query failure.
    """
    import subprocess
    try:
        ps_cmd = (
            f'$svc = New-Object -ComObject "Schedule.Service"; '
            f'$svc.Connect(); '
            f'$task = $svc.GetFolder("\\").GetTask("\\{task_name}"); '
            f'$task.State'
        )
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=10,
            creationflags=0x08000000,  # CREATE_NO_WINDOW
        )
        output = (proc.stdout or "").strip()
        if proc.returncode == 0 and output.isdigit():
            return int(output)
        return -1
    except Exception:
        return -1


def _wait_for_task_ready(
    task_name: str,
    profile: str,
    request_id: str,
    old_pid: int,
    origin: str,
    timeout: float = 30.0,
) -> bool:
    """Wait for the Scheduled Task to reach READY state (COM state == 3).

    Uses COM API numeric state (locale-independent):
    0=UNKNOWN, 1=DISABLED, 2=QUEUED, 3=READY, 4=RUNNING.
    Only READY (3) allows proceeding with /Run.

    Returns True if task reached READY within timeout.
    Returns False if timed out or query failed (caller must NOT proceed).
    """
    from hermes_cli.gateway_restart_state import append_restart_log

    deadline = time.monotonic() + timeout
    poll_count = 0

    while time.monotonic() < deadline:
        poll_count += 1
        state = _query_task_state_com(task_name)
        state_name = _TASK_STATE_NAMES.get(state, f"INVALID({state})")

        if state == _TASK_STATE_READY:
            append_restart_log(
                request_id=request_id, profile=profile, old_pid=old_pid,
                origin=origin, state="waiting_task_ready",
                reason="task_state_ready",
                detail=f"polls={poll_count}, state={state_name}({state})",
            )
            # Brief stability window after reaching READY
            time.sleep(1.5)
            return True

        # Not READY — log and continue polling
        if poll_count == 1:
            append_restart_log(
                request_id=request_id, profile=profile, old_pid=old_pid,
                origin=origin, state="waiting_task_ready",
                reason="task_state_poll",
                detail=f"state={state_name}({state})",
            )

        # Query failure (-1) → continue polling (transient), not immediate fail
        time.sleep(2.0)

    # Timeout — task never reached READY
    final_state = _query_task_state_com(task_name)
    final_name = _TASK_STATE_NAMES.get(final_state, f"INVALID({final_state})")
    append_restart_log(
        request_id=request_id, profile=profile, old_pid=old_pid,
        origin=origin, state="waiting_task_ready",
        reason="task_stop_timeout",
        detail=f"polls={poll_count}, final_state={final_name}({final_state})",
    )
    return False


# ---------------------------------------------------------------------------
# Phase 3: Start new gateway
# ---------------------------------------------------------------------------

def _start_new_gateway(
    profile: str,
    request_id: str,
    old_pid: int,
    origin: str,
    task_name: str,
    task_registered: bool | None = None,
    hermes_home: str = "",
) -> tuple[int, str]:
    """Start a new gateway.  Returns (new_pid, launcher).

    C3: ``hermes_home`` and ``profile`` are forwarded to _direct_spawn_gateway
    so the spawned process inherits intent-captured values.
    """
    from hermes_cli.gateway_restart_state import append_restart_log, write_status

    # Fail closed: caller must provide registration state from shared probe
    if task_registered is None:
        raise RuntimeError(
            "Scheduled Task registration state is ambiguous; "
            "refusing /Run and direct spawn"
        )

    if task_registered:
        write_status(profile, "starting_task", request_id=request_id)
        append_restart_log(
            request_id=request_id, profile=profile, old_pid=old_pid,
            origin=origin, state="starting_task",
        )

        code = -1
        try:
            from hermes_cli.gateway_windows import _exec_schtasks
            code, out, err = _exec_schtasks(["/Run", "/TN", task_name])
        except Exception as e:
            append_restart_log(
                request_id=request_id, profile=profile, old_pid=old_pid,
                origin=origin, state="starting_task",
                error=f"schtasks_run_exception: {e}",
            )
            # P0-2: schtasks exception → fail closed, no direct spawn
            raise RuntimeError(
                f"schtasks /Run exception: {e}.  "
                "Cannot determine if Scheduled Task was accepted. "
                "Will NOT direct-spawn to avoid dual gateway."
            )

        if code == 0:
            new_pid = _wait_for_launch_evidence(old_pid, timeout=15.0)
            if new_pid > 0:
                append_restart_log(
                    request_id=request_id, profile=profile, old_pid=old_pid,
                    new_pid=new_pid, origin=origin, state="starting_task",
                    launcher="scheduled_task", reason="launch_evidence_ok",
                )
                return new_pid, "scheduled_task"
            # P1-1: schtasks /Run accepted but no evidence — fail closed
            append_restart_log(
                request_id=request_id, profile=profile, old_pid=old_pid,
                origin=origin, state="failed",
                reason="schtasks_run_accepted_but_no_evidence",
            )
            raise RuntimeError(
                "schtasks /Run succeeded (exit 0) but no launch evidence "
                "within 15s.  Scheduled Task may have started with delay. "
                "Will NOT direct-spawn to avoid dual gateway."
            )

        # Non-zero code from schtasks → fail closed
        append_restart_log(
            request_id=request_id, profile=profile, old_pid=old_pid,
            origin=origin, state="failed",
            error=f"schtasks_run_code={code}, will NOT direct-spawn",
        )
        raise RuntimeError(
            f"schtasks /Run returned code {code}.  "
            "Cannot determine if Scheduled Task was accepted. "
            "Will NOT direct-spawn to avoid dual gateway."
        )

    # Direct detached spawn (only when schtasks not installed)
    write_status(profile, "starting_direct_fallback", request_id=request_id)
    append_restart_log(
        request_id=request_id, profile=profile, old_pid=old_pid,
        origin=origin, state="starting_direct_fallback",
    )

    new_pid = _direct_spawn_gateway(hermes_home=hermes_home, profile=profile)
    if new_pid <= 0:
        append_restart_log(
            request_id=request_id, profile=profile, old_pid=old_pid,
            new_pid=0, origin=origin, state="starting_direct_fallback",
            launcher="direct_spawn", error="direct spawn failed (no PID)",
        )
        raise RuntimeError("Direct spawn failed: no PID returned")

    verified_pid = _wait_for_launch_evidence(old_pid, timeout=15.0)
    if verified_pid <= 0:
        append_restart_log(
            request_id=request_id, profile=profile, old_pid=old_pid,
            new_pid=new_pid, origin=origin, state="starting_direct_fallback",
            launcher="direct_spawn", error="no launch evidence after direct spawn",
        )
        raise RuntimeError(
            f"Direct-spawned gateway (PID {new_pid}) did not produce launch evidence"
        )
    new_pid = verified_pid

    append_restart_log(
        request_id=request_id, profile=profile, old_pid=old_pid,
        new_pid=new_pid, origin=origin, state="starting_direct_fallback",
        launcher="direct_spawn",
    )
    return new_pid, "direct_spawn"


def _direct_spawn_gateway(hermes_home: str = "", profile: str = "default") -> int:
    """Spawn a new gateway as a detached process.  Returns the PID.

    C3: ``hermes_home`` and ``profile`` are passed from the intent — set in
    the environment before spawning so the child inherits correct paths.
    No implicit derivation from inherited environment.
    """
    try:
        os.environ.pop("HERMES_GATEWAY_RESTART_WORKER", None)
        os.environ.pop("_HERMES_GATEWAY", None)
        if hermes_home:
            os.environ["HERMES_HOME"] = hermes_home
        if profile and profile != "default":
            os.environ["HERMES_PROFILE"] = profile
        else:
            os.environ.pop("HERMES_PROFILE", None)
        from hermes_cli.gateway_windows import _spawn_detached
        return _spawn_detached()
    except Exception:
        return 0


def _wait_for_launch_evidence(old_pid: int, timeout: float = 15.0) -> int:
    """Wait for evidence that a new gateway came up.  Returns new PID or 0."""
    from gateway.status import get_running_pid

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        new_pid = get_running_pid()
        if new_pid and new_pid != old_pid and new_pid > 0:
            if _is_hermes_gateway_pid(new_pid):
                return new_pid
        time.sleep(1.0)
    return 0


# ---------------------------------------------------------------------------
# Phase 4: Verify
# ---------------------------------------------------------------------------

def _verify_new_gateway(
    profile: str,
    request_id: str,
    old_pid: int,
    new_pid: int,
    origin: str,
    launcher: str,
) -> None:
    """Verify the new gateway is healthy."""
    from hermes_cli.gateway_restart_state import (
        append_restart_log,
        write_status,
        _pid_exists,
    )

    if new_pid <= 0:
        write_status(profile, "failed", request_id=request_id,
                     error="No new gateway PID detected")
        append_restart_log(
            request_id=request_id, profile=profile, old_pid=old_pid,
            new_pid=new_pid, origin=origin, state="failed",
            launcher=launcher, error="No new gateway PID",
        )
        return

    if new_pid == old_pid:
        write_status(profile, "failed", request_id=request_id,
                     error="New PID equals old PID")
        append_restart_log(
            request_id=request_id, profile=profile, old_pid=old_pid,
            new_pid=new_pid, origin=origin, state="failed",
            launcher=launcher, error="New PID == Old PID",
        )
        return

    if not _pid_exists(new_pid):
        write_status(profile, "failed", request_id=request_id,
                     error=f"New PID {new_pid} died immediately")
        append_restart_log(
            request_id=request_id, profile=profile, old_pid=old_pid,
            new_pid=new_pid, origin=origin, state="failed",
            launcher=launcher, error="New PID died",
        )
        return

    # Stability window
    stable_seconds = 3.0
    stable_deadline = time.monotonic() + stable_seconds
    while time.monotonic() < stable_deadline:
        if not _pid_exists(new_pid):
            write_status(profile, "failed", request_id=request_id,
                         error=f"New PID {new_pid} died within {stable_seconds}s stability window")
            append_restart_log(
                request_id=request_id, profile=profile, old_pid=old_pid,
                new_pid=new_pid, origin=origin, state="failed",
                launcher=launcher, error="New PID unstable (died within stability window)",
            )
            return
        time.sleep(0.5)

    # Dual gateway check
    if old_pid > 0 and _pid_exists(old_pid):
        write_status(profile, "failed", request_id=request_id,
                     error=f"Dual gateway detected: old PID {old_pid} still alive alongside new PID {new_pid}")
        append_restart_log(
            request_id=request_id, profile=profile, old_pid=old_pid,
            new_pid=new_pid, origin=origin, state="failed",
            launcher=launcher, error="dual gateway detected",
        )
        return

    write_status(profile, "completed", request_id=request_id,
                 old_pid=old_pid, new_pid=new_pid, launcher=launcher)
    append_restart_log(
        request_id=request_id, profile=profile, old_pid=old_pid,
        new_pid=new_pid, origin=origin, state="completed",
        launcher=launcher,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pid_wait(pid: int, timeout: float = 10.0) -> bool:
    """Wait for PID to exit.  Returns True if it exited within timeout."""
    from hermes_cli.gateway_restart_state import _pid_exists

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.5)
    return False


def _is_ancestor(pid: int) -> bool:
    """Check if pid is an ancestor of the current process."""
    current = os.getpid()
    if _psutil_mod is not None:
        try:
            proc = _psutil_mod.Process(current)
            for parent in proc.parents():
                if parent.pid == pid:
                    return True
            return False
        except _psutil_mod.NoSuchProcess:
            pass

    # Fallback: walk ppid chain
    try:
        ppid = os.getppid()
        visited = set()
        while ppid and ppid not in visited:
            if ppid == pid:
                return True
            visited.add(ppid)
            if sys.platform == "win32":
                import ctypes
                k32 = ctypes.windll.kernel32
                k32.OpenProcess.restype = ctypes.c_void_p
                h = k32.OpenProcess(0x1000, False, ppid)
                if not h:
                    break
                try:
                    import ctypes.wintypes
                    ppid_buf = ctypes.wintypes.DWORD()
                    if k32.GetParentProcessId(h, ctypes.byref(ppid_buf)):
                        ppid = ppid_buf.value
                    else:
                        break
                finally:
                    k32.CloseHandle(h)
            else:
                # POSIX: read ppid from /proc (Linux only)
                try:
                    with open(f"/proc/{ppid}/stat", "r") as f:
                        parts = f.read().split(")")
                        if len(parts) >= 2:
                            fields = parts[1].split()
                            ppid = int(fields[1]) if len(fields) > 1 else 0
                        else:
                            break
                except (OSError, ValueError, IndexError):
                    break
    except Exception:
        pass
    return False


def _fail_closed(
    profile: str,
    request_id: str,
    old_pid: int,
    origin: str,
    error_code: str,
    detail: str,
    nonce: str = "",
) -> None:
    """Fail closed: log rejection and exit.

    P0-4: Pre-lease failures only write JSONL — no status overwrite,
    no resource deletion.
    """
    from hermes_cli.gateway_restart_state import append_restart_log
    append_restart_log(
        request_id=request_id, profile=profile, old_pid=old_pid,
        origin=origin, state="rejected",
        error=f"{error_code}: {detail}",
    )
    sys.exit(1)


def _log_error(code: str, detail: str, **extra: Any) -> None:
    """Log an error to JSONL."""
    from hermes_cli.gateway_restart_state import append_restart_log
    _KNOWN = {"request_id", "profile", "old_pid", "new_pid", "origin",
              "launcher", "reason", "error", "listener_pid", "port"}
    filtered = {k: v for k, v in extra.items() if k in _KNOWN}
    append_restart_log(state="failed", error=f"{code}: {detail}", **filtered)


if __name__ == "__main__":
    main()
