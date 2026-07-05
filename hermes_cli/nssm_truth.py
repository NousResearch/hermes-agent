"""nssm_truth.py — single source of truth for "is the Hermes service actually running?"

Replaces the inline `has_live_gateway_via_pidfile()` block that was copy-pasted
into three scripts today (2026-07-05): nssm-self-heal.py,
fix-gateway-paused-state.py, audit-tools.py. All three had the same bug pattern
(NSSM reports PAUSED while the actual python gateway is alive).

The PID file (gateway.lock / gateway.pid) is the source of truth. NSSM state
is advisory — there's a known false-positive pattern where NSSM loses track of
a running service but the python process is still bound to its port.

Usage:
    from nssm_truth import is_hermes_service_actually_running

    ok, detail = is_hermes_service_actually_running("HermesGateway")
    if ok:
        log(f"live gateway: {detail}")
    else:
        # NSSM/state reports STOPPED, PID file absent, etc.
        log(f"no live gateway: {detail}")
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
from ctypes import wintypes
from pathlib import Path
from typing import Optional

# Windows constants
STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# Path resolution — same pattern as other hermes scripts
def _resolve_hermes_home() -> Path:
    """Find HERMES_HOME by checking known locations in order."""
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    # Try common Windows locations
    candidates = [
        Path(r"C:\Users\bbask\AppData\Local\hermes"),
        Path.home() / "AppData/Local/hermes",
    ]
    for c in candidates:
        if c.exists() and (c / "memories").exists():
            return c
    return candidates[0]


HERMES_HOME = _resolve_hermes_home()

# Possible lock-file names. gateway.lock is the canonical, gateway.pid is the
# legacy alias. We check both for robustness.
LOCK_FILE_NAMES = ("gateway.lock", "gateway.pid")


def is_pid_alive(pid: int) -> bool:
    """Windows: check if PID is still in the process table.

    Uses OpenProcess + GetExitCodeProcess. Exit code 259 (STILL_ACTIVE)
    means the process is alive.
    """
    if pid <= 0:
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except OSError:
        return False
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False
    try:
        code = wintypes.DWORD()
        if kernel32.GetExitCodeProcess(h, ctypes.byref(code)):
            return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(h)
    return False


def _read_lock_file(hermes_home: Path) -> Optional[dict]:
    """Read and parse the first valid gateway.lock / gateway.pid file found."""
    for name in LOCK_FILE_NAMES:
        p = hermes_home / name
        if not p.exists():
            continue
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
    return None


def _lock_file_indicates_live_gateway(data: dict) -> tuple[bool, str]:
    """Given parsed lock data, return (is_live, detail)."""
    pid = data.get("pid")
    argv = data.get("argv") or []
    if not isinstance(pid, int) or pid <= 0:
        return False, "pid missing or invalid in lock file"
    if not isinstance(argv, list) or not argv:
        return False, "argv missing in lock file"

    if not is_pid_alive(pid):
        return False, f"pid {pid} not alive"

    # The argv[0] is the python script path. On Windows it looks like:
    # ...hermes_cli\main.py (NOT hermes_cli.main — that uses dots which
    # don't appear in Windows paths). Join argv and look for the path
    # fragment + the gateway command.
    argv_joined = " ".join(str(a) for a in argv).replace("\\", "/")
    if "hermes_cli/main" not in argv_joined:
        return False, f"argv does not look like hermes gateway: {argv[:2]}"
    if "gateway" not in argv_joined:
        return False, "argv does not contain 'gateway' command"

    return True, f"live gateway per PID file, PID {pid}"


def is_hermes_service_actually_running(
    service_name: str = "HermesGateway",
    hermes_home: Optional[Path] = None,
    check_pid_file: bool = True,
    fallback_to_nssm: bool = True,
) -> tuple[bool, str]:
    """Return (is_running, detail) for the named NSSM service.

    Strategy (in order):
    1. If check_pid_file and a lock file exists with a live python that
       has "hermes_cli/main" + "gateway" in argv, return True. This is the
       source of truth.
    2. If fallback_to_nssm, run `nssm status <service>` and parse the
       STATE line. RUNNING → True, PAUSED/STOPPED → False.

    Args:
        service_name: NSSM service name to query (e.g. "HermesGateway").
        hermes_home: Path to HERMES_HOME. Defaults to HERMES_HOME.
        check_pid_file: Whether to consult gateway.lock / gateway.pid first.
        fallback_to_nssm: Whether to fall back to `nssm status` when no PID
            file match.

    Returns:
        (True, detail) if the service is running (per PID file or NSSM).
        (False, detail) otherwise.
    """
    hh = Path(hermes_home) if hermes_home else HERMES_HOME

    # Strategy 1: PID file
    if check_pid_file:
        data = _read_lock_file(hh)
        if data:
            is_live, detail = _lock_file_indicates_live_gateway(data)
            if is_live:
                return True, detail
            # PID file exists but doesn't indicate a live gateway —
            # fall through to NSSM check (don't trust stale PID files).

    # Strategy 2: NSSM status
    if fallback_to_nssm:
        try:
            r = subprocess.run(
                ["nssm", "status", service_name],
                capture_output=True, text=True, timeout=10,
            )
            out = (r.stdout or "").strip()
            if "STATE" in out:
                # NSSM output looks like: "HermesGateway:\n  STATE: SERVICE_PAUSED\n"
                if "RUNNING" in out:
                    return True, "nssm state RUNNING"
                if "PAUSED" in out:
                    return False, "nssm state PAUSED (no live gateway detected)"
                if "STOPPED" in out:
                    return False, "nssm state STOPPED"
            return False, f"nssm status unclear: {out[:100]}"
        except FileNotFoundError:
            return False, "nssm not in PATH"
        except subprocess.TimeoutExpired:
            return False, "nssm status timed out"
        except Exception as e:
            return False, f"nssm status error: {type(e).__name__}: {e}"

    return False, "no PID file match and nssm check disabled"


# Convenience wrapper preserving the historical single-call API used by
# scripts that just want a bool. Logs to stderr if no logger passed.
def is_hermes_gateway_running() -> bool:
    """Shorthand: return True if the Hermes gateway is actually running."""
    ok, _detail = is_hermes_service_actually_running("HermesGateway")
    return ok


if __name__ == "__main__":
    # CLI for testing
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "verbose":
        print(f"HERMES_HOME = {HERMES_HOME}")
        for n in LOCK_FILE_NAMES:
            p = HERMES_HOME / n
            if p.exists():
                print(f"  {n}: {p.read_text(encoding='utf-8').strip()}")
        print()
    ok, detail = is_hermes_service_actually_running("HermesGateway")
    print(f"is_hermes_service_actually_running: {ok} ({detail})")
    sys.exit(0 if ok else 1)