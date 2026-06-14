"""Windows Service backend for the Hermes Gateway using sc.exe (built-in).

Provides a proper Windows Service registered with the Service Control Manager
(SCM), with auto-restart on failure via SCM RecoveryActions (like systemd's
``Restart=always``). Replaces the Scheduled Task approach for admin users who
want robust crash recovery.

Design
------
- Uses ``sc.exe`` (built into Windows) to create/start/stop/delete the service.
  The service binary is ``hermes.exe gateway run --service`` which calls
  ``StartServiceCtrlDispatcher`` internally to register with the SCM.
- **No pywin32 ServiceFramework dependency.** The service runs as a standard
  Windows Service via ``hermes.exe --service``, which handles all SCM
  communication internally.
- The service runs as a console-less background process — no visible window.
- SCM RecoveryActions implement quadratic backoff (1s, 2s, 4s, 9s, 16s, 25s,
  36s, 49s, 64s) with a 60s reset period — matching Tailscale's production
  pattern on Windows.
- Non-admin users fall back to the existing Startup-folder ``.cmd`` entry.
- Graceful shutdown via the planned-stop marker (already handled by the
  gateway's ``SvcStop`` implementation in ``gateway.py``).

Usage
-----
Managed through ``hermes gateway install --service-type service`` on Windows.
The standard ``hermes gateway start|stop|restart|status`` commands auto-detect
whether the service or Scheduled Task is installed.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RECOVERY_DELAYS_MS = [1000, 2000, 4000, 9000, 16000, 25000, 36000, 49000, 64000]
_RECOVERY_RESET_PERIOD_S = 60

_SERVICE_NAME_DEFAULT = "HermesGateway"
_SERVICE_DISPLAY_NAME = "Hermes Agent Gateway"
_SERVICE_DESCRIPTION = (
    "Hermes Agent Gateway - Messaging Platform Integration "
    "(Telegram, Discord, Slack, WhatsApp, and more). "
    "Auto-restarts on failure via SCM RecoveryActions."
)

_CREATE_NO_WINDOW = 0x08000000
_SC_TIMEOUT = 15


def _profile_suffix() -> str:
    try:
        from hermes_cli.gateway import _profile_suffix as _gw_suffix
        return _gw_suffix()
    except ImportError:
        return ""


def get_service_name() -> str:
    suffix = _profile_suffix()
    if not suffix:
        return _SERVICE_NAME_DEFAULT
    return f"{_SERVICE_NAME_DEFAULT}_{suffix}"


def get_service_display_name() -> str:
    suffix = _profile_suffix()
    if not suffix:
        return _SERVICE_DISPLAY_NAME
    return f"{_SERVICE_DISPLAY_NAME} ({suffix})"


def _get_hermes_exe_path() -> Path:
    """Return the path to ``hermes.exe`` in the current venv."""
    home = os.environ.get("HERMES_HOME", "")
    if home:
        candidate = Path(home) / "hermes-agent" / "venv" / "Scripts" / "hermes.exe"
        if candidate.exists():
            return candidate
    # Fallback: relative to this module
    candidate = Path(__file__).resolve().parent.parent / "venv" / "Scripts" / "hermes.exe"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("Cannot locate hermes.exe")


def _get_binpath() -> str:
    """Build the service ``binpath`` for ``sc create``."""
    exe = _get_hermes_exe_path()
    # Use python.exe (not pythonw.exe) because the service needs to interact
    # with the SCM via StartServiceCtrlDispatcher
    return f'"{exe}" gateway run --service --replace'


# ---------------------------------------------------------------------------
# sc.exe helpers
# ---------------------------------------------------------------------------


def _exec_sc(args: list[str], timeout: int = _SC_TIMEOUT) -> subprocess.CompletedProcess:
    """Run ``sc.exe`` with a timeout and CREATE_NO_WINDOW."""
    # Verify sc.exe exists (defensive — should always be present on Windows)
    sc_path = shutil.which("sc")
    if sc_path is None:
        raise RuntimeError("sc.exe not found on PATH — cannot manage Windows Services")

    return subprocess.run(
        ["sc", *args],
        capture_output=True, text=True, timeout=timeout,
        creationflags=_CREATE_NO_WINDOW,
    )


def _set_service_failure_actions() -> None:
    """Configure SCM RecoveryActions via ``sc.exe failure``."""
    svc_name = get_service_name()
    actions = "/".join(f"restart/{d}" for d in _RECOVERY_DELAYS_MS)
    r = _exec_sc([
        "failure", svc_name,
        f"reset={_RECOVERY_RESET_PERIOD_S}",
        f"actions={actions}",
    ])
    if r.returncode != 0:
        detail = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(f"sc failure failed (code {r.returncode}): {detail}")


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------


def install_service(*, start_now: bool = True) -> None:
    """Register the Hermes gateway as a Windows Service using ``sc.exe``.

    1. Delete any existing service (idempotent).
    2. ``sc create`` — registers the service with auto-start.
    3. ``sc failure`` — sets quadratic-backoff RecoveryActions.
    4. Optionally start the service immediately.

    Args:
        start_now: If True, start the service after install.
    """
    svc_name = get_service_name()
    svc_display = get_service_display_name()
    binpath = _get_binpath()

    print(f"Installing Windows Service '{svc_name}'...")
    print(f"  Binary: {binpath}")

    # ── Delete existing first (idempotent) ──────────────────────────────
    _exec_sc(["delete", svc_name], timeout=10)
    # Poll until SCM fully releases the service record
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        r = _exec_sc(["query", svc_name], timeout=5)
        if r.returncode != 0:
            break  # Service gone — safe to recreate
        time.sleep(0.5)
    else:
        print("  ⚠ Previous service instance still exists; proceeding anyway")

    # ── Create service ──────────────────────────────────────────────────
    # start= auto  → SERVICE_AUTO_START (starts at boot)
    r = _exec_sc([
        "create", svc_name,
        f"binpath={binpath}",
        f"displayname={svc_display}",
        "start=auto",
    ])
    if r.returncode != 0:
        detail = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(f"sc create failed (code {r.returncode}): {detail}")
    print("  ✓ Service created (auto-start)")

    # ── Set description ─────────────────────────────────────────────────
    _exec_sc(["description", svc_name, _SERVICE_DESCRIPTION], timeout=10)

    # ── Set RecoveryActions ─────────────────────────────────────────────
    try:
        _set_service_failure_actions()
        print(f"  ✓ SCM RecoveryActions configured (quadratic backoff, {_RECOVERY_RESET_PERIOD_S}s reset)")
    except Exception as e:
        print(f"  ⚠ Could not set RecoveryActions: {e}")
        print("    Service will NOT auto-restart on crash.")

    print(f"✓ Windows Service '{svc_name}' installed.")
    print(f"  Display name: {svc_display}")

    if start_now:
        start_service()
        print(f"✓ Service '{svc_name}' started.")
    else:
        print(f"  Start: hermes gateway start")


def uninstall_service() -> bool:
    """Remove the Windows Service from the SCM.

    Returns True if the service existed and was removed.
    """
    svc_name = get_service_name()
    if not is_service_installed():
        return False

    # Stop first if running
    status = get_service_status()
    if status.get("state") == "running":
        try:
            stop_service()
        except Exception:
            pass
        time.sleep(1)

    print(f"Removing Windows Service '{svc_name}'...")
    r = _exec_sc(["delete", svc_name], timeout=10)
    if r.returncode == 0:
        print(f"✓ Windows Service '{svc_name}' removed.")
        return True
    else:
        detail = (r.stderr or r.stdout or "").strip()
        print(f"  ⚠ sc delete returned code {r.returncode}: {detail}")
    return False


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def start_service() -> None:
    """Start the Windows Service via ``sc start``."""
    svc_name = get_service_name()
    # sc start waits for the service to reach RUNNING state; the --service
    # flag in gateway.py calls StartServiceCtrlDispatcher to register with SCM
    r = _exec_sc(["start", svc_name], timeout=60)
    if r.returncode != 0:
        detail = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(f"sc start failed (code {r.returncode}): {detail}")


def stop_service() -> None:
    """Stop the Windows Service via ``sc stop``."""
    svc_name = get_service_name()
    r = _exec_sc(["stop", svc_name], timeout=30)
    if r.returncode != 0:
        detail = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(f"sc stop failed (code {r.returncode}): {detail}")


def restart_service() -> None:
    """Restart the Windows Service (stop then start)."""
    stop_service()
    time.sleep(1)
    start_service()


# ---------------------------------------------------------------------------
# Status (uses win32service API — locale-independent, unlike sc query parsing)
# ---------------------------------------------------------------------------


def is_service_installed() -> bool:
    """Return True if the Windows Service is registered with the SCM."""
    import win32service
    svc_name = get_service_name()
    try:
        hscm = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
        try:
            hs = win32service.OpenService(hscm, svc_name, win32service.SERVICE_QUERY_STATUS)
            win32service.CloseServiceHandle(hs)
            return True
        except win32service.error:
            return False
        finally:
            win32service.CloseServiceHandle(hscm)
    except win32service.error:
        return False


def get_service_status() -> dict:
    """Query the service status via ``win32serviceutil.QueryServiceStatus``.

    Uses the win32service API directly rather than parsing ``sc query``
    output, which is locale-dependent.

    Returns a dict with keys ``state`` (str) and ``pid`` (int or None),
    or an empty dict if the service is not installed.
    """
    import win32service
    import win32serviceutil
    svc_name = get_service_name()
    if not is_service_installed():
        return {}

    try:
        status = win32serviceutil.QueryServiceStatus(svc_name)
        # status: (serviceType, currentState, controlsAccepted,
        #          win32ExitCode, serviceSpecificExitCode, checkPoint, waitHint)
        state_map = {
            win32service.SERVICE_STOPPED: "stopped",
            win32service.SERVICE_START_PENDING: "start_pending",
            win32service.SERVICE_STOP_PENDING: "stop_pending",
            win32service.SERVICE_RUNNING: "running",
            win32service.SERVICE_CONTINUE_PENDING: "continue_pending",
            win32service.SERVICE_PAUSE_PENDING: "pause_pending",
            win32service.SERVICE_PAUSED: "paused",
        }
        current_state = status[1]
        return {
            "state": state_map.get(current_state, f"unknown ({current_state})"),
            "pid": None,
        }
    except Exception as e:
        logger.warning("Failed to query service status: %s", e)
        return {}