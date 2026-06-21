"""Windows Service wrapper for Hermes Agent Gateway.

This script registers as a proper Windows Service with the Service Control
Manager (SCM). When installed, the SCM manages its lifecycle and RecoveryActions
provide automatic crash recovery — equivalent to systemd Restart=always on Linux.

Installation:
    python _hermes_gateway_service.py install
    python _hermes_gateway_service.py start

Architecture:
    SCM → pythonw.exe _hermes_gateway_service.py
                │
                ├── SvcDoRun() → background thread: asyncio.run(start_gateway())
                ├── SvcStop()  → write planned-stop marker + signal stop_event
                └── RecoveryActions: quadratic backoff restart on failure
"""

from __future__ import annotations

import sys
import os
import threading
import logging
from pathlib import Path

# Platform guard
if sys.platform != "win32":
    raise RuntimeError("This module is Windows-only")

try:
    import win32service
    import win32serviceutil
except ImportError:
    raise RuntimeError(
        "pywin32 is required for Windows Service support. "
        "Install with: pip install pywin32"
    )

SERVICE_NAME = "HermesGateway"
SERVICE_DISPLAY_NAME = "Hermes Agent Gateway"
SERVICE_DESCRIPTION = (
    "Hermes Agent AI gateway \u2014 messaging platform integration, "
    "cron scheduler, and session management."
)

# Recovery actions: quadratic backoff (Tailscale pattern)
_RECOVERY_DELAYS_MS = [1000, 2000, 4000, 9000, 16000, 25000, 36000, 49000, 64000]
_RECOVERY_RESET_PERIOD_S = 60


class HermesGatewayService(win32serviceutil.ServiceFramework):
    """Windows Service that runs the Hermes Agent Gateway."""

    _svc_name_ = SERVICE_NAME
    _svc_display_name_ = SERVICE_DISPLAY_NAME
    _svc_description_ = SERVICE_DESCRIPTION

    def __init__(self, args):
        super().__init__(args)
        self._stop_event = threading.Event()
        self._gateway_thread = None

    def SvcDoRun(self):
        # CRITICAL: Report SERVICE_RUNNING immediately (R-04)
        # SCM expects this within 30 seconds. MCP discovery blocks 120s.
        self.ReportServiceStatus(win32service.SERVICE_RUNNING)

        # Start gateway in background daemon thread (R-03)
        self._gateway_thread = threading.Thread(
            target=self._run_gateway,
            daemon=True,
        )
        self._gateway_thread.start()

        # Block until SvcStop() signals us
        self._stop_event.wait()

    def _run_gateway(self):
        """Run the gateway in a background thread."""
        try:
            # Set environment for detached mode
            os.environ["HERMES_GATEWAY_DETACHED"] = "1"

            # Import and run gateway
            from hermes_cli.gateway import run_gateway

            run_gateway()
        except Exception as exc:
            logging.exception("Gateway crashed: %s", exc)
        finally:
            # If gateway exits on its own, stop the service
            self._stop_event.set()

    def SvcStop(self):
        """Called by SCM to stop the service."""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)

        # Write planned-stop marker (reuse existing mechanism, R-06)
        try:
            from gateway.status import write_planned_stop_marker
            import ctypes

            pid = ctypes.windll.kernel32.GetCurrentProcessId()
            write_planned_stop_marker(pid)
        except Exception:
            pass

        # Signal SvcDoRun to return
        self._stop_event.set()


def configure_recovery_actions():
    """Set SCM RecoveryActions for quadratic backoff restart."""
    try:
        scm = win32service.OpenSCManager(
            None, None, win32service.SC_MANAGER_ALL_ACCESS
        )
        try:
            svc = win32service.OpenService(
                scm, SERVICE_NAME, win32service.SERVICE_ALL_ACCESS
            )
            try:
                action_tuples = [(1, d) for d in _RECOVERY_DELAYS_MS]  # 1 = SC_ACTION_RESTART
                win32service.ChangeServiceConfig2(
                    svc,
                    win32service.SERVICE_CONFIG_FAILURE_ACTIONS,
                    (_RECOVERY_RESET_PERIOD_S, None, None, action_tuples),
                )
                win32service.ChangeServiceConfig2(
                    svc,
                    win32service.SERVICE_CONFIG_FAILURE_ACTIONS_FLAG,
                    {"fFailureActionsOnNonCrashFailures": True},
                )
                win32service.ChangeServiceConfig2(
                    svc,
                    win32service.SERVICE_CONFIG_DESCRIPTION,
                    SERVICE_DESCRIPTION,
                )
            finally:
                win32service.CloseServiceHandle(svc)
        finally:
            win32service.CloseServiceHandle(scm)
    except Exception as exc:
        logging.warning("Could not configure RecoveryActions: %s", exc)


def main():
    """Entry point for sc.exe."""
    # Configure logging to file (Services run in Session 0)
    try:
        from hermes_constants import get_hermes_home

        log_dir = Path(get_hermes_home()) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(
                    str(log_dir / "gateway-service.log"), encoding="utf-8"
                )
            ],
        )
    except Exception:
        logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1 and sys.argv[1].lower() == "install":
        win32serviceutil.HandleCommandLine(HermesGatewayService)
        configure_recovery_actions()
        print(f"\u2713 Hermes Gateway installed as Windows Service ({SERVICE_NAME})")
        return

    win32serviceutil.HandleCommandLine(HermesGatewayService)


if __name__ == "__main__":
    main()
