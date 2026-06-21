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
                ├── SvcStop()  → write planned-stop marker + graceful wait
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

# Service name is read from environment (set by install_service) or defaults
_DEFAULT_SERVICE_NAME = "HermesGateway"
SERVICE_NAME = os.environ.get("HERMES_SERVICE_NAME", _DEFAULT_SERVICE_NAME)
SERVICE_DISPLAY_NAME = "Hermes Agent Gateway"
SERVICE_DESCRIPTION = (
    "Hermes Agent AI gateway \u2014 messaging platform integration, "
    "cron scheduler, and session management."
)

# Recovery actions: quadratic backoff (Tailscale pattern)
_RECOVERY_DELAYS_MS = [1000, 2000, 4000, 9000, 16000, 25000, 36000, 49000, 64000]
_RECOVERY_RESET_PERIOD_S = 60

# Graceful shutdown timeout for gateway thread (HIGH 1)
_GRACEFUL_SHUTDOWN_TIMEOUT_S = 30

# Track gateway exit status for RecoveryActions semantics
_gateway_exit_code = 0


def _resolve_exit_code(exc_code: object) -> int:
    """Map SystemExit.code to an integer exit code.

    SystemExit(None)       => 0  (no explicit code, treat as success)
    SystemExit(0)          => 0  (explicit success)
    SystemExit(int)        => that int
    SystemExit(non-int)    => 1  (string or other object)
    """
    if exc_code is None:
        return 0
    if isinstance(exc_code, int):
        return exc_code
    return 1


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
        global _gateway_exit_code

        # CRITICAL: Report SERVICE_RUNNING immediately (R-04)
        # SCM expects this within 30 seconds. MCP discovery blocks 120s.
        self.ReportServiceStatus(win32service.SERVICE_RUNNING)

        # Start gateway in background thread (R-03)
        # NOTE: NOT daemon=True — we need the thread to stay alive for
        # graceful shutdown in SvcStop (HIGH 1).
        self._gateway_thread = threading.Thread(
            target=self._run_gateway,
            daemon=False,
        )
        self._gateway_thread.start()

        # Block until SvcStop() signals us
        self._stop_event.wait()

        # If gateway crashed (non-zero exit), report SERVICE_STOPPED with
        # non-zero exit code so SCM triggers RecoveryActions
        if _gateway_exit_code != 0:
            self.ReportServiceStatus(
                win32service.SERVICE_STOPPED,
                win32service.NO_ERROR,
                _gateway_exit_code,
            )

    def _run_gateway(self):
        """Run the gateway in a background thread."""
        global _gateway_exit_code
        try:
            # Set environment for detached mode
            os.environ["HERMES_GATEWAY_DETACHED"] = "1"

            # Import and run gateway
            from hermes_cli.gateway import run_gateway

            run_gateway()
            # Normal exit
            _gateway_exit_code = 0
        except SystemExit as exc:
            _gateway_exit_code = _resolve_exit_code(exc.code)
            if _gateway_exit_code != 0:
                logging.warning(
                    "Gateway exited with non-zero code %s", _gateway_exit_code
                )
        except Exception as exc:
            # Unhandled exception = failure
            _gateway_exit_code = 1
            logging.exception("Gateway crashed: %s", exc)
        finally:
            # Signal SvcDoRun to return
            self._stop_event.set()

    def SvcStop(self):
        """Called by SCM to stop the service.

        Implements graceful shutdown (HIGH 1):
        1. Report SERVICE_STOP_PENDING
        2. Write planned-stop marker so gateway knows this is a planned stop
        3. Wait for gateway thread to exit gracefully (up to _GRACEFUL_SHUTDOWN_TIMEOUT_S)
        4. Periodically report STOP_PENDING to prevent SCM timeout
        5. Report SERVICE_STOPPED
        """
        self.ReportServiceStatus(
            win32service.SERVICE_STOP_PENDING,
            waitHint=(_GRACEFUL_SHUTDOWN_TIMEOUT_S + 5) * 1000,
        )

        # Write planned-stop marker
        try:
            self._write_planned_stop_marker()
        except Exception as exc:
            logging.warning("Failed to write planned-stop marker: %s", exc)

        # Signal SvcDoRun to return (unblocks _stop_event.wait())
        self._stop_event.set()

        # Graceful wait for gateway thread to finish (HIGH 1)
        if self._gateway_thread is not None and self._gateway_thread.is_alive():
            deadline = _GRACEFUL_SHUTDOWN_TIMEOUT_S
            elapsed = 0
            poll_interval = 2  # seconds
            while elapsed < deadline:
                self._gateway_thread.join(timeout=poll_interval)
                if not self._gateway_thread.is_alive():
                    logging.info(
                        "Gateway thread exited gracefully after %ds", elapsed
                    )
                    break
                elapsed += poll_interval
                # Keep reporting STOP_PENDING to prevent SCM timeout (30s default)
                remaining = max(1, deadline - elapsed)
                self.ReportServiceStatus(
                    win32service.SERVICE_STOP_PENDING,
                    waitHint=remaining * 1000,
                )
            else:
                logging.warning(
                    "Gateway thread did not exit within %ds, proceeding with stop",
                    _GRACEFUL_SHUTDOWN_TIMEOUT_S,
                )

        # Final: report stopped
        self.ReportServiceStatus(win32service.SERVICE_STOPPED)

    def _write_planned_stop_marker(self):
        """Write planned-stop marker file (self-contained implementation).

        This is a simplified version of gateway.status.write_planned_stop_marker
        that doesn't require importing gateway.status (which may not be
        available in the SCM service environment).

        Parent directory is created if it doesn't exist (HIGH 2).
        """
        import json
        import ctypes
        from datetime import datetime, timezone

        # Get HERMES_HOME from environment or resolve default
        hermes_home = os.environ.get("HERMES_HOME")
        if not hermes_home:
            # Try to resolve from user profile
            appdata = os.environ.get("APPDATA", "")
            if appdata:
                hermes_home = os.path.join(appdata, "hermes")
            else:
                # Fallback to user home
                hermes_home = os.path.join(os.path.expanduser("~"), ".hermes")

        marker_dir = Path(hermes_home)
        # Ensure parent directory exists (HIGH 2)
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker_path = marker_dir / ".gateway-planned-stop.json"

        # Get current process ID
        pid = ctypes.windll.kernel32.GetCurrentProcessId()

        # Get process start time (simplified - use current time as fallback)
        try:
            import psutil
            target_start_time = int(psutil.Process(pid).create_time())
        except Exception:
            target_start_time = None

        record = {
            "target_pid": pid,
            "target_start_time": target_start_time,
            "stopper_pid": pid,  # Self-stop
            "written_at": datetime.now(timezone.utc).isoformat(),
        }

        # Write atomically (same pattern as gateway.status._write_json_file)
        tmp_path = marker_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(record, f)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(marker_path)

        logging.info("Planned-stop marker written to %s", marker_path)


def configure_recovery_actions(service_name: str | None = None):
    """Set SCM RecoveryActions for quadratic backoff restart."""
    name = service_name or SERVICE_NAME
    try:
        scm = win32service.OpenSCManager(
            None, None, win32service.SC_MANAGER_ALL_ACCESS
        )
        try:
            svc = win32service.OpenService(
                scm, name, win32service.SERVICE_ALL_ACCESS
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
    global SERVICE_NAME

    # Allow service name override via command-line argument
    # Usage: python _hermes_gateway_service.py --name HermesGateway-Profile install
    if "--name" in sys.argv:
        idx = sys.argv.index("--name")
        if idx + 1 < len(sys.argv):
            SERVICE_NAME = sys.argv[idx + 1]
            # Update class attribute
            HermesGatewayService._svc_name_ = SERVICE_NAME
            # Remove from argv so HandleCommandLine doesn't see it
            sys.argv.pop(idx)
            sys.argv.pop(idx)

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
        configure_recovery_actions(SERVICE_NAME)
        print(f"\u2713 Hermes Gateway installed as Windows Service ({SERVICE_NAME})")
        return

    win32serviceutil.HandleCommandLine(HermesGatewayService)


if __name__ == "__main__":
    main()
