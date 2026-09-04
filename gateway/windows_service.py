"""`gateway run --service`: the Windows SCM frontend for the gateway.

The MSIX registers HermesGateway as a Windows Service whose Executable
is the payload's own launcher (hermes.exe — the distlib-minted PE; no
rust, no separate shim binary, plan: gateway-msix-windows-service).
When the SCM starts the service, this module IS the service process:

  SCM START → this frontend reports Running in well under a second
  (stdlib + pywin32 imports only — the heavy hermes import happens in
  the CHILD it spawns: `hermes gateway run` with no --service), then
  spawns the gateway as a child and waits.

  SCM STOP  → writes the EXISTING planned-stop marker
  (write_planned_stop_marker(child_pid)) — the gateway's own watcher
  thread (the #33778 windows-graceful-stop path) drains it exactly like
  `hermes gateway stop` does. The frontend waits out the drain window
  (below the SCM's WaitToKillServiceTimeout, with margin — Task 3),
  then force-kills the child and reports Stopped.

The graceful-drain logic is NOT duplicated here — it is the gateway's
own, exercised daily by `hermes gateway stop`. This file is only the SCM
protocol translation layer.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Optional

SERVICE_NAME = "HermesGateway"

# Task 3's contract: the SCM force-kills a service that hasn't reported
# Stopped within WaitToKillServiceTimeout (30s default, but hardening
# guides set 5s). The frontend enforces its own deadline BELOW the
# window so the force-kill decision stays hermes-owned (receipt-visible
# in the gateway's own log tail), never an SCM surprise.
_SCM_WINDOW_FALLBACK_S = 30
_STOP_MARGIN_S = 5


def run_as_service() -> int:
    """The SCM frontend loop. Returns the process exit code."""
    try:
        import win32serviceutil  # noqa: F401
        import servicemanager
    except ImportError:
        # Started outside the SCM (a user ran --service by hand): say so,
        # never hang. The real gateway doesn't need this flag.
        print(
            "gateway --service is the Windows service frontend; it is started "
            "by the Service Control Manager (or `hermes gateway service on`). "
            "For a foreground gateway run: hermes gateway run",
            file=sys.stderr,
        )
        return 2

    import win32service
    import win32serviceutil as svcutil

    class _GatewayFrontend(win32serviceutil.ServiceFramework):
        _svc_name_ = SERVICE_NAME
        _svc_display_name_ = "Hermes Gateway"
        _svc_description_ = (
            "Hermes messaging gateway — bots keep running without the "
            "desktop app. Managed by hermes; enable via "
            "`hermes gateway service on`."
        )

        def __init__(self, args):
            super().__init__(args)
            self.stop_requested = False
            self.child: Optional[subprocess.Popen] = None
            self.sewh = None

        def SvcStop(self):
            """SCM stop: nudge the child GRACEFULLY (the planned-stop
            marker — the gateway's own drain path), then honor the
            frontend's own deadline in SvcRun."""
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            self.stop_requested = True
            if self.child and self.child.poll() is None:
                _write_stop_marker(self.child.pid)

        def SvcDoRun(self):
            import servicemanager

            servicemanager.LogMsg(
                servicemanager.EVENTLOG_INFORMATION_TYPE,
                servicemanager.PYS_SERVICE_STARTED,
                (self._svc_name_, ""),
            )
            try:
                self._run_child()
            except Exception:
                import traceback

                servicemanager.LogErrorMsg(
                    "HermesGateway frontend failure:\n" + traceback.format_exc()
                )
            finally:
                servicemanager.LogMsg(
                    servicemanager.EVENTLOG_INFORMATION_TYPE,
                    servicemanager.PYS_SERVICE_STOPPED,
                    (self._svc_name_, ""),
                )

        def _run_child(self):
            # The REAL gateway: this launcher, no --service, so every
            # code path (watchdog, planned-stop watcher, graceful drain)
            # is the one `hermes gateway run` uses daily.
            argv = [
                sys.executable,
                "-m",
                "hermes_cli.main",
                "gateway",
                "run",
            ]
            self.child = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd(),
            )

            # Wait for exit or the graceful-stop request.
            drain_deadline = None
            while True:
                rc = self.child.poll()
                if rc is not None:
                    return
                if self.stop_requested:
                    if drain_deadline is None:
                        drain_deadline = time.monotonic() + _drain_window_s()
                    elif time.monotonic() >= drain_deadline:
                        # Drain window exhausted — force-kill the child.
                        # (Deliberately NOT a tree-walk taskkill: the
                        # #85265 class. The child's own children die with
                        # it under job semantics; the receipt-visible
                        # unclean-stop class is what remains.)
                        self.child.kill()
                        self.child.wait()
                        return
                time.sleep(0.25)

    svcutil.Initialize(_GatewayFrontend, None)
    svcutil.StartServiceCtrlDispatcher()
    return 0


def _drain_window_s() -> float:
    """The frontend's stop deadline: the gateway's configured drain
    timeout, capped below the SCM's WaitToKillServiceTimeout."""
    fallback = float(_SCM_WINDOW_FALLBACK_S - _STOP_MARGIN_S)
    try:
        raw = os.environ.get("HERMES_RESTART_DRAIN_TIMEOUT", "").strip()
        if raw:
            return min(float(raw), fallback)
    except (TypeError, ValueError):
        pass
    return fallback


def _write_stop_marker(child_pid: int) -> None:
    """The graceful nudge: the SAME marker `hermes gateway stop` writes;
    the child's planned-stop watcher thread consumes it and drains."""
    try:
        from gateway.status import write_planned_stop_marker

        write_planned_stop_marker(child_pid)
    except Exception:
        # Marker machinery unavailable (partial payload): fall back to
        # terminate, which the child's #53107 hardening handles.
        try:
            os.kill(child_pid, 15)  # SIGTERM-shaped nudge on windows = TerminateProcess
        except Exception:
            pass
