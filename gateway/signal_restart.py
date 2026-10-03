"""Signal-driven gateway restart and planned-stop primitives."""
from __future__ import annotations

import os
import signal

from gateway.process_liveness import _wait_for_pid_exit

def _graceful_restart_via_sigusr1(pid: int, drain_timeout: float, *, on_progress=None) -> bool:
    """SIGUSR1 (drain-aware restart) a gateway PID and wait for exit; False if unsent or it outlived the timeout.

    gateway/run.py maps SIGUSR1 to ``request_restart(via_service=True)``: refuse new turns, drain,
    ``stop()``, exit; the supervisor relaunches. ``drain_timeout`` must cover after-turn wait + drain
    — pass ``resolve_restart_exit_wait_budget(...)``. ``on_progress`` (zero-arg) runs on every poll so
    a long wait can report what the gateway is still holding for (``update_cmd_drain_report``).
    """
    if not hasattr(signal, "SIGUSR1") or pid <= 0:
        return False
    try:
        os.kill(pid, signal.SIGUSR1)  # windows-footgun: ok — POSIX signal, guarded by hasattr(signal, 'SIGUSR1') above
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False

    return _wait_for_pid_exit(pid, max(drain_timeout, 1.0), on_progress=on_progress)

def _mark_planned_stop(pid: int | None = None) -> None:
    """Best-effort planned-stop marker for ``pid`` (default: the recorded gateway PID)."""
    try:
        from gateway.status import get_running_pid, write_planned_stop_marker
        if pid is None:
            pid = get_running_pid(cleanup_stale=False)
        if pid is not None:
            write_planned_stop_marker(pid)
    except Exception:
        pass
