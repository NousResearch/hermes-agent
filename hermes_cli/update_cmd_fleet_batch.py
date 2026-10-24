"""Nonblocking signal phase for independent systemd gateway drains."""

import os
import signal
import time


def _identity_exit_result(pid, started):
    """None: original still running; True: exited; False: identity unproven."""
    from gateway.status import _pid_exists, get_process_start_time

    if started is None:
        return False
    current = get_process_start_time(pid)
    if current is None:
        return not _pid_exists(pid)
    return None if current == started else True


class _DrainIdentityResolved(Exception):
    """Interrupt a numeric-PID wait once its original identity is no longer live."""

    def __init__(self, result):
        self.result = result


def _wait_for_original_exit(pid, started, timeout, on_progress):
    from hermes_cli import gateway

    def progress():
        result = _identity_exit_result(pid, started)
        if result is not None:
            raise _DrainIdentityResolved(result)
        on_progress()

    result = _identity_exit_result(pid, started)
    if result is not None:
        return result
    try:
        exited = gateway._wait_for_pid_exit(pid, timeout, on_progress=progress)
    except _DrainIdentityResolved as resolved:
        return resolved.result
    result = _identity_exit_result(pid, started)
    return exited if result is None else result


def begin_systemd_drain(pid: int, budget: float, label: str, *, self_restart_pending=None):
    """Signal now; return reconciliation work with a deadline measured from that signal.

    Ancestors retain their asynchronous self-restart bookkeeping. Wedged processes use
    the existing bounded escalation only during reconciliation, never blocking signals
    to healthy siblings. No signal is sent to serve units (their prepared PID is zero).
    """
    from hermes_cli import gateway, update_cmd_fleet as fleet
    from gateway.status import get_process_start_time

    if pid <= 0:
        return lambda: False
    if gateway._is_pid_ancestor_of_current_process(pid):
        accepted = fleet._drain_or_signal_gateway_for_update(
            pid, budget, label, self_restart_pending=self_restart_pending)
        return lambda: accepted
    started = get_process_start_time(pid)
    if started is None:
        # No process-targeting authority: reconcile through the already-scoped unit.
        return lambda: False
    if gateway.probe_gateway_loop_liveness(pid) == gateway.GATEWAY_LOOP_WEDGED:
        def reconcile_wedged():
            result = _identity_exit_result(pid, started)
            if result is not None:
                return result
            wedged = gateway.probe_gateway_loop_liveness(pid) == gateway.GATEWAY_LOOP_WEDGED
            result = _identity_exit_result(pid, started)
            if result is not None:
                return result
            if not wedged:
                return _signal_and_wait(pid, started, budget, label)()
            gateway._escalate_wedged_gateway(pid)
            result = _identity_exit_result(pid, started)
            return True if result is None else result

        return reconcile_wedged
    return _signal_and_wait(pid, started, budget, label)


def _signal_and_wait(pid, started, budget, label):
    from hermes_cli import update_cmd_fleet as fleet

    if not hasattr(signal, "SIGUSR1"):
        return lambda: False
    result = _identity_exit_result(pid, started)
    if result is not None:
        return lambda: result

    print(f"  → {label}: draining (up to {budget:.0f}s)...")
    deadline = time.monotonic() + max(budget, 1.0)
    try:
        os.kill(pid, signal.SIGUSR1)  # windows-footgun: ok — guarded by hasattr above
    except ProcessLookupError:
        return lambda: True
    except (PermissionError, OSError):
        return lambda: False

    from hermes_cli.update_cmd_drain_report import drain_progress_reporter
    on_progress = drain_progress_reporter(fleet._gateway_home_for_pid(pid), budget_s=budget)

    def wait_for_drain():
        return _wait_for_original_exit(
            pid, started, max(0.0, deadline - time.monotonic()), on_progress)

    return wait_for_drain
