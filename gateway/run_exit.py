"""Bounded process exit after gateway teardown."""

from __future__ import annotations


def exit_after_graceful_shutdown(exit_code: int, *, mark_lifecycle: bool = True) -> None:
    """Release process locks, flush serving telemetry when owned, and hard-exit."""
    from gateway.run import _best_effort, os, suppress, sys

    for stream in (sys.stdout, sys.stderr):
        with suppress(Exception):
            stream.flush()

    def _release_locks() -> None:
        from gateway.status import remove_pid_file, release_gateway_runtime_lock
        remove_pid_file()
        release_gateway_runtime_lock()

    def _mark_exited() -> None:
        if mark_lifecycle:
            from hermes_cli.observability.shared_metrics_process import stamp_exit
            stamp_exit("clean")
            from gateway.lifecycle_ledger import mark_exited
            mark_exited(exit_code, reason="graceful_shutdown")

    def _flush_metrics() -> None:
        from hermes_cli.observability.shared_metrics_process import flush_before_hard_exit
        flush_before_hard_exit()

    def _write_cpu_profile() -> None:
        from hermes_cli.cpu_profile import write_cpu_profile
        write_cpu_profile()

    def _drain_logs() -> None:
        from hermes_logging import drain_log_queue
        drain_log_queue(timeout=1.0)

    steps = [_release_locks]
    if mark_lifecycle:
        steps.extend((_mark_exited, _flush_metrics))
    steps.extend((_write_cpu_profile, _drain_logs))
    for step in steps:
        _best_effort(step)
    os._exit(exit_code)
