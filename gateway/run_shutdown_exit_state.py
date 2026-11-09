"""Serving gateway exit-state writes, kept outside the shutdown facade."""

from __future__ import annotations

import logging
from contextlib import suppress

logger = logging.getLogger("gateway.run")


async def persist_exit_state(runner, ctx) -> None:
    """Persist serving shutdown markers and terminal runtime status."""
    from gateway.run import _hermes_home, _shutdown_gateway_health_export
    if getattr(runner, "control_only", False):
        return
    if getattr(runner, "_startup_cleanup_only", False):
        return _shutdown_gateway_health_export(runner)
    if not ctx.timed_out:
        with suppress(Exception):
            (_hermes_home / ".clean_shutdown").touch()
    else:
        logger.info("Skipping .clean_shutdown marker — drain timed out with interrupted agents; "
                    "next startup will recover their interrupted turns.")
    if ctx.active_agents:
        runner._increment_restart_failure_counts(set(ctx.active_agents.keys()))
    if runner._restart_requested and runner._restart_command_source is None:
        from gateway.run_shutdown import _log_suppressed
        from gateway.restart_notice import write_planned_restart_marker
        with _log_suppressed(logging.DEBUG, "Failed to write planned restart notification marker: %s"):
            write_planned_restart_marker(
                via_service=bool(runner._restart_via_service),
                detached=bool(runner._restart_detached),
            )
    if runner._restart_requested and runner._restart_via_service:
        from gateway.run_shutdown import GATEWAY_SERVICE_RESTART_EXIT_CODE
        runner._exit_code = GATEWAY_SERVICE_RESTART_EXIT_CODE
        runner._exit_reason = runner._exit_reason or "Gateway restart requested"
    runner._draining = False
    if getattr(runner, "_signal_initiated_shutdown", False) and not runner._restart_requested:
        logger.info(
            "Gateway stopped by an unexpected signal — persisting "
            "gateway_state=running so container_boot auto-starts on the next boot (issue #42675)"
        )
        runner._update_runtime_status("running", runner._exit_reason)
    else:
        runner._update_runtime_status("stopped", runner._exit_reason)
    try:
        from gateway.status import flush_runtime_status_async
        flush_timeout = 2.0
        budget = getattr(runner, "_launchd_exit_timeout_s", None)
        signal_stop = getattr(runner, "_stop_requested_by_signal", False)
        if signal_stop and isinstance(budget, (int, float)) and budget > 0:
            flush_timeout = max(0.0, min(flush_timeout, budget - ctx.elapsed()))
        if not await flush_runtime_status_async(timeout=flush_timeout):
            logger.warning("Timed out flushing terminal gateway runtime status")
    except Exception:
        logger.debug("Failed to flush terminal gateway runtime status", exc_info=True)
    _shutdown_gateway_health_export(runner)
    logger.info("Gateway stopped (total teardown %.2fs)", ctx.elapsed())
