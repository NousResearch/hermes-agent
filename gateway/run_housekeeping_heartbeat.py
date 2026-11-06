"""The housekeeping "Runtime heartbeat" chore: re-stamp ``gateway_state.json`` every tick.

Besides refreshing ``updated_at``, the stamp re-publishes the live ``active_agents`` aggregate.
``_persist_active_agents`` writes the count only at messaging-turn boundaries, but the aggregate
(``_active_work_count``) also sums cron jobs, API-server runs and deferred workers, which start
and end without a write (#122813). A turn that ends while a cron job runs persists 1; the job
finishes; without this the file keeps 1 under a seconds-fresh ``updated_at`` until the next inbound
turn, and every external idle gate reading it reports the gateway busy. Recomputing on the tick
bounds that staleness to one housekeeping interval for every work source, current or future.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def runtime_heartbeat(runner: Any) -> None:
    """Stamp the heartbeat; with a runner, also publish its live work count.

    An unreadable count still stamps ``updated_at`` (the wedge detector keys off its age) and
    keeps the last published value.
    """
    from gateway.run import _write_runtime_status_quiet

    fields: dict[str, Any] = {}
    if runner is not None:
        try:
            fields["active_agents"] = runner._active_work_count()
        except Exception:
            logger.debug("Runtime heartbeat: active work count unavailable", exc_info=True)
    _write_runtime_status_quiet(**fields)
