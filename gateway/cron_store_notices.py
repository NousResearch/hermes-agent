"""Home-channel notices for cron store degraded-state transitions (see ``cron/store_health.py``).

The cron tick runs on a worker thread inside the gateway. When a profile's cron store becomes
unwritable it reports ONE transition, and again when it recovers. Each one is posted to that
profile's home channels, honouring that profile's warning-notification opt-out, through the same
send path the state.db warning uses.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

from agent.i18n import t

logger = logging.getLogger(__name__)

# A store flapping across the disk-full line must not post a notice pair every tick: "recovered"
# waits this long and is dropped if the store degrades again meanwhile.
NOTICE_REPEAT_SECONDS = 3600.0


def install_cron_store_notices(runner, loop: asyncio.AbstractEventLoop) -> None:
    """Route store transitions from the ticker thread onto the gateway loop. Installed once the
    adapters are connected, so a store that degraded during boot is announced then. The last
    notice a home channel sees always matches the store's real state."""
    from cron.store_health import degraded_record, degraded_records, set_transition_listener

    # Loop-only state (mutated in loop callbacks, never on the ticker thread).
    announced: set[str] = set()  # stores whose current outage got an "unwritable" notice
    pending: dict[str, asyncio.TimerHandle] = {}  # store -> its deferred "recovered" notice

    def send(event, record) -> None:
        task = loop.create_task(send_cron_store_notice(runner, event, record))
        task.add_done_callback(lambda done: _log_notice_failure(done, event, record.store))

    def send_recovered(record) -> None:
        pending.pop(record.store, None)
        if degraded_record(Path(record.store)) is None:  # else it degraded again: outage goes on
            announced.discard(record.store)
            send("recovered", record)

    def on_loop(event, record) -> None:
        store = record.store
        if event == "unwritable":
            if (handle := pending.pop(store, None)) is not None:
                handle.cancel()  # never told it recovered: the outage notice still stands
            elif store not in announced and degraded_record(Path(store)) is not None:
                announced.add(store)  # recheck: a boot replay can land after the store recovered
                send(event, record)
        elif store in announced and store not in pending:  # no lone "recovered"
            pending[store] = loop.call_later(NOTICE_REPEAT_SECONDS, send_recovered, record)

    def on_transition(event, record) -> None:  # the ticker thread
        # A closed loop raises before any coroutine exists; store_health._notify logs and swallows it.
        loop.call_soon_threadsafe(on_loop, event, record)

    set_transition_listener(on_transition)
    for record in degraded_records():
        on_transition("unwritable", record)


def _log_notice_failure(done, event: str, store: str) -> None:
    try:
        done.result()
    except asyncio.CancelledError:
        return
    except Exception:  # the ticker thread already moved on: log, never raise
        logger.warning("Cron store %s notice for %s failed", event, store, exc_info=True)


async def send_cron_store_notice(runner, event: str, record) -> None:
    """Post one ``unwritable``/``recovered`` notice to the home channels of the store's profile."""
    from gateway.run import _async_profile_runtime_scope
    from gateway.warning_notifications import present_notification
    from hermes_constants import get_routing_process_hermes_home, hermes_home_key

    fields = record.notice_fields()
    key = "gateway.cron_store.unwritable" if event == "unwritable" else "gateway.cron_store.recovered"
    message = t(key, **fields)
    store_home = hermes_home_key(Path(record.store).parent)
    served_homes = getattr(runner, "_served_profile_homes", None) or {}
    logger.info("Broadcasting cron store %s notice for %s", event, record.store)
    for profile, platform, _cfg, home, transport in list(runner._served_home_channel_transports()):
        served_home = served_homes.get(profile) if profile is not None else None
        if hermes_home_key(served_home or get_routing_process_hermes_home()) != store_home:
            continue
        # The opt-out is the owning profile's; the launch profile needs no extra scope.
        scope = _async_profile_runtime_scope(Path(served_home)) if served_home else contextlib.nullcontext()
        async with scope:
            await present_notification(
                lambda: runner._send_home_channel_message(
                    platform, home, transport, message, "Cron store notice failed for %s:%s: %s"),
                platform=platform)
