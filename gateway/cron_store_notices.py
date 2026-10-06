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


def install_cron_store_notices(runner, loop: asyncio.AbstractEventLoop) -> None:
    """Route store transitions from the ticker thread onto the gateway loop. Installed once the
    adapters are connected, so a store that degraded during boot is announced then."""
    from cron.store_health import degraded_records, set_transition_listener

    def on_transition(event, record) -> None:
        asyncio.run_coroutine_threadsafe(send_cron_store_notice(runner, event, record), loop)

    set_transition_listener(on_transition)
    for record in degraded_records():
        on_transition("unwritable", record)


def _profile_home(profile):
    from hermes_constants import get_routing_process_hermes_home
    if profile is None:
        return get_routing_process_hermes_home()
    from hermes_cli.profiles import get_profile_dir
    return get_profile_dir(profile)


async def send_cron_store_notice(runner, event: str, record) -> None:
    """Post one ``unwritable``/``recovered`` notice to the home channels of the store's profile."""
    from gateway.run import _async_profile_runtime_scope
    from gateway.warning_notifications import present_notification
    from hermes_constants import hermes_home_key

    fields = record.notice_fields()
    key = "gateway.cron_store.unwritable" if event == "unwritable" else "gateway.cron_store.recovered"
    message = t(key, **fields)
    store_home = hermes_home_key(Path(record.store).parent)
    logger.info("Broadcasting cron store %s notice for %s", event, record.store)
    for profile, platform, _cfg, home, transport in list(runner._served_home_channel_transports()):
        profile_home = _profile_home(profile)
        if hermes_home_key(profile_home) != store_home:
            continue
        # The opt-out is the owning profile's; the launch profile needs no extra scope.
        scope = (_async_profile_runtime_scope(Path(profile_home)) if profile is not None
                 else contextlib.nullcontext())
        async with scope:
            await present_notification(
                lambda: runner._send_home_channel_message(
                    platform, home, transport, message, "Cron store notice failed for %s:%s: %s"),
                platform=platform)
