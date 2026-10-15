"""Native (session-authority) admission of a relay-delivered inbound event.

``RelayAdapter._on_inbound`` hands every non-command inbound here when the runner owns a
session authority: the event is committed to the durable FIFO before the frame is ACKed.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger("gateway.relay.adapter")

# Native admission refusals a redelivery cannot change (sender, route or payload).
_PERMANENT_REFUSALS = frozenset({'permission_denied', 'invalid_params', 'admission_conflict',
                                 'profile_mismatch', 'not_found'})


async def admit_native_inbound(adapter, runner, event) -> None:
    """Commit *event* through ``admit_producer``; returns once the SQLite receipt exists.

    Waits only for the receipt, never inference or outbound ACKs (those need the same WS
    reader). A permanent refusal returns, so the transport ACKs it (a replay would be refused
    again); any other failure raises and the ACK is withheld so the connector redelivers.
    """
    from gateway.platforms.webhook_ingress import admit_producer
    # The connector does not transmit a provider timestamp. Local arrival
    # time must not change the fingerprint of a retried provider identity.
    event.timestamp = datetime.fromtimestamp(0, UTC)
    if event.source.profile and not getattr(runner.config, 'multiplex_profiles', False):
        # The connector's profile route (Team Gateway) selects among a multiplexed
        # gateway's profiles; a single-profile gateway serves it from its one profile.
        # _capture_scope already kept the stamp for egress (persona attribution).
        event.source.profile = None
    try:
        receipt = await admit_producer(adapter, event)
    except RuntimeStoreError as exc:
        if exc.reason not in _PERMANENT_REFUSALS:
            raise
        logger.warning("relay inbound refused (%s) for %s chat=%s; acknowledged, not replayed",
                       exc.reason, event.source.platform.value, event.source.chat_id)
        return
    if receipt is None and getattr(event, '_consumer_reply', None):
        # A post_gateway_admission plugin consumed it (never committed); its reply is
        # sent off this reader, like an executed admission's.
        from gateway.session_ingress import deliver_response
        task = asyncio.create_task(deliver_response(
            adapter, event, runner._session_key_for_source(event.source), event._consumer_reply))
        adapter._background_tasks.add(task)
        task.add_done_callback(adapter._background_tasks.discard)
