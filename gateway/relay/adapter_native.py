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


def drop_unrouted_profile_stamp(runner, event) -> None:
    """The connector's profile route (Team Gateway) selects among a multiplexed gateway's
    profiles; a single-profile gateway serves the event from its one profile. Clear the stamp
    before admission so nothing durable (admission, session origin) names a profile this gateway
    does not route by. Call after ``_capture_scope``, which keeps it for egress attribution."""
    if event.source.profile and not getattr(runner.config, 'multiplex_profiles', False):
        event.source.profile = None


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
    drop_unrouted_profile_stamp(runner, event)
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


async def dispatch_passthrough_event(adapter, event) -> None:
    """Hand a connector-forwarded interaction (already normalized to a ``MessageEvent``) to the
    ordinary adapter path. It arrived over the same authenticated relay socket as a text frame, so
    it carries the same callback provenance native admission requires; the WS reader installs that
    only around ``inbound`` frames, so the passthrough leg installs it here."""
    from gateway.session_ingress_context import relay_callback
    runner = getattr(adapter._message_handler, '__self__', None)
    if runner is not None:
        drop_unrouted_profile_stamp(runner, event)
    with relay_callback(adapter._transport, event):
        await adapter.handle_message(event)
