"""Native admission of a connector-delivered relay message (``RelayAdapter._on_inbound``)."""
import asyncio
import logging

from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger("gateway.relay.adapter")


async def admit_relay_delivery(adapter, runner, event):
    """Admit *event*; the transport ACKs the frame once this returns, or once the returned task does.

    Without producer hooks admission waits only for the SQLite receipt, on the socket reader. With
    one, ``pre_gateway_dispatch`` or ``post_gateway_admission`` plugin code runs before the commit and
    may await ``adapter.send()``, whose ``outbound_result`` only that reader delivers, or wait for
    another inbound message: the admission then runs in a task off the reader, after any earlier
    admission of the same chat, so one chat's deliveries still commit in arrival order."""
    from gateway.run_inbound_consumer import POST_ADMISSION_HOOK
    from hermes_cli.lifecycle import has_hook
    if not (has_hook('pre_gateway_dispatch') or has_hook(POST_ADMISSION_HOOK)):
        await _admit(adapter, runner, event)
        return None
    lanes = adapter._admission_lanes
    lane = (event.source.platform.value, event.source.chat_id)
    task = asyncio.create_task(_admit_after(lanes.get(lane), adapter, runner, event))
    lanes[lane] = task
    task.add_done_callback(lambda done: lanes.pop(lane) if lanes.get(lane) is done else None)
    adapter._background_tasks.add(task)  # cancelled with the adapter's work on shutdown
    task.add_done_callback(adapter._background_tasks.discard)
    return task


async def _admit_after(previous, adapter, runner, event):
    if previous is not None:
        await asyncio.wait([previous])  # its outcome is its own frame's ACK, not this one's
    await _admit(adapter, runner, event)


async def _admit(adapter, runner, event):
    # A permanent refusal returns, so the transport ACKs it (a replay would be refused again: main's
    # runner dropped it too); any other failure withholds the ACK so the connector redelivers.
    from gateway.platforms.webhook_ingress import admit_producer
    from gateway.relay.adapter import _PERMANENT_REFUSALS
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
