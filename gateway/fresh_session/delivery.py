"""The published boundary is immediate; its quiet note follows the owning reply's delivery."""
import asyncio

from gateway.platforms.base import BasePlatformAdapter, _split_post_delivery_entry
from gateway.session_authorities import active_authority
from gateway.session_ingress import _execution_scope, _registry


async def send_note(runner, source):
    await runner._delivery_adapter_for(source).send(
        source.chat_id, 'Continuing in a fresh session in this topic.',
        metadata={'thread_id': str(source.thread_id), '_interim_send': True})


def defer_note(runner, source, key, generation, terminal):
    authority = active_authority(runner)
    admission_id = terminal[0] if terminal else None
    delivery_event = None
    if authority is not None and admission_id:
        delivery_event = getattr(authority, 'marker_adopters', {}).get(admission_id)
        pending = authority.pending_deliveries.get(admission_id)
        if delivery_event is None and pending is not None:
            delivery_event = pending[1]

    delivery_done = None
    if delivery_event is not None:
        delivery_done = delivery_event._hermes_reply_delivery_done = asyncio.Event()

    async def notify():
        if delivery_done is not None:
            await delivery_done.wait()
            if not getattr(delivery_event, '_hermes_reply_delivered', False):
                return
        if authority is None:
            return await send_note(runner, source)
        with await _execution_scope(authority, None):
            await send_note(runner, source)

    adapter = runner._delivery_adapter_for(source)
    if authority is not None and admission_id:
        slots = _registry(authority, 'post_delivery')
        previous = slots.get(admission_id)
        callback = (BasePlatformAdapter._chain_callbacks(_split_post_delivery_entry(previous[2])[1], notify)
                    if previous else notify)
        slots[admission_id] = (previous[0] if previous else {}, key, (generation, callback))
    else:
        adapter.register_post_delivery_callback(key, notify, generation=generation)
