"""Native test plugin that interrupts real key import and admitted room dispatch."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from hermes_constants import get_hermes_home
from mautrix.types import EventType


async def _barrier(home: Path, marker: str, value: str) -> None:
    await asyncio.to_thread((home / marker).write_text, value, encoding="utf-8")
    try:
        await asyncio.Event().wait()
    finally:
        await asyncio.to_thread(
            (home / f"{marker}-cancelled").write_text, value, encoding="utf-8"
        )


def register(ctx) -> None:
    def attach(client, adapter):
        home = get_hermes_home()
        receive_key = client.crypto._receive_room_key
        accept_intake = client.sync_store.accept_intake
        accepted: dict[str, asyncio.Event] = {}

        async def diagnostics(marker, event_id):
            store = client.sync_store
            data = {
                "event_id": event_id,
                "adapter_running": adapter._running,
                "cursor": await store.get_next_batch(),
                "accepted_events": sorted(store._accepted_events),
                "pending_batches": [
                    str(event.message_id)
                    for event in adapter._pending_text_batches.values()
                ],
            }
            await asyncio.to_thread(
                (home / f"{marker}-diagnostics.json").write_text,
                json.dumps(data),
                encoding="utf-8",
            )

        async def import_key(event):
            await receive_key(event)
            if not (home / "key-import-blocked").exists():
                await _barrier(
                    home, "key-import-blocked", str(event.content.session_id)
                )
                return
            await asyncio.to_thread(
                (home / "key-import-restored").write_text,
                str(event.content.session_id),
                encoding="utf-8",
            )

        async def receipt(event_id):
            await accept_intake(event_id)
            await diagnostics("intake-accepted", event_id)
            accepted.setdefault(event_id, asyncio.Event()).set()

        async def incomplete_sibling(event):
            if (
                event.sender == client.mxid
                or not (home / "room-input-release").exists()
                or (home / "intake-blocked").exists()
                or event.content.body != "Watch the split answer"
            ):
                return
            event_id = str(event.event_id)
            await accepted.setdefault(event_id, asyncio.Event()).wait()
            await diagnostics("intake-blocked", event_id)
            await _barrier(home, "intake-blocked", event_id)

        flush_batch = adapter._flush_text_batch

        async def buffered_input(key):
            pending = adapter._pending_text_batches.get(key)
            if pending is None or pending.text != "Prime encrypted thread":
                await flush_batch(key)
                return
            if not (home / "text-buffered").exists():
                assert adapter._text_batch_delay_seconds == 0.6
                await diagnostics("text-buffered", str(pending.message_id))
                await _barrier(home, "text-buffered", str(pending.message_id))
                return
            await adapter._flush_text_batch_now(key)

        handle_message = adapter.handle_message

        async def startup_replay(event):
            await handle_message(event)
            if getattr(event, "_hermes_startup_restore_replay", False) is True:
                assert event._gateway_accepted is True
                await diagnostics("startup-replayed", str(event.message_id))

        owner = type(adapter)(adapter.config)
        owner._client = client
        owner._store_dir = adapter._store_dir
        owner._user_id = adapter._user_id
        send_final = adapter.send_final_ledgered

        async def replacement_final(
            event, session_key, text_content, metadata, **kwargs
        ):
            adapter._final_delivery_adapter = lambda _source: owner
            result, delivery_adapter = await send_final(
                event,
                session_key,
                text_content,
                metadata,
                **kwargs,
            )
            assert delivery_adapter is owner
            if result.success and result.message_id:
                ids = (*result.continuation_message_ids, result.message_id)
                assert (
                    adapter._followup_delivery_events.latest(event.source.chat_id, ids)
                    is None
                )
                assert owner._followup_delivery_events.latest(event.source.chat_id, ids)
                await asyncio.to_thread(
                    (home / "replacement-delivery").write_text,
                    "\n".join(ids),
                    encoding="utf-8",
                )
            return result, delivery_adapter

        adapter._flush_text_batch = buffered_input
        adapter.handle_message = startup_replay
        adapter.send_final_ledgered = replacement_final
        client.crypto._receive_room_key = import_key
        client.sync_store.accept_intake = receipt
        client.add_event_handler(EventType.ROOM_MESSAGE, incomplete_sibling)

    ctx.register_platform_handler("matrix", attach)
