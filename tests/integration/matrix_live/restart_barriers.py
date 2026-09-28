"""Native test plugin that interrupts real key import and admitted room dispatch."""

from __future__ import annotations

import asyncio
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
    def attach(client, _adapter):
        home = get_hermes_home()
        receive_key = client.crypto._receive_room_key
        accept_intake = client.sync_store.accept_intake
        accepted: dict[str, asyncio.Event] = {}

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
            await _barrier(home, "intake-blocked", event_id)

        client.crypto._receive_room_key = import_key
        client.sync_store.accept_intake = receipt
        client.add_event_handler(EventType.ROOM_MESSAGE, incomplete_sibling)

    ctx.register_platform_handler("matrix", attach)
