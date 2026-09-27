"""Persist Synapse event times from an independent Matrix client for text and media."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time

import pytest
from nio import AsyncClient, RoomGetEventResponse, RoomSendResponse, UploadResponse

from tests.integration.matrix_live.conftest import LiveGateway, LiveRoom


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/"
    "oAAAAABJRU5ErkJggg=="
)
_READ_USER_ROW = """
import json
import sqlite3
import sys

with sqlite3.connect('/opt/data/state.db') as db:
    row = db.execute(
        "SELECT timestamp, content FROM messages "
        "WHERE role = 'user' AND platform_message_id = ? ORDER BY id DESC LIMIT 1",
        (sys.argv[1],),
    ).fetchone()
print(json.dumps(row))
"""


def _persisted_user_row(gateway: LiveGateway, event_id: str) -> tuple[float, str] | None:
    result = gateway.container.exec(["/opt/hermes/.venv/bin/python", "-c", _READ_USER_ROW, event_id])
    output = result.output.decode(errors="replace")
    assert result.exit_code == 0, output
    row = json.loads(output)
    return (float(row[0]), row[1]) if row is not None else None


async def _assert_persisted_event(
    client: AsyncClient, gateway: LiveGateway, room_id: str, sent: RoomSendResponse, marker: str,
) -> None:
    received = await client.room_get_event(room_id, sent.event_id)
    assert isinstance(received, RoomGetEventResponse), received
    server_time = received.event.source["origin_server_ts"] / 1000

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        row = _persisted_user_row(gateway, sent.event_id)
        if row is not None:
            timestamp, content = row
            assert marker in content
            assert timestamp == pytest.approx(server_time, rel=0, abs=0.000001)
            return
        await asyncio.sleep(0.25)
    pytest.fail(
        f"No persisted Matrix user row for {sent.event_id}. Gateway logs:\n"
        + gateway.container.get_wrapped_container().logs().decode(errors="replace")[-6000:]
    )


def test_independent_client_server_timestamps_reach_persisted_turns(
    gateway: LiveGateway,
    live_room: LiveRoom,
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        try:
            await client.sync(timeout=0)
            text = await client.room_send(
                live_room.room_id, "m.room.message",
                {"msgtype": "m.text", "body": "Server timestamp [in:text]"},
            )
            assert isinstance(text, RoomSendResponse), text
            await _assert_persisted_event(client, gateway, live_room.room_id, text, "[in:text]")

            upload, _ = await client.upload(
                io.BytesIO(_PNG), content_type="image/png", filename="pixel.png", filesize=len(_PNG),
            )
            assert isinstance(upload, UploadResponse), upload
            image = await client.room_send(
                live_room.room_id, "m.room.message",
                {
                    "msgtype": "m.image", "body": "Server timestamp [in:image]",
                    "url": upload.content_uri,
                    "info": {"mimetype": "image/png", "size": len(_PNG)},
                },
            )
            assert isinstance(image, RoomSendResponse), image
            await _assert_persisted_event(client, gateway, live_room.room_id, image, "[in:image]")
        finally:
            await client.close()

    asyncio.run(exchange())
