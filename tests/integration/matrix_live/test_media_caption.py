"""A separate Matrix client sends media whose caption resembles a filename."""

from __future__ import annotations

import asyncio
import io
import json
import time

import pytest
from nio import RoomMessageText, RoomSendResponse, UploadResponse

from tests.integration.matrix_live.conftest import LiveGateway, LiveRoom


def test_media_caption_and_filename_reach_model_through_cache(
    gateway: LiveGateway,
    live_room: LiveRoom,
) -> None:
    payload = b"Live Matrix document content marker\n"
    caption = "clip.mp4"
    filename = "report.txt"

    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        try:
            await client.sync(timeout=0)
            uploaded, decryption = await client.upload(
                io.BytesIO(payload), content_type="text/plain", filename=filename, filesize=len(payload)
            )
            assert isinstance(uploaded, UploadResponse), uploaded
            assert decryption is None

            sent = await client.room_send(
                live_room.room_id,
                "m.room.message",
                {
                    "msgtype": "m.file",
                    "body": caption,
                    "filename": filename,
                    "url": uploaded.content_uri,
                    "info": {"mimetype": "text/plain", "size": len(payload)},
                },
            )
            assert isinstance(sent, RoomSendResponse), sent

            deadline = time.monotonic() + 15
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    response = await asyncio.wait_for(client.sync(timeout=250), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                joined = response.rooms.join.get(live_room.room_id)
                if not joined:
                    continue
                replies = [
                    (event.sender, event.body)
                    for event in joined.timeline.events
                    if isinstance(event, RoomMessageText) and event.sender != live_room.observer.user_id
                ]
                if not replies:
                    continue

                assert replies == [(live_room.bot.user_id, "Matrix live reply")]
                requests = gateway.model.main_requests()
                assert len(requests) == 1
                user_messages = [
                    message for message in requests[0]["messages"] if message["role"] == "user"
                ]
                assert len(user_messages) == 1
                document_note, user_text, *_ = user_messages[0]["content"].split("\n\n")
                assert user_text == caption

                cached = gateway.container.exec([
                    "find", "/opt/data/cache/documents", "-type", "f", "-name", f"doc_*_{filename}"
                ])
                assert cached.exit_code == 0
                paths = cached.output.decode().splitlines()
                assert len(paths) == 1
                content = gateway.container.exec(["cat", paths[0]])
                assert (content.exit_code, content.output) == (0, payload)
                assert f"'{filename}'" in document_note
                assert paths[0] in document_note
                return

            pytest.fail("No Matrix reply to the media event within 15 seconds")
        finally:
            await client.close()

    asyncio.run(exchange())


@pytest.mark.parametrize("gateway_extra_config", ["gateway:\n  max_inbound_media_bytes: 10\n"])
@pytest.mark.parametrize("payload, info", [
    pytest.param(b"small", {"mimetype": "text/plain", "size": 11}, id="declared-size"),
    pytest.param(b"eleven byte", {"mimetype": "text/plain"}, id="size-omitted"),
])
def test_oversized_media_exposes_caption_and_filename_without_caching(
    gateway: LiveGateway,
    live_room: LiveRoom,
    payload: bytes,
    info: dict[str, object],
) -> None:
    caption = "Please review this file"
    filename = "oversized.txt"

    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        try:
            await client.sync(timeout=0)
            uploaded, decryption = await client.upload(
                io.BytesIO(payload), content_type="text/plain", filename=filename, filesize=len(payload)
            )
            assert isinstance(uploaded, UploadResponse), uploaded
            assert decryption is None

            sent = await client.room_send(
                live_room.room_id,
                "m.room.message",
                {
                    "msgtype": "m.file",
                    "body": caption,
                    "filename": filename,
                    "url": uploaded.content_uri,
                    "info": info,
                },
            )
            assert isinstance(sent, RoomSendResponse), sent

            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                try:
                    response = await asyncio.wait_for(client.sync(timeout=250), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                joined = response.rooms.join.get(live_room.room_id)
                if not joined:
                    continue
                replies = [
                    (event.sender, event.body)
                    for event in joined.timeline.events
                    if isinstance(event, RoomMessageText) and event.sender != live_room.observer.user_id
                ]
                if not replies:
                    continue

                assert replies == [(live_room.bot.user_id, "Matrix live reply")]
                requests = gateway.model.main_requests()
                assert len(requests) == 1
                user_messages = [
                    message for message in requests[0]["messages"] if message["role"] == "user"
                ]
                assert len(user_messages) == 1
                model_context = json.dumps(user_messages[0]["content"])
                assert caption in model_context
                assert f"[matrix file attachment too large: {filename}]" in model_context

                cached = gateway.container.exec([
                    "/opt/hermes/.venv/bin/python", "-c",
                    "from pathlib import Path; "
                    f"assert not list(Path('/opt/data/cache/documents').glob('doc_*_{filename}'))",
                ])
                assert (cached.exit_code, cached.output) == (0, b"")
                return

            pytest.fail("No Matrix reply to the oversized media event within 15 seconds")
        finally:
            await client.close()

    asyncio.run(asyncio.wait_for(exchange(), timeout=20))


@pytest.mark.parametrize("failure, msgtype, mimetype, filename", [
    pytest.param("missing", "m.file", "text/plain", "missing.txt", id="homeserver-404"),
    pytest.param("invalid-image", "m.image", "image/png", "invalid.png", id="image-cache-rejection"),
    pytest.param("missing-key-metadata", "m.audio", "audio/ogg", "encrypted.ogg", id="encrypted-missing-hash"),
])
def test_failed_media_reaches_model_without_download_url_or_cache(
    gateway: LiveGateway,
    live_room: LiveRoom,
    failure: str,
    msgtype: str,
    mimetype: str,
    filename: str,
) -> None:
    caption = "Please inspect this attachment"
    kind = {"m.file": "file", "m.image": "image", "m.audio": "audio"}[msgtype]
    marker = f"[matrix {kind} attachment could not be downloaded: {filename}]"

    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        try:
            await client.sync(timeout=0)
            url = "mxc://matrix.test/missing-media-contract"
            if failure != "missing":
                payload = b"This payload is not an image or an audio container"
                uploaded, decryption = await client.upload(
                    io.BytesIO(payload), content_type=mimetype, filename=filename, filesize=len(payload),
                )
                assert isinstance(uploaded, UploadResponse), uploaded
                assert decryption is None
                url = uploaded.content_uri
            content = {
                "msgtype": msgtype, "body": caption, "filename": filename,
                "info": {"mimetype": mimetype},
            }
            if failure == "missing-key-metadata":
                content["file"] = {
                    "url": url, "key": {"k": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"},
                    "iv": "AAAAAAAAAAAAAAAAAAAAAA", "hashes": {}, "v": "v2",
                }
            else:
                content["url"] = url
            sent = await client.room_send(live_room.room_id, "m.room.message", content)
            assert isinstance(sent, RoomSendResponse), sent

            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                try:
                    response = await asyncio.wait_for(client.sync(timeout=250), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                joined = response.rooms.join.get(live_room.room_id)
                if not joined:
                    continue
                replies = [
                    (event.sender, event.body) for event in joined.timeline.events
                    if isinstance(event, RoomMessageText) and event.sender != live_room.observer.user_id
                ]
                if not replies:
                    continue
                assert replies == [(live_room.bot.user_id, "Matrix live reply")]
                requests = gateway.model.main_requests()
                assert len(requests) == 1
                user_messages = [
                    message for message in requests[0]["messages"] if message["role"] == "user"
                ]
                user_messages = [
                    {**message, "content": message["content"].split("\n\n[System note:", 1)[0]}
                    for message in user_messages
                ]
                assert user_messages == [{"role": "user", "content": f"{caption}\n{marker}"}]
                cached = gateway.container.exec([
                    "/opt/hermes/.venv/bin/python", "-c",
                    "from pathlib import Path; "
                    "files = [str(path) for kind in ('audio', 'documents', 'images') "
                    "for path in (Path('/opt/data/cache') / kind).rglob('*') if path.is_file()]; "
                    "assert files == [], files",
                ])
                assert (cached.exit_code, cached.output) == (0, b"")
                return
            pytest.fail("No Matrix reply to the failed media event within 15 seconds")
        finally:
            await client.close()

    asyncio.run(asyncio.wait_for(exchange(), timeout=20))
