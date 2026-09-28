"""A separate client observes actual cron delivery to aliases and threads."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest
from nio import (
    JoinResponse,
    RoomCreateResponse,
    RoomMessageText,
    RoomPreset,
    RoomPutStateResponse,
    RoomSendResponse,
)
from testcontainers.core.container import DockerContainer

from tests.integration.matrix_live.conftest import LiveRoom, _wait_for


_DELIVER = """
import asyncio
import json
from cron.scheduler_delivery import _deliver_result
from gateway.config import Platform, load_gateway_config
from plugins.platforms.matrix.adapter import MatrixAdapter
from pathlib import Path

async def main():
    config = load_gateway_config()
    adapter = MatrixAdapter(config.platforms[Platform.MATRIX])
    assert await adapter.connect(), 'Matrix connect failed'
    try:
        loop = asyncio.get_running_loop()
        job = {'id': 'matrix-live-cron', 'deliver': TARGET, 'attach_to_session': False}
        if UPLOAD_TRANSITION:
            from types import SimpleNamespace
            from tools import send_message_senders
            from tools.send_message_tool import _send_to_platform
            from plugins.platforms.matrix.standalone import _HTTPDelivery
            send_message_senders._live_adapter = lambda *args, **kwargs: (SimpleNamespace(_gateway_loop=loop), adapter)
            async def reject_http(*args, **kwargs):
                raise AssertionError('Encrypted SDK media must not use plaintext HTTP')
            _HTTPDelivery.request = reject_http
            assert adapter._encryption and adapter._client.crypto
            original_upload = adapter._client.upload_media
            async def paused_upload(*args, **kwargs):
                uri = await original_upload(*args, **kwargs)
                Path('/opt/data/upload-ready').touch()
                async with asyncio.timeout(10):
                    while not Path('/opt/data/release-upload').exists():
                        await asyncio.sleep(0.01)
                return uri
            adapter._client.upload_media = paused_upload
            attachment = Path('/opt/data/report.txt')
            attachment.write_bytes(b'Private cron attachment')
            result = await _send_to_platform(
                Platform.MATRIX, adapter.config, TARGET.removeprefix('matrix:').split('/')[0], '',
                media_files=[(str(attachment), False)], thread_id=THREAD,
            )
            error = result.get('error')
        else:
            error = await asyncio.to_thread(
                _deliver_result, job, 'Cron alias report',
                {Platform.MATRIX: adapter}, loop,
            )
        print(json.dumps({'error': error}))
    finally:
        await adapter.disconnect()

asyncio.run(main())
"""


@pytest.mark.parametrize(
    "destination",
    [
        "alias",
        "home",
        "home_colon",
        "room",
        "password",
        "missing_alias",
        "mxid",
        "private_alias",
        "encrypted",
        "upload_encryption",
    ],
)
def test_cron_alias_and_home_thread_delivery(
    gateway_image,
    synapse,
    live_room: LiveRoom,
    tmp_path: Path,
    destination,
):
    async def create():
        client = live_room.observer.client(live_room.homeserver)
        try:
            room = await client.room_create(
                alias="cron-target",
                name="Explicit cron destination",
                preset=RoomPreset.private_chat
                if destination == "private_alias"
                else RoomPreset.public_chat,
            )
            assert isinstance(room, RoomCreateResponse), room
            if destination == "encrypted":
                state = await client.room_put_state(
                    room.room_id,
                    "m.room.encryption",
                    {"algorithm": "m.megolm.v1.aes-sha2"},
                )
                assert isinstance(state, RoomPutStateResponse), state
            if destination == "room":
                bot = live_room.bot.client(live_room.homeserver)
                try:
                    joined = await bot.join(room.room_id)
                    assert isinstance(joined, JoinResponse), joined
                finally:
                    await bot.close()
            root = await client.room_send(
                room.room_id,
                "m.room.message",
                {"msgtype": "m.notice", "body": "Cron thread root"},
            )
            assert isinstance(root, RoomSendResponse), root
            return room.room_id, root.event_id
        finally:
            await client.close()

    room_id, root_id = asyncio.run(create())
    alias = "#cron-target:matrix.test"
    if destination == "missing_alias":
        alias = "#missing:matrix.test"
    if destination == "mxid":
        alias = live_room.observer.user_id
    inline = (
        f"{alias}:${root_id[1:]}"
        if destination == "home_colon"
        else f"{alias}/{root_id}"
    )
    target = (
        "matrix"
        if destination in {"home", "home_colon"}
        else f"matrix:{room_id if destination == 'room' else alias}/{root_id}"
    )
    home = tmp_path / "home"
    home.mkdir()
    config = {
        "cron": {"wrap_response": False},
        "platforms": {
            "matrix": {
                "enabled": True,
                "token": ""
                if destination == "password"
                else live_room.bot.access_token,
                "home_channel": {"platform": "matrix", "chat_id": inline},
                "extra": {
                    "homeserver": "http://synapse:8008",
                    "e2ee_mode": "optional"
                    if destination == "upload_encryption"
                    else "off",
                    "allowed_users": [live_room.observer.user_id],
                    "user_id": live_room.bot.user_id,
                    "password": "matrix-test-password"
                    if destination == "password"
                    else "",
                },
            }
        },
    }
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    _, _, network = synapse
    with DockerContainer(
        gateway_image,
        network=network,
        entrypoint="/bin/sleep",
        command="infinity",
        working_dir="/opt/hermes",
    ).with_volume_mapping(home, "/opt/data", "rw") as sender:
        command = [
            "/opt/hermes/.venv/bin/python",
            "-c",
            f"TARGET = {target!r}\nTHREAD = {root_id!r}\n"
            f"UPLOAD_TRANSITION = {destination == 'upload_encryption'!r}\n" + _DELIVER,
        ]
        if destination == "upload_encryption":
            with ThreadPoolExecutor(max_workers=1) as executor:
                pending = executor.submit(sender.exec, command)
                _wait_for(
                    lambda: (home / "upload-ready").exists(),
                    "native upload barrier",
                    timeout=120,
                )

                async def enable_encryption():
                    client = live_room.observer.client(live_room.homeserver)
                    try:
                        state = await client.room_put_state(
                            room_id,
                            "m.room.encryption",
                            {"algorithm": "m.megolm.v1.aes-sha2"},
                        )
                        assert isinstance(state, RoomPutStateResponse), state
                    finally:
                        await client.close()

                asyncio.run(asyncio.wait_for(enable_encryption(), timeout=5))
                (home / "release-upload").touch()
                result = pending.result(timeout=15)
        else:
            result = sender.exec(command)
        output = result.output.decode(errors="replace")
        assert result.exit_code == 0, output
        errors = [
            json.loads(line)["error"]
            for line in output.splitlines()
            if line.startswith('{"error":')
        ]
        assert len(errors) == 1, output
        if destination == "upload_encryption":
            assert (
                errors[0] and alias in errors[0] and "encryption changed" in errors[0]
            ), output
        elif destination in {"missing_alias", "mxid", "private_alias", "encrypted"}:
            assert errors[0] and alias in errors[0], output
            assert "live adapter" in errors[0] and "delivery error" in errors[0], output
        else:
            assert errors == [None], output

    async def observe():
        client = live_room.observer.client(live_room.homeserver)
        try:
            response = await client.sync(timeout=0)
            if destination == "upload_encryption":
                bot_events = [
                    event
                    for event in response.rooms.join[room_id].timeline.events
                    if event.sender == live_room.bot.user_id
                    and event.source["type"] in {"m.room.message", "m.room.encrypted"}
                ]
                assert bot_events == []
                return
            events = [
                event
                for event in response.rooms.join[room_id].timeline.events
                if isinstance(event, RoomMessageText)
                and event.sender == live_room.bot.user_id
            ]
            if destination in {"missing_alias", "mxid", "private_alias", "encrypted"}:
                assert events == []
                return
            assert [
                (event.body, event.source["content"]["m.relates_to"])
                for event in events
            ] == [
                (
                    "Cron alias report",
                    {
                        "rel_type": "m.thread",
                        "event_id": root_id,
                        "is_falling_back": True,
                        "m.in_reply_to": {"event_id": root_id},
                    },
                )
            ]
        finally:
            await client.close()

    asyncio.run(asyncio.wait_for(observe(), timeout=15))
