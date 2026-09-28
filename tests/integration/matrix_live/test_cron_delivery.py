"""A separate client observes actual cron delivery to aliases and threads."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

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

from tests.integration.matrix_live.conftest import LiveRoom


_DELIVER = """
import asyncio
import json
from cron.scheduler_delivery import _deliver_result
from gateway.config import Platform, load_gateway_config
from plugins.platforms.matrix.adapter import MatrixAdapter

async def main():
    config = load_gateway_config()
    adapter = MatrixAdapter(config.platforms[Platform.MATRIX])
    assert await adapter.connect(), 'Matrix connect failed'
    try:
        loop = asyncio.get_running_loop()
        job = {'id': 'matrix-live-cron', 'deliver': TARGET, 'attach_to_session': False}
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
                    "e2ee_mode": "off",
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
        result = sender.exec([
            "/opt/hermes/.venv/bin/python",
            "-c",
            "TARGET = " + repr(target) + "\n" + _DELIVER,
        ])
        output = result.output.decode(errors="replace")
        assert result.exit_code == 0, output
        errors = [
            json.loads(line)["error"]
            for line in output.splitlines()
            if line.startswith('{"error":')
        ]
        assert len(errors) == 1, output
        if destination in {"missing_alias", "mxid", "private_alias", "encrypted"}:
            assert errors[0] and alias in errors[0], output
            assert "live adapter" in errors[0] and "delivery error" in errors[0], output
        else:
            assert errors == [None], output

    async def observe():
        client = live_room.observer.client(live_room.homeserver)
        try:
            response = await client.sync(timeout=0)
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
