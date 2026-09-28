"""Separate clients verify native actions, stickers and model input withdrawal."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from nio import RoomRedactResponse

from tests.integration.matrix_live.conftest import (
    LinuxNioObserver,
    LiveGateway,
    LiveRoom,
    _wait_for,
)


@pytest.fixture
def gateway_extra_config() -> str:
    return "display:\n  busy_input_mode: queue\n"


@pytest.mark.parametrize("gateway", ["pause-image-context"], indirect=True)
@pytest.mark.parametrize("encrypted", [False, True])
def test_native_emotes_and_stickers_reach_model_and_withdraw_only_new_input(
    tmp_path: Path,
    gateway: LiveGateway,
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
    record_property: Callable[[str, object], None],
    encrypted: bool,
) -> None:
    started = time.monotonic()
    prefix = (
        "import asyncio, json\nfrom rich_content_client import accepted_exchange, paused_sticker, withdrawn_reply\n"
        f"ROOM = {live_room.room_id!r}\nBOT = {live_room.bot.user_id!r}\n"
        f"DEVICE = {live_room.bot.device_id!r}\nENCRYPTED = {encrypted!r}\n"
    )
    try:
        output = linux_nio_observer.run_python(
            prefix
            + (
                "result = asyncio.run(asyncio.wait_for(accepted_exchange(ROOM, BOT, DEVICE, encrypted=ENCRYPTED), timeout=20))\n"
                "print(json.dumps(result))\n"
            )
        )
        received = json.loads(output.strip().splitlines()[-1])
        requests = gateway.model.main_requests()
        assert len(requests) == 2
        emote_text = json.dumps(requests[0]["messages"][-1]["content"])
        assert f"[emote by {live_room.observer.user_id}] /new waves" in emote_text
        sticker_parts = requests[1]["messages"][-1]["content"]
        assert isinstance(sticker_parts, list)
        assert "[sticker: Friendly fox.png]" in json.dumps(sticker_parts)
        pixels = [
            part["image_url"]["url"]
            for part in sticker_parts
            if part["type"] == "image_url"
        ]
        assert len(pixels) == 1
        assert base64.b64decode(pixels[0].split(",", 1)[1]).startswith(
            b"\x89PNG\r\n\x1a\n"
        )
        previous = requests[1]["messages"]
        output = linux_nio_observer.run_python(
            prefix
            + (
                f"ROOT = {received['root']!r}\nURL = {received['url']!r}\n"
                "event_id = asyncio.run(asyncio.wait_for(paused_sticker(ROOM, ROOT, URL, encrypted=ENCRYPTED), timeout=10))\n"
                "print(json.dumps(event_id))\n"
            )
        )
        target = json.loads(output.strip().splitlines()[-1])
        home = tmp_path / "hermes"
        _wait_for(
            lambda: (home / "context-started").exists(),
            "sticker context barrier",
            timeout=10,
        )
        (home / "expected-media-change").write_text(target, encoding="utf-8")

        async def redact() -> None:
            client = live_room.observer.client(live_room.homeserver)
            try:
                response = await client.room_redact(live_room.room_id, target)
                assert isinstance(response, RoomRedactResponse), response
            finally:
                await client.close()

        asyncio.run(asyncio.wait_for(redact(), timeout=5))
        _wait_for(
            lambda: (home / "media-change-observed").exists(),
            "sticker withdrawal barrier",
            timeout=10,
        )
        (home / "queued-context-release").write_text("release", encoding="utf-8")
        linux_nio_observer.run_python(
            prefix
            + (
                "asyncio.run(asyncio.wait_for(withdrawn_reply(ROOM, BOT, encrypted=ENCRYPTED), timeout=10))\n"
            )
        )
        requests = gateway.model.main_requests()
        assert len(requests) == 3
        prior_sticker = {
            **previous[-1],
            "content": sticker_parts[0]["text"] + "\n[screenshot]",
        }
        assert requests[2]["messages"][: len(previous)] == [
            *previous[:-1], prior_sticker,
        ]
        assert requests[1]["messages"] == previous
        current = requests[2]["messages"][-1]["content"]
        assert isinstance(current, str)
        assert "[redacted]" in current and "Withdrawn fox" not in current
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))
