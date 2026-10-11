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
from tests.integration.matrix_live.context_client import hand_off


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
        "import asyncio, json\nfrom rich_content_client import accepted_exchange, paused_sticker, withdrawn_reply, paused_emote, queued_stickers\n"
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
        assert f"[emote by https://matrix.to/#/{live_room.observer.user_id}] /new waves" in emote_text
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
        hand_off(home / "expected-media-change", target)

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
        hand_off(home / "context-release", "release")
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
            *previous[:-1],
            prior_sticker,
        ]
        assert requests[1]["messages"] == previous
        current = requests[2]["messages"][-1]["content"]
        assert isinstance(current, str)
        assert "[redacted]" in current and "Withdrawn fox" not in current

        for marker in (
            "context-started",
            "context-release",
            "expected-media-change",
            "media-change-observed",
        ):
            (home / marker).unlink()
        output = linux_nio_observer.run_python(
            prefix
            + f"ROOT = {received['root']!r}\n"
            + "print(json.dumps(asyncio.run(asyncio.wait_for(paused_emote(ROOM, ROOT, encrypted=ENCRYPTED), timeout=10))))\n"
        )
        initiating = json.loads(output.strip().splitlines()[-1])
        _wait_for(
            lambda: (home / "context-started").exists(),
            "queued sticker preparation barrier",
            timeout=10,
        )
        output = linux_nio_observer.run_python(
            prefix
            + f"ROOT = {received['root']!r}\nURL = {received['url']!r}\n"
            + "print(json.dumps(asyncio.run(asyncio.wait_for(queued_stickers(ROOM, ROOT, URL, BOT, encrypted=ENCRYPTED), timeout=10))))\n"
        )
        retained, target = json.loads(output.strip().splitlines()[-1])
        # The handlers run concurrently, and the first sticker's handler also sends the busy
        # acknowledgement, so the handlers can finish in either order.
        _wait_for(
            lambda: (
                (home / "rich-events-queued").exists()
                and sorted(
                    (home / "rich-events-queued").read_text(encoding="utf-8").splitlines()
                )
                == sorted([retained, target])
            ),
            "both native stickers queued",
            timeout=10,
        )
        hand_off(
            home / "read-effective-event",
            json.dumps({
                "room": live_room.room_id,
                "event": retained,
                "sender": live_room.observer.user_id,
            }),
        )
        _wait_for(
            lambda: (home / "effective-event-read").exists(),
            "unchanged queued sticker read",
            timeout=10,
        )
        read = json.loads((home / "effective-event-read").read_text(encoding="utf-8"))
        assert read["errors"] == []
        assert read["events"][0]["event_id"] == retained
        assert live_room.bot.user_id in read["events"][0]["body"]
        hand_off(home / "expected-media-change", target)
        asyncio.run(asyncio.wait_for(redact(), timeout=5))
        _wait_for(
            lambda: (home / "media-change-observed").exists(),
            "second queued sticker withdrawal",
            timeout=10,
        )
        hand_off(home / "context-release", "release")
        linux_nio_observer.run_python(
            prefix
            + f"TARGET = {initiating!r}\nROOT = {received['root']!r}\n"
            + "asyncio.run(asyncio.wait_for(withdrawn_reply(ROOM, BOT, encrypted=ENCRYPTED, reply_target=TARGET, thread_root=ROOT, count=2), timeout=10))\n"
        )
        requests = gateway.model.main_requests()
        assert len(requests) == 5
        prior = requests[3]["messages"]
        assert requests[4]["messages"][: len(prior)] == prior
        assert requests[1]["messages"] == previous
        current = requests[4]["messages"][-1]["content"]
        assert isinstance(current, list)
        text = "".join(part.get("text", "") for part in current)
        assert "[sticker: Retained queued sticker]" in text
        assert "[redacted]" in text and "Withdrawn queued sticker" not in text
        assert f"[sticker: {live_room.bot.user_id}" not in text
        pixels = [
            part["image_url"]["url"] for part in current if part["type"] == "image_url"
        ]
        assert len(pixels) == 1
        assert base64.b64decode(pixels[0].split(",", 1)[1]).startswith(
            b"\x89PNG\r\n\x1a\n"
        )
    except Exception as exc:
        requests = [
            [
                {
                    **message,
                    "content": (
                        message["content"][:2000]
                        if isinstance(message["content"], str)
                        else message["content"]
                    ),
                }
                for message in request["messages"]
                if message["role"] != "system"
            ]
            for request in gateway.model.main_requests()
        ]
        logs = gateway.container.get_wrapped_container().logs().decode(errors="replace")
        gateway_log = tmp_path / "hermes" / "logs" / "gateway.log"
        if gateway_log.exists():
            logs += "\n" + gateway_log.read_text(encoding="utf-8", errors="replace")
        pytest.fail(
            f"{exc}\nNon-system model messages:\n{json.dumps(requests)}\nGateway log tail:\n{logs[-6000:]}",
            pytrace=False,
        )
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))
