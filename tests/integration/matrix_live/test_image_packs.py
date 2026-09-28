"""Separate clients verify standard and legacy packs and selected native stickers."""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import pytest

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.conftest import (
    LinuxNioObserver,
    LiveGateway,
    LiveRoom,
)


@pytest.mark.parametrize("gateway", ["image-packs"], indirect=True)
@pytest.mark.parametrize("transport", ["plain", "encrypted", "transition"])
def test_current_bot_packs_send_native_sticker_with_thread_and_reply_context(
    gateway: LiveGateway,
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
    transport: str,
    record_property: Callable[[str, object], None],
) -> None:
    started = time.monotonic()
    encrypted = transport == "encrypted"
    prefix = (
        "import asyncio, json\nfrom image_packs_client import prepare, ask, reply_to_sticker, enable_encryption\n"
        f"ROOM={live_room.room_id!r}\nBOT={live_room.bot.user_id!r}\nDEVICE={live_room.bot.device_id!r}\n"
        f"BOT_TOKEN={live_room.bot.access_token!r}\nENCRYPTED={encrypted!r}\n"
    )
    try:
        output = linux_nio_observer.run_python(
            prefix
            + (
                "result = asyncio.run(asyncio.wait_for(prepare(ROOM, BOT, DEVICE, BOT_TOKEN, encrypted=ENCRYPTED), timeout=20))\n"
                "print(json.dumps(result))\n"
            )
        )
        prepared = json.loads(output.strip().splitlines()[-1])
        prefix += f"ROOT={prepared['root']!r}\nURL={prepared['url']!r}\n"
        gateway.model.push(
            ToolCall("tool_search", {"queries": ["Matrix image pack stickers"]}),
            ToolCall(
                "tool_call",
                {
                    "calls": [
                        {"name": "matrix_image_packs", "arguments": {"action": "list"}}
                    ]
                },
            ),
            Text("Packs listed"),
        )
        linux_nio_observer.run_python(
            prefix
            + (
                "asyncio.run(asyncio.wait_for(ask(ROOM, BOT, ROOT, 'List sticker packs', 'Packs listed', encrypted=ENCRYPTED), timeout=15))\n"
            )
        )
        requests = gateway.model.main_requests()
        assert len(requests) == 4
        results = [
            json.loads(message["content"])
            for message in requests[-1]["messages"]
            if message["role"] == "tool"
        ]
        catalog = results[-1]
        assert {key: value for key, value in catalog.items() if key != "packs"} == {
            "errors": [],
            "truncated": False,
            "untrusted_data": True,
            "account_user_id": live_room.bot.user_id,
        }
        assert sorted(
            (
                pack["source"],
                pack["event_type"],
                pack["state_key"],
                pack["account_user_id"],
                pack["items"][0]["body"],
            )
            for pack in catalog["packs"]
        ) == sorted([
            ("room", "m.room.image_pack", "", None, "Room fox.png"),
            ("room", "m.room.image_pack", "named", None, "Referenced fox.png"),
            ("room", "im.ponies.room_emotes", "legacy", None, "Legacy fox.png"),
            (
                "account_reference",
                "m.room.image_pack",
                "named",
                live_room.bot.user_id,
                "Referenced fox.png",
            ),
            (
                "account_reference",
                "im.ponies.room_emotes",
                "legacy",
                live_room.bot.user_id,
                "Legacy fox.png",
            ),
            (
                "bot_account",
                "im.ponies.user_emotes",
                None,
                live_room.bot.user_id,
                "Bot private fox.png",
            ),
        ])
        selected = next(
            pack["items"][0]
            for pack in catalog["packs"]
            if pack["source"] == "account_reference"
            and pack["event_type"] == "m.room.image_pack"
        )
        if transport == "transition":
            linux_nio_observer.run_python(
                prefix
                + "asyncio.run(asyncio.wait_for(enable_encryption(ROOM, BOT, DEVICE), timeout=15))\n"
            )
            prefix += "ENCRYPTED=True\n"
        gateway.model.push(
            ToolCall(
                "tool_call",
                {
                    "calls": [
                        {
                            "name": "matrix_image_packs",
                            "arguments": {
                                "action": "send",
                                "selection_id": selected["selection_id"],
                            },
                        }
                    ]
                },
            ),
            Text("Sticker sent"),
        )
        output = linux_nio_observer.run_python(
            prefix
            + (
                "result = asyncio.run(asyncio.wait_for(ask(ROOM, BOT, ROOT, 'Send the selected sticker', 'Sticker sent', "
                "encrypted=ENCRYPTED, sticker_body='Referenced fox.png', url=URL), timeout=15))\nprint(json.dumps(result))\n"
            )
        )
        sent = json.loads(output.strip().splitlines()[-1])
        requests = gateway.model.main_requests()
        assert len(requests) == 6
        tools = [
            json.loads(message["content"])
            for message in requests[-1]["messages"]
            if message["role"] == "tool"
        ]
        assert tools[-1] == {"success": True, "event_id": sent["sticker"]}
        gateway.model.push(Text("Sticker context checked"))
        linux_nio_observer.run_python(
            prefix
            + (
                f"STICKER={sent['sticker']!r}\n"
                "asyncio.run(asyncio.wait_for(reply_to_sticker(ROOM, BOT, ROOT, STICKER, encrypted=ENCRYPTED), timeout=15))\n"
            )
        )
        requests = gateway.model.main_requests()
        assert (
            len(requests),
            len(gateway.model.aux_requests()),
            len(gateway.model.requests),
        ) == (7, 0, 7)
        inbound = requests[-1]["messages"][-1]["content"]
        assert "Referenced fox.png" in json.dumps(inbound)
        assert "Requester private sentinel" not in json.dumps(requests)
    except Exception as exc:
        requests = gateway.model.main_requests()
        latest_tools = [
            message
            for message in (requests[-1]["messages"] if requests else [])
            if message["role"] == "tool"
        ]
        gateway_output = (
            gateway.container
            .get_wrapped_container()
            .logs()
            .decode(errors="replace")[-6000:]
        )
        gateway_log = gateway.container.exec([
            "tail",
            "-c",
            "6000",
            "/opt/data/logs/gateway.log",
        ]).output.decode(errors="replace")
        raise AssertionError(
            f"{exc}\nLatest model tool outputs: {latest_tools!r}\n"
            f"Gateway log tail:\n{gateway_log}\nGateway output tail:\n{gateway_output}"
        ) from exc
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))
