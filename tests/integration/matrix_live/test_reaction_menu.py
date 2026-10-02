"""An independent Matrix client selects a menu through the real Linux gateway."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from nio import RoomSendResponse

from tests.fakes.fake_llm_provider import Response, Text, ToolCall
from tests.integration.matrix_live.conftest import LinuxNioObserver, LiveGateway, LiveRoom, _register


@pytest.fixture
def gateway_script():
    arguments = {
        "prompt": "Choose the next route", "context_id": "live-route",
        "options": [
            {"emoji": "✅", "label": "First route", "payload": "Take the first route"},
            {"emoji": "❌", "label": "Second route", "payload": "Take the second route"},
        ],
    }
    return [
        ToolCall("tool_search", {"queries": ["Matrix reaction menu choices"]}),
        ToolCall("tool_call", {"calls": [{"name": "present_menu", "arguments": arguments}]}),
        Text("Menu ready"), Text("Selected the first route"),
    ]


@pytest.fixture
def model_responder(gateway_script: list[Response]) -> list[Response]:
    return gateway_script


@pytest.fixture
def gateway_home_setup() -> Callable[[Path], None]:
    def use_yaml_allowlist(home: Path) -> None:
        path = home / ".env"
        path.write_text(
            "".join(
                line for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
                if not line.startswith("MATRIX_ALLOWED_USERS=")
            ),
            encoding="utf-8",
        )

    return use_yaml_allowlist


@pytest.fixture
def gateway_config():
    return (
        "platform_toolsets:\n  matrix: [reaction_menu]\n"
        "platforms:\n  matrix:\n    enabled: true\n"
        "    allowed_users: '@alice:matrix.test,@bob:matrix.test'\n"
        "    require_mention: false\n    thread_require_mention: false\n"
        "    reactions: false\n    auto_thread: false\n"
        "updates:\n  check: false\n"
    )


def test_requester_selects_once_in_original_thread(gateway: LiveGateway, live_room: LiveRoom):
    async def exchange():
        alice = live_room.observer.client(live_room.homeserver)
        bot = live_room.bot.client(live_room.homeserver)
        bob_account = await _register(live_room.homeserver, "bob")
        bob = bob_account.client(live_room.homeserver)
        room = live_room.room_id
        events = []

        async def send(client, content, event_type="m.room.message"):
            result = await client.room_send(room, event_type, content)
            assert isinstance(result, RoomSendResponse), result
            return result.event_id

        async def receive(predicate, description):
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if predicate():
                    return
                try:
                    response = await asyncio.wait_for(alice.sync(timeout=250), deadline - time.monotonic())
                except asyncio.TimeoutError:
                    break
                joined = response.rooms.join.get(room)
                if joined:
                    events.extend(event.source for event in joined.timeline.events)
            pytest.fail(
                f"No {description} within 15 seconds:\n"
                + json.dumps({"events": events, "model_requests": gateway.model.requests}, default=str)
                + "\n"
                + gateway.container.get_wrapped_container().logs().decode(errors="replace")[-6000:]
            )

        def messages(body):
            return [event for event in events if event.get("sender") == live_room.bot.user_id
                    and event.get("content", {}).get("body") == body]

        try:
            await alice.sync(timeout=0)
            await alice.room_invite(room, bob_account.user_id)
            await bob.join(room)
            root = await send(bot, {"msgtype": "m.text", "body": "Menu thread root"})
            relation = {"rel_type": "m.thread", "event_id": root, "is_falling_back": True,
                        "m.in_reply_to": {"event_id": root}}
            await send(alice, {
                "msgtype": "m.text", "body": "@hermes:matrix.test Offer a route [in:menu]",
                "m.mentions": {"user_ids": [live_room.bot.user_id]}, "m.relates_to": relation,
            })

            def menu_messages():
                return [event for event in events if event.get("sender") == live_room.bot.user_id
                        and event.get("content", {}).get("body", "").startswith("Choose the next route")]

            await receive(lambda: menu_messages() and messages("Menu ready"), "delivered menu and completed opening turn")
            menus = menu_messages()
            assert len(menus) == 1
            menu = menus[0]
            assert menu["content"]["m.relates_to"]["event_id"] == root
            assert menu["content"]["m.relates_to"]["rel_type"] == "m.thread"
            target = menu["event_id"]
            await receive(lambda: {"✅", "❌"} <= {
                event.get("content", {}).get("m.relates_to", {}).get("key") for event in events
                if event.get("type") == "m.reaction" and event.get("sender") == live_room.bot.user_id
                and event.get("content", {}).get("m.relates_to", {}).get("event_id") == target
            }, "seeded menu reactions")

            reaction = {"m.relates_to": {"rel_type": "m.annotation", "event_id": target, "key": "✅"}}
            await send(bob, reaction, "m.reaction")
            await receive(lambda: messages("Only the user who requested this action can use these controls."), "wrong-actor refusal")
            assert len(gateway.model.main_requests()) == 3
            await send(alice, reaction, "m.reaction")
            await send(alice, {"m.relates_to": {
                "rel_type": "m.annotation", "event_id": target, "key": "❌",
            }}, "m.reaction")
            await receive(lambda: messages("Selected the first route"), "one choice follow-up")
            replies = messages("Selected the first route")
            assert len(replies) == 1
            assert replies[0]["content"]["m.relates_to"]["event_id"] == root
            assert replies[0]["content"]["m.relates_to"]["rel_type"] == "m.thread"

            await send(alice, {"msgtype": "m.text", "body": "/status", "m.relates_to": relation})
            await receive(lambda: any(event.get("sender") == live_room.bot.user_id
                                      and "Status" in event.get("content", {}).get("body", "") for event in events), "post-choice control barrier")
            requests = gateway.model.main_requests()
            assert len(requests) == 4
            search = [message for message in requests[1]["messages"] if message["role"] == "tool"]
            assert "present_menu" in json.loads(search[-1]["content"])["tools"]
            menu_result = [message for message in requests[2]["messages"] if message["role"] == "tool"]
            assert json.loads(menu_result[-1]["content"])["status"] == "menu_presented"
            previous, followup = requests[2]["messages"], requests[3]["messages"]
            assert followup[:len(previous)] == previous
            assert followup[-1]["role"] == "user"
            assert "[menu-choice]" in followup[-1]["content"]
            choice_text = followup[-1]["content"].split("[menu-choice]\n", 1)[1]
            assert json.loads(choice_text) == {
                "prompt": "Choose the next route", "context_id": "live-route", "emoji": "✅",
                "label": "First route", "payload": "Take the first route",
            }
            roles = [message["role"] for message in followup if message["role"] != "system"]
            assert all(left != right for left, right in zip(roles, roles[1:]))
        finally:
            await asyncio.gather(alice.close(), bob.close(), bot.close())

    asyncio.run(exchange())


def _choice(request: dict) -> dict:
    latest = request["messages"][-1]
    assert latest["role"] == "user"
    return json.loads(latest["content"].split("[menu-choice]\n", 1)[1])


def test_requester_selects_in_encrypted_room(
    gateway: LiveGateway, live_room: LiveRoom, linux_nio_observer: LinuxNioObserver,
):
    code = (
        "import asyncio, json; from menu_client import select_menu; "
        f"print(json.dumps(asyncio.run(select_menu({live_room.room_id!r}, {live_room.bot.user_id!r}))))"
    )
    result = linux_nio_observer.container.exec(["/opt/hermes/.venv/bin/python", "-c", code])
    output = result.output.decode(errors="replace")
    assert result.exit_code == 0, (
        f"Encrypted menu exchange failed:\n{output}\n"
        + gateway.container.get_wrapped_container().logs().decode(errors="replace")[-6000:]
    )
    assert json.loads(output.splitlines()[-1]) == {
        "menu_body": "Choose the next route\n\n✅ First route\n❌ Second route\n\nReact to choose within five minutes.",
        "followup_body": "Selected the first route",
        "undecrypted": [],
    }
    requests = gateway.model.main_requests()
    assert len(requests) == 4
    assert _choice(requests[3]) == {
        "prompt": "Choose the next route", "context_id": "live-route", "emoji": "✅",
        "label": "First route", "payload": "Take the first route",
    }
