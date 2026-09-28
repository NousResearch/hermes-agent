"""An independent Matrix client selects a menu through the real Linux gateway."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from nio import RoomSendResponse

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.conftest import LiveGateway, LiveRoom, _register


@pytest.fixture
def gateway_script():
    return [ToolCall("present_menu", {
        "prompt": "Choose the next route", "context_id": "live-route",
        "options": [
            {"emoji": "✅", "label": "First route", "payload": "Take the first route"},
            {"emoji": "❌", "label": "Second route", "payload": "Take the second route"},
        ],
    }), Text("Menu ready"), Text("Selected the first route")]


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
            pytest.fail(f"No {description} within 15 seconds:\n" + gateway.container.get_wrapped_container().logs().decode(errors="replace")[-6000:])

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
            await send(alice, {"msgtype": "m.text", "body": "Offer a route [in:menu]", "m.relates_to": relation})

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
            assert len(gateway.model.main_requests()) == 2
            await send(alice, reaction, "m.reaction")
            await send(alice, reaction, "m.reaction")
            await receive(lambda: messages("Selected the first route"), "one choice follow-up")
            replies = messages("Selected the first route")
            assert len(replies) == 1
            assert replies[0]["content"]["m.relates_to"]["event_id"] == root
            assert replies[0]["content"]["m.relates_to"]["rel_type"] == "m.thread"

            await send(alice, {"msgtype": "m.text", "body": "/status", "m.relates_to": relation})
            await receive(lambda: any(event.get("sender") == live_room.bot.user_id
                                      and "Status" in event.get("content", {}).get("body", "") for event in events), "post-choice control barrier")
            requests = gateway.model.main_requests()
            assert len(requests) == 3
            previous, followup = requests[1]["messages"], requests[2]["messages"]
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
