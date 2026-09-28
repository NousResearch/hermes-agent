"""Separate nio clients verify server notification counts and scoped tool receipts."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from textwrap import dedent
from urllib.parse import quote

import pytest
from nio import (
    PushEventMatch, PushNotify, PushRuleKind, PushSetTweak,
    ReceiptEvent, Response, RoomInfo, RoomReadMarkersResponse, SetPushRuleResponse, SyncResponse,
)
from nio.responses import EmptyResponse

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.conftest import (
    LinuxNioObserver, LiveGateway, LiveRoom, MatrixFeedbackSettings,
)


@pytest.fixture
def matrix_feedback() -> MatrixFeedbackSettings:
    return MatrixFeedbackSettings("disabled", False)


@pytest.fixture
def gateway_auxiliary_config() -> str:
    return (
        "auxiliary:\n  background_review:\n    enabled: false\n"
        "  title_generation:\n    model_upgrade_enabled: false\n"
    )


@dataclass
class _AccountDataResponse(Response):
    content: dict

    @classmethod
    def from_dict(cls, content: dict) -> _AccountDataResponse:
        assert "errcode" not in content, content
        return cls(content)


def _client_code(room: LiveRoom) -> str:
    return (
        f"ROOM = {room.room_id!r}\nBOT = {room.bot.user_id!r}\nDEVICE = {room.bot.device_id!r}\n"
        + dedent("""
            import asyncio
            import json
            from nio import RoomMessageText, RoomSendResponse, SyncResponse
            from client import open_encrypted_client

            async def send(client, body, *, thread=None, notice=False):
                content = {"msgtype": "m.notice" if notice else "m.text", "body": body}
                if thread:
                    content["m.relates_to"] = {
                        "rel_type": "m.thread", "event_id": thread,
                        "is_falling_back": True, "m.in_reply_to": {"event_id": thread},
                    }
                sent = await client.room_send(ROOM, "m.room.message", content, ignore_unverified_devices=True)
                assert isinstance(sent, RoomSendResponse), sent
                return sent.event_id

            async def reply(client, expected):
                while True:
                    response = await client.sync(timeout=250)
                    assert isinstance(response, SyncResponse), response
                    joined = response.rooms.join.get(ROOM)
                    if joined:
                        for event in joined.timeline.events:
                            if isinstance(event, RoomMessageText) and event.sender == BOT and event.body == expected:
                                return event.event_id
        """)
    )


@pytest.mark.parametrize("encrypted", [False, True], ids=["plain", "encrypted"])
def test_model_observes_counts_and_explicitly_marks_only_the_selected_scope(
    gateway: LiveGateway, live_room: LiveRoom, linux_nio_observer: LinuxNioObserver,
    encrypted: bool, record_property: Callable[[str, object], None],
) -> None:
    code = f"ENCRYPTED = {encrypted!r}\n" + _client_code(live_room)
    started = time.monotonic()
    handshake = linux_nio_observer.run_python(
        code + dedent("""
            from nio import JoinedMembersResponse, KeysQueryResponse, KeysUploadResponse, RoomPutStateResponse

            async def prepare():
                client = open_encrypted_client()
                try:
                    await client.sync(timeout=0, full_state=True)
                    if ENCRYPTED:
                        assert isinstance(await client.keys_upload(), KeysUploadResponse)
                        changed = await client.room_put_state(ROOM, "m.room.encryption", {"algorithm": "m.megolm.v1.aes-sha2"})
                        assert isinstance(changed, RoomPutStateResponse), changed
                        while not client.rooms[ROOM].encrypted:
                            await client.sync(timeout=250)
                        assert isinstance(await client.joined_members(ROOM), JoinedMembersResponse)
                        assert isinstance(await client.keys_query(), KeysQueryResponse)
                        assert DEVICE in client.device_store[BOT]
                    await send(client, "Establish unread test session")
                    print(json.dumps(await reply(client, "Matrix live reply")))
                finally:
                    await client.close()

            asyncio.run(asyncio.wait_for(prepare(), timeout=20))
        """)
    )
    baseline_event = json.loads(handshake.strip().splitlines()[-1])

    async def exchange() -> None:
        bot = live_room.bot.client(live_room.homeserver)
        observer = live_room.observer.client(live_room.homeserver)
        receipts = []
        room_path = f"/_matrix/client/v3/user/{quote(live_room.bot.user_id, safe='')}/rooms/{quote(live_room.room_id, safe='')}/account_data"

        async def account_data(event_type: str) -> dict:
            response = await bot._send(_AccountDataResponse, "GET", f"{room_path}/{event_type}")
            return response.content

        async def observe() -> None:
            response = await observer.sync(timeout=0)
            assert isinstance(response, SyncResponse), response
            joined = response.rooms.join.get(live_room.room_id)
            if joined is not None:
                receipts.extend(receipt for event in joined.ephemeral if isinstance(event, ReceiptEvent)
                                for receipt in event.receipts if receipt.user_id == live_room.bot.user_id)

        async def counts(predicate: Callable[[RoomInfo], bool]) -> RoomInfo:
            async with asyncio.timeout(10):
                while True:
                    response = await bot.sync(timeout=250, full_state=True, sync_filter={
                        "room": {"timeline": {"unread_thread_notifications": True}},
                    })
                    assert isinstance(response, SyncResponse), response
                    joined = response.rooms.join.get(live_room.room_id)
                    if joined is not None and predicate(joined):
                        return joined

        def model_phase(label: str, calls: list[tuple[str, dict]]) -> list[dict]:
            request_start = len(gateway.model.main_requests())
            gateway.model.push(
                ToolCall("tool_search", {"queries": ["Matrix unread notification counts mark read"]}),
                *(ToolCall("tool_call", {"calls": [{"name": name, "arguments": arguments}]})
                  for name, arguments in calls),
                Text(label),
            )
            try:
                linux_nio_observer.run_python(code + f"EXPECTED = {label!r}\n" + dedent("""
                    async def ask():
                        client = open_encrypted_client()
                        try:
                            await client.sync(timeout=0, full_state=True)
                            await send(client, "Inspect and acknowledge the requested unread scope")
                            await reply(client, EXPECTED)
                        finally:
                            await client.close()
                    asyncio.run(asyncio.wait_for(ask(), timeout=20))
                """))
            except AssertionError as exc:
                requests = gateway.model.main_requests()[request_start:]
                model_trace = [
                    {
                        "tools": [tool["function"]["name"] for tool in request["tools"]],
                        "messages": [message for message in request["messages"] if message["role"] != "system"],
                    }
                    for request in requests
                ]
                logs = gateway.container.get_wrapped_container().logs().decode(errors="replace")[-6000:]
                pytest.fail(
                    f"Matrix model phase {label!r} failed:\n{exc}\n"
                    f"Model requests:\n{json.dumps(model_trace, indent=2)}\nGateway logs:\n{logs}"
                )
            requests = gateway.model.main_requests()[request_start:]
            assert len(requests) == len(calls) + 2
            messages = [message for message in requests[-1]["messages"]
                        if message["role"] == "tool"]
            search = json.loads(messages[-len(calls) - 1]["content"])
            assert {name for name, _arguments in calls}.issubset(search["tools"])
            return [json.loads(message["content"]) for message in messages[-len(calls):]]

        try:
            rule = await bot.set_pushrule(
                "global", PushRuleKind.override, "hermes-unread-test",
                actions=[PushNotify(), PushSetTweak("highlight", True)],
                conditions=[PushEventMatch("room_id", live_room.room_id),
                            PushEventMatch("sender", live_room.observer.user_id)],
            )
            assert isinstance(rule, SetPushRuleResponse), rule
            reset = await bot.room_read_markers(live_room.room_id, baseline_event, baseline_event)
            assert isinstance(reset, RoomReadMarkersResponse), reset
            marker = await bot._send(EmptyResponse, "PUT", f"{room_path}/m.marked_unread", json.dumps({"unread": True}))
            assert isinstance(marker, EmptyResponse), marker
            fully_read = await account_data("m.fully_read")
            await observe()
            receipts.clear()
            output = linux_nio_observer.run_python(code + dedent("""
                from nio import RoomGetEventResponse

                async def targets():
                    client = open_encrypted_client()
                    try:
                        await client.sync(timeout=0, full_state=True)
                        root = await send(client, "Unread thread root", notice=True)
                        thread = await send(client, "Unread thread notification", thread=root, notice=True)
                        main = await send(client, "Unread main notification", notice=True)
                        target = await client.room_get_event(ROOM, thread)
                        assert isinstance(target, RoomGetEventResponse), target
                        assert target.event.decrypted is ENCRYPTED
                        print(json.dumps([root, thread, main]))
                    finally:
                        await client.close()
                asyncio.run(asyncio.wait_for(targets(), timeout=15))
            """))
            root, thread, main = json.loads(output.strip().splitlines()[-1])
            before = await counts(lambda room: bool((room.unread_thread_notifications or {}).get(root)))
            assert before.unread_thread_notifications is not None
            thread_count = before.unread_thread_notifications[root].notification_count
            assert isinstance(thread_count, int) and thread_count > 0
            assert before.unread_notifications is not None
            main_count = before.unread_notifications.notification_count
            assert isinstance(main_count, int) and main_count > 0
            first = await asyncio.to_thread(model_phase, "Thread acknowledged", [
                ("matrix_unread", {"thread_id": "main"}),
                ("matrix_unread", {"thread_id": root}),
                ("matrix_mark_read", {"thread_id": root, "event_id": thread, "visibility": "private"}),
            ])
            assert all(result["account_user_id"] == live_room.bot.user_id for result in first)
            assert first[0]["notification_count"] > 0 and first[1]["notification_count"] > 0
            assert first[0]["count_basis"] == first[1]["count_basis"] == "bot_account_push_rules"
            assert first[2] == {
                "room_id": live_room.room_id, "account_user_id": live_room.bot.user_id,
                "event_id": thread, "thread_id": root, "visibility": "private",
                "receipt_sent": True, "marked_unread_reset": False,
                "fully_read_marker_changed": False, "counts": "await_sync", "errors": [],
            }
            after_thread = await counts(lambda room: root not in (room.unread_thread_notifications or {})
                                        or (room.unread_thread_notifications or {})[root].notification_count == 0)
            assert after_thread.unread_notifications is not None
            remaining = after_thread.unread_notifications.notification_count
            assert isinstance(remaining, int) and remaining > 0
            await observe()
            assert all(receipt.event_id != thread for receipt in receipts)
            assert await account_data("m.marked_unread") == {"unread": True}
            assert await account_data("m.fully_read") == fully_read

            second = await asyncio.to_thread(model_phase, "Main acknowledged", [
                ("matrix_mark_read", {"thread_id": "main", "event_id": main, "visibility": "public"}),
            ])
            assert second[0]["receipt_sent"] is True and second[0]["marked_unread_reset"] is False
            await observe()
            assert any(receipt.event_id == main and receipt.receipt_type == "m.read" and receipt.thread_id == "main"
                       for receipt in receipts)
            assert await account_data("m.marked_unread") == {"unread": True}
            assert await account_data("m.fully_read") == fully_read
            last = await asyncio.to_thread(model_phase, "Room acknowledged", [
                ("matrix_mark_read", {"thread_id": "room", "event_id": main, "visibility": "private"}),
            ])
            assert (last[0]["receipt_sent"], last[0]["marked_unread_reset"], last[0]["fully_read_marker_changed"]) == (True, True, False)
            assert await account_data("m.marked_unread") == {"unread": False}
            assert await account_data("m.fully_read") == fully_read
            await observe()
            assert all(receipt.receipt_type != "m.read.private" for receipt in receipts)
        finally:
            await bot.close()
            await observer.close()

    try:
        asyncio.run(asyncio.wait_for(exchange(), timeout=90))
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))
