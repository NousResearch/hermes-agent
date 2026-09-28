"""Discover the session action and verify native threads with an independent client."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable

import pytest

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.conftest import (
    LinuxNioObserver,
    LiveGateway,
    LiveRoom,
)


def _probe(
    observer: LinuxNioObserver, room: LiveRoom, encrypted: bool, expression: str
):
    output = observer.run_python(
        "import asyncio, json\n"
        "from thread_create_client import ThreadCreationProbe\n"
        f"probe = ThreadCreationProbe({room.room_id!r}, {room.bot.user_id!r}, {encrypted!r})\n"
        "async def run():\n"
        "    try:\n"
        f"        return await {expression}\n"
        "    finally:\n"
        "        await probe.close()\n"
        "print(json.dumps(asyncio.run(run())))\n"
    )
    return json.loads(output.strip().splitlines()[-1])


def _script(
    gateway: LiveGateway, arguments: dict, final: str = "Created in original"
) -> threading.Event:
    responses = deque([
        ToolCall("tool_search", {"queries": ["Matrix create thread"]}),
        ToolCall(
            "tool_call",
            {"calls": [{"name": "matrix_thread_create", "arguments": arguments}]},
        ),
        Text(final),
        Text("Thread final"),
    ])
    completed = threading.Event()

    def respond(record):
        response = responses.popleft()
        if isinstance(response, Text) and response.text == final:
            completed.set()
        return response

    gateway.model._responder = respond
    return completed


@pytest.mark.parametrize("live_room", ["group"], indirect=True)
@pytest.mark.parametrize(
    "existing,encrypted", [(False, False), (True, False), (False, True), (True, True)]
)
def test_model_creates_a_native_thread_and_later_replies_in_its_own_session(
    gateway: LiveGateway,
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
    record_property: Callable[[str, object], None],
    existing: bool,
    encrypted: bool,
) -> None:
    started = time.monotonic()
    try:
        prepared = _probe(linux_nio_observer, live_room, encrypted, "probe.prepare()")
        arguments = {"message": "Initial thread message [new:initial]"}
        arguments.update(
            {"root_event_id": prepared["existing"]}
            if existing
            else {"root_text": "New root [new:root]"}
        )
        _script(gateway, arguments)
        observed = _probe(
            linux_nio_observer,
            live_room,
            encrypted,
            f"probe.create_and_follow({prepared['original']!r}, {prepared['existing']!r} if {existing!r} else None)",
        )
        requests = gateway.model.main_requests()
        assert len(requests) == 5
        search = [
            message for message in requests[2]["messages"] if message["role"] == "tool"
        ]
        assert "matrix_thread_create" in json.loads(search[-1]["content"])["tools"]
        results = [
            message for message in requests[3]["messages"] if message["role"] == "tool"
        ]
        assert json.loads(results[-1]["content"]) == observed
        old_context = json.dumps(requests[3]["messages"])
        new_context = json.dumps(requests[4]["messages"])
        assert "[old:sentinel]" in old_context and "[old:create]" in old_context
        assert (
            "[new:root]" in new_context
            and "[new:initial]" in new_context
            and "[new:question]" in new_context
        )
        assert (
            "[old:sentinel]" not in new_context
            and "[old:question]" not in new_context
            and "[old:create]" not in new_context
        )
        systems = [
            [message for message in request["messages"] if message["role"] == "system"]
            for request in requests[:4]
        ]
        assert all(system == systems[0] for system in systems)
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))


@pytest.mark.parametrize("live_room", ["group"], indirect=True)
def test_homeserver_denies_the_existing_root_reply_without_claiming_delivery(
    gateway: LiveGateway,
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
    record_property: Callable[[str, object], None],
) -> None:
    started = time.monotonic()
    try:
        prepared = _probe(linux_nio_observer, live_room, False, "probe.prepare()")
        completed = _script(
            gateway,
            {"root_event_id": prepared["existing"], "message": "Denied initial reply"},
            "Permission denied",
        )
        _probe(
            linux_nio_observer,
            live_room,
            False,
            f"probe.request_denied_creation({prepared['original']!r})",
        )
        assert completed.wait(timeout=15), (
            "No model-visible homeserver permission error"
        )
        messages = gateway.model.main_requests()[-1]["messages"]
        result = json.loads(
            [message for message in messages if message["role"] == "tool"][-1][
                "content"
            ]
        )
        assert result["success"] is False
        assert "MForbidden" in result["error"]
        assert {key: value for key, value in result.items() if key != "error"} == {
            "success": False,
            "room_id": live_room.room_id,
            "root_event_id": prepared["existing"],
        }
        _probe(
            linux_nio_observer,
            live_room,
            False,
            f"probe.assert_wire_type({prepared['existing']!r})",
        )
        _probe(
            linux_nio_observer,
            live_room,
            False,
            f"probe.assert_no_thread_replies({prepared['existing']!r})",
        )
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))


@pytest.mark.parametrize("live_room", ["group"], indirect=True)
def test_separate_client_encryption_transition_refuses_plaintext_after_admission(
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
    record_property: Callable[[str, object], None],
) -> None:
    started = time.monotonic()
    try:
        assert (
            _probe(
                linux_nio_observer,
                live_room,
                False,
                f"asyncio.wait_for(probe.assert_admission_encryption_transition_refused({live_room.bot.access_token!r}), timeout=15)",
            )
            == "missing encryption keys"
        )
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))
