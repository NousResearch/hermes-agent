"""Matrix edit follow-ups retain the original route and wait for a turn boundary."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import unquote

import pytest

from gateway.config import GatewayConfig, PlatformConfig
from gateway.run import GatewayRunner
from gateway.run_busy import GatewayBusySessionMixin
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.matrix.adapter import MatrixAdapter


ROOM = "!room:example.org"
ALICE = "@alice:example.org"


def original_event(**changes):
    return {
        "event_id": "$original", "room_id": ROOM, "sender": ALICE,
        "type": "m.room.message",
        "content": {
            "msgtype": "m.text", "body": "original question",
            "m.relates_to": {"rel_type": "m.thread", "event_id": "$original-thread"},
        },
        **changes,
    }


def edit_event(body="correction", event_id="$edit", **changes):
    return SimpleNamespace(
        room_id=ROOM, sender=ALICE, event_id=event_id, timestamp=0,
        content={
            "msgtype": "m.text", "body": f"* {body}",
            "m.relates_to": {"rel_type": "m.replace", "event_id": "$original"},
            "m.new_content": {
                "msgtype": "m.text", "body": body,
                "m.relates_to": {"rel_type": "m.thread", "event_id": "$forged-thread"},
            },
        },
        **changes,
    )


def adapter_for(monkeypatch, policy=None, raw=None):
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")
    monkeypatch.setenv("MATRIX_PROCESS_EDITS", "true")
    extra = {"homeserver": "https://matrix.example.org", "user_id": "@hermes:example.org"}
    if policy is not None:
        extra["process_edits"] = policy
    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="token", extra=extra))
    adapter._text_batch_delay_seconds = 0
    events = {"$original": raw or original_event()}
    async def request(method, path):
        return events.get(unquote(path.rsplit("/", 1)[-1]), {})
    adapter._client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock(side_effect=request)), events=events)
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._resolve_room_identity = AsyncMock(return_value=SimpleNamespace(
        display_name="Project", room_topic=None, server_name="example.org", members_digest="members",
    ))
    monkeypatch.setattr(adapter, "_background_read_receipt", lambda *args: None)
    adapter.set_authorization_check(lambda *args, **kwargs: True)
    adapter._threads.mark("$original-thread")
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "changes", "allowed"),
    [
        (None, {}, False),
        ({ROOM: False}, {}, False),
        ({"!other:example.org": True}, {}, False),
        ({ROOM: True}, {}, True),
        ({ROOM: True}, {"sender": "@mallory:example.org"}, False),
        ({ROOM: True}, {"room_id": "!other:example.org"}, False),
        ({ROOM: True}, {"event_id": "$other"}, False),
        ({ROOM: True}, {"unsigned": {"redacted_because": {"event_id": "$redaction"}}}, False),
    ],
)
async def test_edit_followup_uses_room_opt_in_and_original_author_and_route(monkeypatch, policy, changes, allowed):
    adapter = adapter_for(monkeypatch, policy, original_event(**changes))
    adapter.handle_message = AsyncMock()

    await adapter._on_room_message(edit_event())

    forwarded = [call.args[0] for call in adapter.handle_message.await_args_list]
    observed = [
        (event.text, event.source.chat_id, event.source.user_id, event.source.thread_id,
         event.internal, event.allow_gateway_control, event.metadata)
        for event in forwarded
    ]
    assert observed == ([
        ("correction", ROOM, ALICE, "$original-thread", False, False,
         {"edited_message": True, "edited_message_original_id": "$original"}),
    ] if allowed else [])


class BusyRunner(GatewayBusySessionMixin):
    _BUSY_QUEUE_MAX_PENDING = 20

    def __init__(self, adapter, mode):
        self.adapter = adapter
        self.mode = mode
        self.state = SimpleNamespace(conversation=SimpleNamespace(queued_events=[]), turn=SimpleNamespace(agent=None))
        self._draining = False
        self._route_plaintext_approval_while_busy = AsyncMock(return_value=False)
        self.steer = AsyncMock(side_effect=AssertionError("edit entered the active model turn"))
        self._resolve_busy_steer_or_redirect = self.steer

    def _defer_for_startup_restore(self, event):
        return False

    def _delivery_adapter_for(self, source):
        return self.adapter

    def _session_state(self, key):
        return self.state

    def _peek_session_state(self, key):
        return self.state

    def _is_user_authorized_for_source(self, source):
        return True

    def _admit_bot_message_for_source(self, source):
        return True

    def _effective_busy_input_mode(self, source):
        return self.mode

    def _effective_busy_text_mode(self, source):
        return "interrupt"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["interrupt", "steer", "queue"])
@pytest.mark.parametrize("position", ["slot", "overflow"])
@pytest.mark.parametrize("policy_change", ["unchanged", "opt-out", "room-denied", "mention-required", "redacted", "edit-redacted", "sender-denied"])
async def test_pending_edits_coalesce_without_entering_the_active_turn(monkeypatch, mode, position, policy_change):
    adapter = adapter_for(monkeypatch, {ROOM: True})
    runner = BusyRunner(adapter, mode)
    adapter.set_message_handler(AsyncMock())
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    adapter._event_session_key = lambda event: "session"
    adapter._active_sessions["session"] = asyncio.Event()
    adapter._session_tasks["session"] = asyncio.current_task()

    unrelated = MessageEvent("unrelated pending message")
    if position == "overflow":
        adapter._pending_messages["session"] = unrelated

    for body, event_id in [("first correction", "$edit1"), ("latest correction", "$edit2")]:
        incoming = edit_event(body, event_id)
        adapter._client.events[event_id] = {
            "room_id": ROOM, "sender": ALICE, "event_id": event_id,
            "type": "m.room.message", "content": incoming.content,
        }
        await adapter._on_room_message(incoming)

    pending = [adapter._pending_messages["session"], *runner.state.conversation.queued_events]
    if position == "overflow":
        assert pending.pop(0) is unrelated
    assert [(event.text, event.message_id, event.source.user_id, event.source.thread_id) for event in pending] == [
        ("latest correction", "$edit2", ALICE, "$original-thread"),
    ]
    adapter._message_handler.assert_not_awaited()
    runner.steer.assert_not_awaited()
    queued = replace(pending[0])
    cold_runner = object.__new__(GatewayRunner)
    cold_runner.config = GatewayConfig()
    monkeypatch.setattr(cold_runner, "_is_user_authorized_for_source", lambda source: policy_change != "sender-denied")
    monkeypatch.setattr(cold_runner, "_intake_adapter_for", lambda source: adapter)
    monkeypatch.setattr(cold_runner, "_peek_session_state", lambda key: None)
    monkeypatch.setattr(type(adapter), "fetch_inbound_context", AsyncMock(return_value=None))
    monkeypatch.setattr(type(adapter), "fetch_mention_context", AsyncMock(return_value=None))
    monkeypatch.setattr(type(adapter), "take_turn_channel_context", lambda *args: None)
    if policy_change == "opt-out":
        adapter._process_edits = frozenset()
    if policy_change == "room-denied":
        adapter._is_allowed_matrix_room_event.return_value = False
    if policy_change == "mention-required":
        adapter._thread_require_mention = True
    if policy_change == "redacted":
        adapter._event_context_cache.redact(ROOM, "$original")
    if policy_change == "edit-redacted":
        adapter._client.events["$edit2"]["unsigned"] = {"redacted_because": {"event_id": "$redaction"}}
        adapter._event_context_cache._entries.clear()

    prepared = await cold_runner._prepare_profile_scoped_inbound_message_text(
        event=queued, source=queued.source, history=[{"role": "user", "content": "previous turn"}],
        session_key="session",
    )
    if prepared is not None:
        assert "latest correction" in prepared and "Correction to earlier message $original" in prepared
    assert (prepared is not None, queued._queue_at_turn_boundary, queued._pending_coalesce_key) == (
        policy_change == "unchanged", True, ("matrix-edit", ROOM, ALICE, "$original"),
    )
    adapter._active_sessions.clear()
    adapter._session_tasks.clear()


def cold_runner_for(monkeypatch, adapter):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    monkeypatch.setattr(runner, "_is_user_authorized_for_source", lambda source: True)
    monkeypatch.setattr(runner, "_intake_adapter_for", lambda source: adapter)
    monkeypatch.setattr(runner, "_peek_session_state", lambda key: None)
    monkeypatch.setattr(type(adapter), "fetch_inbound_context", AsyncMock(return_value=None))
    monkeypatch.setattr(type(adapter), "fetch_mention_context", AsyncMock(return_value=None))
    monkeypatch.setattr(type(adapter), "take_turn_channel_context", lambda *args: None)
    return runner


class QueueTextRunner(BusyRunner):
    def _effective_busy_text_mode(self, source):
        return "queue"


@pytest.mark.asyncio
@pytest.mark.parametrize("text_first", [False, True], ids=["text-after-correction", "text-before-correction"])
async def test_queue_mode_text_takes_its_own_turn_beside_a_pending_correction(monkeypatch, text_first):
    adapter = adapter_for(monkeypatch, {ROOM: True})
    adapter._busy_text_mode = "queue"
    runner = QueueTextRunner(adapter, "queue")
    adapter.set_message_handler(AsyncMock())
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    adapter._event_session_key = lambda event: "session"
    adapter._active_sessions["session"] = asyncio.Event()
    adapter._session_tasks["session"] = asyncio.current_task()

    async def correct():
        incoming = edit_event("latest correction", "$edit2")
        adapter._client.events["$edit2"] = {
            "room_id": ROOM, "sender": ALICE, "event_id": "$edit2",
            "type": "m.room.message", "content": incoming.content,
        }
        await adapter._on_room_message(incoming)

    async def follow_up():
        text = await adapter._build_inbound_event(
            ROOM, ALICE, "$new", "and one more thing", {"msgtype": "m.text", "body": "and one more thing"}, {},
        )
        await adapter.handle_message(text)

    for step in ([follow_up, correct] if text_first else [correct, follow_up]):
        await step()

    cold_runner = cold_runner_for(monkeypatch, adapter)
    turns = []
    while True:
        await adapter._flush_text_debounce_now("session")
        event = runner._promote_queued_event("session", adapter, adapter._pending_messages.pop("session", None))
        if event is None:
            break
        prepared = await cold_runner._prepare_profile_scoped_inbound_message_text(
            event=event, source=event.source, history=[{"role": "user", "content": "previous"}],
            session_key="session",
        )
        turns.append((event.message_id, prepared))
    adapter._active_sessions.clear()
    adapter._session_tasks.clear()

    correction = ("$edit2", "[Correction to earlier message $original]\n\n[Alice] latest correction")
    ordinary = ("$new", "and one more thing")
    assert turns == ([ordinary, correction] if text_first else [correction, ordinary])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "original_relation"),
    [
        ("!help", {"rel_type": "m.thread", "event_id": "$original-thread"}),
        ("> quoted line\n\nmy answer", {"m.in_reply_to": {"event_id": "$parent"}}),
    ],
    ids=["bang-command", "quote-on-reply"],
)
async def test_correction_reaches_the_turn_as_typed(monkeypatch, body, original_relation):
    raw = original_event()
    raw["content"]["m.relates_to"] = original_relation
    adapter = adapter_for(monkeypatch, {ROOM: True}, raw)
    adapter._client.events["$parent"] = {
        "event_id": "$parent", "room_id": ROOM, "sender": ALICE, "type": "m.room.message",
        "content": {"msgtype": "m.text", "body": "parent"},
    }
    adapter.handle_message = AsyncMock()
    incoming = edit_event(body, "$edit")
    adapter._client.events["$edit"] = {
        "room_id": ROOM, "sender": ALICE, "event_id": "$edit",
        "type": "m.room.message", "content": incoming.content,
    }

    await adapter._on_room_message(incoming)
    assert adapter.handle_message.await_args is not None
    event = adapter.handle_message.await_args.args[0]

    assert (event.text, event.message_type, event.get_command(), await adapter.validate_inbound_event(event)) == (
        body, MessageType.TEXT, None, True,
    )
