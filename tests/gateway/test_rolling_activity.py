"""Behavior contracts for opt-in rolling gateway activity."""

import asyncio
import logging
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.rolling_activity import finish_turn_activity, start_hygiene_activity, terminal_header
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from tests.gateway.test_run_progress_topics import (
    CommentaryAgent,
    ManyProgressLinesAgent,
    QueuedCommentaryAgent,
    SmallLimitProgressAdapter,
    _run_with_agent,
)


class RollingCaptureAdapter(BasePlatformAdapter):
    MAX_MESSAGE_LENGTH = 120

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.DISCORD)
        self.sent = []
        self.edits = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append((content, reply_to, metadata))
        return SendResult(success=True, message_id="activity-1")

    async def edit_message(self, chat_id, message_id, content, metadata=None) -> SendResult:
        self.edits.append((message_id, content, metadata))
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class Utf16RollingAdapter(SmallLimitProgressAdapter):
    @property
    def message_len_fn(self):
        return lambda text: len(text.encode("utf-16-le")) // 2


class OrderedRollingAdapter(SmallLimitProgressAdapter):
    MAX_MESSAGE_LENGTH = 500

    def __init__(self, platform=Platform.DISCORD):
        super().__init__(platform=platform)
        self.timeline = []

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.timeline.append(("send", content))
        return await super().send(chat_id, content, reply_to, metadata)

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.timeline.append(("edit", content))
        return await super().edit_message(chat_id, message_id, content)


class RetryableTerminalAdapter(RollingCaptureAdapter):
    def __init__(self, failures=1):
        super().__init__()
        self.failures = failures

    async def edit_message(self, chat_id, message_id, content, metadata=None) -> SendResult:
        self.edits.append((message_id, content, metadata))
        if self.failures:
            self.failures -= 1
            return SendResult(success=False, error="temporary failure", retryable=True)
        return SendResult(success=True, message_id=message_id)


class NativeTaskCardCaptureAdapter(RollingCaptureAdapter):
    def __init__(self):
        super().__init__()
        self.stops = 0

    async def send_native_task_card_progress(self, **_kwargs) -> SendResult:
        return SendResult(success=True, message_id="native-card")

    async def stop_native_task_card_progress(self, *_args, **_kwargs) -> None:
        self.stops += 1


class SlowToolAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        import time

        self.tool_progress_callback("tool.started", "terminal", "long command", {})
        time.sleep(0.25)
        return {"final_response": "done", "messages": [], "api_calls": 1}


def _turn(adapter, *, initial_message_id=None):
    source = SessionSource(platform=Platform.DISCORD, chat_id="chat", chat_type="channel")
    ctx = TurnContext(
        source=source,
        progress_grouping="rolling",
        tool_progress_enabled=True,
        progress_queue=queue.Queue(),
        initial_progress_msg_id=initial_message_id,
        _progress_metadata={"non_conversational": True},
        _progress_reply_to="trigger",
        _run_still_current=lambda: True,
    )
    runner = SimpleNamespace(_delivery_adapter_for=lambda _source: adapter)
    return ctx, TurnRunner(runner, ctx)


@pytest.mark.asyncio
async def test_rolling_sender_keeps_one_bounded_non_conversational_bubble():
    adapter = RollingCaptureAdapter()
    ctx, turn = _turn(adapter)
    task = asyncio.create_task(turn.send_progress_messages())
    ctx.progress_queue.put(("__activity_start__",))
    for index in range(5):
        ctx.progress_queue.put(f"activity-{index}-" + "x" * 40)
    ctx.activity_result = {"completed": True}

    await finish_turn_activity(ctx, task, logging.getLogger(__name__))

    assert len(adapter.sent) == 1
    assert adapter.sent[0][1] == "trigger"
    assert adapter.sent[0][2]["non_conversational"] is True
    assert adapter.edits
    assert {message_id for message_id, _, _ in adapter.edits} == {"activity-1"}
    assert all(len(text) <= adapter.MAX_MESSAGE_LENGTH for text, _, _ in adapter.sent)
    assert all(len(text) <= adapter.MAX_MESSAGE_LENGTH for _, text, _ in adapter.edits)
    final = adapter.edits[-1][1]
    assert final.startswith("✅ Completed")
    assert "earlier activities omitted" in final
    assert "activity-4" in final
    assert "activity-0" not in final


@pytest.mark.asyncio
async def test_rolling_sender_reuses_hygiene_message_and_metadata_compatible_edit():
    adapter = RollingCaptureAdapter()
    ctx, turn = _turn(adapter, initial_message_id="hygiene-1")
    task = asyncio.create_task(turn.send_progress_messages())
    ctx.progress_queue.put(("__activity_start__",))
    ctx.progress_queue.put("🔧 one tool")
    ctx.activity_result = {"completed": True}

    await finish_turn_activity(ctx, task, logging.getLogger(__name__))

    assert adapter.sent == []
    assert {message_id for message_id, _, _ in adapter.edits} == {"hygiene-1"}
    assert adapter.edits[-1][2]["non_conversational"] is True


@pytest.mark.asyncio
async def test_native_task_card_finish_marker_stops_without_cleanup_cancellation():
    adapter = NativeTaskCardCaptureAdapter()
    ctx, turn = _turn(adapter)
    ctx._native_slack_task_cards = True
    task = asyncio.create_task(turn.send_progress_messages())
    ctx.activity_result = {"completed": True}

    await asyncio.wait_for(
        finish_turn_activity(ctx, task, logging.getLogger(__name__)), timeout=2.0
    )

    assert task.done() and not task.cancelled()
    assert adapter.stops == 1


@pytest.mark.asyncio
async def test_retryable_terminal_edit_retains_finish_intent_and_message_identity():
    adapter = RetryableTerminalAdapter()
    ctx, turn = _turn(adapter, initial_message_id="activity-1")
    task = asyncio.create_task(turn.send_progress_messages())
    ctx.activity_result = {"completed": True}

    await asyncio.wait_for(
        finish_turn_activity(ctx, task, logging.getLogger(__name__)), timeout=2.0
    )

    assert task.done() and not task.cancelled()
    assert [message_id for message_id, _, _ in adapter.edits] == ["activity-1", "activity-1"]
    assert adapter.edits[-1][1] == "✅ Completed"


@pytest.mark.asyncio
async def test_persistent_retryable_terminal_edit_failure_is_bounded():
    adapter = RetryableTerminalAdapter(failures=10)
    ctx, turn = _turn(adapter, initial_message_id="activity-1")
    task = asyncio.create_task(turn.send_progress_messages())
    ctx.activity_result = {"completed": True}

    await asyncio.wait_for(
        finish_turn_activity(ctx, task, logging.getLogger(__name__)), timeout=2.0
    )

    assert task.done() and not task.cancelled()
    assert len(adapter.edits) == 3


@pytest.mark.asyncio
async def test_blocked_inbound_after_hygiene_closes_preflight_without_completed(monkeypatch):
    import gateway.run as gateway_run
    import gateway.run_turn as gateway_turn

    adapter = RollingCaptureAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._delivery_adapter_for = lambda _source: adapter
    runner._hmwa_open_session = AsyncMock(return_value=(False, False))
    runner._set_session_env = lambda _context: ()
    runner._pinned_session_context_prompt = lambda *_args, **_kwargs: "context"
    runner._hmwa_acquire_turn_lease = AsyncMock()
    runner._mark_durable_active_turn = AsyncMock()
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        load_transcript=AsyncMock(return_value=[{"role": "user"}] * 4),
    )

    async def _hygiene(*args):
        args[-1][0] = "hygiene-1"
        return args[4]

    runner._hmwa_run_session_hygiene = _hygiene
    runner._hmwa_first_contact_notes = AsyncMock()
    runner._voice_channel_sidecar_note = lambda *_args: None

    warning = "Context reference blocked"

    async def _blocked(**_kwargs):
        await adapter.send("chat", warning)
        return None

    runner._prepare_profile_scoped_inbound_message_text = _blocked
    runner._run_agent_progress_threading = lambda *_args: ({"non_conversational": True}, None, None)
    runner._reply_anchor_for_event = lambda event: event.message_id
    runner._clear_session_env = lambda _tokens: None
    monkeypatch.setattr(gateway_turn, "build_session_context", lambda *_args: {})
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    source = SessionSource(platform=Platform.DISCORD, chat_id="chat", chat_type="channel")
    event = MessageEvent(text="@file:blocked", source=source, message_id="trigger")
    entry = SimpleNamespace(session_id="session", session_key="agent:main:discord:chat")

    prepared, _tokens = await runner._hmwa_prepare_turn(
        event, source, entry, entry.session_key, entry.session_key, 1
    )

    assert prepared is None
    assert [content for content, _, _ in adapter.sent] == [warning]
    assert adapter.edits[-1][0] == "hygiene-1"
    assert adapter.edits[-1][1] == "⏹️ Stopped"
    assert all("Completed" not in content for _, content, _ in adapter.edits)


@pytest.mark.asyncio
async def test_active_gateway_path_keeps_one_bounded_editable_tail(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        ManyProgressLinesAgent,
        session_id="sess-progress-rolling-tail",
        config_data={
            "display": {
                "tool_progress": "all",
                "tool_progress_grouping": "rolling",
                "interim_assistant_messages": False,
            }
        },
        platform=Platform.DISCORD,
        chat_id="C123",
        thread_id="thread-1",
        adapter_cls=Utf16RollingAdapter,
    )

    assert result["final_response"] == "done"
    assert len(adapter.sent) == 1
    assert adapter.sent[0]["metadata"]["non_conversational"] is True
    assert {call["message_id"] for call in adapter.edits} == {"progress-1"}
    rendered = [adapter.sent[0]["content"], *(call["content"] for call in adapter.edits)]
    effective_limit = adapter.MAX_MESSAGE_LENGTH - 64
    assert all(adapter.message_len_fn(text) <= effective_limit for text in rendered)
    assert adapter.edits[-1]["content"].startswith("✅ Completed")
    assert "earlier activities omitted" in adapter.edits[-1]["content"]
    assert adapter.oversized_sends == []
    assert adapter.oversized_edits == []


@pytest.mark.asyncio
async def test_active_gateway_path_reuses_preflight_message(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        ManyProgressLinesAgent,
        session_id="sess-progress-rolling-preflight",
        initial_progress_msg_id="preflight-1",
        config_data={
            "display": {
                "tool_progress": "all",
                "tool_progress_grouping": "rolling",
                "interim_assistant_messages": False,
            }
        },
        platform=Platform.DISCORD,
        adapter_cls=Utf16RollingAdapter,
    )

    assert result["final_response"] == "done"
    assert adapter.sent == []
    assert {call["message_id"] for call in adapter.edits} == {"preflight-1"}


@pytest.mark.asyncio
async def test_active_gateway_path_routes_commentary_into_activity(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-progress-rolling-commentary",
        config_data={
            "display": {
                "tool_progress": "all",
                "tool_progress_grouping": "rolling",
                "interim_assistant_messages": True,
            },
            "streaming": {"enabled": False},
        },
        platform=Platform.DISCORD,
        adapter_cls=SmallLimitProgressAdapter,
    )

    assert result["final_response"] == "done"
    assert len(adapter.sent) == 1
    assert adapter.edits[-1]["content"].startswith("✅ Completed")
    assert "I'll inspect the repo first." in adapter.edits[-1]["content"]
    assert {call["message_id"] for call in adapter.edits} == {"progress-1"}


@pytest.mark.asyncio
async def test_active_gateway_path_routes_heartbeat_into_activity(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.05")
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        SlowToolAgent,
        session_id="sess-progress-rolling-heartbeat",
        config_data={
            "display": {
                "tool_progress": "all",
                "tool_progress_grouping": "rolling",
                "interim_assistant_messages": False,
                "long_running_notifications": True,
            }
        },
        platform=Platform.DISCORD,
        adapter_cls=SmallLimitProgressAdapter,
    )

    assert result["final_response"] == "done"
    assert len(adapter.sent) == 1
    assert "Working — 0 min" in adapter.edits[-1]["content"]
    assert {call["message_id"] for call in adapter.edits} == {"progress-1"}


@pytest.mark.asyncio
async def test_rolling_queued_followup_starts_after_parent_terminal_flush(monkeypatch, tmp_path):
    QueuedCommentaryAgent.calls = 0
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        QueuedCommentaryAgent,
        session_id="sess-progress-rolling-queued-order",
        pending_text="follow up",
        config_data={
            "display": {
                "tool_progress": "all",
                "tool_progress_grouping": "rolling",
                "interim_assistant_messages": True,
            }
        },
        platform=Platform.DISCORD,
        chat_id="C123",
        thread_id="thread-1",
        adapter_cls=OrderedRollingAdapter,
    )

    assert result["final_response"] == "final response 2"
    parent_terminal = next(
        index for index, (_, content) in enumerate(adapter.timeline)
        if content.startswith("✅ Completed")
    )
    child_start = next(
        index for index, (_, content) in enumerate(adapter.timeline[parent_terminal + 1 :], parent_terminal + 1)
        if content.startswith("⏳ Working…")
    )
    assert parent_terminal < child_start


@pytest.mark.parametrize(
    ("result", "cancelled", "expected"),
    [
        ({"completed": True}, False, "✅ Completed"),
        ({"interrupted": True}, False, "⏹️ Stopped"),
        ({"completed": True}, True, "⏹️ Stopped"),
        ({"failed": True}, False, "⚠️ Needs attention"),
        ({"partial": True}, False, "⚠️ Needs attention"),
        ({"error": "timeout"}, False, "⚠️ Needs attention"),
        (None, False, "⚠️ Needs attention"),
    ],
)
def test_terminal_header_contract(result, cancelled, expected):
    assert terminal_header(result, cancelled=cancelled) == expected


@pytest.mark.asyncio
async def test_hygiene_activity_uses_prospective_thread_and_legacy_mode_precedence(monkeypatch):
    adapter = RollingCaptureAdapter()
    runner = object.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda _source: adapter
    runner._get_proxy_url = lambda: None
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chat",
        chat_type="channel",
        delivered_via_upstream_relay=True,
        prospective_thread_id="future-thread",
    )
    event = MessageEvent(text="hello", source=source, message_id="trigger")
    config = {"display": {"tool_progress": "all", "tool_progress_grouping": "rolling"}}

    message_id = await start_hygiene_activity(runner, event, source, config)

    assert message_id == "activity-1"
    assert adapter.sent[0][1] == "trigger"
    assert adapter.sent[0][2]["reply_to_message_id"] == "trigger"
    assert adapter.sent[0][2]["non_conversational"] is True
    assert adapter.sent[0][2]["_interim_send"] is True

    adapter.sent.clear()
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")
    inherited = {"display": {"tool_progress_grouping": "rolling"}}
    assert await start_hygiene_activity(runner, event, source, inherited) is None
    assert adapter.sent == []
