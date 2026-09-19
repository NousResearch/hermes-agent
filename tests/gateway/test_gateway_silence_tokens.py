"""Gateway intentional-silence token behavior and agent:start/agent:end pairing."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource
from gateway.response_filters import (
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
)


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="12345",
    )


def _event(*, internal: bool = False):
    return MessageEvent(
        text="side chatter",
        source=_source(),
        message_id="msg-42",
        internal=internal,
    )


def _runner(monkeypatch, tmp_path):
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._clear_session_env = MagicMock()
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-1001:12345",
        session_id="sess-silent",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


def test_exact_silence_tokens_are_intentional_silence():
    for token in ("[SILENT]", " SILENT ", "NO_REPLY", "no reply"):
        assert is_intentional_silence_response(token)


def test_blank_and_prose_mentions_are_not_silence():
    assert not is_intentional_silence_response("")
    assert not is_intentional_silence_response("Use NO_REPLY when no answer is needed.")
    assert not is_intentional_silence_response("The reply was [SILENT], intentionally.")


def test_failed_agent_result_never_counts_as_intentional_silence():
    assert is_intentional_silence_agent_result({"failed": False}, "NO_REPLY")
    assert not is_intentional_silence_agent_result({"failed": True}, "NO_REPLY")


@pytest.mark.asyncio
async def test_human_turn_gets_a_visible_fallback_for_a_silence_marker(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "[SILENT]",
        "messages": [
            {"role": "user", "content": "side chatter"},
            {"role": "assistant", "content": "[SILENT]"},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert "silence marker" in response
    assert "Try again or rephrase" in response


@pytest.mark.asyncio
async def test_internal_silence_token_suppresses_delivery_but_preserves_transcript(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "[SILENT]",
        "messages": [
            {"role": "user", "content": "side chatter"},
            {"role": "assistant", "content": "[SILENT]"},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
    })

    response = await runner._handle_message_with_agent(
        _event(internal=True), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response == ""
    appended = [call.args[1] for call in runner.session_store.append_to_transcript.call_args_list]
    assert {"role": "assistant", "content": "[SILENT]"}.items() <= appended[-1].items()
    assert [msg["role"] for msg in appended if msg.get("role") in {"user", "assistant"}] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_scheduled_heartbeat_silence_suppresses_delivery(monkeypatch, tmp_path):
    """A poller-stamped heartbeat turn may end on a bare marker (#113031); the event stays
    non-internal so authorization and the emergency stop still apply to it."""
    runner = _runner(monkeypatch, tmp_path)
    entry = runner.session_store.get_or_create_session.return_value
    entry.suspended = False
    runner.session_store.lookup_by_session_key.return_value = entry
    runner._run_agent = AsyncMock(return_value={
        "final_response": "NO_REPLY",
        "messages": [], "tools": [], "history_offset": 0, "last_prompt_tokens": 0,
        "api_calls": 1, "failed": False,
    })
    event = _event()
    event._heartbeat_session_id = entry.session_id

    assert await runner._handle_message_with_agent(event, _source(), entry.session_key, 1) == ""
    assert not event.internal


@pytest.mark.asyncio
async def test_queued_human_turn_also_gets_the_visible_fallback():
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner._deliver_queued_first_response = AsyncMock()
    turn_ctx = SimpleNamespace(
        session_key="agent:main:telegram:group:-1001:12345",
        stream_consumer_holder=[None],
        mute_notification_reply=False,
        persist_user_display_kind=None,
        source=_source(),
        _status_thread_metadata=None,
        event_message_id=None,
        inbound_message_id="msg-42",
        run_generation=1,
    )
    result = {"final_response": "NO_REPLY", "failed": False}

    await runner._run_agent_deliver_first_response(
        turn_ctx, None, result, result, None,
    )

    assert "silence marker" in runner._deliver_queued_first_response.await_args.args[0]


@pytest.mark.asyncio
async def test_queued_terminal_turn_owns_the_silence_verdict(monkeypatch, tmp_path):
    """The chain's LAST turn decides whether a bare marker may vanish, not the opener."""
    runner = _runner(monkeypatch, tmp_path)
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(return_value={"final_response": "NO_REPLY", "messages": []})
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value="agent:main:telegram:group:-1001:12345")
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="follow-up")
    runner._adapter_for_source = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    turn_ctx = SimpleNamespace(
        source=_source(), session_id="sid", session_key="agent:main:telegram:group:-1001:12345",
        run_generation=1, _interrupt_depth=0, history=[], _status_thread_metadata=None,
        context_prompt=None, result_holder=[None])
    pending_event = SimpleNamespace(source=_source(), message_id="43", channel_prompt=None,
                                    message_type=None, internal=True, metadata={})

    merged = await gateway_run.GatewayRunner._run_agent_queued_followup(
        runner, turn_ctx, adapter=None, pending="hi again", pending_event=pending_event,
        response="resp", result={"interrupted": True, "messages": []}, stream_task=None)

    assert runner._run_agent.await_args.kwargs["persist_user_display_kind"] == "internal_notification"
    assert merged["queued_terminal_display_kind"] == "internal_notification"

    def _result(terminal_kind):
        return {
            "final_response": "[SILENT]", "tools": [], "history_offset": 0, "last_prompt_tokens": 0,
            "api_calls": 1, "failed": False, "queued_terminal_inbound_id": "43",
            "queued_terminal_display_kind": terminal_kind,
            "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "[SILENT]"}],
        }

    # Human opener, internal terminal turn: silent.
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_result("internal_notification"))
    assert await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1) == ""
    # Internal opener, human terminal turn: visible fallback.
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_result(None))
    response = await runner._handle_message_with_agent(
        _event(internal=True), _source(), "agent:main:telegram:group:-1001:12345", 1)
    assert "silence marker" in response


@pytest.mark.asyncio
async def test_empty_success_still_gets_empty_response_warning(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": ""},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert "no response was generated" in response


@pytest.mark.asyncio
async def test_prose_mentioning_silence_token_is_delivered(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    text = "Use [SILENT] when no answer is needed."
    runner._run_agent = AsyncMock(return_value={
        "final_response": text,
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": text},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response == text


@pytest.mark.asyncio
async def test_agent_end_hook_includes_model_and_provider(monkeypatch, tmp_path):
    """Gateway hooks receive the actual model/provider for post-turn routing."""
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "done",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
        "model": "gpt-5.6-terra",
        "provider": "openai-codex",
    })

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    end_context = next(
        call.args[1]
        for call in runner.hooks.emit.await_args_list
        if call.args[0] == "agent:end"
    )
    assert end_context["model"] == "gpt-5.6-terra"
    assert end_context["provider"] == "openai-codex"


# --- agent:start / agent:end pairing (regression for #113057) ---
# Every attempted agent:start must be closed by exactly one agent:end, including turns that never
# reach the normal end (raise, cancellation, superseded generation, heartbeat owner change).

_SESSION_KEY = "agent:main:telegram:group:-1001:12345"
_DONE = {"final_response": "done", "tools": [], "history_offset": 0, "last_prompt_tokens": 0,
         "api_calls": 1, "failed": False,
         "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "done"}]}


def _end_contexts(runner):
    return [call.args[1] for call in runner.hooks.emit.await_args_list if call.args[0] == "agent:end"]


def _cancel_on(event_type):
    """Handlers are awaited, so a cancellation can land inside an ``agent:*`` emit."""
    def _emit(emitted_type, _context):
        if emitted_type == event_type:
            raise asyncio.CancelledError()
    return _emit


def _assert_closed_pair(runner):
    starts = [call.args[1] for call in runner.hooks.emit.await_args_list if call.args[0] == "agent:start"]
    ends = _end_contexts(runner)
    assert len(starts) == 1
    assert len(ends) == 1
    assert ends[0]["session_id"] == starts[0]["session_id"]
    assert ends[0]["failed"] is True
    assert (ends[0]["response"], ends[0]["model"], ends[0]["provider"]) == ("", "", "")


@pytest.mark.asyncio
async def test_raising_turn_closes_its_agent_start(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(side_effect=RuntimeError("backend exploded"))
    await runner._handle_message_with_agent(_event(), _source(), _SESSION_KEY, 1)
    _assert_closed_pair(runner)


@pytest.mark.asyncio
async def test_cancelled_turn_closes_its_agent_start(monkeypatch, tmp_path):
    """Cancelling inside the abandoned emit must not skip the session-context restore."""
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(side_effect=asyncio.CancelledError())
    runner.hooks.emit = AsyncMock(side_effect=_cancel_on("agent:end"))

    with pytest.raises(asyncio.CancelledError):
        await runner._handle_message_with_agent(_event(), _source(), _SESSION_KEY, 1)
    _assert_closed_pair(runner)
    runner._clear_session_env.assert_called_once()


@pytest.mark.asyncio
async def test_abandoned_run_closes_its_agent_start(monkeypatch, tmp_path):
    """A discarded turn still announced a start, so it must still end (both abandon paths)."""
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_DONE)
    runner._is_session_run_current = MagicMock(return_value=False)
    runner._hmwa_discard_stale_result = MagicMock()

    await runner._handle_message_with_agent(_event(), _source(), _SESSION_KEY, 1)
    _assert_closed_pair(runner)
    runner._hmwa_discard_stale_result.assert_called_once()

    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_DONE)
    monkeypatch.setattr(
        "gateway.run_heartbeat_acceptance.heartbeat_owner_is_current", lambda *_a, **_kw: False
    )

    await runner._handle_message_with_agent(_event(), _source(), _SESSION_KEY, 1)
    _assert_closed_pair(runner)
    runner._run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_post_turn_side_effects_do_not_add_a_second_end(monkeypatch, tmp_path):
    """The watcher drain runs after agent:end: cancelling it must not add a second end."""
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_DONE)
    runner._drain_watch_notifications = AsyncMock(side_effect=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await runner._handle_message_with_agent(_event(), _source(), _SESSION_KEY, 1)
    ends = _end_contexts(runner)
    assert len(ends) == 1
    assert ends[0]["response"] == "done"
    assert "failed" not in ends[0]


@pytest.mark.asyncio
async def test_cancelling_an_emit_keeps_exactly_one_pair(monkeypatch, tmp_path):
    """Hook handlers are awaited, so a cancellation can land inside either emit: it must leave one
    terminal event for the start, never a duplicate one."""
    # Cancel inside the agent:start emit: the attempted start is still closed, once.
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_DONE)
    runner.hooks.emit = AsyncMock(side_effect=_cancel_on("agent:start"))

    with pytest.raises(asyncio.CancelledError):
        await runner._handle_message_with_agent(_event(), _source(), _SESSION_KEY, 1)
    _assert_closed_pair(runner)

    # Cancel inside the agent:end emit: no second, failed end for the turn that was ending.
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_DONE)
    runner._drain_watch_notifications = AsyncMock()
    runner.hooks.emit = AsyncMock(side_effect=_cancel_on("agent:end"))

    with pytest.raises(asyncio.CancelledError):
        await runner._handle_message_with_agent(_event(), _source(), _SESSION_KEY, 1)
    ends = _end_contexts(runner)
    assert len(ends) == 1
    assert "failed" not in ends[0]
    runner._drain_watch_notifications.assert_not_awaited()
