"""Gateway intentional-silence token behavior."""

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


def _event(*, internal: bool = False, reply_expected=None):
    return MessageEvent(
        text="side chatter",
        source=_source(),
        message_id="msg-42",
        internal=internal,
        reply_expected=reply_expected,
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
@pytest.mark.parametrize("reply_expected", [None, True], ids=["adapter-unknown", "addressed"])
async def test_human_turn_gets_a_visible_fallback_for_a_silence_marker(monkeypatch, tmp_path, reply_expected):
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
        _event(reply_expected=reply_expected), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response and not is_intentional_silence_response(response)


@pytest.mark.asyncio
async def test_unaddressed_human_turn_suppresses_silence_without_warning(monkeypatch, tmp_path, caplog):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "[SILENT]",
        "messages": [{"role": "user", "content": "side chatter"},
                     {"role": "assistant", "content": "[SILENT]"}],
        "tools": [], "history_offset": 0, "last_prompt_tokens": 0,
        "api_calls": 1, "failed": False,
    })
    with caplog.at_level("DEBUG"):
        response = await runner._handle_message_with_agent(
            _event(reply_expected=False), _source(), "agent:main:telegram:group:-1001:12345", 1
        )
    assert response == ""
    assert not any(record.levelname == "WARNING" and "silence marker" in record.message
                   for record in caplog.records)
    assert any(record.levelname == "DEBUG" and "unaddressed" in record.message
               for record in caplog.records)


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
        reply_expected=None,
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

    delivered = runner._deliver_queued_first_response.await_args.args[0]
    assert delivered and not is_intentional_silence_response(delivered)


@pytest.mark.asyncio
async def test_queued_terminal_turn_owns_the_silence_verdict(monkeypatch, tmp_path):
    """The chain's LAST turn decides whether a bare marker may vanish, not the opener."""
    runner = _runner(monkeypatch, tmp_path)
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(return_value={"final_response": "NO_REPLY", "messages": []})
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value="agent:main:telegram:group:-1001:12345")
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="follow-up")
    runner._delivery_adapter_for = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    turn_ctx = SimpleNamespace(
        source=_source(), session_id="sid", session_key="agent:main:telegram:group:-1001:12345",
        run_generation=1, _interrupt_depth=0, history=[], _status_thread_metadata=None,
        context_prompt=None, result_holder=[None])
    pending_event = SimpleNamespace(source=_source(), message_id="43", channel_prompt=None,
                                    message_type=None, internal=True, metadata={}, reply_expected=True)

    merged = await gateway_run.GatewayRunner._run_agent_queued_followup(
        runner, turn_ctx, adapter=None, pending="hi again", pending_event=pending_event,
        response="resp", result={"interrupted": True, "messages": []}, stream_task=None)

    followup = runner._run_agent.await_args.kwargs
    assert followup["persist_user_display_kind"] == "internal_notification"
    assert followup["reply_expected"] is True
    assert followup["persist_user_display_metadata"]["reply_expected"] is True
    assert merged["queued_terminal_display_kind"] == "internal_notification"
    assert merged["queued_terminal_reply_expected"] is True

    def _result(terminal_kind, terminal_reply_expected=None):
        return {
            "final_response": "[SILENT]", "tools": [], "history_offset": 0, "last_prompt_tokens": 0,
            "api_calls": 1, "failed": False, "queued_terminal_inbound_id": "43",
            "queued_terminal_display_kind": terminal_kind,
            "queued_terminal_reply_expected": terminal_reply_expected,
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
    assert response and not is_intentional_silence_response(response)
    # Unaddressed opener, addressed terminal turn: visible fallback.
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_result(None, True))
    response = await runner._handle_message_with_agent(
        _event(reply_expected=False), _source(), "agent:main:telegram:group:-1001:12345", 1)
    assert response and not is_intentional_silence_response(response)


@pytest.mark.parametrize("opener, absorbed, merged", [
    (False, True, True), (False, None, None), (True, False, True), (False, False, False),
])
def test_one_turn_answering_several_messages_is_addressed_if_any_was(opener, absorbed, merged):
    """A merged pending message answers both texts, so an addressed one keeps the fallback."""
    from gateway.platforms.base import merge_pending_message_event

    pending = {"k": _event(reply_expected=opener)}
    merge_pending_message_event(pending, "k", _event(reply_expected=absorbed), merge_text=True)
    assert pending["k"].reply_expected is merged


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

    assert response.strip()


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
