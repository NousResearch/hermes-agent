"""Gateway intentional-silence token behavior."""

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


@pytest.mark.asyncio
@pytest.mark.parametrize("reply,failed,interrupted", [
    ("[SILENT]", False, False), ("NO_REPLY", False, False),
    (" SILENT ", False, False), ("no reply", False, False),
    ("[静默]", False, False), ("静默", False, False),
    ("[沉默]", False, False), ("沉默", False, False),
    ("A useful update", False, False),
    ("Use [SILENT] when no answer is needed.", False, False),
    ("[SILENT] A useful update", False, False),
    ("[SILENT]", True, False), ("", True, False), ("", False, False),
    ("Operation interrupted.", False, True),
])
async def test_scheduled_heartbeat_final_delivery_keeps_human_guard(
    monkeypatch, tmp_path, reply, failed, interrupted,
):
    """Poller/admission/final-delivery contract with a fake model and transport (#112254)."""
    from evals.heartbeat_idle_wire import WireAdapter
    from gateway.config import PlatformConfig
    from hermes_cli.heartbeat import HeartbeatState, save_heartbeat

    runner = _runner(monkeypatch, tmp_path)
    source = _source()
    key = "agent:main:telegram:group:-1001:12345"
    entry = runner.session_store.get_or_create_session.return_value
    runner.session_store.peek_session_id.return_value = entry.session_id
    runner.session_store.lookup_by_session_key.return_value = entry
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "hello"}, {"role": "assistant", "content": "hello"},
    ]
    adapter = WireAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    adapter.wire = []
    runner.adapters = {Platform.TELEGRAM: adapter}
    seen = []

    async def model(**kwargs):
        seen.append(kwargs)
        return {"final_response": reply, "failed": failed, "interrupted": interrupted,
                "error": "test failure" if failed else None,
                "messages": [{"role": "user", "content": kwargs["message"]},
                             {"role": "assistant", "content": reply}], "api_calls": 1}

    runner._run_agent = model

    async def handler(event):
        if not seen:
            assert event._heartbeat_session_id == entry.session_id
            assert event.internal is False
        try:
            return await runner._handle_message_with_agent(event, source, key, 1)
        finally:
            # The enclosing gateway handler normally releases the turn lease.
            runner._release_turn_lease(key, 1)

    adapter.set_message_handler(handler)
    save_heartbeat(entry.session_id, HeartbeatState(prompt="check status", interval_seconds=60, created_at=1))
    try:
        await runner._heartbeat_poll_once({key: (source, entry.session_id)})
        while adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks))
        assert len(seen) == 1
        assert seen[0]["heartbeat_turn"] is True
        assert seen[0]["persist_user_display_kind"] is None
        appended = [call.args[1] for call in runner.session_store.append_to_transcript.call_args_list]
        assert all(not msg.get("display_kind") for msg in appended)
        if reply in {"[SILENT]", "NO_REPLY", " SILENT ", "no reply", "[静默]", "静默", "[沉默]", "沉默"} and not failed:
            assert adapter.wire == []
        else:
            assert adapter.wire
            if not failed and reply:
                assert adapter.wire == [reply]
            elif failed and not reply:
                assert "couldn't finish" in adapter.wire[-1]
            elif not reply:
                assert "no response was generated" in adapter.wire[-1]

        if not failed and reply:
            appended = [call.args[1] for call in runner.session_store.append_to_transcript.call_args_list]
            assert any(msg.get("role") == "assistant" and msg.get("content") == reply for msg in appended)
        # Copying the scheduled prompt or metadata cannot grant a human silence privileges.
        if reply in {"[SILENT]", "NO_REPLY", " SILENT ", "no reply", "[静默]", "静默", "[沉默]", "沉默"} and not failed:
            event = _event()
            event.text = seen[0]["message"]
            event.metadata["_heartbeat_session_id"] = entry.session_id
            await adapter.handle_message(event)
            while adapter._background_tasks:
                await asyncio.gather(*list(adapter._background_tasks))
            assert len(adapter.wire) == 1
            assert "silence marker" in adapter.wire[0]
    finally:
        await adapter.cancel_session_processing(key)
        if adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks), return_exceptions=True)
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("heartbeat_terminal", [True, False])
@pytest.mark.parametrize("status", [{}, {"failed": True}, {"interrupted": True}, {"partial": True}, {"completed": False}])
async def test_queued_heartbeat_silence_permission_belongs_to_terminal_event(monkeypatch, tmp_path, heartbeat_terminal, status):
    runner = _runner(monkeypatch, tmp_path)
    key = "agent:main:telegram:group:-1001:12345"
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._is_goal_continuation_event = lambda event: False
    runner._session_key_for_source = lambda source: key
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="follow-up")
    runner._adapter_for_source = lambda source: None
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner.session_store.lookup_by_session_key.return_value = runner.session_store.get_or_create_session.return_value
    opener, terminal = _event(), _event()
    (terminal if heartbeat_terminal else opener)._heartbeat_session_id = "sess-silent"
    ctx = SimpleNamespace(source=_source(), session_id="sess-silent", session_key=key,
                          run_generation=1, _interrupt_depth=0, history=[], _status_thread_metadata=None,
                          context_prompt=None, result_holder=[None])
    runner._run_agent = AsyncMock(return_value={"final_response": "[SILENT]", "messages": [], "failed": False, **status})
    merged = await runner._run_agent_queued_followup(
        ctx, adapter=None, pending="follow-up", pending_event=terminal, response="",
        result={"interrupted": True, "messages": []}, stream_task=None)
    runner._run_agent = AsyncMock(return_value=merged)
    response = await runner._handle_message_with_agent(opener, _source(), key, 1)
    assert runner._run_agent.await_count == 1
    assert merged["queued_terminal_heartbeat_turn"] is heartbeat_terminal
    assert merged["queued_terminal_display_kind"] is None
    if status.get("failed"):
        assert response == (
            "[SILENT]\n\nYour request was not processed. Send it again if you still want me to carry it out."
        )  # Preserve the base failed-result fallback (no API calls in this fixture).
    elif heartbeat_terminal and not status:
        assert response == ""
    else:
        assert "silence marker" in response


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_heartbeat", [False, True])
async def test_nested_queue_preserves_innermost_heartbeat_provenance(monkeypatch, tmp_path, terminal_heartbeat):
    runner = _runner(monkeypatch, tmp_path)
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._is_goal_continuation_event = lambda event: False
    runner._session_key_for_source = lambda source: "key"
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="follow-up")
    runner._adapter_for_source = lambda source: None
    runner._refresh_agent_cache_message_count = AsyncMock()
    pending_event = _event()
    if not terminal_heartbeat:
        pending_event._heartbeat_session_id = "sid"
    terminal_result = {
        "final_response": "[SILENT]", "messages": [], "completed": True,
        "queued_terminal_inbound_id": "innermost",
        "queued_terminal_display_kind": None,
        "queued_terminal_heartbeat_turn": terminal_heartbeat,
    }
    runner._run_agent = AsyncMock(return_value=terminal_result)
    ctx = SimpleNamespace(source=_source(), session_id="sid", session_key="key",
                          run_generation=1, _interrupt_depth=0, history=[],
                          _status_thread_metadata=None, context_prompt=None, result_holder=[None])
    merged = await runner._run_agent_queued_followup(
        ctx, adapter=None, pending="follow-up", pending_event=pending_event,
        response="", result={"interrupted": True, "messages": []}, stream_task=None)
    assert merged["queued_terminal_inbound_id"] == "innermost"
    assert merged["queued_terminal_heartbeat_turn"] is terminal_heartbeat
    assert merged["queued_terminal_display_kind"] is None
    assert runner._run_agent.await_args.kwargs["heartbeat_turn"] is not terminal_heartbeat


@pytest.mark.asyncio
@pytest.mark.parametrize("provenance", ["human", "internal", "heartbeat"])
@pytest.mark.parametrize("state", ["interrupted", "interrupted_before_start", "partial", "incomplete"])
@pytest.mark.parametrize("reply", ["[SILENT]", "", "Useful partial update"])
async def test_noncompleted_final_delivery_characterization(monkeypatch, tmp_path, provenance, state, reply):
    """Heartbeat markers need completion; legacy internal and empty policies stay intact."""
    runner = _runner(monkeypatch, tmp_path)
    event = _event(internal=provenance == "internal")
    runner.session_store.lookup_by_session_key.return_value = runner.session_store.get_or_create_session.return_value
    if provenance == "heartbeat":
        setattr(event, "_heartbeat_session_id", "sess-silent")
    result = {"final_response": reply, "failed": False, "completed": state != "incomplete",
              "interrupted": state.startswith("interrupted"), "partial": state == "partial",
              "api_calls": 0 if state == "interrupted_before_start" else 1,
              "messages": [{"role": "user", "content": "check"},
                           {"role": "assistant", "content": reply}]}
    runner._run_agent = AsyncMock(return_value=result)
    response = await runner._handle_message_with_agent(
        event, _source(), "agent:main:telegram:group:-1001:12345", 1)
    assert is_intentional_silence_agent_result(result, reply) == (reply == "[SILENT]")
    if reply == "[SILENT]":
        if provenance != "internal":
            assert "silence marker" in response
        else:
            assert response == ""
    elif reply:
        assert response == reply
    elif state == "interrupted":
        assert response == ""
    elif state == "interrupted_before_start":
        assert "interrupted before processing started" in response
    elif state == "partial":
        assert "stop before finishing" in response
    else:
        assert "no response was generated" in response


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
@pytest.mark.parametrize("heartbeat_turn", [False, True])
@pytest.mark.parametrize("status", [{}, {"failed": True}, {"interrupted": True}, {"partial": True}, {"completed": False}])
async def test_queued_first_response_silence_policy(heartbeat_turn, status):
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner._deliver_queued_first_response = AsyncMock()
    turn_ctx = SimpleNamespace(
        session_key="agent:main:telegram:group:-1001:12345",
        stream_consumer_holder=[None],
        persist_user_display_kind=None,
        heartbeat_turn=heartbeat_turn,
        source=_source(),
        _status_thread_metadata=None,
        event_message_id=None,
        inbound_message_id="msg-42",
        run_generation=1,
    )
    result = {"final_response": "NO_REPLY", "failed": False, **status}

    await runner._run_agent_deliver_first_response(
        turn_ctx, None, result, result, None,
    )

    if heartbeat_turn and not status:
        runner._deliver_queued_first_response.assert_not_awaited()
    elif status.get("failed"):
        assert runner._deliver_queued_first_response.await_args.args[0] == "NO_REPLY"
    else:
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
                                    message_type=None, internal=True)

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
