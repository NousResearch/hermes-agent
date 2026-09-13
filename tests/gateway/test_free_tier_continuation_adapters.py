"""Messaging must preserve gate outcomes across its dict-to-text boundaries."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_goals import GatewayGoalsMixin
from gateway.run_turn import GatewayTurnMixin
from gateway.session import SessionSource
from gateway.session_state import SessionState
from gateway.turn_context import TurnContext
from gateway.run_turn_runner import TurnRunner
from hermes_cli.free_tier_usage import LIMIT_NOTICE


def gate(refused=True):
    return {"final_response": LIMIT_NOTICE, "failed": refused, "completed": not refused,
            "failure_reason": "free_tier_limit" if refused else None,
            "refusal_reason": "free_tier_limit" if refused else None,
            "failure_retryable": False, "retryable": True, "continuation_required": True,
            "free_tier": {"capped": True}, "messages": [], "api_calls": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
async def test_gate_outcome_skips_string_only_post_turn_hooks(streamed):
    runner = SimpleNamespace(_final_text_for_post_turn_hooks=GatewayGoalsMixin._final_text_for_post_turn_hooks,
        async_session_store=SimpleNamespace(get_or_create_session=AsyncMock()),
        _post_turn_goal_continuation=AsyncMock(), _post_turn_loop_completion=AsyncMock())
    event = SimpleNamespace(_agent_turn_result=gate(), _streamed_final_response=LIMIT_NOTICE)
    await GatewayGoalsMixin._run_post_turn_hooks(runner, agent_result=None if streamed else LIMIT_NOTICE,
                                                source=None, is_internal=False, event=event)
    runner.async_session_store.get_or_create_session.assert_not_called()
    runner._post_turn_goal_continuation.assert_not_called()
    runner._post_turn_loop_completion.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("refused", [False, True])
async def test_gate_parks_fifo_without_recursive_agent_work(refused):
    source = SessionSource(platform=Platform.LOCAL, chat_id="chat", user_id="user")
    event = MessageEvent(text="queued draft", message_type=MessageType.TEXT, source=source)
    later = MessageEvent(text="later draft", message_type=MessageType.TEXT, source=source)
    command = MessageEvent(text="/model", message_type=MessageType.TEXT, source=source)
    state = SessionState()
    state.conversation.queued_events = [later, command]
    adapter = SimpleNamespace(_pending_messages={"key": event})
    adapter.get_pending_message = lambda key: adapter._pending_messages.pop(key, None)
    runner = SimpleNamespace(_session_state=lambda key: state)
    assert await GatewayTurnMixin._run_agent_drain_pending(runner, gate(refused), adapter, source, "key") == (None, None)
    assert adapter._pending_messages == {"key": command}
    assert state.conversation.queued_events == [event, later]
    # After the off-turn command, the existing FIFO rescue/drain reaches both drafts.
    from gateway.run_busy import GatewayBusySessionMixin
    adapter._pending_messages.pop("key")
    runner._overflow_queue = lambda key: state.conversation.queued_events
    assert GatewayBusySessionMixin._rescue_orphaned_overflow(runner, "key", adapter) is event
    assert adapter.get_pending_message("key") is later
    assert state.conversation.queued_events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_raced_followup_refusal_restores_only_unaccepted_envelope(nested):
    source = SessionSource(platform=Platform.LOCAL, chat_id="chat", user_id="user")
    event = MessageEvent(text="original", source=source, message_id="inbound",
                         media_urls=["image.png"], media_types=["image/png"])
    later = MessageEvent(text="later", source=source)
    state = SessionState()
    state.conversation.queued_events = [later] if not nested else [event, later]
    refused = gate()
    if nested:
        refused["queued_terminal_inbound_id"] = "nested-inbound"
    runner = SimpleNamespace(_MAX_INTERRUPT_DEPTH=8, _run_agent=AsyncMock(return_value=refused),
        _is_goal_continuation_event=lambda _: False, _session_key_for_source=lambda _: "key",
        _prepare_profile_scoped_inbound_message_text=AsyncMock(return_value="expanded"),
        _reply_anchor_for_event=lambda _: None, _adapter_for_source=lambda _: None,
        _refresh_agent_cache_message_count=AsyncMock(), _session_state=lambda _: state)
    ctx = TurnContext(source=source, session_id="sid", session_key="key", history=[])
    result = await GatewayTurnMixin._run_agent_queued_followup(
        runner, ctx, None, event.text, event, "", {"interrupted": True, "messages": []}, None,
    )
    assert state.conversation.queued_events == [event, later]
    assert state.conversation.queued_events[0] is event
    assert event.media_urls == ["image.png"]
    assert result["refusal_reason"] == "free_tier_limit"


@pytest.mark.asyncio
async def test_messaging_preflight_retains_exact_event_before_preparation(monkeypatch):
    source = SessionSource(platform=Platform.LOCAL, chat_id="chat", user_id="user")
    event = MessageEvent(text="original @file:input", message_type=MessageType.TEXT, source=source)
    state = SessionState()
    adapter = SimpleNamespace(_pending_messages={})
    runner = SimpleNamespace(_free_tier_refusal_for_source=Mock(return_value=gate()),
        _session_state=lambda key: state, _hmwa_resolve_session=AsyncMock(),
        _adapter_for_source=lambda _: adapter)
    response = await GatewayTurnMixin._handle_message_with_agent(runner, event, source, "key", 1)
    assert LIMIT_NOTICE in response
    assert "send a message to resume the queue" in response
    assert "resend any unanswered messages" in response
    assert state.conversation.queued_events == [event]
    assert adapter._pending_messages == {}
    assert event.text == "original @file:input"
    assert event._agent_turn_result["failure_retryable"] is False
    runner._hmwa_resolve_session.assert_not_called()


@pytest.mark.asyncio
async def test_heartbeat_never_claims_due_tick_while_gated():
    runner = SimpleNamespace(_free_tier_refusal_for_source=Mock(return_value=gate()),
        _warm_goals_session_db=AsyncMock(side_effect=AssertionError("do not claim a tick")))
    await GatewayGoalsMixin._heartbeat_poll_watch(runner, {}, "key", None, "sid")


@pytest.mark.asyncio
async def test_loop_never_claims_due_tick_while_gated(monkeypatch):
    from hermes_cli import loops
    source = SessionSource(platform=Platform.LOCAL, chat_id="chat")
    runner = SimpleNamespace(_free_tier_refusal_for_source=Mock(return_value=gate()),
        _adapters_for_profile=lambda _: {Platform.LOCAL: object()},
        _build_process_event_source=lambda _: source, _session_key_for_source=lambda _: "key",
        _running_agents={}, _run_in_executor_with_context=AsyncMock())
    monkeypatch.setattr(loops, "goal_blocks_loop_tick", lambda _: False)
    state = SimpleNamespace(awaiting_response=False, next_due_at=0,
                            route={"platform": "local", "chat_id": "chat"})
    await GatewayGoalsMixin._loop_wakeup_fire_one(runner, "sid", state, 100, set())
    runner._run_in_executor_with_context.assert_not_called()


def test_preflight_uses_actual_lifetime_usage_and_selected_route(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from hermes_cli import auth, auth_nous, free_tier_usage as usage
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    guest = {"auth_method": "anonymous", "anon_token": "anon_adapter_fixture"}
    auth._save_active_provider_state("nous", guest)
    auth_nous._write_shared_nous_state(guest)
    identity = usage.current_identity()
    for _ in range(usage.TOOL_CALL_CAP):
        usage.record_completed_tool(identity)
    route = {"base_url": "https://welcome-api.nousresearch.com/v1"}
    runner = SimpleNamespace(_profile_scope_for_source=lambda _: nullcontext(),
        _resolve_session_agent_runtime=lambda **kw: ("model", route))
    assert GatewayTurnMixin._free_tier_refusal_for_source(runner, None, "key")["failure_reason"] == usage.LIMIT_REASON
    route["base_url"] = "http://localhost:8080/v1"
    assert GatewayTurnMixin._free_tier_refusal_for_source(runner, None, "key") is None
    route["base_url"] = "https://api.openai.com/v1"
    assert GatewayTurnMixin._free_tier_refusal_for_source(runner, None, "key") is None
    assert usage.identity_status(identity)["capped"]  # route changes never reset allowance


def test_completed_gate_footer_reaches_stream_seal_without_history_change():
    history = [{"role": "assistant", "content": "answer"}]
    result = gate(False)
    result.update(final_response="answer\n\n" + LIMIT_NOTICE, messages=history)
    tr = TurnRunner(MagicMock(), TurnContext())
    consumer = SimpleNamespace(finish=Mock())
    tr._finish_stream_consumer(result, [], consumer)
    consumer.finish.assert_called_once_with(result["final_response"])
    assert history == [{"role": "assistant", "content": "answer"}]


def test_turn_runner_preserves_gate_metadata():
    runner = MagicMock()
    runner._resolve_session_agent_runtime.return_value = ("model", {})
    runner._provider_routing = {}
    ctx = TurnContext(source=SessionSource(platform=Platform.LOCAL, chat_id="chat"),
                      message="input", history=[], session_id="sid", session_key="key", user_config={})
    tr = TurnRunner(runner, ctx)
    agent = SimpleNamespace(model="model")
    ctx.agent_holder[0] = agent
    tr._combined_ephemeral_prompt = lambda: ""
    tr._setup_stream_consumer = lambda _: (None, None, None, False)
    tr._resolve_turn_agent = lambda *a: (agent, False)
    tr._wire_turn_agent_callbacks = lambda *a: None
    tr._load_turn_history = lambda *a: ([], None, set())
    tr._prepare_turn_message = lambda *a: (None, None)
    tr._run_conversation_with_approval = lambda *a: gate()
    tr._sync_session_after_run = lambda *a: (False, "sid", 0)
    result = tr.run_sync()
    assert result["failure_retryable"] is False
    assert result["retryable"] is True  # manual resend after recovery, never an automatic retry
    assert result["refusal_reason"] == "free_tier_limit"
    assert result["continuation_required"] is True
    assert result["free_tier"]["capped"] is True
