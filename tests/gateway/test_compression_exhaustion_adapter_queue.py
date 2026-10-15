"""Native adapter FIFO behavior during terminal compression and manual recovery."""
import asyncio
import threading
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from tests.gateway.test_compression_exhaustion_reset_policy import (
    mark, policy, native_message, native_recovery_command, reload_entry, exhaust,
)

pytest_plugins = ("tests.gateway.test_compression_exhaustion_reset_policy",)

@pytest.mark.asyncio
@pytest.mark.parametrize("command,outcome", [
    ("new", "primary_failure"),
    ("compress", "primary_failure"),
    ("resume", "primary_failure"),
    ("resume", "missing_target"),
    ("retry", "paused_control"),
    ("new", "committed_control"),
])
async def test_paused_manual_command_preserves_accepted_adapter_queue_on_refusal(
    native_env, monkeypatch, command, outcome,
):
    from gateway.config import PlatformConfig
    from gateway.platforms.base import SendResult
    from tests.gateway.test_pending_drain_no_recursion import _StubAdapter

    e = native_env
    r = e.runner
    mark(e)
    policy(e, "compression:\n  exhaustion_action: pause\n")
    adapter = _StubAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="reply"))
    adapter._message_handler = r._primary_message_handler()
    r.adapters[Platform.TELEGRAM] = adapter
    pending = MessageEvent(text="accepted before recovery", source=e.source, message_id="pending")
    overflow = MessageEvent(text="/queue second in FIFO", source=e.source, message_id="overflow")
    adapter._active_sessions[e.key] = asyncio.Event()
    try:
        await adapter.handle_message(pending)
        assert pending._gateway_accepted and adapter._pending_messages[e.key] is pending
        await r._busy_queue_command(overflow, e.key, e.source)
    finally:
        adapter._active_sessions.pop(e.key, None)
    assert [item.text for item in r._overflow_queue(e.key)] == ["second in FIFO"]

    text = native_recovery_command(e, monkeypatch, command)
    if outcome == "missing_target":
        text = "/resume nonexistent-target"
    elif outcome == "primary_failure":
        monkeypatch.setattr(e.db, "replace_gateway_routing_entries",
                            Mock(side_effect=OSError("primary unavailable")))
    try:
        await adapter.handle_message(MessageEvent(text=text, source=e.source, message_id="command"))
        while adapter._background_tasks:
            await asyncio.gather(*tuple(adapter._background_tasks))
            await asyncio.sleep(0)
        current = reload_entry(e)
        assert current is not None
        if outcome == "committed_control":
            # A successful native /new deliberately clears its old queue (#111142).
            assert current.session_id != e.entry.session_id and not current.compression_paused
        else:
            assert current.session_id == e.entry.session_id and current.compression_paused
            assert pending._gateway_accepted
            assert adapter._pending_messages.get(e.key) is pending
            assert [item.text for item in r._overflow_queue(e.key)] == ["second in FIFO"]
            r._hmwa_prepare_turn.assert_not_awaited()
            r._run_agent.assert_not_awaited()
    finally:
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["reset", "pause", "reset_failure"])
async def test_terminal_model_pending_steer_survives_with_adapter_fifo(
    native_env, monkeypatch, action,
):
    from gateway.config import PlatformConfig
    from gateway.platforms.base import SendResult
    from tests.gateway.test_compression_failure_session_sync import (
        _CompressionThenFailureAgent, _install_compression_failure_agent,
    )
    from tests.gateway.test_pending_drain_no_recursion import _StubAdapter

    e = native_env
    r = e.runner
    del r._hmwa_prepare_turn
    del r._run_agent
    r._agent_cache = {}
    r._agent_cache_lock = threading.Lock()
    r._prefill_messages = []
    r._ephemeral_system_prompt = ""
    r._reasoning_config = r._service_tier = r._fallback_model = None
    r._provider_routing = {}
    r.hooks.loaded_hooks = []
    r._voice_mode = {}
    r._get_proxy_url = lambda: None
    r._resolve_session_agent_runtime = lambda **kwargs: (
        "gpt-5.4", {"provider": "custom", "api_key": "fixture-key",
                    "base_url": "https://model.invalid/v1", "api_mode": "chat_completions"},
    )
    adapter = _StubAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="reply"))
    adapter._message_handler = r._primary_message_handler()
    r.adapters[Platform.TELEGRAM] = adapter
    accepted = MessageEvent(text="adapter accepted first", source=e.source, message_id="accepted")
    overflow = MessageEvent(text="/queue overflow second", source=e.source, message_id="overflow")
    # The model's steer payload is user data, never authority for another slash command.
    steer = "/new literal user steer" if action == "reset" else "late native steer third"
    model_inputs = []
    loop = asyncio.get_running_loop()

    async def enqueue_waiting():
        await adapter.handle_message(accepted)
        assert accepted._gateway_accepted
        await r._busy_queue_command(overflow, e.key, e.source)
        if action == "reset_failure":
            # A paused route requires primary persistence; fresh resets retain JSON fallback.
            mark(e)
            monkeypatch.setattr(e.db, "replace_gateway_routing_entries",
                                Mock(side_effect=OSError("primary unavailable")))

    class ModelBoundary(_CompressionThenFailureAgent):
        def run_conversation(self, user_message, conversation_history=None, task_id=None, **kwargs):
            model_inputs.append(user_message)
            if len(model_inputs) == 1:
                asyncio.run_coroutine_threadsafe(enqueue_waiting(), loop).result(5)
            terminal = len(model_inputs) == 1
            response = "context full" if terminal else "completed"
            return {
                "final_response": response, "failed": terminal, "completed": not terminal,
                "compression_exhausted": terminal, "compression_deferred": False,
                "failure_reason": "context_overflow" if terminal else None,
                "pending_steer": steer if terminal else None,
                "messages": [*(conversation_history or []),
                             {"role": "user", "content": user_message},
                             {"role": "assistant", "content": response}],
                "api_calls": 1, "agent_persisted": False,
            }

    _install_compression_failure_agent(monkeypatch, ModelBoundary)
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda *_: "gpt-5.4")
    policy(e, f"compression:\n  exhaustion_action: {'pause' if action == 'pause' else 'reset'}\n")
    event = MessageEvent(text="first request", source=e.source, message_id="initial")
    try:
        await adapter.handle_message(event)
        while adapter._background_tasks:
            await asyncio.gather(*tuple(adapter._background_tasks))
            await asyncio.sleep(0)
        assert accepted._gateway_accepted
        if action == "reset":
            current = reload_entry(e)
            assert current is not None
            assert current.session_id != e.entry.session_id and not current.compression_paused
            assert "first request" in model_inputs[0]
            assert accepted.text in model_inputs[1]
            assert model_inputs[2:] == ["overflow second", steer]
            assert e.key not in adapter._pending_messages
            assert not r._overflow_queue(e.key)
        else:
            current = e.store.lookup_by_session_key(e.key)
            assert current.session_id == e.entry.session_id and current.compression_paused
            assert model_inputs == ["first request"]
            assert adapter._pending_messages.get(e.key) is accepted
            assert [item.text for item in r._overflow_queue(e.key)] == ["overflow second", steer]
    finally:
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["exhausted", "refused", "reset", "completed"])
async def test_fired_loop_tick_pauses_on_compression_without_rescheduling(
    native_env, monkeypatch, outcome,
):
    from hermes_cli.loops import LoopManager

    e = native_env
    monkeypatch.setenv("HERMES_HOME", str(e.tmp))  # LoopManager and SessionStore share the real state.db.
    del e.runner._post_turn_loop_completion
    del e.runner._run_post_turn_hooks
    manager = LoopManager(session_id=e.entry.session_id)
    manager.set("check the task", interval_seconds=60)
    assert manager.fire_tick() is not None
    assert manager.state is not None and manager.state.awaiting_response
    event = MessageEvent(text="loop wakeup", source=e.source)
    if outcome == "exhausted":
        policy(e, "compression:\n  exhaustion_action: pause\n")
        reply, _, event = await exhaust(e)
        assert "paused" in reply.lower() and getattr(event, "_gateway_skip_goal_continuation", False)
    elif outcome == "reset":
        policy(e, "compression:\n  exhaustion_action: reset\n")
        reply, _, event = await exhaust(e)
        reset_entry = reload_entry(e)
        assert reset_entry is not None
        assert "auto-reset" in reply and reset_entry.session_id != e.entry.session_id
    elif outcome == "refused":
        mark(e)
        policy(e, "compression:\n  exhaustion_action: pause\n")
        reply = await e.runner._handle_message(event)
        assert "paused" in str(reply).lower()
    else:
        reply = "completed the task"
    await e.runner._run_post_turn_hooks(agent_result=reply, source=e.source,
                                        is_internal=True, event=event)
    persisted = LoopManager(session_id=e.entry.session_id).state
    assert persisted is not None and persisted.ticks_fired == 1
    assert not persisted.awaiting_response
    if outcome == "completed":
        assert persisted.status == "active" and persisted.next_due_at > persisted.last_fired_at
    else:
        assert persisted.status == "paused"
        if outcome == "reset":
            reset_entry = reload_entry(e)
            assert reset_entry is not None
            assert LoopManager(session_id=reset_entry.session_id).state is None
        e.runner._post_turn_goal_continuation.assert_not_awaited()
