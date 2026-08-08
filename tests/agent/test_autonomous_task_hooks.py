import logging
import sys
from types import ModuleType, SimpleNamespace

from agent.turn_finalizer import finalize_turn


def _install_lightweight_conversation_loop(monkeypatch):
    module = ModuleType("agent.conversation_loop")
    setattr(module, "logger", logging.getLogger("test.conversation_loop"))
    setattr(module, "_notify_context_engine_turn_complete", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "agent.conversation_loop", module)


def _agent_for_finalize(max_iterations=90):
    return SimpleNamespace(
        max_iterations=max_iterations,
        iteration_budget=SimpleNamespace(remaining=20, used=70, max_total=max_iterations),
        quiet_mode=True,
        model="test-model",
        provider="test-provider",
        base_url="https://example.invalid",
        session_id="session-123",
        context_compressor=SimpleNamespace(last_prompt_tokens=0),
        _tool_guardrail_halt_decision=None,
        _response_was_previewed=False,
        _turn_failed_file_mutations={},
        session_input_tokens=0,
        session_output_tokens=0,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_estimated_cost_usd=0.0,
        session_cost_status="ok",
        session_cost_source="test",
        request_overrides={},
        clear_interrupt=lambda: None,
        _drain_pending_steer=lambda: None,
        _save_trajectory=lambda *args, **kwargs: None,
        _cleanup_task_resources=lambda *args, **kwargs: None,
        _drop_trailing_empty_response_scaffolding=lambda messages: None,
        _apply_persist_user_message_override=lambda messages: None,
        _persist_session=lambda *args, **kwargs: None,
        _file_mutation_verifier_enabled=lambda: False,
        _turn_completion_explainer_enabled=lambda: False,
        _handle_max_iterations=lambda messages, api_call_count: "budget summary",
        _emit_status=lambda *args, **kwargs: None,
        _skill_nudge_interval=0,
        _iters_since_skill=0,
        valid_tool_names=set(),
        _trigger_review_background=lambda *args, **kwargs: None,
        _sync_external_memory_for_turn=lambda *args, **kwargs: None,
        _spawn_background_review=lambda *args, **kwargs: None,
        _interrupt_message=None,
        platform="test",
    )


def test_autonomous_task_hook_allows_only_metadata_fields(monkeypatch):
    from agent.autonomous_task_hooks import notify_turn_budget_or_closeout

    captured = {}

    def fake_invoke(hook_name, **kwargs):
        captured["hook_name"] = hook_name
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", fake_invoke)

    notify_turn_budget_or_closeout(
        event="max_turn_or_tool_iteration_closeout",
        turns_used=90,
        max_turns=90,
        session_id="session-123",
        messages=[{"role": "user", "content": "must not leak"}],
        tool_outputs=["must not leak"],
        secrets={"token": "must not leak"},
    )

    assert captured == {
        "hook_name": "autonomous_task_turn_budget",
        "kwargs": {
            "event": "max_turn_or_tool_iteration_closeout",
            "turns_used": 90,
            "max_turns": 90,
            "session_id": "session-123",
        },
    }


def test_autonomous_task_hook_is_fail_open(monkeypatch):
    from agent.autonomous_task_hooks import notify_turn_budget_or_closeout

    def boom(*args, **kwargs):
        raise RuntimeError("hook consumer failed")

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", boom)

    assert notify_turn_budget_or_closeout(
        event="preemptive_turn_budget_threshold",
        turns_used=70,
        max_turns=90,
        session_id="session-123",
    ) is False


def test_turn_finalizer_emits_preemptive_metadata_only_hook(monkeypatch):
    calls = []

    def fake_notify(**kwargs):
        calls.append(kwargs)
        return True

    _install_lightweight_conversation_loop(monkeypatch)
    monkeypatch.setattr("agent.autonomous_task_hooks.notify_turn_budget_or_closeout", fake_notify)
    agent = _agent_for_finalize(max_iterations=90)
    result = finalize_turn(
        agent,
        final_response="Done.",
        api_call_count=70,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "do work"}, {"role": "assistant", "content": "Done."}],
        conversation_history=[],
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do work",
        original_user_message="do work",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert result["completed"] is True
    assert calls == [
        {
            "event": "preemptive_turn_budget_threshold",
            "turns_used": 70,
            "max_turns": 90,
            "session_id": "session-123",
        }
    ]


def test_turn_finalizer_emits_max_turn_metadata_only_hook(monkeypatch):
    calls = []

    def fake_notify(**kwargs):
        calls.append(kwargs)
        return True

    _install_lightweight_conversation_loop(monkeypatch)
    monkeypatch.setattr("agent.autonomous_task_hooks.notify_turn_budget_or_closeout", fake_notify)
    agent = _agent_for_finalize(max_iterations=90)
    agent.iteration_budget = SimpleNamespace(remaining=0, used=90, max_total=90)
    result = finalize_turn(
        agent,
        final_response=None,
        api_call_count=90,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "do work"}],
        conversation_history=[],
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do work",
        original_user_message="do work",
        _should_review_memory=False,
        _turn_exit_reason="budget_exhausted",
    )

    assert result["final_response"] == "budget summary"
    assert calls == [
        {
            "event": "max_turn_or_tool_iteration_closeout",
            "turns_used": 90,
            "max_turns": 90,
            "session_id": "session-123",
        }
    ]
