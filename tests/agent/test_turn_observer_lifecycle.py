"""Observer starts must be balanced across loop exits, without duplicate ends."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.conversation_loop import run_conversation
from agent.turn_context import _collect_pre_llm_call_context
from agent.turn_finalizer import finalize_turn

_LIFECYCLE_PAIR = ("pre_llm_call", "on_session_end")


def _agent():
    """Turn-loop surface plus the collaborators the real ``finalize_turn`` calls unguarded."""
    agent = SimpleNamespace(
        session_id="session", model="model", provider="provider", base_url="",
        platform="cli", _persist_disabled=False, max_iterations=60,
        iteration_budget=SimpleNamespace(remaining=60, used=0, max_total=60),
        context_compressor=SimpleNamespace(last_real_prompt_tokens=0, last_prompt_tokens=0),
        quiet_mode=True, valid_tool_names=[], persisted_messages=None, _session_messages=None,
        _tool_guardrail_halt_decision=None, _interrupt_message=None,
        _response_was_previewed=False, _skill_nudge_interval=0, _iters_since_skill=0,
        _last_turn_usage=None, _current_streamed_assistant_text="", _turn_origin=None,
        log_prefix="[test]", session_cost_status="unknown", session_cost_source="test",
    )
    for token_key in (
        "session_input_tokens", "session_output_tokens", "session_cache_read_tokens",
        "session_cache_write_tokens", "session_reasoning_tokens", "session_prompt_tokens",
        "session_completion_tokens", "session_total_tokens", "session_estimated_cost_usd",
    ):
        setattr(agent, token_key, 0)
    agent._file_mutation_verifier_enabled = lambda: False
    agent._turn_completion_explainer_enabled = lambda: False
    agent._format_turn_completion_explanation = lambda *a, **k: ""
    agent._handle_max_iterations = lambda *a, **k: ""
    agent._drain_pending_steer = lambda: None
    agent._sync_external_memory_for_turn = lambda *a, **k: None
    agent._emit_status = agent._safe_print = lambda *a, **k: None
    agent._save_trajectory = agent._cleanup_task_resources = lambda *a, **k: None
    agent._drop_trailing_empty_response_scaffolding = lambda messages: None
    agent._persist_session = lambda messages, history: None
    agent.clear_interrupt = lambda: None
    return agent


_FINALIZED_TURN = {
    "final_response": "done", "api_call_count": 1, "interrupted": False, "failed": False,
    "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "done"}],
    "conversation_history": [], "effective_task_id": "task", "turn_id": "turn",
    "user_message": "hi", "original_user_message": "hi", "_should_review_memory": False,
    "_turn_exit_reason": "text_response(stop)",
}


@pytest.mark.parametrize("exit_kind,observer_fails", [
    ("early", False), ("error", False), ("interrupt", False), ("finalized", False),
    ("early", True), ("error", True), ("interrupt", True),
])
def test_observed_turn_has_one_correlated_end_across_exit_paths(exit_kind, observer_fails):
    agent = _agent()
    events = []

    def hook(name, **kwargs):
        events.append((name, kwargs))
        if name == "on_session_end" and observer_fails:
            raise OSError("observer failure must not replace the turn outcome")
        return []

    def turn(*args, **kwargs):
        _collect_pre_llm_call_context(
            agent, effective_task_id="task", turn_id="turn", original_user_message="hello",
            messages=[], conversation_history=None,
        )
        if exit_kind == "error":
            raise RuntimeError("failure")
        if exit_kind == "interrupt":
            raise KeyboardInterrupt
        if exit_kind == "finalized":
            return finalize_turn(agent, **_FINALIZED_TURN)
        return {"messages": [], "completed": True}

    with (
        patch("agent.conversation_loop._run_conversation_turn", side_effect=turn),
        patch("hermes_cli.lifecycle.invoke_hook", side_effect=hook),
        patch("agent.turn_context.export_current_turn_boundary", side_effect=lambda _a, r, _m: r),
        patch("agent.conversation_loop._close_durable_failed_turn"),
    ):
        if exit_kind in {"error", "interrupt"}:
            with pytest.raises(RuntimeError if exit_kind == "error" else KeyboardInterrupt):
                run_conversation(agent, "hello")
        else:
            run_conversation(agent, "hello")
    pair = [(name, payload) for name, payload in events if name in _LIFECYCLE_PAIR]
    assert [name for name, _ in pair] == list(_LIFECYCLE_PAIR)
    start, end = (payload for _, payload in pair)
    assert {key: start[key] for key in ("session_id", "task_id", "turn_id")} == {
        key: end[key] for key in ("session_id", "task_id", "turn_id")
    }
    if exit_kind == "interrupt":
        assert end["interrupted"] and not end["completed"]
    if exit_kind == "error":
        assert end["failed"] and not end["completed"]


@pytest.mark.parametrize("persist_disabled", [False, True])
def test_unobserved_exit_does_not_emit_end_or_mask_failure(persist_disabled):
    agent = SimpleNamespace(_persist_disabled=persist_disabled)
    with (
        patch("agent.conversation_loop._run_conversation_turn", side_effect=ValueError("original")),
        patch("hermes_cli.lifecycle.invoke_hook") as hook,
        pytest.raises(ValueError, match="original"),
    ):
        run_conversation(agent, "hello")
    hook.assert_not_called()
