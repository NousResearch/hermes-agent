"""Regression tests for execute_code outer turn iteration_budget accounting."""

from types import SimpleNamespace

import agent.conversation_loop
from agent.iteration_budget import IterationBudget
from agent.turn_iteration_prep import _maybe_inject_iteration_budget_warning
import agent.turn_tool_round as round_module


def _tool_call(call_id: str, name: str, arguments: str = '{"code": "print(1)"}'):
    return SimpleNamespace(
        id=call_id,
        call_id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _make_stub_agent(max_turns: int = 5):
    budget = IterationBudget(max_turns)
    agent = SimpleNamespace(
        iteration_budget=budget,
        max_iterations=max_turns,
        session_id="sess-16",
        quiet_mode=True,
        verbose_logging=False,
        _stream_needs_break=False,
        _session_messages=[],
        _tool_guardrail_halt_decision=None,
        _incremental_persistence_failed=False,
        _iteration_budget_warning_injected=False,
        budget_warning_ratio=0.8,
        valid_tool_names={"execute_code", "read_file"},
        stream_delta_callback=None,
        _hold_stream=False,
        _cap_delegate_task_calls=lambda tcs: tcs,
        _deduplicate_tool_calls=lambda tcs: tcs,
        _flush_messages_to_session_db=lambda *_a, **_k: True,
        _emit_interim_assistant_message=lambda *_a, **_k: None,
        _execute_tool_calls=lambda assistant_message, msgs, *_a: msgs.append(
            {
                "role": "tool",
                "name": assistant_message.tool_calls[0].function.name,
                "tool_call_id": assistant_message.tool_calls[0].id,
                "content": "ok",
            }
        ),
        _touch_activity=lambda *_a, **_k: None,
        _persist_session_activity=lambda *_a, **_k: None,
        _emit_diagnostic_status=lambda *_a, **_k: None,
        _safe_print=lambda *_a, **_k: None,
    )
    return agent


def test_execute_code_only_turn_does_not_refund_outer_iteration_budget(monkeypatch):
    agent = _make_stub_agent(max_turns=5)

    monkeypatch.setattr(
        round_module,
        "validate_tool_calls",
        lambda *_a, **_kw: SimpleNamespace(
            action="ok",
            mixed_invalid_batch=False,
            result=None,
        ),
    )
    monkeypatch.setattr(
        round_module,
        "stage_tool_call_message",
        lambda _ag, **kw: (
            {"role": "assistant", "content": "", "tool_calls": kw["assistant_message"].tool_calls},
            False,
        ),
    )
    monkeypatch.setattr("agent.prepared_compaction.maybe_prepare", lambda *_a, **_k: None)
    monkeypatch.setattr("hermes_cli.observability.shared_metrics_harness.finish_tool_round", lambda *_a, **_k: None)
    monkeypatch.setattr(
        round_module,
        "compress_after_tool_results",
        lambda _ag, **kw: SimpleNamespace(
            end_turn=False,
            messages=kw["messages"],
            active_system_prompt=kw["active_system_prompt"],
            conversation_history=kw["conversation_history"],
            compression_attempts=kw["compression_attempts"],
            final_response=kw["final_response"],
            turn_exit_reason=kw["turn_exit_reason"],
            current_turn_user_idx=kw["current_turn_user_idx"],
        ),
    )

    messages = [{"role": "user", "content": "run scripts"}]
    for turn_idx in range(1, 6):
        assert agent.iteration_budget.consume() is True
        verdict = round_module.run_tool_round(
            agent,
            assistant_message=SimpleNamespace(
                content="",
                tool_calls=[_tool_call(f"call_{turn_idx}", "execute_code")],
                provider_data=None,
            ),
            finish_reason="tool_calls",
            messages=messages,
            conversation_history=[],
            api_call_count=turn_idx,
            effective_task_id="task-16",
            user_message="run scripts",
            system_message="",
            active_system_prompt="",
            compression_attempts=0,
            max_compression_attempts=2,
            final_response=None,
            failed=False,
            _turn_exit_reason=None,
            truncated_tool_call_retries=0,
            current_turn_user_idx=0,
        )
        assert verdict.action == "continue"
        assert agent.iteration_budget.used == turn_idx

    # After 5 outer execute_code turns against max_turns=5, budget is exhausted (not stuck at 0/5).
    assert agent.iteration_budget.remaining == 0
    assert agent.iteration_budget.consume() is False


def test_budget_warning_fires_at_threshold_across_execute_code_turns():
    agent = _make_stub_agent(max_turns=5)
    messages = [
        {"role": "user", "content": "analyze data"},
        {"role": "tool", "name": "execute_code", "tool_call_id": "call_4", "content": "step 4 output"},
    ]

    # Simulate 4 consumed execute_code outer turns (4/5 == 80%).
    for _ in range(4):
        assert agent.iteration_budget.consume() is True

    injected = _maybe_inject_iteration_budget_warning(agent, messages)
    assert injected is True
    assert agent._iteration_budget_warning_injected is True
    assert "You have used 4 of 5 iterations" in messages[-1]["content"]
