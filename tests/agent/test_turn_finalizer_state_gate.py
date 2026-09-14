from types import SimpleNamespace
from unittest.mock import MagicMock

import agent.turn_finalizer as finalizer
from agent.state_candidate_evaluator import CandidateStatus, DeltaType, StateCandidateResult
from agent.state_response_policy import ResponsePolicy


def test_gate_stop_passes_through_finalizer_and_persists(monkeypatch):
    persisted = []
    agent = MagicMock()
    agent.max_iterations = 3
    agent.model = "fixture-model"
    agent.provider = "fixture-provider"
    agent.base_url = "fixture-url"
    agent.session_id = "session-1"
    agent.platform = "test"
    agent.context_compressor.last_prompt_tokens = 0
    agent.request_overrides = {}
    agent._tool_guardrail_halt_decision = None
    agent._response_was_previewed = False
    agent._skill_nudge_interval = 0
    agent._iters_since_skill = 0
    agent.valid_tool_names = set()
    agent._last_turn_usage = None
    agent._interrupt_message = None
    agent._persist_session.side_effect = lambda messages, history: persisted.append(list(messages)) or True
    agent._drain_pending_steer.return_value = None

    monkeypatch.setattr(finalizer, "_resolve_budget_fallback", lambda agent, **kwargs: (
        kwargs["final_response"], kwargs["_turn_exit_reason"], None
    ))
    monkeypatch.setattr(finalizer, "_drop_transcript_scaffolding", lambda agent, messages: None)
    monkeypatch.setattr(finalizer, "_recover_final_from_stream", lambda agent, response, interrupted, failed: (response, False))
    monkeypatch.setattr(finalizer, "_append_file_mutation_footer", lambda agent, response, logger: response)
    monkeypatch.setattr(finalizer, "_explain_abnormal_exit", lambda agent, response, reason, fallback, logger: response)
    monkeypatch.setattr(finalizer, "_apply_output_hooks", lambda agent, response, logger, **kwargs: (response, False, None))
    # Keep transcript-tail shaping real: this is the boundary under test.
    monkeypatch.setattr(finalizer, "_micro_compact_after_turn", lambda *args: None)
    monkeypatch.setattr(finalizer, "_last_turn_reasoning", lambda messages: None)
    monkeypatch.setattr(finalizer, "_log_turn_exit", lambda *args: None)
    monkeypatch.setattr(finalizer, "_notify_context_engine_turn_complete", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(finalizer, "_invoke_hook_safely", lambda *args, **kwargs: [])

    result = finalizer.finalize_turn(
        agent,
        final_response="確認が必要です。",
        api_call_count=0,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "更新内容"}],
        conversation_history=[],
        effective_task_id=None,
        turn_id="turn-1",
        user_message="更新内容",
        original_user_message="更新内容",
        _should_review_memory=False,
        _turn_exit_reason="state_gate_ask_confirmation",
        blocked=True,
    )

    assert len(persisted) == 1
    assert [row["role"] for row in persisted[0]] == ["user", "assistant"]
    assert persisted[0][1]["content"] == "確認が必要です。"
    assert result["final_response"] == "確認が必要です。"
    assert result["persistence_confirmed"] is True
    assert result["completed"] is True
    assert result["blocked"] is True
    assert result["turn_exit_reason"] == "state_gate_ask_confirmation"
    assert result["failed"] is False
