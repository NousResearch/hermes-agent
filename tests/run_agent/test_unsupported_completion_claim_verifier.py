"""Regression tests for housekeeping-only completion-claim guardrails."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from agent.turn_finalizer import finalize_turn
from run_agent import AIAgent


def _bare_agent(tool_names: list[str]) -> AIAgent:
    agent = object.__new__(AIAgent)
    agent._turn_tool_evidence = [
        {"tool": name, "is_error": False} for name in tool_names
    ]
    return agent


def test_housekeeping_only_done_claim_gets_footer():
    agent = _bare_agent(["todo", "skill_view"])

    footer = agent._format_unsupported_completion_claim_footer(
        "STATUS: DONE\nImplemented and verified."
    )

    assert "todo, skill_view" in footer
    assert "not supported by this turn's tool evidence" in footer


def test_non_housekeeping_tool_evidence_suppresses_footer():
    agent = _bare_agent(["todo", "terminal"])

    footer = agent._format_unsupported_completion_claim_footer(
        "STATUS: DONE\nImplemented and verified."
    )

    assert footer == ""


def test_housekeeping_only_normal_reply_stays_quiet():
    agent = _bare_agent(["memory", "session_search"])

    footer = agent._format_unsupported_completion_claim_footer(
        "I found the session notes you asked about."
    )

    assert footer == ""


def test_read_context_verified_language_stays_quiet():
    agent = _bare_agent(["session_search"])

    footer = agent._format_unsupported_completion_claim_footer(
        "I verified the prior session history and tested the search query wording."
    )

    assert footer == ""


def test_housekeeping_only_negated_claim_stays_quiet():
    agent = _bare_agent(["todo"])

    footer = agent._format_unsupported_completion_claim_footer(
        "This is not implemented and could not be verified from context."
    )

    assert footer == ""


def _finalizer_agent(tool_names: list[str], persisted: list[list[dict]]) -> AIAgent:
    agent = _bare_agent(tool_names)
    agent.max_iterations = 5
    agent.iteration_budget = SimpleNamespace(remaining=4, used=1, max_total=5)
    agent.quiet_mode = True
    agent.model = "test-model"
    agent.provider = "test-provider"
    agent.base_url = "http://test.invalid/v1"
    agent.session_id = "session-1"
    agent.context_compressor = SimpleNamespace(last_prompt_tokens=0)
    agent._turn_failed_file_mutations = {}
    agent._tool_guardrail_halt_decision = None
    agent._response_was_previewed = False
    agent._interrupt_message = None
    agent._skill_nudge_interval = 0
    agent._iters_since_skill = 0
    agent.valid_tool_names = set()

    for attr in (
        "session_input_tokens",
        "session_output_tokens",
        "session_cache_read_tokens",
        "session_cache_write_tokens",
        "session_reasoning_tokens",
        "session_prompt_tokens",
        "session_completion_tokens",
        "session_total_tokens",
        "session_estimated_cost_usd",
    ):
        setattr(agent, attr, 0)
    agent.session_cost_status = ""
    agent.session_cost_source = ""

    agent._save_trajectory = lambda *_a, **_k: None
    agent._cleanup_task_resources = lambda *_a, **_k: None
    agent._drop_trailing_empty_response_scaffolding = lambda *_a, **_k: None
    agent._persist_session = lambda messages, _history: persisted.append(
        deepcopy(messages)
    )
    agent._turn_completion_explainer_enabled = lambda: False
    agent._drain_pending_steer = lambda: None
    agent.clear_interrupt = lambda: None
    agent._sync_external_memory_for_turn = lambda **_k: None
    agent._spawn_background_review = lambda **_k: None
    return agent


def test_finalizer_updates_returned_and_persisted_assistant_message(monkeypatch):
    try:
        import hermes_cli.plugins as plugins

        monkeypatch.setattr(plugins, "invoke_hook", lambda *_a, **_k: [])
    except Exception:
        pass

    persisted: list[list[dict]] = []
    agent = _finalizer_agent(["todo"], persisted)
    messages = [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "STATUS: DONE\nImplemented and verified."},
    ]

    result = finalize_turn(
        agent,
        final_response=messages[-1]["content"],
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="fix it",
        original_user_message="fix it",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert "not supported by this turn's tool evidence" in result["final_response"]
    assert messages[-1]["content"] == result["final_response"]
    assert persisted[-1][-1]["content"] == result["final_response"]


def test_finalizer_persists_transform_hook_output(monkeypatch):
    import hermes_cli.plugins as plugins

    monkeypatch.setattr(
        plugins,
        "invoke_hook",
        lambda hook, **kwargs: ["transformed delivery"]
        if hook == "transform_llm_output"
        else [],
    )
    persisted: list[list[dict]] = []
    agent = _finalizer_agent(["terminal"], persisted)
    messages = [
        {"role": "user", "content": "report"},
        {"role": "assistant", "content": "original response"},
    ]

    result = finalize_turn(
        agent,
        final_response="original response",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="report",
        original_user_message="report",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert result["final_response"] == "transformed delivery"
    assert messages[-1]["content"] == "transformed delivery"
    assert persisted[-1][-1]["content"] == "transformed delivery"
