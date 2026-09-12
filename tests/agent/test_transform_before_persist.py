"""Regression tests: ``transform_llm_output`` must run before persistence.

The hook used to fire after ``_persist_session``, so a plugin's rewritten
response was delivered to the user but never landed in the durable session
— ``/resume`` replayed the original text and ``post_llm_call`` observers
synced the untransformed transcript. These tests pin the new contract:

1. the hook fires before persistence,
2. the closing assistant row in the persisted transcript carries the
   transformed text (with the flush marker dropped so the rewrite is
   actually written to state.db),
3. the returned result dict reports the transformation.
"""

import pytest

from agent.turn_finalizer import finalize_turn


class _StubBudget:
    used = 0
    max_total = 10
    remaining = 9


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface that ``finalize_turn`` reads from."""

    def __init__(self):
        self.max_iterations = 10
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
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
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

        self.events = []
        self.persisted_snapshots = []

    # --- observed surfaces ----------------------------------------------
    def _persist_session(self, messages, conversation_history):
        self.events.append("persist")
        self.persisted_snapshots.append([dict(m) for m in messages])

    # --- harmless no-ops -------------------------------------------------
    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, *a, **k):
        pass

    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _handle_max_iterations(self, messages, n):
        return "PARTIAL SUMMARY FROM MODEL"

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


def _run(agent, *, final_response="Original"):
    messages = [
        {"role": "user", "content": "do a thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file contents"},
    ]
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason="stop",
    )


def test_transform_runs_before_persist(monkeypatch):
    agent = _StubAgent()

    def mock_invoke_hook(hook_name, **kwargs):
        agent.events.append(f"hook:{hook_name}")
        if hook_name == "transform_llm_output":
            return ["Transformed"]
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", mock_invoke_hook)

    result = _run(agent)

    assert "hook:transform_llm_output" in agent.events
    assert "persist" in agent.events
    assert agent.events.index("hook:transform_llm_output") < agent.events.index(
        "persist"
    )
    assert result["final_response"] == "Transformed"
    assert result["response_transformed"] is True
    assert result["pre_transform_response"] == "Original"


def test_persisted_transcript_carries_transformed_text(monkeypatch):
    agent = _StubAgent()

    def mock_invoke_hook(hook_name, **kwargs):
        if hook_name == "transform_llm_output":
            return ["Transformed"]
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", mock_invoke_hook)

    result = _run(agent)

    snapshot = agent.persisted_snapshots[0]
    tail = snapshot[-1]
    assert tail["role"] == "assistant"
    # The pure tool-call tail was filled with the TRANSFORMED response, not
    # the original, so /resume replays what the user actually saw.
    assert tail["content"] == "Transformed"
    assert "_db_persisted" not in tail
    assert result["final_response"] == "Transformed"


def test_no_hook_result_leaves_response_and_transcript_intact(monkeypatch):
    agent = _StubAgent()

    def mock_invoke_hook(hook_name, **kwargs):
        if hook_name == "transform_llm_output":
            return [None, ""]
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", mock_invoke_hook)

    result = _run(agent)

    assert result["final_response"] == "Original"
    assert result["response_transformed"] is False
    tail = agent.persisted_snapshots[0][-1]
    assert tail["content"] == "Original"


def test_post_llm_call_receives_transformed_response(monkeypatch):
    agent = _StubAgent()
    seen_responses = []

    def mock_invoke_hook(hook_name, **kwargs):
        if hook_name == "post_llm_call":
            seen_responses.append(kwargs.get("assistant_response"))
        if hook_name == "transform_llm_output":
            return ["Transformed"]
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", mock_invoke_hook)

    _run(agent)

    assert seen_responses == ["Transformed"]


def test_hook_failure_does_not_break_the_turn(monkeypatch):
    agent = _StubAgent()

    def mock_invoke_hook(hook_name, **kwargs):
        if hook_name == "transform_llm_output":
            raise RuntimeError("plugin exploded")
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", mock_invoke_hook)

    result = _run(agent)

    assert result["final_response"] == "Original"
    assert result["response_transformed"] is False
    # The hook failure is contained (warned, not raised) and the turn's
    # cleanup chain reports no errors.
    assert not result.get("cleanup_errors")
