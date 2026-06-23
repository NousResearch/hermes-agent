"""Regression test for #8049.

When the post-loop cleanup chain in ``finalize_turn`` raises — trajectory
save (file I/O), resource teardown (remote VM/browser), or session
persistence (SQLite) — the partial ``final_response`` the caller is waiting
for must still be returned.  Previously any of those raised straight out of
``run_conversation``, so a subprocess wrapper saw an empty stdout with no
traceback and lost the whole turn.
"""

import pytest

from agent.turn_finalizer import finalize_turn


class _StubBudget:
    used = 5
    max_total = 3
    remaining = 0


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface that ``finalize_turn`` reads from."""

    def __init__(self, *, raise_in):
        self._raise_in = set(raise_in)
        self.max_iterations = 3
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

    # --- fallible cleanup surfaces -------------------------------------
    def _save_trajectory(self, *a, **k):
        if "save_trajectory" in self._raise_in:
            raise RuntimeError("trajectory disk full")

    def _cleanup_task_resources(self, *a, **k):
        if "cleanup_task_resources" in self._raise_in:
            raise RuntimeError("docker teardown EOF")

    def _drop_trailing_empty_response_scaffolding(self, *a, **k):
        pass

    def _persist_session(self, *a, **k):
        if "persist_session" in self._raise_in:
            raise RuntimeError("sqlite database is locked")

    # --- harmless no-ops ------------------------------------------------
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


def _run(
    agent,
    *,
    final_response=None,
    api_call_count=3,
    turn_exit_reason="unknown",
):
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
        api_call_count=api_call_count,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason=turn_exit_reason,
    )


def test_all_cleanup_steps_raise_response_still_returned():
    agent = _StubAgent(
        raise_in=("save_trajectory", "cleanup_task_resources", "persist_session")
    )
    result = _run(agent)
    assert result["final_response"] == "PARTIAL SUMMARY FROM MODEL"
    labels = [e.split(":")[0] for e in result["cleanup_errors"]]
    assert labels == ["save_trajectory", "cleanup_task_resources", "persist_session"]


@pytest.mark.parametrize(
    "step", ["save_trajectory", "cleanup_task_resources", "persist_session"]
)
def test_single_cleanup_step_raises_does_not_skip_others(step):
    agent = _StubAgent(raise_in=(step,))
    result = _run(agent)
    # Response survives.
    assert result["final_response"] == "PARTIAL SUMMARY FROM MODEL"
    # Exactly the failing step is recorded; the others ran without error.
    assert result["cleanup_errors"] == [
        next(
            e
            for e in result["cleanup_errors"]
            if e.startswith(step)
        )
    ]
    assert len(result["cleanup_errors"]) == 1


def test_clean_turn_has_no_cleanup_errors_key():
    agent = _StubAgent(raise_in=())
    result = _run(agent)
    assert result["final_response"] == "PARTIAL SUMMARY FROM MODEL"
    assert result["completed"] is False
    assert "cleanup_errors" not in result


def test_text_response_on_last_allowed_call_is_completed():
    agent = _StubAgent(raise_in=())
    result = _run(
        agent,
        final_response="final report",
        api_call_count=agent.max_iterations,
        turn_exit_reason="text_response(finish_reason=stop)",
    )
    assert result["final_response"] == "final report"
    assert result["completed"] is True


# ---------------------------------------------------------------------------
# Auto session summary hook
# ---------------------------------------------------------------------------

class _StubAgentWithSummary(_StubAgent):
    """Stub agent with auto_session_summary enabled."""

    def __init__(self, session_id, *, raise_in=()):
        super().__init__(raise_in=raise_in)
        self.session_id = session_id

    def _auto_session_summary_enabled(self):
        return True


def test_auto_summary_appends_after_completed_turn(tmp_path, monkeypatch):
    """After a completed turn, auto-summary appends a timestamped entry."""
    monkeypatch.setattr("tools.session_summary_tool.get_hermes_home", lambda: tmp_path / "hermes_test")

    session_id = "test-session"
    agent = _StubAgentWithSummary(session_id)

    result = _run(
        agent,
        final_response="All tests pass.",
        api_call_count=3,
        turn_exit_reason="text_response(finish_reason=stop)",
    )
    assert result["completed"] is True

    summary_path = tmp_path / "hermes_test" / "sessions" / session_id / "running_summary.md"
    assert summary_path.exists(), f"Expected summary at {summary_path}"
    text = summary_path.read_text()
    assert "All tests pass" in text or "3 API calls" in text or "tool" in text.lower()


def test_auto_summary_skipped_when_interrupted(tmp_path, monkeypatch):
    """Auto-summary does NOT append when the turn was interrupted."""
    monkeypatch.setattr("tools.session_summary_tool.get_hermes_home", lambda: tmp_path / "hermes_test")

    session_id = "test-session"
    agent = _StubAgentWithSummary(session_id)

    result = finalize_turn(
        agent,
        final_response="partial work",
        api_call_count=2,
        interrupted=True,
        failed=False,
        messages=[{"role": "user", "content": "do a thing"}],
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason="interrupted_by_user",
    )
    assert result["interrupted"] is True

    summary_path = tmp_path / "hermes_test" / "sessions" / session_id / "running_summary.md"
    # Should NOT exist — interrupted turns don't get auto-summary
    assert not summary_path.exists(), "Interrupted turn should not create summary"


def test_auto_summary_skipped_when_disabled(tmp_path, monkeypatch):
    """Auto-summary does NOT append when config is disabled."""
    monkeypatch.setattr("tools.session_summary_tool.get_hermes_home", lambda: tmp_path / "hermes_test")

    session_id = "test-session"
    agent = _StubAgentWithSummary(session_id)
    # Override to disable
    agent._auto_session_summary_enabled = lambda: False  # type: ignore[assignment]

    result = _run(
        agent,
        final_response="All tests pass.",
        api_call_count=3,
        turn_exit_reason="text_response(finish_reason=stop)",
    )
    assert result["completed"] is True

    summary_path = tmp_path / "hermes_test" / "sessions" / session_id / "running_summary.md"
    assert not summary_path.exists(), "Disabled auto-summary should not create file"


def test_auto_summary_failure_does_not_affect_result(tmp_path, monkeypatch):
    """If the summary write fails, the turn result is still returned cleanly."""
    monkeypatch.setattr("tools.session_summary_tool.get_hermes_home", lambda: tmp_path / "hermes_test")

    session_id = "test-session"
    agent = _StubAgentWithSummary(session_id)

    # Make the sessions dir read-only so the write fails
    sessions_dir = tmp_path / "hermes_test" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.chmod(0o444)  # read-only

    result = _run(
        agent,
        final_response="All tests pass.",
        api_call_count=3,
        turn_exit_reason="text_response(finish_reason=stop)",
    )
    # Turn still completes normally
    assert result["completed"] is True
    assert result["final_response"] == "All tests pass."

    # Restore permissions for cleanup
    sessions_dir.chmod(0o755)
