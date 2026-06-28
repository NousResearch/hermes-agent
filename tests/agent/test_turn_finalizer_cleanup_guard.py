"""Regression test for #8049.

When the post-loop cleanup chain in ``finalize_turn`` raises — trajectory
save (file I/O), resource teardown (remote VM/browser), or session
persistence (SQLite) — the partial ``final_response`` the caller is waiting
for must still be returned.  Previously any of those raised straight out of
``run_conversation``, so a subprocess wrapper saw an empty stdout with no
traceback and lost the whole turn.

Also includes tests for protocol violation auto-detection (#1000):
when a kanban worker exits with a text response without calling
kanban_complete or kanban_block, turn_finalizer must auto-call
_record_task_failure to prevent "protocol_violation" crash detection.
"""

import unittest.mock
from pathlib import Path

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
# Protocol violation auto-detection (#1000)
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    from hermes_cli import kanban_db as _kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _kb.init_db()
    return home


def test_protocol_violation_auto_blocked_on_text_response(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When a kanban worker exits with text_response (no tool call),
    turn_finalizer should auto-call _record_task_failure to prevent
    protocol_violation crash detection."""
    from hermes_cli import kanban_db as _kb

    # Create a running task in the isolated DB
    with _kb.connect_closing() as conn:
        task_id = _kb.create_task(
            conn, title="protocol-violation-test", assignee="worker"
        )
        with _kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        claimed = _kb.claim_task(conn, task_id, claimer="worker")
        assert claimed is not None

    # Set the env var so the finalizer knows it's a kanban worker
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)

    # Mock _record_task_failure so we can assert on call args without
    # actually modifying the (already isolated) DB
    mock_record = unittest.mock.MagicMock()
    monkeypatch.setattr(_kb, "_record_task_failure", mock_record)

    # Call finalize_turn with a normal text_response exit and NO
    # kanban_complete/kanban_block tool calls in messages
    agent = _StubAgent(raise_in=())
    result = _run(
        agent,
        final_response="Done.",
        api_call_count=1,
        turn_exit_reason="text_response(finish_reason=stop)",
    )

    # The turn should be marked completed
    assert result["completed"] is True

    # _record_task_failure should have been called with the right args
    mock_record.assert_called_once()
    call_args, call_kwargs = mock_record.call_args
    assert call_args[1] == task_id
    assert call_kwargs["outcome"] == "crashed"
    assert call_kwargs["failure_limit"] == 1
    assert call_kwargs["release_claim"] is True
    assert call_kwargs["end_run"] is True
    assert call_kwargs["event_payload_extra"]["auto_detected"] is True
    assert "exit_reason" in call_kwargs["event_payload_extra"]

    # A WARNING should have been logged
    assert "Protocol violation auto-detected for task" in caplog.text
    assert task_id in caplog.text


def test_protocol_violation_skipped_when_complete_called(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When a kanban worker exits with text_response but DID call
    kanban_complete/kanban_block, turn_finalizer should NOT auto-call
    _record_task_failure."""
    from hermes_cli import kanban_db as _kb

    # Create a running task
    with _kb.connect_closing() as conn:
        task_id = _kb.create_task(
            conn, title="protocol-violation-skip", assignee="worker"
        )
        with _kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        claimed = _kb.claim_task(conn, task_id, claimer="worker")
        assert claimed is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)

    mock_record = unittest.mock.MagicMock()
    monkeypatch.setattr(_kb, "_record_task_failure", mock_record)

    # Messages include a kanban_complete tool call (the worker DID complete)
    messages_with_complete = [
        {"role": "user", "content": "do a thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    agent = _StubAgent(raise_in=())
    result = finalize_turn(
        agent,
        final_response="Done.",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages_with_complete,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert result["completed"] is True

    # _record_task_failure should NOT have been called
    mock_record.assert_not_called()

    # No protocol violation warning
    assert "Protocol violation auto-detected" not in caplog.text


def test_protocol_violation_skipped_for_non_kanban(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When HERMES_KANBAN_TASK is not set, the auto-detection block
    should not fire."""
    # Ensure no kanban env
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    agent = _StubAgent(raise_in=())
    result = _run(
        agent,
        final_response="Done.",
        api_call_count=1,
        turn_exit_reason="text_response(finish_reason=stop)",
    )
    assert result["completed"] is True
    assert "Protocol violation auto-detected" not in caplog.text
