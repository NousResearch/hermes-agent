"""Interrupted history retains actual events; provider projection owns role bridges.

Non-interrupted failure/recovery responses remain visible and durable.
"""

from copy import deepcopy

import pytest

from hermes_state import SessionDB
from agent.turn_recovery import abort_turn_on_interrupt

from agent.turn_finalizer import finalize_turn


class _StubBudget:
    used = 1
    max_total = 90
    remaining = 89


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface that ``finalize_turn`` reads from."""

    def __init__(self):
        self.max_iterations = 90
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.log_prefix = ""
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
        self.persisted_messages = None
        # #95514 stream-recovery state read by the finalizer; None on a clean stub.
        self._current_streamed_assistant_text: str | None = None

    # --- fallible cleanup surfaces (all succeed here) ------------------
    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        # A clean interrupt sets no empty-response scaffolding flags, so
        # the real method returns early and leaves the tool tail in place.
        # Model that here as a no-op.
        pass

    def _persist_session(self, messages, conversation_history):
        # Snapshot the role sequence at the moment of persistence.
        self.persisted_messages = [dict(m) for m in messages]

    # --- harmless no-ops ------------------------------------------------
    def _emit_status(self, *a, **k):
        pass

    def _vprint(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self, **kwargs):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


def _interrupted_tool_tail():
    """A transcript interrupted after a successful tool, before any
    assistant text — the exact #48879 shape."""
    return [
        {"role": "user", "content": "edit the file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "patch", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok edited"},
    ]


def _finalize(agent, messages, *, interrupted, final_response=None):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=1,
        interrupted=interrupted,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="edit the file",
        original_user_message="edit the file",
        _should_review_memory=False,
        _turn_exit_reason="interrupted_by_user",
    )


def _assert_no_tool_then_user(messages):
    for i in range(len(messages) - 1):
        if messages[i].get("role") == "tool":
            assert messages[i + 1].get("role") != "user", (
                f"role-alternation violation: tool → user at index {i}"
            )


@pytest.mark.parametrize("exit_path", ["finalizer", "recovery"])
@pytest.mark.parametrize("has_partial", [False, True])
def test_interruption_preserves_actual_history_after_reload(tmp_path, exit_path, has_partial):
    agent = _StubAgent()
    messages = _interrupted_tool_tail()
    if has_partial:
        messages.append({"role": "assistant", "content": "Partial answer", "display_metadata": {"interrupted": True}})
    original = deepcopy(messages)
    db_path = tmp_path / "state.db"
    with SessionDB(db_path=db_path) as db:
        db.create_session(agent.session_id, source="cli")
        agent._persist_session = lambda rows, _history: db.replace_messages(agent.session_id, rows)
        if exit_path == "finalizer":
            result = _finalize(agent, messages, interrupted=True, final_response="Operation interrupted.")
        else:
            result = abort_turn_on_interrupt(
                agent, messages, None, 1, abort_message="Stopped", interrupt_text="Operation interrupted.",
            )
    with SessionDB(db_path=db_path) as reopened:
        saved = reopened.get_messages_as_conversation(agent.session_id)
    assert result["interrupted"] is True
    assert result["completed"] is False
    assert [(row["role"], row["content"]) for row in messages] == [
        (row["role"], row["content"]) for row in original
    ]
    # Neither the durable transcript nor the next user turn acquires a fabricated answer.
    assert [(row["role"], row["content"]) for row in saved] == [
        (row["role"], row["content"]) for row in original
    ]
    assert saved[1]["tool_calls"] == original[1]["tool_calls"]
    assert saved[2]["tool_call_id"] == original[2]["tool_call_id"]


def test_interrupt_without_tool_tail_adds_nothing():
    # Interrupt while the tail is already an assistant/user message: no
    # synthetic close needed.
    agent = _StubAgent()
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "partial reply"},
    ]
    before = len(messages)
    _finalize(agent, messages, interrupted=True, final_response="partial reply")
    assert len(messages) == before
    assert messages[-1]["role"] == "assistant"


def test_interrupted_turn_with_diagnostic_text_is_not_completed():
    """An interrupt mid-call leaves a diagnostic ``final_response`` ("Operation interrupted:
    waiting for model response"); the result must still say ``completed=False`` like the
    sibling producers (turn_recovery, codex_runtime) — the gateway stream gate and the API run
    status trust that flag (#111770)."""
    agent = _StubAgent()
    result = _finalize(
        agent, [{"role": "user", "content": "hi"}], interrupted=True,
        final_response="Operation interrupted: waiting for model response (0.1s elapsed).",
    )
    assert result["interrupted"] is True
    assert result["completed"] is False
    assert result["failed"] is False


def _pending_tool_result_tail():
    """A non-interrupted turn that fell out of the loop after a tool result, with no
    follow-up assistant text — the #55316/#54756 "silent stop" shape."""
    return [
        {"role": "user", "content": "summarize the log"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "log output"},
    ]


def test_non_interrupted_tool_tail_gets_visible_close():
    """A turn that stops on a tool tail WITHOUT an interrupt must not return a silent,
    ready-looking result: the finalizer fails the turn, mints the ``pending_tool_result``
    exit reason, and persists a visible assistant close so the durable transcript does
    not end at a raw ``tool`` row (#55316, #54756)."""
    agent = _StubAgent()
    messages = _pending_tool_result_tail()
    result = _finalize(agent, messages, interrupted=False, final_response=None)

    assert result["turn_exit_reason"] == "pending_tool_result"
    assert result["failed"] is True
    assert result["completed"] is False
    assert result["final_response"].strip()
    # The durable tail is an assistant row, not the raw tool result.
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"].strip()
    assert agent.persisted_messages is not None
    assert agent.persisted_messages[-1]["role"] == "assistant"
    follow_on = agent.persisted_messages + [{"role": "user", "content": "continue"}]
    _assert_no_tool_then_user(follow_on)


def test_tool_tail_with_streamed_text_is_recovered_not_marked_pending():
    """A turn whose stream already delivered text is #95514's stream-recovery case; the
    ``pending_tool_result`` close must not fire over it."""
    agent = _StubAgent()
    agent._current_streamed_assistant_text = "Here is the summary you asked for."
    messages = _pending_tool_result_tail()
    result = _finalize(agent, messages, interrupted=False, final_response=None)

    assert result["turn_exit_reason"] != "pending_tool_result"
    assert result["final_response"] == "Here is the summary you asked for."
    assert messages[-1]["role"] == "assistant"


def test_tool_tail_with_non_tool_last_role_is_untouched():
    """Only the tool-tail shape triggers the close; a plain user-tail turn keeps its
    existing exit reason."""
    agent = _StubAgent()
    result = _finalize(
        agent, [{"role": "user", "content": "hi"}], interrupted=False, final_response=None,
    )
    assert result["turn_exit_reason"] == "interrupted_by_user"  # unchanged passthrough
    assert result["failed"] is False
