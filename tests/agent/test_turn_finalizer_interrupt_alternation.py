"""Regression test for #48879.

When a turn is interrupted via ``/stop`` right after a tool completes — but
before the assistant streams any final text — the transcript tail is a raw
``tool`` message. Persisting that tail unmodified means the next user message
lands as ``... tool → user``, a role-alternation violation that strict
providers (Gemini, Claude) react to by hallucinating a continuation of the
user's message before transitioning into the assistant persona.

``finalize_turn`` closes the tool-call sequence on interrupt by appending a
synthetic ``assistant`` message before persistence. ``final_response`` is
typically empty on an interrupt, so the placeholder text is used rather than
an empty-content assistant turn.
"""

import pytest

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

    def _safe_print(self, *a, **k):
        pass

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


def test_interrupt_after_tool_closes_sequence_with_placeholder():
    agent = _StubAgent()
    messages = _interrupted_tool_tail()
    _finalize(agent, messages, interrupted=True, final_response=None)

    # Tail must now be an assistant message, not a raw tool result.
    assert messages[-1]["role"] == "assistant"
    # Empty final_response falls back to the explicit placeholder rather
    # than persisting an empty-content assistant turn.
    assert messages[-1]["content"] == "Operation interrupted."

    # The persisted snapshot is alternation-safe: appending a new user
    # message would follow an assistant, not an orphan tool.
    assert agent.persisted_messages is not None
    assert agent.persisted_messages[-1]["role"] == "assistant"
    follow_on = agent.persisted_messages + [{"role": "user", "content": "forget it"}]
    _assert_no_tool_then_user(follow_on)


def test_interrupt_after_tool_keeps_delivered_text_when_present():
    agent = _StubAgent()
    messages = _interrupted_tool_tail()
    _finalize(agent, messages, interrupted=True, final_response="Partial answer so far")

    assert messages[-1]["role"] == "assistant"
    # Real delivered text is preserved, not clobbered by the placeholder.
    assert messages[-1]["content"] == "Partial answer so far"


def test_non_interrupted_tool_tail_is_left_untouched():
    # A turn that ends on a tool tail WITHOUT an interrupt (mid-progress
    # tool loop) must not get a synthetic close — that is normal dialog
    # state handled elsewhere.
    agent = _StubAgent()
    messages = _interrupted_tool_tail()
    _finalize(agent, messages, interrupted=False, final_response=None)
    assert messages[-1]["role"] == "tool"


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
