"""Regression tests for #102806 (defect 1): a tool result must never become the
user-visible reply without the model composing it.

The reporter's fork registers ``archive_search`` with ``direct_result=True`` in an
out-of-tree plugin (``plugins/archive/__init__.py``), so the raw FTS output
(``- [27] … | - [17] …``) is promoted to the turn's final response and forwarded to
the platform verbatim — the user sees stale search hits from other sessions instead
of the model's "Saved and verified.".

``direct_result`` is not a core symbol (no consumer exists in origin/main — see the
PR body), so the fix belongs at the single chokepoint every turn flows through:
``agent.turn_finalizer.finalize_turn``. Whatever a caller (a fork, a plugin harness,
a future delivery path) puts in ``final_response``, the finalizer must refuse to
deliver a reply that is verbatim a tool row's output with no model-composed text
after it.

Two failure shapes, both driven through the real ``finalize_turn``:

1. a stray ``archive_search`` right after a successful ``archive_save`` (the save
   already produced the user-facing receipt, so the search output is pure noise);
2. a bare ``archive_search`` with no subsequent model turn at all.
"""

from agent.turn_finalizer import finalize_turn

# Raw FTS output as it reached the reporter's Telegram DM.
RAW_FTS = "- [27] 2026-08-19 deploy checklist | - [17] 2026-08-24 milk and eggs"
RECEIPT = "Saved and verified."


class _StubBudget:
    used = 1
    max_total = 10
    remaining = 9


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface ``finalize_turn`` reads from."""

    def __init__(self, *, explainer=True):
        self.max_iterations = 10
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-102806"
        self.quiet_mode = True
        self.platform = "telegram"
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self._explainer = explainer
        for attr in (
            "session_input_tokens", "session_output_tokens", "session_cache_read_tokens",
            "session_cache_write_tokens", "session_reasoning_tokens", "session_prompt_tokens",
            "session_completion_tokens", "session_total_tokens", "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, *a, **k):
        pass

    def _persist_session(self, *a, **k):
        pass

    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _handle_max_iterations(self, messages, n):
        return "SUMMARY FROM MODEL"

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return self._explainer

    def _format_turn_completion_explanation(self, turn_exit_reason, persistence_cause=None):
        from agent.turn_explainers import TurnExplainersMixin
        return TurnExplainersMixin._format_turn_completion_explanation(
            turn_exit_reason, persistence_cause
        )

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


def _call(agent, messages, *, final_response, turn_exit_reason="text_response(stop)"):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-102806",
        turn_id="turn-102806",
        user_message="save https://example.com/post",
        original_user_message="save https://example.com/post",
        _should_review_memory=False,
        _turn_exit_reason=turn_exit_reason,
    )


def _search_call(call_id):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "function": {"name": "archive_search", "arguments": "{}"}}],
    }


# --------------------------------------------------------------------------
# Defect 1, failure shape 1: stray search immediately after a successful save
# --------------------------------------------------------------------------

def test_stray_search_after_save_does_not_become_the_reply():
    messages = [
        {"role": "user", "content": "save https://example.com/post"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "s1", "function": {"name": "archive_save", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "s1", "content": RECEIPT},
        _search_call("q1"),
        {"role": "tool", "tool_call_id": "q1", "content": RAW_FTS},
    ]
    result = _call(_StubAgent(), messages, final_response=RAW_FTS)
    assert RAW_FTS not in result["final_response"]
    assert "[27]" not in result["final_response"]


# --------------------------------------------------------------------------
# Defect 1, failure shape 2: bare search, no follow-up model turn
# --------------------------------------------------------------------------

def test_bare_search_with_no_model_turn_does_not_become_the_reply():
    messages = [
        {"role": "user", "content": "search the archive"},
        _search_call("q1"),
        {"role": "tool", "tool_call_id": "q1", "content": RAW_FTS},
    ]
    result = _call(_StubAgent(), messages, final_response=RAW_FTS)
    assert RAW_FTS not in result["final_response"]
    assert "[27]" not in result["final_response"]


def test_withheld_raw_tool_output_is_explained_not_silent():
    messages = [
        {"role": "user", "content": "search the archive"},
        _search_call("q1"),
        {"role": "tool", "tool_call_id": "q1", "content": RAW_FTS},
    ]
    result = _call(_StubAgent(), messages, final_response=RAW_FTS)
    assert result["final_response"].startswith("⚠️ No reply:")
    assert result["turn_exit_reason"] == "uncomposed_tool_output"


def test_withheld_raw_tool_output_fails_closed_when_explainer_disabled():
    """No reply beats raw junk from another session when the explainer is off."""
    messages = [
        {"role": "user", "content": "search the archive"},
        _search_call("q1"),
        {"role": "tool", "tool_call_id": "q1", "content": RAW_FTS},
    ]
    result = _call(_StubAgent(explainer=False), messages, final_response=RAW_FTS)
    assert result["final_response"] == ""


# --------------------------------------------------------------------------
# Bidirectional: the guard must not touch genuinely model-composed replies
# --------------------------------------------------------------------------

def test_model_composed_reply_is_delivered_unchanged():
    messages = [
        {"role": "user", "content": "save https://example.com/post"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "s1", "function": {"name": "archive_save", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "s1", "content": RECEIPT},
        {"role": "assistant", "content": RECEIPT},
    ]
    result = _call(_StubAgent(), messages, final_response=RECEIPT)
    assert result["final_response"] == RECEIPT


def test_composed_reply_that_quotes_tool_output_is_untouched():
    quoted = f"I searched the archive. Closest hit: {RAW_FTS} — want me to open it?"
    messages = [
        {"role": "user", "content": "search the archive"},
        _search_call("q1"),
        {"role": "tool", "tool_call_id": "q1", "content": RAW_FTS},
        {"role": "assistant", "content": quoted},
    ]
    result = _call(_StubAgent(), messages, final_response=quoted)
    assert result["final_response"] == quoted


def test_plain_chat_reply_without_tools_is_untouched():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    result = _call(_StubAgent(), messages, final_response="Hello!")
    assert result["final_response"] == "Hello!"


def test_tool_output_text_mentioned_in_an_earlier_turn_is_not_matched():
    """A tool row from a PRIOR turn must not suppress this turn's reply."""
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "q0", "function": {"name": "archive_search", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "q0", "content": RAW_FTS},
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": RAW_FTS},
    ]
    result = _call(_StubAgent(), messages, final_response=RAW_FTS)
    assert result["final_response"] == RAW_FTS
