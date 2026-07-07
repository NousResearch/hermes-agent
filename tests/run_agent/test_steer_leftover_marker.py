"""Regression / reproduction for #60543.

`/steer` has three delivery paths.  Two of them wrap the user's message in the
`[OUT-OF-BAND USER MESSAGE]` marker so the model trusts it as a genuine user
instruction rather than treating it as tool output / prompt injection:

  * mid-batch drain  -> agent_runtime_helpers.apply_pending_steer_to_tool_results
                        (agent_runtime_helpers.py:3109, calls format_steer_marker)
  * pre-API drain    -> conversation_loop.py:718 (calls format_steer_marker)

The THIRD path — the "leftover" path, taken when a steer lands *after* the
final assistant turn (no more tool batches to drain into) — does NOT:

  * turn_finalizer.py:439-441 stashes the RAW steer text into
    result["pending_steer"] (via _drain_pending_steer(), unmarked — correct,
    marking is the delivery layer's job), and then
  * cli.py:12658-12662 delivers it with `self._pending_input.put(_leftover_steer)`
    — the raw text, WITHOUT format_steer_marker().

Net effect: a steer that arrives in that timing window reaches the model as
bare `/steer` command text on the next turn instead of an out-of-band user
message — the bug reported in #60543.

These tests reproduce the divergence.  `test_leftover_delivery_carries_oob_marker`
is expected to FAIL until the leftover handler wraps the text with
format_steer_marker().
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from agent.prompt_builder import STEER_MARKER_OPEN, format_steer_marker
from agent.turn_finalizer import finalize_turn
from run_agent import AIAgent

STEER_TEXT = "actually, stop and summarize what you have so far"


def _bare_agent() -> AIAgent:
    """AIAgent without __init__ + manual steer state — matches test_steer.py."""
    agent = object.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    return agent


class _FinalizeAgent:
    """Minimal agent surface that finalize_turn reads from — mirrors the stub
    in tests/agent/test_turn_finalizer_final_response_persistence.py, but with
    a pending steer that lands after the final assistant turn (issue #60543).
    """

    def __init__(self, steer_text: str):
        self.max_iterations = 90
        self.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=90)
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "sess-test"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = True
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.persisted_messages = None
        self._steer_slot = steer_text

    def _handle_max_iterations(self, messages, api_call_count):
        raise AssertionError("not expected")

    def _emit_status(self, *_a, **_kw):
        pass

    def _safe_print(self, *_a, **_kw):
        pass

    def _save_trajectory(self, *_a, **_kw):
        pass

    def _cleanup_task_resources(self, *_a, **_kw):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = list(messages)

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        # A steer that arrived after the final assistant turn — the leftover
        # scenario. Drains once (like the real lock-guarded slot).
        text, self._steer_slot = self._steer_slot, None
        return text

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kw):
        pass


def _run_finalize(agent) -> dict:
    return finalize_turn(
        agent,
        final_response="Here is what I found.",
        api_call_count=2,
        interrupted=False,
        failed=False,
        messages=[
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "content": "on it",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "terminal", "content": "ok"},
        ],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="do it",
        original_user_message="do it",
        _should_review_memory=False,
        _turn_exit_reason="done",
    )


@pytest.fixture(autouse=True)
def _no_plugin_hooks(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])


def test_sibling_drain_path_marks_the_steer():
    """Baseline: the mid-batch path DOES wrap the steer in the OOB marker."""
    agent = _bare_agent()
    agent.steer(STEER_TEXT)
    messages = [{"role": "tool", "content": "output", "tool_call_id": "c1"}]
    agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=1)
    assert STEER_MARKER_OPEN in messages[-1]["content"]
    assert STEER_TEXT in messages[-1]["content"]


def test_finalize_turn_hands_back_raw_unmarked_steer():
    """finalize_turn stashes the RAW steer (marking is the delivery layer's
    job). This is the value cli.py's leftover handler receives."""
    agent = _FinalizeAgent(STEER_TEXT)
    result = _run_finalize(agent)
    assert result.get("pending_steer") == STEER_TEXT
    assert STEER_MARKER_OPEN not in result["pending_steer"]


def test_leftover_delivery_carries_oob_marker():
    """#60543: the leftover steer, once DELIVERED to the input queue (and thus
    to the model on the next turn), must carry the [OUT-OF-BAND USER MESSAGE]
    marker — exactly like the other two delivery paths.

    Mirrors the real CLI handler at cli.py:12658-12662, which delivers the
    leftover steer via the shared ``format_leftover_steer_for_delivery`` seam:

        _leftover_steer = result.get("pending_steer") if result else None
        if _leftover_steer and hasattr(self, '_pending_input'):
            from agent.agent_runtime_helpers import format_leftover_steer_for_delivery
            ...
            self._pending_input.put(format_leftover_steer_for_delivery(_leftover_steer))
    """
    import queue

    from agent.agent_runtime_helpers import format_leftover_steer_for_delivery

    agent = _FinalizeAgent(STEER_TEXT)
    result = _run_finalize(agent)

    # --- mirror of the real cli.py leftover-steer delivery block ---
    pending_input: "queue.Queue[str]" = queue.Queue()
    _leftover_steer = result.get("pending_steer") if result else None
    if _leftover_steer:
        pending_input.put(format_leftover_steer_for_delivery(_leftover_steer))
    # ---------------------------------------------------------------

    delivered = pending_input.get_nowait()
    assert STEER_TEXT in delivered
    assert STEER_MARKER_OPEN in delivered, (
        "leftover /steer reached the model as raw command text, not as an "
        "out-of-band user message (#60543)"
    )
    # Clean standalone turn — no leading blank lines from the append-oriented
    # marker helper.
    assert not delivered.startswith("\n")


def test_delivery_helper_matches_marker_contract():
    """The leftover-delivery helper must emit the same open/close marker the
    system prompt tells the model to trust (see STEER_CHANNEL_NOTE)."""
    from agent.agent_runtime_helpers import format_leftover_steer_for_delivery
    from agent.prompt_builder import STEER_MARKER_CLOSE

    out = format_leftover_steer_for_delivery(STEER_TEXT)
    assert STEER_MARKER_OPEN in out
    assert STEER_MARKER_CLOSE in out
    assert STEER_TEXT in out
    assert out == format_steer_marker(STEER_TEXT).strip()


def _gateway_deliver_leftover(result):
    """Mirror of the gateway leftover-steer handler (gateway/run.py:19419-19449):
    the delivery line followed by the slash-command discard safety net. Returns
    the ``pending`` value the gateway would hand to the agent as the next turn.
    """
    from agent.agent_runtime_helpers import format_leftover_steer_for_delivery
    from hermes_cli.commands import resolve_command

    pending = None
    pending_event = None
    if result and not pending and not pending_event:
        _leftover_steer = result.get("pending_steer")
        if _leftover_steer:
            pending = format_leftover_steer_for_delivery(_leftover_steer)

    # Safety net: gateway discards pending text that is a slash command.
    if pending and pending.strip().startswith("/"):
        parts = pending.strip().split(None, 1)
        cmd_word = parts[0][1:].lower() if parts else ""
        if cmd_word and resolve_command(cmd_word):
            pending = None
    return pending


def test_gateway_leftover_delivery_carries_oob_marker():
    """#60543 (TUI/gateway surface): the gateway serves the TUI, Telegram,
    Slack and WhatsApp. Its leftover-steer handler must also deliver the steer
    wrapped in the [OUT-OF-BAND USER MESSAGE] marker, not raw text."""
    agent = _FinalizeAgent(STEER_TEXT)
    result = _run_finalize(agent)

    pending = _gateway_deliver_leftover(result)

    assert pending is not None
    assert STEER_TEXT in pending
    assert STEER_MARKER_OPEN in pending, (
        "gateway delivered leftover /steer as raw command text, not as an "
        "out-of-band user message (#60543)"
    )


def test_gateway_marked_steer_survives_slash_command_net():
    """The marker must keep the steer from being swallowed by the gateway's
    slash-command discard net — even when the steer text itself looks like a
    command (e.g. a user steering with '/stop the search'). Wrapped, it no
    longer starts with '/', so it is delivered rather than silently dropped."""
    agent = _FinalizeAgent("/stop the search and summarize")
    result = _run_finalize(agent)

    pending = _gateway_deliver_leftover(result)

    assert pending is not None, "marked steer was wrongly discarded by the slash-command net"
    assert STEER_MARKER_OPEN in pending
    assert "/stop the search and summarize" in pending


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
