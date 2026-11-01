"""Interrupted-turn reasoning persistence (#134946).

Reasoning streamed before an interrupt was shown live but never attached to the persisted
row: every interrupted turn in the store lost it. The accumulator mirrors
``_current_streamed_assistant_text`` and both interrupt paths (plain interrupt, user
redirect) attach it to the row's ``reasoning_content`` — never to ``content`` /
``api_content``, which the wire replays byte-for-byte."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from agent.agent_runtime_helpers_placeholders import (
    _INTERRUPTED_PLACEHOLDER,
    hidden_interrupt_placeholder_row,
)
from agent.stream_delivery import StreamDeliveryMixin
from agent.turn_api_call import handle_api_interrupt
from agent.turn_retry_state import TurnRetryState
from run_agent import AIAgent


class _DeliveryStub(StreamDeliveryMixin):
    """Minimal host for the delivery mixin: hook enqueue is pre-disabled."""

    def __init__(self):
        self.reasoning_callback = None
        self._stream_reasoning_hooks_enabled = False


def _bare_agent(streamed: str = "", reasoning: str = "") -> AIAgent:
    agent = object.__new__(AIAgent)
    agent._pending_redirect = None
    agent._pending_redirect_lock = threading.Lock()
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._current_streamed_assistant_text = streamed
    agent._current_streamed_assistant_reasoning = reasoning
    agent._strip_think_blocks = lambda content: content
    agent.quiet_mode = True
    agent.log_prefix = ""
    agent.thinking_callback = None
    agent._print_fn = lambda *args, **kwargs: None
    agent._persist_session = lambda *args, **kwargs: None
    return agent


def _interrupt(streamed: str = "", reasoning: str = ""):
    messages = [{"role": "user", "content": "start"}]
    verdict = handle_api_interrupt(
        _bare_agent(streamed, reasoning), _retry=TurnRetryState(), thinking_spinner=None,
        messages=messages, conversation_history=[], api_start_time=time.time(),
        interrupted=False, final_response=None,
    )
    return messages, verdict


# ── the accumulator ───────────────────────────────────────────────────────────


def test_streamed_reasoning_accumulates_and_resets():
    stub = _DeliveryStub()
    stub._fire_reasoning_delta("Let me ")
    stub._fire_reasoning_delta("check the config.", inline=True)

    assert stub._current_streamed_assistant_reasoning == "Let me check the config."

    stub._reset_stream_delivery_tracking()
    assert stub._current_streamed_assistant_reasoning == ""


def test_superseded_writer_reasoning_is_not_accumulated():
    stub = _DeliveryStub()
    stub._current_streamed_assistant_reasoning = "live writer's reasoning"

    # A stale writer token is fenced exactly like content deltas (#65991).
    stub._stream_writer_tls = SimpleNamespace(token=1)
    stub._stream_writer_token = 2
    stub._fire_reasoning_delta("retry interleaved")

    assert stub._current_streamed_assistant_reasoning == "live writer's reasoning"


# ── the plain interrupt row ───────────────────────────────────────────────────


def test_interrupt_with_partial_keeps_reasoning_on_row():
    messages, _ = _interrupt(streamed="Visible draft.", reasoning="I was inspecting the entry point.")

    row = messages[-1]
    assert (row["role"], row["content"]) == ("assistant", "Visible draft.")
    assert row["reasoning_content"] == "I was inspecting the entry point."


def test_reasoning_only_interrupt_persists_a_hidden_row():
    """A model interrupted mid-thought with no visible text left no trace at all; the
    hidden placeholder row keeps the reasoning recoverable after a reload."""
    messages, verdict = _interrupt(reasoning="Half a plan: first…")

    row = messages[-1]
    assert row["role"] == "assistant"
    assert row["display_kind"] == "hidden"
    assert row["api_content"] == _INTERRUPTED_PLACEHOLDER
    assert row["reasoning_content"] == "Half a plan: first…"
    assert verdict.final_response.startswith("Operation interrupted: waiting for model response")


def test_runaway_interrupt_does_not_persist_reasoning():
    """Runaway bytes must not be re-seeded through the reasoning channel either (#112764)."""
    looped_reasoning = "I. " * 1941

    messages, _ = _interrupt(streamed="I. " * 1941, reasoning=looped_reasoning)

    assert messages[-1]["api_content"] == _INTERRUPTED_PLACEHOLDER
    assert "reasoning_content" not in messages[-1]


def test_empty_interrupt_appends_no_row():
    messages, _ = _interrupt()

    assert [m["role"] for m in messages] == ["user"]


# ── the redirect (user correction) row ────────────────────────────────────────


def _redirect(streamed: str = "", reasoning: str = "", messages=None):
    from agent.conversation_loop import _apply_active_turn_redirect

    agent = _bare_agent(streamed, reasoning)
    agent._stream_needs_break = False
    messages = messages if messages is not None else [{"role": "user", "content": "start"}]
    _apply_active_turn_redirect(agent, messages, "Change course.")
    return agent, messages


def test_redirect_hidden_row_carries_reasoning():
    _, messages = _redirect(reasoning="Interrupted mid-thought.")

    placeholder = messages[-2]
    assert placeholder["display_kind"] == "hidden"
    assert placeholder["api_content"] == _INTERRUPTED_PLACEHOLDER
    assert placeholder["reasoning_content"] == "Interrupted mid-thought."
    # Scaffold/replay channels stay untouched: reasoning rides its own field only.
    assert placeholder["content"] == ""
    assert "Interrupted mid-thought." not in placeholder["api_content"]
    assert "Interrupted mid-thought." not in messages[-1]["api_content"]


def test_redirect_visible_row_carries_reasoning_and_clears_accumulators():
    agent, messages = _redirect(
        streamed="Visible draft.", reasoning="Reasoning that preceded the draft."
    )

    row = messages[-2]
    assert (row["role"], row["content"]) == ("assistant", "Visible draft.")
    assert row["reasoning_content"] == "Reasoning that preceded the draft."
    # The drained row must not leak into the rebuilt request's own interrupt row.
    assert agent._current_streamed_assistant_reasoning == ""
    assert agent._current_streamed_assistant_text == ""


def test_redirect_runaway_drops_reasoning():
    _, messages = _redirect(streamed="I. " * 1941, reasoning="I. " * 1941)

    placeholder = messages[-2]
    assert placeholder["display_kind"] == "hidden"
    assert "reasoning_content" not in placeholder


def test_placeholder_row_without_reasoning_keeps_legacy_shape():
    assert hidden_interrupt_placeholder_row() == {
        "role": "assistant", "content": "", "display_kind": "hidden",
        "api_content": _INTERRUPTED_PLACEHOLDER,
    }
