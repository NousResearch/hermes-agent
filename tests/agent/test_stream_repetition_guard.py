"""The streaming repetition guard: a loop is cut while it streams, not after it finishes.

Incident (HYPEST box, 2026-09-27): the model degenerated into a ~160-char cycle three times in
one Telegram thread. Every existing repetition checkpoint reads the streamed text only when the
turn ENDS (interrupt, truncation, final response), so each loop ran until a human sent /stop:
78,641 and 29,951 visible chars. The checks below pin that a runaway loop interrupts itself
during accumulation, and that ordinary long or mildly repetitive replies never trip it.
"""

from __future__ import annotations

from unittest.mock import patch

from agent.repetition_guard import (
    STREAM_GUARD_FIRST_CHECK_CHARS,
    STREAM_LOOP_INTERRUPT_REASON,
)
from run_agent import AIAgent

# The exact cycle from the incident transcript.
_INCIDENT_CYCLE = (
    "De bot en de werkers blijven lopen. Ik zoek het punt waar de tekst binnenkomt. "
    "Daar stop ik de herhaling. De bot en de werkers blijven lopen. Ik zoek dat punt nu. "
)


def _make_agent():
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


def _stream(agent, text: str, chunk: int = 37) -> None:
    # Like the real stream reader: stops feeding once the turn is interrupted.
    for i in range(0, len(text), chunk):
        if getattr(agent, "_interrupt_requested", False):
            return
        agent._record_streamed_assistant_text(text[i : i + chunk])


class TestStreamingLoopIsCutMidStream:
    def test_incident_cycle_interrupts_before_second_threshold(self):
        agent = _make_agent()
        _stream(agent, _INCIDENT_CYCLE * 400)  # ~65k chars if nothing stops it
        assert agent._interrupt_requested is True
        assert agent._tool_interrupt_reason == STREAM_LOOP_INTERRUPT_REASON
        # Stopped at the first threshold, not after the whole loop was accumulated.
        assert len(agent._current_streamed_assistant_text) < STREAM_GUARD_FIRST_CHECK_CHARS * 2

    def test_short_looping_reply_is_left_alone(self):
        # Below the first threshold the guard never runs: "say X 50 times" must stream.
        agent = _make_agent()
        _stream(agent, _INCIDENT_CYCLE * 20)  # ~3.3k chars
        assert agent._interrupt_requested is False

    def test_long_distinct_reply_is_left_alone(self):
        agent = _make_agent()
        prose = " ".join(
            f"Paragraaf {i} beschrijft een eigen onderwerp met unieke woorden zoals "
            f"kwartel-{i} en meander-{i}, zodat geen venster ooit herhaalt."
            for i in range(700)
        )
        assert len(prose) > STREAM_GUARD_FIRST_CHECK_CHARS * 2
        _stream(agent, prose)
        assert agent._interrupt_requested is False

    def test_guard_fires_once_per_threshold_not_per_delta(self):
        # The check is O(n); running it on every token would be quadratic. It must run at the
        # threshold crossings only, so a reply that crosses one threshold is checked once there.
        agent = _make_agent()
        calls = []
        with patch(
            "agent.stream_delivery.is_runaway_repetition",
            side_effect=lambda t: calls.append(len(t)) or False,
        ):
            _stream(agent, "x" * (STREAM_GUARD_FIRST_CHECK_CHARS + 500), chunk=1)
        assert len(calls) == 1
        assert calls[0] >= STREAM_GUARD_FIRST_CHECK_CHARS
