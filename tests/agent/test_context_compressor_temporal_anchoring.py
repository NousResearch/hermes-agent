"""Tests for temporal anchoring in context-compaction summaries.

The summarizer is handed the current date and instructed to rewrite completed
actions as absolute, dated, past-tense facts (e.g. "email John" ->
"Sent the proposal email to John on 2026-06-07"). This keeps a resumed
conversation from re-issuing work that already happened. Date resolution is
best-effort: a clock failure must omit the rule, never block compaction.

These exercise ``_generate_summary`` directly -- the function that builds the
summarizer prompt. ``test_context_compressor_summary_continuity`` already
proves ``compress()`` routes into ``_generate_summary``.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import hermes_time
from agent.context_compressor import ContextCompressor, HISTORICAL_TASK_HEADING


def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=1,
            quiet_mode=True,
        )


def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _turns():
    return [
        {"role": "user", "content": "do the first thing"},
        {"role": "assistant", "content": "did the first thing"},
        {"role": "user", "content": "do the second thing"},
        {"role": "assistant", "content": "did the second thing"},
    ]


def _fixed_now():
    return datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)






def test_clock_failure_omits_rule_but_compaction_still_runs():
    compressor = _compressor()

    def _boom():
        raise RuntimeError("clock unavailable")

    with patch.object(hermes_time, "now", _boom), patch(
        "agent.context_compressor.call_llm", return_value=_response("summary")
    ) as mock_call:
        result = compressor._generate_summary(_turns())

    # call_llm was still invoked -> compaction was not blocked by the clock error.
    assert mock_call.called
    assert result is not None
    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "TEMPORAL ANCHORING" not in prompt
    # Structured template still intact.
    assert HISTORICAL_TASK_HEADING in prompt


def test_anchoring_rule_uses_date_from_hermes_time_now():
    """The date is taken from hermes_time.now(), which respects the user's TZ."""
    compressor = _compressor()
    fixed = datetime(2025, 12, 31, 23, 30, tzinfo=timezone.utc)
    with patch.object(hermes_time, "now", lambda: fixed), patch(
        "agent.context_compressor.call_llm", return_value=_response("summary")
    ) as mock_call:
        compressor._generate_summary(_turns())

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "2025-12-31" in prompt
