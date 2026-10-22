"""#142355: one over-reported provider ``usage.prompt_tokens`` must not kill a session.

The preflight estimate prefers the provider usage anchor. A provider that once reported
598,980 prompt tokens for a ~241k request on a 500k window made every later turn fail
closed with "start a new session": compression reclaimed nothing (there was nothing to
reclaim), the anchor's priced prefix survived, and no request was ever sent to replace it.

An anchor above the window while the local estimate fits is self-contradictory (the
provider accepted that request), so it is discarded with a warning naming both numbers.
When both signals agree the request is over the window, the turn fails closed as before.
"""
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_context import PreflightCompressionTimedOut, _preflight_request_tokens
from agent.turn_context_compaction import CompactionOutcome, _run_preflight_passes
from agent.usage_anchor import USAGE_ANCHOR_MODEL_CONFIG_KEY, anchored_context_tokens, capture_usage_anchor


def _transcript(chars_per_turn=4_000, turns=10):
    messages = []
    for i in range(turns):
        messages.append({"role": "user", "content": f"question {i} " + "x" * chars_per_turn})
        messages.append({"role": "assistant", "content": f"answer {i} " + "y" * chars_per_turn})
    return messages


def _agent(messages, *, prompt_tokens, window):
    db = MagicMock()
    agent = SimpleNamespace(
        tools=None, api_mode="", provider="openai", model="m", session_id="s1", _session_db=db,
        _persist_disabled=False,
        _usage_anchor=capture_usage_anchor(prompt_tokens, 100, messages),
        context_compressor=SimpleNamespace(
            protect_first_n=3, protect_last_n=3, threshold_tokens=window // 2, context_length=window,
            summary_target_ratio=0.5, should_compress=lambda tokens: tokens >= window // 2,
        ),
        max_compression_attempts=3, _emit_status=MagicMock(), _compression_blocked_transient=None,
    )
    agent._compress_context = lambda msgs, system, **kw: (msgs, system)  # reclaims nothing
    return agent


def test_over_window_anchor_with_a_fitting_local_estimate_is_discarded(caplog):
    messages = _transcript()
    agent = _agent(messages, prompt_tokens=598_980, window=500_000)
    assert anchored_context_tokens(messages, agent._usage_anchor) > 500_000

    with caplog.at_level(logging.WARNING, logger="agent.turn_context"):
        tokens = _preflight_request_tokens(agent, messages, "sys")

    assert tokens < 500_000, "the implausible anchor still drives the estimate"
    assert agent._usage_anchor is None
    assert agent._request_pressure_anchored is False
    # The clear reaches the session row, so a restarted process does not restore it.
    agent._session_db.patch_session_model_config.assert_called_once_with(
        "s1", {USAGE_ANCHOR_MODEL_CONFIG_KEY: None},
    )
    warning = " ".join(r.getMessage() for r in caplog.records)
    assert "599,080" in warning  # the anchored figure (prompt 598,980 + completion 100)
    assert "500,000" in warning
    assert f"{tokens:,}" in warning


def test_session_survives_preflight_after_the_bad_reading():
    """The reported field failure: a pass reclaims nothing on an over-window anchor whose
    local estimate fits. The turn must not end with "start a new session"."""
    messages = _transcript()
    agent = _agent(messages, prompt_tokens=598_980, window=500_000)
    out = CompactionOutcome(
        messages=messages, active_system_prompt="sys", conversation_history=None, current_turn_user_idx=0,
    )
    with patch("agent.turn_context_compaction.automatic_compaction_status_message", return_value=""):
        _run_preflight_passes(agent, out, agent.context_compressor, 598_980, "sys", "t")  # no raise
    assert agent._usage_anchor is None


def test_anchor_and_local_estimate_both_over_the_window_still_fail_closed():
    messages = _transcript(chars_per_turn=40_000, turns=10)  # ~200k local tokens on a 100k window
    agent = _agent(messages, prompt_tokens=210_000, window=100_000)

    tokens = _preflight_request_tokens(agent, messages, "sys")

    assert tokens == anchored_context_tokens(messages, agent._usage_anchor)
    assert agent._usage_anchor is not None
    out = CompactionOutcome(
        messages=messages, active_system_prompt="sys", conversation_history=None, current_turn_user_idx=0,
    )
    with patch("agent.turn_context_compaction.automatic_compaction_status_message", return_value=""):
        with pytest.raises(PreflightCompressionTimedOut, match="Start a new session with /new"):
            _run_preflight_passes(agent, out, agent.context_compressor, tokens, "sys", "t")


def test_anchor_inside_the_window_is_still_authoritative():
    messages = _transcript()
    agent = _agent(messages, prompt_tokens=233_654, window=500_000)
    expected = anchored_context_tokens(messages, agent._usage_anchor)

    assert _preflight_request_tokens(agent, messages, "sys") == expected
    assert agent._usage_anchor is not None
    assert agent._request_pressure_anchored is True
    agent._session_db.patch_session_model_config.assert_not_called()


@pytest.mark.parametrize("window", [0, None])
def test_unknown_window_keeps_the_anchor(window):
    messages = _transcript()
    agent = _agent(messages, prompt_tokens=598_980, window=500_000)
    agent.context_compressor.context_length = window

    assert _preflight_request_tokens(agent, messages, "sys") == anchored_context_tokens(messages, agent._usage_anchor)
    assert agent._usage_anchor is not None
