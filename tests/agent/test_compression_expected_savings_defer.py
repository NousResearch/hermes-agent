"""Expected-savings guard: defer a negligible-reclaim pass BEFORE the first summary.

The pre-existing `_feasibility_skip` only arms `_FEASIBILITY_SKIP_MIDDLE_FRACTION`
AFTER a real-usage ineffectiveness strike, so the FIRST expensive low-reclaim pass
always goes through. The expected-savings guard runs on the first attempt too: when
the reclaimable middle cannot cover the over-threshold deficit, it DEFERS the pass
(transcript returned unchanged, no summary LLM call, no messages dropped). Forced
runs and sessions near the provider's hard context limit bypass it.

Fixtures use a pinned trigger (`threshold_tokens`) so the "negligible middle" shape is
deterministic; the message shapes below were measured against the real boundary helpers.
"""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor

# Trigger pinned in the tests; deficit = current_tokens - threshold.
_PINNED_THRESHOLD = 40000
# Real calibrations that the auto caller passes: over the trigger, below the hard-limit bypass.
_OVER_TRIGGER = 45000


def _compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
        _ = c.context_length  # resolve under the mock
    c.threshold_tokens = _PINNED_THRESHOLD
    return c


def _negligible_middle():
    """Measured middle ≈ 474 tokens (< 4000): the target low-reclaim shape."""
    msgs = [{"role": "system", "content": "system prompt"}]
    for i in range(6):
        msgs.append({"role": "user", "content": f"eq{i}"})
        msgs.append({"role": "assistant", "content": "e" * 300})
    for i in range(2):
        msgs.append({"role": "user", "content": f"rq{i}"})
        msgs.append({"role": "assistant", "content": "r" * 24000})
    return msgs


def _grown_middle():
    """Measured middle ≈ 6483 tokens (>= 4000, above the deferred size): real material."""
    msgs = [{"role": "system", "content": "system prompt"}]
    for i in range(6):
        msgs.append({"role": "user", "content": f"eq{i}"})
        msgs.append({"role": "assistant", "content": "e" * 300})
    for i in range(3):
        msgs.append({"role": "user", "content": f"rq{i}"})
        msgs.append({"role": "assistant", "content": "r" * 24000})
    return msgs


def test_negligible_middle_is_deferred_on_first_attempt():
    """No prior strike: a tiny middle still defers instead of paying for a summary."""
    compressor = _compressor()
    msgs = _negligible_middle()
    assert compressor._ineffective_compression_count == 0

    with patch.object(compressor, "_generate_summary") as mock_gen:
        result = compressor.compress(list(msgs), force=False, current_tokens=_OVER_TRIGGER)

    mock_gen.assert_not_called()
    assert compressor._last_reclaim_deferred is True
    # Deferred, not skipped: this is not the strike-neutral feasibility path.
    assert compressor._prellm_skip_count == 0
    assert compressor._last_feasibility_skip is False
    # No messages dropped: the transcript comes back unchanged.
    assert result == msgs
    telemetry = compressor._last_compression_telemetry or {}
    assert telemetry.get("failure_class") == "reclaim_deferred"
    assert telemetry.get("reclaim_deferred") is True


def test_defer_does_not_strike_or_touch_streaks():
    """A defer is neither an ineffective strike nor a fallback-streak verdict."""
    compressor = _compressor()
    msgs = _negligible_middle()

    with patch.object(compressor, "_generate_summary"):
        compressor.compress(list(msgs), force=False, current_tokens=_OVER_TRIGGER)

    assert compressor._ineffective_compression_count == 0
    assert compressor._fallback_compression_streak == 0
    assert not compressor._automatic_compression_blocked_locally()


def test_force_bypasses_expected_savings_defer():
    """Manual /compress (force) always exercises the full summary path."""
    compressor = _compressor()
    msgs = _negligible_middle()

    with patch.object(compressor, "_generate_summary", return_value="LLM summary") as mock_gen:
        compressor.compress(list(msgs), force=True)

    mock_gen.assert_called_once()
    assert compressor._last_reclaim_deferred is False


def test_near_hard_context_limit_bypasses_defer():
    """Near the provider's real window a smaller reclaim must still be allowed."""
    compressor = _compressor()
    msgs = _negligible_middle()
    near_limit = compressor._reclaim_defer_hard_limit()  # int(window * 0.9)

    with patch.object(compressor, "_generate_summary", return_value="LLM summary") as mock_gen:
        compressor.compress(list(msgs), force=False, current_tokens=near_limit)

    mock_gen.assert_called_once()
    assert compressor._last_reclaim_deferred is False


def test_below_trigger_and_unknown_tokens_keep_legacy_behavior():
    """Under the trigger (or an unknown count) the guard is inert: pre-existing paths decide."""
    compressor = _compressor()
    msgs = _negligible_middle()

    for current in (None, _PINNED_THRESHOLD - 1):
        compressor._last_reclaim_deferred = False
        with patch.object(compressor, "_generate_summary", return_value="LLM summary"):
            compressor.compress(list(msgs), force=False, current_tokens=current)
        # Not the expected-savings defer; whatever other guard fires is the legacy behavior.
        assert compressor._last_reclaim_deferred is False


def test_grown_middle_attempts_again_after_defer():
    """Once the compressible middle can cover the deficit, a fresh attempt runs."""
    compressor = _compressor()

    with patch.object(compressor, "_generate_summary") as mock_gen:
        compressor.compress(list(_negligible_middle()), force=False, current_tokens=_OVER_TRIGGER)
    mock_gen.assert_not_called()
    assert compressor._last_reclaim_deferred is True

    with patch.object(compressor, "_generate_summary", return_value="LLM summary") as mock_gen:
        compressor.compress(list(_grown_middle()), force=False, current_tokens=_PINNED_THRESHOLD + 2000)

    mock_gen.assert_called_once()
    assert compressor._last_reclaim_deferred is False


def test_small_deficit_is_not_deferred_even_with_small_middle():
    """Reviewer scenario: barely over threshold, a small middle can still clear the deficit."""
    compressor = _compressor()

    with patch.object(compressor, "_generate_summary", return_value="LLM summary") as mock_gen:
        compressor.compress(list(_negligible_middle()), force=False, current_tokens=_PINNED_THRESHOLD + 100)

    mock_gen.assert_called_once()
    assert compressor._last_reclaim_deferred is False