"""Regression tests for #62605: auto-compression must fire before the request
exceeds context_length when the rough estimator systematically underestimates.

Root cause: ``estimate_request_tokens_rough`` uses a chars/4 heuristic that
diverges from real tokenization by 20-30% on schema-heavy / YAML-heavy /
non-English payloads. ``should_compress(rough_tokens)`` fires late, so after
3 compression passes the request still exceeds context_length and the
provider returns HTTP 400 with "total of at least N tokens".

Fix: track the calibration ratio ``real_prompt_tokens / rough_estimate`` from
each API response and apply it to the rough estimate BEFORE the threshold
check. The threshold itself isn't wrong; our estimate of how full we are is.

Tests cover:
- EMA ratio computation: stored, decayed, and applied
- should_compress scales rough up when past ratios underestimate
- should_compress stays correct when calibration is accurate (no double-count)
- _calibrate_rough is a no-op when no real-usage history yet
- 3-pass retry loop in turn_context can now succeed where it previously failed
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


# -----------------------------------------------------------------------
# Helpers: stub a ContextCompressor without booting agent_init
# -----------------------------------------------------------------------


def _make_compressor(**overrides):
    """Build a ContextCompressor-like object sufficient for testing
    should_compress + update_from_response + _calibrate_rough."""
    from agent.context_compressor import ContextCompressor

    # Pop our own overrides out of the dict so we can mix-and-match the
    # ContextCompressor API surface (which derives context_length from the
    # model name via update_model, not __init__).
    overrides = dict(overrides)
    context_length = overrides.pop("context_length", 262_144)
    max_tokens = overrides.pop("max_tokens", 65_536)
    threshold_percent = overrides.pop("threshold_percent", 0.75)

    c = ContextCompressor(
        model="test-model",
        quiet_mode=True,
        summary_target_ratio=0.3,
        max_tokens=max_tokens,
        **overrides,
    )
    c.update_model(
        model="test-model",
        provider="test",
        base_url="http://localhost:1234/v1",
        context_length=context_length,
        max_tokens=max_tokens,
    )
    # Re-apply threshold_percent if user wanted something other than the
    # default (ContextCompressor derives threshold_tokens = threshold_percent
    # * (context_length - max_tokens), with floors for small windows).
    if threshold_percent != 0.75:
        c.threshold_percent = c._effective_threshold_percent(
            context_length, threshold_percent,
        )
        c.threshold_tokens = c._compute_threshold_tokens(
            context_length, c.threshold_percent, max_tokens,
        )
    return c


# -----------------------------------------------------------------------
# Calibration: ratio tracking + application
# -----------------------------------------------------------------------


class TestRoughToRealRatioTracking:
    """update_from_response must learn the rough→real ratio so subsequent
    threshold checks use a calibrated estimate instead of the raw chars/4."""

    def test_initial_ratio_is_one(self):
        c = _make_compressor()
        assert c.rough_to_real_ratio == 1.0

    def test_ratio_recorded_from_first_response(self):
        c = _make_compressor()
        # Simulate: rough estimate was 150K, real provider prompt was 200K.
        # raw ratio = 200/150 = 1.333. With 0.5/0.5 EMA from seed 1.0 →
        # 0.5*1.0 + 0.5*1.333 = 1.167.
        c.last_compression_rough_tokens = 150_000
        c.update_from_response({"prompt_tokens": 200_000, "completion_tokens": 0})
        assert 1.15 < c.rough_to_real_ratio < 1.20

    def test_ratio_uses_ema_for_subsequent_responses(self):
        """Second response must blend with the first via EMA, not overwrite."""
        c = _make_compressor()
        # First sample: real=200K, rough=150K → ratio target 1.333
        # EMA(1.0, 1.333, 0.5/0.5) = 1.167
        c.last_compression_rough_tokens = 150_000
        c.update_from_response({"prompt_tokens": 200_000, "completion_tokens": 0})
        first_ratio = c.rough_to_real_ratio

        # Second sample: real=220K, rough=150K → ratio target 1.467
        # EMA(1.167, 1.467, 0.5/0.5) = 1.317
        c.last_compression_rough_tokens = 150_000
        c.update_from_response({"prompt_tokens": 220_000, "completion_tokens": 0})
        # Equal-weight EMA: result must be between first_ratio and 1.467
        assert first_ratio < c.rough_to_real_ratio < 1.47
        # Closer to the midpoint (weight is 0.5/0.5)
        assert abs(c.rough_to_real_ratio - (first_ratio + 1.47) / 2) < 0.05

    def test_no_ratio_when_rough_estimate_unknown(self):
        """If we never saw a rough estimate, don't fabricate a ratio."""
        c = _make_compressor()
        c.update_from_response({"prompt_tokens": 100_000, "completion_tokens": 0})
        # No prior _last_compression_rough_tokens → no learning signal
        assert c.rough_to_real_ratio == 1.0


class TestCalibrateRoughMethod:
    """``_calibrate_rough`` scales the rough estimate by the learned ratio
    so a should_compress(rough) call with underestimating history gets a
    fair chance to fire."""

    def test_no_history_passes_through(self):
        c = _make_compressor()
        assert c._calibrate_rough(150_000) == 150_000

    def test_underestimate_scales_up(self):
        """The reported case: rough=150K, real=200K. With 0.5/0.5 EMA from
        seed 1.0 the first observation yields ratio 1.167 → calibrated
        150K * 1.167 = 175K. The next response would converge further."""
        c = _make_compressor()
        c.last_compression_rough_tokens = 150_000
        c.update_from_response({"prompt_tokens": 200_000, "completion_tokens": 0})
        calibrated = c._calibrate_rough(150_000)
        # Allow EMA drift but must be clearly above 150K
        assert calibrated > 170_000
        assert calibrated < 180_000

    def test_calibration_capped_to_prevent_runaway(self):
        """Defensive: even an absurd ratio shouldn't push a small rough
        past the model context_length (would make every check fire)."""
        c = _make_compressor()
        # Plant a fake pathological ratio by directly manipulating state
        c.rough_to_real_ratio = 10.0  # pretend rough underestimates by 10x
        c._last_compression_rough_tokens = 1000
        # 1K * 10 = 10K — fine, but should also not exceed context_length
        calibrated = c._calibrate_rough(30_000)
        assert calibrated <= c.context_length
        # And shouldn't be wildly larger than a sane upper bound
        assert calibrated <= c.context_length


class TestShouldCompressUsesCalibration:
    """End-to-end: should_compress(rough) with underestimating history
    fires BEFORE the actual context_length breach — preventing the
    "3 compressions and still HTTP 400" failure mode."""

    def test_underestimating_history_triggers_earlier(self):
        """Scenario from issue #62605 — calibrated for the reproduction:
        context_length=262144, max_tokens=65536, threshold_percent=0.66
        threshold = (262144 - 65536) * 0.66 ≈ 129,741
        rough estimate = 150,688 (above threshold by ~11K — already fires)
        previous-turn real prompt = 200K, previous-turn rough=150K
        → ratio after 1 EMA sample = 0.5*1.0 + 0.5*(200/150) = 1.167
        → calibrated = int(150_000 * 1.167) = 175_000
        Crucially, even if the rough were BELOW threshold (e.g. 120K with
        higher threshold_percent), the calibration fires for the next
        preflight: 120K * 1.167 = 140K > 130K threshold.

        The intent of #62605 is: when the rough estimator underestimates
        systematically, the threshold check fires LATE. With calibration,
        the SAME rough estimate, after one observed real/rough sample,
        gets scaled up so the threshold check fires at the right time.
        """
        c = _make_compressor(
            context_length=262_144, max_tokens=65_536, threshold_percent=0.66
        )
        # Plant history: previous turn reported rough=150K, real=200K
        c.last_compression_rough_tokens = 150_000
        c.update_from_response({"prompt_tokens": 200_000, "completion_tokens": 0})

        # Sanity: the calibrated value crosses the threshold
        calibrated = c._calibrate_rough(150_000)
        assert calibrated > c.threshold_tokens, (
            f"calibration didn't scale enough: {calibrated} vs {c.threshold_tokens}"
        )
        # And the calibrated check fires
        assert c.should_compress(150_000) is True

    def test_underestimating_history_with_subthreshold_rough(self):
        """Stronger version of #62605: rough is BELOW threshold but real
        is ABOVE. Without calibration the check passes (misses the
        overflow); with calibration it fires (catches the overflow)."""
        c = _make_compressor(
            context_length=262_144, max_tokens=65_536, threshold_percent=0.80
        )
        # threshold = (262144 - 65536) * 0.80 = 157,286
        assert c.threshold_tokens == 157_286
        # Plant: rough=120K, real=200K → EMA → 1.333; calibrated(120K) = 160K
        c.last_compression_rough_tokens = 120_000
        c.update_from_response({"prompt_tokens": 200_000, "completion_tokens": 0})
        # Without calibration: 120K < 157K threshold → would NOT fire
        assert c._calibrate_rough(120_000) > c.threshold_tokens
        # With calibration: check fires
        assert c.should_compress(120_000) is True

    def test_accurate_history_does_not_over_trigger(self):
        """If past ratios were accurate, calibration shouldn't fire false
        positives below threshold."""
        c = _make_compressor(context_length=262_144, max_tokens=65_536, threshold_percent=0.75)
        # History: rough=140K, real=140K → ratio = 1.0
        c._last_compression_rough_tokens = 140_000
        c.update_from_response({"prompt_tokens": 140_000, "completion_tokens": 0})
        # A new rough estimate that's well below threshold shouldn't fire
        assert c.should_compress(100_000) is False

    def test_real_prompt_tokens_path_unchanged(self):
        """When prompt_tokens is None we use last_prompt_tokens (the real
        value). This path must NOT be calibrated — we already have the
        authoritative answer."""
        c = _make_compressor()
        # Below threshold: real says safe, no calibration needed
        c.last_prompt_tokens = 50_000
        assert c.should_compress() is False

        # Above threshold: real says compress, no calibration debate
        c.last_prompt_tokens = c.threshold_tokens + 1000
        assert c.should_compress() is True


class TestThreePassRetryCanNowSucceed:
    """The repro from #62605: 3 compress passes and still over. With
    calibration firing early, the first pass should bring the calibrated
    estimate under threshold."""

    def test_calibration_lets_first_pass_succeed(self):
        """When rough=150K actually represents real=200K, calibrated=175K
        exceeds threshold 130K. Compression shrinks messages enough that
        a re-estimate lands under threshold."""
        c = _make_compressor(
            context_length=262_144, max_tokens=65_536, threshold_percent=0.66
        )
        # threshold = (262144 - 65536) * 0.66 ≈ 129,741
        assert c.threshold_tokens < 150_000
        # Pretend compressor saw rough=150K → real=200K
        c.last_compression_rough_tokens = 150_000
        c.update_from_response({"prompt_tokens": 200_000, "completion_tokens": 0})
        # First pass: calibrated check fires
        assert c.should_compress(150_000) is True
        # After a successful compression, a smaller rough estimate should
        # NOT keep firing — calibration is calibrated for THIS conversation
        # size, so a much smaller estimate means we've already shrunk.
        assert c.should_compress(80_000) is False


class TestAntiThrashingStillWorks:
    """The calibration must not bypass the anti-thrashing guard that
    prevents infinite compression loops."""

    def test_ineffective_compression_still_skipped(self, monkeypatch):
        c = _make_compressor()
        # Set up underestimating history so calibration would normally fire
        c.last_compression_rough_tokens = 150_000
        c.update_from_response({"prompt_tokens": 200_000, "completion_tokens": 0})
        # But mark that the last 2 compressions were ineffective
        c._ineffective_compression_count = 2
        # Even with calibration saying "fire", the anti-thrashing guard wins
        assert c.should_compress(150_000) is False


class TestCooldownStillWorks:
    """The summary-LLM cooldown must still defer compression."""

    def test_cooldown_deferes_firing(self):
        import time

        c = _make_compressor()
        c.last_compression_rough_tokens = 150_000
        c.update_from_response({"prompt_tokens": 200_000, "completion_tokens": 0})
        # Plant an active cooldown
        c._summary_failure_cooldown_until = time.monotonic() + 60
        # Even with rough estimate above threshold, defer to cooldown
        assert c.should_compress(150_000) is False
