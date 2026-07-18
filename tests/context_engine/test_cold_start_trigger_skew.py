"""Cold-start trigger-skew prior (empty-history false-fire fix).

Root cause (2026-07-18 investigation): the compaction TRIGGER calibrates the
rough token estimate by the measured real/rough skew, but on an EMPTY skew
history ``_current_skew()`` returns 1.0 (raw rough). The rough estimator
over-counts schema/dense-heavy sessions (empirically p10=0.67, min=0.10 across
5,268 samples), so an uncalibrated first preflight on a large resumed session
FALSE-FIRES a premature LOSSY compaction (observed: raw 316,953 calibrated at
skew 1.000 >= 279,000 threshold while REAL usage was ~48% of the window).

The fix: a TRIGGER-ONLY cold-start prior (the conservative ``skew_floor``) on
empty history, so the first uncalibrated decision defers instead of false-firing.
The window hard-frac ceiling still backstops a genuine overflow regardless of
skew, so deferring can never cause a 413. Display skew (``_current_skew``) stays
identity-on-empty so the shown estimate remains an honest 'not yet measured'
(Greptile #111 contract).
"""
from __future__ import annotations

from agent.context_engine import ContextEngine


class _Engine(ContextEngine):
    """Minimal concrete engine exercising the shared ABC calibration methods."""

    def __init__(self, *, threshold_tokens: int, context_length: int,
                 skew_floor: float = 0.55, hard_frac: float = 0.95) -> None:
        self.threshold_tokens = threshold_tokens
        self.context_length = context_length
        self._skew_floor = skew_floor
        self._hard_frac = hard_frac
        self._recent_skews = []
        self._ineffective_compression_count = 0
        self._fallback_compression_streak = 0
        self._summary_failure_cooldown_until = 0.0
        self.quiet_mode = True

    @property
    def name(self) -> str:
        return "test-engine"

    def update_from_response(self, usage) -> None:
        pass

    def should_compress(self, prompt_tokens: int = None) -> bool:
        tokens = prompt_tokens if prompt_tokens is not None else 0
        if tokens < self.threshold_tokens:
            return False
        # Minimal anti-thrash mirror of ContextCompressor._automatic_compression_blocked.
        if (self._ineffective_compression_count >= 2
                or self._fallback_compression_streak >= 2):
            return False
        return True

    def compress(self, messages, current_tokens=None, focus_topic=None):
        return messages


def test_empty_history_display_skew_stays_identity() -> None:
    """Display contract: no reading yet ⇒ _current_skew == 1.0 (honest)."""
    e = _Engine(threshold_tokens=279_000, context_length=372_000)
    assert e._current_skew() == 1.0
    # calibrated_tokens (display path) must therefore be identity on empty.
    assert e.calibrated_tokens(316_953) == 316_953


def test_empty_history_trigger_skew_uses_cold_start_floor() -> None:
    """Trigger contract: no reading yet ⇒ trigger uses the conservative floor,
    NOT 1.0."""
    e = _Engine(threshold_tokens=279_000, context_length=372_000, skew_floor=0.55)
    assert e._trigger_skew() == 0.55


def test_logged_false_fire_is_now_deferred() -> None:
    """The exact 2026-07-18 incident: raw 316,953 on a 372K window, threshold
    279,000, empty history. At skew 1.0 it FIRED (calibrated 316,953 >= 279,000);
    real usage was ~178K (it fit). With the cold-start floor it must NOT fire."""
    e = _Engine(threshold_tokens=279_000, context_length=372_000, skew_floor=0.55)
    # 316,953 * 0.55 = 174,324 < 279,000 -> defer
    assert e.should_compress_calibrated(316_953) is False


def test_hard_frac_ceiling_still_fires_regardless_of_cold_start() -> None:
    """Overflow backstop: raw at/above the window hard-frac compacts even with
    the conservative cold-start skew (a 413 can never be deferred away)."""
    e = _Engine(threshold_tokens=279_000, context_length=372_000, skew_floor=0.55)
    ceiling = int(372_000 * 0.95)  # 353,400
    assert e.should_compress_calibrated(ceiling) is True
    assert e.should_compress_calibrated(ceiling + 10_000) is True


def test_populated_history_trigger_matches_current_skew() -> None:
    """Once real readings pair, trigger skew == display skew (median), so the
    cold-start prior governs ONLY the pre-first-reading window."""
    e = _Engine(threshold_tokens=279_000, context_length=372_000, skew_floor=0.55)
    e._recent_skews = [0.90, 0.92, 0.95]  # measured
    assert e._trigger_skew() == e._current_skew() == 0.92
    # calibrated 316,953 * 0.92 = 291,596 >= 279,000 -> fires on real accounting
    assert e.should_compress_calibrated(316_953) is True


def test_misconfigured_near_zero_floor_clamped_to_trigger_min() -> None:
    """Greptile PR #392 P2: a near-zero skew_floor (e.g. 0.01) must NOT shrink the
    cold-start trigger estimate to ~1% of raw (which would make the soft threshold
    unreachable until the 95% hard-frac ceiling). The trigger prior clamps UP to
    _TRIGGER_SKEW_MIN so the soft threshold stays reachable."""
    e = _Engine(threshold_tokens=279_000, context_length=372_000, skew_floor=0.01)
    assert e._recent_skews == []
    assert e._trigger_skew() == e._TRIGGER_SKEW_MIN
    # At the min (0.5): a request genuinely over the soft threshold still fires,
    # rather than being suppressed to ~1% of raw and waiting for the 95% ceiling.
    # 560,000 * 0.5 = 280,000 >= 279,000 -> fires (soft threshold reachable).
    assert e.should_compress_calibrated(560_000) is True


def test_zero_floor_clamped_to_trigger_min_not_disabled() -> None:
    """An explicit 0.0 floor (or the old `or DEFAULT` truthiness edge) clamps to
    the positive minimum, never to 0 (which would zero the estimate entirely)."""
    e = _Engine(threshold_tokens=279_000, context_length=372_000, skew_floor=0.0)
    # skew_floor=0.0 fails the ContextCompressor (0,1] validation -> default 0.7,
    # but even a raw 0.0 reaching _trigger_skew clamps to the min, never 0.
    assert e._trigger_skew() >= e._TRIGGER_SKEW_MIN
    assert e._trigger_skew() > 0.0


def test_cold_start_never_below_floor_after_data_either() -> None:
    """A measured median below the floor is clamped up to the floor by
    _current_skew, so trigger and display agree on the floor as the ratchet."""
    e = _Engine(threshold_tokens=279_000, context_length=372_000, skew_floor=0.60)
    e._recent_skews = [0.20, 0.25, 0.30]  # densest sessions, below floor
    assert e._current_skew() == 0.60  # clamped
    assert e._trigger_skew() == 0.60
