"""Tests for the opt-in idle-triggered compaction policy.

Covers ``agent.turn_context._should_idle_compact`` — the pure predicate that
decides whether a session resuming after an idle gap should compact up front.
The predicate is intentionally side-effect-free so the policy can be verified
without constructing a live agent or DB.
"""

from agent.turn_context import _should_idle_compact


def _decide(**overrides):
    """Call the predicate with sensible defaults (idle + large context => fire)."""
    kwargs = dict(
        enabled=True,
        idle_after_seconds=1800,
        idle_gap_seconds=3600.0,
        tokens=100_000,
        floor_tokens=40_000,
        cooldown_active=False,
    )
    kwargs.update(overrides)
    return _should_idle_compact(**kwargs)


class TestShouldIdleCompact:

    def test_disabled_when_idle_after_zero(self):
        # 0 is the documented "off" value — must never fire regardless of gap.
        assert _decide(idle_after_seconds=0, idle_gap_seconds=10_000.0) is False


    def test_disabled_when_compression_off(self):
        assert _decide(enabled=False) is False




    def test_fires_just_above_floor(self):
        assert _decide(tokens=40_001, floor_tokens=40_000) is True


class TestPostCompactionFloor:
    """The floor also honours what the previous pass actually produced (#97239).

    ``floor_tokens`` is the theoretical target (threshold × target_ratio); a
    real pass lands well above it because the system prompt, the tool schemas
    and the protected head/tail are incompressible. Without this, an already
    compacted session re-summarises itself on every idle resume forever.
    """

    def test_unrecorded_last_compaction_keeps_original_floor(self):
        # 0 = nothing compacted yet (or state reset) — original semantics.
        assert _decide(tokens=40_001, floor_tokens=40_000,
                       last_compaction_tokens=0) is True

    def test_skips_when_transcript_has_not_grown_since_last_compaction(self):
        # Previous pass produced 44,000; the transcript is still ~that size.
        assert _decide(tokens=44_100, floor_tokens=40_000,
                       last_compaction_tokens=44_000) is False

    def test_fires_once_a_full_floor_of_new_content_accumulated(self):
        assert _decide(tokens=84_001, floor_tokens=40_000,
                       last_compaction_tokens=44_000) is True

    def test_does_not_fire_at_exactly_the_raised_floor(self):
        assert _decide(tokens=84_000, floor_tokens=40_000,
                       last_compaction_tokens=44_000) is False

    def test_reported_session_stops_recompacting_itself(self):
        """Exact numbers from issue #97239.

        The 17:01 pass reduced 64,105 -> 44,579 tokens; the 17:36 resume
        re-fired on that same transcript because 44,579 > the 25,502
        theoretical floor, blocking the prompt for another 256 s.
        """
        common = dict(idle_after_seconds=1, idle_gap_seconds=747.0,
                      floor_tokens=25_502)
        # Before the fix the second resume fired: 44,579 > 25,502.
        assert _decide(tokens=44_579, last_compaction_tokens=0, **common) is True
        # With the previous pass's real output known, it sits the round out.
        assert _decide(tokens=44_579, last_compaction_tokens=44_579,
                       **common) is False

    def test_an_effective_pass_still_raises_the_floor(self):
        # 100K -> 10K is a good pass; another one is worth it only once about
        # a floor's worth of new content has landed on top of the 10K.
        assert _decide(tokens=30_000, floor_tokens=25_000,
                       last_compaction_tokens=10_000) is False
        assert _decide(tokens=35_001, floor_tokens=25_000,
                       last_compaction_tokens=10_000) is True

    def test_other_gates_still_win_over_the_raised_floor(self):
        # Growth alone must not defeat the cooldown / opt-out gates.
        assert _decide(tokens=200_000, floor_tokens=40_000,
                       last_compaction_tokens=44_000,
                       cooldown_active=True) is False
        assert _decide(tokens=200_000, floor_tokens=40_000,
                       last_compaction_tokens=44_000,
                       idle_after_seconds=0) is False
        assert _decide(tokens=200_000, floor_tokens=40_000,
                       last_compaction_tokens=44_000,
                       idle_gap_seconds=0.5) is False

