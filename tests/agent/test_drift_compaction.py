"""Tests for the opt-in JEV drift-triggered compaction policy.

Covers ``agent.turn_context._should_drift_compact`` — the pure predicate that decides
whether a topic change may compact the old context early — and the lexical dependence
heuristic behind it. Both are intentionally side-effect-free so the policy can be
verified without constructing a live agent or DB.
"""

from agent.turn_context import (
    _DRIFT_DEPENDENT_MIN,
    _DRIFT_INDEPENDENT_MAX,
    _should_drift_compact,
    drift_dependence_score,
    drift_score_ambiguous,
)


def _decide(**overrides):
    """Call the predicate with sensible defaults (new topic + large context => fire)."""
    kwargs = dict(
        enabled=True,
        tokens=60_000,
        window_tokens=65_536,
        min_ratio=0.5,
        dependence_score=0.0,
        confirm_independent=None,
        cooldown_active=False,
        in_task=False,
        floor_tokens=20_000,
        last_compaction_tokens=0,
    )
    kwargs.update(overrides)
    return _should_drift_compact(**kwargs)


class TestShouldDriftCompact:

    def test_independent_new_direction_fires(self):
        assert _decide() is True

    def test_disabled(self):
        assert _decide(enabled=False) is False

    def test_never_during_task_continuation(self):
        # The #1 false-drift failure mode: a follow-up that looks like a new topic but
        # depends on in-flight work.
        assert _decide(in_task=True) is False

    def test_never_below_min_ratio(self):
        # 30k < 0.5 * 65,536 — drift compaction is opportunistic, not an overflow valve.
        assert _decide(tokens=30_000) is False

    def test_never_below_floor(self):
        assert _decide(tokens=25_000, min_ratio=0.0, floor_tokens=40_000) is False

    def test_last_compaction_raises_floor(self):
        # Must gain a floor's worth of NEW content after a compaction (#97239 contract).
        assert _decide(
            tokens=45_000, min_ratio=0.0, floor_tokens=20_000, last_compaction_tokens=40_000
        ) is False

    def test_dependent_continuation_never_fires(self):
        assert _decide(dependence_score=0.9) is False

    def test_ambiguous_band_requires_model_ok(self):
        ambiguous = (_DRIFT_INDEPENDENT_MAX + _DRIFT_DEPENDENT_MIN) / 2
        assert drift_score_ambiguous(ambiguous) is True
        assert _decide(dependence_score=ambiguous, confirm_independent=None) is False
        assert _decide(dependence_score=ambiguous, confirm_independent=False) is False
        assert _decide(dependence_score=ambiguous, confirm_independent=True) is True

    def test_no_signal_requires_model_ok(self):
        assert _decide(dependence_score=None, confirm_independent=None) is False
        assert _decide(dependence_score=None, confirm_independent=True) is True

    def test_cooldown_active_blocks(self):
        assert _decide(cooldown_active=True) is False

    def test_zero_window_never_fires(self):
        assert _decide(window_tokens=0) is False


class TestDependenceScore:

    def test_disjoint_texts_score_low(self):
        score = drift_dependence_score(
            "redeploy the kubernetes cluster in frankfurt",
            ["the quarterly budget spreadsheet needs revision", "invoice totals look wrong"],
        )
        assert score is not None
        assert score <= _DRIFT_INDEPENDENT_MAX

    def test_shared_vocabulary_scores_high(self):
        score = drift_dependence_score(
            "why did the gateway crash loop keep respawning",
            ["the gateway crash loop kept respawning the telegram adapter"],
        )
        assert score is not None
        assert score >= _DRIFT_DEPENDENT_MIN

    def test_too_little_signal_is_none(self):
        assert drift_dependence_score("ok", ["nothing useful here"]) is None
        assert drift_dependence_score("a real message with words", []) is None

    def test_ambiguous_band_detection(self):
        assert drift_score_ambiguous(None) is True
        assert drift_score_ambiguous(0.0) is False
        assert drift_score_ambiguous(1.0) is False
