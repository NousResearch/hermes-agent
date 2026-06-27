"""T0 + T1 — skew-pairing consume fix + COMPACTION_SKEW telemetry.

T0 (Phase 0): `record_skew_from_real` must CONSUME `_last_rough_sent` (reset to 0)
after using it, so a single `note_rough_sent` pairs with exactly ONE real reading —
not the N growing real readings of a multi-call turn (the stale-rough contamination
the pass-1 review found in shipped P2).

T1 (Phase 1): emit one `COMPACTION_SKEW rough=… real=… ratio=… task=… model=… ctx=…`
info line per FRESH pair, best-effort (never raises into the hot path), plus a
dedicated append-only sample sink for the v0.2 floor tune.

Spec: ~/.hermes/plans/2026-06-27_skew-telemetry-and-render-harness-SPEC.md (v0.3).
"""
from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.context_engine import ContextEngine


@pytest.fixture(autouse=True)
def _isolated_skew_sink(tmp_path, monkeypatch):
    """Redirect the skew-sample sink to a tmp HERMES_HOME so these tests never
    write to the live ~/.hermes/state/skew-samples.log (INV-4/isolation)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield


@pytest.fixture
def compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.75,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )


# ─────────────────────────── Phase 0 — T0 consume ───────────────────────────

class TestT0ConsumeStaleRough:
    def test_two_reals_after_one_rough_record_one_ratio(self, compressor):
        """RED before T0: a single note_rough_sent followed by two
        update_from_response (the multi-call turn shape) appends TWO ratios with
        the same stale rough / different reals. After T0: exactly ONE."""
        compressor.note_rough_sent(1000)
        compressor.update_from_response({"prompt_tokens": 800, "completion_tokens": 5})
        compressor.update_from_response({"prompt_tokens": 850, "completion_tokens": 5})
        compressor.update_from_response({"prompt_tokens": 900, "completion_tokens": 5})
        assert len(compressor._recent_skews) == 1
        # the ONE recorded ratio is the first real ÷ rough
        assert abs(compressor._recent_skews[0] - 0.8) < 1e-9

    def test_rough_consumed_resets_to_zero(self, compressor):
        compressor.note_rough_sent(1000)
        compressor.update_from_response({"prompt_tokens": 800, "completion_tokens": 5})
        assert compressor._last_rough_sent == 0

    def test_fresh_rough_each_turn_records_each(self, compressor):
        """Two real turns, each with its own note_rough_sent → two ratios."""
        compressor.note_rough_sent(1000)
        compressor.update_from_response({"prompt_tokens": 800, "completion_tokens": 5})
        compressor.note_rough_sent(2000)
        compressor.update_from_response({"prompt_tokens": 1000, "completion_tokens": 5})
        assert len(compressor._recent_skews) == 2
        assert abs(compressor._recent_skews[0] - 0.8) < 1e-9
        assert abs(compressor._recent_skews[1] - 0.5) < 1e-9

    def test_no_rough_no_record(self, compressor):
        """An update_from_response with no preceding (unconsumed) note_rough_sent
        records nothing — the aux-route shape."""
        compressor.update_from_response({"prompt_tokens": 800, "completion_tokens": 5})
        assert compressor._recent_skews == []


class TestT0FenceFormulaUnchanged:
    """INV-1 (1): the calibration FORMULA is byte-unchanged — for a HAND-SET
    history, _current_skew / calibrated_tokens give the same answer as always."""

    def test_current_skew_median_unchanged(self, compressor):
        compressor._skew_floor = 0.6
        compressor._recent_skews = [0.7, 0.8, 0.9]
        assert abs(compressor._current_skew() - 0.8) < 1e-9  # median, clamped

    def test_calibrated_tokens_unchanged(self, compressor):
        compressor._skew_floor = 0.6
        compressor._recent_skews = [0.8, 0.8, 0.8]
        assert compressor.calibrated_tokens(1000) == 800

    def test_floor_clamp_unchanged(self, compressor):
        compressor._skew_floor = 0.7
        compressor._recent_skews = [0.4, 0.4, 0.4]
        assert compressor._current_skew() == 0.7


class TestT0BehaviorTriggerMoves:
    """INV-1 (2): T0 changes WHICH ratios populate the median, so the live trigger
    point MOVES (toward correct). Same raw event sequence, different resulting skew
    than the pre-T0 contaminated behavior would have produced."""

    def test_median_reflects_single_correct_pair_not_contaminated_set(self, compressor):
        # Multi-call turn: rough 1000, reals 800/850/900/950/990 (growing as the
        # turn appends tool results). Pre-T0 the median would be over the set
        # {0.8,0.85,0.9,0.95,0.99} = 0.9 (contaminated by stale-rough pairing).
        # Post-T0 only the FIRST pair (0.8) is recorded → median 0.8.
        compressor.note_rough_sent(1000)
        for real in (800, 850, 900, 950, 990):
            compressor.update_from_response({"prompt_tokens": real, "completion_tokens": 5})
        assert compressor._recent_skews == [0.8]
        assert abs(compressor._current_skew() - 0.8) < 1e-9
        # The contaminated median (0.9) is what pre-T0 produced; prove we are NOT that.
        contaminated_median = 0.9
        assert compressor._current_skew() != contaminated_median


# ─────────────────────────── Phase 1 — T1 telemetry ───────────────────────────

class TestT1SkewTelemetry:
    def test_emits_one_line_per_fresh_pair(self, compressor, caplog):
        with caplog.at_level(logging.INFO):
            compressor.note_rough_sent(1000)
            compressor.update_from_response({"prompt_tokens": 800, "completion_tokens": 5})
        lines = [r.getMessage() for r in caplog.records if "COMPACTION_SKEW" in r.getMessage()]
        assert len(lines) == 1
        assert "rough=1000" in lines[0]
        assert "real=800" in lines[0]
        assert "ratio=0.800" in lines[0]
        assert "task=main" in lines[0]

    def test_second_real_no_rough_emits_no_line(self, compressor, caplog):
        with caplog.at_level(logging.INFO):
            compressor.note_rough_sent(1000)
            compressor.update_from_response({"prompt_tokens": 800, "completion_tokens": 5})
            # no new note_rough_sent → consumed → no second line
            compressor.update_from_response({"prompt_tokens": 850, "completion_tokens": 5})
        lines = [r.getMessage() for r in caplog.records if "COMPACTION_SKEW" in r.getMessage()]
        assert len(lines) == 1

    def test_no_rough_emits_no_line(self, compressor, caplog):
        with caplog.at_level(logging.INFO):
            compressor.update_from_response({"prompt_tokens": 800, "completion_tokens": 5})
        assert not [r for r in caplog.records if "COMPACTION_SKEW" in r.getMessage()]

    def test_lcm_engine_also_emits(self, caplog):
        from plugins.context_engine.lcm.config import LCMConfig
        from plugins.context_engine.lcm.engine import LCMEngine
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            eng = LCMEngine(config=LCMConfig(), hermes_home=td)
            with caplog.at_level(logging.INFO):
                eng.note_rough_sent(2000)
                eng.update_from_response({"prompt_tokens": 1000, "completion_tokens": 5})
            lines = [r.getMessage() for r in caplog.records if "COMPACTION_SKEW" in r.getMessage()]
            assert len(lines) == 1
            assert "ratio=0.500" in lines[0]


class TestT1NeverRaises:
    """INV-2: the emit is best-effort — a broken handler / missing attr must never
    propagate out of record_skew_from_real into the live turn."""

    def test_broken_logger_handler_does_not_propagate(self, compressor):
        with patch("agent.context_engine.logger") as mock_logger:
            mock_logger.info.side_effect = RuntimeError("handler exploded")
            compressor.note_rough_sent(1000)
            # must NOT raise — the ratio is still recorded, the log failure is swallowed
            compressor.update_from_response({"prompt_tokens": 800, "completion_tokens": 5})
            assert compressor._recent_skews == [0.8]

    def test_missing_model_ctx_attrs_no_attributeerror(self, compressor):
        """The emit reads model/context_length via getattr — if a subclass/instance
        doesn't have them set, the emit must not hit an AttributeError (B-2)."""
        # Simulate the bare-ABC case: attributes absent on the instance.
        try:
            del compressor.model
        except AttributeError:
            pass
        try:
            del compressor.context_length
        except AttributeError:
            pass
        compressor.note_rough_sent(1000)
        compressor.record_skew_from_real(800)  # must not raise
        assert compressor._recent_skews == [0.8]


class TestT1AuxPathNoEmit:
    """D-7: an aux-route update_from_response (tagged _aux_task, no preceding
    note_rough_sent of its own) emits NO COMPACTION_SKEW line — keeps task=main
    separable for the floor tune."""

    def test_aux_task_without_rough_emits_nothing(self, compressor, caplog):
        compressor._aux_task = "compression"
        with caplog.at_level(logging.INFO):
            # aux route: no note_rough_sent for this engine → last_rough=0 → no pair
            compressor.update_from_response({"prompt_tokens": 800, "completion_tokens": 5})
        assert not [r for r in caplog.records if "COMPACTION_SKEW" in r.getMessage()]
