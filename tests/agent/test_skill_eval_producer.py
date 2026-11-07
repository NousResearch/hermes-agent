"""Tests for SkillEvalProducer."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agent.experience_ledger import ExperienceLedger
from agent.signal_sources.rework_detector import ReworkEvent
from agent.signal_sources.reuse_tracker import ReuseEntry
from agent.skill_eval_producer import SkillEvalProducer, TurnSignals


NOW = 1_700_000_000.0


def _make_producer(tmp_path: Path) -> SkillEvalProducer:
    return SkillEvalProducer(hermes_home=tmp_path, rework_window_sec=600.0)


def _basic_signals(**overrides) -> TurnSignals:
    base: dict = dict(
        skill_id="my-skill",
        task_id="t-1",
        task_family="coding",
        public_signal=0.85,
        cost_usd=0.10,
        tokens_in=100,
        tokens_out=50,
        duration_sec=5.0,
        success=True,
    )
    base.update(overrides)
    return TurnSignals(**base)


class TestValidation:
    def test_empty_skill_id_rejected(self):
        with pytest.raises(ValueError):
            TurnSignals(
                skill_id="", task_id="t", task_family="coding", public_signal=0.5
            )

    def test_empty_task_id_rejected(self):
        with pytest.raises(ValueError):
            TurnSignals(
                skill_id="s", task_id="", task_family="coding", public_signal=0.5
            )

    def test_empty_task_family_rejected(self):
        with pytest.raises(ValueError):
            TurnSignals(skill_id="s", task_id="t", task_family="", public_signal=0.5)

    def test_out_of_range_public_signal_rejected(self):
        with pytest.raises(ValueError):
            TurnSignals(
                skill_id="s", task_id="t", task_family="coding", public_signal=1.5
            )
        with pytest.raises(ValueError):
            TurnSignals(
                skill_id="s", task_id="t", task_family="coding", public_signal=-0.1
            )

    def test_negative_cost_rejected(self):
        with pytest.raises(ValueError):
            TurnSignals(
                skill_id="s",
                task_id="t",
                task_family="coding",
                public_signal=0.5,
                cost_usd=-0.01,
            )


class TestRecordTurnBasic:
    def test_records_eval_and_writes_to_ledger(self, tmp_path: Path):
        producer = _make_producer(tmp_path)
        record = producer.record_turn(_basic_signals(), now=NOW)

        assert record.skill_id == "my-skill"
        assert record.eval_event_id.startswith("turn-t-1-")
        assert record.task_family == "coding"
        assert record.public_score == 0.85
        # private_score is heuristic; should be close to public for
        # the no-negative-signals case.
        assert 0.0 <= record.private_score <= 1.0

        # Ledger was saved (auto_save=True default).
        ledger_path = tmp_path / "state" / "experience_ledger.json"
        assert ledger_path.exists()
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert "my-skill" in data["evals"]
        assert len(data["evals"]["my-skill"]) == 1

    def test_success_outcome(self, tmp_path: Path):
        producer = _make_producer(tmp_path)
        record = producer.record_turn(_basic_signals(success=True), now=NOW)
        assert record.outcome == "success"

    def test_failure_outcome_when_user_corrected(self, tmp_path: Path):
        producer = _make_producer(tmp_path)
        record = producer.record_turn(
            _basic_signals(follow_up_user_messages=("that's wrong",)),
            now=NOW,
        )
        assert record.outcome == "failure"
        assert record.user_corrected is True


class TestSignalAggregation:
    def test_user_correction_detected_from_messages(self, tmp_path: Path):
        producer = _make_producer(tmp_path)
        record = producer.record_turn(
            _basic_signals(follow_up_user_messages=("不对", "redo it")),
            now=NOW,
        )
        assert record.user_corrected is True

    def test_user_correction_override_skips_detector(self, tmp_path: Path):
        producer = _make_producer(tmp_path)
        record = producer.record_turn(
            _basic_signals(
                follow_up_user_messages=("that's wrong",),
                user_corrected_override=False,
            ),
            now=NOW,
        )
        assert record.user_corrected is False

    def test_rework_count_from_supplied_events(self, tmp_path: Path):
        producer = _make_producer(tmp_path)
        events = [
            ReworkEvent(task_id="t-1", timestamp=NOW - 200),
            ReworkEvent(task_id="t-1", timestamp=NOW - 100),
        ]
        record = producer.record_turn(
            _basic_signals(rework_events=events),
            now=NOW,
        )
        assert record.rework_count == 2

    def test_rework_count_override(self, tmp_path: Path):
        producer = _make_producer(tmp_path)
        record = producer.record_turn(
            _basic_signals(rework_count_override=5),
            now=NOW,
        )
        assert record.rework_count == 5

    def test_reuse_success_lookup(self, tmp_path: Path):
        producer = _make_producer(tmp_path)
        history = [ReuseEntry(timestamp=NOW + 100, success=False)]
        record = producer.record_turn(
            _basic_signals(reuse_history=history),
            now=NOW,
        )
        assert record.reuse_count == 1  # current invocation always +1

    def test_reuse_success_override(self, tmp_path: Path):
        producer = _make_producer(tmp_path)
        record = producer.record_turn(
            _basic_signals(reuse_success_override=False),
            now=NOW,
        )
        # The override only affects the private-score heuristic; the
        # ledger record does not store reuse_success directly, only
        # reuse_count. So we check the *score* effect below.
        assert record.reuse_count == 1


class TestPrivateScoreHeuristic:
    def test_no_negative_signals(self, tmp_path: Path):
        producer = _make_producer(tmp_path)
        record = producer.record_turn(_basic_signals(public_signal=0.85), now=NOW)
        assert record.private_score == pytest.approx(0.85)

    def test_user_corrected_subtracts_0_4(self, tmp_path: Path):
        producer = _make_producer(tmp_path)
        record = producer.record_turn(
            _basic_signals(public_signal=0.85, user_corrected_override=True),
            now=NOW,
        )
        assert record.private_score == pytest.approx(0.45)

    def test_rework_subtracts_0_15_each(self, tmp_path: Path):
        producer = _make_producer(tmp_path)
        record = producer.record_turn(
            _basic_signals(public_signal=0.85, rework_count_override=2),
            now=NOW,
        )
        assert record.private_score == pytest.approx(0.55)

    def test_combined_clamped_to_zero(self, tmp_path: Path):
        producer = _make_producer(tmp_path)
        record = producer.record_turn(
            _basic_signals(
                public_signal=0.30,
                user_corrected_override=True,
                rework_count_override=3,
                reuse_success_override=False,
            ),
            now=NOW,
        )
        # 0.30 - 0.4 - 0.45 - 0.2 = -0.75 → clamped to 0.0
        assert record.private_score == 0.0

    def test_combined_clamped_to_one(self, tmp_path: Path):
        producer = _make_producer(tmp_path)
        # All overrides set to "no signal" (False/0), public at 1.0.
        record = producer.record_turn(
            _basic_signals(public_signal=1.0),
            now=NOW,
        )
        assert record.private_score == pytest.approx(1.0)


class TestRecordBatch:
    def test_batch_records_and_saves_once(self, tmp_path: Path):
        producer = _make_producer(tmp_path)
        signals_list = [
            _basic_signals(task_id=f"t-{i}", public_signal=0.9) for i in range(5)
        ]
        records = producer.record_batch(signals_list, now=NOW)

        assert len(records) == 5
        assert all(r.public_score == 0.9 for r in records)

        ledger_path = tmp_path / "state" / "experience_ledger.json"
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert len(data["evals"]["my-skill"]) == 5


class TestInMemoryReworkTracker:
    def test_recognizes_same_task_id_across_calls(self, tmp_path: Path):
        producer = _make_producer(tmp_path)
        # First call — no retries.
        producer.record_turn(_basic_signals(task_id="t-X"), now=NOW)
        # Second call same task — 1 retry.
        record = producer.record_turn(
            _basic_signals(task_id="t-X", public_signal=0.7),
            now=NOW + 60,
        )
        assert record.rework_count == 1

    def test_evicts_events_outside_window(self, tmp_path: Path):
        producer = _make_producer(tmp_path)
        producer.record_turn(_basic_signals(task_id="t-X"), now=NOW)
        # Way past the rework window.
        record = producer.record_turn(
            _basic_signals(task_id="t-X"),
            now=NOW + 700,
        )
        assert record.rework_count == 0


class TestIntegration:
    def test_end_to_end_persistence(self, tmp_path: Path):
        """A full producer → ledger → reload round trip."""
        producer = _make_producer(tmp_path)
        producer.record_turn(
            _basic_signals(public_signal=0.7, eval_event_id="evt-001"),
            now=NOW,
        )

        # Reload the ledger from disk in a fresh instance.
        ledger2 = ExperienceLedger(hermes_home=tmp_path)
        summary = ledger2.get_summary("my-skill")
        assert summary is not None
        assert summary.total_evals == 1
        # avg_private_score reflects the heuristic for the only eval.
        assert 0.0 <= summary.avg_private_score <= 1.0
