"""Tests for the worked-example turn wrapper."""

from __future__ import annotations

import json
from pathlib import Path

from agent.hermes_eval_hook import wrap_turn


class TestWrapTurn:
    def test_records_eval_and_returns_record(self, tmp_path: Path):
        record = wrap_turn(
            skill_id="my-skill",
            task_id="t-1",
            task_family="coding",
            hermes_home=tmp_path,
            public_signal=0.85,
            cost_usd=0.12,
            tokens_in=100,
            tokens_out=50,
            duration_sec=3.5,
            success=True,
        )
        assert record is not None
        assert record.skill_id == "my-skill"
        assert record.cost_usd == 0.12

        # Persisted to ledger.
        ledger_path = tmp_path / "state" / "experience_ledger.json"
        assert ledger_path.exists()

    def test_user_corrected_from_follow_up_messages(self, tmp_path: Path):
        record = wrap_turn(
            skill_id="my-skill",
            task_id="t-1",
            task_family="coding",
            hermes_home=tmp_path,
            public_signal=0.85,
            user_messages=["that's wrong, try again"],
        )
        assert record is not None
        assert record.user_corrected is True

    def test_does_not_raise_on_producer_error(self, tmp_path: Path, monkeypatch):
        """A broken producer must not break the caller."""
        from agent import skill_eval_producer as producer_mod

        def _broken_record_turn(self, signals, *, now=None):
            raise RuntimeError("ledger broken")

        monkeypatch.setattr(
            producer_mod.SkillEvalProducer,
            "record_turn",
            _broken_record_turn,
        )

        # wrap_turn swallows the exception and returns None.
        result = wrap_turn(
            skill_id="my-skill",
            task_id="t-1",
            task_family="coding",
            hermes_home=tmp_path,
            public_signal=0.85,
        )
        assert result is None

    def test_existing_producer_instance_is_reused(self, tmp_path: Path):
        """Passing an explicit producer avoids creating a second one."""
        from agent.skill_eval_producer import SkillEvalProducer

        producer = SkillEvalProducer(hermes_home=tmp_path)
        record = wrap_turn(
            skill_id="my-skill",
            task_id="t-1",
            task_family="coding",
            hermes_home=tmp_path,
            public_signal=0.7,
            producer=producer,
        )
        assert record is not None
        # Same producer's ledger should have one record.
        assert producer.ledger.total_evals == 1
