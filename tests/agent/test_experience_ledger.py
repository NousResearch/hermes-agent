"""Tests for AIDE² P0-1: Experience Ledger."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.experience_ledger import ExperienceLedger, SkillEval, SkillSummary


# ========================================================================
# SkillEval tests
# ========================================================================

class TestSkillEval:
    def test_roundtrip(self):
        eval_record = SkillEval(
            skill_id="github-pr-workflow",
            eval_event_id="evt-001",
            task_family="coding",
            public_score=0.9,
            private_score=0.7,
            cost_usd=0.15,
            outcome="success",
        )
        d = eval_record.to_dict()
        restored = SkillEval.from_dict(d)
        assert restored.skill_id == "github-pr-workflow"
        assert restored.public_score == 0.9
        assert restored.private_score == 0.7

    def test_defaults(self):
        ev = SkillEval(
            skill_id="test", eval_event_id="x", task_family="general"
        )
        assert ev.public_score == 0.0
        assert ev.private_score == 0.0
        assert ev.cost_usd == 0.0
        assert ev.outcome == "unknown"
        assert ev.lineage == ""


# ========================================================================
# ExperienceLedger tests
# ========================================================================

class TestExperienceLedger:
    def _make_ledger(self, tmp_path: Path):
        return ExperienceLedger(
            hermes_home=tmp_path,
            max_history_per_skill=10,
        )

    def test_empty_ledger(self, tmp_path):
        ledger = self._make_ledger(tmp_path)
        assert ledger.total_evals == 0
        assert ledger.skill_count == 0

    def test_record_eval(self, tmp_path):
        ledger = self._make_ledger(tmp_path)
        ledger.record_eval(SkillEval(
            skill_id="test-skill",
            eval_event_id="evt-001",
            task_family="coding",
            public_score=0.8,
            private_score=0.6,
            cost_usd=0.10,
            outcome="success",
        ))
        assert ledger.total_evals == 1
        assert ledger.skill_count == 1

    def test_summary_computation(self, tmp_path):
        ledger = self._make_ledger(tmp_path)
        for i in range(5):
            ledger.record_eval(SkillEval(
                skill_id="my-skill",
                eval_event_id=f"evt-{i:03d}",
                task_family="coding",
                public_score=0.8 + i * 0.05,
                private_score=0.5 + i * 0.1,
                cost_usd=0.10 + i * 0.02,
                outcome="success" if i < 4 else "partial",
                tokens_in=1000 + i * 100,
                tokens_out=500 + i * 50,
            ))

        summary = ledger.get_summary("my-skill")
        assert summary is not None
        assert summary.total_evals == 5
        assert summary.success_rate == 0.8  # 4/5
        assert summary.avg_private_score == pytest.approx(0.7, abs=0.01)

    def test_save_and_load(self, tmp_path):
        ledger = self._make_ledger(tmp_path)
        ledger.record_eval(SkillEval(
            skill_id="persisted-skill",
            eval_event_id="evt-save",
            task_family="research",
            public_score=0.9,
            private_score=0.85,
            cost_usd=0.25,
            outcome="success",
        ))
        ledger.save()

        # New ledger instance
        ledger2 = self._make_ledger(tmp_path)
        assert ledger2.skill_count == 1
        summary = ledger2.get_summary("persisted-skill")
        assert summary is not None
        assert summary.avg_private_score == 0.85

    def test_stale_detection(self, tmp_path):
        ledger = self._make_ledger(tmp_path)
        # Low private score + old = stale
        old_time = time.time() - 60 * 86400  # 60 days ago
        ledger.record_eval(SkillEval(
            skill_id="old-skill",
            eval_event_id="evt-old",
            task_family="general",
            public_score=0.3,
            private_score=0.2,
            outcome="failure",
        ))
        # Override the created_at to simulate old eval
        ledger._evals["old-skill"][0].created_at = old_time

        summary = ledger.get_summary("old-skill")
        assert summary is not None
        assert summary.is_stale

    def test_needs_improvement(self, tmp_path):
        ledger = self._make_ledger(tmp_path)
        # 3 evals with low private score = needs improvement
        for i in range(3):
            ledger.record_eval(SkillEval(
                skill_id="bad-skill",
                eval_event_id=f"evt-bad-{i}",
                task_family="coding",
                public_score=0.9,  # Agent thinks it's great!
                private_score=0.3,  # But it's actually bad
                outcome="partial",
            ))

        summary = ledger.get_summary("bad-skill")
        assert summary is not None
        assert summary.needs_improvement
        # Public/private gap indicates potential reward hacking
        assert summary.avg_public_score > summary.avg_private_score

    def test_get_top_and_worst(self, tmp_path):
        ledger = self._make_ledger(tmp_path)
        for skill, score in [("good", 0.9), ("ok", 0.5), ("bad", 0.2)]:
            ledger.record_eval(SkillEval(
                skill_id=skill,
                eval_event_id=f"evt-{skill}",
                task_family="general",
                private_score=score,
                outcome="success" if score > 0.5 else "partial",
            ))

        top = ledger.get_top_skills(2)
        assert top[0].skill_id == "good"
        assert top[1].skill_id == "ok"

        worst = ledger.get_worst_skills(1)
        assert worst[0].skill_id == "bad"

    def test_user_correction_tracking(self, tmp_path):
        ledger = self._make_ledger(tmp_path)
        ledger.record_eval(SkillEval(
            skill_id="correction-skill",
            eval_event_id="evt-corr",
            task_family="coding",
            public_score=1.0,
            private_score=0.5,
            user_corrected=True,
            rework_count=2,
            outcome="partial",
        ))

        summary = ledger.get_summary("correction-skill")
        assert summary is not None
        assert summary.user_correction_rate == 1.0

    def test_history_trimming(self, tmp_path):
        ledger = ExperienceLedger(
            hermes_home=tmp_path,
        )
        ledger.max_history = 3
        for i in range(5):
            ledger.record_eval(SkillEval(
                skill_id="hot-skill",
                eval_event_id=f"evt-{i}",
                task_family="general",
                private_score=0.5 + i * 0.1,
            ))

        assert len(ledger._evals["hot-skill"]) == 3  # Trimmed
        assert ledger.total_evals == 3

    def test_get_all_summaries(self, tmp_path):
        ledger = self._make_ledger(tmp_path)
        for i in range(3):
            ledger.record_eval(SkillEval(
                skill_id=f"skill-{i}",
                eval_event_id=f"evt-{i}",
                task_family="general",
                private_score=0.3 + i * 0.3,
            ))

        summaries = ledger.get_all_summaries()
        assert len(summaries) == 3
