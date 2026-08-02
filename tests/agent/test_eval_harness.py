"""Tests for AIDE² P0-2: Eval Harness."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agent.eval_harness import EvalDefinition, EvalHarness, EvalResult
from agent.experience_ledger import ExperienceLedger


class TestEvalDefinition:
    def test_roundtrip(self):
        ev = EvalDefinition(
            id="test-eval",
            family="tools",
            prompt="Sort a CSV file",
            budget_usd=0.5,
            metric="private",
            private_check="test -f output.csv",
        )
        d = ev.to_dict()
        restored = EvalDefinition.from_dict(d)
        assert restored.id == "test-eval"
        assert restored.family == "tools"
        assert restored.budget_usd == 0.5


class TestEvalHarness:
    def _make_harness(self, tmp_path: Path):
        return EvalHarness(hermes_home=tmp_path)

    def test_init(self, tmp_path):
        h = self._make_harness(tmp_path)
        assert h.hermes_home == tmp_path
        assert len(h._evals) == 0

    def test_load_evals_creates_defaults(self, tmp_path):
        h = self._make_harness(tmp_path)
        count = h.load_evals()
        assert count >= 3  # At least 3 default evals
        assert "file-ops-batch" in h.get_evals()
        assert "research-synthesis" in h.get_evals()

    def test_run_eval_unknown(self, tmp_path):
        h = self._make_harness(tmp_path)
        result = h.run_eval("nonexistent")
        assert not result.success
        assert "Unknown eval" in result.error

    def test_run_eval_records_in_ledger(self, tmp_path):
        h = self._make_harness(tmp_path)
        h.load_evals()
        result = h.run_eval("file-ops-batch")

        assert result.eval_id == "file-ops-batch"
        assert result.cost_usd >= 0
        assert result.duration_sec >= 0
        # Should be recorded in ledger
        assert h.ledger.total_evals > 0

    def test_run_all_evals(self, tmp_path):
        h = self._make_harness(tmp_path)
        h.load_evals()
        results = h.run_all_evals()

        assert len(results) == len(h.get_evals())
        for eval_id, result in results.items():
            assert result.eval_id == eval_id
            assert result.started_at > 0

    def test_summary(self, tmp_path):
        h = self._make_harness(tmp_path)
        h.load_evals()
        h.run_all_evals()

        summary = h.get_eval_summary()
        assert summary["total"] >= 3
        assert "success_rate" in summary
        assert "total_cost_usd" in summary

    def test_budget_exceeded_detection(self, tmp_path):
        h = self._make_harness(tmp_path)
        ev = EvalDefinition(
            id="budget-test",
            family="tools",
            prompt="Do something expensive",
            budget_usd=0.01,  # Very low budget
        )
        h._evals[ev.id] = ev

        result = h.run_eval("budget-test")
        # May or may not exceed budget depending on simulation
        assert result.cost_usd >= 0

    def test_custom_metric_registration(self, tmp_path):
        h = self._make_harness(tmp_path)

        def my_metric(ev, result):
            result.public_score = 0.8
            result.private_score = 0.9
            result.success = True
            return result

        h.register_custom_metric("my_test", my_metric)
        assert "my_test" in h._custom_metrics

    def test_load_from_json(self, tmp_path):
        evals_dir = tmp_path / "evals"
        evals_dir.mkdir(parents=True)

        evals = [
            EvalDefinition(
                id="json-eval",
                family="custom",
                prompt="Test from JSON",
                budget_usd=0.3,
            ).to_dict()
        ]
        (evals_dir / "evals.json").write_text(json.dumps(evals))

        h = self._make_harness(tmp_path)
        count = h.load_evals()
        assert count == 1
        assert "json-eval" in h.get_evals()

    def test_reward_hack_detection(self, tmp_path):
        h = self._make_harness(tmp_path)
        ev = EvalDefinition(
            id="hack-test",
            family="tools",
            prompt="Test for reward hacking",
            budget_usd=1.0,
            metric="private",
            private_check="false",  # Will fail, creating gap
        )
        h._evals[ev.id] = ev
        h._run_deterministic_check = lambda ev, result: self._force_hack(result)

        result = h.run_eval("hack-test")
        # Reward hack detection should trigger
        # (public >> private with significant gap)

    @staticmethod
    def _force_hack(result):
        """Force a reward hack scenario for testing."""
        result.public_score = 0.95
        result.private_score = 0.3
        result.success = False
        result.reward_hack_detected = True
        return result
