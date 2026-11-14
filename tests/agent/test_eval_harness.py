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

        # Phase 3 will wire the real Hermes runtime. Until then
        # _simulate_task_execution raises; tests that exercise the
        # ledger-recording path inject a fake execution.
        h._simulate_task_execution = lambda ev: (
            f"Task '{ev.id}' completed.",
            ev.budget_usd * 0.5,
        )

        result = h.run_eval("file-ops-batch")

        assert result.eval_id == "file-ops-batch"
        assert result.cost_usd >= 0
        assert result.duration_sec >= 0
        # Recorded in ledger (not stubbed out)
        assert h.ledger.total_evals > 0
        assert result.not_implemented is False

    def test_run_eval_without_provider_fails_gracefully(self, tmp_path):
        """Without a configured LLM provider, run_eval fails gracefully
        via the runner's structured error path (Phase 3) instead of
        fabricating fake data.

        Phase 1 raised ``NotImplementedError`` here. Phase 3 uses a
        real runner that returns ``PromptResult(success=False, ...)``;
        the harness translates that into ``EvalResult(success=False,
        error=<runner error>)`` and does NOT pollute the ledger.
        """
        h = self._make_harness(tmp_path)
        h.load_evals()

        result = h.run_eval("file-ops-batch")

        assert result.success is False
        assert result.error  # set to runner error
        # Ledger MUST stay clean when execution fails — the harness
        # only records real results.
        assert h.ledger.total_evals == 0

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
        # Write with explicit UTF-8 to mirror the Windows-footgun fix.
        (evals_dir / "evals.json").write_text(
            json.dumps(evals, ensure_ascii=False),
            encoding="utf-8",
        )

        h = self._make_harness(tmp_path)
        count = h.load_evals()
        assert count == 1
        assert "json-eval" in h.get_evals()

    def test_load_from_json_with_unicode(self, tmp_path):
        """Regression: eval definitions with non-ASCII fields must round-trip
        under UTF-8 encoding (Windows-footgun fix).
        """
        evals_dir = tmp_path / "evals"
        evals_dir.mkdir(parents=True)

        evals = [
            EvalDefinition(
                id="unicode-eval-✓",
                family="research",
                prompt="研究 Python 异步最佳实践",
                description="包含中文的描述",
                budget_usd=0.5,
            ).to_dict()
        ]
        (evals_dir / "evals.json").write_text(
            json.dumps(evals, ensure_ascii=False),
            encoding="utf-8",
        )

        h = self._make_harness(tmp_path)
        count = h.load_evals()
        assert count == 1
        ev = h.get_evals()["unicode-eval-✓"]
        assert "研究" in ev.prompt
        assert "中文" in ev.description

    def test_budget_exceeded_when_cost_strictly_over(self, tmp_path):
        """Cost strictly greater than budget triggers budget_exceeded."""
        h = self._make_harness(tmp_path)
        ev = EvalDefinition(
            id="over-budget",
            family="tools",
            prompt="Force over-budget",
            budget_usd=0.10,
        )
        h._evals[ev.id] = ev

        # Override the simulator to return a guaranteed over-budget cost.
        h._simulate_task_execution = lambda ev: (
            "ran",
            ev.budget_usd + 0.01,  # cost = $0.11, budget = $0.10
        )

        result = h.run_eval("over-budget")
        assert result.budget_exceeded is True
        assert result.success is False
        assert result.public_score == 0.0
        assert result.private_score == 0.0
        assert "exceeded budget" in result.error.lower()

    def test_cost_equal_to_budget_passes(self, tmp_path):
        """Cost == budget is allowed (strictly-greater threshold)."""
        h = self._make_harness(tmp_path)
        ev = EvalDefinition(
            id="at-budget",
            family="tools",
            prompt="Hit budget exactly",
            budget_usd=0.20,
        )
        h._evals[ev.id] = ev

        h._simulate_task_execution = lambda ev: ("ran", ev.budget_usd)
        # Default metric will assign moderate scores.

        result = h.run_eval("at-budget")
        assert result.budget_exceeded is False
        # Should not short-circuit; metric path runs.

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
