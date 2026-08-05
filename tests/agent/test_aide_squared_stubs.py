"""Tests for the stub state of AIDE² execution paths.

Phase 1 marked all execution paths as ``NotImplementedError`` stubs so
that no fake results enter the Experience Ledger. These tests pin that
contract:

- ``run_eval`` / ``run_all_evals`` report stub state and skip ledger
  recording when no execution is injected.
- ``DelegationEvolution._dispatch_strategy`` / ``_fork_strategy`` raise.
- ``HermesSquaredEngine._run_validation_eval`` / ``_apply_mutation`` raise,
  and the cycle handles that gracefully (proposal rejected, no SKILL.md
  written).

Once Phase 3 and Phase 4 land, replace each ``test_*_stub_*`` test with a
real-execution test that monkeypatches the real implementation.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent.delegation_evolution import DelegationEvolution, EvolutionResult
from agent.eval_harness import EvalHarness
from agent.experience_ledger import ExperienceLedger, SkillEval
from agent.hermes_squared import HermesSquaredEngine


class TestEvalHarnessStub:
    def test_simulate_task_execution_raises(self, tmp_path: Path):
        h = EvalHarness(hermes_home=tmp_path)
        ev = h._evals.get("file-ops-batch")  # populated by default
        if ev is None:
            h.load_evals()
            ev = list(h.get_evals().values())[0]

        with pytest.raises(NotImplementedError) as exc:
            h._simulate_task_execution(ev)
        assert "Phase 3" in str(exc.value)

    def test_run_llm_judge_raises(self, tmp_path: Path):
        from agent.eval_harness import EvalResult

        h = EvalHarness(hermes_home=tmp_path)
        ev = list(h.get_evals().values())[0] if h.get_evals() else None
        if ev is None:
            h.load_evals()
            ev = list(h.get_evals().values())[0]

        with pytest.raises(NotImplementedError) as exc:
            h._run_llm_judge(ev, EvalResult(eval_id=ev.id, skill_id="", success=False))
        assert "Phase 3" in str(exc.value)

    def test_run_eval_reports_stub(self, tmp_path: Path):
        h = EvalHarness(hermes_home=tmp_path)
        h.load_evals()
        result = h.run_eval("file-ops-batch")

        assert result.not_implemented is True
        assert result.success is False
        assert "stub until Phase 3" in result.error
        # Ledger MUST stay clean when execution is stubbed.
        assert h.ledger.total_evals == 0

    def test_run_all_evals_reports_stub(self, tmp_path: Path):
        h = EvalHarness(hermes_home=tmp_path)
        h.load_evals()
        results = h.run_all_evals()

        assert len(results) == len(h.get_evals())
        for r in results.values():
            assert r.not_implemented is True
            assert r.success is False
        assert h.ledger.total_evals == 0


class TestDelegationEvolutionStub:
    def test_dispatch_strategy_raises(self, tmp_path: Path):
        de = DelegationEvolution(hermes_home=tmp_path)
        with pytest.raises(NotImplementedError) as exc:
            asyncio.run(de._dispatch_strategy("aggressive", "goal", None, "task-1"))
        assert "Phase 3" in str(exc.value)

    def test_fork_strategy_raises(self, tmp_path: Path):
        from agent.delegation_evolution import StrategyResult

        de = DelegationEvolution(hermes_home=tmp_path)
        best = StrategyResult(strategy="conservative", score=0.7, lineage_id="t-c")
        with pytest.raises(NotImplementedError) as exc:
            asyncio.run(de._fork_strategy(best, "goal", None, "task-1"))
        assert "Phase 3" in str(exc.value)

    def test_bandit_selection_works_without_dispatch(self, tmp_path: Path):
        """Bandit / fork *algorithm* is functional; only dispatch is stubbed.

        We exercise _select_strategies with no history to verify the
        algorithm path is independent of the stubbed dispatch path.
        """
        de = DelegationEvolution(hermes_home=tmp_path)
        strategies = de._select_strategies(max_agents=3)
        # Should fall back to defaults when no history.
        assert strategies == ["aggressive", "conservative", "creative"]


class TestHermesSquaredStub:
    def test_apply_mutation_raises_and_does_not_corrupt_skill(self, tmp_path: Path):
        # Set up a fake skill SKILL.md so _apply_proposal finds a file.
        skills_dir = tmp_path / "skills" / "demo-skill"
        skills_dir.mkdir(parents=True)
        skill_file = skills_dir / "SKILL.md"
        original = "# demo\n\noriginal content\n"
        skill_file.write_text(original, encoding="utf-8")

        engine = HermesSquaredEngine(hermes_home=tmp_path)
        from agent.hermes_squared import ImprovementProposal

        proposal = ImprovementProposal(
            proposal_id="p1",
            skill_id="demo-skill",
            proposal_type="optimize",
            description="test",
            changes={"strategy": "optimize"},
            expected_private_score=0.7,
            current_private_score=0.5,
        )

        # _apply_proposal must catch NotImplementedError and NOT touch
        # SKILL.md.
        asyncio.run(engine._apply_proposal(proposal))

        # SKILL.md untouched.
        assert skill_file.read_text(encoding="utf-8") == original
        assert proposal.status == "rejected_stub"

    def test_run_validation_eval_raises(self, tmp_path: Path):
        engine = HermesSquaredEngine(hermes_home=tmp_path)
        with pytest.raises(NotImplementedError) as exc:
            asyncio.run(engine._run_validation_eval("eval-1", "demo-skill"))
        assert "Phase 3" in str(exc.value)

    def test_validate_proposal_returns_failure_on_stub(self, tmp_path: Path):
        skills_dir = tmp_path / "skills" / "demo-skill"
        skills_dir.mkdir(parents=True)
        skill_file = skills_dir / "SKILL.md"
        original = "# demo\noriginal\n"
        skill_file.write_text(original, encoding="utf-8")

        engine = HermesSquaredEngine(hermes_home=tmp_path)
        from agent.hermes_squared import ImprovementProposal

        proposal = ImprovementProposal(
            proposal_id="p1",
            skill_id="demo-skill",
            proposal_type="optimize",
            description="test",
            changes={"strategy": "optimize"},
            expected_private_score=0.7,
            current_private_score=0.5,
        )

        result = asyncio.run(engine._validate_proposal(proposal))

        # Stub: validation fails gracefully.
        assert result["improved"] is False
        # Either "Skill file not found" or the stub error message is fine;
        # the important invariants are improved=False and SKILL.md intact.
        assert "cost_usd" in result
        assert skill_file.read_text(encoding="utf-8") == original
