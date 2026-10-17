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
    """Phase 3 replaced the _simulate_task_execution / _run_llm_judge
    stubs with real implementations behind injectable runners/judges.
    These tests pin the *injection* contract: passing a fake runner
    lets callers exercise the harness without touching
    auxiliary_client.
    """

    def test_runner_injection(self, tmp_path: Path):
        """Passing an explicit runner to EvalHarness replaces the
        default. This is how Phase 1 stubs became Phase 3 real code.
        """
        from agent.eval_runner import EvalInvocation, PromptResult

        class _FakeRunner:
            def execute_prompt(self, invocation):
                return PromptResult(
                    text="<skipped>",
                    tokens_in=0,
                    tokens_out=0,
                    success=True,
                    model="fake",
                )

            def run_private_check(self, invocation):
                from agent.eval_runner import PrivateCheckResult

                return PrivateCheckResult(
                    exit_code=0,
                    stdout="",
                    stderr="",
                    duration_sec=0.01,
                    success=True,
                )

        h = EvalHarness(hermes_home=tmp_path, runner=_FakeRunner())
        assert isinstance(h.runner, _FakeRunner)

    def test_llm_judge_injection(self, tmp_path: Path):
        from agent.llm_judge import JudgeScore

        class _FakeJudge:
            def judge(self, prompt, response):
                return JudgeScore(score=80, reasoning="ok", success=True, model="fake")

        h = EvalHarness(hermes_home=tmp_path, judge=_FakeJudge())
        assert isinstance(h.judge, _FakeJudge)


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

        # Phase 4 replaces the stub with a real mutator. When no LLM
        # provider is configured, the engine surfaces the failure as
        # ``apply_failed`` rather than corrupting SKILL.md.
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

        asyncio.run(engine._apply_proposal(proposal))

        # SKILL.md untouched (mutator failed gracefully).
        assert skill_file.read_text(encoding="utf-8") == original
        assert proposal.status == "apply_failed"

    def test_run_validation_eval_handles_missing_provider(self, tmp_path: Path):
        """Phase 4 replaces the NotImplementedError stub with a real
        ``EvalHarness.run_eval`` call. Without a configured provider
        the call surfaces a structured failure rather than raising.
        """
        engine = HermesSquaredEngine(hermes_home=tmp_path)
        result = asyncio.run(engine._run_validation_eval("eval-1", "demo-skill"))
        # Without a configured provider, success=False with a reason.
        assert result["success"] is False
        assert result["private_score"] == 0.0

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
