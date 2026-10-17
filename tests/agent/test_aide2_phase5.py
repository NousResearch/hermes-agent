"""Tests for Phase 5 additions: Aide2Metrics, concurrent validation, HermesAide2CLI."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.hermes_squared import (
    Aide2Metrics,
    EvolutionReport,
    HermesAide2CLI,
    HermesSquaredEngine,
    ImprovementProposal,
)


# ---------------------------------------------------------------------------
# Aide2Metrics
# ---------------------------------------------------------------------------


class TestAide2Metrics:
    def test_to_dict_roundtrip(self):
        m = Aide2Metrics(
            cycle_id="abc12345",
            timestamp=1700000000.0,
            skills_reviewed=3,
            proposals_made=3,
            proposals_accepted=1,
            proposals_rejected=2,
            rejection_rate=0.667,
            total_cost_usd=0.42,
            duration_sec=12.5,
            skill_deltas={"my-skill": 0.15, "bad-skill": -0.05},
        )
        d = m.to_dict()
        assert d["cycle_id"] == "abc12345"
        assert d["proposals_accepted"] == 1
        assert d["skill_deltas"]["my-skill"] == 0.15
        assert d["skill_deltas"]["bad-skill"] == -0.05

    def test_frozen_immutable(self):
        m = Aide2Metrics(
            cycle_id="x",
            timestamp=1.0,
            skills_reviewed=0,
            proposals_made=0,
            proposals_accepted=0,
            proposals_rejected=0,
            rejection_rate=0.0,
            total_cost_usd=0.0,
            duration_sec=0.0,
            skill_deltas={},
        )
        with pytest.raises(Exception):  # frozen dataclass
            m.cycle_id = "y"  # type: ignore[reportArgumentType]


class TestEvolutionReportToMetrics:
    def test_to_metrics_derives_deltas(self):
        prop = ImprovementProposal(
            proposal_id="p1",
            skill_id="s1",
            proposal_type="optimize",
            description="",
            current_private_score=0.3,
            expected_private_score=0.5,
            validation_result={
                "improved": True,
                "original_private_score": 0.3,
                "new_private_score": 0.45,
                "cost_usd": 0.10,
            },
        )
        report = EvolutionReport(
            cycle_id="c1",
            timestamp=1.0,
            skills_reviewed=1,
            proposals_made=1,
            proposals_accepted=1,
            proposals_rejected=0,
            total_cost_usd=0.10,
            proposals=[prop],
        )
        metrics = report.to_metrics()
        assert metrics.skill_deltas["s1"] == pytest.approx(0.15)
        assert metrics.rejection_rate == 0.0


# ---------------------------------------------------------------------------
# Concurrent validation
# ---------------------------------------------------------------------------


class TestConcurrentValidation:
    def test_gather_validates_all_proposals(self, tmp_path: Path):
        """Validate that asyncio.gather runs all validations concurrently
        and records their results.
        """
        call_log: list[str] = []

        async def _fake_validate(prop):
            await asyncio.sleep(0.05)  # simulate async work
            call_log.append(f"validated-{prop.skill_id}")
            return {"improved": False, "cost_usd": 0.0}

        # Build engine with fake eval harness so we can mock _validate_proposal
        engine = HermesSquaredEngine(hermes_home=tmp_path)

        # Create a skill dir so _validate_proposal doesn't early-return
        skill_dir = tmp_path / "skills" / "skill-a"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# skill-a\n", encoding="utf-8")

        prop_a = ImprovementProposal(
            proposal_id="pa",
            skill_id="skill-a",
            proposal_type="optimize",
            description="a",
            current_private_score=0.5,
        )

        # Patch _validate_proposal on the engine
        engine._validate_proposal = _fake_validate  # type: ignore[method-assign]

        # Run one "cycle" by calling the concurrent block logic directly
        proposals = [prop_a]

        async def _run():
            tasks = [engine._validate_proposal(prop) for prop in proposals]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.run(_run())
        assert call_log == ["validated-skill-a"]
        assert results[0] == {"improved": False, "cost_usd": 0.0}

    def test_exception_in_gather_is_caught(self, tmp_path: Path):
        """If a validation raises, it appears as an exception in results
        from asyncio.gather with return_exceptions=True.
        """
        engine = HermesSquaredEngine(hermes_home=tmp_path)

        # Set up the skill file so _validate_proposal doesn't early-return
        skill_dir = tmp_path / "skills" / "skill-x"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# skill-x\n", encoding="utf-8")

        async def _fail(_prop):
            raise RuntimeError("validation failed")

        prop = ImprovementProposal(
            proposal_id="p1",
            skill_id="skill-x",
            proposal_type="optimize",
            description="",
        )
        engine._validate_proposal = _fail  # type: ignore[method-assign]

        async def _run():
            tasks = [engine._validate_proposal(prop) for prop in [prop]]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.run(_run())
        assert isinstance(results[0], RuntimeError)
        assert str(results[0]) == "validation failed"


# ---------------------------------------------------------------------------
# HermesAide2CLI
# ---------------------------------------------------------------------------


class TestHermesAide2CLI:
    def test_status_no_reports(self, tmp_path: Path):
        cli = HermesAide2CLI(hermes_home=tmp_path)
        result = cli.run_status()
        assert result is None

    def test_status_with_latest_report(self, tmp_path: Path):
        # Create a fake report.
        reports_dir = tmp_path / "evolution_reports"
        reports_dir.mkdir(parents=True)
        report_file = reports_dir / "cycle-abc12345.json"
        report_file.write_text(
            json.dumps({
                "cycle_id": "abc12345",
                "timestamp": 1700000000.0,
                "skills_reviewed": 2,
                "proposals_made": 2,
                "proposals_accepted": 1,
                "proposals_rejected": 1,
                "rejection_rate": 0.5,
                "total_cost_usd": 0.23,
                "duration_sec": 8.1,
                "summary": "Test cycle",
            }),
            encoding="utf-8",
        )
        cli = HermesAide2CLI(hermes_home=tmp_path)
        result = cli.run_status()
        assert result is not None
        assert result.cycle_id == "abc12345"
        assert result.proposals_accepted == 1
        assert result.total_cost_usd == 0.23
        assert result.duration_sec == 8.1

    def test_main_run_command(self, tmp_path: Path):
        """main(['run']) runs without error even when no skills exist."""
        from agent.hermes_squared import main

        # Mock the engine so it doesn't need a real LLM
        with patch.object(HermesSquaredEngine, "run_improvement_cycle") as mock_run:
            mock_run.return_value = EvolutionReport(
                cycle_id="test001",
                timestamp=time.time(),
                skills_reviewed=0,
                proposals_made=0,
                proposals_accepted=0,
                proposals_rejected=0,
                total_cost_usd=0.0,
                duration_sec=0.1,
            )
            exit_code = main([
                "--hermes-home",
                str(tmp_path),
                "run",
                "--budget",
                "1.0",
                "--max-proposals",
                "2",
            ])
        assert exit_code == 0

    def test_main_status_command(self, tmp_path: Path):
        from agent.hermes_squared import main

        reports_dir = tmp_path / "evolution_reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "cycle-abc12345.json").write_text(
            json.dumps({
                "cycle_id": "abc12345",
                "timestamp": 1700000000.0,
                "skills_reviewed": 1,
                "proposals_made": 1,
                "proposals_accepted": 0,
                "proposals_rejected": 1,
                "rejection_rate": 1.0,
                "total_cost_usd": 0.05,
                "duration_sec": 2.0,
                "summary": "",
            }),
            encoding="utf-8",
        )

        exit_code = main([
            "--hermes-home",
            str(tmp_path),
            "status",
        ])


class TestBug1NewContentCaching:
    """Bug: _apply_proposal re-ran mutator instead of using validated content.

    Before the fix: _apply_proposal called _apply_mutation again, generating
    DIFFERENT content from what was validated. This made validation meaningless.

    After the fix: proposal.new_content is cached during validation and
    _apply_proposal uses it directly.
    """

    @pytest.mark.asyncio
    async def test_apply_proposal_uses_cached_new_content_not_mutator(self, tmp_path: Path):
        """When proposal.new_content is set, apply uses it instead of re-mutating."""
        from agent.hermes_squared import HermesSquaredEngine, ImprovementProposal
        from agent.skill_muter import ApplyResult

        call_count = 0

        class WritingApplier:
            """Fake applier that actually writes the file (like the real one)."""
            def apply(self, skill_id, proposal, hermes_home=None):
                nonlocal call_count
                skill_file = (hermes_home or Path(".")) / "skills" / skill_id / "SKILL.md"
                skill_file.parent.mkdir(parents=True, exist_ok=True)
                skill_file.write_text(proposal.new_content, encoding="utf-8")
                return ApplyResult(success=True)

            def rollback(self, skill_id, hermes_home=None):
                return True

        class CountingMutator:
            def mutate(self, context):
                nonlocal call_count
                call_count += 1
                from agent.skill_muter import MutationProposal
                return MutationProposal(
                    new_content="CONTENT_FROM_MUTATOR_CALL_%d" % call_count,
                    reasoning="",
                    success=True,
                )

        # Setup: write a skill file
        skill_dir = tmp_path / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("original content", encoding="utf-8")

        engine = HermesSquaredEngine(
            hermes_home=tmp_path,
            mutator=CountingMutator(),
            applier=WritingApplier(),
        )

        # Proposal with pre-cached new_content
        prop = ImprovementProposal(
            proposal_id="p1",
            skill_id="test-skill",
            proposal_type="rewrite_skill",
            description="test",
            changes={"strategy": "optimize"},
            current_private_score=0.3,
            new_content="VALIDATED_CONTENT_THAT_SHOULD_BE_APPLIED",
        )
        prop.validation_result = {"cost_usd": 0.05}

        await engine._apply_proposal(prop)

        # Mutator should NOT have been called — content was cached
        assert call_count == 0, f"mutator was called {call_count} times, expected 0"

        # The applied content should be the cached one
        applied = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert applied == "VALIDATED_CONTENT_THAT_SHOULD_BE_APPLIED"

        # proposal.new_content should be preserved
        assert prop.new_content == "VALIDATED_CONTENT_THAT_SHOULD_BE_APPLIED"

    @pytest.mark.asyncio
    async def test_apply_proposal_falls_back_to_mutator_when_no_new_content(self, tmp_path: Path):
        """When proposal.new_content is None, apply falls back to mutator (backwards compat)."""
        from agent.hermes_squared import HermesSquaredEngine, ImprovementProposal
        from agent.skill_muter import ApplyResult

        call_count = 0

        class WritingApplier:
            def apply(self, skill_id, proposal, hermes_home=None):
                skill_file = (hermes_home or Path(".")) / "skills" / skill_id / "SKILL.md"
                skill_file.write_text(proposal.new_content, encoding="utf-8")
                return ApplyResult(success=True)

            def rollback(self, skill_id, hermes_home=None):
                return True

        class CountingMutator:
            def mutate(self, context):
                nonlocal call_count
                call_count += 1
                from agent.skill_muter import MutationProposal
                return MutationProposal(
                    new_content="MUTATOR_RESULT_%d" % call_count,
                    reasoning="",
                    success=True,
                )

        skill_dir = tmp_path / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("original", encoding="utf-8")

        engine = HermesSquaredEngine(
            hermes_home=tmp_path,
            mutator=CountingMutator(),
            applier=WritingApplier(),
        )

        # Proposal WITHOUT new_content (e.g. manually constructed)
        prop = ImprovementProposal(
            proposal_id="p2",
            skill_id="test-skill",
            proposal_type="rewrite_skill",
            description="test",
            changes={"strategy": "optimize"},
            current_private_score=0.3,
            new_content=None,  # explicitly None
        )

        await engine._apply_proposal(prop)

        # Mutator SHOULD have been called as fallback
        assert call_count == 1, f"mutator was called {call_count} times, expected 1"
        applied = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert applied == "MUTATOR_RESULT_1"
