"""Hermes² — Outer Loop self-improvement engineer (skeleton).

Inspired by AIDE²'s outer-loop agent: this cron-driven engineer periodically
reads the Experience Ledger, identifies skills needing improvement, proposes
mutations, validates them through the Eval Harness, and only retains changes
that improve the private score.

⚠️  STUB IMPLEMENTATION WARNING ⚠️
The execution paths that *produce* eval results and *apply* mutations are
intentional stubs that raise ``NotImplementedError`` until Phases 3 and 4:

- ``_run_validation_eval`` (Phase 3): runs the eval prompt against Hermes
  via auxiliary_client + isolated chat session. Currently raises.
- ``_apply_mutation`` (Phase 4): writes a new SKILL.md based on LLM-generated
  content. Currently raises to prevent corrupting user SKILL.md files with
  hard-coded text appendages (the previous string-append implementation was
  actively harmful — it would add noise like "## Validation Steps" sections
  to every skill regardless of what the skill does).

Working parts (fully tested):
- Cycle orchestration (load → worst-skills → propose → validate → accept/reject)
- Proposal generation from SkillSummary symptoms
- Budget enforcement
- Stale-fallback: if ``_apply_proposal`` finds no SKILL.md, skip
- Report serialization

Key principles from AIDE² (for the full Phase 3+4 implementation):
1. Read experience ledger → find worst performers
2. Propose 1-3 mutations (rewrite SKILL.md / add skill / adjust memory)
3. Validate via eval harness with private score
4. Only retain if private score improves AND cost doesn't exceed budget
5. ~90% rejection rate (strict evaluation)

Usage (Phase 3+4):
    # Create cron job (every Sunday 2am)
    hermes cron create --prompt "Run Hermes² self-improvement cycle" \\
        --schedule "0 2 * * 0" --skills "hermes-agent"

    # Or run manually
    from agent.hermes_squared import HermesSquaredEngine
    engine = HermesSquaredEngine()
    report = await engine.run_improvement_cycle()
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.eval_harness import EvalHarness
from agent.experience_ledger import ExperienceLedger, SkillEval, SkillSummary

logger = logging.getLogger(__name__)


@dataclass
class ImprovementProposal:
    """A proposed improvement to a skill or config."""

    proposal_id: str
    skill_id: str
    proposal_type: str  # rewrite_skill/add_skill/adjust_memory/new_config
    description: str
    changes: Dict[str, Any] = field(default_factory=dict)
    expected_private_score: float = 0.0
    current_private_score: float = 0.0
    status: str = "proposed"  # proposed/validating/accepted/rejected
    validation_result: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "skill_id": self.skill_id,
            "proposal_type": self.proposal_type,
            "description": self.description,
            "changes": self.changes,
            "expected_private_score": self.expected_private_score,
            "current_private_score": self.current_private_score,
            "status": self.status,
            "validation_result": self.validation_result,
            "created_at": self.created_at,
        }


@dataclass
class EvolutionReport:
    """Report from one improvement cycle."""

    cycle_id: str
    timestamp: float
    skills_reviewed: int = 0
    proposals_made: int = 0
    proposals_accepted: int = 0
    proposals_rejected: int = 0
    rejection_rate: float = 0.0
    total_cost_usd: float = 0.0
    duration_sec: float = 0.0
    proposals: List[ImprovementProposal] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp,
            "skills_reviewed": self.skills_reviewed,
            "proposals_made": self.proposals_made,
            "proposals_accepted": self.proposals_accepted,
            "proposals_rejected": self.proposals_rejected,
            "rejection_rate": self.rejection_rate,
            "total_cost_usd": self.total_cost_usd,
            "duration_sec": self.duration_sec,
            "proposals": [p.to_dict() for p in self.proposals],
            "summary": self.summary,
        }


class HermesSquaredEngine:
    """Outer-loop self-improvement engine.

    Implements AIDE²'s double-loop structure:
    - Inner loop = normal Hermes operation (running skills/tasks)
    - Outer loop = this engine (improving the inner loop)

    Cycle:
    1. Read Experience Ledger → find worst skills
    2. For each bad skill, propose improvement(s)
    3. Run eval to validate
    4. Accept if private score improves + cost within budget
    5. Generate evolution report
    """

    def __init__(
        self,
        hermes_home: Optional[Path] = None,
        max_proposals_per_cycle: int = 3,
        improvement_budget_usd: float = 5.0,
        min_private_score_threshold: float = 0.6,
    ):
        self.hermes_home = hermes_home or Path.home() / ".hermes"
        self.max_proposals = max_proposals_per_cycle
        self.budget = improvement_budget_usd
        self.min_score_threshold = min_private_score_threshold

        self.ledger = ExperienceLedger(hermes_home=self.hermes_home)
        self.eval_harness = EvalHarness(
            hermes_home=self.hermes_home,
            ledger=self.ledger,
        )
        self.evals_dir = self.hermes_home / "evals"
        self.reports_dir = self.evals_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    async def run_improvement_cycle(self) -> EvolutionReport:
        """Execute one full improvement cycle."""
        cycle_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        logger.info(
            "Hermes²: starting improvement cycle %s",
            cycle_id,
        )

        report = EvolutionReport(
            cycle_id=cycle_id,
            timestamp=start_time,
        )

        # Step 1: Read experience ledger
        self.eval_harness.load_evals()

        # Step 2: Find worst skills
        worst_skills = self.ledger.get_worst_skills(self.max_proposals)
        needing_improvement = self.ledger.get_skills_needing_improvement()

        # Combine and deduplicate
        candidates = {}
        for s in worst_skills + needing_improvement:
            if s.skill_id not in candidates:
                candidates[s.skill_id] = s

        report.skills_reviewed = len(candidates)
        logger.info(
            "Hermes²: found %d candidate skills for improvement",
            report.skills_reviewed,
        )

        # Step 3: Generate proposals
        proposals = []
        for skill_id, summary in candidates.items():
            if summary.avg_private_score >= self.min_score_threshold:
                continue  # Already good enough

            prop = await self._generate_proposal(skill_id, summary)
            if prop:
                proposals.append(prop)
                report.proposals_made += 1

            if report.proposals_made >= self.max_proposals:
                break

        report.proposals = proposals
        logger.info(
            "Hermes²: generated %d improvement proposals",
            report.proposals_made,
        )

        # Step 4: Validate proposals through eval harness
        for prop in proposals:
            prop.status = "validating"
            prop.validation_result = await self._validate_proposal(prop)

            # Check if improvement is real
            if prop.validation_result and prop.validation_result.get("improved", False):
                prop.status = "accepted"
                report.proposals_accepted += 1
                await self._apply_proposal(prop)
            else:
                prop.status = "rejected"
                report.proposals_rejected += 1

            report.total_cost_usd += prop.validation_result.get("cost_usd", 0.0)

            # Check budget
            if report.total_cost_usd > self.budget:
                logger.warning(
                    "Hermes²: budget exceeded (%.2f > %.2f), stopping cycle",
                    report.total_cost_usd,
                    self.budget,
                )
                break

        # Step 5: Calculate rejection rate (AIDE² reference: ~90%)
        if report.proposals_made > 0:
            report.rejection_rate = report.proposals_rejected / report.proposals_made

        report.duration_sec = time.time() - start_time
        report.summary = self._generate_summary(report)

        # Save report
        self._save_report(report)

        logger.info(
            "Hermes²: cycle %s complete — %d/%d accepted, rejection rate=%.0f%%, cost=$%.2f",
            cycle_id,
            report.proposals_accepted,
            report.proposals_made,
            report.rejection_rate * 100,
            report.total_cost_usd,
        )

        return report

    async def _generate_proposal(
        self,
        skill_id: str,
        summary: SkillSummary,
    ) -> Optional[ImprovementProposal]:
        """Generate an improvement proposal for a skill."""
        proposal_id = str(uuid.uuid4())[:8]

        # Determine improvement strategy based on symptoms
        if summary.avg_public_score > summary.avg_private_score + 0.3:
            # Reward hacking detected — need stricter validation
            strategy = "add_validation"
            description = (
                f"Skill '{skill_id}' shows reward hacking pattern "
                f"(public={summary.avg_public_score:.2f} vs private={summary.avg_private_score:.2f}). "
                f"Adding stricter private validation checks."
            )
        elif summary.user_correction_rate > 0.5:
            # User frequently corrects — skill instructions unclear
            strategy = "rewrite_skill"
            description = (
                f"Skill '{skill_id}' has high correction rate "
                f"({summary.user_correction_rate:.0%}). "
                f"Rewriting SKILL.md for clarity."
            )
        elif summary.success_rate < 0.5:
            # Low success rate — fundamental issues
            strategy = "fundamental_rewrite"
            description = (
                f"Skill '{skill_id}' has low success rate "
                f"({summary.success_rate:.0%}). "
                f"Complete rewrite needed."
            )
        else:
            # General improvement
            strategy = "optimize"
            description = (
                f"Skill '{skill_id}' private score is low "
                f"({summary.avg_private_score:.2f}). "
                f"Optimizing for better performance."
            )

        # Expected improvement (optimistic but realistic)
        expected_score = min(summary.avg_private_score + 0.15, 0.95)

        return ImprovementProposal(
            proposal_id=proposal_id,
            skill_id=skill_id,
            proposal_type=strategy,
            description=description,
            changes={"strategy": strategy},
            expected_private_score=expected_score,
            current_private_score=summary.avg_private_score,
        )

    async def _validate_proposal(
        self,
        proposal: ImprovementProposal,
    ) -> Dict[str, Any]:
        """Validate a proposal by running it through eval harness."""
        skill_id = proposal.skill_id
        strategy = proposal.changes.get("strategy", "optimize")

        # Apply the proposed change temporarily
        skill_dir = self.hermes_home / "skills" / skill_id
        skill_file = skill_dir / "SKILL.md"

        if not skill_file.exists():
            return {
                "improved": False,
                "reason": "Skill file not found",
                "cost_usd": 0.0,
            }

        # Save original content
        original_content = skill_file.read_text(encoding="utf-8")

        try:
            # Apply mutation
            mutated_content = self._apply_mutation(
                original_content,
                strategy,
                skill_id,
            )
            skill_file.write_text(mutated_content, encoding="utf-8")

            # Run eval
            eval_id = f"validate-{proposal.proposal_id}"
            result = await self._run_validation_eval(eval_id, skill_id)

            # Compare scores
            improved = result.get("private_score", 0.0) > proposal.current_private_score

            return {
                "improved": improved,
                "original_private_score": proposal.current_private_score,
                "new_private_score": result.get("private_score", 0.0),
                "cost_usd": result.get("cost_usd", 0.0),
                "eval_success": result.get("success", False),
            }

        except Exception as e:
            return {
                "improved": False,
                "reason": str(e),
                "cost_usd": 0.0,
            }
        finally:
            # Restore original content (proposal not yet accepted)
            skill_file.write_text(original_content, encoding="utf-8")

    def _apply_mutation(
        self,
        content: str,
        strategy: str,
        skill_id: str,
    ) -> str:
        """Generate a new SKILL.md from ``content`` given ``strategy``.

        ⚠️  STUB: raises ``NotImplementedError`` until Phase 4. The real
        implementation will call an LLM (via ``auxiliary_client``) with
        the current SKILL.md, the skill's evaluation summary, and the
        proposed strategy, then return the LLM's full new SKILL.md content
        for the caller to apply via file_ops V4A patch.

        The previous implementation appending hard-coded markdown
        sections ("## Validation Steps", "## Performance Notes", etc.)
        was **actively harmful** — it polluted every skill with the same
        generic noise regardless of what the skill does, and the
        ``rewrite_skill`` variant even mangled headings. Stubbing here
        until LLM-driven mutation is real.
        """
        raise NotImplementedError(
            f"HermesSquaredEngine._apply_mutation is a stub until Phase 4. "
            f"Strategy {strategy!r} on skill {skill_id!r} would be applied "
            f"via LLM-generated content; until then this raises to prevent "
            f"corrupting the user's SKILL.md. See docs/aide-squared-roadmap.md."
        )

    async def _run_validation_eval(
        self,
        eval_id: str,
        skill_id: str,
    ) -> Dict[str, Any]:
        """Run a validation eval for a specific skill.

        ⚠️  STUB: raises ``NotImplementedError`` until Phase 3. The real
        implementation will call ``EvalHarness.run_eval`` (with the
        helper wired through ``auxiliary_client`` in Phase 3) and return
        the actual measured private_score and cost.

        Tests and callers that need a non-stub result should monkeypatch
        this method.
        """
        raise NotImplementedError(
            f"HermesSquaredEngine._run_validation_eval is a stub until "
            f"Phase 3. Cannot validate proposal for skill {skill_id!r} "
            f"(eval {eval_id!r}) without a real EvalHarness execution "
            f"path. See docs/aide-squared-roadmap.md."
        )

    async def _apply_proposal(self, proposal: ImprovementProposal) -> None:
        """Apply an accepted proposal permanently to the user's SKILL.md.

        Stubbed until Phase 4. The real implementation will back up the
        current SKILL.md to ``<skill_dir>/SKILL.md.bak`` before mutation,
        call ``_apply_mutation`` to get LLM-generated content, write it
        back, and roll back from the backup if anything fails.

        Until Phase 4, ``_apply_mutation`` raises ``NotImplementedError``,
        which we catch and log so an accidentally-triggered apply (e.g.
        from a direct API call) does not corrupt SKILL.md or leave
        orphan backup files.
        """
        skill_dir = self.hermes_home / "skills" / proposal.skill_id
        skill_file = skill_dir / "SKILL.md"

        if not skill_file.exists():
            return

        try:
            # Apply the mutation permanently
            content = skill_file.read_text(encoding="utf-8")
            mutated = self._apply_mutation(
                content,
                proposal.changes.get("strategy", "optimize"),
                proposal.skill_id,
            )
            skill_file.write_text(mutated, encoding="utf-8")

            # Record in ledger with lineage
            self.ledger.record_eval(
                SkillEval(
                    skill_id=proposal.skill_id,
                    eval_event_id=f"evolved-{proposal.proposal_id}",
                    task_family="self_improvement",
                    public_score=proposal.expected_private_score,
                    private_score=proposal.expected_private_score,
                    cost_usd=proposal.validation_result.get("cost_usd", 0.0)
                    if proposal.validation_result
                    else 0.0,
                    outcome="success",
                    lineage=proposal.proposal_id,
                )
            )
            self.ledger.save()

            logger.info(
                "Hermes²: applied improvement to %s (proposal %s)",
                proposal.skill_id,
                proposal.proposal_id,
            )
        except NotImplementedError as e:
            logger.warning(
                "Hermes²: cannot apply proposal %s for skill %s — "
                "_apply_mutation is a stub until Phase 4. %s",
                proposal.proposal_id,
                proposal.skill_id,
                e,
            )
            proposal.status = "rejected_stub"
        except Exception as e:
            logger.error(
                "Hermes²: failed to apply proposal %s for skill %s: %s",
                proposal.proposal_id,
                proposal.skill_id,
                e,
            )
            proposal.status = "apply_failed"

    def _generate_summary(self, report: EvolutionReport) -> str:
        """Generate a human-readable summary of the cycle."""
        lines = [
            f"Hermes² Evolution Report (Cycle {report.cycle_id[:8]})",
            f"=" * 50,
            f"Skills reviewed: {report.skills_reviewed}",
            f"Proposals made: {report.proposals_made}",
            f"Proposals accepted: {report.proposals_accepted}",
            f"Proposals rejected: {report.proposals_rejected}",
            f"Rejection rate: {report.rejection_rate:.0%}",
            f"Total cost: ${report.total_cost_usd:.2f}",
            f"Duration: {report.duration_sec:.1f}s",
        ]

        if report.proposals:
            lines.append("\nProposals:")
            for prop in report.proposals:
                status_icon = "✅" if prop.status == "accepted" else "❌"
                lines.append(
                    f"  {status_icon} {prop.skill_id}: {prop.proposal_type} "
                    f"({prop.current_private_score:.2f} → {prop.expected_private_score:.2f})"
                )

        return "\n".join(lines)

    def _save_report(self, report: EvolutionReport) -> None:
        """Save evolution report to disk."""
        report_file = self.reports_dir / f"cycle-{report.cycle_id}.json"
        report_file.write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8",
        )
        logger.info("Hermes²: saved report to %s", report_file)
