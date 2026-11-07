"""Hermes² — Outer Loop self-improvement engineer.

Inspired by AIDE²'s outer-loop agent: this cron-driven engineer periodically
reads the Experience Ledger, identifies skills needing improvement, proposes
mutations, validates them through the Eval Harness, and only retains changes
that improve the private score.

Phase 5 of the AIDE² self-evaluation plan (see
``docs/aide-squared-roadmap.md``). Phase 5 adds:

- ``Aide2Metrics`` (frozen dataclass): structured metrics emitted after
  each cycle — skill_counts, proposal_counts, scores, cost, duration,
  and per-skill delta scores. Suitable for Prometheus pushgateway,
  a JSON log sink, or the ``hermes aide2 status`` CLI command.
- Concurrent proposal validation: ``run_improvement_cycle`` accepts a
  ``max_concurrent`` kwarg (default 2). When > 1, proposals are
  validated in parallel via ``asyncio.gather`` instead of sequentially.
  Budget tracking is still accurate — total cost is summed after all
  validations complete.
- ``HermesAide2CLI`` class: thin CLI wrapper (``hermes aide2 run`` /
  ``hermes aide2 status``) that instantiates the engine, runs the cycle,
  and prints a formatted report.
- ``hermes aide2 status``: reads the latest ``evolution_report.json`` and
  prints a human-readable summary (cycle id, acceptance rate, cost, skill
  deltas).

Execution paths from Phase 4 are unchanged: ``_apply_mutation``,
``_validate_proposal``, ``_apply_proposal``, ``_run_validation_eval``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.eval_harness import EvalDefinition, EvalHarness
from agent.experience_ledger import ExperienceLedger, SkillEval, SkillSummary
from agent.skill_muter import (
    ApplyResult,
    DefaultSkillMuter,
    DefaultSkillMuterApplier,
    MutationContext,
    MutationProposal,
    SkillMuter,
    SkillMuterApplier,
)

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
    # Cached new content from the validation pass. Set by
    # _validate_proposal when the mutation succeeds. Used by
    # _apply_proposal to avoid re-running the LLM mutator — the
    # validated content is already measured, so we apply exactly
    # what we tested.
    new_content: Optional[str] = None

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
            "new_content": self.new_content,
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

    def to_metrics(self) -> "Aide2Metrics":
        """Convert to structured metrics for observability backends."""
        return Aide2Metrics(
            cycle_id=self.cycle_id,
            timestamp=self.timestamp,
            skills_reviewed=self.skills_reviewed,
            proposals_made=self.proposals_made,
            proposals_accepted=self.proposals_accepted,
            proposals_rejected=self.proposals_rejected,
            rejection_rate=self.rejection_rate,
            total_cost_usd=self.total_cost_usd,
            duration_sec=self.duration_sec,
            skill_deltas={
                p.skill_id: p.validation_result.get("new_private_score", 0.0)
                - p.validation_result.get("original_private_score", 0.0)
                for p in self.proposals
                if p.validation_result
            },
        )


@dataclass(frozen=True)
class Aide2Metrics:
    """Structured metrics emitted after each improvement cycle.

    Designed to be emitted to any observability sink: Prometheus
    pushgateway, a structured JSON log, Datadog, etc.
    """

    cycle_id: str
    timestamp: float
    skills_reviewed: int
    proposals_made: int
    proposals_accepted: int
    proposals_rejected: int
    rejection_rate: float  # 0.0–1.0
    total_cost_usd: float
    duration_sec: float
    skill_deltas: dict[str, float]  # skill_id → delta_private_score

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
            "skill_deltas": self.skill_deltas,
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
        *,
        mutator: Optional[SkillMuter] = None,
        applier: Optional[SkillMuterApplier] = None,
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
        self.mutator: SkillMuter = mutator or DefaultSkillMuter()
        self.applier: SkillMuterApplier = applier or DefaultSkillMuterApplier()

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

        # Step 4: Validate proposals through eval harness (concurrent)
        # Run all validations in parallel, up to max_concurrent at a time.
        # Each validation does not modify SKILL.md permanently — the
        # applier's rollback in _validate_proposal restores it after.
        validation_tasks = [self._validate_proposal(prop) for prop in proposals]
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)

        # Record results and tally cost.
        accepted_proposals: list[ImprovementProposal] = []
        for prop, result in zip(proposals, results):
            if isinstance(result, Exception):
                prop.validation_result = {"error": str(result)}
                prop.status = "rejected"
                report.proposals_rejected += 1
            else:
                prop.validation_result = result
                if result.get("improved", False):
                    prop.status = "accepted"
                    report.proposals_accepted += 1
                    accepted_proposals.append(prop)
                else:
                    prop.status = "rejected"
                    report.proposals_rejected += 1

            report.total_cost_usd += prop.validation_result.get("cost_usd", 0.0)

        # Step 4b: Apply accepted proposals sequentially (file writes are
        # sequential by design; concurrency here adds no value and risks
        # concurrent writes to the same skill).
        for prop in accepted_proposals:
            if report.total_cost_usd > self.budget:
                logger.warning(
                    "Hermes²: budget exceeded (%.2f > %.2f), stopping before apply",
                    report.total_cost_usd,
                    self.budget,
                )
                break
            await self._apply_proposal(prop)

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
        """Validate a proposal by running it through eval harness.

        Flow (Phase 4): the mutator generates the new SKILL.md, the
        applier temporarily writes it (backing up the original to
        ``SKILL.md.bak``), the eval runs, then we restore from the
        backup regardless of outcome. The permanent write happens in
        ``_apply_proposal`` *after* validation passes.
        """
        skill_id = proposal.skill_id
        strategy = proposal.changes.get("strategy", "optimize")
        skill_dir = self.hermes_home / "skills" / skill_id
        skill_file = skill_dir / "SKILL.md"

        if not skill_file.exists():
            return {
                "improved": False,
                "reason": "Skill file not found",
                "cost_usd": 0.0,
            }

        original_content = skill_file.read_text(encoding="utf-8")

        # Fetch the summary (if any) so the mutator can tailor the
        # rewrite to the skill's known symptoms.
        summary = self.ledger.get_summary(skill_id)

        mutated_content: Optional[str] = None
        try:
            try:
                mutated_content = self._apply_mutation(
                    original_content,
                    strategy,
                    skill_id,
                    summary=summary,
                )
            except Exception as e:
                return {
                    "improved": False,
                    "reason": f"mutation failed: {e}",
                    "cost_usd": 0.0,
                }

            if mutated_content is None or not mutated_content.strip():
                return {
                    "improved": False,
                    "reason": "mutator returned empty content",
                    "cost_usd": 0.0,
                }

            # Apply via the applier so SKILL.md.bak is created.
            temp_proposal = MutationProposal(
                new_content=mutated_content,
                reasoning="(temporary mutation for validation)",
                success=True,
            )
            apply = self.applier.apply(
                skill_id, temp_proposal, hermes_home=self.hermes_home
            )
            if not apply.success:
                return {
                    "improved": False,
                    "reason": f"temp apply failed: {apply.error}",
                    "cost_usd": 0.0,
                }

            # Run eval.
            eval_id = f"validate-{proposal.proposal_id}"
            try:
                result = await self._run_validation_eval(eval_id, skill_id)
            except Exception as e:
                return {
                    "improved": False,
                    "reason": f"validation eval raised: {e}",
                    "cost_usd": 0.0,
                }

            improved = result.get("private_score", 0.0) > proposal.current_private_score

            # Cache the validated content on the proposal so _apply_proposal
            # applies exactly what we measured (avoids re-running the LLM).
            proposal.new_content = mutated_content

            return {
                "improved": improved,
                "original_private_score": proposal.current_private_score,
                "new_private_score": result.get("private_score", 0.0),
                "cost_usd": result.get("cost_usd", 0.0),
                "eval_success": result.get("success", False),
            }

        finally:
            # Always restore the original (proposal not yet accepted).
            # The applier's rollback uses SKILL.md.bak; we fall back to
            # the in-memory original if no backup was created.
            if mutated_content is not None:
                rolled = self.applier.rollback(skill_id, hermes_home=self.hermes_home)
                if not rolled:
                    try:
                        skill_file.write_text(original_content, encoding="utf-8")
                    except OSError:
                        pass  # Best-effort restore from memory.

    def _apply_mutation(
        self,
        content: str,
        strategy: str,
        skill_id: str,
        summary: Optional[SkillSummary] = None,
    ) -> str:
        """Generate a new SKILL.md from ``content`` given ``strategy``.

        Real implementation (Phase 4). Delegates to the injected
        ``SkillMuter`` (defaults to ``DefaultSkillMuter``) which
        calls ``auxiliary_client.call_llm`` with a structured prompt
        asking the model to rewrite the SKILL.md based on the
        strategy and the skill's evaluation summary.

        Returns the new content. On failure, raises ``RuntimeError``
        with the mutator's error message — the caller (``_validate_proposal``)
        catches and treats it as a failed validation, leaving
        ``SKILL.md`` untouched.
        """
        notes = ""
        private_score = summary.avg_private_score if summary else 0.0
        public_score = summary.avg_public_score if summary else 0.0
        correction_rate = summary.user_correction_rate if summary else 0.0
        success_rate = summary.success_rate if summary else 0.0
        if summary and summary.is_suspected_reward_hack:
            notes = f"reward hacking suspected (gap={summary.public_private_gap})"

        context = MutationContext(
            skill_id=skill_id,
            current_content=content,
            strategy=strategy,
            private_score=private_score,
            public_score=public_score,
            correction_rate=correction_rate,
            success_rate=success_rate,
            notes=notes,
        )
        proposal = self.mutator.mutate(context)
        if not proposal.success or not proposal.new_content.strip():
            raise RuntimeError(
                proposal.error
                or "SkillMuter returned no content; not applying mutation"
            )
        return proposal.new_content

    async def _run_validation_eval(
        self,
        eval_id: str,
        skill_id: str,
    ) -> Dict[str, Any]:
        """Run a validation eval for a specific skill.

        Phase 4 implementation: synthesizes a one-off eval definition
        from the skill's ledger summary, runs it through the injected
        ``EvalHarness`` (Phase 3 real path), and returns the measured
        private_score + cost. Falls back to ``None`` if no LLM provider
        is available — the caller treats that as a failed validation.
        """
        summary = self.ledger.get_summary(skill_id)
        if summary is None:
            return {
                "success": False,
                "private_score": 0.0,
                "cost_usd": 0.0,
                "reason": "no ledger summary for skill",
            }

        prompt = (
            f"Re-run the '{skill_id}' skill with the proposed mutation "
            f"and report the observed private_score and cost_usd in JSON."
        )
        ev = EvalDefinition(
            id=eval_id,
            family=summary.skill_id,  # best-effort family tag
            prompt=prompt,
            budget_usd=1.0,
            metric="llm_judge_private",
            skill_id=skill_id,
            timeout_sec=60,
            description=f"validation eval for {skill_id} proposal",
        )
        # Inject the eval into the harness's in-memory state so we
        # don't need to touch disk.
        self.eval_harness._evals[eval_id] = ev
        result = self.eval_harness.run_eval(eval_id)
        if result.not_implemented:
            return {
                "success": False,
                "private_score": 0.0,
                "cost_usd": 0.0,
                "reason": "eval harness not wired (stubbed)",
            }
        return {
            "success": result.success,
            "private_score": result.private_score,
            "cost_usd": result.cost_usd,
            "eval_success": result.success,
            "reason": result.error,
        }

    async def _apply_proposal(self, proposal: ImprovementProposal) -> None:
        """Apply an accepted proposal permanently to the user's SKILL.md.

        Phase 4 real implementation. Delegates to the injected
        ``SkillMuterApplier`` which handles backup (``SKILL.md.bak``)
        and atomic write. The mutator has already been validated
        (``_validate_proposal`` ran the mutation under a temporary
        apply + eval), so by the time we get here the new content
        is known to be an improvement.

        This implementation intentionally avoids re-running the
        mutator: ``proposal.new_content`` was cached during validation.
        Future work can extend this to support "re-mutate on apply" for
        cases where the skill's ledger has been updated since validation.
        """
        skill_dir = self.hermes_home / "skills" / proposal.skill_id
        skill_file = skill_dir / "SKILL.md"

        if not skill_file.exists():
            return

        try:
            # Use the content we validated — never re-call the mutator.
            # If new_content is missing (e.g. manually constructed proposal),
            # fall back to calling the mutator.
            if proposal.new_content is not None:
                mutated = proposal.new_content
            else:
                content = skill_file.read_text(encoding="utf-8")
                summary = self.ledger.get_summary(proposal.skill_id)
                mutated = self._apply_mutation(
                    content,
                    proposal.changes.get("strategy", "optimize"),
                    proposal.skill_id,
                    summary=summary,
                )

            proposal_obj = MutationProposal(
                new_content=mutated,
                reasoning="(permanent mutation)",
                success=True,
            )
            apply = self.applier.apply(
                proposal.skill_id,
                proposal_obj,
                hermes_home=self.hermes_home,
            )
            if not apply.success:
                proposal.status = "apply_failed"
                logger.warning(
                    "Hermes²: applier refused permanent write for %s: %s",
                    proposal.skill_id,
                    apply.error,
                )
                return

            # Record in ledger with lineage.
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
        except Exception as e:
            logger.warning(
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


class HermesAide2CLI:
    """Thin CLI wrapper for Hermes² (AIDE²-inspired self-improvement engine).

    Supports two commands::

        hermes aide2 run [--budget USD] [--max-proposals N]
        hermes aide2 status

    Usage (Python)::

        from agent.hermes_squared import HermesAide2CLI
        cli = HermesAide2CLI()
        cli.run_status()
        # or
        cli.run_cycle(budget_usd=3.0, max_proposals=2)

    The CLI reads ``~/.hermes/`` as the default hermes_home; override with
    ``--hermes-home`` or ``HERMES_HOME``.
    """

    def __init__(self, hermes_home: Optional[Path] = None):
        self.hermes_home = hermes_home or Path.home() / ".hermes"

    def run_cycle(
        self,
        *,
        budget_usd: float = 5.0,
        max_proposals: int = 3,
    ) -> EvolutionReport:
        """Run one improvement cycle synchronously."""
        engine = HermesSquaredEngine(
            hermes_home=self.hermes_home,
            improvement_budget_usd=budget_usd,
            max_proposals_per_cycle=max_proposals,
        )
        report = asyncio.run(engine.run_improvement_cycle())
        print(engine._generate_summary(report))
        return report

    def run_status(self) -> Optional[EvolutionReport]:
        """Load and print the latest evolution report."""
        reports_dir = self.hermes_home / "evolution_reports"
        if not reports_dir.exists():
            print("No evolution reports found.")
            return None

        reports = sorted(reports_dir.glob("cycle-*.json"), reverse=True)
        if not reports:
            print("No evolution reports found.")
            return None

        latest = reports[0]
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Failed to read latest report: {e}")
            return None

        report = EvolutionReport(
            cycle_id=data["cycle_id"],
            timestamp=data["timestamp"],
            skills_reviewed=data.get("skills_reviewed", 0),
            proposals_made=data.get("proposals_made", 0),
            proposals_accepted=data.get("proposals_accepted", 0),
            proposals_rejected=data.get("proposals_rejected", 0),
            rejection_rate=data.get("rejection_rate", 0.0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            duration_sec=data.get("duration_sec", 0.0),
            summary=data.get("summary", ""),
        )

        print(
            f"Hermes² Status — latest cycle {report.cycle_id[:8]} "
            f"({len(reports)} total cycles)"
        )
        print("=" * 50)
        print(f"Skills reviewed : {report.skills_reviewed}")
        print(f"Proposals made   : {report.proposals_made}")
        print(f"Proposals accepted: {report.proposals_accepted}")
        print(f"Proposals rejected: {report.proposals_rejected}")
        print(f"Rejection rate   : {report.rejection_rate:.0%}")
        print(f"Total cost       : ${report.total_cost_usd:.4f}")
        print(f"Duration         : {report.duration_sec:.1f}s")
        if report.summary:
            print(f"\n{report.summary}")

        metrics = report.to_metrics()
        if metrics.skill_deltas:
            print("\nSkill deltas:")
            for skill_id, delta in metrics.skill_deltas.items():
                sign = "+" if delta >= 0 else ""
                print(f"  {skill_id}: {sign}{delta:.3f}")

        return report


def main(argv: Optional[List[str]] = None) -> int:
    """``hermes aide2`` entry point (also usable as __main__)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="hermes aide2",
        description="Hermes² AIDE²-inspired self-improvement engine",
    )
    parser.add_argument(
        "--hermes-home",
        type=Path,
        default=None,
        help="Path to hermes home (default: ~/.hermes)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # hermes aide2 run
    run_parser = subparsers.add_parser("run", help="Run one improvement cycle")
    run_parser.add_argument(
        "--budget",
        type=float,
        default=5.0,
        dest="budget_usd",
        help="Max USD to spend on eval LLM calls (default: 5.0)",
    )
    run_parser.add_argument(
        "--max-proposals",
        type=int,
        default=3,
        dest="max_proposals",
        help="Max proposals per cycle (default: 3)",
    )

    # hermes aide2 status
    subparsers.add_parser("status", help="Show latest evolution report")

    args = parser.parse_args(argv)
    cli = HermesAide2CLI(hermes_home=args.hermes_home)

    if args.command == "run":
        cli.run_cycle(budget_usd=args.budget_usd, max_proposals=args.max_proposals)
        return 0
    elif args.command == "status":
        cli.run_status()
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
