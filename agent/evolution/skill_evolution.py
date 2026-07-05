"""Recursive Skill Evolution — skills auto-improve from failures. Each generation adds targeted procedures."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# How many failures before a skill gets patched
DEFAULT_PATCH_THRESHOLD = 2
# Max patch history per skill
MAX_PATCH_HISTORY = 20
# Max skill size (prevents unbounded growth)
MAX_SKILL_SIZE_BYTES = 20 * 1024  # 20KB


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SkillGeneration:
    """One generation of a skill's evolution."""
    generation: int
    created_at: str
    patch_description: str  # What was changed
    failure_evidence: str    # What failure triggered this patch
    skill_size_bytes: int
    success_count_after: int = 0
    failure_count_after: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "created_at": self.created_at,
            "patch_description": self.patch_description,
            "failure_evidence": self.failure_evidence,
            "skill_size_bytes": self.skill_size_bytes,
            "success_count_after": self.success_count_after,
            "failure_count_after": self.failure_count_after,
        }


@dataclass
class SkillEvolutionRecord:
    """Tracks a skill's evolution over time."""
    skill_name: str
    generations: List[SkillGeneration] = field(default_factory=list)
    total_successes: int = 0
    total_failures: int = 0
    total_patches: int = 0
    failures_since_last_patch: int = 0  # Reset after each patch
    created_at: str = ""
    last_updated_at: str = ""

    @property
    def current_generation(self) -> int:
        return len(self.generations)

    @property
    def effectiveness(self) -> float:
        """Success rate: what fraction of sessions with this skill succeeded?"""
        total = self.total_successes + self.total_failures
        return self.total_successes / total if total > 0 else 0.0

    @property
    def is_improving(self) -> bool:
        """Is the skill getting better over generations?"""
        if len(self.generations) < 2:
            return False
        recent = self.generations[-2:]  # Last 2 generations
        return recent[1].success_count_after > recent[0].success_count_after

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "generations": [g.to_dict() for g in self.generations],
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "total_patches": self.total_patches,
            "failures_since_last_patch": self.failures_since_last_patch,
            "current_generation": self.current_generation,
            "effectiveness": self.effectiveness,
            "is_improving": self.is_improving,
            "created_at": self.created_at,
            "last_updated_at": self.last_updated_at,
        }


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class SkillEvolutionTracker:
    """Tracks and recursively improves skills based on real outcomes.

    Thread-safe. Persists evolution state to disk. Integrates with
    auto-trigger and improvement proposer.
    """

    def __init__(self):
        self._records: Dict[str, SkillEvolutionRecord] = {}
        self._active_skills: set = set()  # Skills loaded this session
        self._session_failures: List[str] = []  # Failure types this session
        self._load()

    # ── Session lifecycle ──────────────────────────────────────────────

    def start_session(self, skills_loaded: List[str]) -> None:
        """Record which skills are active this session."""
        self._active_skills = set(skills_loaded)
        self._session_failures = []

    def record_failure(self, failure_type: str, description: str = "") -> None:
        """Record a failure that occurred despite active skills."""
        self._session_failures.append(f"{failure_type}: {description}"[:200])

    def end_session(self, was_successful: bool) -> List[str]:
        """Finalize session. Returns list of skills that should be patched."""
        patched = []

        for skill_name in self._active_skills:
            record = self._get_or_create_record(skill_name)

            if was_successful:
                record.total_successes += 1
                if record.generations:
                    record.generations[-1].success_count_after += 1
                record.failures_since_last_patch = 0  # Reset on success
            else:
                record.total_failures += 1
                record.failures_since_last_patch += 1
                if record.generations:
                    record.generations[-1].failure_count_after += 1

                # Check if skill needs patching
                if record.failures_since_last_patch >= DEFAULT_PATCH_THRESHOLD and self._session_failures:
                    patched_skill = self._patch_skill(skill_name, record)
                    if patched_skill:
                        patched.append(skill_name)

            record.last_updated_at = datetime.now(timezone.utc).isoformat()

        self._save()
        return patched

    # ── Skill patching ─────────────────────────────────────────────────

    def _patch_skill(self, skill_name: str, record: SkillEvolutionRecord) -> bool:
        """Apply a recursive improvement patch to a skill.

        This is the meta-level improvement: the skill gets better
        because it accumulated real failure evidence.
        """
        skill_path = self._skill_path(skill_name)
        if not skill_path.exists():
            return False

        content = skill_path.read_text()

        # Prevent unbounded growth
        if len(content) > MAX_SKILL_SIZE_BYTES:
            logger.debug("Skill %s at %d bytes, skipping patch (max %d)",
                        skill_name, len(content), MAX_SKILL_SIZE_BYTES)
            return False

        # Build targeted improvement patch
        new_content = self._build_patch(record, content)
        if new_content == content:
            return False  # No meaningful change

        # Gate through safety check — validate new content
        from agent.evolution.regression_gate import RegressionGate
        from agent.evolution.improvement_proposer import ImprovementProposal, ImprovementActionType

        proposal = ImprovementProposal(
            action_type=ImprovementActionType.SKILL_CREATE,
            target=skill_name,
            description=f"Recursive improvement: added failure case (gen {record.current_generation + 1})",
            rationale=f"Skill failed to prevent {len(self._session_failures)} failure(s). Adding specific cases.",
            content=new_content,
        )

        gate = RegressionGate()
        result = gate.evaluate(proposal)
        if not result.passed:
            logger.debug("Gate blocked recursive patch for %s: %s",
                        skill_name, result.failures)
            return False

        # Apply
        skill_path.write_text(new_content)

        # Record the generation
        record.generations.append(SkillGeneration(
            generation=record.current_generation + 1,
            created_at=datetime.now(timezone.utc).isoformat(),
            patch_description=f"Added {len(self._session_failures)} failure case(s)",
            failure_evidence="; ".join(self._session_failures[-3:]),
            skill_size_bytes=len(new_content),
        ))
        record.total_patches += 1
        record.failures_since_last_patch = 0  # Reset counter after patch

        logger.info("Skill %s evolved to gen %d (+%d bytes, %d failure cases)",
                    skill_name, record.current_generation,
                    len(new_content) - len(content), len(self._session_failures))
        return True

    def _build_patch(self, record: SkillEvolutionRecord, content: str) -> str:
        """Generate a TARGETED skill improvement — not just a failure log.

        SOTA approach: analyze failure patterns → generate specific procedure
        changes → add concrete verification commands → update triggers.
        """
        if not self._session_failures:
            return content

        # Analyze failure patterns
        failure_types = set()
        failure_details = []
        for f in self._session_failures[-5:]:
            parts = f.split(":", 1)
            ftype = parts[0].strip() if parts else "unknown"
            detail = parts[1].strip() if len(parts) > 1 else ""
            failure_types.add(ftype)
            if detail:
                failure_details.append(detail)

        patches = []

        # 1. Add concrete verification step for each failure type
        if "missing_verification" in failure_types:
            new_step = self._generate_verification_step(failure_details)
            if new_step and new_step not in content:
                # Insert into Procedure section
                if "## Procedure" in content:
                    content = content.replace(
                        "## Procedure",
                        "## Procedure\n\n" + new_step.strip()
                    )
                else:
                    content += f"\n\n## Procedure\n\n{new_step.strip()}\n"

        if "loop_detected" in failure_types:
            loop_guidance = self._generate_loop_guidance(failure_details)
            if loop_guidance and loop_guidance not in content:
                if "## How to Run" in content:
                    content = content.replace(
                        "## How to Run",
                        "## How to Run\n\n" + loop_guidance.strip()
                    )

        # 2. Update Pitfalls with actionable prevention (not just logs)
        prevention = self._generate_prevention(failure_types, failure_details)
        if prevention and prevention not in content:
            if "## Pitfalls" in content:
                content = content.replace(
                    "## Pitfalls",
                    "## Pitfalls\n\n" + prevention.strip()
                )
            else:
                content += f"\n\n## Pitfalls\n\n{prevention.strip()}\n"

        # 3. Add verification commands that catch these failures
        verify_cmd = self._generate_verification_commands(failure_types)
        if verify_cmd and "## Verification" in content:
            # Append to existing verification
            content += verify_cmd

        return content

    def _generate_verification_step(self, details: List[str]) -> str:
        """Generate a concrete procedure step based on failure details."""
        steps = []
        for d in details[:2]:
            if "exit code" in d.lower():
                steps.append(
                    "1. After running any command, check the exit code: "
                    "`echo $?` must return 0. If not, the command FAILED."
                )
            elif "file" in d.lower() or "output" in d.lower():
                steps.append(
                    "1. Verify every output file exists and is non-empty: "
                    "`test -s <file> && echo 'exists' || echo 'MISSING'`"
                )
            elif "permission" in d.lower():
                steps.append(
                    "1. Check file permissions before assuming success: "
                    "`ls -la <file>` — verify read/write access."
                )
        if not steps:
            steps.append(
                "1. Before declaring completion, run ALL verification "
                "commands. Check each result. Only proceed if ALL pass."
            )
        return "\n".join(steps)

    def _generate_loop_guidance(self, details: List[str]) -> str:
        """Generate anti-loop guidance from failure evidence."""
        return (
            "1. If any command fails, diagnose the error BEFORE retrying.\n"
            "2. Never retry the same command more than twice without changing something.\n"
            "3. If stuck, switch to a DIFFERENT approach — don't repeat."
        )

    def _generate_prevention(self, failure_types: set, details: List[str]) -> str:
        """Generate actionable prevention advice, not just failure logs."""
        lines = []
        for ftype in failure_types:
            if ftype == "missing_verification":
                lines.append(
                    "- **Missing verification**: Agent completed work without checking results. "
                    "Always run verification commands BEFORE declaring done. Check exit codes, "
                    "file existence, and output correctness."
                )
            elif ftype == "loop_detected":
                lines.append(
                    "- **Loop detection**: Agent repeated the same failing action. "
                    "After 2 identical failures, STOP. Diagnose the root cause. "
                    "Try a completely different approach."
                )
            elif ftype == "user_correction":
                lines.append(
                    "- **User correction**: Output was incorrect. Read the correction carefully. "
                    "Adjust the approach based on what the user said was wrong."
                )
            else:
                lines.append(
                    f"- **{ftype}**: Failure detected. Review the procedure above "
                    "and ensure all verification steps are followed."
                )
        return "\n".join(lines)

    def _generate_verification_commands(self, failure_types: set) -> str:
        """Generate concrete verification commands for detected failure types."""
        cmds = []
        if "missing_verification" in failure_types:
            cmds.append("# Verify: check exit code of last command\ntest $? -eq 0 && echo 'OK' || echo 'FAILED'")
        if "missing_verification" in failure_types:
            cmds.append("# Verify: check output files exist\nls -la <expected_output_files>")
        if not cmds:
            return ""
        return "\n\n" + "\n".join(cmds) + "\n"

    # ── Query API ──────────────────────────────────────────────────────

    def get_evolution(self, skill_name: str) -> Optional[SkillEvolutionRecord]:
        """Get the full evolution history of a skill."""
        return self._records.get(skill_name)

    def get_improving_skills(self) -> List[str]:
        """List skills that are measurably improving."""
        return [
            name for name, record in self._records.items()
            if record.is_improving
        ]

    def get_generation_summary(self) -> Dict[str, Any]:
        """Summary of all skill evolution across generations."""
        total_gens = sum(r.current_generation for r in self._records.values())
        total_patches = sum(r.total_patches for r in self._records.values())
        improving = len(self.get_improving_skills())

        # Calculate collective effectiveness
        total_successes = sum(r.total_successes for r in self._records.values())
        total_failures = sum(r.total_failures for r in self._records.values())
        total = total_successes + total_failures

        return {
            "skills_tracked": len(self._records),
            "total_generations": total_gens,
            "total_patches": total_patches,
            "skills_improving": improving,
            "collective_effectiveness": total_successes / total if total > 0 else 0.0,
            "total_sessions": total,
            "oldest_skill_generation": max(
                (r.current_generation for r in self._records.values()), default=0
            ),
        }

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_or_create_record(self, skill_name: str) -> SkillEvolutionRecord:
        if skill_name not in self._records:
            now = datetime.now(timezone.utc).isoformat()
            self._records[skill_name] = SkillEvolutionRecord(
                skill_name=skill_name,
                created_at=now,
                last_updated_at=now,
            )
        return self._records[skill_name]

    @staticmethod
    def _skill_path(skill_name: str) -> Path:
        return get_hermes_home() / "skills" / skill_name / "SKILL.md"

    # ── Persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        path = self._store_path()
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8-sig") as f:
                data = json.load(f)
            for name, rd in data.get("records", {}).items():
                record = SkillEvolutionRecord(
                    skill_name=name,
                    total_successes=rd.get("total_successes", 0),
                    total_failures=rd.get("total_failures", 0),
                    total_patches=rd.get("total_patches", 0),
                    created_at=rd.get("created_at", ""),
                    last_updated_at=rd.get("last_updated_at", ""),
                )
                record.generations = [
                    SkillGeneration(**g) for g in rd.get("generations", [])
                ]
                self._records[name] = record
        except Exception as e:
            logger.debug("Failed to load skill evolution: %s", e)

    def _save(self) -> None:
        path = self._store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "records": {
                        name: record.to_dict()
                        for name, record in self._records.items()
                    },
                }, f, indent=2, default=str)
        except Exception as e:
            logger.debug("Failed to save skill evolution: %s", e)

    def _store_path(self) -> Path:
        return get_hermes_home() / "evolution" / "skill_evolution.json"


# ── Singleton ──────────────────────────────────────────────────────────

_tracker: Optional[SkillEvolutionTracker] = None


def get_skill_evolution_tracker() -> SkillEvolutionTracker:
    global _tracker
    if _tracker is None:
        _tracker = SkillEvolutionTracker()
    return _tracker
