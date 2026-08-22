"""Deterministic mandatory-skill gates for explicit LAH workflow turns."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class GovernancePhase(str, Enum):
    UNCLASSIFIED = "UNCLASSIFIED"
    ROUTER_REQUIRED = "ROUTER_REQUIRED"
    ROUTER_PASSED = "ROUTER_PASSED"
    DECOMPOSER_REQUIRED = "DECOMPOSER_REQUIRED"
    GOVERNANCE_PREREQUISITES_PASSED = "GOVERNANCE_PREREQUISITES_PASSED"
    ROUTER_FAILED = "ROUTER_FAILED"
    DECOMPOSER_FAILED = "DECOMPOSER_FAILED"
    MANDATORY_SKILL_MISSING = "MANDATORY_SKILL_MISSING"
    MANDATORY_SKILL_RESOLUTION_FAILED = "MANDATORY_SKILL_RESOLUTION_FAILED"
    MANDATORY_SKILL_LOAD_FAILED = "MANDATORY_SKILL_LOAD_FAILED"
    MANDATORY_GATE_ORDER_INVALID = "MANDATORY_GATE_ORDER_INVALID"


MANDATORY_SKILLS = {
    GovernancePhase.ROUTER_REQUIRED: ("0", "lah-repo-router"),
    GovernancePhase.DECOMPOSER_REQUIRED: ("0.5", "mission-decomposer"),
}

# Keep harmless metadata needed to discover/load mandatory skills available.
GOVERNANCE_METADATA_TOOLS = frozenset(
    {"skill_view", "skills_list", "read_file", "search_files", "session_search"}
)


def classify_governed_mission(text: str | None) -> bool:
    """Recognize explicit LAH workflow selection, not generic repository work."""
    value = (text or "").lower()
    return bool(
        re.search(r"\blah-workflow(?:-small-model)?\b", value)
        or re.search(r"\bgoverned\s+lah\s+mission\b", value)
        or re.search(r"\bmission\s*:\s*lah_[a-z0-9_]+", value)
        or (
            re.search(r"\bmission\b", value)
            and re.search(r"\blah_[a-z0-9_]+\b", value)
            and re.search(r"\blah-repo-router\b|\bmission-decomposer\b", value)
        )
    )


@dataclass(frozen=True)
class GovernanceDecision:
    allowed: bool
    result: str = ""


@dataclass
class GovernedSkillState:
    governed: bool = False
    authority_valid: bool = True
    authority_errors: tuple[str, ...] = ()
    phase: GovernancePhase = GovernancePhase.UNCLASSIFIED
    failure_reason: str = ""
    observed: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        if self.governed and self.phase is GovernancePhase.UNCLASSIFIED:
            if not self.authority_valid:
                self.phase = GovernancePhase.MANDATORY_SKILL_RESOLUTION_FAILED
                self.failure_reason = "; ".join(self.authority_errors) or "runtime skill authority invalid"
            else:
                self.phase = GovernancePhase.ROUTER_REQUIRED

    @property
    def downstream_allowed(self) -> bool:
        return self.phase is GovernancePhase.GOVERNANCE_PREREQUISITES_PASSED

    @property
    def blocked(self) -> bool:
        return self.governed and not self.downstream_allowed

    def before_tool(self, tool_name: str, args: Mapping[str, Any] | None) -> GovernanceDecision:
        with self._lock:
            if not self.governed or self.downstream_allowed:
                return GovernanceDecision(True)

            if self.phase in {
                GovernancePhase.ROUTER_FAILED,
                GovernancePhase.DECOMPOSER_FAILED,
                GovernancePhase.MANDATORY_SKILL_MISSING,
                GovernancePhase.MANDATORY_SKILL_RESOLUTION_FAILED,
                GovernancePhase.MANDATORY_SKILL_LOAD_FAILED,
                GovernancePhase.MANDATORY_GATE_ORDER_INVALID,
            }:
                return GovernanceDecision(False, self._receipt("MANDATORY_GATE_FAILED"))

            expected = MANDATORY_SKILLS.get(self.phase)
            if expected is None:
                return GovernanceDecision(False, self._receipt("MANDATORY_GATE_ORDER_INVALID"))

            gate, expected_name = expected
            observed_name = args.get("name") if isinstance(args, Mapping) else None
            if tool_name == "skill_view" and observed_name == expected_name:
                return GovernanceDecision(True)

            if tool_name == "skill_view":
                self.failure_reason = (
                    f"expected canonical invocation_name '{expected_name}', "
                    f"observed '{observed_name or ''}'"
                )
                return GovernanceDecision(False, self._receipt("MANDATORY_GATE_ORDER_INVALID", gate, observed_name))

            self.failure_reason = f"downstream tool '{tool_name}' requested before Gate {gate}"
            return GovernanceDecision(False, self._receipt("GOVERNANCE_PREREQUISITE_REQUIRED", gate, tool_name))

    def observe_skill_result(self, requested_name: str, result: Mapping[str, Any] | None) -> None:
        with self._lock:
            if not self.governed or self.downstream_allowed:
                return
            expected = MANDATORY_SKILLS.get(self.phase)
            if expected is None or requested_name != expected[1]:
                return

            data = dict(result or {})
            success = data.get("success") is True
            resolved = data.get("skill_name") or data.get("name")
            self.observed.append({
                "gate": expected[0],
                "expected": expected[1],
                "requested": requested_name,
                "resolved": resolved,
                "success": success,
            })
            if not success:
                self.failure_reason = str(data.get("error") or "skill resolution/load failed")
                self.phase = (
                    GovernancePhase.ROUTER_FAILED
                    if expected[0] == "0"
                    else GovernancePhase.DECOMPOSER_FAILED
                )
                return
            if resolved and resolved != requested_name:
                self.failure_reason = f"resolved skill '{resolved}' differs from '{requested_name}'"
                self.phase = GovernancePhase.MANDATORY_SKILL_RESOLUTION_FAILED
                return
            self.phase = (
                GovernancePhase.DECOMPOSER_REQUIRED
                if expected[0] == "0"
                else GovernancePhase.GOVERNANCE_PREREQUISITES_PASSED
            )

    def _receipt(self, reason: str, gate: str | None = None, observed: Any = None) -> str:
        expected = MANDATORY_SKILLS.get(self.phase)
        gates = []
        if expected:
            gates.append({
                "gate": expected[0],
                "skill": expected[1],
                "status": "BLOCKED",
                "observed": observed,
                "reason": self.failure_reason or reason,
            })
        elif self.governed:
            failed_gate = "0.5" if self.phase is GovernancePhase.DECOMPOSER_FAILED else "0"
            failed_skill = "mission-decomposer" if failed_gate == "0.5" else "lah-repo-router"
            gates.append({
                "gate": failed_gate,
                "skill": failed_skill,
                "status": "BLOCKED",
                "observed": observed,
                "reason": self.failure_reason or reason,
            })
        for item in self.observed:
            gates.append({
                "gate": item["gate"],
                "skill": item["expected"],
                "status": "PASS" if item["success"] else "FAIL",
            })
        return json.dumps({
            "error": "governed_mission_blocked",
            "governance": {
                "phase": self.phase.value,
                "required_gates": gates,
                "downstream_execution_allowed": False,
                "reason": self.failure_reason or reason,
            },
        }, ensure_ascii=False)
