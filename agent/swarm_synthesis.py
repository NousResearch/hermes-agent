"""Deterministic parent synthesis for swarm jobs.

The synthesis layer is the parent-owned truth table: it decides which child
claims are supported by required evidence and what must be disclosed before the
operator presents a job as complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from agent.swarm_evidence import EvidencePacket, validate_evidence_packet
from agent.swarm_state import SwarmJob


@dataclass
class SwarmSynthesis:
    verified_claims: List[str] = field(default_factory=list)
    unverified_claims: List[str] = field(default_factory=list)
    blocked_tasks: List[str] = field(default_factory=list)
    weak_tasks: List[str] = field(default_factory=list)
    missing_evidence: Dict[str, int] = field(default_factory=dict)
    unresolved: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    safe_to_present_complete: bool = True
    user_disclosure: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified_claims": list(self.verified_claims),
            "unverified_claims": list(self.unverified_claims),
            "blocked_tasks": list(self.blocked_tasks),
            "weak_tasks": list(self.weak_tasks),
            "missing_evidence": dict(self.missing_evidence),
            "unresolved": list(self.unresolved),
            "risks": list(self.risks),
            "safe_to_present_complete": self.safe_to_present_complete,
            "user_disclosure": self.user_disclosure,
        }


def synthesize_swarm_result(job: SwarmJob, *, persist: bool = False) -> SwarmSynthesis:
    requirements = job.routing_plan.evidence_requirements if job.routing_plan else []
    synthesis = SwarmSynthesis()

    for task in job.tasks:
        result = task.result or {}
        packet = EvidencePacket.from_result(result)
        validation = validate_evidence_packet(packet, requirements)

        if task.status in {"blocked", "awaiting_permission"} or task.permission_required:
            synthesis.blocked_tasks.append(task.task_id)
        if isinstance(result, dict) and result.get("weak_output"):
            synthesis.weak_tasks.append(task.task_id)

        synthesis.unresolved.extend(packet.unresolved)
        synthesis.risks.extend(packet.risks)
        for kind in validation.missing_required_kinds:
            synthesis.missing_evidence[kind] = synthesis.missing_evidence.get(kind, 0) + 1

        claims = packet.claims or ([packet.summary] if packet.summary else [])
        if validation.passed:
            synthesis.verified_claims.extend(claims)
        else:
            synthesis.unverified_claims.extend(claims)

    incomplete = bool(
        synthesis.unverified_claims
        or synthesis.blocked_tasks
        or synthesis.weak_tasks
        or synthesis.missing_evidence
        or job.status in {"failed", "partially_completed", "awaiting_permission", "cancelled"}
    )
    synthesis.safe_to_present_complete = not incomplete
    synthesis.user_disclosure = _build_disclosure(synthesis)
    if persist:
        job.metadata["swarm_synthesis"] = synthesis.to_dict()
    return synthesis


def _build_disclosure(synthesis: SwarmSynthesis) -> str:
    if synthesis.safe_to_present_complete:
        return "All delegated claims are supported by required evidence."
    parts: List[str] = []
    if synthesis.missing_evidence:
        missing = ", ".join(f"{kind} x{count}" for kind, count in sorted(synthesis.missing_evidence.items()))
        parts.append(f"Missing required evidence: {missing}.")
    if synthesis.unverified_claims:
        parts.append(f"Unverified claims: {len(synthesis.unverified_claims)}.")
    if synthesis.weak_tasks:
        parts.append(f"Weak child outputs: {', '.join(synthesis.weak_tasks)}.")
    if synthesis.blocked_tasks:
        parts.append(f"Blocked tasks: {', '.join(synthesis.blocked_tasks)}.")
    return " ".join(parts) or "Swarm result is not safe to present as complete."


__all__ = ["SwarmSynthesis", "synthesize_swarm_result"]
