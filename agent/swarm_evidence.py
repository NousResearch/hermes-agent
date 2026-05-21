"""Structured evidence packets for swarm child outputs.

This module stays deterministic and side-effect free. It parses child result
payloads into a small evidence contract and validates that required evidence
kinds are actually present with kind-specific fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from agent.swarm_state import EvidenceRequirement


@dataclass
class EvidenceItem:
    kind: str
    description: str = ""
    command: str = ""
    output: str = ""
    url: str = ""
    path: str = ""
    approved_by: str = ""
    approval_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "EvidenceItem":
        if isinstance(value, EvidenceItem):
            return value
        if isinstance(value, str):
            return cls(kind="artifact", description=value)
        if not isinstance(value, dict):
            return cls(kind="artifact", description=repr(value))
        return cls(
            kind=str(value.get("kind") or value.get("type") or "artifact"),
            description=str(value.get("description") or value.get("summary") or value.get("text") or ""),
            command=str(value.get("command") or ""),
            output=str(value.get("output") or value.get("stdout") or value.get("result") or ""),
            url=str(value.get("url") or value.get("source") or ""),
            path=str(value.get("path") or value.get("file") or value.get("artifact") or ""),
            approved_by=str(value.get("approved_by") or value.get("approver") or ""),
            approval_id=str(value.get("approval_id") or value.get("permission_id") or ""),
            metadata={k: v for k, v in value.items() if k not in {"kind", "type", "description", "summary", "text", "command", "output", "stdout", "result", "url", "source", "path", "file", "artifact", "approved_by", "approver", "approval_id", "permission_id"}},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "description": self.description,
            "command": self.command,
            "output": self.output,
            "url": self.url,
            "path": self.path,
            "approved_by": self.approved_by,
            "approval_id": self.approval_id,
            "metadata": dict(self.metadata),
        }


@dataclass
class EvidencePacket:
    summary: str = ""
    claims: List[str] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    confidence: str = "unknown"
    side_effects_requested: List[str] = field(default_factory=list)
    side_effects_performed: List[str] = field(default_factory=list)

    @classmethod
    def from_result(cls, result: Any) -> "EvidencePacket":
        if isinstance(result, EvidencePacket):
            return result
        if isinstance(result, str):
            return cls(summary=result)
        if not isinstance(result, dict):
            return cls(summary=repr(result))
        raw_evidence = result.get("evidence") or []
        if isinstance(raw_evidence, (str, dict)):
            raw_evidence = [raw_evidence]
        return cls(
            summary=str(result.get("summary") or result.get("content") or result.get("message") or ""),
            claims=_string_list(result.get("claims") or []),
            evidence=[EvidenceItem.from_value(item) for item in raw_evidence if item is not None],
            assumptions=_string_list(result.get("assumptions") or []),
            unresolved=_string_list(result.get("unresolved") or result.get("open_issues") or []),
            risks=_string_list(result.get("risks") or []),
            confidence=str(result.get("confidence") or "unknown"),
            side_effects_requested=_string_list(result.get("side_effects_requested") or []),
            side_effects_performed=_string_list(result.get("side_effects_performed") or []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "claims": list(self.claims),
            "evidence": [item.to_dict() for item in self.evidence],
            "assumptions": list(self.assumptions),
            "unresolved": list(self.unresolved),
            "risks": list(self.risks),
            "confidence": self.confidence,
            "side_effects_requested": list(self.side_effects_requested),
            "side_effects_performed": list(self.side_effects_performed),
        }


@dataclass
class EvidenceValidationResult:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    missing_required_kinds: List[str] = field(default_factory=list)
    satisfied_kinds: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "missing_required_kinds": list(self.missing_required_kinds),
            "satisfied_kinds": list(self.satisfied_kinds),
            "unsupported_claims": list(self.unsupported_claims),
        }


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    return [str(value)]


def _requirement_kind(requirement: Any) -> str:
    if isinstance(requirement, EvidenceRequirement):
        return requirement.kind
    if isinstance(requirement, dict):
        return str(requirement.get("kind") or "")
    return str(getattr(requirement, "kind", "") or "")


def _requirement_required(requirement: Any) -> bool:
    if isinstance(requirement, EvidenceRequirement):
        return requirement.required
    if isinstance(requirement, dict):
        return bool(requirement.get("required", True))
    return bool(getattr(requirement, "required", True))


def _item_satisfies_kind(item: EvidenceItem, kind: str) -> bool:
    actual = item.kind.lower().strip()
    expected = kind.lower().strip()
    if actual != expected:
        return False
    text = " ".join([item.description, item.command, item.output, item.url, item.path, item.approved_by, item.approval_id]).lower()
    if expected == "citation":
        return bool(item.url or item.metadata.get("title") or "http" in text or "source" in text)
    if expected == "test":
        return bool(item.command and item.output and any(marker in text for marker in ("pass", "fail", "pytest", "test", "lint", "typecheck")))
    if expected == "command_output":
        return bool(item.command and item.output)
    if expected == "dry_run":
        return bool((item.command or item.output or item.description) and any(marker in text for marker in ("dry-run", "dry run", "payload", "would", "no-op", "contract")))
    if expected == "artifact":
        return bool(item.path or item.url or item.description)
    if expected == "human_approval":
        # Child output cannot prove human approval. Parent/job permission state must
        # verify approvals in a later live execution layer.
        return False
    return bool(text.strip())


def validate_evidence_packet(packet: EvidencePacket | Dict[str, Any], requirements: Iterable[Any]) -> EvidenceValidationResult:
    packet = EvidencePacket.from_result(packet)
    required_kinds = []
    for requirement in requirements or []:
        kind = _requirement_kind(requirement)
        if kind and _requirement_required(requirement):
            required_kinds.append(kind)

    satisfied: List[str] = []
    missing: List[str] = []
    reasons: List[str] = []
    for kind in required_kinds:
        if any(_item_satisfies_kind(item, kind) for item in packet.evidence):
            if kind not in satisfied:
                satisfied.append(kind)
        else:
            missing.append(kind)
            reasons.append(f"missing_{kind}")

    unsupported_claims = list(packet.claims) if missing and packet.claims else []
    if packet.side_effects_performed and "human_approval" in missing:
        reasons.append("side_effects_without_approval")

    return EvidenceValidationResult(
        passed=not missing and not (packet.side_effects_performed and "human_approval" in missing),
        reasons=reasons,
        missing_required_kinds=missing,
        satisfied_kinds=satisfied,
        unsupported_claims=unsupported_claims,
    )


__all__ = [
    "EvidenceItem",
    "EvidencePacket",
    "EvidenceValidationResult",
    "validate_evidence_packet",
]
