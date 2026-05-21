"""Anti-illusion learning/weak-output helpers for swarm results.

MVP scope is intentionally small and deterministic: annotate weak outputs; do
not retry children or write durable memory from telemetry.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from agent.swarm_evidence import EvidencePacket, validate_evidence_packet

_VAGUE_SUCCESS_PHRASES = (
    "done",
    "fixed",
    "looks good",
    "should work",
    "seems fine",
    "all good",
    "ok",
)

_EVIDENCE_KEYS = ("evidence", "citations", "sources", "artifacts", "tests", "commands", "command_output", "dry_run", "approval")
_KIND_KEYS = {
    "citation": ("citations", "sources"),
    "test": ("tests", "commands", "command_output"),
    "command_output": ("command_output", "commands"),
    "dry_run": ("dry_run", "dry_run_output", "payload_contract"),
    "artifact": ("artifact", "artifacts", "path", "url", "file"),
    "human_approval": ("approval", "human_approval", "approved_by", "permission_id"),
}


def _has_evidence(result: Dict[str, Any]) -> bool:
    for key in _EVIDENCE_KEYS:
        value = result.get(key)
        if value:
            return True
    return False


def _requirement_kind(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("kind") or "")
    return str(getattr(item, "kind", "") or "")


def _requirement_required(item: Any) -> bool:
    if isinstance(item, dict):
        return bool(item.get("required", True))
    return bool(getattr(item, "required", True))


def _has_kind_evidence(result: Dict[str, Any], kind: str) -> bool:
    return any(result.get(key) for key in _KIND_KEYS.get(kind, (kind,)))


def detect_weak_output(result: Dict[str, Any], evidence_requirements: Iterable[Any] = ()) -> Dict[str, Any]:
    """Return weak-output annotation for a child result.

    Weak means the child result is not trustworthy enough for parent synthesis
    without review. This is not a failure signal by itself; it is an evidence
    insufficiency marker.
    """

    text = " ".join(str(result.get(key) or "") for key in ("summary", "content", "message", "status")).strip().lower()
    required_kinds: List[str] = [_requirement_kind(item) for item in evidence_requirements if _requirement_required(item)]
    reasons: List[str] = []

    if required_kinds and not _has_evidence(result):
        reasons.append("no_evidence")
    if text in _VAGUE_SUCCESS_PHRASES or any(phrase in text for phrase in ("looks good", "should work", "seems fine")):
        reasons.append("vague_success_language")
    for kind in required_kinds:
        validation = validate_evidence_packet(EvidencePacket.from_result(result), [item for item in evidence_requirements if _requirement_kind(item) == kind])
        if kind and not validation.passed:
            for reason in validation.reasons or [f"missing_{kind}"]:
                if reason not in reasons:
                    reasons.append(reason)

    return {
        "weak": bool(reasons),
        "reasons": reasons,
        "required_evidence": [kind for kind in required_kinds if kind],
        "recommended_status": "needs_review" if reasons else "completed",
    }


__all__ = ["detect_weak_output"]
