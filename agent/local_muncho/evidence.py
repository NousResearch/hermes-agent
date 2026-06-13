"""Deterministic evidence-first validation for Local Muncho responses."""

from __future__ import annotations

import re
from typing import Sequence

from agent.local_muncho.types import (
    EvidenceValidationResult,
    RuntimeContext,
    ToolEvidence,
)


REQUIRED_FRAME_FIELDS = (
    "VERDICT",
    "TL;DR",
    "CATEGORY",
    "EVIDENCE_CHECKED",
    "EVIDENCE_GAP",
    "STATUS",
    "NEXT_ACTION",
    "APPROVAL_NEEDED",
    "RISK",
)

_CLAIM_RE = re.compile(
    r"\b(sent|routed|tracked|fixed|approved|completed)\b",
    re.IGNORECASE,
)
_SECRET_RE = re.compile(
    r"("
    r"\b(?:access[_-]?token|auth[_-]?token|api[_-]?key|secret|password)\s*[:=]\s*['\"]?[^'\"\s]{8,}"
    r"|\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"
    r"|\bsk-[A-Za-z0-9]{16,}"
    r")",
    re.IGNORECASE,
)
_CARDISH_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def _has_required_field(text: str, field: str) -> bool:
    return bool(re.search(rf"(?m)^\s*{re.escape(field)}\s*:", text))


def _blocked_frame(reason: str, gaps: Sequence[str]) -> str:
    gap_text = "; ".join(gaps) if gaps else reason
    return "\n".join(
        [
            "VERDICT: BLOCKED",
            "TL;DR: Local Muncho runtime blocked the model text before delivery.",
            "CATEGORY: runtime_guard",
            "EVIDENCE_CHECKED: deterministic-validator",
            f"EVIDENCE_GAP: {gap_text}",
            "STATUS: blocked",
            "NEXT_ACTION: collect durable evidence or request operator approval",
            "APPROVAL_NEEDED: yes",
            "RISK: unverified-visible-claim",
        ]
    )


def validate_final_output(
    text: str,
    *,
    context: RuntimeContext,
    evidence: Sequence[ToolEvidence],
    required_frame: bool = True,
) -> EvidenceValidationResult:
    text = text or ""
    gaps: list[str] = []
    missing: list[str] = []

    if required_frame:
        missing = [field for field in REQUIRED_FRAME_FIELDS if not _has_required_field(text, field)]
        if missing:
            gaps.append("missing required frame fields: " + ", ".join(missing))

    if _SECRET_RE.search(text) or _CARDISH_RE.search(text):
        gaps.append("protected data pattern present in visible text")

    if context.lane == "internal-support":
        lowered = text.lower()
        if "customer hermes" in lowered and "internal support" in lowered:
            gaps.append("Customer Hermes and internal-support lanes are conflated")

    if _CLAIM_RE.search(text):
        durable_evidence = [
            item for item in evidence
            if item.success or item.durable_ref or item.tool_name in {"send_message", "approval"}
        ]
        if not durable_evidence:
            gaps.append("visible completion/send/fix claim has no durable evidence")

    if gaps:
        return EvidenceValidationResult(
            allowed=False,
            reason="; ".join(gaps),
            missing_fields=tuple(missing),
            evidence_gaps=tuple(gaps),
            replacement_text=_blocked_frame("validation failed", gaps),
        )
    return EvidenceValidationResult(allowed=True, reason="validated")
