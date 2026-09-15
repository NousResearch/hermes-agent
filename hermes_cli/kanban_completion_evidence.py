"""Structured, opt-in completion evidence shared by every Kanban surface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EVIDENCE_REQUIRED_CONTRACT = "evidence-required"


class CompletionEvidenceError(ValueError):
    """A declared evidence contract was not satisfied by a concrete receipt."""


@dataclass(frozen=True)
class CompletionEvidence:
    """One human-readable, machine-preserved completion receipt."""

    kind: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "detail": self.detail}


def normalize_completion_evidence(value: Any, *, required: bool) -> list[dict[str, str]]:
    """Validate receipts without accepting free-form placeholders."""
    if value is None:
        if required:
            raise CompletionEvidenceError(
                "completion requires concrete evidence: provide a non-empty evidence list"
            )
        return []
    if not isinstance(value, list):
        raise CompletionEvidenceError("evidence must be a list of {kind, detail} objects")
    normalized: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise CompletionEvidenceError("each evidence item must be a {kind, detail} object")
        kind, detail = item.get("kind"), item.get("detail")
        if not isinstance(kind, str) or not kind.strip() or not isinstance(detail, str) or not detail.strip():
            raise CompletionEvidenceError("each evidence item needs non-empty string kind and detail")
        normalized.append(CompletionEvidence(kind.strip(), detail.strip()).as_dict())
    if required and not normalized:
        raise CompletionEvidenceError(
            "completion requires concrete evidence: provide at least one evidence item"
        )
    return normalized
