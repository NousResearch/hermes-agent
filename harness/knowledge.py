"""Durable knowledge gate: observations become knowledge only when they are
reusable, evidence-backed, non-temporary, deduplicated, and conflict-free.
Unresolvable conflicts become open questions — never invented truths.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Sequence

from .state import KnowledgeItem


def knowledge_id(kind: str, content: str) -> str:
    digest = hashlib.sha256(f"{kind}\x00{content}".encode("utf-8")).hexdigest()[:16]
    return f"k-{digest}"


@dataclass
class KnowledgeCandidate:
    type: str
    content: str
    scope: str = ""


_NON_DURABLE_MARKERS = ("tmp", "temp", "debug-print", "wip", "todo:")


def is_durable(candidate: KnowledgeCandidate) -> bool:
    text = candidate.content.strip()
    if len(text) < 8:
        return False
    lowered = text.lower()
    return not any(marker in lowered for marker in _NON_DURABLE_MARKERS)


def extract(
    candidates: Sequence[KnowledgeCandidate],
    existing: Sequence[KnowledgeItem],
    *,
    has_evidence: bool,
) -> List[KnowledgeItem]:
    """Validate candidates into storable items. No evidence → nothing stored."""
    if not has_evidence:
        return []
    known = {item.content for item in existing}
    items = []
    for candidate in candidates:
        content = candidate.content.strip()
        if not content or content in known:
            continue
        if not is_durable(candidate):
            continue
        known.add(content)
        items.append(
            KnowledgeItem(
                id=knowledge_id(candidate.type, content),
                type=candidate.type,
                content=content,
                scope=candidate.scope,
            )
        )
    return items


# Conflict priority, strongest first (system design §31).
_CONFLICT_PRIORITY = (
    "architecture_decision",
    "verified_source",
    "verified_evidence",
    "project_version",
    "durable_knowledge",
    "model_inference",
)


def resolve_conflict(sources: Sequence[str]) -> str:
    """Strongest source wins; unknown sources rank below everything known.
    Empty input means no conflict to resolve."""
    if not sources:
        return ""
    ranked = sorted(
        sources,
        key=lambda s: (
            _CONFLICT_PRIORITY.index(s)
            if s in _CONFLICT_PRIORITY
            else len(_CONFLICT_PRIORITY)
        ),
    )
    return ranked[0]
