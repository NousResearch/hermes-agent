"""Pre-turn context assembly.

Builds the text Soma passes as `system_message` to AIAgent.run_conversation().
Hermes prepends its own identity / tool blocks; Soma's contribution lives
underneath them as additional context.

Three blocks (each capped, omitted when empty):

  BEHAVIOR RULES
    Every memory tagged "preference" or "behavior". These are short,
    persistent rules the agent must follow regardless of the current
    query. They are listed in full, not query-filtered.

  RELEVANT CONTEXT
    Top-K semantic memories most similar (cosine) to the current user
    message, above a similarity floor. Tag matches always rank above
    the floor so domain/identity facts surface even on tangential queries.

  PROCEDURES
    Top-K procedural memories matching the query. Same query mechanism
    as relevant context but type-restricted.

A separate CURATOR NOTES block is reserved for the background curator
(Phase 5); for now it stays empty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .memory_store import MemoryRecord, MemoryStore

logger = logging.getLogger(__name__)


BEHAVIOR_TAGS = frozenset({"preference", "behavior"})


@dataclass
class BuiltContext:
    text: str
    behavior_count: int = 0
    context_count: int = 0
    procedure_count: int = 0
    curator_notes: List[str] = field(default_factory=list)


class ContextBuilder:
    def __init__(
        self,
        store: MemoryStore,
        *,
        max_behavior: int = 8,
        max_context: int = 8,
        max_procedures: int = 8,
        min_similarity: float = 0.30,
    ):
        self.store = store
        self.max_behavior = max_behavior
        self.max_context = max_context
        self.max_procedures = max_procedures
        self.min_similarity = min_similarity

    def build(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        curator_notes: Optional[Sequence[str]] = None,
    ) -> BuiltContext:
        behavior = self._behavior_block()
        context = self._context_block(query)
        procedures = self._procedure_block(query)
        notes = list(curator_notes or [])

        sections: List[str] = []
        if behavior:
            sections.append(_section("BEHAVIOR RULES", behavior))
        if context:
            sections.append(_section("RELEVANT CONTEXT", context))
        if procedures:
            sections.append(_section("PROCEDURES", procedures))
        if notes:
            sections.append(_section("CURATOR NOTES", notes))

        if not sections:
            return BuiltContext(text="")

        header = (
            "[Soma context — Soma maintains memories about this user "
            "and injects them here. Use them as background; do not "
            "mention them unless asked.]"
        )
        text = header + "\n\n" + "\n\n".join(sections)
        return BuiltContext(
            text=text,
            behavior_count=len(behavior),
            context_count=len(context),
            procedure_count=len(procedures),
            curator_notes=notes,
        )

    # -- Block builders ------------------------------------------------------

    def _behavior_block(self) -> List[str]:
        rules: List[Tuple[float, MemoryRecord]] = []
        for record in self.store.all():
            if any(t in BEHAVIOR_TAGS for t in record.tags):
                # Newer rules win on contradiction; sort by last_seen_at desc.
                rules.append((record.last_seen_at, record))
        rules.sort(key=lambda pair: pair[0], reverse=True)
        return [r.content for _, r in rules[: self.max_behavior]]

    def _context_block(self, query: str) -> List[str]:
        if not query.strip():
            return []
        try:
            results = self.store.query(
                query, top_k=self.max_context, min_similarity=self.min_similarity
            )
        except Exception as exc:
            logger.warning("soma: context query failed: %s", exc)
            return []
        out: List[str] = []
        for record, _sim in results:
            # Skip pure behavior records (already shown above) and procedures
            # (handled separately).
            if any(t in BEHAVIOR_TAGS for t in record.tags):
                continue
            if record.type == "procedural":
                continue
            out.append(record.content)
        return out

    def _procedure_block(self, query: str) -> List[str]:
        if not query.strip():
            return []
        try:
            results = self.store.query(
                query, top_k=self.max_procedures * 2, min_similarity=self.min_similarity
            )
        except Exception as exc:
            logger.warning("soma: procedure query failed: %s", exc)
            return []
        out: List[str] = []
        for record, _sim in results:
            if record.type != "procedural":
                continue
            out.append(record.content)
            if len(out) >= self.max_procedures:
                break
        return out


def _section(title: str, items: Sequence[str]) -> str:
    body = "\n".join(f"- {item}" for item in items)
    return f"## {title}\n{body}"
