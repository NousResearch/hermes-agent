"""Bounded dependencies of the logical final excerpt quoted by a reaction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from plugins.platforms.matrix.reply_context import MatrixEventContext

EXCERPT_PART_LIMIT = 32
REPLY_EXCERPT_CHARS = 500
_UNAVAILABLE = "[event content unavailable]"


def body_digest(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


def source_characters(text: str) -> int:
    text = re.sub(r" \(\d+/\d+\)$", "", text)
    return sum(len(line) for line in text.splitlines() if not line.startswith("```"))


@dataclass(frozen=True)
class ReplyExcerptPart:
    event_id: str
    digest: str


@dataclass(frozen=True)
class ReplyExcerpt:
    parts: tuple[ReplyExcerptPart, ...]
    complete: bool
    target_digest: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"parts": [[part.event_id, part.digest] for part in self.parts],
                "complete": self.complete, "target_digest": self.target_digest}

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> ReplyExcerpt:
        return cls(tuple(ReplyExcerptPart(*part) for part in value["parts"]), value["complete"], value.get("target_digest", ""))


@dataclass
class LogicalReplyContext:
    excerpt: str
    delivery: ReplyExcerpt
    parents: tuple[MatrixEventContext, ...]

    @classmethod
    def capture(cls, adapter: Any, room_id: str, excerpt: str, delivery: ReplyExcerpt) -> LogicalReplyContext:
        return cls(excerpt, delivery, tuple(
            adapter._event_context_cache.retain(room_id, part.event_id) for part in delivery.parts
        ))

    async def refresh(self, adapter: Any, room_id: str) -> None:
        self.parents = tuple([
            await adapter._event_context_cache.refresh(adapter._client, room_id, parent)
            for parent in self.parents
        ])

    def text(self, adapter: Any, room_id: str) -> str:
        parents = tuple(adapter._event_context_cache.recheck(room_id, parent) for parent in self.parents)
        if any(parent.redacted for parent in parents):
            return "[redacted]"
        if any(parent.state_error for parent in parents) or not self.delivery.complete:
            return _UNAVAILABLE
        if not any(parent.sender or parent.text for parent in parents):
            return self.excerpt
        if any(not parent.text for parent in parents):
            return _UNAVAILABLE
        if all(body_digest(parent.text) == part.digest
               for parent, part in zip(parents, self.delivery.parts, strict=True)):
            return self.excerpt
        return "\n".join(parent.text for parent in parents)[:REPLY_EXCERPT_CHARS]
