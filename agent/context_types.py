"""Context type definitions for GL(生成式循环) principle.

Defines clear boundaries between long-term, medium-term, and short-term context.
All enums have sensible defaults for backward compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class ContextScope(Enum):
    """Context lifecycle scope — defines where context lives and how long it persists.

    LONG_TERM:  Cross-session persistent memory (MEMORY.md/USER.md, external providers)
    MEDIUM_TERM: Current compression cycle (context summaries, live for 1+ compress cycles)
    SHORT_TERM: Single conversation turn (session messages, ephemeral within one round-trip)
    """
    LONG_TERM = "long_term"
    MEDIUM_TERM = "medium_term"
    SHORT_TERM = "short_term"

    def __repr__(self) -> str:
        return self.value

@dataclass
class ContextEntry:
    """A single context item with explicit lifecycle scope."""
    scope: ContextScope
    source: str  # "memory", "compression", "session", "reference"
    content: str
    created_at: float
    promoted: bool = False

    def promote(self) -> None:
        """Mark this entry as promoted to a longer-lived scope."""
        self.promoted = True

    def to_event_payload(self) -> dict:
        return {
            "scope": self.scope.value,
            "source": self.source,
            "content_preview": self.content[:200] if self.content else "",
            "promoted": self.promoted,
        }
