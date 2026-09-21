"""Data models for the Idea Box flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class IdeaStatus(str, Enum):
    """Lifecycle states for a captured idea."""

    CAPTURED = "captured"        # Just ingested, dedup pending
    DUPLICATE = "duplicate"     # Dedup found an existing match
    NOVEL = "novel"             # Dedup passed, card ready for review
    CONFIRMED = "confirmed"     # Sahil approved → Kanban triage task
    REJECTED = "rejected"       # Sahil declined → LLM Wiki provenance log


@dataclass(frozen=True)
class SourceRef:
    """Reference to the originating Discord message."""

    platform: str            # "discord"
    channel_id: str           # Discord channel id
    message_id: str           # Discord message id
    user_id: str              # Discord user id
    user_name: str            # Discord display name
    channel_name: str = ""   # Channel name (e.g. "idea-box")
    raw_text: str = ""        # Original ``/idea <text>`` text

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "channel_id": self.channel_id,
            "message_id": self.message_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "channel_name": self.channel_name,
            "raw_text": self.raw_text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceRef":
        return cls(
            platform=data.get("platform", ""),
            channel_id=data.get("channel_id", ""),
            message_id=data.get("message_id", ""),
            user_id=data.get("user_id", ""),
            user_name=data.get("user_name", ""),
            channel_name=data.get("channel_name", ""),
            raw_text=data.get("raw_text", ""),
        )


@dataclass
class IdeaCard:
    """Structured representation of a captured idea.

    Produced by the intake flow after dedup.  Presented to Sahil for
    confirmation before any Kanban task creation or LLM Wiki logging.
    """

    summary: str                          # One-paragraph summary
    source: SourceRef                     # Originating message reference
    tags: list[str] = field(default_factory=list)
    dedup_status: IdeaStatus = IdeaStatus.CAPTURED
    dedup_matches: list[dict] = field(default_factory=list)  # Populated by DedupChecker
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    idea_id: Optional[str] = None         # Set on confirmation (kanban task id)
    rejection_reason: Optional[str] = None  # Set on rejection

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "source": self.source.to_dict(),
            "tags": self.tags,
            "dedup_status": self.dedup_status.value,
            "dedup_matches": self.dedup_matches,
            "created_at": self.created_at,
            "idea_id": self.idea_id,
            "rejection_reason": self.rejection_reason,
        }

    def render_card(self) -> str:
        """Render the card as a Discord-friendly text block for presentation."""
        lines = [
            f"**Idea Card**",
            f"",
            f"**Summary:** {self.summary}",
            f"**Tags:** {', '.join(self.tags) if self.tags else '(none)'}",
            f"**Dedup:** {self.dedup_status.value}",
        ]
        if self.dedup_matches:
            lines.append("")
            lines.append("**Dedup matches:**")
            for m in self.dedup_matches:
                lines.append(f"  - {m.get('source', '?')}: {m.get('title', m.get('summary', ''))}")
        lines.append("")
        lines.append(f"**Source:** {self.source.platform}/{self.source.channel_name} (msg {self.source.message_id})")
        lines.append(f"**Captured:** {self.created_at}")
        return "\n".join(lines)


@dataclass
class DedupResult:
    """Outcome of a deduplication check.

    ``is_duplicate`` is True when any source found a match above its
    similarity threshold.  ``matches`` carries the details so the card
    can surface them to Sahil.
    """

    is_duplicate: bool
    matches: list[dict] = field(default_factory=list)
    checked_sources: list[str] = field(default_factory=list)