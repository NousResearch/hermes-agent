"""Idea Box — bounded, approval-gated idea capture flow.

Captures ideas from Discord ``/idea <text>`` in a dedicated channel, deduplicates
against Kanban open tasks, 30-day session history, and Mnemosyne recall, then
presents a structured card for Sahil's confirmation before promoting to a Kanban
triage task.

Public API:
    IdeaCard      — structured idea representation
    DedupResult   — outcome of deduplication check
    IdeaStatus    — enum for idea lifecycle states
    DedupChecker  — dedup against kanban + session_search + mnemosyne
    IdeaBoxFlow   — full capture → dedup → card → confirm/reject flow
"""

from idea_box.models import DedupResult, IdeaCard, IdeaStatus, SourceRef
from idea_box.dedup import DedupChecker, DedupMatch, DedupSource
from idea_box.flow import IdeaBoxFlow, FlowResult

__all__ = [
    "DedupChecker",
    "DedupMatch",
    "DedupResult",
    "DedupSource",
    "FlowResult",
    "IdeaBoxFlow",
    "IdeaCard",
    "IdeaStatus",
    "SourceRef",
]