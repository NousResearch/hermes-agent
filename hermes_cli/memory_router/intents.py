"""Intent taxonomy for the Memory Router.

Intents are the units of *what a memory request is asking for*. They are
deliberately decoupled from *which layer serves them* — the Router maps an
intent to one or more registered capabilities, never the other way around.

This module is pure data + a tiny enum. It contains no logic that could
touch storage or an LLM.
"""

from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    """Classification of a memory request.

    Inherits from ``str`` so intent values compare equal to their string
    form and serialize cleanly to JSON (useful for provenance records and
    the routing log).
    """

    IDENTITY = "identity"            # -> L1
    PROJECT_STATE = "project_state"  # -> L2 (unavailable in Phase 1)
    DECISION = "decision"           # -> L4 (unavailable in Phase 1)
    HISTORICAL = "historical"        # -> L5 (broad FTS5)
    RELATIONSHIP = "relationship"    # -> L7 optional (unavailable in Phase 1)
    RECENT = "recent"               # -> L3+L2 (unavailable in Phase 1)
    CONTEXT = "context"             # fan-out; only L5 participates in Phase 1
    ARCHIVE = "archive"           # -> L3 (conversation archive; Phase 2/3)
    REMEMBER = "remember"         # -> L1 (markdown write; human-curated)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# Metadata describing where each intent is served and a human description.
# ``phases_available`` records which phases (per docs/memory/memory-architecture.md) bring
# the layer online. The Router only ever routes to *registered + available*
# capabilities, so this table is documentation/audit aid, not routing logic.
INTENT_METADATA: dict[Intent, dict] = {
    Intent.IDENTITY: {
        "layer": "L1",
        "description": "Who the user/agent is; identity files.",
        "phases_available": "Phase 0 (always available)",
    },
    Intent.PROJECT_STATE: {
        "layer": "L2",
        "description": "Active project status, roadmap, progress.",
        "phases_available": "Phase 6",
    },
    Intent.DECISION: {
        "layer": "L4",
        "description": "Architectural decisions / ADRs.",
        "phases_available": "Phase 5",
    },
    Intent.HISTORICAL: {
        "layer": "L5",
        "description": "Broad full-text search over all indexed markdown.",
        "phases_available": "Phase 1",
    },
    Intent.RELATIONSHIP: {
        "layer": "L7",
        "description": "Cross-entity relationships / graph (optional).",
        "phases_available": "Phase 6 (optional)",
    },
    Intent.RECENT: {
        "layer": "L3+L2",
        "description": "Most recent activity across sessions/projects.",
        "phases_available": "Phase 2/3",
    },
    Intent.CONTEXT: {
        "layer": "fan-out",
        "description": "Cross-cutting context assembled from available layers.",
        "phases_available": "Phase 1 (L5 only) -> later fan-out",
    },
    Intent.ARCHIVE: {
        "layer": "L3",
        "description": "Conversation archive retrieval (closed sessions).",
        "phases_available": "Phase 2/3",
    },
    Intent.REMEMBER: {
        "layer": "L1",
        "description": "Persist a human-curated markdown memory (e.g. MEMORY.md).",
        "phases_available": "Phase 4 (write refused unless a writer is wired)",
    },
}


def all_intents() -> list[Intent]:
    """Return every known intent (handy for validation/registry iteration)."""
    return list(Intent)


def is_valid_intent(value: object) -> bool:
    """True if ``value`` is a known intent (accepts Intent or string)."""
    if isinstance(value, Intent):
        return True
    if isinstance(value, str):
        return value in {i.value for i in Intent}
    return False


# Safe default: on low classification confidence, route to the broad index.
DEFAULT_INTENT = Intent.HISTORICAL
