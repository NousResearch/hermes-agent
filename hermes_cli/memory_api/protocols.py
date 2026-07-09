"""Phase 4 — Memory Provider Protocols (structural interfaces).

These are ``typing.Protocol`` definitions, NOT abstract base classes. A backend
qualifies by *shape* (duck typing) — no inheritance required. This satisfies
the directive that the memory system support structural compatibility across:

- SQLite provider (Layer 5 index, Layer 3 archive)
- markdown provider (Layer 1 identity, Layer 2 project notes)
- future ADR provider (Layer 4)
- future Graphiti / Holographic providers

A class need only implement the methods it actually supports; unsupported
operations should raise :class:`~hermes_cli.memory_api.errors.CapabilityError`
rather than silently no-op. Provenance is mandatory on every returned result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class SearchResultLike(Protocol):
    """Structural minimum a memory result must expose.

    The concrete :class:`~hermes_cli.memory_router.provenance.SearchResult`
    already satisfies this. Providers may return their own shape as long as it
    carries these provenance fields.
    """

    source_file: str
    memory_layer: str
    retrieval_method: str
    content: str
    timestamp: Optional[str]


@dataclass
class MemoryResult:
    """Normalized, provenance-bearing result returned by the API facade.

    Wraps any provider result into one uniform shape so callers never depend
    on backend-specific objects. Provenance is preserved verbatim.
    """

    source: str
    provider: str
    layer: str
    retrieval_method: str
    content: str
    timestamp: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None
    intent: Optional[str] = None
    capability: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_search_result(cls, r: Any, provider: str, *, intent: Optional[str] = None) -> "MemoryResult":
        """Build a MemoryResult from any SearchResultLike object."""
        return cls(
            source=getattr(r, "source_file", "") or "",
            provider=provider,
            layer=getattr(r, "memory_layer", "") or "",
            retrieval_method=getattr(r, "retrieval_method", "") or "",
            content=getattr(r, "content", "") or "",
            timestamp=getattr(r, "timestamp", None),
            snippet=getattr(r, "snippet", None),
            score=getattr(r, "score", None),
            intent=intent or getattr(r, "intent", None),
            capability=getattr(r, "capability", None),
            extra=dict(getattr(r, "extra", {}) or {}),
        )


@dataclass
class ContextBundle:
    """Structured (not concatenated) cross-layer context for prompt injection.

    The API preserves source *categories* and *provenance*; it performs NO LLM
    reasoning or ranking. Each category is a list of provenance-bearing
    :class:`MemoryResult` objects. Consumers assemble the final prompt block.
    """

    identity: list[MemoryResult] = field(default_factory=list)
    project: list[MemoryResult] = field(default_factory=list)
    decision: list[MemoryResult] = field(default_factory=list)
    recent: list[MemoryResult] = field(default_factory=list)
    other: list[MemoryResult] = field(default_factory=list)

    def all_results(self) -> list[MemoryResult]:
        out: list[MemoryResult] = []
        for cat in (self.identity, self.project, self.decision, self.recent, self.other):
            out.extend(cat)
        return out

    def is_empty(self) -> bool:
        return not self.all_results()


@dataclass
class NextAction:
    """A single forward-looking next step (Layer 2).

    Flat list; ``blocked_by`` carries lightweight dependencies (ADR ids or
    other action refs). No graph / workflow engine (per Phase 6 scope).
    """

    what: str
    owner: str = "unassigned"
    blocked_by: list[str] = field(default_factory=list)


@dataclass
class ProjectState:
    """Current, curated project truth (Layer 2). Built in Phase 6.

    L2 describes the PRESENT, not the PAST (memory-architecture.md §18.1).
    It holds status, owners, blockers, and next actions — and LINKS to
    history (ADRs / Archive / Search) by reference rather than copying it.

    ``last_verified`` / ``verified_by`` are informational only: when a human
    last confirmed the state. They are NEVER updated automatically.
    """

    project: str                       # project key, e.g. 'hermes-aios'
    title: str = ""
    status: str = ""                   # active | paused | blocked | done | archived
    updated_at: str = ""
    updated_by: str = ""
    owners: list[str] = field(default_factory=list)
    next_actions: list[NextAction] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    links: dict[str, list[str]] = field(default_factory=dict)  # adrs/archive/search
    last_verified: str = ""            # informational only; never auto-set
    verified_by: str = ""              # informational only; never auto-set
    narrative: str = ""                # 1-2 paragraph current-state body
    source: str = ""                   # absolute path to STATUS.md


@dataclass
class DecisionRecord:
    """An architectural decision / ADR (Layer 4). Built in Phase 5.

    Carries full provenance (§17.5): who drafted/created, who approved,
    and when. ``decision()`` returns ONLY accepted records; proposed drafts
    are excluded at the read boundary, never presented as decisions.
    """

    id: str
    title: str
    status: str = ""
    date: str = ""                  # acceptance/creation date (ISO)
    project: str = ""               # project key (hermes-aios, _system, ...)
    context: str = ""
    decision: str = ""
    consequences: str = ""
    source: str = ""
    # supersession
    supersedes: list[str] = field(default_factory=list)
    superseded_by: list[str] = field(default_factory=list)
    # provenance (write metadata, §17.5 directive)
    decision_maker: str = ""      # human who accepted
    proposed_by: str = ""         # who drafted (hermes | joe)
    related_components: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_by: str = ""           # = proposed_by at draft time
    created_at: str = ""           # ISO timestamp of draft
    approved_by: str = ""          # set on accept()
    approved_at: str = ""          # ISO timestamp of accept()


class CapabilityStatus(str, Enum):
    """Lifecycle state of a provider/capability, for graceful degradation."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"   # registered but not ready (e.g. empty index)
    UNBUILT = "unbuilt"           # intentionally not implemented yet (L2/L4)


@runtime_checkable
class MemoryProvider(Protocol):
    """Structural interface every memory backend conforms to.

    Implementing classes need NOT subclass anything. They only need methods
    matching these signatures for the operations they support. Unsupported
    operations must raise
    :class:`~hermes_cli.memory_api.errors.CapabilityError` — never return a
    silent success.

    Read operations return provenance-bearing results or empty lists. Write
    operations return a :class:`MemoryResult` on success or raise.
    """

    # -- metadata -------------------------------------------------------
    @property
    def name(self) -> str:
        """Stable provider identifier (e.g. 'sqlite-index', 'markdown-identity')."""
        ...

    @property
    def layers(self) -> list[str]:
        """Memory layers this provider can serve (e.g. ['L3-archive', 'L5'])."""
        ...

    def status(self) -> CapabilityStatus:
        """Current readiness for graceful degradation."""
        ...

    # -- read operations -------------------------------------------------
    def search(self, query: str, *, limit: int = 10, scope: Optional[str] = None) -> list[SearchResultLike]:
        """Full-text search across this provider's layers."""
        ...

    def archive(self, query: str = "", *, limit: int = 10, session_id: Optional[str] = None) -> list[SearchResultLike]:
        """Retrieve conversation-archive chunks (L3)."""
        ...

    def recent(self, *, limit: int = 10) -> list[SearchResultLike]:
        """Most-recent activity across this provider's layers."""
        ...

    # -- write operations (raise if unsupported; never silent no-op) ------
    def remember(self, content: str, *, layer: str, **meta: Any) -> MemoryResult:
        """Persist a memory. Raises CapabilityError if the provider is read-only."""
        ...

    def project(self, name: str) -> Optional["ProjectState"]:
        """Return a project record, or None if unavailable."""
        ...

    def decision(self, id: Optional[str] = None, topic: Optional[str] = None) -> list[DecisionRecord]:
        """Return ADR records. Raises CapabilityError if not built (Phase 5)."""
        ...
