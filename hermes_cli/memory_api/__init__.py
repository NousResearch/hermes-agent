"""Phase 4 — Memory API abstraction (L6 interface).

This package is the STABLE, versioned front door between callers (agents,
skills, CLI) and the memory subsystem. It defines:

- :class:`MemoryProvider` (typing.Protocol): structural interface a concrete
  backend conforms to. NO shared base class — a SQLite provider, a markdown
  provider, a future ADR provider, and Graphiti/Holographic providers are all
  eligible purely by shape (duck typing), per Joe's directive.
- :class:`MemoryAPI`: the facade callers use. It delegates to the
  :class:`~hermes_cli.memory_router.router.MemoryRouter` (which routes to
  registered capabilities) and to registered :class:`MemoryProvider`
  implementations for write/structured ops.
- Errors + result types that preserve provenance.

Hard constraints (docs/memory/memory-architecture.md §16):
- No behavior change to existing memory paths.
- No new storage, no new backends, no Holographic/Mem0/Graphiti/embeddings.
- Writes are NOT silent no-ops: an unsupported write raises
  :class:`CapabilityError`. No "success" is ever reported for an unpersisted
  write.

Scope of THIS milestone: facade + Protocols + adapters for the existing
search/archive/recent capabilities + contract tests. NOT implemented: ADR
storage, project memory, semantic search, new backends.

(Phase 5 added AdrProvider / DecisionRecord; Phase 6 added ProjectProvider /
ProjectState / NextAction — see docs/memory/memory-architecture.md §17, §18.)
"""

from __future__ import annotations

from .errors import CapabilityError, MemoryAPIError, UnsupportedCapability
from .facade import MemoryAPI
from .protocols import (
    CapabilityStatus,
    ContextBundle,
    DecisionRecord,
    MemoryProvider,
    MemoryResult,
    NextAction,
    ProjectState,
    SearchResultLike,
)
from .providers import IndexMemoryProvider
from .adr import AdrProvider
from .project import ProjectProvider

__all__ = [
    "MemoryAPI",
    "MemoryProvider",
    "IndexMemoryProvider",
    "AdrProvider",
    "ProjectProvider",
    "MemoryResult",
    "SearchResultLike",
    "ContextBundle",
    "ProjectState",
    "NextAction",
    "DecisionRecord",
    "CapabilityStatus",
    "CapabilityError",
    "UnsupportedCapability",
    "MemoryAPIError",
]
