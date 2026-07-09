"""Phase 4 — IndexMemoryProvider adapter.

Adapts the existing :class:`~hermes_cli.memory_index.indexer.MemoryIndex`
(SQLite + FTS5, read-only derived cache) to the structural
:class:`~hermes_cli.memory_api.protocols.MemoryProvider` Protocol. It serves
Layers 3 (archive) and 5 (broad index), plus L1/L5-notes recent activity.

It is structurally compatible — it implements the Protocol methods; no
inheritance. Write operations (``remember``) raise
:class:`~hermes_cli.memory_api.errors.CapabilityError` because the SQLite index
is a derived cache, not an authority (memory-architecture.md: markdown/raw are
the source of truth). The API therefore never reports a fake successful write.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from hermes_cli.memory_api.errors import CapabilityError
from hermes_cli.memory_api.protocols import (
    CapabilityStatus,
    MemoryProvider,
    MemoryResult,
    SearchResultLike,
)
from hermes_cli.memory_index.indexer import MemoryIndex


class IndexMemoryProvider:
    """Read-only MemoryProvider over the SQLite/FTS5 index (L3 + L5)."""

    def __init__(
        self,
        index: Optional[MemoryIndex] = None,
        db_path: Optional[Path] = None,
        hermes_home: Optional[Path] = None,
    ) -> None:
        self.index = index or MemoryIndex(db_path=db_path, hermes_home=hermes_home)

    # -- metadata -------------------------------------------------------
    @property
    def name(self) -> str:
        return "sqlite-index"

    @property
    def layers(self) -> list[str]:
        return ["L3-archive", "L5", "L1-identity", "L5-notes"]

    def status(self) -> CapabilityStatus:
        if not self.index.available():
            return CapabilityStatus.UNAVAILABLE
        return CapabilityStatus.AVAILABLE

    # -- read operations -------------------------------------------------
    def search(self, query: str, *, limit: int = 10, scope: Optional[str] = None) -> list[SearchResultLike]:
        return self.index.search(query, limit=limit, scope=scope)

    def archive(self, query: str = "", *, limit: int = 10, session_id: Optional[str] = None) -> list[SearchResultLike]:
        return self.index.archive(query, limit=limit, session_id=session_id)

    def recent(self, *, limit: int = 10) -> list[SearchResultLike]:
        return self.index.recent(limit=limit)

    # -- write operations (read-only provider: explicit refusal) ----------
    def remember(self, content: str, *, layer: str, **meta: Any) -> MemoryResult:
        raise CapabilityError(
            "remember",
            "sqlite-index is a derived read-only cache; persist via the markdown/memory subsystem",
            layer=layer,
            provider=self.name,
        )

    def project(self, name: str) -> Optional[Any]:
        # L2 not built in Phase 4. Returning None is honest (no data), not a
        # fake success; the facade escalates to CapabilityError if a caller
        # requires a project record.
        return None

    def decision(self, id: Optional[str] = None, topic: Optional[str] = None) -> list[Any]:
        # L4 not built in Phase 5. Explicitly unsupported, not a silent empty.
        raise CapabilityError(
            "decision",
            "ADR/L4 provider not built yet (Phase 5)",
            layer="L4",
            provider=self.name,
        )
