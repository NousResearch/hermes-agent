"""Adapter: MemoryIndex -> Router handler interface.

The Router dispatches by calling ``handler(method, **kwargs)``. This adapter
wraps :class:`~hermes_cli.memory_index.indexer.MemoryIndex` so an index
instance satisfies that callable contract:

  - method == "search"  -> index.search(query, limit, scope)
  - method == "build"   -> index.build(hermes_home)
  - method == "rebuild" -> index.rebuild(hermes_home)

The capability is *always available* as a Router target (it participates in
historical/context intents). If the underlying db is missing or empty, search
simply returns an empty list — graceful, no error. The index can be built
lazily by callers via ``router.route(Intent.HISTORICAL, "build")``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from hermes_cli.memory_index.indexer import MemoryIndex


class IndexCapability:
    """Router handler adapter around :class:`MemoryIndex`."""

    def __init__(
        self,
        index: Optional[MemoryIndex] = None,
        db_path: Optional[Path] = None,
        hermes_home: Optional[Path] = None,
    ) -> None:
        self.index = index or MemoryIndex(db_path=db_path, hermes_home=hermes_home)

    def handle(self, method: str, **kwargs: Any) -> Any:
        if method == "search":
            return self.index.search(
                kwargs.get("query", ""),
                limit=kwargs.get("limit", 10),
                scope=kwargs.get("scope"),
            )
        if method == "archive":
            return self.index.archive(
                kwargs.get("query", ""),
                limit=kwargs.get("limit", 10),
                session_id=kwargs.get("session_id"),
            )
        if method == "recent":
            return self.index.recent(limit=kwargs.get("limit", 10))
        if method == "build":
            home = kwargs.get("hermes_home")
            return self.index.build(Path(home) if home else None)
        if method == "rebuild":
            home = kwargs.get("hermes_home")
            return self.index.rebuild(Path(home) if home else None)
        # Unknown method: treat as a no-op returning an empty search result.
        return []

    # Convenience mirror of availability for tests / callers.
    def available(self) -> bool:
        return True
