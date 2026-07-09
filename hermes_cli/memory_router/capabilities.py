"""Trivial L1 identity capability.

Per the architecture (memory-architecture.md §3), identity memory lives in:
  ~/.hermes/SOUL.md
  ~/.hermes/memories/IDENTITY.md
  ~/.hermes/memories/USER.md
  ~/.hermes/memories/MEMORY.md

This capability is deliberately THIN: it returns *pointers* (source file paths
with provenance) for the identity intent. It does NOT read, parse, summarize,
or interpret the file contents. Any content interpretation is out of scope for
the Router and belongs to callers/agents.
"""

from __future__ import annotations

from typing import Any, Optional

from hermes_constants import get_hermes_home

from .provenance import SearchResult

# Identity files and the layer label each maps to.
_IDENTITY_FILES: tuple[tuple[str, str], ...] = (
    ("SOUL.md", "L1-identity"),
    ("memories/IDENTITY.md", "L1-identity"),
    ("memories/USER.md", "L1-identity"),
    ("memories/MEMORY.md", "L1-identity"),
)


class IdentityCapability:
    """Capability returning identity-file pointers with provenance.

    Handler interface: ``handle(method, **kwargs)``. Only ``"search"`` is
    meaningfully served; others return an empty list.
    """

    def handle(self, method: str, **kwargs: Any) -> list[SearchResult]:
        if method != "search":
            return []
        return self._pointers()

    def _pointers(self) -> list[SearchResult]:
        home = get_hermes_home()
        results: list[SearchResult] = []
        for rel, layer in _IDENTITY_FILES:
            path = home / rel
            if not path.exists():
                continue
            try:
                mtime = path.stat().st_mtime
                from datetime import datetime, timezone

                ts = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            except OSError:
                ts = None
            results.append(
                SearchResult(
                    source_file=str(path),
                    memory_layer=layer,
                    retrieval_method="direct-file",
                    content="",
                    timestamp=ts,
                    snippet=f"identity file: {rel}",
                    intent="identity",
                    capability="L1-identity",
                )
            )
        return results
