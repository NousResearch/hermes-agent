"""Layer 5 — SQLite + FTS5 index over EXISTING markdown (Phase 1).

This is a REGENERABLE CACHE. Markdown files are the source of truth; the
SQLite database is derivable from them and therefore gitignored. Deleting
``index.db`` and rebuilding yields an identical result (deterministic).

Phase 1 indexes:
  - L1 identity:  ~/.hermes/SOUL.md, ~/.hermes/memories/{MEMORY,USER,IDENTITY}.md
  - L2-ish:       ~/.hermes/HERMES_PROJECTS.md, ~/.hermes/HERMES_SESSION.md
  - L5-notes:     any other *.md under ~/.hermes/memories/

No numpy, no LLM, no embeddings. Pure sqlite3 stdlib + FTS5 (with a LIKE
fallback if FTS5 is unavailable at runtime).
"""

from __future__ import annotations

from .capability import IndexCapability
from .indexer import MemoryIndex

# Phase 3: register the archive-lifecycle listener (on_session_end) once the
# memory index package is imported. No close path is touched.
from . import archive_lifecycle as _archive_lifecycle  # noqa: F401  (side-effect: registers hook)

__all__ = ["MemoryIndex", "IndexCapability"]
