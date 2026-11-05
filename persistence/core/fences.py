"""Authority fence counters for the persistence boundary.

The session authority fences runtime ownership with three counters: the store-wide authority epoch, the per-session mutation revision, and the per-session execution generation. Each is an opaque monotonic value minted by the backend: callers carry it back verbatim and never do arithmetic beyond equality. A stale counter is refused with :class:`persistence.FenceStale`.
"""

from __future__ import annotations

from typing import NewType

__all__ = ["AuthorityEpoch", "ExecutionGeneration", "SessionRevision"]

AuthorityEpoch = NewType("AuthorityEpoch", int)
SessionRevision = NewType("SessionRevision", int)
ExecutionGeneration = NewType("ExecutionGeneration", int)
