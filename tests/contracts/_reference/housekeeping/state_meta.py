"""In-memory reference implementation of ``StateMetaRepository`` (test fixture).

Registered onto the assembled backend as the ``state_meta`` attribute per the append-only extension contract in ``backend.py``. Keys live under the ``"state_meta"`` key of the owning scope's namespace, so cross-scope isolation is structural. The sessions reference reads the same table for the retag gate (``retag_worker_sessions``), which is why the table name is shared through :func:`state_meta_table` instead of duplicated.
"""

from __future__ import annotations

import string
from typing import Dict, Optional

from persistence import ProfileScope
from persistence.housekeeping.state_meta import StateMetaEntry

from ..backend import InMemoryBackend, register_repository

__all__ = ["InMemoryStateMetaRepository", "state_meta_table"]

_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def _fold(text: str) -> str:
    """ASCII letters lowered, every other character kept: the case rule of SQLite's built-in ``LIKE``."""
    return text.translate(_ASCII_LOWER)


def state_meta_table(backend: InMemoryBackend, scope: ProfileScope) -> dict[str, str]:
    """The scope's meta table, created on first touch; shared with the sessions reference's retag gate."""
    table: dict[str, str] = backend.scope_store(scope).setdefault("state_meta", {})
    return table


def require_key(key: str) -> None:
    """The boundary's input-shape check: an empty key is a caller error, never a store row."""
    if not key:
        raise ValueError("state_meta key must be non-empty")


class InMemoryStateMetaRepository:
    """See module docstring; semantics follow persistence/housekeeping/state_meta.py exactly."""

    def __init__(self, backend: InMemoryBackend) -> None:
        self._backend = backend

    def get(self, key: str) -> Optional[str]:
        scope = self._backend.scope
        require_key(key)
        return state_meta_table(self._backend, scope).get(key)

    def set(self, key: str, value: str) -> None:
        scope = self._backend.scope
        require_key(key)
        state_meta_table(self._backend, scope)[key] = value

    def list_prefix(self, prefix: str) -> tuple[StateMetaEntry, ...]:
        scope = self._backend.scope
        require_key(prefix)
        table = state_meta_table(self._backend, scope)
        folded = _fold(prefix)
        # Insertion order (``set`` on a stored key keeps its place), as the SQLite table scan reads.
        return tuple(StateMetaEntry(key=key, value=value) for key, value in table.items() if _fold(key[:len(prefix)]) == folded)


register_repository("state_meta", InMemoryStateMetaRepository)
