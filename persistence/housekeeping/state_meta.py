"""Core-owned key/value bookkeeping contract.

The production store keeps a small ``state_meta`` table of core-owned bookkeeping keys: scheduler and repair gates (``cli.py`` startup repairs, the kanban retag gate), per-session goal, loop, and heartbeat state (``hermes_cli/goals.py``, ``hermes_cli/loops.py``, ``hermes_cli/heartbeat.py``), and storage-version stamps. Callers reach those keys through ``StateMetaRepository``, a typed, scope-aware get and set.

Scope of the promise: this is a store for keys the CORE OWNS, addressed by literal string, scope-isolated like every other capability. It is NOT a plugin escape hatch and NOT a generic query surface: there is no enumeration of the whole table, no typed values, no expiry, and no transactional composition with other capabilities (the retag verb that gates itself on a meta key owns that atomicity inside ``SessionRepository.retag_worker_sessions``). A plugin needing durable state uses its own storage; a new core key is a code change in the owning module, never a runtime registration.

Prefix listing (caller ``hermes_cli/loops.py`` reading the ``loop:`` rows): ``list_prefix`` is the ONE bounded read the family admits, because the /loop registry stores one core-owned key per session under a fixed prefix and must find them all without knowing the session ids. It is a read over a caller-named literal prefix, never an enumeration: the prefix is required (empty is a ``ValueError``, so the no-enumeration rule is "a prefix is required"), and in matching wildcard characters in it are literal (``%`` and ``_`` never match anything but themselves) and ASCII letters match either case (``loop:`` also reads ``LOOP:x``), every other character only itself. Entries come back as records in the order the keys were first stored (``set`` on a stored key keeps its place); an unknown prefix reads an empty tuple.

Outcome semantics: ``get`` of an unknown key is the typed ``None`` return (never ``NotFound``); ``set`` is an upsert, so the second write of a key overwrites and the read sees the latest value. Values are opaque strings; the repository never parses them. Keys must be non-empty (``ValueError`` at the boundary, an input-shape violation per the persistence records' convention): no caller writes one, and admitting it would make the empty key a hidden singleton.


Error taxonomy: a store failure surfaces as the store errors (``TransientStoreError`` / ``PermanentStoreError`` / ``StoreUnavailable``); on a read-only assembly ``set`` refuses typed through the same boundary with ``PermanentStoreError`` (a read-only attach never accepts the write on retry; the SQLite adapter translates the attach's ``sqlite3.OperationalError``) while ``get`` keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


__all__ = ["StateMetaEntry", "StateMetaRepository"]


@dataclass(frozen=True, slots=True)
class StateMetaEntry:
    """One stored key/value pair as ``list_prefix`` returns it."""

    key: str
    value: str


@runtime_checkable
class StateMetaRepository(Protocol):
    """Core-owned bookkeeping keys per scope; see the module docstring for what this is not."""

    def get(self, key: str) -> Optional[str]:
        """The stored value for ``key`` in this scope, or ``None`` when the key was never set. Empty ``key`` raises ``ValueError`` before any store access."""
        ...

    def set(self, key: str, value: str) -> None:
        """Upsert ``key`` to ``value`` in this scope: a first write creates, a later write overwrites, and a following ``get`` reads the latest value. Empty ``key`` raises ``ValueError`` before any store access; a read-only assembly refuses with a typed store error and writes nothing."""
        ...

    def list_prefix(self, prefix: str) -> tuple[StateMetaEntry, ...]:
        """Every key in this scope that starts with ``prefix`` (wildcard characters are literal; ASCII letters match either case, every other character only itself), with its value, in the order the keys were first stored (``set`` on a stored key keeps its place); an unmatched prefix reads an empty tuple. Empty ``prefix`` raises ``ValueError`` before any store access: this is a bounded read, never an enumeration. Read-only assemblies serve this read."""
        ...
