"""Memory storage port: the agent's curated memory, as one plugin kind.

One port at the level today's memory tool and memory-provider call sites use, never files and
never SQL: :class:`MemoryStore` is the bounded, entry-based store behind today's ``memory`` tool
and the built-in provider — the agent's own notes (``target="memory"``) and its user profile
(``target="user"``) — with the add / patch / remove edits, the all-or-nothing ``operations``
batch whose char cap is checked on its final result, the staged-write resolutions the write
approval gate takes, and the frozen system-prompt block.

Ownership (``Scope`` docs): the agent's own notes are agent-level records. The user profile is
agent-level on a personal agent one person uses, and a per-participant record whenever more than
one person uses the agent (a group agent, or a personal agent its owner shared) — decided by the
store, never by a caller-supplied id. :meth:`MemoryStore.levels` is what a backend holds; a
write says its level explicitly (``level=``), and a backend that does not hold it refuses with
:class:`MemoryLevelRefused` — the local backend holds the agent level only, until local installs
move from profiles to agents. Reads resolve through the backend's own policy.

This is the interface and its local backend only: no production caller goes through it yet, and
nothing in config.yaml selects a backend. Today's callers still construct
``tools.memory_tool_store.MemoryStore`` directly; moving them behind this port, call site by call
site, is a follow-up, and until it lands a provider plugin stores nothing that Hermes reads.
:func:`resolve_memory_storage` takes the backend's name explicitly: ``"local"`` (or empty) is
``hermes_memory_port_local`` (today's ``MEMORY.md`` / ``USER.md`` under the profile's
``memories/``, at today's char caps); any other name is a provider from
``plugins/memory_storage/<name>/`` or ``$HERMES_HOME/plugins/<name>/``. A named provider that is
missing or unavailable is an error, never a fallback to the local store.

Providers load in-process only. Under ``plugins.isolation: host`` a user provider is refused with
the reason: a host process would hand back an opaque object for the port, and its calls would not
carry the caller's scope. Serving one from the plugin host needs a typed proxy for the port and a
validated scope envelope bound around every call, which is a follow-up.

Every operation reads its pin (tenant, agent, owner, actor) from ``hermes_scope.current_scope()``
when it runs, never from a path, ``HERMES_HOME`` or an argument; a port exposes no way to pin
anything else. A backend refuses a pin it cannot serve with
:class:`~hermes_state_port.StoragePinRefused`. Memory is the agent's own learning, not
session-owned data, so a turn's fence does not gate these writes: any activity whose scope pins
the agent (a turn, an approval replay, the gateway's ``/memory``) may write it, with or without
``can_administer``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

from hermes_state_port import LEVELS, StoragePinRefused

__all__ = [
    "LEVELS", "MemoryLevelRefused", "MemoryStorage", "MemoryStorageProvider", "MemoryStore",
    "StoragePinRefused", "resolve_memory_storage",
]


class MemoryLevelRefused(PermissionError):
    """The write names an ownership level this backend does not hold (see :data:`LEVELS`), or a
    level its target never takes: the agent's own notes are agent-level records only. Says
    nothing about the content."""


class MemoryStore(ABC):
    """Curated memory for the pinned agent: the agent's own notes and its user profile.

    Every result is the dict today's callers get from ``tools.memory_tool_store.MemoryStore``:
    ``{"success": bool, ...}`` with the model-facing messages, usage and matched entries, so a
    call site moves behind the port without rewriting its responses.
    """

    #: The ownership levels this backend holds (:data:`LEVELS`). The local backend holds the
    #: agent level only.
    levels: frozenset[str] = frozenset({"agent"})

    def _check_level(self, target: str, level: str) -> None:
        """Refuse a write whose level the store does not hold, or that the target never takes:
        the agent's own notes are agent-level records; the user profile may also be a
        per-participant record on a backend that holds that level."""
        if target == "memory" and level != "agent":
            raise MemoryLevelRefused("the agent's own notes are agent-level records")
        if level not in self.levels:
            raise MemoryLevelRefused(
                f"this memory store holds {sorted(self.levels)} records, not {level!r}")

    @abstractmethod
    def target_enabled(self, target: str) -> bool:
        """Whether writes and the prompt block for *target* are enabled for this agent."""

    @abstractmethod
    def char_limit(self, target: str) -> int:
        """The target's whole-store char cap. A write whose result would exceed it is refused
        before anything is written."""

    @abstractmethod
    def entries(self, target: str) -> list[str]:
        """The target's entries as they stand now, oldest first (missing and foreign look the
        same: both refuse or both come back empty through the pin)."""

    @abstractmethod
    def system_prompt_block(self, target: str) -> Optional[str]:
        """The frozen session-start block for the prompt (``None`` when empty): mid-session
        writes do not change it, keeping the prefix cache intact."""

    @abstractmethod
    def reload(self) -> None:
        """Re-read the backing store and refresh the frozen system-prompt blocks."""

    @abstractmethod
    def resolve_entry(self, target: str, old_text: str, verb: str) -> dict[str, Any]:
        """``{"success": True, "matched_entry": <full entry>}`` for the entry *old_text* selects
        now, or the error the edit would return — the read the write-approval gate pins a staged
        write to."""

    @abstractmethod
    def resolve_batch_entries(self, target: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
        """Dry-run :meth:`apply_batch`: the same scan, walk and budget checks, nothing written;
        on success ``{"success": True, "matched_entries": [...]}`` in batch order."""

    # -- writes; each says its level explicitly ---------------------------------------------

    @abstractmethod
    def add(self, target: str, content: str, *, level: str = "agent") -> dict[str, Any]:
        """Append one entry unless it would exceed the cap (refused before anything is written)."""

    @abstractmethod
    def replace(self, target: str, old_text: str, new_content: str, matched_entry: Optional[str] = None,
                *, level: str = "agent") -> dict[str, Any]:
        """Replace the WHOLE entry containing *old_text* with *new_content*; *matched_entry* pins
        the exact entry a staged write was reviewed against."""

    @abstractmethod
    def patch(self, target: str, old_text: str, new_text: str, matched_entry: Optional[str] = None,
              *, level: str = "agent") -> dict[str, Any]:
        """Replace the one *old_text* span inside its entry with *new_text*; the rest of the
        entry is kept verbatim."""

    @abstractmethod
    def remove(self, target: str, old_text: str, matched_entry: Optional[str] = None,
               *, level: str = "agent") -> dict[str, Any]:
        """Remove the entry containing *old_text*."""

    @abstractmethod
    def apply_batch(self, target: str, operations: list[dict[str, Any]], *,
                    level: str = "agent") -> dict[str, Any]:
        """Apply add/replace/patch/remove operations atomically against the FINAL budget: one
        call can free space and add entries. All-or-nothing — any failed or over-cap result
        writes nothing, the cap included."""


@dataclass(frozen=True)
class MemoryStorage:
    """One opened memory port. ``close`` gives back whatever ``open`` took."""

    store: MemoryStore
    close: Callable[[], None] = lambda: None


class MemoryStorageProvider(ABC):
    """The plugin kind: supplies the memory port. ``name``, ``open`` and ``assert_isolating``
    are required; every other method has a safe default, and growth stays additive."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``"local"``."""

    #: The ownership levels this backend holds (see :data:`LEVELS`).
    levels: frozenset[str] = frozenset({"agent"})

    def is_available(self) -> bool:
        """Whether this backend can run here. No network calls."""
        return True

    @abstractmethod
    def open(self, home: Any = None) -> MemoryStorage:
        """The port for the bound scope. *home* names the profile the caller wants, for tools
        that serve a named profile; with none, the bound scope's agent, else the ambient one. The
        store's identity is frozen here; the pin is checked by every operation."""

    @abstractmethod
    def assert_isolating(self) -> None:
        """Raise unless this backend isolates tenants and agents from each other in storage
        itself (a Postgres backend whose role is a superuser or has ``BYPASSRLS`` fails).
        Required, with no default: a backend that has not checked must not pass. Nothing calls it
        yet: the first caller that serves a scope from a resolved provider must call it, and
        refuse the provider when it raises, before :meth:`open`."""


def resolve_memory_storage(name: str) -> MemoryStorageProvider:
    """The provider named *name* (``"local"`` or empty: the local backend). Raises ``LookupError``
    with the reason for a named provider that is missing, refused or unavailable: storage never
    falls back to another store."""
    name = name.strip()
    if name in ("", "local"):
        from hermes_memory_port_local import LocalMemoryStorageProvider
        return LocalMemoryStorageProvider()
    from plugins.memory_storage import MemoryStorageRefused, load_memory_storage
    try:
        provider = load_memory_storage(name)
    except MemoryStorageRefused as refused:
        raise LookupError(f"memory storage {name!r}: {refused}") from refused
    if provider is None:
        raise LookupError(f"memory storage {name!r}: no such memory storage provider")
    if not provider.is_available():
        raise LookupError(f"memory storage {name!r} is not available here")
    return provider
