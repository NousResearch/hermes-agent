"""The local memory storage backend: today's ``MEMORY.md`` / ``USER.md`` behind the memory port.

:class:`LocalMemoryStore` wraps the ``tools.memory_tool_store.MemoryStore` the memory tool uses
today — the same files at the same paths (``<profile home>/memories/MEMORY.md`` and ``USER.md``),
the same char caps read from the profile's config, the same lock, drift guard, threat scan and
response dicts. Each operation is the same call today's callers make, after one check of the pin:
the bound scope (``hermes_scope.current_scope()``) must be a local one whose agent is the store's
own profile home. A local store holds one agent and only agent-level records, so any other pin
(another tenant, another profile) is refused with ``StoragePinRefused`` before anything is read
or written, and a write naming another level is refused with ``MemoryLevelRefused``. With no
scope bound, the pin is the ambient profile (``get_hermes_home()``), as every local caller has it
today.

The store's identity is frozen when the port opens: the profile home it was opened for. Today's
store resolves its files through ``get_hermes_home()`` on every call, so the pin also requires
the home the files will resolve under to be the frozen one — a scope that names this profile
without moving the ambient home (``bind_state(home_override=False)``) must not redirect a write
into whichever profile the ambient home names. A hosted backend never resolves a tenant's
identity from a filesystem path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from hermes_memory_port import MemoryLevelRefused, MemoryStorage, MemoryStorageProvider, MemoryStore
from hermes_scope import LOCAL_TENANT, current_scope
from hermes_state_port import StoragePinRefused


class LocalMemoryStore(MemoryStore):
    """Today's on-disk memory store, pinned to one profile home."""

    levels = frozenset({"agent"})

    def __init__(self, store: Any, home: Path) -> None:
        from hermes_constants import hermes_home_key
        self.store = store
        self.home = home
        self._home_key = hermes_home_key(home)

    def _pin(self) -> None:
        from hermes_constants import get_hermes_home, hermes_home_key
        scope = current_scope()
        if scope is not None and scope.tenant != LOCAL_TENANT:
            raise StoragePinRefused(f"a local memory store serves tenant {LOCAL_TENANT!r} only")
        agent = scope.agent if scope is not None else get_hermes_home()
        # Both sides canonical and absolute: a relative agent is the directory it names now.
        if hermes_home_key(agent) != self._home_key:
            raise StoragePinRefused("the bound agent is not this memory store's profile")
        # The wrapped store resolves its files through the ambient home on every call, so that
        # home must be the pinned profile too — the files a pinned operation reads and writes are
        # decided by it, and a bind that names this profile without moving the home would point
        # them at whichever profile the ambient home names instead.
        if hermes_home_key() != self._home_key:
            raise StoragePinRefused("the ambient profile is not this memory store's")

    def target_enabled(self, target: str) -> bool:
        self._pin()
        return self.store.target_enabled(target)

    def char_limit(self, target: str) -> int:
        self._pin()
        return self.store._char_limit(target)

    def entries(self, target: str) -> list[str]:
        self._pin()
        return self.store._read_file(self.store._path_for(target))

    def system_prompt_block(self, target: str) -> Optional[str]:
        self._pin()
        return self.store.format_for_system_prompt(target)

    def reload(self) -> None:
        self._pin()
        self.store.load_from_disk()

    def resolve_entry(self, target: str, old_text: str, verb: str) -> dict[str, Any]:
        self._pin()
        return self.store.resolve_entry(target, old_text, verb)

    def resolve_batch_entries(self, target: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
        self._pin()
        return self.store.resolve_batch_entries(target, operations)

    def add(self, target: str, content: str, *, level: str = "agent") -> dict[str, Any]:
        self._pin()
        self._check_level(target, level)
        return self.store.add(target, content)

    def replace(self, target: str, old_text: str, new_content: str, matched_entry: Optional[str] = None,
                *, level: str = "agent") -> dict[str, Any]:
        self._pin()
        self._check_level(target, level)
        return self.store.replace(target, old_text, new_content, matched_entry)

    def patch(self, target: str, old_text: str, new_text: str, matched_entry: Optional[str] = None,
              *, level: str = "agent") -> dict[str, Any]:
        self._pin()
        self._check_level(target, level)
        return self.store.patch(target, old_text, new_text, matched_entry)

    def remove(self, target: str, old_text: str, matched_entry: Optional[str] = None,
               *, level: str = "agent") -> dict[str, Any]:
        self._pin()
        self._check_level(target, level)
        return self.store.remove(target, old_text, matched_entry)

    def apply_batch(self, target: str, operations: list[dict[str, Any]], *,
                    level: str = "agent") -> dict[str, Any]:
        self._pin()
        self._check_level(target, level)
        return self.store.apply_batch(target, operations)


class LocalMemoryStorageProvider(MemoryStorageProvider):
    """Today's memory: one ``memories/`` directory per profile home, holding that agent's records."""

    levels = frozenset({"agent"})

    @property
    def name(self) -> str:
        return "local"

    def open(self, home: Any = None) -> MemoryStorage:
        """The port for *home*, or for the bound scope's agent (else the ambient home), over the
        store today's callers construct: the profile's configured caps and flags, loaded from
        disk. The store's identity is frozen here; the pin is checked by every operation."""
        scope = current_scope()
        if home is None:
            if scope is not None and scope.tenant != LOCAL_TENANT:
                raise StoragePinRefused(f"a local memory store serves tenant {LOCAL_TENANT!r} only")
            from hermes_constants import get_hermes_home
            home = scope.agent if scope is not None else get_hermes_home()
        home = Path(home)
        from hermes_scope import push_state
        from tools.memory_tool import load_on_disk_store
        # Only the home moves, for the construction alone: the profile's own config names the
        # caps and flags, exactly as today's callers read them.
        with push_state(home=home):
            store = load_on_disk_store()
        return MemoryStorage(LocalMemoryStore(store, home))

    def assert_isolating(self) -> None:
        """A local store is one profile's files and refuses every pin but its own
        (``LocalMemoryStore._pin``); there is no shared role that could bypass that."""
        return
