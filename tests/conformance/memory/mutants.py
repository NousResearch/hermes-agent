"""Broken local memory backends the memory conformance suite must reject (``test_mutants.py``).

Each passed an earlier version of the suite. ``unpinned_writes`` and ``unpinned_reads`` drop the
pin check outright; ``tenant_blind_pin`` keeps the agent half only, so another tenant naming the
profile's home passes; ``ambient_home_ignored`` drops the check that the files resolve under the
pinned profile, so a bind that names the profile without moving the ambient home redirects the
write; ``add_cap_writes_anyway`` writes the entry and then returns the over-cap refusal;
``batch_cap_after_write`` applies a batch operation by operation and refuses over-cap only at
the end, so a refused batch leaves partial state; ``level_refusal_ignored`` accepts a write at a
level the store does not hold. Each mutant is applied by the ``backend`` fixture when
``HERMES_MEMORY_MUTANT`` names it.
"""

from typing import Any


def _write_without_pin(name: str):
    def method(self, *args, **kwargs):
        return getattr(self.store, name)(*args, **{k: v for k, v in kwargs.items() if k != "level"})
    return method


def _read_without_pin(name: str):
    def method(self, *args, **kwargs):
        store = self.store
        if name == "reload":
            return store.load_from_disk()
        target = args[0]
        if name == "entries":
            return store._read_file(store._path_for(target))
        if name == "system_prompt_block":
            return store.format_for_system_prompt(target)
        if name == "char_limit":
            return store._char_limit(target)
        return getattr(store, name)(*args, **kwargs)
    return method


def _unpinned_writes(monkeypatch):
    """Every write keeps its level check but skips the pin check."""
    from hermes_memory_port_local import LocalMemoryStore
    for name in ("add", "replace", "patch", "remove", "apply_batch"):
        monkeypatch.setattr(LocalMemoryStore, name, _write_without_pin(name))


def _unpinned_reads(monkeypatch):
    """Every read and the reload skip the pin check."""
    from hermes_memory_port_local import LocalMemoryStore
    for name in ("entries", "system_prompt_block", "target_enabled", "char_limit",
                 "resolve_entry", "resolve_batch_entries", "reload"):
        monkeypatch.setattr(LocalMemoryStore, name, _read_without_pin(name))


def _tenant_blind_pin(monkeypatch):
    """The pin's agent half alone: another tenant naming the store's home passes."""
    from hermes_constants import get_hermes_home, hermes_home_key
    from hermes_scope import current_scope
    from hermes_state_port import StoragePinRefused
    from hermes_memory_port_local import LocalMemoryStore

    def pin(self):
        scope = current_scope()
        agent = scope.agent if scope is not None else get_hermes_home()
        if hermes_home_key(agent) != self._home_key:
            raise StoragePinRefused("the bound agent is not this memory store's profile")
        if hermes_home_key() != self._home_key:
            raise StoragePinRefused("the ambient profile is not this memory store's")
    monkeypatch.setattr(LocalMemoryStore, "_pin", pin)


def _ambient_home_ignored(monkeypatch):
    """The pin no longer requires the ambient home to be the pinned profile, so a bind that names
    the profile without moving the home redirects the write into the ambient profile's files."""
    from hermes_constants import get_hermes_home, hermes_home_key
    from hermes_scope import LOCAL_TENANT, current_scope
    from hermes_state_port import StoragePinRefused
    from hermes_memory_port_local import LocalMemoryStore

    def pin(self):
        scope = current_scope()
        if scope is not None and scope.tenant != LOCAL_TENANT:
            raise StoragePinRefused(f"a local memory store serves tenant {LOCAL_TENANT!r} only")
        agent = scope.agent if scope is not None else get_hermes_home()
        if hermes_home_key(agent) != self._home_key:
            raise StoragePinRefused("the bound agent is not this memory store's profile")
    monkeypatch.setattr(LocalMemoryStore, "_pin", pin)


def _add_cap_writes_anyway(monkeypatch):
    """A refused over-cap add writes the entry anyway, then returns the refusal."""
    from hermes_memory_port_local import LocalMemoryStore
    original = LocalMemoryStore.add

    def add(self, target, content, *, level="agent"):
        result = original(self, target, content, level=level)
        if result.get("success") is False and "chars" in result.get("error", ""):
            entries = self.store._read_file(self.store._path_for(target)) + [content.strip()]
            self.store._write_file(self.store._path_for(target), entries)
        return result
    monkeypatch.setattr(LocalMemoryStore, "add", add)


def _batch_cap_after_write(monkeypatch):
    """apply_batch applies each operation as it goes and refuses over-cap only at the end, so a
    refused batch leaves its earlier operations applied."""
    from hermes_memory_port_local import LocalMemoryStore

    def apply_batch(self, target, operations, *, level="agent"):
        self._pin()
        self._check_level(target, level)
        for op in operations or []:
            act = op.get("action")
            content = (op.get("content") or op.get("new_text") or "").strip()
            old_text = (op.get("old_text") or "").strip()
            if act == "add":
                self.store.add(target, content)
            elif act == "remove":
                self.store.remove(target, old_text)
            elif act == "patch":
                self.store.patch(target, old_text, content)
        from tools.memory_tool_store import ENTRY_DELIMITER
        entries = self.store._read_file(self.store._path_for(target))
        limit = self.store._char_limit(target)
        if len(ENTRY_DELIMITER.join(entries)) > limit:
            return {"success": False, "done": True, "target": target, "error": (
                f"Applying all operations would put memory over {limit:,} chars.")}
        return {"success": True, "done": True, "target": target,
                "usage": f"{len(ENTRY_DELIMITER.join(entries)):,}/{limit:,}",
                "entry_count": len(entries), "note": "Write saved."}
    monkeypatch.setattr(LocalMemoryStore, "apply_batch", apply_batch)


def _level_refusal_ignored(monkeypatch):
    """Writes at a level the store does not hold are accepted at the agent level."""
    from hermes_memory_port_local import LocalMemoryStore
    monkeypatch.setattr(LocalMemoryStore, "_check_level", lambda self, target, level: None)


MUTANTS = {
    "unpinned_writes": _unpinned_writes,
    "unpinned_reads": _unpinned_reads,
    "tenant_blind_pin": _tenant_blind_pin,
    "ambient_home_ignored": _ambient_home_ignored,
    "add_cap_writes_anyway": _add_cap_writes_anyway,
    "batch_cap_after_write": _batch_cap_after_write,
    "level_refusal_ignored": _level_refusal_ignored,
}


def apply_mutant(monkeypatch):
    import os
    name = os.environ.get("HERMES_MEMORY_MUTANT")
    if name:
        MUTANTS[name](monkeypatch)
