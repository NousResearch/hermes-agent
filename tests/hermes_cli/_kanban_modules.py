"""Test helper: one attribute namespace over the decomposed ``kanban_db`` modules.

Tests ported from the pre-split monolith address everything as ``kb.<name>``.
Reads resolve against the split module that owns the name (never the
deprecated compat shim); writes (``monkeypatch.setattr``) land on that owner,
so patching a seam affects the code that actually calls it.
"""

from __future__ import annotations

import importlib

_OWNERS = (
    "hermes_cli.kanban_db_dispatch",
    "hermes_cli.kanban_db_workspace",
    "hermes_cli.kanban_db_notify",
    "hermes_cli.kanban_db_connect",
    "hermes_cli.kanban_db_graph",
    "hermes_cli.kanban_db",
)


class KanbanModules:
    def __init__(self) -> None:
        object.__setattr__(self, "_mods", [importlib.import_module(m) for m in _OWNERS])

    def _owner(self, name: str):
        for mod in self._mods:
            if name in vars(mod):
                return mod
        return None

    def __getattr__(self, name: str):
        owner = self._owner(name)
        if owner is None:
            raise AttributeError(name)
        return vars(owner)[name]

    def __setattr__(self, name: str, value) -> None:
        owner = self._owner(name) or self._mods[0]
        setattr(owner, name, value)

    def __delattr__(self, name: str) -> None:
        owner = self._owner(name)
        if owner is None:
            raise AttributeError(name)
        delattr(owner, name)
