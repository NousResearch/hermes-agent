"""The plugin host child recognises a provider object whichever module name its base was imported by."""
from __future__ import annotations

import sys
import types

from hermes_agent.compat import moves
from hermes_agent.hermes_cli import plugin_host_child, plugin_isolation


def test_a_base_imported_under_its_moved_name_still_matches(monkeypatch):
    """HOST_OBJECT_BASES lists the old module name; a plugin that imported the base by its current name
    loaded only that name, and its provider object must still be recognised."""
    current = types.ModuleType("test_provider_base_current")

    class Base:
        pass

    current.Base = Base
    monkeypatch.setitem(sys.modules, "test_provider_base_current", current)
    monkeypatch.setitem(moves.MOVES, "test_provider_base_old", "test_provider_base_current")
    monkeypatch.setattr(plugin_isolation, "HOST_OBJECT_BASES", {"probe": "test_provider_base_old:Base"})

    assert plugin_host_child._provider_base(type("Impl", (Base,), {})()) is Base
