"""Plugin context-reference providers across a plugin reload (force re-discovery), within one load and across
profiles. Only the successful-reload test is new behavior; the rest pin what already holds on main."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import types
from pathlib import Path

import hermes_yaml as yaml
import pytest

from hermes_agent.agent.context_references import _context_reference_providers, get_context_reference_providers
from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginManager, _plugin_home_scope

_PROVIDER = '''
from agent.context_references import ContextReferenceProvider

class Tickets(ContextReferenceProvider):
    prefix = "tkt"

    def __init__(self, label="{label}"):
        self.label = label

    async def autocomplete(self, query, *, limit=10):
        return []

    async def expand(self, target):
        return self.label + " " + target

def register(ctx):
    ctx.register_context_reference(Tickets())
'''
_REFUSED = "prefix 'tkt' is already registered"


def _write_plugin(home: Path, name: str, label: str, extra: str = "") -> None:
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": name, "version": "1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(_PROVIDER.replace("{label}", label) + extra, encoding="utf-8")


def _write_config(home: Path, *names: str) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": list(names)}}), encoding="utf-8")


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    _context_reference_providers.clear()
    yield home
    _context_reference_providers.clear()


def _expand(target: str) -> str | None:
    provider = get_context_reference_providers().get("tkt")
    return None if provider is None else asyncio.run(provider.expand(target))


def test_successful_reload_replaces_the_plugins_provider(home):
    _write_plugin(home, "tickets", "v1")
    _write_config(home, "tickets")
    manager = PluginManager()
    try:
        manager.discover_and_load()
        assert _expand("7") == "v1 7"

        _write_plugin(home, "tickets", "v2-reloaded")
        manager.discover_and_load(force=True)
        assert manager._plugins["tickets"].error is None
        assert _expand("7") == "v2-reloaded 7"
    finally:
        manager.unload()


def test_reload_publishes_when_earlier_registrations_are_disposed_during_it(home, monkeypatch):
    # Registrations made before this plugin's load can be disposed while it loads (a supervised task
    # finishing, a handle released from a callback); the manager's registration list gets shorter then.
    handles: list = []
    shared = types.ModuleType("ctxref_shared")
    shared.__dict__["handles"] = handles
    monkeypatch.setitem(sys.modules, "ctxref_shared", shared)
    holder = home / "plugins" / "aaa-holder"
    holder.mkdir(parents=True)
    (holder / "plugin.yaml").write_text(yaml.safe_dump({"name": "aaa-holder", "version": "1.0"}), encoding="utf-8")
    (holder / "__init__.py").write_text(
        "import ctxref_shared\n\ndef register(ctx):\n"
        "    ctxref_shared.handles[:] = [ctx.on_unload(lambda: None), ctx.on_unload(lambda: None)]\n",
        encoding="utf-8")
    _write_plugin(home, "tickets", "v1")
    _write_config(home, "aaa-holder", "tickets")
    manager = PluginManager()
    try:
        manager.discover_and_load()
        assert _expand("7") == "v1 7"

        _write_plugin(home, "tickets", "v2")
        source = home / "plugins" / "tickets" / "__init__.py"
        source.write_text(source.read_text(encoding="utf-8").replace(
            "def register(ctx):\n",
            "def register(ctx):\n"
            "    import ctxref_shared\n"
            "    for handle in ctxref_shared.handles:\n"
            "        handle.dispose()\n",
        ), encoding="utf-8")
        manager.discover_and_load(force=True)
        assert handles and not any(handle.active for handle in handles)  # disposed mid-load
        assert manager._plugins["tickets"].error is None
        assert _expand("7") == "v2 7"
    finally:
        manager.unload()


def test_failed_reload_keeps_the_working_provider(home):
    _write_plugin(home, "tickets", "working")
    _write_config(home, "tickets")
    manager = PluginManager()
    try:
        manager.discover_and_load()
        working = get_context_reference_providers()["tkt"]

        _write_plugin(home, "tickets", "half-loaded", extra="    raise RuntimeError('configuration unavailable')\n")
        manager.discover_and_load(force=True)
        assert "configuration unavailable" in manager._plugins["tickets"].error
        assert get_context_reference_providers()["tkt"] is working
    finally:
        manager.unload()


def test_duplicate_prefix_in_one_load_keeps_the_first_provider(home, caplog):
    _write_plugin(home, "tickets", "first", extra="    ctx.register_context_reference(Tickets('second'))\n")
    _write_config(home, "tickets")
    manager = PluginManager()
    try:
        with caplog.at_level(logging.WARNING):
            manager.discover_and_load()
        assert manager._plugins["tickets"].error is None
        assert _expand("7") == "first 7"
        assert _REFUSED in caplog.text

        manager.unload("tickets")  # disabling a plugin leaves its provider registered
        assert _expand("7") == "first 7"
    finally:
        manager.unload()


def test_another_plugin_cannot_take_a_registered_prefix(home, caplog):
    _write_plugin(home, "tickets-a", "a")
    _write_plugin(home, "tickets-b", "b")
    _write_config(home, "tickets-a", "tickets-b")
    manager = PluginManager()
    try:
        with caplog.at_level(logging.WARNING):
            manager.discover_and_load()
        first = get_context_reference_providers()["tkt"]
        assert _REFUSED in caplog.text

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            manager.discover_and_load(force=True)
        assert _REFUSED in caplog.text
        assert _expand("1") == asyncio.run(first.expand("1"))
    finally:
        manager.unload()


def test_profiles_share_the_first_profiles_provider(home, tmp_path, caplog):
    # One process, one manager per profile (multiplexed gateway): A and C enable the plugin, B does not.
    # The prefix is process-wide: every profile sees A's provider and C's registration is refused.
    homes = {}
    for name in ("A", "B", "C"):
        homes[name] = tmp_path / f"profile-{name}"
        _write_plugin(homes[name], "xref", name)
        _write_config(homes[name], *(("xref",) if name != "B" else ()))
    managers = {name: PluginManager(scope_key=str(path)) for name, path in homes.items()}
    try:
        for name, manager in managers.items():
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                manager.discover_and_load()
            assert (_REFUSED in caplog.text) is (name == "C")

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            managers["C"].discover_and_load(force=True)
        assert _REFUSED in caplog.text
        managers["A"].discover_and_load(force=True)
        for manager in managers.values():
            with _plugin_home_scope(manager.home_path):
                assert _expand("1") == "A 1"
    finally:
        for manager in managers.values():
            manager.unload()
