"""A plugin that is both a general plugin and the active memory provider registers once."""

import logging
import textwrap

import pytest

from hermes_cli.plugins import get_plugin_manager
from plugins.memory import load_memory_provider
from tools.registry import registry


def _install(home, monkeypatch, *, enabled=True):
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(home / "empty"))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    monkeypatch.chdir(home)
    (home / "config.yaml").write_text(
        f"plugins:\n  enabled: {'[dual]' if enabled else '[]'}\nmemory:\n  provider: dual\n"
    )
    plugin = home / "plugins" / "dual"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: dual\nversion: 1.0.0\nkind: standalone\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(textwrap.dedent('''\
        from agent.memory_provider import MemoryProvider

        class Provider(MemoryProvider):
            name = "dual"
            def is_available(self): return True
            def initialize(self, session_id, **kwargs): pass
            def get_tool_schemas(self): return []

        def handler(args, **kwargs):
            return __name__

        def register(ctx):
            ctx.register_memory_provider(Provider())
            ctx.register_system_prompt_section("dual.guide", "DUAL GUIDE")
            ctx.register_tool("dual_probe", "dual", {"name": "dual_probe", "description": "probe",
                              "parameters": {"type": "object", "properties": {}}}, handler)
    '''))
    return get_plugin_manager()


@pytest.mark.parametrize("order", ["plugin-first", "memory-first", "memory-only"])
def test_dual_kind_plugin_registers_section_and_tool_once(tmp_path, monkeypatch, caplog, order):
    manager = _install(tmp_path, monkeypatch, enabled=order != "memory-only")
    caplog.set_level(logging.WARNING)
    try:
        if order == "plugin-first":
            manager.discover_and_load()
        # Every new session loads the provider again.
        assert load_memory_provider("dual") is not None
        assert load_memory_provider("dual") is not None
        if order == "memory-first":
            manager.discover_and_load()
            assert load_memory_provider("dual") is not None
        assert manager._plugins.get("dual") is None or manager._plugins["dual"].error is None
        assert not [r for r in caplog.records if "already registered" in r.getMessage()]
        assert list(manager._system_prompt_sections) == ["dual.guide"]
        entry = registry.get_entry("dual_probe", scope=manager.scope_key)
        owner = entry.handler({})
        if order == "memory-only":
            assert owner.startswith("_hermes_user_memory.")
        else:  # the general load's copy owns the tool, as it owns the hooks
            assert owner.startswith("hermes_plugins.")
        manager.unload()
        assert not manager._system_prompt_sections
        assert registry.get_entry("dual_probe", scope=manager.scope_key) is None
    finally:
        manager.unload()
