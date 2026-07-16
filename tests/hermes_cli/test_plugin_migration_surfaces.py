from types import SimpleNamespace

from agent.memory_provider import MemoryProvider
from hermes_cli.plugins import PluginContext, PluginManifest
from plugins.memory import (
    clear_registered_memory_providers,
    discover_memory_providers,
    list_memory_provider_names,
    load_memory_provider,
)
from tools.registry import registry


class _MemoryProvider(MemoryProvider):
    """Test memory provider."""

    @property
    def name(self):
        return "entrypoint-test-memory"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.session_id = session_id

    def get_tool_schemas(self):
        return []


def _context():
    manager = SimpleNamespace(_plugin_tool_names=set())
    manifest = PluginManifest(name="migration-test", source="entrypoint")
    return PluginContext(manifest, manager)


def test_plugin_tool_preserves_result_bound():
    name = "plugin_result_bound_migration_test"
    ctx = _context()
    try:
        ctx.register_tool(
            name=name,
            toolset="migration-test",
            schema={"name": name, "description": "test", "parameters": {}},
            handler=lambda _args: "ok",
            max_result_size_chars=1234,
        )
        assert registry.get_entry(name).max_result_size_chars == 1234
    finally:
        registry.deregister(name)


def test_entrypoint_memory_provider_is_discoverable_and_fresh():
    clear_registered_memory_providers()
    try:
        _context().register_memory_provider(_MemoryProvider())

        assert "entrypoint-test-memory" in list_memory_provider_names()
        discovered = {
            name: available
            for name, _desc, available in discover_memory_providers()
        }
        assert discovered["entrypoint-test-memory"] is True
        first = load_memory_provider("entrypoint-test-memory")
        second = load_memory_provider("entrypoint-test-memory")
        assert isinstance(first, _MemoryProvider)
        assert isinstance(second, _MemoryProvider)
        assert first is not second
    finally:
        clear_registered_memory_providers()
