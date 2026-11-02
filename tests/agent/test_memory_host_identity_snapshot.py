"""Hosted prefetch validates both sides against the same loaded provider identity."""

from agent.memory_manager import MemoryManager
from hermes_cli import plugins as plugins_mod
from plugins.memory import load_memory_provider
from tests.hermes_cli.test_plugin_host import _home_with_plugins


def test_host_identity_remains_stable_when_provider_mutates_its_name(tmp_path, monkeypatch):
    home = _home_with_plugins(tmp_path, monkeypatch, {})
    provider_dir = home / "plugins" / "memprobe"
    provider_dir.mkdir(parents=True)
    (provider_dir / "__init__.py").write_text('''
from agent.memory_provider import MemoryObservation, MemoryPrefetchResult, MemoryProvider
class Probe(MemoryProvider):
    name = "memprobe"
    def is_available(self): return True
    def initialize(self, session_id, **kwargs): pass
    def get_tool_schemas(self): return []
    def prefetch(self, query, *, session_id=""):
        self.name = "other"
        return MemoryPrefetchResult(context="kept", observations=(
            MemoryObservation("recall", "test", 1, {"ok": True}, provider="memprobe"),
            MemoryObservation("recall", "test", 1, {"bad": True}, provider="other")))
''', encoding="utf-8")
    host = plugins_mod.get_plugin_manager()._plugin_host()
    manager = MemoryManager()
    try:
        provider = load_memory_provider("memprobe", register_skills=False)
        assert provider is not None
        manager.add_provider(provider)
        result = manager.prefetch_all_result("question")
        assert result.context == "kept"
        assert [item.provider for item in result.observations] == ["memprobe"]
        assert [item.provider for item in manager.prefetch_all_result("again").observations] == ["memprobe"]
    finally:
        manager.shutdown_all()
        host.shutdown()
