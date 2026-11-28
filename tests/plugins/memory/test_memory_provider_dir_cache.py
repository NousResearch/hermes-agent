"""Provider-dir verdicts are memoized on the directory's module signature (#147444)."""
import plugins.memory as memory_plugins


def test_verdict_cached_until_a_module_changes(tmp_path, monkeypatch):
    plugin = tmp_path / "big_plugin"
    plugin.mkdir()
    (plugin / "__init__.py").write_text("def register(ctx):\n    pass\n")
    (plugin / "store.py").write_text("x = 1\n")
    monkeypatch.setattr(memory_plugins, "_PROVIDER_DIR_VERDICTS", {})
    calls = []
    real = memory_plugins._defines_memory_provider
    monkeypatch.setattr(memory_plugins, "_defines_memory_provider", lambda p: calls.append(p) or real(p))

    assert memory_plugins._is_memory_provider_dir(plugin) is False
    assert memory_plugins._is_memory_provider_dir(plugin) is False
    assert len(calls) == 1  # second call served from the cache, no re-parse

    (plugin / "store.py").write_text(
        "from agent.memory_provider import MemoryProvider\nclass P(MemoryProvider):\n    pass\n")
    assert memory_plugins._is_memory_provider_dir(plugin) is True  # size/mtime changed -> re-evaluated
    assert len(calls) == 2

    (plugin / "extra.py").write_text("y = 2\n")
    memory_plugins._is_memory_provider_dir(plugin)
    assert len(calls) == 3  # a new module also invalidates
