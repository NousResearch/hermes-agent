"""Exercise dual-kind plugins through the real general and memory loaders."""

import textwrap

import pytest

from hermes_cli.plugins import get_plugin_manager
from plugins.memory import load_memory_provider


def _install(home, monkeypatch, *, label="first", enabled=True):
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
    (plugin / "plugin.yaml").write_text("name: dual\nversion: 1.0.0\nkind: standalone\n")
    (plugin / "values.py").write_text(f"LABEL = {label!r}\n")
    (plugin / "__init__.py").write_text(textwrap.dedent('''\
        from agent.memory_provider import MemoryProvider
        from .values import LABEL

        class Provider(MemoryProvider):
            name = "dual"
            def is_available(self): return True
            def initialize(self, session_id, **kwargs): pass
            def get_tool_schemas(self): return []

        def make_hook(label):
            def callback(**kwargs):
                return {"context": label}
            return callback

        def register(ctx):
            ctx.register_memory_provider(Provider())
            ctx.register_hook("pre_llm_call", make_hook(LABEL))
            ctx.register_hook("pre_llm_call", make_hook("second"))
    '''))
    return get_plugin_manager()


def _contexts(manager):
    return manager.invoke_hook("pre_llm_call", session_id="")


@pytest.mark.parametrize("order", ["plugin-first", "memory-first", "memory-only"])
def test_dual_kind_plugin_hooks_run_once(tmp_path, monkeypatch, order):
    manager = _install(tmp_path, monkeypatch, enabled=order != "memory-only")
    try:
        if order == "plugin-first":
            manager.discover_and_load()
        provider = load_memory_provider("dual")
        assert provider is not None and provider.name == "dual"
        if order == "memory-first":
            manager.discover_and_load()
        expected = [{"context": "first"}, {"context": "second"}]
        assert _contexts(manager) == expected
        # New provider instances must not append another fallback hook group.
        assert load_memory_provider("dual") is not provider
        assert _contexts(manager) == expected
        manager.unload()
        assert _contexts(manager) == []
        assert not manager._memory_hook_registrations
        if order != "memory-only":
            manager.discover_and_load(force=True)
        assert load_memory_provider("dual") is not None
        assert _contexts(manager) == expected
    finally:
        manager.unload()


def test_equal_names_in_different_homes_keep_their_own_imports(tmp_path, monkeypatch):
    managers = []
    try:
        for label in ("alpha", "beta"):
            manager = _install(tmp_path / label, monkeypatch, label=label)
            managers.append(manager)
            assert load_memory_provider("dual") is not None
            manager.discover_and_load()
            assert _contexts(manager) == [{"context": label}, {"context": "second"}]
        assert _contexts(managers[0]) == [{"context": "alpha"}, {"context": "second"}]
        managers[1].unload()
        assert _contexts(managers[0]) == [{"context": "alpha"}, {"context": "second"}]
    finally:
        for manager in managers:
            manager.unload()


def test_failed_general_registration_leaves_no_callable(tmp_path, monkeypatch):
    manager = _install(tmp_path, monkeypatch)
    try:
        assert load_memory_provider("dual") is not None
        source = tmp_path / "plugins" / "dual" / "__init__.py"
        source.write_text(source.read_text() + '\n    raise RuntimeError("broken registration")\n')
        manager.discover_and_load()
        assert _contexts(manager) == []
        assert not manager._plugins["dual"].enabled
    finally:
        manager.unload()


def test_same_name_different_sources_are_not_suppressed(tmp_path, monkeypatch):
    import shutil

    home = tmp_path / "home"
    manager = _install(home, monkeypatch)
    project = tmp_path / "project"
    source = project / ".hermes" / "plugins" / "dual"
    shutil.copytree(home / "plugins" / "dual", source)
    (source / "values.py").write_text('LABEL = "project"\n')
    monkeypatch.chdir(project)
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "1")
    try:
        assert load_memory_provider("dual") is not None
        manager.discover_and_load()
        assert _contexts(manager) == [
            {"context": "first"}, {"context": "second"},
            {"context": "project"}, {"context": "second"},
        ]
    finally:
        manager.unload()


@pytest.mark.parametrize("entry_kind", ["module", "function"])
def test_entry_point_registration_uses_same_hook_ownership(tmp_path, monkeypatch, entry_kind):
    from types import SimpleNamespace
    from plugins.memory import _load_provider_from_entry_point

    manager = _install(tmp_path, monkeypatch)
    try:
        manager.discover_and_load()
        module = manager._plugins["dual"].module
        target = module if entry_kind == "module" else module.register
        entry = SimpleNamespace(name="dual", load=lambda: target)
        assert _load_provider_from_entry_point(entry) is not None
        assert _contexts(manager) == [{"context": "first"}, {"context": "second"}]
    finally:
        manager.unload()


@pytest.mark.parametrize("order", ["plugin-first", "memory-first"])
def test_reexported_register_uses_plugin_source(tmp_path, monkeypatch, order):
    manager = _install(tmp_path, monkeypatch)
    plugin = tmp_path / "plugins" / "dual"
    original = plugin / "__init__.py"
    (plugin / "implementation.py").write_text(original.read_text())
    original.write_text("from .implementation import register, Provider  # MemoryProvider\n")
    try:
        if order == "plugin-first":
            manager.discover_and_load()
        assert load_memory_provider("dual") is not None
        if order == "memory-first":
            manager.discover_and_load()
        assert _contexts(manager) == [{"context": "first"}, {"context": "second"}]
    finally:
        manager.unload()


def test_suppressed_hook_still_returns_disposable_handle(tmp_path, monkeypatch):
    manager = _install(tmp_path, monkeypatch)
    source = tmp_path / "plugins" / "dual" / "__init__.py"
    source.write_text(source.read_text().split("def register(ctx):")[0] + textwrap.dedent('''\
        def register(ctx):
            handle = ctx.register_hook("pre_llm_call", make_hook("discard"))
            assert handle.active
            handle.dispose()
            assert not handle.active
            provider = Provider()
            provider.configured = True
            ctx.register_memory_provider(provider)
            ctx.register_hook("pre_llm_call", make_hook("kept"))
    '''))
    try:
        manager.discover_and_load()
        provider = load_memory_provider("dual")
        assert provider is not None and provider.configured
        assert _contexts(manager) == [{"context": "kept"}]
    finally:
        manager.unload()
