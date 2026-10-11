"""Pluggable kanban backend seam (:mod:`hermes_cli.kanban_backend`).

Covers the PR-1 contract:
- default resolution (no config, no registration) returns the in-tree modules;
- a plugin-registered backend is returned and USED by a converted call site;
- a configured-but-unregistered backend fails closed (no silent fallback);
- a backend module missing the connection surface is rejected at the seam;
- the plugin-side registrar (``PluginContext.register_kanban_backend``) registers
  a real module under the plugin key and unregisters on unload.
"""

import sys
import types

import pytest

from hermes_cli import kanban_backend


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate the process-global backend registry and config override per test."""
    saved = dict(kanban_backend._registry._modules)
    yield
    kanban_backend._registry._modules.clear()
    kanban_backend._registry._modules.update(saved)


def _install_backend(monkeypatch, name="testbackend"):
    """Register an alternate backend module and configure it active."""
    calls = []

    class _Closing:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def connect(db_path=None, *, board=None):
        calls.append("connect")
        raise AssertionError("test backend connect() must not open a real DB in this test")

    module = types.ModuleType("test_kanban_backend_module")
    module.connect = connect
    module.connect_closing = lambda *a, **kw: _Closing()
    module.write_txn = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("not used here"))
    module.marker = f"backend:{name}"
    kanban_backend.register_backend(name, module)
    return module, calls


def _config_plugin_backend(monkeypatch, name):
    """Point ``plugins.kanban_backend`` at *name* without touching config.yaml."""
    import hermes_cli.kanban_backend as seam
    monkeypatch.setattr(seam, "_configured_backend_name", lambda: name, raising=True)


def test_default_resolution_is_in_tree_modules():
    import hermes_cli.kanban_db as kb
    import hermes_cli.kanban_db_connect as kbc
    assert kanban_backend.get_kanban_db() is kb
    assert kanban_backend.get_kanban_db_connect() is kbc


def test_registered_backend_is_returned(monkeypatch):
    module, _ = _install_backend(monkeypatch)
    _config_plugin_backend(monkeypatch, "testbackend")
    assert kanban_backend.get_kanban_db() is module
    assert kanban_backend.get_kanban_db_connect() is module


def test_registered_backend_used_by_converted_call_site(monkeypatch, tmp_path):
    """A converted call site (hermes_cli.kanban_ops) resolves through the seam:
    its module-level ``kb`` binding is a lazy proxy that follows the active backend."""
    module, _ = _install_backend(monkeypatch)
    _config_plugin_backend(monkeypatch, "testbackend")
    import hermes_cli.kanban_ops as kanban_ops
    assert kanban_ops.kb.marker == "backend:testbackend"
    assert kanban_ops.kbc.connect is module.connect


def test_unconfigured_backend_fails_closed(monkeypatch):
    _config_plugin_backend(monkeypatch, "ghost")
    with pytest.raises(kanban_backend.KanbanBackendError) as exc:
        kanban_backend.get_kanban_db()
    assert "ghost" in str(exc.value)
    assert "plugins.kanban_backend" in str(exc.value)
    # And the connection resolver fails the same way (no silent default fallback).
    with pytest.raises(kanban_backend.KanbanBackendError):
        kanban_backend.get_kanban_db_connect()


def test_backend_missing_connection_surface_rejected(monkeypatch):
    bare = types.ModuleType("bare_backend")
    bare.marker = "bare"
    kanban_backend.register_backend("bare", bare)
    _config_plugin_backend(monkeypatch, "bare")
    # get_kanban_db still returns it (module surface is the plugin's business)...
    assert kanban_backend.get_kanban_db() is bare
    # ...but the connection seam refuses it loudly.
    with pytest.raises(kanban_backend.KanbanBackendError) as exc:
        kanban_backend.get_kanban_db_connect()
    assert "connect()" in str(exc.value)


def test_blank_config_value_means_default(monkeypatch):
    import hermes_cli.kanban_backend as seam
    monkeypatch.setattr(seam, "_configured_backend_name", lambda: None, raising=True)
    import hermes_cli.kanban_db as kb
    assert kanban_backend.get_kanban_db() is kb


def test_seam_import_does_not_import_plugin_system():
    """Reloading the seam with the plugin system absent from sys.modules must not
    pull hermes_cli.plugins in (no import-time plugin dependency)."""
    import importlib
    saved = {k: v for k, v in sys.modules.items()
             if k == "hermes_cli.plugins" or k.startswith("hermes_cli.plugins.")}
    for k in list(saved):
        del sys.modules[k]
    try:
        import hermes_cli.kanban_backend as seam
        importlib.reload(seam)
        assert "hermes_cli.plugins" not in sys.modules
    finally:
        sys.modules.update(saved)


def test_plugin_context_registrar_registers_and_unloads(monkeypatch, tmp_path):
    """PluginContext.register_kanban_backend registers under the plugin key; unload removes it."""
    import hermes_yaml as yaml

    from hermes_cli.plugins import PluginManager

    home = tmp_path / ".hermes"
    plugin_dir = home / "plugins" / "kanbanplug"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "kanbanplug", "version": "1.0"}), encoding="utf-8")
    backend_src = (
        "import types\n"
        "MODULE = types.ModuleType('kanbanplug_backend_module')\n"
        "def _connect(db_path=None, *, board=None):\n"
        "    raise RuntimeError('not opened in test')\n"
        "MODULE.connect = _connect\n"
        "def _connect_closing(*a, **kw):\n"
        "    raise RuntimeError('not opened in test')\n"
        "MODULE.connect_closing = _connect_closing\n"
        "def _write_txn(*a, **kw):\n"
        "    raise RuntimeError('not opened in test')\n"
        "MODULE.write_txn = _write_txn\n"
        "def register(ctx):\n"
        "    ctx.register_kanban_backend(MODULE)\n"
    )
    (plugin_dir / "__init__.py").write_text(backend_src, encoding="utf-8")
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["kanbanplug"]}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    manager = PluginManager()
    manager.discover_and_load()
    assert any(k == "kanbanplug" for k in manager._plugins), dict(manager._plugins)
    assert kanban_backend.registered_backends() == ["kanbanplug"]
    # The registered module is the plugin's own MODULE object.
    registered = kanban_backend._registry.get("kanbanplug")
    assert registered is not None and registered.__name__ == "kanbanplug_backend_module"

    manager.unload()
    assert kanban_backend.registered_backends() == []
