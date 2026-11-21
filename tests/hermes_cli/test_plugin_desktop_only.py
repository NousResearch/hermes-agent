"""Directory plugins with no Python half (a manifest plus ``desktop/plugin.js`` or a
``dashboard/`` payload, no root ``.py`` files) load cleanly instead of warning
"Failed to load plugin ... No __init__.py" on every start (#132741, #132762).
Real temp home, real plugin directory, real discovery path — no loader mocks."""

from __future__ import annotations

import json
import logging

import hermes_yaml as yaml
import pytest

from hermes_cli.plugins import PluginManager


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A temp HERMES_HOME whose bundled plugin dir is empty, so only the fixture plugin loads."""
    from hermes_cli import plugins as plugins_mod

    home = tmp_path / "home"
    home.mkdir()
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    return home


def _write_plugin(
    home,
    name: str,
    *,
    kind: str | None = None,
    root_py: bool = False,
    with_init: bool = False,
):
    plugin = home / "plugins" / name
    (plugin / "desktop").mkdir(parents=True)
    manifest = {"name": name, "version": "1.0.0", "description": "desktop widget"}
    if kind:
        manifest["kind"] = kind
    (plugin / "plugin.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (plugin / "desktop" / "plugin.js").write_text(
        "// titlebar widget\n", encoding="utf-8"
    )
    if root_py:
        (plugin / "widget.py").write_text(
            "def register(ctx):\n    pass\n", encoding="utf-8"
        )
    if with_init:
        (plugin / "__init__.py").write_text(
            "def register(ctx):\n    pass\n", encoding="utf-8"
        )
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": [name]}}), encoding="utf-8"
    )
    return plugin


def _load(home) -> PluginManager:
    manager = PluginManager()
    manager.discover_and_load()
    return manager


def test_desktop_only_plugin_without_kind_loads_clean(home, caplog):
    # deepseek-whale ships no `kind` field; the file shape (no __init__.py, no root .py) decides.
    _write_plugin(home, "deepseek-widget")
    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        manager = _load(home)
    loaded = manager._plugins["deepseek-widget"]
    assert loaded.enabled
    assert loaded.error is None
    assert [r for r in caplog.records if "deepseek-widget" in r.getMessage()] == []


def test_standalone_kind_desktop_only_plugin_loads_clean(home, caplog):
    _write_plugin(home, "agent-log", kind="standalone")
    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        manager = _load(home)
    loaded = manager._plugins["agent-log"]
    assert loaded.enabled
    assert loaded.error is None
    assert [r for r in caplog.records if "agent-log" in r.getMessage()] == []


def test_directory_plugin_with_root_py_still_requires_init(home, caplog):
    # The skip must not swallow the real failure: a directory that ships Python files but no
    # package marker stays a load error, not a silent desktop-only skip.
    _write_plugin(home, "half-broken", root_py=True)
    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        manager = _load(home)
    loaded = manager._plugins["half-broken"]
    assert not loaded.enabled
    assert loaded.error and "No __init__.py" in loaded.error


def test_dashboard_only_plugin_with_subdir_backend_loads_clean(home, caplog):
    # home-dashboard (#132762): the Python backend lives at dashboard/plugin_api.py, inside a
    # subdirectory the web server imports itself — the plugin root still has no Python half.
    plugin = home / "plugins" / "home-dashboard"
    (plugin / "dashboard").mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        yaml.safe_dump({
            "name": "home-dashboard",
            "version": "1.5.0",
            "description": "home widgets",
        }),
        encoding="utf-8",
    )
    (plugin / "dashboard" / "manifest.json").write_text(
        json.dumps({"name": "home-dashboard", "label": "Home", "api": "plugin_api.py"}),
        encoding="utf-8",
    )
    (plugin / "dashboard" / "plugin_api.py").write_text(
        "router = None  # FastAPI APIRouter in the real plugin\n", encoding="utf-8"
    )
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["home-dashboard"]}}), encoding="utf-8"
    )
    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        manager = _load(home)
    loaded = manager._plugins["home-dashboard"]
    assert loaded.enabled
    assert loaded.error is None
    assert [r for r in caplog.records if "home-dashboard" in r.getMessage()] == []
