"""A nested directory plugin and a same-name entry point are one registry row."""
from __future__ import annotations

from importlib.metadata import EntryPoint
from types import SimpleNamespace

import pytest

from hermes_cli.plugins import ENTRY_POINTS_GROUP, PluginManager, _reset_plugin_managers_for_tests


@pytest.fixture(autouse=True)
def _reset_managers():
    _reset_plugin_managers_for_tests()
    yield
    _reset_plugin_managers_for_tests()


def _write_plugin(directory, description: str) -> None:
    directory.mkdir(parents=True)
    (directory / "plugin.yaml").write_text(
        f"name: demo\nversion: '0.0.1'\ndescription: {description}\n", encoding="utf-8")
    (directory / "__init__.py").write_text("def register(ctx):\n    return None\n", encoding="utf-8")


def _load(monkeypatch, tmp_path, entry_names: list[str], *, nested: bool = True, bundled: bool = False):
    bundled_root = tmp_path / "bundled"
    bundled_root.mkdir()
    if bundled:
        _write_plugin(bundled_root / "tools" / "demo", "bundled")
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled_root))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    if nested:
        _write_plugin(home / "plugins" / "demo" / "demo_pkg", "nested")
    enabled = "".join(f"    - {name}\n" for name in ("demo", *entry_names))
    (home / "config.yaml").write_text(
        f"plugins:\n  enabled:\n{enabled}  load_timeout_seconds: 0\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    ep_dir = tmp_path / "eps"
    ep_dir.mkdir()
    points = []
    for name in entry_names:
        (ep_dir / f"{name}_alias.py").write_text(
            "def register(ctx):\n    return None\n", encoding="utf-8")
        points.append(EntryPoint(name=name, value=f"{name}_alias:register", group=ENTRY_POINTS_GROUP))
    monkeypatch.syspath_prepend(str(ep_dir))
    monkeypatch.setattr(
        "hermes_cli.plugins_discovery.importlib.metadata.entry_points",
        lambda: SimpleNamespace(select=lambda group: points if group == ENTRY_POINTS_GROUP else []))
    manager = PluginManager(str(home))
    manager.discover_and_load(force=True)
    return manager


def test_nested_same_name_entrypoint_is_one_registry_row(monkeypatch, tmp_path):
    rows = _load(monkeypatch, tmp_path, ["demo"]).list_plugins()
    assert [row["key"] for row in rows if row["name"] == "demo"] == ["demo/demo_pkg"]


def test_different_name_entrypoint_stays_loaded(monkeypatch, tmp_path):
    rows = _load(monkeypatch, tmp_path, ["other"]).list_plugins()
    other = next(row for row in rows if row["name"] == "other")
    assert other["source"] == "entrypoint" and other["enabled"] and not other["error"]


def test_bundled_tools_demo_and_entrypoint_demo_both_register(monkeypatch, tmp_path):
    rows = _load(monkeypatch, tmp_path, ["demo"], nested=False, bundled=True).list_plugins()
    bundled_row = next(row for row in rows if row["key"] == "tools/demo")
    entry = next(row for row in rows if row["key"] == "demo")
    assert bundled_row["source"] == "bundled" and bundled_row["enabled"] and not bundled_row["error"]
    assert entry["source"] == "entrypoint" and entry["enabled"] and not entry["error"]


def test_nested_disable_matches_gate_manifest(monkeypatch):
    from hermes_cli.plugins_discovery import IN_BOTH_LISTS, directory_entrypoint_precedence, gate_manifest
    from hermes_cli.plugins_manifest import PluginManifest

    manifest = PluginManifest(name="langfuse", source="user", key="tools/langfuse", path=".")
    enabled = {"langfuse", "tools/langfuse"}
    cases = (set(), {"langfuse"}, {"tools/langfuse"}, {"observability/langfuse"}, {"other"})
    for disabled in cases:
        monkeypatch.setattr(
            "hermes_cli.plugins_discovery._get_disabled_plugins",
            lambda disabled=disabled: set(disabled))
        _keys, names = directory_entrypoint_precedence([manifest])
        verdict = gate_manifest(manifest, set(disabled), enabled)
        gate_disabled = verdict.error in {"disabled via config", IN_BOTH_LISTS}
        assert ("langfuse" not in names) is gate_disabled
