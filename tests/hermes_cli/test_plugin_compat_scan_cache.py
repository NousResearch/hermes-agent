"""``scan_plugin`` caches its AST scan on the plugin dir's file signature.

A multiplex gateway runs plugin discovery once per served profile; before the cache every profile
re-parsed every external plugin's source (~0.4s × profiles on the gateway boot path).
"""
from __future__ import annotations

from pathlib import Path

from hermes_cli import plugin_compat


def _install_plugin(root: Path) -> Path:
    plugin = root / "demo-plugin"
    plugin.mkdir()
    (plugin / "__init__.py").write_text("from old.facade import thing\n", encoding="utf-8")
    return plugin


def test_scan_plugin_parses_each_plugin_once_until_its_files_change(tmp_path, monkeypatch) -> None:
    manifest = {"old.facade": {"thing": "new.home.thing"}}
    monkeypatch.setattr(plugin_compat, "load_manifest", lambda: manifest)
    monkeypatch.setattr(plugin_compat, "_scan_cache", {})
    parses = []
    real_scan_source = plugin_compat.scan_source
    monkeypatch.setattr(plugin_compat, "scan_source",
                        lambda src, rel, m: parses.append(rel) or real_scan_source(src, rel, m))
    plugin = _install_plugin(tmp_path)

    first = plugin_compat.scan_plugin(plugin)
    second = plugin_compat.scan_plugin(plugin)          # a second profile discovering the same plugin
    assert [h.old for h in first] == ["old.facade.thing"]
    assert second == first
    assert parses == ["__init__.py"], "second discovery must reuse the first scan"

    (plugin / "__init__.py").write_text("from old.facade import thing, other\n", encoding="utf-8")
    third = plugin_compat.scan_plugin(plugin)           # size changed -> rescanned
    assert parses == ["__init__.py", "__init__.py"]
    assert [h.old for h in third] == ["old.facade.thing"]

    # A caller-supplied manifest is never served from (or written to) the cache.
    plugin_compat.scan_plugin(plugin, {"old.facade": {"other": "x"}})
    assert len(parses) == 3
