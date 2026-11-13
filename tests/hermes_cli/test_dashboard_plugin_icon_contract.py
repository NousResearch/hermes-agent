"""Dashboard discovery must keep icon metadata decodable by typed clients (#147777)."""

import json

import pytest

from hermes_cli import web_server_dashboard as dashboard


def _write_plugin(root, name, **fields):
    directory = root / name / "dashboard"
    directory.mkdir(parents=True)
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"name": name, **fields}), encoding="utf-8")
    return manifest


@pytest.mark.parametrize("icon", [
    {"type": "image", "src": "dist/logo.png", "alt": "Logo"},
    None, [], 42, True,
])
def test_non_string_icon_does_not_poison_discovered_manifests(tmp_path, caplog, icon):
    manifest = _write_plugin(tmp_path, "custom-icon", icon=icon)
    _write_plugin(tmp_path, "other-plugin", icon="Activity")

    plugins = dashboard._discover_dashboard_plugins([(tmp_path, "user")])
    public = json.loads(json.dumps([dashboard._strip_dashboard_manifest(p) for p in plugins]))
    by_name = {p["name"]: p for p in public}

    assert by_name["custom-icon"]["icon"] == "Puzzle"
    assert by_name["other-plugin"]["icon"] == "Activity"
    assert all(isinstance(p["icon"], str) for p in public)
    assert all(not key.startswith("_") for p in public for key in p)
    assert json.loads(manifest.read_text(encoding="utf-8"))["icon"] == icon
    assert "custom-icon" in caplog.text
    assert "icon" in caplog.text and "string" in caplog.text and "Puzzle" in caplog.text


def test_string_icons_and_missing_icon_keep_existing_discovery_behavior(tmp_path, caplog):
    _write_plugin(tmp_path, "named", icon="BarChart3")
    _write_plugin(tmp_path, "unknown-name", icon="FutureLucideIcon")
    _write_plugin(tmp_path, "default")

    plugins = dashboard._discover_dashboard_plugins([(tmp_path, "user")])

    assert {p["name"]: p["icon"] for p in plugins} == {
        "named": "BarChart3", "unknown-name": "FutureLucideIcon", "default": "Puzzle",
    }
    assert not caplog.records
