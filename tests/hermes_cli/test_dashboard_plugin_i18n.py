"""Dashboard extension manifests carry localization keys alongside source copy."""

import json


def test_presentation_keys_carried_through(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manifest = {
        "name": "localized-nav",
        "label": "Localized Nav",
        "labelKey": "kanban",
        "descriptionKey": "kanban",
        "tab": {"path": "/localized-nav"},
        "entry": "dist/index.js",
    }
    plug_dir = tmp_path / "plugins" / "localized-nav" / "dashboard"
    plug_dir.mkdir(parents=True)
    (plug_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    from hermes_cli import web_server
    web_server._dashboard_plugins_cache = None
    plugins = web_server._get_dashboard_plugins(force_rescan=True)
    entry = next(p for p in plugins if p["name"] == "localized-nav")
    assert entry["label"] == "Localized Nav"
    assert entry["labelKey"] == "kanban"
    assert entry["descriptionKey"] == "kanban"
