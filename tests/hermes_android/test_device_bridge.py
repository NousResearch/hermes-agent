import json

from hermes_android.device_bridge import read_device_capabilities


def test_read_device_capabilities_defaults_to_workspace(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    payload = read_device_capabilities()

    assert payload["workspace_path"] == str(hermes_home / "workspace")
    assert payload["workspace_file_count"] == 0
    assert payload["workspace_files"] == []
    assert payload["accessibility_enabled"] is False
    assert payload["shared_tree_uri"] == ""


def test_read_device_capabilities_merges_android_state_file(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    workspace = hermes_home / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "notes.txt").write_text("hello", encoding="utf-8")
    (hermes_home / "android-device-state.json").write_text(
        json.dumps(
            {
                "shared_tree_uri": "content://example/tree/1",
                "shared_tree_label": "Docs",
                "accessibility_enabled": True,
                "accessibility_connected": True,
                "available_global_actions": ["home", "back"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    payload = read_device_capabilities()

    assert payload["shared_tree_uri"] == "content://example/tree/1"
    assert payload["shared_tree_label"] == "Docs"
    assert payload["accessibility_enabled"] is True
    assert payload["accessibility_connected"] is True
    assert payload["available_global_actions"] == ["home", "back"]
    assert payload["workspace_file_count"] == 1
    assert payload["workspace_files"][0]["relative_path"] == "notes.txt"
