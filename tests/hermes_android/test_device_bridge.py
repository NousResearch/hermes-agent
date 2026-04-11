import json

from hermes_android import device_bridge
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
    assert payload["linux_enabled"] is False
    assert payload["workspace_file_count"] == 1
    assert payload["workspace_files"][0]["relative_path"] == "notes.txt"


def test_shared_folder_bridge_wrappers_decode_json(monkeypatch):
    calls = []

    def fake_call(method_name, *args):
        calls.append((method_name, args))
        if method_name == "listEntriesJson":
            return json.dumps({"entries": [{"relative_path": "docs/note.txt"}]})
        if method_name == "readTextFileJson":
            return json.dumps({"content": "hello from shared folder"})
        if method_name == "writeTextFileJson":
            return json.dumps({"success": True, "relative_path": "docs/note.txt"})
        raise AssertionError(method_name)

    monkeypatch.setattr(device_bridge, "_call_shared_folder_bridge", fake_call)

    listed = device_bridge.list_shared_folder_entries("docs", recursive=True, limit=25)
    read_back = device_bridge.read_shared_folder_file("docs/note.txt", max_chars=2048)
    written = device_bridge.write_shared_folder_file("docs/note.txt", "hello", create_directories=True)

    assert listed["entries"][0]["relative_path"] == "docs/note.txt"
    assert read_back["content"] == "hello from shared folder"
    assert written["success"] is True
    assert calls == [
        ("listEntriesJson", ("docs", True, 25)),
        ("readTextFileJson", ("docs/note.txt", 2048)),
        ("writeTextFileJson", ("docs/note.txt", "hello", True)),
    ]


def test_accessibility_bridge_wrappers_decode_json(monkeypatch):
    calls = []

    def fake_call(method_name, *args):
        calls.append((method_name, args))
        if method_name == "snapshotJson":
            return json.dumps({"active_package": "com.example", "nodes": [{"text": "Submit"}]})
        if method_name == "performActionJson":
            return json.dumps({"success": True, "action": args[0], "matched_node": {"text": "Submit"}})
        raise AssertionError(method_name)

    monkeypatch.setattr(device_bridge, "_call_accessibility_bridge", fake_call)

    snapshot = device_bridge.read_accessibility_snapshot(limit=12)
    action = device_bridge.perform_accessibility_action(
        action="click",
        text_contains="submit",
        content_description_contains="",
        view_id="",
        package_name="com.example",
        value="",
        index=1,
    )

    assert snapshot["active_package"] == "com.example"
    assert snapshot["nodes"][0]["text"] == "Submit"
    assert action["success"] is True
    assert action["matched_node"]["text"] == "Submit"
    assert calls == [
        ("snapshotJson", (12,)),
        ("performActionJson", ("click", "submit", "", "", "com.example", "", 1)),
    ]
