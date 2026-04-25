import json
import subprocess
import sys


def test_capabilities_json_declares_launch_metadata_support():
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "capabilities", "--json"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schemaVersion"] == 1
    assert payload["cliVersion"]
    assert payload["capabilities"]["supportsLaunchMetadataArg"] is True
    assert payload["capabilities"]["supportsLaunchMetadataEnv"] is True
    assert payload["capabilities"]["supportsResume"] is True


def test_chat_accepts_launch_metadata_flag():
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "chat", "--launch-metadata", "metadata.json", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--launch-metadata" in result.stdout
    assert "unrecognized arguments" not in result.stderr


def test_launch_metadata_loader_reads_runtime_metadata_without_history(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    metadata = tmp_path / "launch.json"
    metadata.write_text(json.dumps({
        "version": 1,
        "createdAt": "2026-04-23T00:00:00.000Z",
        "forgeSessionId": "forge-session",
        "taskRunId": "task-run",
        "workspace": {"runtimePath": "/repo"},
        "selectedFilePaths": [],
        "attachmentPaths": [],
        "imagePaths": [],
        "bridgeAvailability": {"enabled": False, "available": False, "capabilities": []},
        "cliSession": {"status": "fresh"},
    }), encoding="utf-8")

    import cli
    loaded, diagnostic = cli._load_launch_metadata(str(metadata))

    assert diagnostic["status"] == "loaded"
    assert diagnostic["source"] == "arg"
    assert loaded["forgeSessionId"] == "forge-session"
    assert "workspace" in diagnostic["fields"]
