from __future__ import annotations

from unittest.mock import MagicMock

import agent.file_safety as fs

from plugins.memory.retaindb import RetainDBMemoryProvider


def test_tools_only_initialize_skips_automatic_startup_writes(tmp_path, monkeypatch):
    (tmp_path / "SOUL.md").write_text("# Test identity", encoding="utf-8")
    monkeypatch.setenv("RETAINDB_API_KEY", "test-key")
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    client = MagicMock()
    queue_cls = MagicMock()
    thread_cls = MagicMock()
    monkeypatch.setattr("plugins.memory.retaindb._Client", lambda *args: client)
    monkeypatch.setattr("plugins.memory.retaindb._WriteQueue", queue_cls)
    monkeypatch.setattr("plugins.memory.retaindb.threading.Thread", thread_cls)

    provider = RetainDBMemoryProvider()
    provider.initialize(
        "cron-session",
        memory_provider_mode="tools",
        hermes_home=str(tmp_path),
    )

    assert provider._client is client
    assert provider._queue is None
    queue_cls.assert_not_called()
    thread_cls.assert_not_called()


def test_upload_file_rejects_hermes_credential_store(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    auth_json = hermes_home / "auth.json"
    auth_json.write_text('{"OPENAI_API_KEY":"sk-test-secret"}', encoding="utf-8")
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: hermes_home)

    provider = RetainDBMemoryProvider()
    provider._client = MagicMock()

    result = provider._dispatch("retaindb_upload_file", {"local_path": str(auth_json)})

    assert "error" in result
    assert "credential store" in result["error"]
    provider._client.upload_file.assert_not_called()


def test_upload_file_allows_regular_file(tmp_path):
    note = tmp_path / "note.md"
    note.write_text("# Note\n", encoding="utf-8")
    provider = RetainDBMemoryProvider()
    provider._client = MagicMock()
    provider._client.upload_file.return_value = {
        "file": {"id": "file-1", "name": "note.md"},
    }

    result = provider._dispatch("retaindb_upload_file", {"local_path": str(note)})

    provider._client.upload_file.assert_called_once()
    assert provider._client.upload_file.call_args.args[0] == note.read_bytes()
    assert result["file"]["id"] == "file-1"
