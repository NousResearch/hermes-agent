from __future__ import annotations

from unittest.mock import MagicMock

import agent.file_safety as fs

from plugins.memory.retaindb import RetainDBMemoryProvider


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


def _capture_initialized_client(monkeypatch, tmp_path):
    """Patch _Client/_WriteQueue/get_hermes_home; return a dict capturing args."""
    import hermes_constants

    import plugins.memory.retaindb as retaindb_module

    captured: dict = {}

    class _FakeClient:
        def __init__(self, api_key, base_url, project):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["project"] = project
            self.project = project

    monkeypatch.setattr(retaindb_module, "_Client", _FakeClient)
    monkeypatch.setattr(retaindb_module, "_WriteQueue", lambda *a, **k: MagicMock())
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    return retaindb_module, captured


def test_initialize_reads_base_url_and_project_from_config_yaml(tmp_path, monkeypatch):
    """#68209: non-secret base_url/project come from config.yaml when env is unset."""
    for var in ("RETAINDB_API_KEY", "RETAINDB_BASE_URL", "RETAINDB_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    retaindb_module, captured = _capture_initialized_client(monkeypatch, tmp_path)
    monkeypatch.setattr(
        retaindb_module,
        "_load_retaindb_config",
        lambda: {"base_url": "https://retaindb.example.com/", "project": "cfg-project"},
    )

    RetainDBMemoryProvider().initialize("sess-1")

    assert captured["base_url"] == "https://retaindb.example.com"  # trailing slash stripped
    assert captured["project"] == "cfg-project"


def test_initialize_env_overrides_config_yaml(tmp_path, monkeypatch):
    for var in ("RETAINDB_API_KEY", "RETAINDB_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("RETAINDB_BASE_URL", "https://env.example.com")
    retaindb_module, captured = _capture_initialized_client(monkeypatch, tmp_path)
    monkeypatch.setattr(
        retaindb_module,
        "_load_retaindb_config",
        lambda: {"base_url": "https://cfg.example.com", "project": "cfg-project"},
    )

    RetainDBMemoryProvider().initialize("sess-1")

    assert captured["base_url"] == "https://env.example.com"


def test_initialize_falls_back_to_default_base_url(tmp_path, monkeypatch):
    for var in ("RETAINDB_API_KEY", "RETAINDB_BASE_URL", "RETAINDB_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    retaindb_module, captured = _capture_initialized_client(monkeypatch, tmp_path)
    monkeypatch.setattr(retaindb_module, "_load_retaindb_config", lambda: {})

    RetainDBMemoryProvider().initialize("sess-1")

    assert captured["base_url"] == retaindb_module._DEFAULT_BASE_URL
    assert captured["project"] == "default"
