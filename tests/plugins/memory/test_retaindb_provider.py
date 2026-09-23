from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import agent.file_safety as fs

import pytest

import plugins.memory.retaindb as retaindb
from plugins.memory.retaindb import RetainDBMemoryProvider


def test_write_queue_closes_owner_connection(tmp_path):
    queue = retaindb._WriteQueue(object(), tmp_path / "retaindb.db")
    owner_conn = queue._local.conn
    worker = retaindb.threading.Thread(target=queue._get_conn)
    worker.start()
    worker.join()
    queue.shutdown()
    assert not queue._connections
    with pytest.raises(sqlite3.ProgrammingError):
        owner_conn.execute("SELECT 1")


def test_write_queue_ignores_enqueue_after_shutdown(tmp_path):
    queue = retaindb._WriteQueue(object(), tmp_path / "retaindb.db")
    queue.shutdown()

    queue.enqueue("user", "session", [])

    assert not queue._connections


def test_prefetch_does_not_spawn_when_previous_batch_is_alive(monkeypatch):
    provider = RetainDBMemoryProvider()
    provider._client = object()

    class _RunningThread:
        def join(self, timeout):
            pass

        def is_alive(self):
            return True

    previous = _RunningThread()
    provider._prefetch_threads = [previous]
    created = []

    class _Thread:
        def __init__(self, *args, **kwargs):
            created.append((args, kwargs))

        def start(self):
            pass

    monkeypatch.setattr(retaindb.threading, "Thread", _Thread)
    provider.queue_prefetch("query")
    assert provider._prefetch_threads == [previous]
    assert not created


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


def test_retaindb_config_loader_uses_readonly_config(monkeypatch):
    import hermes_cli.config as config_mod
    import plugins.memory.retaindb as retaindb_module

    backing_config = {
        "memory": {
            "retaindb": {
                "base_url": "https://saved.example",
                "project": "saved-project",
            }
        }
    }
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: backing_config)
    monkeypatch.setattr(
        config_mod,
        "load_config",
        MagicMock(side_effect=AssertionError("read-only provider path must not load a mutable copy")),
    )

    config = retaindb_module._load_retaindb_config()

    assert config == backing_config["memory"]["retaindb"]
    assert config is not backing_config["memory"]["retaindb"]


def test_initialize_reads_real_dashboard_config_file(tmp_path, monkeypatch):
    for var in ("RETAINDB_API_KEY", "RETAINDB_BASE_URL", "RETAINDB_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / "config.yaml").write_text(
        """\
memory:
  provider: retaindb
  retaindb:
    base_url: https://retaindb.saved.example/
    project: dashboard-project
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _retaindb_module, captured = _capture_initialized_client(monkeypatch, tmp_path)

    RetainDBMemoryProvider().initialize("sess-1")

    assert captured["base_url"] == "https://retaindb.saved.example"
    assert captured["project"] == "dashboard-project"


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


def test_initialize_combines_scoped_secret_with_dashboard_config(tmp_path, monkeypatch):
    """Rebase regression: scoped secrets and non-secret config must coexist."""
    from agent.secret_scope import (
        is_multiplex_active,
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )

    monkeypatch.setenv("RETAINDB_API_KEY", "env-other-profile")
    monkeypatch.delenv("RETAINDB_BASE_URL", raising=False)
    monkeypatch.delenv("RETAINDB_PROJECT", raising=False)
    retaindb_module, captured = _capture_initialized_client(monkeypatch, tmp_path)
    monkeypatch.setattr(
        retaindb_module,
        "_load_retaindb_config",
        lambda: {"base_url": "https://dashboard.example.com/", "project": "dashboard-project"},
    )

    previous_multiplex_state = is_multiplex_active()
    set_multiplex_active(True)
    token = set_secret_scope({"RETAINDB_API_KEY": "scoped-key"})
    try:
        RetainDBMemoryProvider().initialize("sess-1")
    finally:
        reset_secret_scope(token)
        set_multiplex_active(previous_multiplex_state)

    assert captured == {
        "api_key": "scoped-key",
        "base_url": "https://dashboard.example.com",
        "project": "dashboard-project",
    }


def test_initialize_falls_back_to_default_base_url(tmp_path, monkeypatch):
    for var in ("RETAINDB_API_KEY", "RETAINDB_BASE_URL", "RETAINDB_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    retaindb_module, captured = _capture_initialized_client(monkeypatch, tmp_path)
    monkeypatch.setattr(retaindb_module, "_load_retaindb_config", lambda: {})

    RetainDBMemoryProvider().initialize("sess-1")

    assert captured["base_url"] == retaindb_module._DEFAULT_BASE_URL
    assert captured["project"] == "default"
