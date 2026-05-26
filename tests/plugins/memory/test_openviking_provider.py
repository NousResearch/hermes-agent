import json
import os
import stat
import zipfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import plugins.memory.openviking as openviking_module
from plugins.memory.openviking import OpenVikingMemoryProvider, _VikingClient


def _clear_openviking_env(monkeypatch):
    for key in (
        "OPENVIKING_ENDPOINT",
        "OPENVIKING_API_KEY",
        "OPENVIKING_ACCOUNT",
        "OPENVIKING_USER",
        "OPENVIKING_AGENT",
        "OPENVIKING_CLI_CONFIG_FILE",
    ):
        monkeypatch.delenv(key, raising=False)


def _prompt_from_values(values: dict[str, str], *, forbidden: set[str] | None = None):
    forbidden = forbidden or set()

    def _prompt(label, default=None, secret=False):
        if label in forbidden:
            raise AssertionError(f"{label} should not be prompted")
        return values.get(label, default or "")

    return _prompt


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_openviking_env_writer_restricts_file_permissions(tmp_path):
    env_path = tmp_path / ".env"

    openviking_module._write_env_vars(env_path, {"OPENVIKING_API_KEY": "secret"})

    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_ovcli_config_writer_restricts_file_permissions(tmp_path):
    config_path = tmp_path / "ovcli.conf"

    openviking_module._write_ovcli_config(
        config_path,
        {"endpoint": "http://remote.example", "api_key": "secret"},
    )

    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600


def test_linked_ovcli_config_is_read_at_runtime(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    ovcli_path = tmp_path / "ovcli.conf"
    ovcli_path.write_text(
        json.dumps({
            "url": "http://openviking-one.local",
            "api_key": "key-one",
            "account": "acct-one",
            "user": "alice",
            "agent_id": "agent-one",
        }),
        encoding="utf-8",
    )
    provider_config = {"use_ovcli_config": True, "ovcli_config_path": str(ovcli_path)}

    settings = openviking_module._resolve_connection_settings(provider_config)

    assert settings == {
        "endpoint": "http://openviking-one.local",
        "api_key": "key-one",
        "account": "acct-one",
        "user": "alice",
        "agent": "agent-one",
    }

    ovcli_path.write_text(
        json.dumps({
            "url": "http://openviking-two.local",
            "api_key": "key-two",
            "agent_id": "agent-two",
        }),
        encoding="utf-8",
    )

    settings = openviking_module._resolve_connection_settings(provider_config)

    assert settings == {
        "endpoint": "http://openviking-two.local",
        "api_key": "key-two",
        "account": "",
        "user": "",
        "agent": "agent-two",
    }


def test_openviking_env_overrides_linked_ovcli_config(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    ovcli_path = tmp_path / "ovcli.conf"
    ovcli_path.write_text(
        json.dumps({
            "url": "http://openviking.local",
            "api_key": "file-key",
            "account": "file-account",
            "user": "file-user",
            "agent_id": "file-agent",
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://env.local")
    monkeypatch.setenv("OPENVIKING_API_KEY", "env-key")
    monkeypatch.setenv("OPENVIKING_ACCOUNT", "env-account")
    monkeypatch.setenv("OPENVIKING_USER", "env-user")
    monkeypatch.setenv("OPENVIKING_AGENT", "env-agent")

    settings = openviking_module._resolve_connection_settings({
        "use_ovcli_config": True,
        "ovcli_config_path": str(ovcli_path),
    })

    assert settings == {
        "endpoint": "http://env.local",
        "api_key": "env-key",
        "account": "env-account",
        "user": "env-user",
        "agent": "env-agent",
    }


def test_post_setup_link_existing_ovcli_clears_hermes_env(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    env_path = hermes_home / ".env"
    env_path.write_text(
        "OPENVIKING_ENDPOINT=http://old.local\n"
        "OPENVIKING_ACCOUNT=old-account\n"
        "OTHER_KEY=keep\n",
        encoding="utf-8",
    )
    ovcli_path = tmp_path / "ovcli.conf"
    original_ovcli = json.dumps({"url": "http://openviking.local"})
    ovcli_path.write_text(original_ovcli, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(ovcli_path))

    from hermes_cli import memory_setup

    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: 0)
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    assert config["memory"]["provider"] == "openviking"
    assert config["memory"]["openviking"]["use_ovcli_config"] is True
    assert config["memory"]["openviking"]["ovcli_config_path"] == str(ovcli_path)
    env_text = env_path.read_text(encoding="utf-8")
    assert "OPENVIKING_" not in env_text
    assert "OTHER_KEY=keep" in env_text
    assert ovcli_path.read_text(encoding="utf-8") == original_ovcli


def test_post_setup_copy_existing_ovcli_writes_hermes_env(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    ovcli_path = tmp_path / "ovcli.conf"
    original_ovcli = json.dumps({
        "url": "http://openviking.local",
        "api_key": "test-key",
        "account": "acct",
        "user": "alice",
        "agent_id": "agent",
    })
    ovcli_path.write_text(original_ovcli, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(ovcli_path))

    from hermes_cli import memory_setup

    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: 1)
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    assert config["memory"]["provider"] == "openviking"
    assert config["memory"]["openviking"]["use_ovcli_config"] is False
    env_text = (hermes_home / ".env").read_text(encoding="utf-8")
    assert "OPENVIKING_ENDPOINT=http://openviking.local" in env_text
    assert "OPENVIKING_API_KEY=test-key" in env_text
    assert "OPENVIKING_ACCOUNT=acct" in env_text
    assert "OPENVIKING_USER=alice" in env_text
    assert "OPENVIKING_AGENT=agent" in env_text
    assert ovcli_path.read_text(encoding="utf-8") == original_ovcli


def test_post_setup_manual_remote_root_writes_ovcli_and_links(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    env_path = hermes_home / ".env"
    env_path.write_text("OPENVIKING_ENDPOINT=http://old.local\n", encoding="utf-8")
    ovcli_path = tmp_path / "ovcli.conf"
    ovcli_path.write_text(json.dumps({"url": "http://old.local"}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(ovcli_path))

    from hermes_cli import memory_setup

    choices = iter([2, 1, 0])
    monkeypatch.setattr(
        memory_setup,
        "_curses_select",
        lambda *args, **kwargs: next(choices),
    )
    monkeypatch.setattr(
        memory_setup,
        "_prompt",
        _prompt_from_values({
            "OpenViking server URL": "https://openviking.example",
            "OpenViking root API key": "root-secret",
            "OpenViking account": "acct",
            "OpenViking user": "alice",
            "OpenViking agent": "agent",
        }),
    )
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    assert config["memory"]["provider"] == "openviking"
    assert config["memory"]["openviking"]["use_ovcli_config"] is True
    assert config["memory"]["openviking"]["ovcli_config_path"] == str(ovcli_path)
    assert env_path.read_text(encoding="utf-8") == ""
    data = json.loads(ovcli_path.read_text(encoding="utf-8"))
    assert data == {
        "url": "https://openviking.example",
        "api_key": "root-secret",
        "root_api_key": "root-secret",
        "account": "acct",
        "user": "alice",
        "agent_id": "agent",
    }


def test_post_setup_manual_remote_user_keeps_only_hermes_env(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    ovcli_path = tmp_path / "ovcli.conf"
    original_ovcli = json.dumps({"url": "http://old.local"})
    ovcli_path.write_text(original_ovcli, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(ovcli_path))

    from hermes_cli import memory_setup

    choices = iter([2, 0, 1])
    monkeypatch.setattr(
        memory_setup,
        "_curses_select",
        lambda *args, **kwargs: next(choices),
    )
    monkeypatch.setattr(
        memory_setup,
        "_prompt",
        _prompt_from_values(
            {
                "OpenViking server URL": "https://openviking.example",
                "OpenViking user API key": "user-secret",
                "OpenViking agent": "agent",
            },
            forbidden={
                "OpenViking account",
                "OpenViking root API key",
                "OpenViking user",
            },
        ),
    )
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    assert config["memory"]["provider"] == "openviking"
    assert config["memory"]["openviking"]["use_ovcli_config"] is False
    assert ovcli_path.read_text(encoding="utf-8") == original_ovcli
    env_text = (hermes_home / ".env").read_text(encoding="utf-8")
    assert "OPENVIKING_ENDPOINT=https://openviking.example" in env_text
    assert "OPENVIKING_API_KEY=user-secret" in env_text
    assert "OPENVIKING_AGENT=agent" in env_text
    assert "OPENVIKING_ACCOUNT" not in env_text
    assert "OPENVIKING_USER" not in env_text


def test_post_setup_manual_remote_requires_api_key(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    ovcli_path = tmp_path / "ovcli.conf"
    original_ovcli = json.dumps({"url": "http://old.local"})
    ovcli_path.write_text(original_ovcli, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(ovcli_path))

    from hermes_cli import config as hermes_config
    from hermes_cli import memory_setup

    save_config = MagicMock()
    monkeypatch.setattr(hermes_config, "save_config", save_config)
    choices = iter([2, 0])
    monkeypatch.setattr(
        memory_setup,
        "_curses_select",
        lambda *args, **kwargs: next(choices),
    )
    monkeypatch.setattr(
        memory_setup,
        "_prompt",
        _prompt_from_values({
            "OpenViking server URL": "https://openviking.example",
            "OpenViking user API key": "",
        }),
    )
    config = {"memory": {"provider": "builtin"}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    save_config.assert_not_called()
    assert config == {"memory": {"provider": "builtin"}}
    assert ovcli_path.read_text(encoding="utf-8") == original_ovcli
    assert not (hermes_home / ".env").exists()


def test_post_setup_manual_root_requires_account_and_user(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    ovcli_path = tmp_path / "ovcli.conf"
    original_ovcli = json.dumps({"url": "http://old.local"})
    ovcli_path.write_text(original_ovcli, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(ovcli_path))

    from hermes_cli import config as hermes_config
    from hermes_cli import memory_setup

    save_config = MagicMock()
    choices = iter([2, 1])
    monkeypatch.setattr(hermes_config, "save_config", save_config)
    monkeypatch.setattr(
        memory_setup,
        "_curses_select",
        lambda *args, **kwargs: next(choices),
    )
    monkeypatch.setattr(
        memory_setup,
        "_prompt",
        _prompt_from_values({
            "OpenViking server URL": "https://openviking.example",
            "OpenViking root API key": "root-secret",
            "OpenViking account": "",
            "OpenViking user": "alice",
        }),
    )
    config = {"memory": {"provider": "builtin"}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    save_config.assert_not_called()
    assert config == {"memory": {"provider": "builtin"}}
    assert ovcli_path.read_text(encoding="utf-8") == original_ovcli
    assert not (hermes_home / ".env").exists()


def test_post_setup_manual_local_allows_blank_api_key(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    ovcli_path = tmp_path / "ovcli.conf"
    original_ovcli = json.dumps({"url": "http://old.local"})
    ovcli_path.write_text(original_ovcli, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(ovcli_path))

    from hermes_cli import memory_setup

    choices = iter([2, 0, 1])
    monkeypatch.setattr(
        memory_setup,
        "_curses_select",
        lambda *args, **kwargs: next(choices),
    )
    monkeypatch.setattr(
        memory_setup,
        "_prompt",
        _prompt_from_values(
            {
                "OpenViking server URL": "http://localhost:1933",
                "OpenViking agent": "agent",
            },
            forbidden={
                "OpenViking account",
                "OpenViking root API key",
                "OpenViking user",
                "OpenViking user API key",
            },
        ),
    )
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    assert config["memory"]["provider"] == "openviking"
    assert config["memory"]["openviking"]["use_ovcli_config"] is False
    assert ovcli_path.read_text(encoding="utf-8") == original_ovcli
    env_text = (hermes_home / ".env").read_text(encoding="utf-8")
    assert "OPENVIKING_ENDPOINT=http://localhost:1933" in env_text
    assert "OPENVIKING_AGENT=agent" in env_text
    assert "OPENVIKING_API_KEY" not in env_text
    assert "OPENVIKING_ACCOUNT" not in env_text
    assert "OPENVIKING_USER" not in env_text


def test_post_setup_cancel_existing_ovcli_writes_nothing(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    env_path = hermes_home / ".env"
    original_env = "OPENVIKING_ENDPOINT=http://old.local\nOTHER_KEY=keep\n"
    env_path.write_text(original_env, encoding="utf-8")
    ovcli_path = tmp_path / "ovcli.conf"
    ovcli_path.write_text(json.dumps({"url": "http://openviking.local"}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(ovcli_path))

    from hermes_cli import config as hermes_config
    from hermes_cli import memory_setup

    save_config = MagicMock()
    monkeypatch.setattr(hermes_config, "save_config", save_config)
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: -1)
    config = {"memory": {"provider": "builtin"}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    save_config.assert_not_called()
    assert config == {"memory": {"provider": "builtin"}}
    assert env_path.read_text(encoding="utf-8") == original_env


def test_post_setup_invalid_existing_ovcli_writes_nothing(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    env_path = hermes_home / ".env"
    original_env = "OPENVIKING_ENDPOINT=http://old.local\nOTHER_KEY=keep\n"
    env_path.write_text(original_env, encoding="utf-8")
    ovcli_path = tmp_path / "ovcli.conf"
    ovcli_path.write_text("{", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(ovcli_path))

    from hermes_cli import config as hermes_config
    from hermes_cli import memory_setup

    save_config = MagicMock()
    monkeypatch.setattr(hermes_config, "save_config", save_config)
    monkeypatch.setattr(
        memory_setup,
        "_curses_select",
        MagicMock(side_effect=AssertionError("picker should not open for invalid ovcli.conf")),
    )
    config = {"memory": {"provider": "builtin"}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    save_config.assert_not_called()
    assert config == {"memory": {"provider": "builtin"}}
    assert env_path.read_text(encoding="utf-8") == original_env


def test_post_setup_creates_minimal_ovcli_and_links(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    ovcli_path = tmp_path / "missing" / "ovcli.conf"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(ovcli_path))

    from hermes_cli import memory_setup

    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        memory_setup,
        "_prompt",
        lambda label, default=None, secret=False: default or "",
    )
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    assert config["memory"]["provider"] == "openviking"
    assert config["memory"]["openviking"]["use_ovcli_config"] is True
    data = json.loads(ovcli_path.read_text(encoding="utf-8"))
    assert data == {
        "url": "http://127.0.0.1:1933",
        "agent_id": "hermes",
    }
    env_path = hermes_home / ".env"
    if env_path.exists():
        assert env_path.read_text(encoding="utf-8") == ""


def test_post_setup_cancel_missing_ovcli_does_not_prompt_or_create(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    ovcli_path = tmp_path / "missing" / "ovcli.conf"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(ovcli_path))

    from hermes_cli import config as hermes_config
    from hermes_cli import memory_setup

    save_config = MagicMock()
    monkeypatch.setattr(hermes_config, "save_config", save_config)
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: -1)
    monkeypatch.setattr(
        memory_setup,
        "_prompt",
        MagicMock(side_effect=AssertionError("prompts should not run after cancel")),
    )
    config = {"memory": {"provider": "builtin"}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    save_config.assert_not_called()
    assert config == {"memory": {"provider": "builtin"}}
    assert not ovcli_path.exists()
    assert not (hermes_home / ".env").exists()


def test_tool_search_sorts_by_raw_score_across_buckets():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.post.return_value = {
        "result": {
            "memories": [
                {"uri": "viking://memories/1", "score": 0.9003, "abstract": "memory result"},
            ],
            "resources": [
                {"uri": "viking://resources/1", "score": 0.9004, "abstract": "resource result"},
            ],
            "skills": [
                {"uri": "viking://skills/1", "score": 0.8999, "abstract": "skill result"},
            ],
            "total": 3,
        }
    }

    result = json.loads(provider._tool_search({"query": "ranking"}))

    assert [entry["uri"] for entry in result["results"]] == [
        "viking://resources/1",
        "viking://memories/1",
        "viking://skills/1",
    ]
    assert [entry["score"] for entry in result["results"]] == [0.9, 0.9, 0.9]
    assert result["total"] == 3


def test_tool_search_sorts_missing_raw_score_after_negative_scores():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.post.return_value = {
        "result": {
            "memories": [
                {"uri": "viking://memories/missing", "abstract": "missing score"},
            ],
            "resources": [
                {"uri": "viking://resources/negative", "score": -0.25, "abstract": "negative score"},
            ],
            "skills": [
                {"uri": "viking://skills/positive", "score": 0.1, "abstract": "positive score"},
            ],
            "total": 3,
        }
    }

    result = json.loads(provider._tool_search({"query": "ranking"}))

    assert [entry["uri"] for entry in result["results"]] == [
        "viking://skills/positive",
        "viking://memories/missing",
        "viking://resources/negative",
    ]
    assert [entry["score"] for entry in result["results"]] == [0.1, 0.0, -0.25]
    assert result["total"] == 3


def test_tool_add_resource_uploads_existing_local_file(tmp_path):
    sample = tmp_path / "sample.md"
    sample.write_text("# Local resource\n", encoding="utf-8")
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.upload_temp_file.return_value = "upload_sample.md"
    provider._client.post.return_value = {
        "status": "ok",
        "result": {"root_uri": "viking://resources/sample"},
    }

    result = json.loads(provider._tool_add_resource({
        "url": str(sample),
        "reason": "local test",
        "wait": True,
    }))

    provider._client.upload_temp_file.assert_called_once_with(sample)
    provider._client.post.assert_called_once_with("/api/v1/resources", {
        "reason": "local test",
        "wait": True,
        "source_name": "sample.md",
        "temp_file_id": "upload_sample.md",
    })
    assert result["status"] == "added"
    assert result["root_uri"] == "viking://resources/sample"


def test_tool_add_resource_uploads_file_uri(tmp_path):
    sample = tmp_path / "sample.md"
    sample.write_text("# Local resource\n", encoding="utf-8")
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.upload_temp_file.return_value = "upload_sample.md"
    provider._client.post.return_value = {
        "status": "ok",
        "result": {"root_uri": "viking://resources/sample"},
    }

    result = json.loads(provider._tool_add_resource({
        "url": sample.as_uri(),
        "reason": "file uri test",
    }))

    provider._client.upload_temp_file.assert_called_once_with(sample)
    provider._client.post.assert_called_once_with("/api/v1/resources", {
        "reason": "file uri test",
        "source_name": "sample.md",
        "temp_file_id": "upload_sample.md",
    })
    assert result["status"] == "added"
    assert result["root_uri"] == "viking://resources/sample"


def test_tool_add_resource_uploads_existing_local_directory_and_cleans_zip(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    nested = docs / "nested"
    nested.mkdir()
    (nested / "api.md").write_text("# API\n", encoding="utf-8")
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    uploaded_paths = []
    provider._client.upload_temp_file.side_effect = (
        lambda path: uploaded_paths.append(path) or "upload_docs.zip"
    )
    provider._client.post.return_value = {
        "status": "ok",
        "result": {"root_uri": "viking://resources/docs"},
    }

    result = json.loads(provider._tool_add_resource({
        "url": str(docs),
        "reason": "directory test",
        "wait": True,
    }))

    assert uploaded_paths
    assert uploaded_paths[0].suffix == ".zip"
    assert not uploaded_paths[0].exists()
    provider._client.post.assert_called_once_with("/api/v1/resources", {
        "reason": "directory test",
        "wait": True,
        "source_name": "docs",
        "temp_file_id": "upload_docs.zip",
    })
    assert result["status"] == "added"
    assert result["root_uri"] == "viking://resources/docs"


def test_tool_add_resource_directory_zip_skips_symlink_escape(tmp_path):
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("do not upload\n", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    link = docs / "leak.txt"
    try:
        link.symlink_to(secret)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")

    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    archive_entries = {}

    def inspect_upload(path):
        with zipfile.ZipFile(path) as archive:
            archive_entries["names"] = archive.namelist()
            archive_entries["payloads"] = {
                name: archive.read(name)
                for name in archive.namelist()
            }
        return "upload_docs.zip"

    provider._client.upload_temp_file.side_effect = inspect_upload
    provider._client.post.return_value = {
        "status": "ok",
        "result": {"root_uri": "viking://resources/docs"},
    }

    json.loads(provider._tool_add_resource({"url": str(docs)}))

    assert archive_entries["names"] == ["guide.md"]
    assert b"do not upload" not in b"".join(archive_entries["payloads"].values())


def test_tool_add_resource_cleans_local_directory_zip_when_add_fails(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    uploaded_paths = []
    provider._client.upload_temp_file.side_effect = (
        lambda path: uploaded_paths.append(path) or "upload_docs.zip"
    )
    provider._client.post.side_effect = RuntimeError("add failed")

    with pytest.raises(RuntimeError, match="add failed"):
        provider._tool_add_resource({"url": str(docs)})

    assert uploaded_paths
    assert not uploaded_paths[0].exists()


def test_tool_add_resource_cleans_local_directory_zip_when_upload_fails(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    uploaded_paths = []

    def fail_upload(path):
        uploaded_paths.append(path)
        raise RuntimeError("upload failed")

    provider._client.upload_temp_file.side_effect = fail_upload

    with pytest.raises(RuntimeError, match="upload failed"):
        provider._tool_add_resource({"url": str(docs)})

    assert uploaded_paths
    assert not uploaded_paths[0].exists()
    provider._client.post.assert_not_called()


def test_tool_add_resource_rejects_missing_local_path(tmp_path):
    missing = tmp_path / "missing.md"
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()

    result = json.loads(provider._tool_add_resource({"url": str(missing)}))

    assert result["error"] == f"Local resource path does not exist: {missing}"
    provider._client.upload_temp_file.assert_not_called()
    provider._client.post.assert_not_called()


def test_tool_add_resource_sends_remote_url_as_path():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.post.return_value = {
        "status": "ok",
        "result": {"root_uri": "viking://resources/remote"},
    }

    provider._tool_add_resource({"url": "https://example.com/doc.md"})

    provider._client.upload_temp_file.assert_not_called()
    provider._client.post.assert_called_once_with("/api/v1/resources", {
        "path": "https://example.com/doc.md",
    })


@pytest.mark.parametrize("url", [
    "git@github.com:org/repo.git",
    "git@ssh.dev.azure.com:v3/org/project/repo",
    "ssh://git@github.com/org/repo.git",
    "git://github.com/org/repo.git",
])
def test_tool_add_resource_sends_git_remote_sources_as_path(url):
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.post.return_value = {
        "status": "ok",
        "result": {"root_uri": "viking://resources/repo"},
    }

    provider._tool_add_resource({"url": url})

    provider._client.upload_temp_file.assert_not_called()
    provider._client.post.assert_called_once_with("/api/v1/resources", {
        "path": url,
    })


def test_viking_client_upload_temp_file_uses_multipart_identity_headers(tmp_path, monkeypatch):
    sample = tmp_path / "sample.md"
    sample.write_text("# Local resource\n", encoding="utf-8")
    client = _VikingClient(
        "https://example.com",
        api_key="test-key",
        account="test-account",
        user="test-user",
        agent="test-agent",
    )
    captured_kwargs = {}

    def capture_httpx_post(url, **kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"status": "ok", "result": {"temp_file_id": "upload_sample.md"}},
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(client._httpx, "post", capture_httpx_post)

    assert client.upload_temp_file(sample) == "upload_sample.md"

    assert "files" in captured_kwargs
    assert "json" not in captured_kwargs
    headers = captured_kwargs["headers"]
    assert headers["X-OpenViking-Account"] == "test-account"
    assert headers["X-OpenViking-User"] == "test-user"
    assert headers["X-OpenViking-Agent"] == "test-agent"
    assert headers["X-API-Key"] == "test-key"
    assert "Content-Type" not in headers


def test_viking_client_raises_structured_server_error():
    client = _VikingClient.__new__(_VikingClient)
    response = SimpleNamespace(
        status_code=403,
        text='{"status":"error"}',
        json=lambda: {
            "status": "error",
            "error": {
                "code": "PERMISSION_DENIED",
                "message": "direct host filesystem paths are not allowed",
            },
        },
        raise_for_status=lambda: None,
    )

    with pytest.raises(RuntimeError, match="PERMISSION_DENIED"):
        client._parse_response(response)


def test_viking_client_headers_include_bearer_when_api_key_set():
    client = _VikingClient(
        "https://example.com",
        api_key="test-key",
        account="acct",
        user="usr",
        agent="hermes",
    )
    headers = client._headers()
    assert headers["X-API-Key"] == "test-key"
    assert headers["Authorization"] == "Bearer test-key"


def test_viking_client_headers_send_tenant_when_default():
    # account/user set to the literal string "default". OpenViking 0.3.x
    # requires X-OpenViking-Account and X-OpenViking-User for ROOT API key
    # requests to tenant-scoped APIs — omitting them causes
    # INVALID_ARGUMENT errors even when account="default".
    client = _VikingClient(
        "https://example.com",
        api_key="test-key",
        account="default",
        user="default",
        agent="hermes",
    )
    headers = client._headers()
    assert headers["X-OpenViking-Account"] == "default"
    assert headers["X-OpenViking-User"] == "default"
    assert headers["X-OpenViking-Agent"] == "hermes"
    assert headers["Authorization"] == "Bearer test-key"


def test_viking_client_headers_omit_tenant_when_empty():
    client = _VikingClient(
        "https://example.com",
        api_key="",
        account="",
        user="",
        agent="hermes",
    )
    headers = client._headers()
    assert "X-OpenViking-Account" not in headers
    assert "X-OpenViking-User" not in headers
    assert headers["X-OpenViking-Agent"] == "hermes"
    assert "Authorization" not in headers
    assert "X-API-Key" not in headers


def test_viking_client_headers_sent_with_real_tenant_values():
    client = _VikingClient(
        "https://example.com",
        api_key="test-key",
        account="real-account",
        user="real-user",
        agent="hermes",
    )
    headers = client._headers()
    assert headers["X-OpenViking-Account"] == "real-account"
    assert headers["X-OpenViking-User"] == "real-user"


def test_viking_client_health_sends_auth_headers(monkeypatch):
    client = _VikingClient(
        "https://example.com",
        api_key="test-key",
        account="",
        user="",
        agent="hermes",
    )
    captured = {}

    def capture_get(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers") or {}
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(client._httpx, "get", capture_get)
    assert client.health() is True
    assert captured["url"] == "https://example.com/health"
    assert captured["headers"]["Authorization"] == "Bearer test-key"


def test_tool_search_uses_openviking_limit_and_ignores_dead_mode():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.post.return_value = {
        "result": {
            "memories": [],
            "resources": [],
            "skills": [],
            "total": 0,
        }
    }

    result = json.loads(provider._tool_search({
        "query": "ranking",
        "mode": "deep",
        "scope": "viking://user/memories/",
        "limit": 7,
    }))

    assert result == {"results": [], "total": 0}
    provider._client.post.assert_called_once_with(
        "/api/v1/search/find",
        {
            "query": "ranking",
            "target_uri": "viking://user/memories/",
            "limit": 7,
        },
    )


def test_search_schema_does_not_expose_dead_mode():
    properties = openviking_module.SEARCH_SCHEMA["parameters"]["properties"]

    assert "mode" not in properties
    assert "mode=" not in openviking_module.SEARCH_SCHEMA["description"]
    assert "durable memory" in openviking_module.SEARCH_SCHEMA["description"]


def test_read_schema_supports_small_uri_batches():
    properties = openviking_module.READ_SCHEMA["parameters"]["properties"]

    assert "uri" in properties
    assert "uris" in properties
    assert openviking_module.READ_SCHEMA["parameters"]["required"] == []
    assert "up to three" in openviking_module.READ_SCHEMA["description"]


def test_tool_read_batches_up_to_three_unique_uris():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.get.side_effect = [
        {"result": "first"},
        {"result": "second"},
        {"result": "third"},
    ]

    result = json.loads(provider._tool_read({
        "uris": [
            "viking://user/memories/one.md",
            "viking://user/memories/two.md",
            "viking://user/memories/two.md",
            "viking://user/memories/three.md",
            "viking://user/memories/four.md",
        ],
        "level": "full",
    }))

    assert result["requested"] == 4
    assert result["returned"] == 3
    assert result["truncated"] is True
    assert [entry["uri"] for entry in result["results"]] == [
        "viking://user/memories/one.md",
        "viking://user/memories/two.md",
        "viking://user/memories/three.md",
    ]
    assert [entry["content"] for entry in result["results"]] == ["first", "second", "third"]


def test_history_tools_are_not_exposed_to_the_model():
    provider = OpenVikingMemoryProvider()

    schemas = {schema["name"] for schema in provider.get_tool_schemas()}

    assert "viking_history_search" not in schemas
    assert "viking_history_read" not in schemas


def test_system_prompt_focuses_on_openviking_search_and_read():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._endpoint = "http://ov"
    provider._client.get.return_value = {"result": [{"uri": "viking://user/memories/"}]}

    prompt = provider.system_prompt_block()

    assert "durable indexed memory and knowledge" in prompt
    assert "viking_search" in prompt
    assert "viking_read" in prompt
    assert "up to three URIs" in prompt
    assert "viking_history_search" not in prompt
    assert "viking_history_read" not in prompt
    assert "URI diagnostics" in prompt
    assert "Hermes local session search" not in prompt
    assert "do not use" not in prompt.lower()
    assert "avoid session_search" not in prompt.lower()


def test_prefetch_fetches_current_query_when_cached_result_is_missing(monkeypatch):
    provider = OpenVikingMemoryProvider()
    provider._client = object()
    provider._endpoint = "http://ov"
    provider._api_key = "key"
    provider._account = "acct"
    provider._user = "user"
    provider._agent = "agent"

    seen_payloads = []

    class FakeClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "http://ov"
            assert api_key == "key"
            assert account == "acct"
            assert user == "user"
            assert agent == "agent"

        def post(self, path, payload):
            seen_payloads.append((path, payload))
            return {
                "result": {
                    "memories": [
                        {
                            "uri": "viking://memories/1",
                            "score": 0.91,
                            "abstract": "fresh memory",
                        }
                    ],
                    "resources": [],
                    "skills": [],
                }
            }

        def get(self, path, **kwargs):
            return {"result": "fresh memory"}

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeClient)

    result = provider.prefetch("where did we store it?")

    assert result.startswith("## OpenViking Context\nTreat these OpenViking memories as evidence")
    assert "Retrieval status: source=find" in result
    assert "hits=1; merged=1" in result
    assert "Ranked OpenViking evidence:" in result
    assert "- [0.91] memory: viking://memories/1; sources=find" in result
    assert "Abstract: fresh memory" in result
    assert seen_payloads == [
        (
            "/api/v1/search/find",
            {
                "query": "where did we store it?",
                "limit": 32,
            },
        ),
    ]


def test_prefetch_ignores_cached_result_for_different_query_and_fetches_fresh(monkeypatch):
    provider = OpenVikingMemoryProvider()
    provider._client = object()
    provider._endpoint = "http://ov"
    provider._account = "acct"
    provider._user = "user"
    provider._agent = "agent"
    provider._prefetch_result = "- [0.80] stale result (viking://memories/stale)"
    provider._prefetch_query = "old query"

    seen_payloads = []

    class FakeClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "http://ov"

        def post(self, path, payload):
            seen_payloads.append((path, payload))
            return {
                "result": {
                    "memories": [],
                    "resources": [
                        {
                            "uri": "viking://resources/2",
                            "score": 0.95,
                            "abstract": "fresh resource",
                        }
                    ],
                    "skills": [],
                }
            }

        def get(self, path, **kwargs):
            return {"result": "fresh resource"}

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeClient)

    result = provider.prefetch("new query")

    assert "stale result" not in result
    assert "Retrieval status: source=find" in result
    assert "hits=1; merged=1" in result
    assert "- [0.95] resource: viking://resources/2; sources=find" in result
    assert seen_payloads == [
        (
            "/api/v1/search/find",
            {
                "query": "new query",
                "limit": 32,
            },
        ),
    ]


def test_prefetch_detail_uses_relevant_excerpt_from_long_content(monkeypatch):
    provider = OpenVikingMemoryProvider()
    provider._client = object()
    provider._endpoint = "http://ov"
    provider._account = "acct"
    provider._user = "user"
    provider._agent = "agent"

    long_prefix = "\n".join(f"unrelated profile line {idx}" for idx in range(80))
    focused_detail = (
        f"{long_prefix}\n"
        "John does kickboxing for energy.\n"
        "John also practices taekwondo with his family.\n"
        "He enjoys family fitness routines.\n"
    )
    seen_get_paths = []

    class FakeClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "http://ov"

        def post(self, path, payload):
            return {
                "result": {
                    "memories": [
                        {
                            "uri": "viking://memories/john",
                            "score": 0.91,
                            "abstract": "John does kickboxing for energy.",
                        }
                    ],
                    "resources": [],
                    "skills": [],
                }
            }

        def get(self, path, **kwargs):
            seen_get_paths.append(path)
            return {"result": focused_detail}

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeClient)

    result = provider.prefetch("What martial arts has John done?")

    assert "Detail:" in result
    assert "kickboxing" in result
    assert "taekwondo" in result
    assert seen_get_paths == ["/api/v1/content/read"]


def test_prefetch_reranks_query_overlap_without_dropping_low_scores(monkeypatch):
    provider = OpenVikingMemoryProvider()
    provider._client = object()
    provider._endpoint = "http://ov"
    provider._account = "acct"
    provider._user = "user"
    provider._agent = "agent"

    class FakeClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "http://ov"

        def post(self, path, payload):
            return {
                "result": {
                    "memories": [
                        {
                            "uri": "viking://user/memories/generic",
                            "score": 0.8,
                            "abstract": "A generic planning note about errands.",
                        },
                        {
                            "uri": "viking://user/memories/john-martial-arts",
                            "score": 0.62,
                            "level": 2,
                            "category": "events",
                            "abstract": "John practiced kickboxing and taekwondo with his family.",
                        },
                        {
                            "uri": "viking://user/memories/weak",
                            "score": 0.2,
                            "abstract": "John mentioned martial arts once.",
                        },
                    ],
                    "resources": [],
                    "skills": [],
                }
            }

        def get(self, path, **kwargs):
            return {"result": "John practiced kickboxing and taekwondo with his family."}

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeClient)

    result = provider.prefetch("What martial arts has John done?")

    assert "filtered=3" in result
    assert "threshold=0.0" in result
    assert "viking://user/memories/weak" in result
    assert result.index("viking://user/memories/john-martial-arts") < result.index(
        "viking://user/memories/generic"
    )


def test_prefetch_dedupes_equivalent_abstracts(monkeypatch):
    provider = OpenVikingMemoryProvider()
    provider._client = object()
    provider._endpoint = "http://ov"
    provider._account = "acct"
    provider._user = "user"
    provider._agent = "agent"

    class FakeClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "http://ov"

        def post(self, path, payload):
            return {
                "result": {
                    "memories": [
                        {
                            "uri": "viking://user/memories/one",
                            "score": 0.7,
                            "category": "profile",
                            "abstract": "The backpack is in the closet.",
                        },
                        {
                            "uri": "viking://user/memories/two",
                            "score": 0.68,
                            "category": "profile",
                            "abstract": "The backpack is in the closet.",
                        },
                    ],
                    "resources": [],
                    "skills": [],
                }
            }

        def get(self, path, **kwargs):
            return {"result": "The backpack is in the closet."}

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeClient)

    result = provider.prefetch("where is the backpack?")

    assert "merged=2" in result
    assert "filtered=1" in result
    assert "viking://user/memories/one" in result
    assert "viking://user/memories/two" not in result


def test_browse_is_capped_diagnostic_output():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.get.return_value = {
        "result": {
            "uri": "viking://session",
            "type": "dir",
            "children": [
                {
                    "uri": f"viking://session/session-{idx}",
                    "name": f"session-{idx}",
                    "type": "dir",
                    "abstract": "x" * 500,
                }
                for idx in range(40)
            ],
        }
    }

    result = json.loads(provider._tool_browse({"action": "list", "path": "viking://session"}))

    assert result["path"] == "viking://session"
    assert result["note"] == "Diagnostic listing only. Use search/read tools for evidence content."
    assert len(result["entries"]) == 25
    assert result["truncated"] is True
    assert len(result["entries"][0]["abstract"]) < 220


def test_handle_tool_call_reconnects_after_startup_health_failure(monkeypatch):
    instances = []

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            self.endpoint = endpoint
            self.posts = []
            self.index = len(instances)
            instances.append(self)

        def health(self):
            return self.index > 0

        def post(self, path, payload=None, **kwargs):
            self.posts.append((path, payload or {}))
            return {}

    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://openviking.local")
    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)

    provider = OpenVikingMemoryProvider()
    provider.initialize("session-1")

    assert provider._client is None

    result = json.loads(provider.handle_tool_call("viking_remember", {"content": "stable fact"}))

    assert result["status"] == "stored"
    assert len(instances) == 2
    assert instances[1].posts == [
        (
            "/api/v1/sessions/session-1/messages",
            {"role": "user", "parts": [{"type": "text", "text": "[Remember] stable fact"}]},
        )
    ]
