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


def _allow_setup_validation(monkeypatch, *, root_access: bool = False):
    monkeypatch.setattr(
        openviking_module,
        "_validate_openviking_reachability",
        lambda endpoint: (True, ""),
        raising=False,
    )
    monkeypatch.setattr(
        openviking_module,
        "_validate_openviking_auth",
        lambda values: (True, ""),
        raising=False,
    )
    monkeypatch.setattr(
        openviking_module,
        "_validate_openviking_root_access",
        lambda values: (root_access, "" if root_access else "Requires role: root"),
        raising=False,
    )


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
    _allow_setup_validation(monkeypatch, root_access=True)

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
    _allow_setup_validation(monkeypatch)

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


def test_post_setup_manual_validation_failure_writes_nothing(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    ovcli_path = tmp_path / "ovcli.conf"
    original_ovcli = json.dumps({"url": "http://old.local"})
    ovcli_path.write_text(original_ovcli, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(ovcli_path))
    _allow_setup_validation(monkeypatch)
    monkeypatch.setattr(
        openviking_module,
        "_validate_openviking_auth",
        lambda values: (False, "OpenViking authentication validation failed: bad key"),
        raising=False,
    )

    from hermes_cli import config as hermes_config
    from hermes_cli import memory_setup

    save_config = MagicMock()
    choices = iter([2, 0, 1])
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
            "OpenViking user API key": "bad-key",
            "OpenViking agent": "agent",
        }),
    )
    config = {"memory": {"provider": "builtin"}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    save_config.assert_not_called()
    assert config == {"memory": {"provider": "builtin"}}
    assert ovcli_path.read_text(encoding="utf-8") == original_ovcli
    assert not (hermes_home / ".env").exists()


def test_post_setup_manual_retries_base_url_until_reachable(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    ovcli_path = tmp_path / "ovcli.conf"
    ovcli_path.write_text(json.dumps({"url": "http://old.local"}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(ovcli_path))
    monkeypatch.setattr(openviking_module, "_validate_openviking_auth", lambda values: (True, ""))

    reachability_calls = []

    def validate_reachability(endpoint):
        reachability_calls.append(endpoint)
        if endpoint == "http://bad.local:1933":
            return False, "OpenViking server is not reachable at http://bad.local:1933."
        return True, ""

    monkeypatch.setattr(openviking_module, "_validate_openviking_reachability", validate_reachability)
    monkeypatch.setattr(openviking_module, "_validate_openviking_root_access", lambda values: (False, "Requires role: root"))

    from hermes_cli import memory_setup

    prompts = {
        "OpenViking server URL": iter(["http://bad.local:1933", "http://localhost:1933"]),
        "OpenViking agent": iter(["agent"]),
    }

    def fake_prompt(label, default=None, secret=False):
        return next(prompts[label])

    choices = iter([2, 0, 0, 1])
    monkeypatch.setattr(
        memory_setup,
        "_curses_select",
        lambda *args, **kwargs: next(choices),
    )
    monkeypatch.setattr(memory_setup, "_prompt", fake_prompt)
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    assert reachability_calls == ["http://bad.local:1933", "http://localhost:1933"]
    assert config["memory"]["provider"] == "openviking"
    env_text = (hermes_home / ".env").read_text(encoding="utf-8")
    assert "OPENVIKING_ENDPOINT=http://localhost:1933" in env_text


def test_post_setup_manual_retries_user_key_until_status_valid(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    ovcli_path = tmp_path / "ovcli.conf"
    ovcli_path.write_text(json.dumps({"url": "http://old.local"}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(ovcli_path))
    monkeypatch.setattr(openviking_module, "_validate_openviking_reachability", lambda endpoint: (True, ""))
    monkeypatch.setattr(openviking_module, "_validate_openviking_root_access", lambda values: (False, "Requires role: root"))

    auth_calls = []

    def validate_auth(values):
        auth_calls.append(dict(values))
        if values["api_key"] == "bad-key":
            return False, "OpenViking authentication validation failed: bad key"
        return True, ""

    monkeypatch.setattr(openviking_module, "_validate_openviking_auth", validate_auth)

    from hermes_cli import memory_setup

    prompts = {
        "OpenViking server URL": iter(["https://openviking.example"]),
        "OpenViking user API key": iter(["bad-key", "good-key"]),
        "OpenViking agent": iter(["agent", "agent"]),
    }

    def fake_prompt(label, default=None, secret=False):
        return next(prompts[label])

    choices = iter([2, 0, 0, 0, 1])
    monkeypatch.setattr(
        memory_setup,
        "_curses_select",
        lambda *args, **kwargs: next(choices),
    )
    monkeypatch.setattr(memory_setup, "_prompt", fake_prompt)
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    assert [call["api_key"] for call in auth_calls] == ["bad-key", "good-key"]
    env_text = (hermes_home / ".env").read_text(encoding="utf-8")
    assert "OPENVIKING_API_KEY=good-key" in env_text


def test_post_setup_manual_user_key_rejects_root_key(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    ovcli_path = tmp_path / "ovcli.conf"
    ovcli_path.write_text(json.dumps({"url": "http://old.local"}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(ovcli_path))
    monkeypatch.setattr(openviking_module, "_validate_openviking_reachability", lambda endpoint: (True, ""))
    monkeypatch.setattr(openviking_module, "_validate_openviking_auth", lambda values: (True, ""))

    root_checks = []

    def validate_root(values):
        root_checks.append(values["api_key"])
        if values["api_key"] == "root-secret":
            return True, ""
        return False, "Requires role: root"

    monkeypatch.setattr(openviking_module, "_validate_openviking_root_access", validate_root)

    from hermes_cli import memory_setup

    prompts = {
        "OpenViking server URL": iter(["https://openviking.example"]),
        "OpenViking user API key": iter(["root-secret", "user-secret"]),
        "OpenViking agent": iter(["agent", "agent"]),
    }

    def fake_prompt(label, default=None, secret=False):
        return next(prompts[label])

    choices = iter([2, 0, 0, 0, 1])
    monkeypatch.setattr(
        memory_setup,
        "_curses_select",
        lambda *args, **kwargs: next(choices),
    )
    monkeypatch.setattr(memory_setup, "_prompt", fake_prompt)
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    assert root_checks == ["root-secret", "user-secret"]
    env_text = (hermes_home / ".env").read_text(encoding="utf-8")
    assert "OPENVIKING_API_KEY=user-secret" in env_text
    assert "OPENVIKING_API_KEY=root-secret" not in env_text


def test_post_setup_manual_root_key_requires_root_only_validation(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    ovcli_path = tmp_path / "ovcli.conf"
    ovcli_path.write_text(json.dumps({"url": "http://old.local"}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(ovcli_path))
    monkeypatch.setattr(openviking_module, "_validate_openviking_reachability", lambda endpoint: (True, ""))
    monkeypatch.setattr(openviking_module, "_validate_openviking_auth", lambda values: (True, ""))

    root_calls = []

    def validate_root(values):
        root_calls.append(dict(values))
        return True, ""

    monkeypatch.setattr(openviking_module, "_validate_openviking_root_access", validate_root)

    from hermes_cli import memory_setup

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
    choices = iter([2, 1, 1])
    monkeypatch.setattr(
        memory_setup,
        "_curses_select",
        lambda *args, **kwargs: next(choices),
    )
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    assert [call["api_key"] for call in root_calls] == ["root-secret"]
    assert config["memory"]["provider"] == "openviking"


def test_post_setup_manual_retries_root_key_before_account_prompts(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    ovcli_path = tmp_path / "ovcli.conf"
    ovcli_path.write_text(json.dumps({"url": "http://old.local"}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(ovcli_path))
    monkeypatch.setattr(openviking_module, "_validate_openviking_reachability", lambda endpoint: (True, ""))
    monkeypatch.setattr(openviking_module, "_validate_openviking_auth", lambda values: (True, ""))

    def validate_root(values):
        if values["api_key"] == "bad-root":
            return False, "OpenViking root API key validation failed: bad key"
        return True, ""

    monkeypatch.setattr(openviking_module, "_validate_openviking_root_access", validate_root)

    from hermes_cli import memory_setup

    prompt_events = []
    prompts = {
        "OpenViking server URL": iter(["https://openviking.example"]),
        "OpenViking root API key": iter(["bad-root", "good-root"]),
        "OpenViking account": iter(["acct"]),
        "OpenViking user": iter(["alice"]),
        "OpenViking agent": iter(["agent"]),
    }

    def fake_prompt(label, default=None, secret=False):
        prompt_events.append(label)
        return next(prompts[label])

    choices = iter([2, 1, 0, 1, 1])
    monkeypatch.setattr(
        memory_setup,
        "_curses_select",
        lambda *args, **kwargs: next(choices),
    )
    monkeypatch.setattr(memory_setup, "_prompt", fake_prompt)
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    assert prompt_events.index("OpenViking account") > prompt_events.index("OpenViking root API key")
    assert prompt_events.count("OpenViking account") == 1
    env_text = (hermes_home / ".env").read_text(encoding="utf-8")
    assert "OPENVIKING_API_KEY=good-root" in env_text


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
    choices = iter([2, 0, 1])
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
    _allow_setup_validation(monkeypatch, root_access=True)

    from hermes_cli import config as hermes_config
    from hermes_cli import memory_setup

    save_config = MagicMock()
    choices = iter([2, 1, 1])
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
    _allow_setup_validation(monkeypatch)

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


def test_viking_client_validate_auth_uses_authenticated_system_status(monkeypatch):
    client = _VikingClient(
        "https://example.com",
        api_key="test-key",
        account="acct",
        user="alice",
        agent="hermes",
    )
    captured = {}

    def capture_get(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers") or {}
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"status": "ok", "result": {"initialized": True}},
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(client._httpx, "get", capture_get)

    assert client.validate_auth() == {
        "status": "ok",
        "result": {"initialized": True},
    }
    assert captured["url"] == "https://example.com/api/v1/system/status"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["headers"]["X-OpenViking-Account"] == "acct"
    assert captured["headers"]["X-OpenViking-User"] == "alice"


def test_viking_client_validate_root_access_uses_admin_accounts(monkeypatch):
    client = _VikingClient(
        "https://example.com",
        api_key="root-key",
        account="",
        user="",
        agent="hermes",
    )
    captured = {}

    def capture_get(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers") or {}
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"status": "ok", "result": []},
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(client._httpx, "get", capture_get)

    assert client.validate_root_access() == {"status": "ok", "result": []}
    assert captured["url"] == "https://example.com/api/v1/admin/accounts"
    assert captured["headers"]["Authorization"] == "Bearer root-key"
    assert "X-OpenViking-Account" not in captured["headers"]
    assert "X-OpenViking-User" not in captured["headers"]


def test_validate_openviking_reachability_uses_health_only(monkeypatch):
    events = []

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "https://openviking.example"
            assert api_key == ""

        def health(self):
            events.append("health")
            return True

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)

    ok, message = openviking_module._validate_openviking_reachability(
        "https://openviking.example"
    )

    assert ok is True
    assert message == ""
    assert events == ["health"]


def test_validate_openviking_auth_uses_status_without_health(monkeypatch):
    events = []

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "https://openviking.example"
            assert api_key == "test-key"
            assert account == "acct"
            assert user == "alice"
            assert agent == "hermes"

        def validate_auth(self):
            events.append("status")
            return {"status": "ok"}

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)

    ok, message = openviking_module._validate_openviking_auth({
        "endpoint": "https://openviking.example",
        "api_key": "test-key",
        "account": "acct",
        "user": "alice",
        "agent": "hermes",
    })

    assert ok is True
    assert message == ""
    assert events == ["status"]


def test_validate_openviking_root_access_uses_admin_endpoint(monkeypatch):
    events = []

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "https://openviking.example"
            assert api_key == "root-key"
            assert account == ""
            assert user == ""
            assert agent == "hermes"

        def validate_root_access(self):
            events.append("admin")
            return {"status": "ok"}

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)

    ok, message = openviking_module._validate_openviking_root_access({
        "endpoint": "https://openviking.example",
        "api_key": "root-key",
    })

    assert ok is True
    assert message == ""
    assert events == ["admin"]
