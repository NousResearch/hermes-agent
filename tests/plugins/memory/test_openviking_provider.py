import json
import os
import stat
import zipfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import plugins.memory.openviking as openviking_module
from plugins.memory.openviking import (
    OpenVikingMemoryProvider,
    _DEFERRED_COMMIT_TIMEOUT,
    _VikingClient,
)


def _clear_openviking_tenant_env(monkeypatch):
    for name in ("OPENVIKING_ACCOUNT", "OPENVIKING_USER", "OPENVIKING_AGENT"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _isolate_openviking_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(openviking_module.Path, "home", staticmethod(lambda: home))


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
    monkeypatch.setattr(
        openviking_module,
        "_validate_openviking_setup_values",
        lambda values, *, require_api_key=False: (
            True,
            "",
            "root" if root_access else ("user" if values.get("api_key") else None),
        ),
        raising=False,
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_openviking_env_writer_restricts_file_permissions(tmp_path):
    env_path = tmp_path / ".env"

    openviking_module._write_env_vars(env_path, {"OPENVIKING_API_KEY": "secret"})

    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600


def test_openviking_env_writer_strips_embedded_newlines_in_values(tmp_path):
    # A secret pasted with an embedded CR/LF must not spill onto a new line,
    # or the round-trip re-parses the tail as a separate KEY=VALUE entry and
    # injects an arbitrary variable into the persisted credentials file.
    env_path = tmp_path / ".env"

    openviking_module._write_env_vars(
        env_path,
        {"OPENVIKING_API_KEY": "good\nINJECTED_KEY=attacker"},
    )

    lines = env_path.read_text(encoding="utf-8").splitlines()
    assert lines == ["OPENVIKING_API_KEY=goodINJECTED_KEY=attacker"]
    # No injected line means a follow-up read sees no rogue key.
    parsed = dict(line.split("=", 1) for line in lines if "=" in line)
    assert set(parsed) == {"OPENVIKING_API_KEY"}
    assert "INJECTED_KEY" not in parsed


def test_openviking_env_writer_strips_newlines_when_updating_existing_key(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENVIKING_API_KEY=old\n", encoding="utf-8")

    openviking_module._write_env_vars(
        env_path,
        {"OPENVIKING_API_KEY": "new\r\nROGUE=1"},
    )

    lines = env_path.read_text(encoding="utf-8").splitlines()
    assert lines == ["OPENVIKING_API_KEY=newROGUE=1"]
    assert all(not line.startswith("ROGUE=") for line in lines)


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_ovcli_config_writer_restricts_file_permissions(tmp_path):
    config_path = tmp_path / "ovcli.conf"

    openviking_module._write_ovcli_config(
        config_path,
        {"endpoint": "http://remote.example", "api_key": "secret"},
    )

    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600


def test_secret_permission_restriction_logs_chmod_failure(tmp_path, monkeypatch, caplog):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENVIKING_API_KEY=secret\n", encoding="utf-8")

    def fail_chmod(self, mode):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(type(env_path), "chmod", fail_chmod)

    with caplog.at_level("DEBUG", logger=openviking_module.__name__):
        openviking_module._restrict_secret_file_permissions(env_path)

    assert "Could not restrict permissions" in caplog.text
    assert "read-only filesystem" in caplog.text


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
        "account": "",
        "user": "",
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


def test_openviking_cli_config_env_overrides_saved_profile_path(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    saved_path = tmp_path / "ovcli.conf.saved"
    env_path = tmp_path / "ovcli.conf.env"
    saved_path.write_text(
        json.dumps({"url": "http://saved.local", "api_key": "saved-key"}),
        encoding="utf-8",
    )
    env_path.write_text(
        json.dumps({"url": "http://env-profile.local", "api_key": "env-profile-key"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(env_path))

    settings = openviking_module._resolve_connection_settings({
        "use_ovcli_config": True,
        "ovcli_config_path": str(saved_path),
    })

    assert settings["endpoint"] == "http://env-profile.local"
    assert settings["api_key"] == "env-profile-key"


def test_connection_values_omit_stale_identity_for_user_key_with_root_key():
    values = openviking_module._connection_values_from_ovcli({
        "url": "https://openviking.example",
        "api_key": "user-key",
        "root_api_key": "root-key",
        "account": "stale-account",
        "user": "stale-user",
    })

    assert values["api_key"] == "user-key"
    assert values["account"] == ""
    assert values["user"] == ""


def test_discover_ovcli_profiles_lists_saved_profiles_without_active_label(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    openviking_home = tmp_path / ".openviking"
    openviking_home.mkdir()
    env_path = tmp_path / "custom-ovcli.conf"
    env_path.write_text(json.dumps({"url": "http://env.local"}), encoding="utf-8")
    (openviking_home / "ovcli.conf").write_text(
        json.dumps({"url": "https://vps.example", "api_key": "secret"}),
        encoding="utf-8",
    )
    (openviking_home / "ovcli.conf.VPS").write_text(
        json.dumps({"url": "https://vps.example", "api_key": "secret"}),
        encoding="utf-8",
    )
    (openviking_home / "ovcli.conf.bak").write_text(
        json.dumps({"url": "http://backup.local"}),
        encoding="utf-8",
    )
    (openviking_home / "ovcli.conf.bad").write_text("{", encoding="utf-8")
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(env_path))
    monkeypatch.setattr(openviking_module.Path, "home", staticmethod(lambda: tmp_path))

    profiles = openviking_module._discover_ovcli_profiles()

    assert [(profile.source, profile.name, profile.path) for profile in profiles] == [
        ("env", "OPENVIKING_CLI_CONFIG_FILE", env_path),
        ("saved", "VPS", openviking_home / "ovcli.conf.VPS"),
    ]
    assert profiles[1].is_active is True
    assert openviking_module._profile_display_name(profiles[1]) == "VPS"
    assert "active" not in openviking_module._profile_description(profiles[1]).lower()


def test_link_ovcli_profile_removes_stale_inline_config(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENVIKING_ENDPOINT=http://old.local\nOTHER_KEY=keep\n", encoding="utf-8")
    config = {"memory": {}}
    provider_config = {
        "use_ovcli_config": False,
        "endpoint": "http://stale.local",
        "api_key": "stale-key",
        "account": "default",
        "user": "default",
        "agent": "stale-agent",
        "api_key_type": "root",
    }
    ovcli_path = tmp_path / "ovcli.conf.VPS_ROOT"

    openviking_module._link_ovcli_profile(
        config=config,
        provider_config=provider_config,
        env_path=env_path,
        ovcli_path=ovcli_path,
    )

    assert config["memory"]["openviking"] == {
        "use_ovcli_config": True,
        "ovcli_config_path": str(ovcli_path),
    }
    assert "OPENVIKING_ENDPOINT" not in env_path.read_text(encoding="utf-8")
    assert "OTHER_KEY=keep" in env_path.read_text(encoding="utf-8")


def test_post_setup_existing_profile_picker_validates_and_links_saved_profile(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    env_path = hermes_home / ".env"
    env_path.write_text("OPENVIKING_ENDPOINT=http://old.local\nOTHER_KEY=keep\n", encoding="utf-8")
    openviking_home = tmp_path / ".openviking"
    openviking_home.mkdir()
    active_path = openviking_home / "ovcli.conf"
    saved_path = openviking_home / "ovcli.conf.VPS"
    active_path.write_text(json.dumps({"url": "http://active.local"}), encoding="utf-8")
    saved_path.write_text(
        json.dumps({"url": "https://vps.example", "api_key": "user-key"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(openviking_module.Path, "home", staticmethod(lambda: tmp_path))

    from hermes_cli import memory_setup

    validate_calls = []

    def validate_values(values, *, require_api_key=False):
        validate_calls.append(dict(values))
        return True, "", "user"

    monkeypatch.setattr(
        openviking_module,
        "_validate_openviking_setup_values",
        validate_values,
        raising=False,
    )
    choices = iter([0, 0])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: next(choices))
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    assert validate_calls == [{
        "endpoint": "https://vps.example",
        "api_key": "user-key",
        "root_api_key": "",
        "account": "",
        "user": "",
        "agent": "",
    }]
    assert config["memory"]["provider"] == "openviking"
    assert config["memory"]["openviking"] == {
        "use_ovcli_config": True,
        "ovcli_config_path": str(saved_path),
    }
    env_text = env_path.read_text(encoding="utf-8")
    assert "OPENVIKING_" not in env_text
    assert "OTHER_KEY=keep" in env_text


def test_post_setup_create_remote_user_profile_can_mirror_to_openviking_store(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(openviking_module.Path, "home", staticmethod(lambda: tmp_path))
    _allow_setup_validation(monkeypatch)

    from hermes_cli import memory_setup

    choices = iter([1, 0, 1])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: next(choices))
    monkeypatch.setattr(
        memory_setup,
        "_prompt",
        _prompt_from_values({
            "OpenViking server URL": "https://openviking.example",
            "OpenViking user API key": "user-secret",
            "Hermes peer ID in OpenViking": "hermes",
            "OpenViking profile name": "VPS",
        }),
    )
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    mirrored_path = tmp_path / ".openviking" / "ovcli.conf.VPS"
    assert mirrored_path.exists()
    assert json.loads(mirrored_path.read_text(encoding="utf-8")) == {
        "url": "https://openviking.example",
        "api_key": "user-secret",
        "actor_peer_id": "hermes",
    }
    assert config["memory"]["provider"] == "openviking"
    assert config["memory"]["openviking"] == {
        "use_ovcli_config": True,
        "ovcli_config_path": str(mirrored_path),
    }
    env_path = hermes_home / ".env"
    if env_path.exists():
        assert "OPENVIKING_" not in env_path.read_text(encoding="utf-8")


def test_post_setup_create_remote_user_can_keep_hermes_only(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _allow_setup_validation(monkeypatch)

    from hermes_cli import memory_setup

    choices = iter([1, 0, 0])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: next(choices))
    monkeypatch.setattr(
        memory_setup,
        "_prompt",
        _prompt_from_values({
            "OpenViking server URL": "https://openviking.example",
            "OpenViking user API key": "user-secret",
            "Hermes peer ID in OpenViking": "agent",
        }),
    )
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    assert config["memory"]["provider"] == "openviking"
    assert config["memory"]["openviking"] == {"use_ovcli_config": False}
    env_text = (hermes_home / ".env").read_text(encoding="utf-8")
    assert "OPENVIKING_ENDPOINT=https://openviking.example" in env_text
    assert "OPENVIKING_API_KEY=user-secret" in env_text
    assert "OPENVIKING_AGENT=agent" in env_text
    assert not (tmp_path / "home" / ".openviking").exists()


def test_post_setup_create_openviking_service_validates_after_api_key(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import memory_setup

    validation_calls = []

    def validate_values(values, *, require_api_key=False):
        validation_calls.append((dict(values), require_api_key))
        return True, "", "user"

    monkeypatch.setattr(
        openviking_module,
        "_validate_openviking_reachability",
        MagicMock(side_effect=AssertionError("service setup validates only after API key entry")),
    )
    monkeypatch.setattr(openviking_module, "_validate_openviking_setup_values", validate_values)
    choices = iter([0, 0])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: next(choices))
    monkeypatch.setattr(
        memory_setup,
        "_prompt",
        _prompt_from_values(
            {
                "OpenViking API key": "service-secret",
                "Hermes peer ID in OpenViking": "agent",
            },
            forbidden={"OpenViking server URL", "OpenViking user API key", "OpenViking root API key"},
        ),
    )
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    assert validation_calls == [(
        {
            "endpoint": "https://api.vikingdb.cn-beijing.volces.com/openviking",
            "api_key": "service-secret",
            "root_api_key": "",
            "account": "",
            "user": "",
            "agent": "agent",
            "api_key_type": "user",
        },
        True,
    )]
    env_text = (hermes_home / ".env").read_text(encoding="utf-8")
    assert "OPENVIKING_ENDPOINT=https://api.vikingdb.cn-beijing.volces.com/openviking" in env_text
    assert "OPENVIKING_API_KEY=service-secret" in env_text
    assert "OPENVIKING_AGENT=agent" in env_text


def test_post_setup_remote_blank_api_key_cancels_without_saving(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(openviking_module, "_validate_openviking_reachability", lambda endpoint: (True, ""))

    from hermes_cli import config as hermes_config
    from hermes_cli import memory_setup

    save_config = MagicMock()
    monkeypatch.setattr(hermes_config, "save_config", save_config)
    choices = iter([1, 0, 1])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: next(choices))
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
    assert not (hermes_home / ".env").exists()


def test_post_setup_user_key_path_can_route_detected_root_key_to_root_setup(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import memory_setup

    def validate_values(values, *, require_api_key=False):
        assert values["api_key"] == "root-secret"
        return True, "", "root"

    monkeypatch.setattr(openviking_module, "_validate_openviking_reachability", lambda endpoint: (True, ""))
    monkeypatch.setattr(openviking_module, "_validate_openviking_setup_values", validate_values)
    choices = iter([1, 0, 0, 0])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: next(choices))
    prompt_events = []

    def fake_prompt(label, default=None, secret=False):
        if label == "OpenViking root API key":
            raise AssertionError("OpenViking root API key should not be re-prompted")
        prompt_events.append(label)
        values = {
            "OpenViking server URL": "https://openviking.example",
            "OpenViking user API key": "root-secret",
            "OpenViking account": "acct",
            "OpenViking user": "alice",
            "Hermes peer ID in OpenViking": "agent",
        }
        return values.get(label, default or "")

    monkeypatch.setattr(memory_setup, "_prompt", fake_prompt)
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    assert prompt_events.count("Hermes peer ID in OpenViking") == 1
    env_text = (hermes_home / ".env").read_text(encoding="utf-8")
    assert "OPENVIKING_API_KEY=root-secret" in env_text
    assert "OPENVIKING_ACCOUNT=acct" in env_text
    assert "OPENVIKING_USER=alice" in env_text
    assert "OPENVIKING_AGENT=agent" in env_text


def test_post_setup_root_key_path_can_route_detected_user_key_to_user_setup(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import memory_setup

    def validate_values(values, *, require_api_key=False):
        assert values["api_key"] == "user-secret"
        return True, "", "user"

    monkeypatch.setattr(openviking_module, "_validate_openviking_reachability", lambda endpoint: (True, ""))
    monkeypatch.setattr(openviking_module, "_validate_openviking_setup_values", validate_values)
    choices = iter([1, 1, 0, 0])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: next(choices))
    monkeypatch.setattr(
        memory_setup,
        "_prompt",
        _prompt_from_values(
            {
                "OpenViking server URL": "https://openviking.example",
                "OpenViking root API key": "user-secret",
                "Hermes peer ID in OpenViking": "agent",
            },
            forbidden={"OpenViking user API key", "OpenViking account", "OpenViking user"},
        ),
    )
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    env_text = (hermes_home / ".env").read_text(encoding="utf-8")
    assert "OPENVIKING_API_KEY=user-secret" in env_text
    assert "OPENVIKING_AGENT=agent" in env_text
    assert "OPENVIKING_ACCOUNT" not in env_text
    assert "OPENVIKING_USER" not in env_text


def test_manual_root_key_flow_prints_validation_progress(monkeypatch, capsys):
    _clear_openviking_env(monkeypatch)

    monkeypatch.setattr(openviking_module, "_validate_openviking_reachability", lambda endpoint: (True, ""))

    validate_calls = []

    def validate_values(values, *, require_api_key=False):
        validate_calls.append(dict(values))
        return True, "", "root"

    monkeypatch.setattr(openviking_module, "_validate_openviking_setup_values", validate_values)
    choices = iter([1])

    values = openviking_module._prompt_manual_connection_values(
        _prompt_from_values({
            "OpenViking server URL": "https://openviking.example",
            "OpenViking root API key": "root-secret",
            "OpenViking account": "acct",
            "OpenViking user": "alice",
            "Hermes peer ID in OpenViking": "agent",
        }),
        lambda *args, **kwargs: next(choices),
        -1,
    )

    assert values["root_api_key"] == "root-secret"
    assert len(validate_calls) == 2
    output = capsys.readouterr().out
    assert "Checking OpenViking server..." in output
    assert "Validating OpenViking root API key..." in output
    assert "Validating OpenViking API access..." in output


def test_start_local_openviking_server_uses_endpoint_host_and_port(monkeypatch):
    popen_calls = []

    def fake_popen(args, **kwargs):
        popen_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(openviking_module.shutil, "which", lambda name: "/usr/local/bin/openviking-server")
    monkeypatch.setattr(openviking_module.subprocess, "Popen", fake_popen)

    started, message = openviking_module._start_local_openviking_server("http://127.0.0.1:1934")

    assert started is True
    assert "127.0.0.1:1934" in message
    args, kwargs = popen_calls[0]
    assert args == ["/usr/local/bin/openviking-server", "--host", "127.0.0.1", "--port", "1934"]
    assert kwargs["start_new_session"] is True


def test_start_local_openviking_server_writes_output_to_log(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    popen_calls = []

    class FakeProcess:
        pass

    def fake_popen(args, **kwargs):
        popen_calls.append((args, kwargs))
        assert kwargs["stdout"] is kwargs["stderr"]
        assert kwargs["stdout"].name == str(hermes_home / "logs" / "openviking-server.log")
        assert not kwargs["stdout"].closed
        return FakeProcess()

    monkeypatch.setattr(openviking_module.shutil, "which", lambda name: "/usr/local/bin/openviking-server")
    monkeypatch.setattr(openviking_module.subprocess, "Popen", fake_popen)

    started, message = openviking_module._start_local_openviking_server("http://127.0.0.1:1934")

    assert started is True
    assert str(hermes_home / "logs" / "openviking-server.log") in message
    assert popen_calls


def test_https_local_endpoint_is_not_runtime_autostart_eligible(monkeypatch):
    _clear_openviking_env(monkeypatch)
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "https://localhost:1934")

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "https://localhost:1934"

        def health(self):
            return False

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)
    monkeypatch.setattr(
        openviking_module,
        "_start_local_openviking_server",
        MagicMock(side_effect=AssertionError("https localhost endpoint should not auto-start")),
    )

    warnings = []
    provider = OpenVikingMemoryProvider()
    provider.initialize("session-1", platform="cli", warning_callback=warnings.append)

    assert provider._client is None
    assert warnings == [
        "Remote OpenViking server at https://localhost:1934 is not reachable; "
        "OpenViking memory disabled for this Hermes run. "
        "Check the configured endpoint and network connectivity."
    ]


def test_runtime_does_not_autostart_when_local_server_reports_unhealthy(monkeypatch):
    _clear_openviking_env(monkeypatch)
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://localhost:1934")

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "http://localhost:1934"

        def health(self):
            return False

        def health_payload(self):
            return {"healthy": False}

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)
    monkeypatch.setattr(
        openviking_module,
        "_start_local_openviking_server",
        MagicMock(side_effect=AssertionError("responding unhealthy server should not auto-start another process")),
    )

    warnings = []
    provider = OpenVikingMemoryProvider()
    provider.initialize("session-1", platform="cli", warning_callback=warnings.append)

    assert provider._client is None
    assert warnings == [
        "OpenViking server at http://localhost:1934 responded but reported unhealthy status. "
        "OpenViking memory disabled for this Hermes run."
    ]


def test_handle_unreachable_endpoint_does_not_wait_when_autostart_command_missing(monkeypatch, capsys):
    monkeypatch.setattr(
        openviking_module,
        "_start_local_openviking_server",
        lambda endpoint: (False, "openviking-server was not found on PATH."),
    )
    monkeypatch.setattr(
        openviking_module,
        "_wait_for_openviking_health",
        MagicMock(side_effect=AssertionError("should not wait when server did not start")),
    )

    result = openviking_module._handle_unreachable_endpoint(
        "http://127.0.0.1:1934",
        "OpenViking server is not reachable.",
        lambda *args, **kwargs: 0,
        -1,
    )

    assert result is False
    output = capsys.readouterr().out
    assert "openviking-server was not found on PATH." in output
    assert "did not become reachable" not in output


def test_handle_unreachable_endpoint_waits_long_enough_after_autostart(monkeypatch, capsys):
    wait_calls = []

    monkeypatch.setattr(
        openviking_module,
        "_start_local_openviking_server",
        lambda endpoint: (True, "Started openviking-server on 127.0.0.1:1934 in the background."),
    )
    monkeypatch.setattr(
        openviking_module,
        "_wait_for_openviking_health",
        lambda endpoint, *, timeout_seconds=0: wait_calls.append((endpoint, timeout_seconds)) or True,
    )

    result = openviking_module._handle_unreachable_endpoint(
        "http://127.0.0.1:1934",
        "OpenViking server is not reachable.",
        lambda *args, **kwargs: 0,
        -1,
    )

    assert result is True
    assert wait_calls == [("http://127.0.0.1:1934", 60.0)]
    output = capsys.readouterr().out
    assert "Waiting for OpenViking server to become reachable..." in output


def test_manual_setup_does_not_offer_autostart_when_local_server_is_unhealthy(monkeypatch):
    _clear_openviking_env(monkeypatch)

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "http://localhost:1933"

        def health_payload(self):
            return {"healthy": False}

    select_calls = []

    def select(title, options, **kwargs):
        select_calls.append((title, options))
        assert all(label != "Start local OpenViking" for label, _description in options)
        return 1

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)
    monkeypatch.setattr(
        openviking_module,
        "_start_local_openviking_server",
        MagicMock(side_effect=AssertionError("unhealthy local server should not offer auto-start")),
    )

    result = openviking_module._prompt_manual_connection_values(
        _prompt_from_values({"OpenViking server URL": "localhost"}),
        select,
        -1,
    )

    assert result is openviking_module._SETUP_CANCELLED
    assert select_calls == [(
        "  OpenViking server unhealthy",
        [
            ("Retry", "try this step again"),
            ("Cancel setup", "no changes saved"),
        ],
    )]


def test_initialize_autostarts_local_openviking_in_background_when_runtime_health_fails(monkeypatch):
    _clear_openviking_env(monkeypatch)
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://127.0.0.1:1934")
    health_calls = []
    start_calls = []
    waiter_calls = []

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "http://127.0.0.1:1934"

        def health(self):
            health_calls.append("health")
            return False

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)
    monkeypatch.setattr(
        openviking_module,
        "_start_local_openviking_server",
        lambda endpoint: start_calls.append(endpoint) or (True, "started"),
    )
    monkeypatch.setattr(
        openviking_module,
        "_wait_for_openviking_health",
        MagicMock(side_effect=AssertionError("runtime init should not wait synchronously")),
    )

    provider = OpenVikingMemoryProvider()
    monkeypatch.setattr(
        provider,
        "_start_runtime_openviking_waiter",
        lambda **kwargs: waiter_calls.append(kwargs),
        raising=False,
    )
    statuses = []
    provider.initialize("session-1", platform="cli", status_callback=statuses.append)

    assert provider._client is None
    assert health_calls == ["health"]
    assert start_calls == ["http://127.0.0.1:1934"]
    assert len(waiter_calls) == 1
    assert waiter_calls[0]["status_callback"] == statuses.append
    assert any("starting in the background" in message for message in statuses)


def test_runtime_openviking_waiter_attaches_client_after_health_recovers(monkeypatch):
    _clear_openviking_env(monkeypatch)
    wait_calls = []

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            self.endpoint = endpoint
            self.api_key = api_key
            self.account = account
            self.user = user
            self.agent = agent

        def health(self):
            return True

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)
    monkeypatch.setattr(
        openviking_module,
        "_wait_for_openviking_health",
        lambda endpoint, **kwargs: wait_calls.append((endpoint, kwargs)) or True,
    )

    provider = OpenVikingMemoryProvider()
    provider._endpoint = "http://127.0.0.1:1934"
    provider._api_key = "secret"
    provider._account = "acct"
    provider._user = "alice"
    provider._agent = "hermes"
    statuses = []

    provider._finish_runtime_openviking_start(
        status_callback=statuses.append,
        warning_callback=None,
    )

    assert provider._client is not None
    assert provider._client.endpoint == "http://127.0.0.1:1934"
    assert provider._client.api_key == "secret"
    assert wait_calls == [(
        "http://127.0.0.1:1934",
        {"timeout_seconds": openviking_module._LOCAL_OPENVIKING_AUTOSTART_TIMEOUT},
    )]
    assert any("OpenViking memory is active" in message for message in statuses)


def test_runtime_openviking_waiter_warns_when_background_start_times_out(monkeypatch):
    _clear_openviking_env(monkeypatch)
    monkeypatch.setattr(
        openviking_module,
        "_wait_for_openviking_health",
        lambda endpoint, **kwargs: False,
    )
    monkeypatch.setattr(
        openviking_module,
        "_VikingClient",
        MagicMock(side_effect=AssertionError("client should not be rebuilt before health recovers")),
    )

    provider = OpenVikingMemoryProvider()
    provider._endpoint = "http://127.0.0.1:1934"
    warnings = []

    provider._finish_runtime_openviking_start(
        status_callback=None,
        warning_callback=warnings.append,
    )

    assert provider._client is None
    assert warnings == [
        "Local OpenViking server at http://127.0.0.1:1934 is not reachable. "
        "Tried to start openviking-server, but it did not become reachable "
        "within 60 seconds. OpenViking memory disabled for this Hermes run."
    ]


def test_initialize_does_not_autostart_remote_openviking(monkeypatch, caplog):
    _clear_openviking_env(monkeypatch)
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "https://openviking.example")

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "https://openviking.example"

        def health(self):
            return False

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)
    monkeypatch.setattr(
        openviking_module,
        "_start_local_openviking_server",
        MagicMock(side_effect=AssertionError("remote endpoint should not auto-start")),
    )
    monkeypatch.setattr(
        openviking_module,
        "_wait_for_openviking_health",
        MagicMock(side_effect=AssertionError("remote endpoint should not wait")),
    )

    with caplog.at_level("WARNING", logger=openviking_module.__name__):
        provider = OpenVikingMemoryProvider()
        provider.initialize("session-1")

    assert provider._client is None
    assert "Remote OpenViking server at https://openviking.example is not reachable" in caplog.text


def test_initialize_warns_clearly_when_local_runtime_autostart_fails(monkeypatch, caplog):
    _clear_openviking_env(monkeypatch)
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://localhost:1934")

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "http://localhost:1934"

        def health(self):
            return False

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)
    monkeypatch.setattr(
        openviking_module,
        "_start_local_openviking_server",
        lambda endpoint: (False, "openviking-server was not found on PATH."),
    )
    monkeypatch.setattr(
        openviking_module,
        "_wait_for_openviking_health",
        MagicMock(side_effect=AssertionError("should not wait when server did not start")),
    )

    with caplog.at_level("WARNING", logger=openviking_module.__name__):
        provider = OpenVikingMemoryProvider()
        provider.initialize("session-1")

    assert provider._client is None
    assert "Local OpenViking server at http://localhost:1934 is not reachable" in caplog.text
    assert "openviking-server was not found on PATH" in caplog.text


def test_initialize_emits_cli_warning_when_local_runtime_autostart_fails(monkeypatch):
    _clear_openviking_env(monkeypatch)
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://localhost:1934")

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "http://localhost:1934"

        def health(self):
            return False

    warnings = []
    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)
    monkeypatch.setattr(
        openviking_module,
        "_start_local_openviking_server",
        lambda endpoint: (False, "openviking-server was not found on PATH."),
    )

    provider = OpenVikingMemoryProvider()
    provider.initialize("session-1", platform="cli", warning_callback=warnings.append)

    assert provider._client is None
    assert warnings == [
        "Local OpenViking server at http://localhost:1934 is not reachable. "
        "openviking-server was not found on PATH. "
        "OpenViking memory disabled for this Hermes run."
    ]


def test_initialize_does_not_emit_cli_warning_when_callback_absent(monkeypatch):
    _clear_openviking_env(monkeypatch)
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://localhost:1934")

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "http://localhost:1934"

        def health(self):
            return False

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)
    monkeypatch.setattr(
        openviking_module,
        "_start_local_openviking_server",
        lambda endpoint: (False, "openviking-server was not found on PATH."),
    )

    provider = OpenVikingMemoryProvider()
    provider.initialize("session-1", platform="gateway")

    assert provider._client is None


def test_post_setup_local_server_down_can_offer_autostart(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(openviking_module, "_validate_openviking_setup_values", lambda values, *, require_api_key=False: (True, "", None))

    from hermes_cli import memory_setup

    reachability_calls = []

    def validate_reachability(endpoint):
        reachability_calls.append(endpoint)
        return False, "OpenViking server is not reachable." if len(reachability_calls) == 1 else ""

    started = []
    monkeypatch.setattr(openviking_module, "_validate_openviking_reachability", validate_reachability)
    monkeypatch.setattr(openviking_module, "_start_local_openviking_server", lambda endpoint: (started.append(endpoint) or True, "started"))
    monkeypatch.setattr(openviking_module, "_wait_for_openviking_health", lambda endpoint, **kwargs: True)
    choices = iter([1, 0, 0, 0])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: next(choices))
    monkeypatch.setattr(
        memory_setup,
        "_prompt",
        _prompt_from_values({
            "OpenViking server URL": "localhost",
            "Hermes peer ID in OpenViking": "agent",
        }),
    )
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    assert started == ["http://localhost:1933"]
    assert reachability_calls == ["http://localhost:1933"]
    env_text = (hermes_home / ".env").read_text(encoding="utf-8")
    assert "OPENVIKING_ENDPOINT=http://localhost:1933" in env_text
    assert "OPENVIKING_API_KEY" not in env_text


def test_post_setup_invalid_env_profile_can_create_new_config(tmp_path, monkeypatch):
    _clear_openviking_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    ovcli_path = tmp_path / "broken" / "ovcli.conf"
    ovcli_path.parent.mkdir()
    ovcli_path.write_text("{", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(ovcli_path))
    _allow_setup_validation(monkeypatch)

    from hermes_cli import memory_setup

    choices = iter([1, 0, 0])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: next(choices))
    monkeypatch.setattr(
        memory_setup,
        "_prompt",
        _prompt_from_values({
            "OpenViking server URL": "https://openviking.example",
            "OpenViking user API key": "user-secret",
            "Hermes peer ID in OpenViking": "agent",
        }),
    )
    config = {"memory": {}}

    OpenVikingMemoryProvider().post_setup(str(hermes_home), config)

    assert ovcli_path.read_text(encoding="utf-8") == "{"
    assert config["memory"]["openviking"] == {"use_ovcli_config": False}


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


def test_tool_search_sends_limit_not_legacy_top_k():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.post.return_value = {
        "result": {"memories": [], "resources": [], "skills": [], "total": 0}
    }

    provider._tool_search({"query": "session switch", "limit": 7})

    provider._client.post.assert_called_once()
    payload = provider._client.post.call_args.args[1]
    assert payload["limit"] == 7
    assert "top_k" not in payload


def test_tool_search_uses_find_for_normal_search():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.post.return_value = {
        "result": {"memories": [], "resources": [], "skills": [], "total": 0}
    }

    provider._tool_search({"query": "simple lookup", "mode": "fast"})

    provider._client.post.assert_called_once_with("/api/v1/search/find", {
        "query": "simple lookup",
    })


def test_tool_search_uses_session_search_for_deep_search():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._session_id = "session-123"
    provider._client.post.return_value = {
        "result": {"memories": [], "resources": [], "skills": [], "total": 0}
    }

    provider._tool_search({"query": "connect facts", "mode": "deep"})

    provider._client.post.assert_called_once_with("/api/v1/search/search", {
        "query": "connect facts",
        "session_id": "session-123",
    })


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
    assert "X-OpenViking-Account" not in headers
    assert "X-OpenViking-User" not in headers
    assert headers["X-OpenViking-Actor-Peer"] == "test-agent"
    assert "X-OpenViking-Agent" not in headers
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


def test_viking_client_sanitizes_html_error_body():
    client = _VikingClient.__new__(_VikingClient)
    response = SimpleNamespace(
        status_code=523,
        text="""<!DOCTYPE html>
<html>
<head><title>tosaki.top | 523: Origin is unreachable</title></head>
<body>large Cloudflare error page</body>
</html>""",
        json=lambda: (_ for _ in ()).throw(ValueError("not json")),
    )

    with pytest.raises(openviking_module._OpenVikingHTTPError) as exc_info:
        client._parse_response(response)

    message = str(exc_info.value)
    assert "HTTP 523" in message
    assert "Origin is unreachable" in message
    assert "<!DOCTYPE" not in message
    assert "<html" not in message


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
    assert headers["X-OpenViking-Actor-Peer"] == "hermes"
    assert "X-OpenViking-Agent" not in headers
    assert "X-OpenViking-Account" not in headers
    assert "X-OpenViking-User" not in headers


def test_viking_client_headers_send_tenant_in_local_mode():
    # Local/trusted mode needs explicit tenant identity headers.
    client = _VikingClient(
        "https://example.com",
        api_key="",
        account="default",
        user="default",
        agent="hermes",
    )
    headers = client._headers()
    assert headers["X-OpenViking-Account"] == "default"
    assert headers["X-OpenViking-User"] == "default"
    assert headers["X-OpenViking-Actor-Peer"] == "hermes"
    assert "X-OpenViking-Agent" not in headers
    assert "Authorization" not in headers


def test_viking_client_headers_send_tenant_when_empty_falls_back_to_default(monkeypatch):
    _clear_openviking_tenant_env(monkeypatch)
    # Empty account/user strings fall back to "default" in local mode.
    client = _VikingClient(
        "https://example.com",
        api_key="",
        account="",
        user="",
        agent="hermes",
    )
    headers = client._headers()
    assert headers["X-OpenViking-Account"] == "default"
    assert headers["X-OpenViking-User"] == "default"
    assert headers["X-OpenViking-Actor-Peer"] == "hermes"
    assert "X-OpenViking-Agent" not in headers
    assert "Authorization" not in headers
    assert "X-API-Key" not in headers


def test_viking_client_headers_can_include_tenant_for_trusted_retry():
    client = _VikingClient(
        "https://example.com",
        api_key="test-key",
        account="real-account",
        user="real-user",
        agent="hermes",
    )
    headers = client._headers(include_tenant=True)
    assert headers["X-OpenViking-Account"] == "real-account"
    assert headers["X-OpenViking-User"] == "real-user"
    assert headers["Authorization"] == "Bearer test-key"


def test_viking_client_retries_with_tenant_headers_for_trusted_mode(monkeypatch):
    client = _VikingClient(
        "https://example.com",
        api_key="test-key",
        account="acct",
        user="usr",
        agent="hermes",
    )
    captured_headers = []

    def capture_get(url, **kwargs):
        captured_headers.append(kwargs.get("headers") or {})
        if len(captured_headers) == 1:
            return SimpleNamespace(
                status_code=400,
                text="",
                json=lambda: {
                    "status": "error",
                    "error": {
                        "code": "INVALID_ARGUMENT",
                        "message": "Trusted mode requests must include X-OpenViking-Account.",
                    },
                },
                raise_for_status=lambda: None,
            )
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"status": "ok", "result": {"ok": True}},
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(client._httpx, "get", capture_get)

    assert client.get("/api/v1/system/status") == {
        "status": "ok",
        "result": {"ok": True},
    }
    assert "X-OpenViking-Account" not in captured_headers[0]
    assert "X-OpenViking-User" not in captured_headers[0]
    assert captured_headers[1]["X-OpenViking-Account"] == "acct"
    assert captured_headers[1]["X-OpenViking-User"] == "usr"


def test_viking_client_health_sends_auth_headers(monkeypatch):
    _clear_openviking_tenant_env(monkeypatch)
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
    assert captured["headers"]["X-OpenViking-Actor-Peer"] == "hermes"
    assert "X-OpenViking-Agent" not in captured["headers"]
    assert "X-OpenViking-Account" not in captured["headers"]
    assert "X-OpenViking-User" not in captured["headers"]


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
    assert captured["headers"]["X-OpenViking-Actor-Peer"] == "hermes"
    assert "X-OpenViking-Account" not in captured["headers"]
    assert "X-OpenViking-User" not in captured["headers"]


def test_viking_client_validate_root_access_uses_admin_accounts(monkeypatch):
    _clear_openviking_tenant_env(monkeypatch)
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
    assert captured["headers"]["X-OpenViking-Actor-Peer"] == "hermes"
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


def test_validate_openviking_setup_values_blocks_remote_without_api_key(monkeypatch):
    class FakeVikingClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("remote configs without API keys should fail before network validation")

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)

    ok, message, role = openviking_module._validate_openviking_setup_values(
        {"endpoint": "https://openviking.example"},
        require_api_key=True,
    )

    assert ok is False
    assert message == "Remote OpenViking configs require an API key."
    assert role is None


def test_validate_openviking_setup_values_local_dev_no_key_uses_health_only(monkeypatch):
    events = []

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "http://localhost:1933"
            assert api_key == ""

        def health_payload(self):
            events.append("health")
            return {"healthy": True, "auth_mode": "dev"}

        def validate_auth(self):
            raise AssertionError("dev-mode no-key setup should not run authenticated status check")

        def validate_root_access(self):
            raise AssertionError("no-key setup should not run root probe")

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)

    ok, message, role = openviking_module._validate_openviking_setup_values(
        {"endpoint": "localhost", "agent": "hermes"}
    )

    assert ok is True
    assert message == ""
    assert role is None
    assert events == ["health"]


def test_validate_openviking_setup_values_user_key_runs_status_and_classifies_role(monkeypatch):
    events = []

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "https://openviking.example"
            assert api_key == "user-key"
            assert account == ""
            assert user == ""

        def health_payload(self):
            events.append("health")
            return {"healthy": True}

        def validate_auth(self):
            events.append("status")
            return {"status": "ok"}

        def validate_root_access(self):
            events.append("admin")
            raise openviking_module._OpenVikingHTTPError("forbidden", 403)

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)

    ok, message, role = openviking_module._validate_openviking_setup_values(
        {"endpoint": "https://openviking.example", "api_key": "user-key"},
        require_api_key=True,
    )

    assert ok is True
    assert message == ""
    assert role == "user"
    assert events == ["health", "status", "admin"]


def test_validate_openviking_setup_values_root_key_runs_admin_probe(monkeypatch):
    events = []

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            assert endpoint == "https://openviking.example"
            assert api_key == "root-key"
            assert account == "acct"
            assert user == "alice"

        def health_payload(self):
            events.append("health")
            return {"healthy": True}

        def validate_auth(self):
            events.append("status")
            return {"status": "ok"}

        def validate_root_access(self):
            events.append("admin")
            return {"accounts": []}

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)

    ok, message, role = openviking_module._validate_openviking_setup_values(
        {
            "endpoint": "https://openviking.example",
            "api_key": "root-key",
            "account": "acct",
            "user": "alice",
        },
        require_api_key=True,
    )

    assert ok is True
    assert message == ""
    assert role == "root"
    assert events == ["health", "status", "admin"]


@pytest.mark.parametrize(
    ("value", "field", "ok"),
    [
        ("acct", "account", True),
        ("alice@example.com", "user", True),
        ("_system", "account", False),
        ("bad/user", "user", False),
        ("alice@@example.com", "user", False),
        (" alice", "user", False),
    ],
)
def test_validate_openviking_identity_value_matches_cli_rules(value, field, ok):
    valid, _message, normalized = openviking_module._validate_openviking_identity_value(
        value,
        field=field,
    )

    assert valid is ok
    assert bool(normalized) is ok
# ---------------------------------------------------------------------------
# on_session_switch — flush + commit + rotate behavior (hermes-agent#28296)
# ---------------------------------------------------------------------------

def _make_provider_with_session(session_id: str, turn_count: int):
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._session_id = session_id
    provider._turn_count = turn_count
    return provider


def test_on_session_switch_commits_old_session_and_rotates_id():
    provider = _make_provider_with_session("old-sid", turn_count=3)

    provider.on_session_switch("new-sid", parent_session_id="old-sid")

    provider._client.post.assert_called_once_with(
        "/api/v1/sessions/old-sid/commit",
        {"keep_recent_count": 0},
    )
    assert provider._session_id == "new-sid"
    assert provider._turn_count == 0


def test_on_session_switch_skips_commit_for_empty_old_session():
    """No turns accumulated → nothing to extract → no commit call."""
    provider = _make_provider_with_session("old-sid", turn_count=0)

    provider.on_session_switch("new-sid")

    provider._client.post.assert_not_called()
    assert provider._session_id == "new-sid"
    assert provider._turn_count == 0


def test_on_session_switch_commits_pending_tokens_without_turn_count():
    provider = _make_provider_with_session("old-sid", turn_count=0)
    provider._client.get.return_value = {"result": {"pending_tokens": 42}}

    provider.on_session_switch("new-sid")

    provider._client.get.assert_called_once_with("/api/v1/sessions/old-sid")
    provider._client.post.assert_called_once_with(
        "/api/v1/sessions/old-sid/commit",
        {"keep_recent_count": 0},
    )
    assert provider._session_id == "new-sid"
    assert provider._turn_count == 0


def test_on_session_switch_rewound_same_session_only_invalidates_prefetch():
    provider = _make_provider_with_session("same-sid", turn_count=3)
    provider._prefetch_generation = 9
    provider._prefetch_result = "stale recall"

    provider.on_session_switch("same-sid", rewound=True)

    provider._client.get.assert_not_called()
    provider._client.post.assert_not_called()
    assert provider._session_id == "same-sid"
    assert provider._turn_count == 3
    assert provider._prefetch_generation == 10
    assert provider._prefetch_result == ""


def test_on_session_switch_clears_stale_prefetch_result():
    provider = _make_provider_with_session("old-sid", turn_count=1)
    provider._prefetch_result = "stale recall from old session"

    provider.on_session_switch("new-sid")

    assert provider._prefetch_result == ""


def test_on_session_switch_waits_for_inflight_sync_thread():
    """In-flight sync_turn write must drain before the commit fires —
    otherwise the commit can race the last message write."""
    provider = _make_provider_with_session("old-sid", turn_count=2)

    join_calls = []

    class FakeThread:
        def __init__(self):
            self._alive = True

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            join_calls.append(timeout)
            # Simulate a worker that finishes within the join window.
            self._alive = False

    provider._inflight_writers["old-sid"] = {FakeThread()}

    provider.on_session_switch("new-sid")

    assert join_calls, "expected on_session_switch to join the in-flight sync thread"
    provider._client.post.assert_called_once_with(
        "/api/v1/sessions/old-sid/commit",
        {"keep_recent_count": 0},
    )


def test_on_session_switch_noop_on_empty_new_id():
    provider = _make_provider_with_session("old-sid", turn_count=5)

    provider.on_session_switch("")
    provider.on_session_switch("   ")

    provider._client.post.assert_not_called()
    assert provider._session_id == "old-sid"
    assert provider._turn_count == 5


def test_on_session_switch_noop_when_client_missing():
    provider = OpenVikingMemoryProvider()
    provider._client = None
    provider._session_id = "old-sid"
    provider._turn_count = 4

    # Must not raise even though no client is configured.
    provider.on_session_switch("new-sid")

    # State stays untouched — provider is effectively disabled.
    assert provider._session_id == "old-sid"
    assert provider._turn_count == 4


def test_sync_turn_captures_session_id_before_worker_runs():
    """Worker must use the session id snapshotted at sync_turn() call time, not
    re-read self._session_id later — otherwise a delayed worker can write the
    previous turn's messages into the rotated-in NEW session."""
    import threading

    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._endpoint = "http://test"
    provider._api_key = ""
    provider._account = "acct"
    provider._user = "usr"
    provider._agent = "hermes"
    provider._session_id = "old-sid"

    started = threading.Event()
    release = threading.Event()
    captured_paths = []
    captured_payloads = []

    def fake_post(path, payload=None, **kwargs):
        started.set()
        release.wait(timeout=2.0)
        captured_paths.append(path)
        captured_payloads.append(payload)
        return {}

    # Patch _VikingClient inside the worker by stubbing post on a client
    # the constructor will produce. Easiest path: monkeypatch the class.
    real_client_cls = _VikingClient

    class StubClient:
        def __init__(self, *a, **kw):
            pass

        def post(self, path, payload=None, **kwargs):
            return fake_post(path, payload, **kwargs)

    import plugins.memory.openviking as _mod
    _mod._VikingClient = StubClient
    try:
        provider.sync_turn("u", "a")
        # Wait until the worker is parked inside the first post call.
        assert started.wait(timeout=2.0), "worker never entered post()"
        # Rotate the provider's session id while the worker is mid-flight.
        provider._session_id = "new-sid"
        release.set()
        for t in list(provider._inflight_writers.get("old-sid", set())):
            t.join(timeout=2.0)
    finally:
        _mod._VikingClient = real_client_cls

    # The whole turn must target the OLD session id as a single ordered batch.
    assert captured_paths == ["/api/v1/sessions/old-sid/messages/batch"]
    assert captured_payloads == [{
        "messages": [
            {"role": "user", "parts": [{"type": "text", "text": "u"}]},
            {"role": "assistant", "parts": [{"type": "text", "text": "a"}], "peer_id": "hermes"},
        ]
    }]


def test_sync_turn_retries_batch_write_with_fresh_client():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._endpoint = "http://test"
    provider._api_key = ""
    provider._account = "acct"
    provider._user = "usr"
    provider._agent = "hermes"
    provider._session_id = "sid-1"

    clients = []
    captured = []

    class StubClient:
        def __init__(self, *a, **kw):
            self.index = len(clients)
            clients.append(self)

        def post(self, path, payload=None, **kwargs):
            if self.index == 0:
                raise RuntimeError("transient")
            captured.append((path, payload))
            return {}

    import plugins.memory.openviking as _mod
    real_client_cls = _mod._VikingClient
    _mod._VikingClient = StubClient
    try:
        provider.sync_turn("u", "a")
        assert provider._drain_writers("sid-1", timeout=2.0)
    finally:
        _mod._VikingClient = real_client_cls

    assert len(clients) == 2
    assert captured == [(
        "/api/v1/sessions/sid-1/messages/batch",
        {
            "messages": [
                {"role": "user", "parts": [{"type": "text", "text": "u"}]},
                {"role": "assistant", "parts": [{"type": "text", "text": "a"}], "peer_id": "hermes"},
            ]
        },
    )]


def test_sync_turn_structured_messages_include_assistant_peer_id():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._endpoint = "http://test"
    provider._api_key = ""
    provider._account = "acct"
    provider._user = "usr"
    provider._agent = "hermes"
    provider._session_id = "sid-structured"

    captured = []

    class StubClient:
        def __init__(self, *a, **kw):
            pass

        def post(self, path, payload=None, **kwargs):
            captured.append((path, payload))
            return {}

    import plugins.memory.openviking as _mod

    real_client_cls = _mod._VikingClient
    _mod._VikingClient = StubClient
    messages = [
        {"role": "user", "content": [{"type": "input_text", "text": "u"}]},
        {
            "role": "assistant",
            "content": "Looking.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "shell_command", "arguments": json.dumps({"cmd": "pwd"})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "shell_command", "content": "ok"},
        {"role": "assistant", "content": [{"type": "output_text", "text": "a"}]},
    ]
    try:
        provider.sync_turn("u", "a", messages=messages)
        assert provider._drain_writers("sid-structured", timeout=2.0)
    finally:
        _mod._VikingClient = real_client_cls

    assert captured == [(
        "/api/v1/sessions/sid-structured/messages/batch",
        {
            "messages": [
                {"role": "user", "parts": [{"type": "text", "text": "u"}]},
                {"role": "assistant", "parts": [{"type": "text", "text": "Looking."}], "peer_id": "hermes"},
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool",
                            "tool_id": "call-1",
                            "tool_name": "shell_command",
                            "tool_input": {"cmd": "pwd"},
                            "tool_output": "ok",
                            "tool_status": "completed",
                        }
                    ],
                    "peer_id": "hermes",
                },
                {"role": "assistant", "parts": [{"type": "text", "text": "a"}], "peer_id": "hermes"},
            ]
        },
    )]


def test_sync_turn_noop_when_session_id_blank():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._session_id = ""

    provider.sync_turn("u", "a")

    # No turn counted, no worker spawned.
    assert provider._turn_count == 0
    assert provider._inflight_writers == {}


def test_on_session_end_marks_session_clean_after_successful_commit():
    """After a successful commit on_session_end must reset _turn_count so a
    subsequent on_session_switch (fired by /new and compression right after
    commit_memory_session) skips its commit instead of double-committing."""
    provider = _make_provider_with_session("old-sid", turn_count=3)

    provider.on_session_end([])

    provider._client.post.assert_called_once_with(
        "/api/v1/sessions/old-sid/commit",
        {"keep_recent_count": 0},
    )
    assert provider._turn_count == 0


def test_on_session_end_keeps_dirty_when_commit_fails():
    """If the commit fails, leave _turn_count > 0 so on_session_switch retries
    rather than silently dropping extraction for the old session."""
    provider = _make_provider_with_session("old-sid", turn_count=3)
    provider._client.post.side_effect = RuntimeError("commit boom")

    provider.on_session_end([])

    assert provider._turn_count == 3


def test_on_session_end_commits_pending_tokens_without_turn_count():
    provider = _make_provider_with_session("old-sid", turn_count=0)
    provider._client.get.return_value = {"result": {"pending_tokens": 42}}

    provider.on_session_end([])

    provider._client.get.assert_called_once_with("/api/v1/sessions/old-sid")
    provider._client.post.assert_called_once_with(
        "/api/v1/sessions/old-sid/commit",
        {"keep_recent_count": 0},
    )


def test_end_then_switch_does_not_double_commit():
    """Mirrors the /new and compression call order: commit_memory_session
    (→ on_session_end) immediately followed by on_session_switch. The switch
    must NOT issue a second commit on the same session id."""
    provider = _make_provider_with_session("old-sid", turn_count=2)

    provider.on_session_end([])
    provider.on_session_switch("new-sid", parent_session_id="old-sid")

    # Exactly one commit call, on the OLD session, fired by on_session_end.
    provider._client.post.assert_called_once_with(
        "/api/v1/sessions/old-sid/commit",
        {"keep_recent_count": 0},
    )
    assert provider._session_id == "new-sid"
    assert provider._turn_count == 0


def test_end_then_switch_with_pending_tokens_does_not_double_commit():
    provider = _make_provider_with_session("old-sid", turn_count=0)
    provider._client.get.return_value = {"result": {"pending_tokens": 42}}

    provider.on_session_end([])
    provider.on_session_switch("new-sid", parent_session_id="old-sid")

    provider._client.post.assert_called_once_with(
        "/api/v1/sessions/old-sid/commit",
        {"keep_recent_count": 0},
    )
    assert provider._session_id == "new-sid"
    assert provider._turn_count == 0


def test_session_needs_commit_guard_wins_over_stale_turn_count():
    """Regression for hermes-agent#28296 review (M3): once a session is marked
    committed, _session_needs_commit must return False even if turn_count is
    still positive. A racing sync_turn can re-increment _turn_count after the
    commit+reset; without the guard ordering, a follow-up finalizer would
    double-commit the same session. The committed-guard must be checked BEFORE
    the turn_count>0 shortcut."""
    provider = _make_provider_with_session("old-sid", turn_count=5)
    provider._mark_session_committed("old-sid")

    # turn_count is a (stale) 5 but the session is already committed.
    assert provider._session_needs_commit("old-sid", 5) is False
    # An uncommitted session with turns still needs a commit.
    assert provider._session_needs_commit("fresh-sid", 5) is True


def test_on_session_switch_swallows_commit_failure():
    """Commit-on-switch must not propagate exceptions: a failing commit on the
    old session must still allow the rotate to the new session to complete,
    otherwise subsequent sync_turn writes would land in the wrong session."""
    provider = _make_provider_with_session("old-sid", turn_count=2)
    provider._client.post.side_effect = RuntimeError("commit boom")

    provider.on_session_switch("new-sid")

    assert provider._session_id == "new-sid"
    assert provider._turn_count == 0


# ---------------------------------------------------------------------------
# Hung-writer protection: the sync worker can outlive the bounded join
# because each OpenViking POST has _TIMEOUT=30s and there are two per turn.
# Committing while late writes are still in flight would orphan them past
# the commit boundary — they would never be extracted.
# ---------------------------------------------------------------------------

class _HungThread:
    """Thread stand-in that stays alive across joins."""

    def is_alive(self):
        return True

    def join(self, timeout=None):
        # Pretend the join timed out — worker still running.
        return None


def test_on_session_end_skips_commit_when_sync_worker_outlives_join():
    """If the sync worker is still alive after the 10s join, the commit must
    be skipped — late writes from the worker would otherwise land in an
    already-committed session and never be extracted. Leave _turn_count
    intact so the session stays marked dirty."""
    provider = _make_provider_with_session("old-sid", turn_count=3)
    provider._inflight_writers["old-sid"] = {_HungThread()}

    provider.on_session_end([])

    provider._client.post.assert_not_called()
    assert provider._turn_count == 3


def test_on_session_switch_skips_commit_when_sync_worker_outlives_join():
    """Same hazard on the switch path. Rotation must still proceed (the new
    session needs to start) but the old-session commit is skipped to avoid
    orphaning the worker's late writes past commit."""
    provider = _make_provider_with_session("old-sid", turn_count=2)
    provider._inflight_writers["old-sid"] = {_HungThread()}

    provider.on_session_switch("new-sid")

    provider._client.post.assert_not_called()
    assert provider._session_id == "new-sid"
    assert provider._turn_count == 0


# ---------------------------------------------------------------------------
# Orphaned-writer hazard: commit must wait for ALL writers for the session,
# not just the latest tracked one. sync_turn's bounded rate-limit can drop a
# still-alive previous worker — that dropped writer keeps POSTing under the
# old sid and would otherwise land its writes past the commit boundary.
# ---------------------------------------------------------------------------

def test_on_session_end_waits_for_all_writers_not_just_latest():
    provider = _make_provider_with_session("old-sid", turn_count=2)
    provider._inflight_writers["old-sid"] = {_HungThread()}

    provider.on_session_end([])

    provider._client.post.assert_not_called()
    assert provider._turn_count == 2


def test_on_session_switch_waits_for_all_writers_not_just_latest():
    provider = _make_provider_with_session("old-sid", turn_count=2)
    provider._inflight_writers["old-sid"] = {_HungThread()}

    provider.on_session_switch("new-sid")

    provider._client.post.assert_not_called()
    assert provider._session_id == "new-sid"
    assert provider._turn_count == 0


def test_on_session_switch_does_not_block_caller_on_slow_drain():
    """Regression for hermes-agent#28296 review (H1): on_session_switch must
    NOT run the old-session drain/commit on the caller's thread. /new, /branch,
    /resume, /undo call this synchronously on the command thread, so a slow
    writer drain (up to _SESSION_DRAIN_TIMEOUT/_DEFERRED_COMMIT_TIMEOUT) or a
    wedged commit POST must not stall the user-facing command. The rotation is
    cheap and synchronous; the commit is offloaded. Mirrors the #41945
    'do not block the turn thread' contract."""
    import threading
    import time

    provider = _make_provider_with_session("old-sid", turn_count=2)

    drain_entered = threading.Event()
    release_drain = threading.Event()

    def slow_drain(sid, timeout):
        drain_entered.set()
        # Simulate a writer that takes a long time to drain.
        release_drain.wait(timeout=10.0)
        return True

    provider._drain_writers = slow_drain

    start = time.monotonic()
    provider.on_session_switch("new-sid")
    elapsed = time.monotonic() - start

    # The caller returned promptly with state already rotated, even though the
    # drain is still parked on the finalizer thread.
    assert elapsed < 1.0, f"on_session_switch blocked the caller for {elapsed:.2f}s"
    assert provider._session_id == "new-sid"
    assert provider._turn_count == 0
    assert drain_entered.wait(timeout=2.0), "finalizer never started draining"
    # No commit yet — drain is still blocked off-thread.
    provider._client.post.assert_not_called()
    # Let the finalizer finish so it doesn't leak past the test.
    release_drain.set()
    assert provider._drain_finalizers(timeout=5.0)
    provider._client.post.assert_called_once_with(
        "/api/v1/sessions/old-sid/commit",
        {"keep_recent_count": 0},
    )


def test_on_session_switch_defers_old_commit_to_finalizer_thread():
    """The switch path rotates session state synchronously (cheap, in-memory)
    but offloads the old-session drain + commit onto a daemon finalizer so the
    caller's command thread (/new, /branch, /resume) never blocks on the up-to
    -_DEFERRED_COMMIT_TIMEOUT drain or the commit POST. See hermes-agent#28296
    review (the #41945 'do not block the turn thread' contract)."""
    import threading

    provider = _make_provider_with_session("old-sid", turn_count=2)
    committed = threading.Event()
    drain_timeouts = []

    def fake_post(path, payload=None):
        committed.set()
        return {}

    def fake_drain(sid, timeout):
        drain_timeouts.append(timeout)
        return True

    provider._client.post.side_effect = fake_post
    provider._drain_writers = fake_drain

    provider.on_session_switch("new-sid")

    # Rotation is synchronous and immediate — the new session is live at once.
    assert provider._session_id == "new-sid"
    assert provider._turn_count == 0
    # The old-session commit lands on the finalizer thread, not inline.
    assert committed.wait(timeout=5.0), "old session was not finalized off-thread"
    provider._client.post.assert_called_once_with(
        "/api/v1/sessions/old-sid/commit",
        {"keep_recent_count": 0},
    )
    # The finalizer drains with the deferred (longer) budget, not inline 10s.
    assert drain_timeouts == [_DEFERRED_COMMIT_TIMEOUT]


def test_sync_turn_tracks_writer_under_session_id():
    """Every sync_turn writer must register under its captured sid so the
    drain at end/switch sees it even if a later sync_turn replaces the
    latest-tracked reference."""
    import threading

    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._endpoint = "http://test"
    provider._api_key = ""
    provider._account = "acct"
    provider._user = "usr"
    provider._agent = "hermes"
    provider._session_id = "sid-1"

    release = threading.Event()
    started = threading.Event()

    class StubClient:
        def __init__(self, *a, **kw):
            pass

        def post(self, path, payload=None, **kwargs):
            started.set()
            release.wait(timeout=2.0)
            return {}

    import plugins.memory.openviking as _mod
    real_client_cls = _mod._VikingClient
    _mod._VikingClient = StubClient
    try:
        provider.sync_turn("u", "a")
        assert started.wait(timeout=2.0), "worker never entered post()"
        assert len(provider._inflight_writers.get("sid-1", set())) == 1
        release.set()
        for t in list(provider._inflight_writers.get("sid-1", set())):
            t.join(timeout=2.0)
    finally:
        _mod._VikingClient = real_client_cls

    # Worker should have removed itself from the inflight set on exit.
    assert provider._inflight_writers.get("sid-1", set()) == set()


# ---------------------------------------------------------------------------
# on_memory_write: explicit memory writes use content/write and stay outside
# the session transcript/commit boundary.
# ---------------------------------------------------------------------------

def test_on_memory_write_uses_content_write_independent_of_session_rotation():
    import threading

    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._endpoint = "http://test"
    provider._api_key = ""
    provider._account = "acct"
    provider._user = "usr"
    provider._agent = "hermes"
    provider._session_id = "old-sid"

    in_ctor = threading.Event()
    release = threading.Event()
    done = threading.Event()
    captured_paths = []
    captured_payloads = []

    class StubClient:
        def __init__(self, *a, **kw):
            in_ctor.set()
            release.wait(timeout=2.0)

        def post(self, path, payload=None, **kwargs):
            captured_paths.append(path)
            captured_payloads.append(payload)
            done.set()
            return {}

    import plugins.memory.openviking as _mod
    real_client_cls = _mod._VikingClient
    _mod._VikingClient = StubClient
    try:
        provider.on_memory_write("add", "user", "remember this")
        assert in_ctor.wait(timeout=2.0), "worker never entered ctor"
        # Rotate provider's session id while the worker is parked. Memory writes
        # must not become session messages in either the old or new session.
        provider._session_id = "new-sid"
        release.set()
        assert done.wait(timeout=2.0), "worker never reached post()"
    finally:
        _mod._VikingClient = real_client_cls

    assert captured_paths == ["/api/v1/content/write"]
    assert captured_payloads[0]["content"] == "remember this"
    assert captured_payloads[0]["mode"] == "create"
    assert captured_payloads[0]["uri"].startswith(
        "viking://user/peers/hermes/memories/preferences/mem_"
    )


# ---------------------------------------------------------------------------
# Prefetch staleness: a prefetch worker that finishes AFTER a session switch
# must drop its result instead of repopulating the new session with stale
# recall from the old generation. Bump the generation directly (rather than
# calling on_session_switch, whose own join blocks on the test worker) so
# the test isolates the generation-gating behavior.
# ---------------------------------------------------------------------------

def test_queue_prefetch_drops_result_when_generation_changed_mid_flight():
    import threading

    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._endpoint = "http://test"
    provider._api_key = ""
    provider._account = "acct"
    provider._user = "usr"
    provider._agent = "hermes"
    provider._session_id = "old-sid"

    started = threading.Event()
    release = threading.Event()

    class StubClient:
        def __init__(self, *a, **kw):
            pass

        def post(self, path, payload=None, **kwargs):
            started.set()
            release.wait(timeout=2.0)
            return {
                "result": {
                    "memories": [
                        {"uri": "viking://memories/old", "score": 0.9,
                         "abstract": "stale from old session"},
                    ],
                    "resources": [],
                }
            }

    import plugins.memory.openviking as _mod
    real_client_cls = _mod._VikingClient
    _mod._VikingClient = StubClient
    try:
        provider.queue_prefetch("anything")
        assert started.wait(timeout=2.0), "prefetch worker never entered post()"
        # Simulate a session switch by bumping the generation directly.
        # The worker captured the pre-bump generation when it was spawned.
        provider._prefetch_generation += 1
        release.set()
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=2.0)
    finally:
        _mod._VikingClient = real_client_cls

    # The stale result from the pre-bump generation must NOT have been written
    # into the new generation's prefetch slot.
    assert provider._prefetch_result == ""


def test_queue_prefetch_sends_limit_not_legacy_top_k():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._endpoint = "http://test"
    provider._api_key = ""
    provider._account = "acct"
    provider._user = "usr"
    provider._agent = "hermes"

    captured_payloads = []

    class StubClient:
        def __init__(self, *a, **kw):
            pass

        def post(self, path, payload=None, **kwargs):
            captured_payloads.append(payload)
            return {"result": {"memories": [], "resources": []}}

    import plugins.memory.openviking as _mod
    real_client_cls = _mod._VikingClient
    _mod._VikingClient = StubClient
    try:
        provider.queue_prefetch("anything")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=2.0)
    finally:
        _mod._VikingClient = real_client_cls

    assert captured_payloads == [{"query": "anything", "limit": 5}]
    assert "top_k" not in captured_payloads[0]
