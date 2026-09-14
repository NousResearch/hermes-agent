"""Home provenance survives only the same configured delivery destination."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gateway import config as gateway_config, config_env
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig


def home_data(platform=Platform.TELEGRAM):
    return {
        "platform": platform.value,
        "chat_id": "selected-chat",
        "name": "Selected Home",
        "thread_id": "selected-topic",
        "user_id": "selected-owner",
        "scope_id": "selected-scope",
        "selection_id": "selection-a",
        "group_audience_ack": "audience-a",
    }


def test_home_identity_roundtrip():
    data = home_data()
    assert HomeChannel.from_dict(data).to_dict() == data


@pytest.mark.parametrize("invalid", [None, 7, True, {}, []])
def test_non_string_selection_and_audience_are_not_coerced(invalid):
    data = {**home_data(), "selection_id": invalid, "group_audience_ack": invalid}
    result = HomeChannel.from_dict(data).to_dict()
    assert "selection_id" not in result
    assert "group_audience_ack" not in result
    assert result["user_id"] == "selected-owner"


@pytest.mark.parametrize("platform", [Platform.TELEGRAM, Platform.SIGNAL, Platform.WHATSAPP])
@pytest.mark.parametrize("target", ["same", "chat", "thread"])
def test_env_pipeline_restores_only_matching_home(monkeypatch, platform, target):
    from gateway.platform_registry import platform_registry
    from hermes_cli import plugins

    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    monkeypatch.setattr(platform_registry, "plugin_entries", lambda: [])
    values = {
        f"{platform.value.upper()}_HOME_CHANNEL": "other-chat" if target == "chat" else "selected-chat",
        f"{platform.value.upper()}_HOME_CHANNEL_THREAD_ID": "other-topic" if target == "thread" else "selected-topic",
    }
    monkeypatch.setattr(config_env, "getenv", lambda key, default="": values.get(key, default))
    current = GatewayConfig(platforms={platform: PlatformConfig(
        enabled=False, home_channel=HomeChannel.from_dict(home_data(platform)),
    )})
    config_env._apply_env_overrides(current)
    selected = current.get_home_channel(platform)
    assert selected is not None
    assert current.platforms[platform].enabled is False
    for key in ("user_id", "scope_id", "selection_id", "group_audience_ack"):
        assert getattr(selected, key, None) == (home_data(platform)[key] if target == "same" else None)
    if target == "same":
        assert selected.name == "Selected Home"


@pytest.mark.parametrize("enabled_if_new", [False, True])
def test_persistence_serializes_read_and_write_without_enabling_existing_platform(monkeypatch, enabled_if_new):
    from hermes_cli import config as cli_config

    current = {"platforms": {"telegram": {"enabled": False, "extra": {"keep": 7}}}, "unrelated": "kept"}
    seen = []

    def load():
        assert cli_config._CONFIG_LOCK._is_owned()
        seen.append("read")
        return current

    def save(value):
        assert cli_config._CONFIG_LOCK._is_owned()
        seen.append("write")
        assert value["platforms"]["telegram"] == {
            "enabled": False, "extra": {"keep": 7}, "home_channel": home_data(),
        }
        assert value["unrelated"] == "kept"

    monkeypatch.setattr(cli_config, "load_config", load)
    monkeypatch.setattr(cli_config, "save_config", save)
    gateway_config.persist_home_channel(HomeChannel.from_dict(home_data()), enabled_if_new=enabled_if_new)
    assert seen == ["read", "write"]


def test_real_save_and_loader_keep_home_identity_and_unrelated_config(tmp_path, monkeypatch):
    import yaml
    from gateway.platform_registry import platform_registry
    from hermes_cli import config as cli_config, plugins

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    monkeypatch.setattr(platform_registry, "plugin_entries", lambda: [])
    monkeypatch.setattr(platform_registry, "all_entries", lambda: [])
    monkeypatch.setattr(cli_config, "_LOAD_CONFIG_CACHE", {})
    monkeypatch.setattr(cli_config, "_RAW_CONFIG_CACHE", {})
    monkeypatch.setattr(cli_config, "_LAST_EXPANDED_CONFIG_BY_PATH", {})
    path = tmp_path / "config.yaml"
    path.write_text("platforms:\n  telegram:\n    enabled: false\nagent:\n  max_turns: 7\n", encoding="utf-8")
    gateway_config.persist_home_channel(HomeChannel.from_dict(home_data()))
    stored = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert stored["platforms"]["telegram"]["home_channel"] == home_data()
    assert stored["agent"]["max_turns"] == 7
    values = {"TELEGRAM_HOME_CHANNEL": "selected-chat", "TELEGRAM_HOME_CHANNEL_THREAD_ID": "selected-topic"}
    monkeypatch.setattr(config_env, "getenv", lambda key, default="": values.get(key, default))
    loaded = gateway_config.load_gateway_config()
    assert loaded.get_home_channel(Platform.TELEGRAM).to_dict() == home_data()
    assert loaded.platforms[Platform.TELEGRAM].enabled is False


@pytest.mark.parametrize("raw", ["platforms: [broken", "- not-a-mapping\n"])
def test_home_persistence_does_not_replace_corrupt_config(tmp_path, monkeypatch, raw):
    from hermes_cli import config as cli_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cli_config, "_LOAD_CONFIG_CACHE", {})
    monkeypatch.setattr(cli_config, "_RAW_CONFIG_CACHE", {})
    monkeypatch.setattr(cli_config, "_LAST_EXPANDED_CONFIG_BY_PATH", {})
    path = tmp_path / "config.yaml"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        gateway_config.persist_home_channel(HomeChannel.from_dict(home_data()))
    assert path.read_text(encoding="utf-8") == raw


@pytest.mark.parametrize("changed,value", [
    ("platform", "signal"), ("chat_id", "foreign-chat"), ("thread_id", "foreign-topic"),
    ("user_id", "foreign-owner"), ("scope_id", "foreign-scope"),
])
def test_conflicting_overlay_cannot_acquire_prior_selection_or_audience(changed, value):
    from gateway.config_io import restore_matching_home_bindings, snapshot_home_bindings

    current = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(
        home_channel=HomeChannel.from_dict(home_data()),
    )})
    snapshot = snapshot_home_bindings(current)
    incoming = {k: v for k, v in home_data().items() if k not in {
        "user_id", "scope_id", "selection_id", "group_audience_ack",
    }}
    incoming[changed] = value
    replacement = HomeChannel.from_dict(incoming)
    current.platforms[Platform.TELEGRAM].home_channel = replacement
    restore_matching_home_bindings(current, snapshot)
    assert replacement.selection_id is None
    assert replacement.group_audience_ack is None
    assert replacement.to_dict() == incoming


def test_snapshot_is_independent_and_removed_home_is_not_recreated():
    from gateway.config_io import restore_matching_home_bindings, snapshot_home_bindings

    home = HomeChannel.from_dict(home_data())
    current = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(home_channel=home)})
    snapshot = snapshot_home_bindings(current)
    home.chat_id = "changed-in-place"
    assert snapshot[Platform.TELEGRAM].to_dict() == home_data()
    current.platforms[Platform.TELEGRAM].home_channel = None
    restore_matching_home_bindings(current, snapshot)
    assert current.get_home_channel(Platform.TELEGRAM) is None


def test_real_dotenv_cold_reload_keeps_selected_home(tmp_path):
    (tmp_path / "config.yaml").write_text(json.dumps({
        "platforms": {"telegram": {"enabled": False, "home_channel": home_data()}},
    }), encoding="utf-8")
    (tmp_path / ".env").write_text(
        "TELEGRAM_HOME_CHANNEL=selected-chat\nTELEGRAM_HOME_CHANNEL_THREAD_ID=selected-topic\n",
        encoding="utf-8",
    )
    code = """
import json, os
from pathlib import Path
from hermes_cli.env_loader import load_hermes_dotenv
home = Path(os.environ['HERMES_HOME'])
load_hermes_dotenv(hermes_home=home, project_env=home/'absent-project-env', load_external_secrets=False)
from gateway.config import load_gateway_config, Platform
current = load_gateway_config()
assert current.platforms[Platform.TELEGRAM].enabled is False
print('HOME=' + json.dumps(current.get_home_channel(Platform.TELEGRAM).to_dict(), sort_keys=True))
"""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
        env={"HOME": str(tmp_path), "HERMES_HOME": str(tmp_path), "PATH": os.environ["PATH"],
             "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=20, check=True)
    line = next(line for line in result.stdout.splitlines() if line.startswith("HOME="))
    assert json.loads(line.removeprefix("HOME=")) == home_data()
