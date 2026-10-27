"""Successful complete YAML loads retract only their unchanged, bridge-owned leftovers."""
import os

import pytest

from hermes_yaml import YAMLError

from gateway.config_loader import load_yaml_layer
from gateway.platforms._shared import bridged_env_marker


_NAMES = ("TELEGRAM_REACTIONS", "TELEGRAM_ALLOWED_USERS", "TELEGRAM_REQUIRE_MENTION", "SIGNAL_REQUIRE_MENTION")


@pytest.fixture(autouse=True)
def clean_bridge(monkeypatch):
    from hermes_cli import env_loader

    monkeypatch.setattr(env_loader, "_LOADED_DOTENV_KEYS", set())
    monkeypatch.setattr(env_loader, "_MANAGED_DOTENV_KEYS", set())
    for name in _NAMES:
        for key in (name, bridged_env_marker(name)):
            monkeypatch.setenv(key, "")
            monkeypatch.delenv(key)


def _load(home, text):
    (home / "config.yaml").write_text(text, encoding="utf-8")
    data = {}
    load_yaml_layer(home, data)
    return data


@pytest.mark.parametrize("replacement", [
    "telegram:\n  allow_from: ['111']\n",  # one key removed
    "display:\n  tool_preview_length: 100\n",  # whole platform removed
    "{}\n",  # empty mapping
    "",  # empty file
])
def test_removed_yaml_keys_retract_only_their_own_previous_values(tmp_path, replacement):
    _load(tmp_path, "telegram:\n  reactions: false\n  allow_from: ['111']\n")
    assert os.environ["TELEGRAM_REACTIONS"] == "false"

    _load(tmp_path, replacement)

    assert "TELEGRAM_REACTIONS" not in os.environ
    assert bridged_env_marker("TELEGRAM_REACTIONS") not in os.environ
    if "allow_from" in replacement:
        assert os.environ["TELEGRAM_ALLOWED_USERS"] == "111"


@pytest.mark.parametrize("owner", ["shell", "dotenv", "managed"])
def test_removed_key_preserves_operator_owned_value(tmp_path, monkeypatch, owner):
    from hermes_cli import env_loader

    _load(tmp_path, "telegram:\n  reactions: false\n")
    if owner == "shell":
        monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    elif owner == "dotenv":
        (tmp_path / ".env").write_text("TELEGRAM_REACTIONS=false\n", encoding="utf-8")
        env_loader.load_hermes_dotenv(hermes_home=tmp_path, load_external_secrets=False)
        assert "TELEGRAM_REACTIONS" in env_loader.launch_dotenv_keys()
    else:
        monkeypatch.setattr(env_loader, "_MANAGED_DOTENV_KEYS", {"TELEGRAM_REACTIONS"})
    _load(tmp_path, "{}\n")
    assert os.environ["TELEGRAM_REACTIONS"] == ("true" if owner == "shell" else "false")


def test_malformed_yaml_never_retracts_last_successful_value(tmp_path):
    _load(tmp_path, "telegram:\n  reactions: false\n")
    with pytest.raises(YAMLError):
        _load(tmp_path, "telegram: [\n")
    assert os.environ["TELEGRAM_REACTIONS"] == "false"
    assert os.environ[bridged_env_marker("TELEGRAM_REACTIONS")] == "false"


@pytest.mark.parametrize("failure", ["discovery", "hook", "core"])
def test_failed_load_does_not_treat_unseen_keys_as_deleted(tmp_path, monkeypatch, failure):
    from gateway import config_loader
    from gateway.platform_registry import platform_registry
    from hermes_cli import plugins

    _load(tmp_path, "telegram:\n  reactions: false\n")

    def fail(*args, **kwargs):
        raise RuntimeError("test load failure")

    if failure == "discovery":
        monkeypatch.setattr(plugins, "discover_plugins", fail)
    elif failure == "hook":
        monkeypatch.setattr(platform_registry.get("telegram"), "apply_yaml_config_fn", fail)
    else:
        monkeypatch.setattr(config_loader, "bridge_core_env_settings", fail)
    if failure == "core":
        with pytest.raises(RuntimeError, match="test load failure"):
            _load(tmp_path, "telegram: {}\n")
    else:
        _load(tmp_path, "telegram: {}\n")
    assert os.environ["TELEGRAM_REACTIONS"] == "false"
    assert os.environ[bridged_env_marker("TELEGRAM_REACTIONS")] == "false"


def test_secondary_load_neither_refreshes_nor_retracts_launch_environment(tmp_path):
    from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope

    _load(tmp_path, "telegram:\n  reactions: false\n")
    set_multiplex_active(True)
    token = set_secret_scope({})
    try:
        _load(tmp_path, "{}\n")
        _load(tmp_path, "telegram:\n  reactions: true\n")
        assert os.environ["TELEGRAM_REACTIONS"] == "false"
        assert os.environ[bridged_env_marker("TELEGRAM_REACTIONS")] == "false"
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)


def test_core_fallback_retains_ownership_after_platform_block_is_removed(tmp_path):
    _load(tmp_path, "telegram:\n  require_mention: false\n")
    _load(tmp_path, "require_mention: true\n")
    assert os.environ["TELEGRAM_REQUIRE_MENTION"] == "true"
    _load(tmp_path, "{}\n")
    assert "TELEGRAM_REQUIRE_MENTION" not in os.environ


def test_signal_core_key_uses_the_same_refresh_and_retraction_contract(tmp_path):
    _load(tmp_path, "signal:\n  require_mention: true\n")
    _load(tmp_path, "signal:\n  require_mention: false\n")
    assert os.environ["SIGNAL_REQUIRE_MENTION"] == "false"
    _load(tmp_path, "{}\n")
    assert "SIGNAL_REQUIRE_MENTION" not in os.environ


def test_missing_config_file_also_retracts_a_previous_bridge_value(tmp_path):
    _load(tmp_path, "telegram:\n  reactions: false\n")
    (tmp_path / "config.yaml").unlink()
    load_yaml_layer(tmp_path, {})
    assert "TELEGRAM_REACTIONS" not in os.environ
    assert bridged_env_marker("TELEGRAM_REACTIONS") not in os.environ


def test_nested_load_does_not_clear_the_outer_loads_partial_writes(tmp_path, monkeypatch):
    from gateway.platform_registry import platform_registry

    _load(tmp_path, "telegram:\n  reactions: false\n")
    entry = platform_registry.get("telegram")
    hook = entry.apply_yaml_config_fn
    nested = tmp_path / "nested"
    nested.mkdir()

    def nested_hook(yaml_cfg, effective):
        result = hook(yaml_cfg, effective)
        _load(nested, "{}\n")
        assert os.environ["TELEGRAM_REACTIONS"] == "true"
        return result

    monkeypatch.setattr(entry, "apply_yaml_config_fn", nested_hook)
    _load(tmp_path, "telegram:\n  reactions: true\n")
    assert os.environ["TELEGRAM_REACTIONS"] == "true"


def test_a_copied_context_in_another_thread_gets_a_complete_new_load(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    from gateway.platforms._shared import yaml_env_refresh

    _load(tmp_path, "telegram:\n  reactions: false\n")
    with yaml_env_refresh():
        context = copy_context()
    assert "TELEGRAM_REACTIONS" not in os.environ
    _load(tmp_path, "telegram:\n  reactions: false\n")
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(context.run, _load, tmp_path, "{}\n").result(timeout=60)
    assert "TELEGRAM_REACTIONS" not in os.environ
