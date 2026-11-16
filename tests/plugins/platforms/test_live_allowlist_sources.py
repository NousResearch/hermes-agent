"""A live allowlist refresh reads only the config the gate came from, and publishes all-or-nothing.

An adapter built from ``gateway --config custom.yaml`` keeps that file's policy when the home
config.yaml changes; a config.yaml edit whose typed construction fails leaves the adapter AND the
runner's env-reading authorization on the last good lists.
"""

import itertools
import os
import time

import hermes_yaml as yaml
import pytest

from gateway.config import Platform, load_gateway_config

_WRITES = itertools.count()


def _write(home, doc):
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    stamp = time.time_ns() - 10_000_000_000 + next(_WRITES) * 1_000_000
    os.utime(path, ns=(stamp, stamp))


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in list(os.environ):
        if name.startswith(("DISCORD_", "TELEGRAM_", "HERMES_BRIDGED_", "GATEWAY_ALLOW")):
            monkeypatch.delenv(name)
    for var in ("DISCORD_ALLOWED_USERS", "TELEGRAM_ALLOWED_USERS"):  # teardown removes bridged writes
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
        monkeypatch.setenv(f"HERMES_BRIDGED_{var}", "")
        monkeypatch.delenv(f"HERMES_BRIDGED_{var}")
    return tmp_path


def test_custom_config_adapter_ignores_home_config_edits(home):
    from gateway.config_loader import load_gateway_config_file
    from plugins.platforms.discord.adapter import DiscordAdapter

    _write(home, {"discord": {"allow_from": ["222"]}})
    custom = home / "custom.yaml"
    custom.write_text(yaml.safe_dump({"platforms": {"discord": {"enabled": True, "extra": {"allow_from": ["111"]}}}}))
    adapter = DiscordAdapter(load_gateway_config_file(str(custom)).platforms[Platform.DISCORD])
    adapter._snapshot_gate_env()
    adapter._allowed_user_ids = adapter._get_allowed_users()
    adapter._allowed_role_ids = adapter._get_allowed_roles()
    adapter._mark_live_gate_baseline()
    assert (adapter._is_allowed_user("111"), adapter._is_allowed_user("222")) == (True, False)
    _write(home, {"discord": {"allow_from": ["222"]}, "display": {"tool_preview_length": 123}})
    assert (adapter._is_allowed_user("111"), adapter._is_allowed_user("222")) == (True, False)


def test_failed_typed_construction_publishes_no_env_grant(home):
    from gateway.platforms._shared import mark_live_gate_baseline
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from tests.gateway.test_multiplex_interactive_auth import _runner

    _write(home, {"telegram": {"allow_from": ["111"]}})
    adapter = TelegramAdapter(load_gateway_config().platforms[Platform.TELEGRAM])
    mark_live_gate_baseline(adapter)
    runner = _runner(home)
    runner.config.multiplex_profiles = False
    runner.config.profile_routes = []
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.config.platforms = {Platform.TELEGRAM: adapter.config}
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.TELEGRAM))
    assert adapter._is_callback_user_authorized("222", chat_id="222", chat_type="private") is False
    # The Telegram list bridges fine; another platform's block then fails typed construction.
    _write(home, {"telegram": {"allow_from": ["222"]}, "platforms": {"discord": {"home_channel": {"name": "x"}}}})
    assert adapter._is_callback_user_authorized("222", chat_id="222", chat_type="private") is False
    assert os.environ.get("TELEGRAM_ALLOWED_USERS") == "111"
    assert adapter.config.extra.get("allow_from") == ["111"]


def test_a_non_strict_load_that_fails_mid_bridge_restores_the_last_good_env(home, monkeypatch):
    """The non-strict fallback swallows the YAML failure and continues on env/gateway.json, so the outer
    transaction never unwinds: only the abort may undo the partial write, and it must retract nothing."""
    from gateway.platforms._shared import bridged_env_marker

    for var in ("DISCORD_ALLOWED_ROLES", "HERMES_BRIDGED_DISCORD_ALLOWED_ROLES"):  # teardown removes bridged writes
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    _write(home, {"discord": {"allow_from": ["111"], "allowed_roles": ["2"]}})
    load_gateway_config()
    good = {name: (os.environ.get(name), os.environ.get(bridged_env_marker(name)))
            for name in ("DISCORD_ALLOWED_USERS", "DISCORD_ALLOWED_ROLES")}
    assert good["DISCORD_ALLOWED_USERS"][0] == "111" and good["DISCORD_ALLOWED_ROLES"][0] == "2"
    # The Discord list bridges, then the boolean ``telegram:`` block raises later in the same YAML stage.
    _write(home, {"discord": {"allow_from": ["222"]}, "require_mention": True, "telegram": True})
    load_gateway_config()
    assert {name: (os.environ.get(name), os.environ.get(bridged_env_marker(name))) for name in good} == good


@pytest.mark.parametrize("platform,var", [(Platform.TELEGRAM, "TELEGRAM_REPLY_TO_MODE"),
                                          (Platform.DISCORD, "DISCORD_REPLY_TO_MODE")])
def test_a_removed_setting_leftover_never_shapes_the_load_that_retracts_it(home, monkeypatch, platform, var):
    """A restart child inherits a bridged value whose YAML key was since removed: the env-override pass
    of the same load must not read it, and a load that then fails restores it with the last good env."""
    from gateway.platforms._shared import bridged_env_digest, bridged_env_marker

    monkeypatch.setenv(var, "all")
    monkeypatch.setenv(bridged_env_marker(var), bridged_env_digest("all"))
    broken = {"enabled": True, "home_channel": {"name": "x"}}  # fails typed construction after the YAML layer
    _write(home, {"platforms": {platform.value: broken}})
    with pytest.raises(Exception):
        load_gateway_config()
    assert (os.environ.get(var), os.environ.get(bridged_env_marker(var))) == ("all", bridged_env_digest("all"))
    _write(home, {"platforms": {platform.value: {"enabled": True}}})
    assert load_gateway_config().platforms[platform].reply_to_mode != "all"
    assert var not in os.environ and bridged_env_marker(var) not in os.environ
