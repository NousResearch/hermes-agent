"""The QQ Bot ``LEFT_CORE`` row: homes core ran QQ Bot for get the ``qqbot`` catalog plugin, others never.

Real config/.env files under a temp home; the catalog lookup and the install are stand-ins (the live install is
exercised end to end outside the unit suite).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.left_core_migration as lcm

_CREDS = "QQ_APP_ID=a\nQQ_CLIENT_SECRET=s\n"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(lcm, "_attempted", set())
    for name in ("QQ_APP_ID", "QQ_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    import hermes_cli.memory_provider_migration as mpm
    monkeypatch.setattr(mpm, "catalog_source", lambda name: name)


def _home(tmp_path: Path, *, env: str = "", config: str = "") -> Path:
    home = tmp_path / "home"
    home.mkdir()
    if env:
        (home / ".env").write_text(env, encoding="utf-8")
    if config:
        (home / "config.yaml").write_text(config, encoding="utf-8")
    return home


def _qqbot_in_use(home: Path) -> bool:
    return next(f for f in lcm.LEFT_CORE if f.plugin == "qqbot").in_use(home)


# The platform_toolsets block cli-config.yaml.example shipped while QQ Bot was in core; the installers seed
# config.yaml from it, so its qqbot row alone says nothing about use.
_SHIPPED_PLATFORM_TOOLSETS = "platform_toolsets:\n  cli: [hermes-cli]\n  telegram: [hermes-telegram]\n  qqbot: [hermes-qqbot]\n"


@pytest.mark.parametrize(("env", "config", "in_use"), [
    (_CREDS, "", True),
    # Core's is_connected read app_id + client_secret from the block's extra too, per field.
    ("", "platforms:\n  qqbot:\n    extra: {app_id: a, client_secret: s}\n", True),
    ("QQ_CLIENT_SECRET=s\n", "platforms:\n  qqbot:\n    extra: {app_id: a}\n", True),
    ("", "platforms:\n  qqbot:\n    enabled: true\n", True),
    ("", "qqbot:\n  app_id: a\n  client_secret: s\n", True),  # top-level block, bridged into extra
    ("", "platform_toolsets:\n  telegram: [hermes-qqbot]\n", True),
    ("", "", False),
    ("QQ_APP_ID=a\n", "", False),  # half a set: core enabled the adapter, but it never connected
    ("", "platforms:\n  qqbot:\n    extra: {app_id: a}\n", False),
    ("", _SHIPPED_PLATFORM_TOOLSETS, False),
    # An explicit disable kept core's adapter off whatever credentials were set.
    (_CREDS, "platforms:\n  qqbot:\n    enabled: false\n", False),
    (_CREDS, "qqbot:\n  enabled: false\n", False),
    # plugins.disabled naming core's bundled adapter: core's loader never ran it.
    (_CREDS, "plugins:\n  disabled: [platforms/qqbot]\n", False),
    (_CREDS, "plugins:\n  disabled: [qqbot-platform]\n", False),
])
def test_qqbot_in_use_matches_what_core_ran(tmp_path, env, config, in_use):
    assert _qqbot_in_use(_home(tmp_path, env=env, config=config)) is in_use


def test_a_qqbot_home_gets_the_plugin_once_with_the_labs_notice(tmp_path):
    home = _home(tmp_path, env=_CREDS)
    calls, said = [], []

    def install(name):
        calls.append(name)
        (home / "plugins" / name).mkdir(parents=True)
        return {"ok": True}

    assert lcm.migrate_home(home, install=install, say=said.append) == ["qqbot"]
    assert lcm.migrate_home(home, install=install, say=said.append) == []
    assert calls == ["qqbot"]
    assert ("QQ Bot is now a Hermes Labs plugin: built by Nous Research, not supported, and likely not "
            "maintained going forward. Fixes from the community are welcome.") in said[0]


def test_a_fresh_or_shipped_default_home_installs_nothing(tmp_path):
    for config in ("", _SHIPPED_PLATFORM_TOOLSETS):
        home = tmp_path / ("shipped" if config else "fresh")
        home.mkdir()
        if config:
            (home / "config.yaml").write_text(config, encoding="utf-8")
        said = []
        assert lcm.migrate_home(home, install=lambda name: pytest.fail("installed"), say=said.append) == []
        assert said == []


@pytest.mark.parametrize("name", ["platforms/qqbot", "qqbot-platform"])
def test_core_s_old_disable_still_disables_the_installed_plugin(name):
    """``plugins.disabled`` naming core's adapter kept QQ Bot off; it names the catalog plugin now and still wins."""
    from hermes_cli.plugins_discovery import gate_manifest
    from hermes_cli.plugins_manifest import PluginManifest
    manifest = PluginManifest(name="qqbot", source="user", path="qqbot", key="qqbot")
    assert lcm.legacy_plugin_alias(name) == "qqbot"
    assert gate_manifest(manifest, {name}, {"qqbot"}).error == \
        "listed in both plugins.enabled and plugins.disabled — disabled wins"


def test_child_processes_keep_the_qq_strip_set_core_had():
    """With the plugin absent, children lose the QQ names core's manifest made them lose."""
    from tools.environments import local_env_policy as policy
    assert {"QQ_CLIENT_SECRET", "QQ_STT_API_KEY"} <= policy._ADAPTER_SECRET_ENV & policy._ALWAYS_STRIP_KEYS
    assert {"QQ_APP_ID", "QQ_CLIENT_SECRET", "QQ_ALLOWED_USERS", "QQBOT_HOME_CHANNEL"} <= \
        policy._HERMES_PROVIDER_ENV_BLOCKLIST


def test_yuanbao_tools_stay_off_in_qqbot_sessions():
    """A home upgrading across both removals: core never carried the yb_* tools in hermes-qqbot."""
    assert "qqbot" in next(f for f in lcm.LEFT_CORE if f.plugin == "yuanbao").off_platforms


@pytest.mark.parametrize("name", ["QQ_ALLOWED_USERS", "QQ_ALLOW_ALL_USERS", "QQBOT_ALLOWED_USERS"])
def test_qq_gates_stay_profile_gates_with_the_plugin_absent(name):
    """``QQ_*`` and ``QQBOT_*`` gates stayed out of another profile's children while core shipped QQ Bot
    (``Platform.QQBOT`` named the prefix); the row keeps that while the plugin is absent."""
    from tools.environments.local_env_policy import is_profile_gate_env
    assert is_profile_gate_env(name) is True


def test_the_channels_card_keeps_the_qq_secret_masked_without_the_plugin_manifest(monkeypatch):
    """The dashboard reads field metadata from OPTIONAL_ENV_VARS, which lists the plugin's names only when the
    launch home had the plugin when hermes_cli.config was imported. A card built after an in-process install, or
    for a secondary profile's plugin, must still render QQ_CLIENT_SECRET as a password field, as core did."""
    from hermes_cli import config
    from hermes_cli.web_routers import messaging
    for name in ("QQ_APP_ID", "QQ_CLIENT_SECRET"):
        monkeypatch.delitem(config.OPTIONAL_ENV_VARS, name, raising=False)
    assert messaging._messaging_env_info("QQ_CLIENT_SECRET")["is_password"] is True
    assert messaging._messaging_env_info("QQ_APP_ID")["prompt"] == "QQ App ID"


@pytest.mark.parametrize(("config", "in_use"), [
    ("platforms:\n  qqbot-platform:\n    extra: {app_id: a, client_secret: s}\n", True),
    ("platforms:\n  qqbot-platform:\n    enabled: true\n", True),
    ("platforms:\n  qqbot-platform:\n    enabled: false\n", False),
])
def test_the_old_manifest_key_is_read_as_platforms_qqbot(tmp_path, config, in_use):
    """Core resolved ``platforms.qqbot-platform`` (its manifest name) to ``qqbot``; the migration does too."""
    env = _CREDS if not in_use else ""
    assert _qqbot_in_use(_home(tmp_path, env=env, config=config)) is in_use
    from gateway.config import Platform
    from hermes_cli.left_core_migration import legacy_platform_alias
    assert legacy_platform_alias("qqbot-platform") == "qqbot"
    with pytest.raises(ValueError):  # unregistered: no pseudo member until the plugin registers qqbot
        Platform("qqbot-platform")


def test_reload_and_config_set_still_know_the_qq_env_names():
    """reload_env drops a removed QQ_* line and `config set` routes it to .env, as when core shipped QQ Bot,
    even after an in-process install whose manifest this process never read."""
    from hermes_cli import config
    assert {"QQ_APP_ID", "QQ_CLIENT_SECRET", "QQBOT_HOME_CHANNEL", "QQ_HOME_CHANNEL"} <= config._EXTRA_ENV_KEYS


def test_webhook_routes_keep_a_qqbot_target_before_the_plugin_registers():
    from gateway.webhook_targets import is_dispatchable_target
    assert is_dispatchable_target("qqbot") is True


def test_a_mixed_list_naming_hermes_qqbot_keeps_its_core_tools_and_an_unchecked_one_stays_off():
    """``telegram: [hermes-qqbot, file]`` resolved hermes-qqbot's core tools on main, plugin or not (the bundle was
    core tools only). The picker treats it as the preset it was: unchecking ``terminal`` sticks on the next read."""
    from hermes_cli import tools_config
    config = {"platform_toolsets": {"telegram": ["hermes-qqbot", "file"]}}
    enabled = tools_config._get_platform_tools(config, "telegram")
    assert enabled >= {"file", "terminal", "web"}
    tools_config._save_platform_tools(config, "telegram", enabled - {"terminal"})
    assert "hermes-qqbot" not in config["platform_toolsets"]["telegram"]
    assert "terminal" not in tools_config._get_platform_tools(config, "telegram")


def test_a_bare_hermes_qqbot_row_reads_as_its_core_tools_without_the_plugin():
    """``telegram: [hermes-qqbot]`` alone (the composite path) shows the preset's core tools before the plugin
    loads, so a save from the picker keeps them instead of dropping the preset with nothing in its place."""
    from hermes_cli import tools_config
    config = {"platform_toolsets": {"telegram": ["hermes-qqbot"]}}
    enabled = tools_config._get_platform_tools(config, "telegram")
    assert enabled >= {"file", "terminal", "web"}
    tools_config._save_platform_tools(config, "telegram", enabled)
    assert tools_config._get_platform_tools(config, "telegram") >= {"file", "terminal", "web"}


def test_a_newer_yaml_disable_under_the_old_key_beats_a_gateway_json_enable(tmp_path):
    """Layer order decides between the two spellings: config.yaml's ``qqbot-platform: {enabled: false}`` is
    newer than gateway.json's ``qqbot`` enable, as startup reads them."""
    import json
    home = _home(tmp_path, env=_CREDS, config="platforms:\n  qqbot-platform:\n    enabled: false\n")
    (home / "gateway.json").write_text(json.dumps({"platforms": {"qqbot": {"enabled": True}}}), encoding="utf-8")
    assert _qqbot_in_use(home) is False
