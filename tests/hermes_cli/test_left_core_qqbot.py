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


def test_a_qqbot_home_gets_the_plugin_once_with_no_placement_notice(tmp_path):
    home = _home(tmp_path, env=_CREDS)
    calls, said = [], []

    def install(name):
        calls.append(name)
        (home / "plugins" / name).mkdir(parents=True)
        return {"ok": True}

    assert lcm.migrate_home(home, install=install, say=said.append) == ["qqbot"]
    assert lcm.migrate_home(home, install=install, say=said.append) == []
    assert calls == ["qqbot"]
    assert "'qqbot'" in said[0] and "not supported" not in said[0] and "Labs" not in said[0]


def test_a_qqbot_disable_survives_the_automatic_install(tmp_path, monkeypatch):
    """Greptile #159651 (the same path for QQ): a credentialed home core ran QQ Bot for despite a bare
    ``qqbot`` in plugins.disabled gets the plugin through the real install and enable path, yet the entry stays
    and QQ Bot stays off; the outcome line names the enable that turns it on."""
    from types import SimpleNamespace
    import hermes_cli.plugins_cmd_install as pci
    monkeypatch.setattr(pci, "_resolve_source", lambda identifier, catalog_name: (
        SimpleNamespace(name=catalog_name, known_issues=()), catalog_name, [], None))

    def place(entry, identifier, **_kw):  # the clone, network-free
        target = home / "plugins" / entry.name
        target.mkdir(parents=True)
        (target / "plugin.yaml").write_text(f"name: {entry.name}\nversion: 1.0.0\nkind: platform\n", encoding="utf-8")
        return target, {"name": entry.name}, entry.name
    monkeypatch.setattr(pci, "_place_tree", place)
    monkeypatch.setattr("hermes_cli.plugins_admission.admit_plugin_set_change",
                        lambda *a, **k: pytest.fail("the automatic install enabled a plugin the home disabled"))
    home = _home(tmp_path, env=_CREDS, config="plugins:\n  disabled: [qqbot]\n")
    said = []

    assert lcm.migrate_home(home, install=lcm._install_into(home, consent=True), say=said.append) == ["qqbot"]
    from hermes_cli.config import read_user_config_raw
    plugins = read_user_config_raw(home / "config.yaml")["plugins"]
    assert plugins["disabled"] == ["qqbot"] and "qqbot" not in (plugins.get("enabled") or [])
    assert "`hermes plugins enable qqbot` turns it on" in said[0]


@pytest.mark.parametrize(("config", "sentence"), [
    ("plugins:\n  disabled: [qqbot]\n", "plugins.disabled lists 'qqbot', which now turns QQ Bot off"),
    ("plugins:\n  isolation: host\n", "This profile sets plugins.isolation: host"),
    ("", ""),
])
def test_the_notice_says_only_what_changed_for_the_home(tmp_path, config, sentence):
    home = _home(tmp_path, env=_CREDS, config=config)
    (home / "plugins" / "qqbot").mkdir(parents=True)  # the install just placed it
    notice = next(f for f in lcm.LEFT_CORE if f.plugin == "qqbot").notice(home)
    assert sentence in notice if sentence else notice == ""


def test_the_new_key_wins_over_the_old_one_in_the_same_layer(tmp_path):
    home = _home(tmp_path, env=_CREDS,
                 config="platforms:\n  qqbot-platform:\n    enabled: true\n  qqbot:\n    enabled: false\n")
    assert _qqbot_in_use(home) is False


def test_the_gateway_reads_the_old_manifest_key_as_qqbot_once_the_plugin_registers():
    from gateway.config import Platform
    Platform._add_pseudo_member("qqbot")  # what the plugin's platform registration yields
    try:
        assert Platform("qqbot-platform") is Platform("qqbot")
    finally:
        for value in ("qqbot-platform", "qqbot"):
            member = Platform._value2member_map_.pop(value, None)
            if member is not None and member.value == value:
                Platform._member_map_.pop(member._name_, None)


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


def test_an_mcp_server_named_hermes_qqbot_does_not_shadow_the_old_preset(monkeypatch):
    """The static bundle won over a same-named MCP toolset on main; the left-core preset still does."""
    import toolsets
    from hermes_cli import tools_config
    real = toolsets.resolve_toolset
    monkeypatch.setattr(toolsets, "resolve_toolset",
                        lambda name, *a, **k: ["mcp_tool"] if name == "hermes-qqbot" else real(name, *a, **k))
    config = {"platform_toolsets": {"telegram": ["hermes-qqbot", "file"]}}
    assert tools_config._get_platform_tools(config, "telegram") >= {"file", "terminal", "web"}


def test_a_newer_yaml_disable_under_the_old_key_beats_a_gateway_json_enable(tmp_path):
    """Layer order decides between the two spellings: config.yaml's ``qqbot-platform: {enabled: false}`` is
    newer than gateway.json's ``qqbot`` enable, as startup reads them."""
    import json
    home = _home(tmp_path, env=_CREDS, config="platforms:\n  qqbot-platform:\n    enabled: false\n")
    (home / "gateway.json").write_text(json.dumps({"platforms": {"qqbot": {"enabled": True}}}), encoding="utf-8")
    assert _qqbot_in_use(home) is False


def _gateway_qqbot(home: Path, monkeypatch):
    """QQ Bot's ``PlatformConfig`` as the gateway loads *home* once the plugin registers ``qqbot``."""
    from gateway.config import Platform, load_gateway_config
    monkeypatch.setenv("HERMES_HOME", str(home))
    added = "qqbot" not in Platform._value2member_map_
    qqbot = Platform._add_pseudo_member("qqbot") if added else Platform("qqbot")
    try:
        return load_gateway_config().platforms.get(qqbot)
    finally:
        if added:
            for value in ("qqbot-platform", "qqbot"):
                member = Platform._value2member_map_.pop(value, None)
                if member is not None and member.value == value:
                    Platform._member_map_.pop(member._name_, None)


def test_a_yaml_block_under_the_old_key_replaces_gateway_json_credentials_as_the_gateway_reads_it(
        tmp_path, monkeypatch):
    """The gateway maps both spellings to one ``Platform`` and the later block replaces the earlier one whole, so
    gateway.json's ``qqbot`` credentials and enable do not survive a config.yaml ``qqbot-platform`` block: core
    never ran QQ Bot for this home, and the migration does not install it either."""
    import json
    home = _home(tmp_path, config="platforms:\n  qqbot-platform:\n    extra:\n      markdown_support: true\n")
    (home / "gateway.json").write_text(json.dumps({"platforms": {"qqbot": {
        "enabled": True, "extra": {"app_id": "a", "client_secret": "s"}}}}), encoding="utf-8")
    platform = _gateway_qqbot(home, monkeypatch)
    assert platform is not None and platform.enabled is False and "app_id" not in platform.extra
    assert _qqbot_in_use(home) is False


def test_the_later_spelling_in_one_layer_wins_as_the_gateway_reads_it(tmp_path, monkeypatch):
    home = _home(tmp_path, env=_CREDS,
                 config="platforms:\n  qqbot:\n    enabled: false\n  qqbot-platform:\n    enabled: true\n")
    platform = _gateway_qqbot(home, monkeypatch)
    assert platform is not None and platform.enabled is True
    assert _qqbot_in_use(home) is True


@pytest.mark.parametrize("collision", [False, True])
def test_saved_qqbot_preset_still_selects_core_tools_at_runtime(monkeypatch, collision):
    from cron.scheduler import _resolve_cron_enabled_toolsets
    from model_tools import _select_tool_names
    from tools.registry import ToolRegistry
    from toolsets import _resolve_toolset_memo

    registry = ToolRegistry()
    monkeypatch.setattr("tools.registry.registry", registry)
    _resolve_toolset_memo.clear()
    if collision:
        registry.register("mcp_fixture_echo", "mcp-hermes-qqbot", {"name": "mcp_fixture_echo"}, lambda *_: "")
        registry.register_toolset_alias("hermes-qqbot", "mcp-hermes-qqbot")

    picks = _resolve_cron_enabled_toolsets({"enabled_toolsets": ["hermes-qqbot", "file"]}, {})
    tools = _select_tool_names(picks, None, quiet_mode=True)
    assert {"terminal", "web_search"} <= tools
    assert ("mcp_fixture_echo" in tools) is collision


def test_a_qqbot_the_default_installer_placed_and_the_user_removed_stays_removed_and_is_said_off(tmp_path):
    # default_plugins recorded it (qqbot is a default plugin) while core still ran QQ Bot; the user removed that
    # idle tree; creds are still there. The removal sticks, and the home is told QQ Bot is off and how to get it.
    home = _home(tmp_path, env=_CREDS, config="_default_plugins_installed: [qqbot]\n")
    calls, said = [], []
    assert lcm.migrate_home(home, install=lambda name: calls.append(name) or {"ok": True}, say=said.append) == []
    assert calls == []
    off = [line for line in said if "so it is off" in line]
    assert len(off) == 1 and "`hermes plugins install qqbot`" in off[0], said


def test_the_default_installer_skips_a_plugin_the_left_core_migration_placed(tmp_path):
    from hermes_cli import default_plugins
    home = _home(tmp_path, config="_left_core_installed: [qqbot]\n")
    assert "qqbot" in default_plugins._recorded(home)
    assert default_plugins._recorded(home, own_only=True) == set()


def test_one_qqbot_check_merges_the_home_s_platform_block_once(tmp_path, monkeypatch):
    # Split credentials take the predicate past its on/off check to the per-field read: the whole check
    # still merges gateway.json + config.yaml for the platform one time, not once per step.
    home = _home(tmp_path, env="QQ_CLIENT_SECRET=s\n", config="platforms:\n  qqbot:\n    extra: {app_id: a}\n")
    merges, real = [], lcm._merged_platform_block
    monkeypatch.setattr(lcm, "_merged_platform_block", lambda *a, **k: merges.append(a) or real(*a, **k))
    assert _qqbot_in_use(home) is True
    assert len(merges) == 1


@pytest.mark.parametrize("recorded", ["_default_plugins_installed: [qqbot]\n", ""], ids=["default-placed", "user-placed"])
def test_a_present_qqbot_the_home_disables_is_said_off_once_and_stays_disabled(tmp_path, recorded):
    # Core ran QQ Bot despite a bare ``qqbot`` in plugins.disabled; the plugin already in the home honours it, so
    # the upgrade turns QQ Bot off. The disable stays (nothing installs or enables) and the home is told, once.
    home = _home(tmp_path, env=_CREDS, config=recorded + "plugins:\n  disabled: [qqbot]\n")
    (home / "plugins" / "qqbot").mkdir(parents=True)
    calls, said = [], []

    def install(name):
        calls.append(name)
        return {"ok": True}

    assert lcm.migrate_home(home, install=install, say=said.append) == []
    assert lcm.migrate_home(home, install=install, say=said.append) == []
    assert calls == []
    from hermes_cli.config import read_user_config_raw
    assert read_user_config_raw(home / "config.yaml")["plugins"]["disabled"] == ["qqbot"]
    off = [line for line in said if "which now turns QQ Bot off" in line]
    assert len(off) == 1 and "`hermes plugins enable qqbot` turns it on" in off[0], said
