"""The Yuanbao ``LEFT_CORE`` row: homes core ran Yuanbao for get the ``yuanbao`` catalog plugin, others never.

Real config/.env files under a temp home; the catalog lookup and the install are stand-ins (the live install is
exercised end to end outside the unit suite).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.left_core_migration as lcm


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(lcm, "_attempted", set())
    for name in ("YUANBAO_APP_ID", "YUANBAO_APP_KEY", "YUANBAO_APP_SECRET"):
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


# The platform_toolsets block cli-config.yaml.example shipped while Yuanbao was in core; the installers
# seed config.yaml from it, so its yuanbao row alone says nothing about use.
_SHIPPED_PLATFORM_TOOLSETS = (
    "platform_toolsets:\n  cli: [hermes-cli]\n  telegram: [hermes-telegram]\n  qqbot: [hermes-qqbot]\n"
    "  yuanbao: [hermes-yuanbao]\n  teams: [hermes-teams]\n"
)


@pytest.mark.parametrize(("env", "config", "in_use"), [
    ("YUANBAO_APP_ID=a\nYUANBAO_APP_SECRET=s\n", "", True),
    ("YUANBAO_APP_KEY=a\nYUANBAO_APP_SECRET=s\n", "", True),  # legacy App ID name
    ("", "platforms:\n  yuanbao:\n    enabled: true\n", True),
    ("", "platform_toolsets:\n  telegram: [hermes-telegram, yuanbao]\n", True),
    ("", "toolsets: [hermes-yuanbao]\n", True),
    ("", "", False),
    ("YUANBAO_APP_ID=a\n", "", False),  # half a credential set never enabled the adapter
    ("", _SHIPPED_PLATFORM_TOOLSETS, False),
    ("", "platforms:\n  yuanbao:\n    enabled: false\n", False),
    # Core's _is_connected read the credentials from the platform block's extras too, per field.
    ("", "platforms:\n  yuanbao:\n    extra: {app_id: a, app_secret: s}\n", True),
    ("", "platforms:\n  yuanbao: {app_id: a, app_secret: s}\n", True),
    ("YUANBAO_APP_SECRET=s\n", "platforms:\n  yuanbao:\n    extra: {app_id: a}\n", True),
    ("", "platforms:\n  yuanbao:\n    extra: {app_id: a}\n", False),
    # An explicit disable kept core's adapter off whatever credentials were set, legacy block included.
    ("YUANBAO_APP_ID=a\nYUANBAO_APP_SECRET=s\n", "platforms:\n  yuanbao:\n    enabled: false\n", False),
    ("YUANBAO_APP_ID=a\nYUANBAO_APP_SECRET=s\n", "gateway:\n  yuanbao:\n    enabled: false\n", False),
    # A top-level ``yuanbao:`` block counted too: the loader bridged its keys into extra and its enabled won.
    ("", "yuanbao:\n  app_id: a\n  app_secret: s\n", True),
    ("YUANBAO_APP_SECRET=s\n", "yuanbao:\n  app_id: a\n", True),
    ("YUANBAO_APP_ID=a\nYUANBAO_APP_SECRET=s\n", "yuanbao:\n  enabled: false\n", False),
    ("YUANBAO_APP_ID=a\nYUANBAO_APP_SECRET=s\n", "platforms:\n  yuanbao:\n    enabled: false\nyuanbao:\n  enabled: true\n", True),
    ("", "platforms:\n  yuanbao:\n    extra: {app_secret: ''}\nyuanbao:\n  app_id: a\n  app_secret: s\n", False),  # nested extra wins
    ("", "yuanbao:\n  app_id: a\n  app_secret: s\n  home_channel: {platform: yuanbao, chat_id: c1}\n", True),
])
def test_yuanbao_in_use_matches_what_core_ran(tmp_path, env, config, in_use):
    assert lcm.yuanbao_in_use(_home(tmp_path, env=env, config=config)) is in_use


def test_a_disable_in_legacy_gateway_json_vetoes_env_credentials(tmp_path):
    import json
    home = _home(tmp_path, env="YUANBAO_APP_ID=a\nYUANBAO_APP_SECRET=s\n")
    (home / "gateway.json").write_text(json.dumps({"platforms": {"yuanbao": {"enabled": False}}}))
    assert lcm.yuanbao_in_use(home) is False


def test_a_top_level_yuanbao_block_outranks_legacy_gateway_json_extras(tmp_path):
    import json
    home = _home(tmp_path, config="yuanbao:\n  app_id: a\n  app_secret: s\n")
    (home / "gateway.json").write_text(json.dumps({"platforms": {"yuanbao": {"extra": {"app_secret": ""}}}}))
    assert lcm.yuanbao_in_use(home) is True


def test_a_yuanbao_home_gets_the_plugin_once_with_the_labs_notice(tmp_path):
    home = _home(tmp_path, env="YUANBAO_APP_ID=a\nYUANBAO_APP_SECRET=s\n")
    calls, said = [], []

    def install(name):
        calls.append(name)
        (home / "plugins" / name).mkdir(parents=True)
        return {"ok": True}

    assert lcm.migrate_home(home, install=install, say=said.append) == ["yuanbao"]
    assert lcm.migrate_home(home, install=install, say=said.append) == []
    assert calls == ["yuanbao"]
    assert "Yuanbao is now a Hermes Labs plugin: built by Nous Research, not supported" in said[0]
    # The yb_* tools lived only in hermes-yuanbao: every other platform keeps them off.
    from hermes_cli.config import read_user_config_raw
    known = read_user_config_raw(home / "config.yaml")["known_plugin_toolsets"]
    assert known["telegram"] == ["yuanbao"] and known["cli"] == ["yuanbao"] and "yuanbao" not in known


@pytest.mark.parametrize("yuanbao", [
    "    enabled: true\n",  # /sethome saves enabled: true with the channel; credentials in .env
    "    extra: {app_id: a, app_secret: s}\n",  # no enabled key, credentials in the block
])
def test_a_saved_home_channel_does_not_stop_the_migration(tmp_path, yuanbao):
    """``/sethome`` saved ``platforms.yuanbao.home_channel: {platform: yuanbao, ...}``, and
    ``Platform("yuanbao")`` exists only once the plugin loads. Parsing that channel raised, and the
    home's migration installed nothing, Home Assistant included."""
    home = _home(tmp_path, env="YUANBAO_APP_ID=a\nYUANBAO_APP_SECRET=s\n" if "enabled" in yuanbao else "",
                 config="platforms:\n  homeassistant: {enabled: true}\n  yuanbao:\n" + yuanbao
                 + "    home_channel: {platform: yuanbao, chat_id: c1, name: Home}\n")
    installed = lcm.migrate_home(home, install=lambda name: (home / "plugins" / name).mkdir(parents=True)
                                 or {"ok": True}, say=lambda message: None)
    assert sorted(installed) == ["homeassistant", "yuanbao"]


def test_a_fresh_or_shipped_default_home_installs_nothing(tmp_path):
    for config in ("", _SHIPPED_PLATFORM_TOOLSETS):
        home = tmp_path / ("shipped" if config else "fresh")
        home.mkdir()
        if config:
            (home / "config.yaml").write_text(config, encoding="utf-8")
        said = []
        assert lcm.migrate_home(home, install=lambda name: pytest.fail("installed"), say=said.append) == []
        assert said == []


def test_a_secondary_whose_credentials_come_from_its_secret_scope_uses_yuanbao(tmp_path):
    """A multiplexed secondary's external secret sources reach only the scope bound for it, never its .env."""
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    home = _home(tmp_path)
    other = tmp_path / "other"
    creds = {"YUANBAO_APP_ID": "a", "YUANBAO_APP_SECRET": "s"}
    for bound, in_use in ((home, True), (other, False)):
        token = set_secret_scope(creds, profile_home=str(bound))
        try:
            assert lcm.yuanbao_in_use(home) is in_use
        finally:
            reset_secret_scope(token)


@pytest.mark.parametrize("name", ["platforms/yuanbao", "yuanbao-platform"])
def test_plugins_disabled_naming_the_bundled_adapter_is_not_use_and_still_disables_the_plugin(tmp_path, name):
    from hermes_cli.plugins_discovery import gate_manifest
    from hermes_cli.plugins_manifest import PluginManifest
    home = _home(tmp_path, env="YUANBAO_APP_ID=a\nYUANBAO_APP_SECRET=s\n",
                 config=f"plugins:\n  disabled: [{name}]\n")
    assert lcm.yuanbao_in_use(home) is False
    manifest = PluginManifest(name="yuanbao", source="user", path=str(home / "plugins" / "yuanbao"), key="yuanbao")
    assert gate_manifest(manifest, {name}, None).error == "disabled via config"
    assert lcm.legacy_plugin_alias(name) == "yuanbao"  # `hermes plugins <verb> <name>` names the catalog plugin
    # Installed and enabled (before the update, or by hand): core's old disable still wins.
    assert gate_manifest(manifest, {name}, {"yuanbao"}).error == \
        "listed in both plugins.enabled and plugins.disabled — disabled wins"


def test_a_bare_yuanbao_in_plugins_disabled_is_still_use(tmp_path):
    """Core's adapter was ``platforms/yuanbao`` / ``yuanbao-platform``: a bare ``yuanbao`` never stopped it, so
    the home gets the plugin. The install's enable lifts that entry; one that survives it (a managed overlay
    pin) now turns the plugin off, and the notice says so."""
    from hermes_cli.config import read_user_config_raw
    home = _home(tmp_path, env="YUANBAO_APP_ID=a\nYUANBAO_APP_SECRET=s\n",
                 config="plugins:\n  disabled: [yuanbao]\n")
    assert lcm.yuanbao_in_use(home) is True
    sentence = "plugins.disabled lists 'yuanbao', which now turns Yuanbao off; remove it to keep using Yuanbao"
    for lifted in (True, False):
        (home / "config.yaml").write_text("plugins:\n  disabled: [yuanbao]\n", encoding="utf-8")
        said = []

        def install(name):
            (home / "plugins" / name).mkdir(parents=True, exist_ok=True)
            if lifted:  # what the real enable does (_apply_activation)
                (home / "config.yaml").write_text("plugins:\n  enabled: [yuanbao]\n  disabled: []\n", encoding="utf-8")
            return {"ok": True}

        lcm._attempted.clear()
        import shutil
        shutil.rmtree(home / "plugins", ignore_errors=True)
        assert lcm.migrate_home(home, install=install, say=said.append) == ["yuanbao"]
        assert (sentence in said[0]) is (not lifted)
        assert ("yuanbao" in (read_user_config_raw(home / "config.yaml")["plugins"].get("disabled") or [])) is (not lifted)

    # A failed install leaves the entry, but the install command it names lifts it: no sentence there either.
    (home / "config.yaml").write_text("plugins:\n  disabled: [yuanbao]\n", encoding="utf-8")
    shutil.rmtree(home / "plugins", ignore_errors=True)
    lcm._attempted.clear()
    said = []
    assert lcm.migrate_home(home, install=lambda name: {"ok": False, "error": "offline"}, say=said.append) == []
    assert said and sentence not in said[-1]


def test_every_platform_core_knows_but_yuanbao_keeps_the_yb_tools_off():
    from gateway.config import Platform
    from hermes_cli.platforms import PLATFORMS
    feature = next(f for f in lcm.LEFT_CORE if f.plugin == "yuanbao")
    assert "homeassistant" in feature.off_platforms and "yuanbao" not in feature.off_platforms
    assert set(PLATFORMS) <= set(feature.off_platforms)
    # Gateway-only platforms (msgraph_webhook, relay) host sessions in the same process as the adapter.
    assert {p.value for p in Platform} - {"local", "yuanbao"} <= set(feature.off_platforms)


def test_a_legacy_hermes_list_kept_the_yb_tools_off_and_keeps_them_off(tmp_path):
    """tools_config reads a saved ``[hermes]`` as hermes-cli + hermes-api-server, neither of which carried yb_*;
    ``hermes-gateway`` included ``hermes-yuanbao``, so that row keeps them."""
    home = _home(tmp_path, env="YUANBAO_APP_ID=a\nYUANBAO_APP_SECRET=s\n",
                 config="platform_toolsets:\n  telegram: [hermes]\n  slack: [hermes-gateway]\n")
    lcm.migrate_home(home, install=lambda name: (home / "plugins" / name).mkdir(parents=True) or {"ok": True},
                     say=lambda message: None)
    from hermes_cli.config import read_user_config_raw
    known = read_user_config_raw(home / "config.yaml")["known_plugin_toolsets"]
    assert known["telegram"] == ["yuanbao"] and "slack" not in known


@pytest.mark.parametrize("bundle", ["hermes-yuanbao", "hermes-gateway"])
def test_a_disabled_hermes_yuanbao_bundle_keeps_the_yb_tools_disabled(tmp_path, bundle):
    """Core's ``hermes-yuanbao`` bundle was the core tools plus yb_*, and ``hermes-gateway`` included it;
    ``agent.disabled_toolsets`` naming either removed the yb_* tools. Neither does once the bundle is gone,
    so the conversion disables the plugin's ``yuanbao`` toolset there, once, and turns nothing else off."""
    home = _home(tmp_path, env="YUANBAO_APP_ID=a\nYUANBAO_APP_SECRET=s\n",
                 config=f"agent:\n  disabled_toolsets: [{bundle}, memory]\n")
    for _ in range(2):
        lcm.migrate_home(home, install=lambda name: (home / "plugins" / name).mkdir(parents=True, exist_ok=True)
                         or {"ok": True}, say=lambda message: None)
    from hermes_cli.config import read_user_config_raw
    assert read_user_config_raw(home / "config.yaml")["agent"]["disabled_toolsets"] == [
        bundle, "memory", "yuanbao"]
