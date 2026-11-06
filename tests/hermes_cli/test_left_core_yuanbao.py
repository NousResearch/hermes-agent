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
])
def test_yuanbao_in_use_matches_what_core_ran(tmp_path, env, config, in_use):
    assert lcm.yuanbao_in_use(_home(tmp_path, env=env, config=config)) is in_use


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


@pytest.mark.parametrize("name", ["platforms/yuanbao", "yuanbao-platform", "yuanbao"])
def test_plugins_disabled_naming_the_bundled_adapter_is_not_use_and_still_disables_the_plugin(tmp_path, name):
    from hermes_cli.plugins_discovery import gate_manifest
    from hermes_cli.plugins_manifest import PluginManifest
    home = _home(tmp_path, env="YUANBAO_APP_ID=a\nYUANBAO_APP_SECRET=s\n",
                 config=f"plugins:\n  disabled: [{name}]\n")
    assert lcm.yuanbao_in_use(home) is False
    manifest = PluginManifest(name="yuanbao", source="user", path=str(home / "plugins" / "yuanbao"), key="yuanbao")
    assert gate_manifest(manifest, {"platforms/yuanbao"}, None).error == "disabled via config"


def test_every_platform_core_knows_but_yuanbao_keeps_the_yb_tools_off():
    from hermes_cli.platforms import PLATFORMS
    feature = next(f for f in lcm.LEFT_CORE if f.plugin == "yuanbao")
    assert "homeassistant" in feature.off_platforms and "yuanbao" not in feature.off_platforms
    assert set(PLATFORMS) <= set(feature.off_platforms)
