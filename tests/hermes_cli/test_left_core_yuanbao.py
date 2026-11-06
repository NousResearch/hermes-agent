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
