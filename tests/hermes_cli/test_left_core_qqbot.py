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
    ("QQ_APP_ID=a\n", "", False),  # half a credential set never enabled the adapter
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


@pytest.mark.parametrize("name", ["QQ_ALLOWED_USERS", "QQ_ALLOW_ALL_USERS", "QQBOT_ALLOWED_USERS"])
def test_qq_gates_stay_profile_gates_with_the_plugin_absent(name):
    """``QQ_*`` and ``QQBOT_*`` gates stayed out of another profile's children while core shipped QQ Bot
    (``Platform.QQBOT`` named the prefix); the row keeps that while the plugin is absent."""
    from tools.environments.local_env_policy import is_profile_gate_env
    assert is_profile_gate_env(name) is True
