"""``hermes_cli.left_core_migration``: homes that used a feature that left core get its catalog plugin.

Real config/.env files under a temp home and the in-tree catalog; only the network install is a
recording stand-in (the live install is exercised end to end outside the unit suite).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.left_core_migration as lcm


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(lcm, "_attempted", set())
    monkeypatch.setattr(lcm, "_undelivered", {})
    monkeypatch.delenv("HASS_TOKEN", raising=False)
    # The in-tree catalog (this checkout's plugin-catalog/), never the network.
    import hermes_cli.plugin_catalog as pc
    monkeypatch.setattr(pc, "fetch_live_catalog", lambda **_: None)


def _home(tmp_path: Path, name: str = "home", *, env: str = "", config: str = "") -> Path:
    home = tmp_path / name
    home.mkdir(parents=True)
    if env:
        (home / ".env").write_text(env, encoding="utf-8")
    if config:
        (home / "config.yaml").write_text(config, encoding="utf-8")
    return home


@pytest.mark.parametrize(("env", "config"), [
    ("HASS_TOKEN=abc\n", ""),
    ("", "platforms:\n  homeassistant:\n    enabled: true\n"),
    ("", "gateway:\n  platforms:\n    homeassistant:\n      enabled: true\n"),
    ("", "platforms:\n  homeassistant:\n    token: inline-token\n"),
    ("", "platform_toolsets:\n  cli: [hermes-cli, homeassistant]\n"),
    ("", "platform_toolsets:\n  homeassistant: [hermes-homeassistant]\n"),
])
def test_homeassistant_in_use_matches_what_core_activated(tmp_path, env, config):
    assert lcm.homeassistant_in_use(_home(tmp_path, env=env, config=config)) is True


@pytest.mark.parametrize(("env", "config"), [
    ("", ""),
    ("HASS_TOKEN=\n", ""),
    ("", "platforms:\n  homeassistant:\n    enabled: false\n    token: t\n"),
    ("", "platform_toolsets:\n  cli: [hermes-cli]\n"),
    ("OPENAI_API_KEY=sk\n", "platforms:\n  telegram:\n    enabled: true\n"),
])
def test_homeassistant_not_in_use(tmp_path, env, config):
    assert lcm.homeassistant_in_use(_home(tmp_path, env=env, config=config)) is False


def test_process_env_token_counts_only_for_the_active_home(tmp_path, monkeypatch):
    home = _home(tmp_path)
    monkeypatch.setenv("HASS_TOKEN", "from-systemd")
    assert lcm.homeassistant_in_use(home) is False
    assert lcm.homeassistant_in_use(home, process_env=True) is True


def test_migrate_home_installs_the_catalog_plugin_once(tmp_path):
    home = _home(tmp_path, env="HASS_TOKEN=abc\n")
    calls, said = [], []

    def install(name):
        calls.append(name)
        (home / "plugins" / name).mkdir(parents=True)
        return {"ok": True}

    assert lcm.migrate_home(home, install=install, say=said.append) == ["homeassistant"]
    assert calls == ["homeassistant"]
    assert "✓ Home Assistant moved out of core" in said[0]
    # Installed now: a second pass is a no-op and silent.
    said.clear()
    assert lcm.migrate_home(home, install=install, say=said.append) == []
    assert calls == ["homeassistant"] and said == []


def test_migrate_home_reports_a_failed_install_with_the_command(tmp_path):
    home = _home(tmp_path, env="HASS_TOKEN=abc\n")
    said = []
    assert lcm.migrate_home(home, install=lambda n: {"ok": False, "error": "network down"}, say=said.append) == []
    assert "could not be installed automatically: network down" in said[0]
    assert "plugins install homeassistant" in said[0]


def test_home_without_homeassistant_gets_nothing_and_no_notice(tmp_path):
    home = _home(tmp_path, env="OPENAI_API_KEY=sk\n")
    said = []
    assert lcm.migrate_home(home, install=lambda n: pytest.fail("installed"), say=said.append) == []
    assert said == []


def test_a_disabled_or_present_plugin_is_left_alone(tmp_path):
    home = _home(tmp_path, env="HASS_TOKEN=abc\n")
    (home / "plugins" / "homeassistant").mkdir(parents=True)
    assert lcm.migrate_home(home, install=lambda n: pytest.fail("reinstalled"), say=print) == []


def test_catalog_miss_is_reported_not_installed(tmp_path, monkeypatch):
    import hermes_cli.memory_provider_migration as mpm
    monkeypatch.setattr(mpm, "catalog_source", lambda name: None)
    home = _home(tmp_path, env="HASS_TOKEN=abc\n")
    said = []
    assert lcm.migrate_home(home, install=lambda n: pytest.fail("installed"), say=said.append) == []
    assert "cannot find in the plugin catalog" in said[0]


def test_migrate_all_homes_migrates_only_the_profile_that_used_it(tmp_path, monkeypatch):
    import pm.plugins_state as ps
    a = _home(tmp_path, "a", env="HASS_TOKEN=abc\n")
    b = _home(tmp_path, "b", env="OPENAI_API_KEY=sk\n")
    monkeypatch.setattr(ps, "dependency_homes", lambda: [a, b])
    installed_into = []

    def install_into(home):
        def _install(name):
            installed_into.append((home, name))
            return {"ok": True}
        return _install

    monkeypatch.setattr(lcm, "_install_into", install_into)
    said = []
    assert lcm.migrate_all_homes(say=said.append) == ["homeassistant"]
    assert installed_into == [(a, "homeassistant")]
    assert len(said) == 1 and str(a) in said[0]


def test_startup_honours_lazy_install_opt_out(tmp_path, monkeypatch):
    home = _home(tmp_path, env="HASS_TOKEN=abc\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    import pm.install
    monkeypatch.setattr(pm.install, "lazy_installs_allowed", lambda: False)
    monkeypatch.setattr(lcm, "_install_into", lambda h: pytest.fail("installed"))
    said = []
    assert lcm.recover_at_startup(say=said.append) == []
    assert "allow_lazy_installs is off" in said[0] and "plugins install homeassistant" in said[0]
    # Once per process per home.
    assert lcm.recover_at_startup(say=said.append) == [] and len(said) == 1


def test_gateway_start_outcome_waits_for_the_first_agent(tmp_path, monkeypatch):
    home = _home(tmp_path, env="HASS_TOKEN=abc\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    import pm.install
    monkeypatch.setattr(pm.install, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr(lcm, "_install_into", lambda h: (lambda name: {"ok": True}))
    assert lcm.recover_at_startup() == ["homeassistant"]  # gateway start: nobody to tell yet
    said = []
    assert lcm.recover_at_startup(say=said.append) == []  # first agent: delivered, not re-attempted
    assert len(said) == 1 and said[0].startswith("✓ Home Assistant moved out of core")
    assert lcm.recover_at_startup(say=said.append) == [] and len(said) == 1


def test_in_tree_catalog_ships_the_homeassistant_entry():
    from hermes_cli.plugin_catalog import get_catalog_entry
    entry = get_catalog_entry("homeassistant")
    assert entry is not None and entry.category == "platform" and entry.tier == "official"
    assert set(entry.capabilities.provides_tools) == {
        "ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service"}
