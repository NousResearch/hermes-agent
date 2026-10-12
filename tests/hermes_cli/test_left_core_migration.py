"""``hermes_cli.left_core_migration``: homes that used a feature that left core get its catalog plugin.

Real config/.env files under a temp home; the catalog lookup and the network install are stand-ins
(the live install is exercised end to end outside the unit suite).
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
    # A catalog that lists every left-core plugin; never the network.
    import hermes_cli.memory_provider_migration as mpm
    monkeypatch.setattr(mpm, "catalog_source", lambda name: name)


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


_JSON_ON = '{"platforms": {"homeassistant": {"enabled": true, "token": "t"}}}'


@pytest.mark.parametrize(("gateway_json", "config", "in_use"), [
    (_JSON_ON, "", True),
    ("", "gateway:\n  homeassistant:\n    enabled: true\n    token: t\n", True),
    ("", "gateway:\n  platforms:\n    homeassistant:\n      enabled: true\n      token: t\n"
         "platforms:\n  homeassistant:\n    extra:\n      url: http://ha.local\n", True),
    (_JSON_ON, "platforms:\n  homeassistant:\n    enabled: false\n", False),
])
def test_in_use_reads_the_platform_config_the_gateway_loader_merged(tmp_path, gateway_json, config, in_use):
    """legacy gateway.json, gateway.<platform> shorthand and nested+top-level blocks enabled core's
    adapter; the last explicit ``enabled`` (here config.yaml over gateway.json) still wins."""
    home = _home(tmp_path, config=config)
    if gateway_json:
        (home / "gateway.json").write_text(gateway_json, encoding="utf-8")
    assert lcm.homeassistant_in_use(home) is in_use


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


def test_migration_keeps_a_toolset_off_where_core_had_it_off(tmp_path):
    """Core's homeassistant toolset was on only where a platform's saved list (or default composite)
    carried it; as a plugin toolset it is on everywhere not in known_plugin_toolsets."""
    from hermes_cli.tools_config import _enabled_plugin_toolsets
    home = _home(tmp_path, env="HASS_TOKEN=abc\n", config=(
        "platform_toolsets:\n  telegram: [web, terminal, file]\n  discord: [hermes-discord]\n"
        "  slack: [web, homeassistant]\n"))
    lcm.migrate_home(home, install=lambda n: {"ok": True}, say=lambda m: None)
    config = lcm._read_config(home)
    enabled = {p: bool(_enabled_plugin_toolsets(config, p, sel, {"homeassistant"})) for p, sel in {
        "telegram": ["web", "terminal", "file"], "discord": ["hermes-discord"], "slack": ["web", "homeassistant"],
        "cli": ["hermes-cli"], "webhook": ["hermes-webhook"], "acp": ["hermes-acp"]}.items()}
    assert enabled == {"telegram": False, "discord": True, "slack": True, "cli": True,
                       "webhook": False, "acp": False}


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


def test_a_preinstalled_plugin_gets_the_scope_conversion_once(tmp_path):
    """A plugin installed before the update (inert while core shipped HA) is not proof the core-era
    toolset scope was converted; the conversion runs once and never undoes a later user choice."""
    from hermes_cli.config import read_user_config_raw, save_config
    from hermes_cli.tools_config import _enabled_plugin_toolsets, _save_platform_tools
    home = _home(tmp_path, env="HASS_TOKEN=abc\n", config="platform_toolsets:\n  telegram: [web]\n")
    (home / "plugins" / "homeassistant").mkdir(parents=True)
    selections = {"telegram": ["web"], "acp": ["hermes-acp"], "webhook": ["hermes-webhook"]}

    def ha_on() -> dict:
        config = lcm._read_config(home)
        selections["telegram"] = config["platform_toolsets"]["telegram"]
        return {p: bool(_enabled_plugin_toolsets(config, p, sel, {"homeassistant"})) for p, sel in selections.items()}

    lcm.migrate_home(home, install=lambda n: pytest.fail("reinstalled"), say=print)
    assert ha_on() == {"telegram": False, "acp": False, "webhook": False}

    config = read_user_config_raw(home / "config.yaml")
    _save_platform_tools(config, "telegram", {"web", "homeassistant"})  # the user's later choice
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(home)
    try:
        save_config(config)
    finally:
        reset_hermes_home_override(token)
    lcm.migrate_home(home, install=lambda n: pytest.fail("reinstalled"), say=print)
    assert ha_on() == {"telegram": True, "acp": False, "webhook": False}


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



def test_the_migration_installs_once_so_a_removal_sticks(tmp_path, monkeypatch):
    """At most one automatic install per home: after `hermes plugins remove`, neither `hermes update`
    nor startup reinstalls it. Until an install succeeds (catalog miss, failed install) it retries."""
    import hermes_cli.memory_provider_migration as mpm
    import pm.install
    home = _home(tmp_path, env="HASS_TOKEN=abc\n")
    plugin = home / "plugins" / "homeassistant"
    calls = []

    def install(name):
        calls.append(name)
        plugin.mkdir(parents=True)
        return {"ok": True}

    quiet = lambda message: None
    monkeypatch.setattr(mpm, "catalog_source", lambda name: None)
    assert lcm.migrate_home(home, install=install, say=quiet) == []
    monkeypatch.setattr(mpm, "catalog_source", lambda name: name)
    assert lcm.migrate_home(home, install=lambda n: {"ok": False, "error": "network down"}, say=quiet) == []
    assert lcm.migrate_home(home, install=install, say=quiet) == ["homeassistant"]

    import shutil
    shutil.rmtree(plugin)  # `hermes plugins remove homeassistant`
    assert lcm.migrate_home(home, install=install, say=quiet) == []
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(pm.install, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr(lcm, "_install_into", lambda h: install)
    assert lcm.recover_at_startup(say=quiet) == []
    assert calls == ["homeassistant"] and not plugin.exists()


def test_a_failed_startup_install_backs_off_and_reports_one_line(tmp_path, monkeypatch):
    """With the catalog or git unreachable, only the first agent start in a while pays the network
    round trip, and its notice is one line naming the cause; `hermes update` always retries."""
    import hermes_cli.memory_provider_migration as mpm
    import pm.install
    home = _home(tmp_path, env="HASS_TOKEN=abc\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(pm.install, "lazy_installs_allowed", lambda: True)
    lookups = []
    monkeypatch.setattr(mpm, "catalog_source", lambda name: lookups.append(name) or name)
    git_error = ("Could not download the plugin from https://github.com/x/y. Check the address.\n"
                 "Details: Cloning into '/h/plugins/.install-1'...\n"
                 "fatal: unable to access 'https://github.com/x/y/': Could not resolve host: github.com\n")
    failing = lambda name: {"ok": False, "error": git_error}
    monkeypatch.setattr(lcm, "_install_into", lambda h: failing)
    said = []
    assert lcm.recover_at_startup(say=said.append) == []
    assert len(said) == 1 and "\n" not in said[0] and "Could not resolve host: github.com" in said[0]
    assert "plugins install homeassistant" in said[0]

    lcm._attempted.clear()  # the next `hermes chat` process
    assert lcm.recover_at_startup(say=said.append) == []
    assert lookups == ["homeassistant"] and len(said) == 1
    assert lcm.migrate_home(home, install=failing, say=said.append) == []
    assert lookups == ["homeassistant", "homeassistant"]


@pytest.mark.parametrize(("auth_json", "config", "in_use"), [
    ('{"providers": {"spotify": {"refresh_token": "r"}}}', "", True),
    ('{"providers": {"spotify": {"access_token": "a"}}}', "", True),
    ("", "platform_toolsets:\n  cli: [hermes-cli, spotify]\n", True),
    ('{"providers": {"spotify": {"client_id": "only-the-app"}}}', "", False),
    ('{"providers": {"nous": {"access_token": "a"}}}', "platform_toolsets:\n  cli: [hermes-cli]\n", False),
])
def test_spotify_in_use_is_a_stored_login_or_the_toolset(tmp_path, auth_json, config, in_use):
    """Core's Spotify tools were opt-in and login-gated: a login in auth.json or the toolset selected is
    use; an app client id alone (or another provider's login) is not."""
    home = _home(tmp_path, config=config)
    if auth_json:
        (home / "auth.json").write_text(auth_json, encoding="utf-8")
    assert lcm.spotify_in_use(home) is in_use


def test_old_spotify_commands_point_at_the_plugin(tmp_path, monkeypatch):
    """`hermes auth spotify` / `hermes spotify` name the new command and how to get it for this home."""
    home = _home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    hint = lcm.moved_command_hint("hermes auth", "spotify", "status")
    assert "`hermes auth status spotify` is now `hermes spotify status`" in hint
    assert "plugins install spotify" in hint
    assert "plugins install spotify" in lcm.moved_command_hint("hermes", "spotify")
    (home / "plugins" / "spotify").mkdir(parents=True)  # installed but disabled
    assert "plugins enable spotify" in lcm.moved_command_hint("hermes", "spotify")
    assert lcm.moved_command_hint("hermes", "honcho") == ""
