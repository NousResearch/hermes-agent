"""Config v52: agent-browser and the built-in browser tools left Hermes.

browser.backend is dropped (``off`` gets a one-time notice); AGENT_BROWSER_* from a profile's own
.env (plus the process env for the launch home only) moves to browser.* keys that are unset or hold the
default core let the env override.
"""

import hermes_yaml as yaml
import pytest

from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_AB_ENV = ("AGENT_BROWSER_EXECUTABLE_PATH", "AGENT_BROWSER_PROFILE", "AGENT_BROWSER_ARGS",
           "AGENT_BROWSER_HEADED", "AGENT_BROWSER_ENGINE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _AB_ENV:
        monkeypatch.delenv(name, raising=False)


def _home(path, config, env=""):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (path / ".env").write_text(env, encoding="utf-8")
    return path


def _raw(home):
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))


def _migrate(home, *, quiet=False):
    from hermes_cli.config import migrate_config

    token = set_hermes_home_override(home)
    try:
        return migrate_config(interactive=False, quiet=quiet)
    finally:
        reset_hermes_home_override(token)


def test_backend_off_is_dropped_with_a_notice(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    home = _home(tmp_path / "a", {"_config_version": 51, "browser": {"backend": "off", "inactivity_timeout": 60}})
    _migrate(home)
    out = capsys.readouterr().out
    assert _raw(home)["browser"] == {"inactivity_timeout": 60}
    assert _raw(home)["_config_version"] == DEFAULT_CONFIG["_config_version"]
    assert "browser_exec is the browser now" in out and "`browser` toolset in `hermes tools`" in out
    # Once: a second pass has nothing left to say.
    _migrate(home)
    assert "browser_exec" not in capsys.readouterr().out


def test_quiet_migration_hands_the_notice_to_the_caller(tmp_path, monkeypatch, capsys):
    """`hermes update` migrates quietly and re-prints only ``config_added``; the notice must be in it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    home = _home(tmp_path / "a", {"_config_version": 51, "browser": {"backend": "off"}})
    results = _migrate(home, quiet=True)
    assert capsys.readouterr().out == ""
    assert any("browser_exec is the browser now" in note and "`browser` toolset" in note
               for note in results["config_added"])


def _boot_agents(home, monkeypatch, starts: int) -> list[str]:
    """Boot maintenance for *home* (its config step), then *starts* agents in that process; the startup
    warnings they show."""
    import hermes_cli.left_core_migration as lcm
    from hermes_cli import post_update

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(lcm, "_attempted", set())
    post_update.step_migrate_config()
    said = []
    for _ in range(starts):
        lcm.recover_at_startup(say=said.append)
    return said


def test_boot_migration_shows_the_off_notice_at_the_first_agent_start_once(tmp_path, monkeypatch):
    """Boot migrates 51 -> 52 quietly, before any agent exists: the ``off`` notice reaches the first
    agent's startup warnings once, and neither a second agent nor the next boot repeats it."""
    import hermes_cli.left_core_migration as lcm

    monkeypatch.setattr(lcm, "_undelivered", {})
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    home = _home(tmp_path / "a", {"_config_version": 51, "browser": {"backend": "off"}})
    said = _boot_agents(home, monkeypatch, starts=2)
    assert _raw(home)["_config_version"] == DEFAULT_CONFIG["_config_version"]
    assert len(said) == 1
    assert "browser_exec is the browser now" in said[0] and "`browser` toolset in `hermes tools`" in said[0]
    assert _boot_agents(home, monkeypatch, starts=1) == []


@pytest.mark.parametrize(("value", "reason"), [
    ("camofox", "Camofox is now the camofox plugin (`hermes plugins install camofox`)"),
    ("local", "browser_exec launches Hermes's packaged Chromium itself"),
    ("browserbase", "a cloud browser is chosen with browser.cloud_provider"),
    ("Browser-Use", "browser_exec (Browser Use mode) is the only browser now"),
    ("agent-browser", "agent-browser left Hermes"),
    ("lynx", "browser_exec is the only browser now"),
])
def test_a_removed_backend_value_is_named_with_its_reason(tmp_path, monkeypatch, value, reason):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    home = _home(tmp_path / "a", {"_config_version": 51, "browser": {"backend": value}})
    results = _migrate(home, quiet=True)
    assert "backend" not in _raw(home)["browser"]
    note = next(note for note in results["config_added"] if "browser.backend" in note)
    assert note.startswith(f"removed browser.backend: {value} — ") and reason in note


@pytest.mark.parametrize("value", ["", "browser-use", "Browser-Use", None])
def test_other_backend_values_are_dropped_silently(tmp_path, monkeypatch, capsys, value):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    home = _home(tmp_path / "a", {"_config_version": 51, "browser": {"backend": value, "headed": True}})
    _migrate(home)
    assert _raw(home)["browser"] == {"headed": True}
    assert "browser_exec" not in capsys.readouterr().out


def test_unversioned_config_still_loses_backend(tmp_path, monkeypatch):
    """The installer seeds config.yaml without a stamp; a legacy key is legacy however old the file."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    home = _home(tmp_path / "a", {"browser": {"backend": "off"}})
    _migrate(home, quiet=True)
    assert "backend" not in (_raw(home).get("browser") or {})


def test_dotenv_settings_move_to_unset_keys_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    env = ("AGENT_BROWSER_EXECUTABLE_PATH=/opt/chrome/chrome\n"
           "AGENT_BROWSER_PROFILE=/data/profile\n"
           "AGENT_BROWSER_ARGS=--no-sandbox, --lang=de\n"
           "AGENT_BROWSER_HEADED=true\n"
           "AGENT_BROWSER_ENGINE=lightpanda\n")
    home = _home(tmp_path / "a", {"_config_version": 51, "browser": {"user_data_dir": "/mine"}}, env)
    results = _migrate(home, quiet=True)
    browser = _raw(home)["browser"]
    assert browser == {
        "executable_path": "/opt/chrome/chrome",
        "user_data_dir": "/mine",  # already set: the user's value wins
        "chromium_args": ["--no-sandbox", "--lang=de"],
        "headed": True,
        "engine": "lightpanda",
    }
    assert any("AGENT_BROWSER_ARGS → browser.chromium_args" in item for item in results["config_added"])
    # .env is left as it was.
    assert (home / ".env").read_text(encoding="utf-8") == env


def test_env_wins_over_the_defaults_core_wrote(tmp_path, monkeypatch):
    """Core resolved ``engine: auto`` to AGENT_BROWSER_ENGINE and ``headed`` as config OR env, and its
    `hermes tools` Local Browser row wrote ``engine: auto``: those values do not block the move. A
    non-default engine won on core and stays."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    env = "AGENT_BROWSER_ENGINE=lightpanda\nAGENT_BROWSER_HEADED=true\n"
    defaults = _home(tmp_path / "a", {"_config_version": 51, "browser": {"engine": "auto", "headed": False}}, env)
    chosen = _home(tmp_path / "b", {"_config_version": 51, "browser": {"engine": "chrome", "headed": "false"}}, env)
    _migrate(defaults, quiet=True)
    _migrate(chosen, quiet=True)
    assert _raw(defaults)["browser"] == {"engine": "lightpanda", "headed": True}
    assert _raw(chosen)["browser"] == {"engine": "chrome", "headed": True}


def test_unquoted_backend_off_gets_the_notice(tmp_path, monkeypatch, capsys):
    """YAML 1.1 reads an unquoted ``backend: off`` as False; core treated it as off."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    home = tmp_path / "a"
    home.mkdir()
    (home / "config.yaml").write_text("_config_version: 51\nbrowser:\n  backend: off\n", encoding="utf-8")
    results = _migrate(home)
    assert "backend" not in (_raw(home).get("browser") or {})
    assert "browser_exec is the browser now" in capsys.readouterr().out
    assert any("browser.backend: off" in note for note in results["config_added"])


def test_relative_profile_keeps_the_directory_core_used(tmp_path, monkeypatch):
    """agent-browser resolved a relative AGENT_BROWSER_PROFILE against Hermes's cwd; browser.user_data_dir
    anchors relative paths at HERMES_HOME, so the migration pins the absolute directory."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    home = _home(tmp_path / "a", {"_config_version": 51}, "AGENT_BROWSER_PROFILE=prof/rel\n")
    _migrate(home, quiet=True)
    assert _raw(home)["browser"]["user_data_dir"] == str(cwd / "prof" / "rel")


def test_a_pinned_relative_profile_names_the_absolute_path_on_update_and_boot(tmp_path, monkeypatch):
    """The relative AGENT_BROWSER_PROFILE was resolved against whatever directory the migrating process ran
    in: `hermes update` and the first agent after a boot migration both name the absolute path pinned."""
    import hermes_cli.left_core_migration as lcm

    monkeypatch.setattr(lcm, "_undelivered", {})
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    pinned = str(cwd / "prof" / "rel")
    updated = _home(tmp_path / "a", {"_config_version": 51}, "AGENT_BROWSER_PROFILE=prof/rel\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    results = _migrate(updated, quiet=True)
    assert any(pinned in note and "prof/rel" in note for note in results["config_added"])

    booted = _home(tmp_path / "b", {"_config_version": 51}, "AGENT_BROWSER_PROFILE=prof/rel\n")
    said = _boot_agents(booted, monkeypatch, starts=1)
    assert _raw(booted)["browser"]["user_data_dir"] == pinned
    assert len(said) == 1 and pinned in said[0] and "prof/rel" in said[0]


def test_falsy_headed_and_unknown_engine_carry_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    home = _home(tmp_path / "a", {"_config_version": 51},
                 "AGENT_BROWSER_HEADED=0\nAGENT_BROWSER_ENGINE=webkit\n")
    _migrate(home, quiet=True)
    assert not _raw(home).get("browser")


def test_pm_store_chromium_path_is_not_pinned(tmp_path, monkeypatch):
    """pm (and the Docker image) exported its own Chromium under AGENT_BROWSER_EXECUTABLE_PATH; that is the
    default now, and a copied versioned store path would break at the next Chromium update."""
    from pm import paths

    store = tmp_path / "tools"
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    pm_binary = paths.store_root() / "chromium-1208" / "chrome-linux64" / "chrome"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    home = _home(tmp_path / "a", {"_config_version": 51})
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", str(pm_binary))
    _migrate(home, quiet=True)
    assert "executable_path" not in (_raw(home).get("browser") or {})


def test_profiles_are_islands_a_b_a(tmp_path, monkeypatch, capsys):
    """Each home is decided by its own files; the process env counts for the launch home only."""
    a = _home(tmp_path / "a", {"_config_version": 51, "browser": {"backend": "off"}},
              "AGENT_BROWSER_EXECUTABLE_PATH=/opt/a/chrome\n")
    b = _home(tmp_path / "b", {"_config_version": 51, "browser": {"inactivity_timeout": 300}})
    monkeypatch.setenv("HERMES_HOME", str(a))  # A is the launch home
    monkeypatch.setenv("AGENT_BROWSER_HEADED", "1")  # e.g. a systemd unit for A

    _migrate(a)
    out_a = capsys.readouterr().out
    _migrate(b)
    out_b = capsys.readouterr().out
    _migrate(a)
    out_a2 = capsys.readouterr().out

    assert _raw(a)["browser"] == {"executable_path": "/opt/a/chrome", "headed": True}
    assert "browser_exec is the browser now" in out_a
    assert _raw(b)["browser"] == {"inactivity_timeout": 300}
    assert "browser_exec" not in out_b and "agent-browser" not in out_b
    assert "browser_exec" not in out_a2 and "agent-browser" not in out_a2


@pytest.mark.parametrize("stamp", [51, None])
def test_extension_control_block_is_dropped(tmp_path, monkeypatch, stamp):
    """Browser-extension control left with the built-in browser tools; nothing reads the key now."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    config = {"browser": {"extension_control": {"enabled": True, "developer_mode": False}, "headed": True}}
    if stamp is not None:
        config["_config_version"] = stamp
    home = _home(tmp_path / "a", config)
    results = _migrate(home, quiet=True)
    assert _raw(home)["browser"] == {"headed": True}
    assert "removed browser.extension_control" in results["config_added"]
