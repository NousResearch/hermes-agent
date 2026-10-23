"""AGENT_BROWSER_* settings survive homes that never run the versioned v52 step.

The runtime reads only ``browser.*`` from config.yaml now; the v52 step moves a home's legacy
AGENT_BROWSER_* .env settings there. A home whose config.yaml is missing (reads as current), never
stamped (boot used to skip it as below the support floor) or first stamped by an ordinary save
never reaches that step, so the move happens at its first stamp instead, exactly once.
"""

import hermes_yaml as yaml
import pytest

from hermes_cli.config_defaults import DEFAULT_CONFIG

_AB_ENV = ("AGENT_BROWSER_EXECUTABLE_PATH", "AGENT_BROWSER_PROFILE", "AGENT_BROWSER_ARGS",
           "AGENT_BROWSER_HEADED", "AGENT_BROWSER_ENGINE")
_LEGACY = ("AGENT_BROWSER_ENGINE=lightpanda\nAGENT_BROWSER_PROFILE=/probe/chrome-profile\n"
           "AGENT_BROWSER_HEADED=true\n")
_KEPT = {"engine": "lightpanda", "headed": True, "user_data_dir": "/probe/chrome-profile"}
_LATEST = DEFAULT_CONFIG["_config_version"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _AB_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CAMOFOX_URL", raising=False)


def _home(tmp_path, monkeypatch, config_text):
    """A real HERMES_HOME holding the legacy .env and *config_text* (None: no config.yaml)."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text(_LEGACY, encoding="utf-8")
    if config_text is not None:
        (home / "config.yaml").write_text(config_text, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _boot():
    from hermes_cli import post_update
    return post_update.step_migrate_config()


def _first_save():
    from hermes_cli.config import load_config, save_config
    save_config(load_config())


def _runtime():
    """What the browser launcher resolves for the active home right now."""
    from tools import browser_session_cloud as session
    session.reset_caches()
    return {"engine": session.get_browser_engine(), "headed": session.is_headed_mode(),
            "user_data_dir": session.user_data_dir()}


def _stamp(home):
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")).get("_config_version")


_CONFIGS = {"missing": None, "unstamped": "{}\n", "v51": "_config_version: 51\n"}


@pytest.mark.parametrize("case", list(_CONFIGS))
def test_boot_keeps_the_legacy_browser_settings(tmp_path, monkeypatch, case):
    home = _home(tmp_path, monkeypatch, _CONFIGS[case])
    assert _boot()["ok"]
    assert _runtime() == _KEPT
    assert _stamp(home) == _LATEST


@pytest.mark.parametrize("case", ["missing", "unstamped"])
def test_a_save_before_any_migration_keeps_the_legacy_browser_settings(tmp_path, monkeypatch, case):
    """The first save stamps the current version, so no config step will ever see this home."""
    home = _home(tmp_path, monkeypatch, _CONFIGS[case])
    _first_save()
    assert _stamp(home) == _LATEST
    _boot()
    assert _runtime() == _KEPT


def test_explicit_migration_recovers_a_home_without_config(tmp_path, monkeypatch):
    from hermes_cli.config import migrate_config

    home = _home(tmp_path, monkeypatch, None)
    migrate_config(interactive=False, quiet=True)
    assert _runtime() == _KEPT
    assert _stamp(home) == _LATEST


def test_the_retained_env_is_converted_once_never_over_later_edits(tmp_path, monkeypatch):
    """.env keeps the AGENT_BROWSER_* lines; the user's later choice must still win for good, even the
    defaults (``engine: auto``, ``headed: false``) that the v52 step lets the env override."""
    from hermes_cli.config import load_config, migrate_config, save_config

    _home(tmp_path, monkeypatch, None)
    _boot()
    config = load_config()
    config["browser"].update(engine="auto", headed=False, user_data_dir="/mine")
    save_config(config)
    _boot()
    migrate_config(interactive=False, quiet=True)
    _first_save()
    assert _runtime() == {"engine": "auto", "headed": False, "user_data_dir": "/mine"}


@pytest.mark.parametrize("case", ["missing", "unstamped"])
def test_boot_leaves_a_home_without_legacy_settings_as_it_was(tmp_path, monkeypatch, case):
    home = _home(tmp_path, monkeypatch, _CONFIGS[case])
    (home / ".env").write_text("", encoding="utf-8")
    assert _boot()["ok"]
    if _CONFIGS[case] is None:
        assert not (home / "config.yaml").exists()
    else:
        assert (home / "config.yaml").read_text(encoding="utf-8") == _CONFIGS[case]
