"""The dotenv loader must resolve its default home the way everyone else does.

``load_hermes_dotenv`` fell back to a bare
``os.getenv("HERMES_HOME", Path.home() / ".hermes")``, which hard-codes the
POSIX layout.  ``get_hermes_home()`` / ``get_env_path()`` — and therefore
``hermes status`` and ``hermes doctor`` — resolve the platform-native default
instead (``%LOCALAPPDATA%\\hermes`` on Windows).

``hermes_cli/main.py`` runs the startup load without passing ``hermes_home=``,
and ``HERMES_HOME`` is only exported into the environment when a profile is
actually resolved.  So on Windows, with no profile and no ``HERMES_HOME``, the
process loaded one ``.env`` while both reporting commands described another.

These tests pin the delegation rather than the Windows path, so they assert the
same invariant on every platform: with ``HERMES_HOME`` unset, the loader reads
whatever ``hermes_constants`` calls the platform default.
"""

import os

import hermes_constants
from hermes_cli.env_loader import load_hermes_dotenv


def _write_env(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _pin_platform_default(monkeypatch, home):
    """Make the platform-native default resolve to ``home`` on any OS."""
    monkeypatch.setattr(
        hermes_constants, "_get_platform_default_hermes_home", lambda: home
    )


def test_loader_follows_platform_default_home(tmp_path, monkeypatch):
    """The regression: the startup load must read the file status/doctor report."""
    home = tmp_path / "platform-home"
    _write_env(home / ".env", "PLATFORM_DEFAULT_KEY=from-platform-home\n")
    monkeypatch.delenv("HERMES_HOME", raising=False)
    _pin_platform_default(monkeypatch, home)
    monkeypatch.delenv("PLATFORM_DEFAULT_KEY", raising=False)

    loaded = load_hermes_dotenv(load_external_secrets=False)

    assert loaded == [home / ".env"]
    assert os.environ["PLATFORM_DEFAULT_KEY"] == "from-platform-home"


def test_hermes_home_env_var_still_wins_over_platform_default(tmp_path, monkeypatch):
    """Behavior preserved: an explicit HERMES_HOME still beats the default."""
    platform_home = tmp_path / "platform-home"
    _write_env(platform_home / ".env", "WRONG_HOME_KEY=from-platform-home\n")
    env_home = tmp_path / "env-home"
    _write_env(env_home / ".env", "RIGHT_HOME_KEY=from-env-home\n")
    _pin_platform_default(monkeypatch, platform_home)
    monkeypatch.setenv("HERMES_HOME", str(env_home))
    monkeypatch.delenv("WRONG_HOME_KEY", raising=False)
    monkeypatch.delenv("RIGHT_HOME_KEY", raising=False)

    loaded = load_hermes_dotenv(load_external_secrets=False)

    assert loaded == [env_home / ".env"]
    assert os.environ["RIGHT_HOME_KEY"] == "from-env-home"
    assert "WRONG_HOME_KEY" not in os.environ


def test_explicit_hermes_home_argument_still_wins(tmp_path, monkeypatch):
    """Behavior preserved: the keyword argument beats env var and default."""
    platform_home = tmp_path / "platform-home"
    _write_env(platform_home / ".env", "WRONG_HOME_KEY=from-platform-home\n")
    explicit_home = tmp_path / "explicit-home"
    _write_env(explicit_home / ".env", "EXPLICIT_KEY=from-explicit-home\n")
    monkeypatch.delenv("HERMES_HOME", raising=False)
    _pin_platform_default(monkeypatch, platform_home)
    monkeypatch.delenv("WRONG_HOME_KEY", raising=False)
    monkeypatch.delenv("EXPLICIT_KEY", raising=False)

    loaded = load_hermes_dotenv(
        hermes_home=explicit_home, load_external_secrets=False
    )

    assert loaded == [explicit_home / ".env"]
    assert os.environ["EXPLICIT_KEY"] == "from-explicit-home"
    assert "WRONG_HOME_KEY" not in os.environ
