"""get_env_value must be scope-aware — the last scope-blind reader (#67027).

Under a multiplexed profile turn, ``os.environ`` can hold another profile's
value. ``get_env_value`` previously returned it before any scope check;
its siblings (``get_env_value_prefer_dotenv``, ``gateway.config._getenv``)
were already scope-aware. Salvaged premise from PR #67065 (@webtecnica),
reimplemented to delegate policy fully to ``agent.secret_scope.get_secret``
(the PR's own diff fell through to ``os.environ`` on a scoped miss,
re-opening the leak it targeted).
"""

import contextlib

import pytest

from agent.secret_scope import (
    UnscopedSecretError,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_cli.config import get_env_value


@contextlib.contextmanager
def _scope(secrets, *, multiplex: bool):
    set_multiplex_active(multiplex)
    token = set_secret_scope(secrets) if secrets is not None else None
    try:
        yield
    finally:
        if token is not None:
            reset_secret_scope(token)
        set_multiplex_active(False)


def test_scoped_value_wins_over_environ(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-other-profile")
    with _scope({"OPENAI_API_KEY": "sk-this-profile"}, multiplex=True):
        assert get_env_value("OPENAI_API_KEY") == "sk-this-profile"


def test_multiplexed_scope_miss_does_not_leak_environ(monkeypatch, tmp_path):
    """The #67027 repro: envless named profile must NOT inherit the other
    profile's credential from process environ during a multiplexed turn."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-other-profile")
    with _scope({}, multiplex=True):  # profile has no .env / empty scope
        assert get_env_value("OPENAI_API_KEY") is None


def test_multiplex_off_scope_miss_falls_back_to_environ(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")
    with _scope({}, multiplex=False):
        assert get_env_value("OPENAI_API_KEY") == "sk-from-shell"


def test_no_scope_single_profile_behaves_like_legacy(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-legacy")
    with _scope(None, multiplex=False):
        assert get_env_value("OPENAI_API_KEY") == "sk-legacy"


def test_multiplex_active_unscoped_read_fails_closed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    with _scope(None, multiplex=True):
        # No scope installed — must raise, not silently serve environ.
        with pytest.raises(UnscopedSecretError):
            get_env_value("OPENAI_API_KEY")


def test_dotenv_fallback_still_works(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    (home / ".env").write_text("MY_TEST_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("MY_TEST_KEY", raising=False)
    with _scope(None, multiplex=False):
        assert get_env_value("MY_TEST_KEY") == "from-dotenv"
