"""Signal authorization helpers honor per-profile secret scope (#93522)."""

import contextlib

import pytest

from agent.secret_scope import reset_secret_scope, set_secret_scope
from plugins.platforms.signal.adapter import _sig_secret


@contextlib.contextmanager
def _scope(secrets):
    token = set_secret_scope(secrets)
    try:
        yield
    finally:
        reset_secret_scope(token)


def test_helper_reads_profile_scope_first(monkeypatch):
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "FROM-DEFAULT-ENV")
    with _scope({"SIGNAL_ALLOWED_USERS": "from-profile-scope"}):
        assert _sig_secret("SIGNAL_ALLOWED_USERS", "") == "from-profile-scope"


@pytest.fixture
def multiplex_on(monkeypatch):
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)


def test_helper_does_not_leak_default_env_into_scoped_miss(multiplex_on, monkeypatch):
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "*")
    with _scope({}):
        assert _sig_secret("SIGNAL_ALLOWED_USERS", "restricted-default") == "restricted-default"


def test_helper_falls_back_to_environ_without_scope(monkeypatch):
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "legacy-env-value")
    assert _sig_secret("SIGNAL_ALLOWED_USERS", "") == "legacy-env-value"
