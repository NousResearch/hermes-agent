"""Regression: _auth_env must not inherit process env under multiplex.

When gateway.multiplex_profiles is on, each turn's secret_scope is authoritative.
A profile that omits allowlist / allow-all keys must not pick up those values
from os.environ (which may belong to another profile in the same process).
"""

from types import SimpleNamespace

import pytest

import agent.secret_scope as secret_scope
from gateway.authz_mixin import _auth_env
from gateway.config import Platform
from gateway.session import SessionSource


@pytest.fixture(autouse=True)
def _reset_multiplex():
    secret_scope.set_multiplex_active(False)
    yield
    secret_scope.set_multiplex_active(False)


@pytest.fixture
def clear_auth_env(monkeypatch):
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "TELEGRAM_ALLOW_BOTS",
        "DISCORD_ALLOW_BOTS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    runner.adapters = {}
    runner._profile_adapters = {}
    runner.config = None
    return runner


def _dm_source(user_id: str) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="dm-1",
        chat_type="dm",
        user_id=user_id,
    )


def test_auth_env_ignores_process_allowlist_under_multiplex(clear_auth_env, monkeypatch):
    """Scope omits TELEGRAM_ALLOWED_USERS; process env must not leak through."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "env-allowed")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"TELEGRAM_BOT_TOKEN": "x"})
    try:
        assert _auth_env("TELEGRAM_ALLOWED_USERS") == ""
    finally:
        secret_scope.reset_secret_scope(token)


def test_auth_env_ignores_process_allow_all_under_multiplex(clear_auth_env, monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"TELEGRAM_BOT_TOKEN": "x"})
    try:
        assert _auth_env("GATEWAY_ALLOW_ALL_USERS") == ""
    finally:
        secret_scope.reset_secret_scope(token)


def test_is_user_authorized_denies_foreign_process_allowlist(clear_auth_env, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "env-allowed")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"TELEGRAM_BOT_TOKEN": "x"})
    try:
        runner = _make_runner()
        assert runner._is_user_authorized(_dm_source("env-allowed")) is False
    finally:
        secret_scope.reset_secret_scope(token)


def test_is_user_authorized_denies_foreign_process_allow_all(clear_auth_env, monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"TELEGRAM_BOT_TOKEN": "x"})
    try:
        runner = _make_runner()
        assert runner._is_user_authorized(_dm_source("stranger")) is False
    finally:
        secret_scope.reset_secret_scope(token)


def test_auth_env_honors_scoped_allowlist_under_multiplex(clear_auth_env):
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"TELEGRAM_ALLOWED_USERS": "scoped-user"})
    try:
        assert _auth_env("TELEGRAM_ALLOWED_USERS") == "scoped-user"
        runner = _make_runner()
        assert runner._is_user_authorized(_dm_source("scoped-user")) is True
        assert runner._is_user_authorized(_dm_source("other")) is False
    finally:
        secret_scope.reset_secret_scope(token)


def test_auth_env_still_reads_process_env_when_multiplex_off(clear_auth_env, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "env-allowed")
    secret_scope.set_multiplex_active(False)
    assert _auth_env("TELEGRAM_ALLOWED_USERS") == "env-allowed"


def test_auth_env_returns_default_when_multiplex_unscoped(clear_auth_env, monkeypatch):
    """No secret scope under multiplex must not fall through to os.environ."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "env-allowed")
    secret_scope.set_multiplex_active(True)
    assert _auth_env("TELEGRAM_ALLOWED_USERS") == ""


def test_auth_env_empty_scoped_value_does_not_fall_through(clear_auth_env, monkeypatch):
    """Explicit empty string in scope must not inherit process env under multiplex."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "env-allowed")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"TELEGRAM_ALLOWED_USERS": ""})
    try:
        assert _auth_env("TELEGRAM_ALLOWED_USERS") == ""
    finally:
        secret_scope.reset_secret_scope(token)


def test_group_chat_allowlist_ignores_process_env_under_multiplex(clear_auth_env, monkeypatch):
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-100A")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"TELEGRAM_BOT_TOKEN": "x"})
    try:
        runner = _make_runner()
        src = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100A",
            chat_type="channel",
            user_id=None,
        )
        assert runner._is_user_authorized(src) is False
    finally:
        secret_scope.reset_secret_scope(token)


def test_group_chat_allowlist_honors_profile_scope(clear_auth_env):
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"TELEGRAM_GROUP_ALLOWED_CHATS": "-100B"})
    try:
        runner = _make_runner()
        src = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100B",
            chat_type="channel",
            user_id=None,
        )
        assert runner._is_user_authorized(src) is True
    finally:
        secret_scope.reset_secret_scope(token)


def test_allow_bots_ignores_process_env_under_multiplex(clear_auth_env, monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"DISCORD_BOT_TOKEN": "x"})
    try:
        runner = _make_runner()
        src = SessionSource(
            platform=Platform.DISCORD,
            chat_id="dm-1",
            chat_type="dm",
            user_id="bot-1",
            is_bot=True,
        )
        assert runner._is_user_authorized(src) is False
    finally:
        secret_scope.reset_secret_scope(token)


def test_unauthorized_dm_behavior_ignores_process_allowlist_under_multiplex(
    clear_auth_env, monkeypatch
):
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "env-owner")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"TELEGRAM_BOT_TOKEN": "x"})
    try:
        runner = _make_runner()
        assert runner._get_unauthorized_dm_behavior(Platform.TELEGRAM) == "pair"
    finally:
        secret_scope.reset_secret_scope(token)


def test_unauthorized_dm_behavior_honors_scoped_allowlist(clear_auth_env):
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"GATEWAY_ALLOWED_USERS": "owner1"})
    try:
        runner = _make_runner()
        assert runner._get_unauthorized_dm_behavior(Platform.TELEGRAM) == "ignore"
    finally:
        secret_scope.reset_secret_scope(token)
