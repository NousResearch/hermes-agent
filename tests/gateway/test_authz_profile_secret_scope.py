"""Regression tests: authz_mixin.py allowlist checks must honor profile-scoped
secrets (via ``agent.secret_scope``), not just ``os.environ``.

Three call sites in ``GatewayAuthorizationMixin`` read allowlist env vars with
raw ``os.getenv()`` instead of the module's own ``_auth_env()`` helper, which
checks the active profile secret scope first. Under multiplexing, an operator
who sets an allowlist via a profile's ``secret_scope`` (rather than the
process environment) had that allowlist silently ignored at exactly these
three spots, while every other allowlist check in the same class correctly
honored it. Fixed by routing all three through ``_auth_env()``.
"""

from types import SimpleNamespace

import pytest

import agent.secret_scope as secret_scope
from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in (
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "TELEGRAM_ALLOW_BOTS",
        "TELEGRAM_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def scoped_secrets():
    """Install a profile secret scope, auto-reset after the test."""

    def _install(secrets):
        secret_scope.set_multiplex_active(True)
        token = secret_scope.set_secret_scope(secrets)
        return token

    installed = []

    def _install_and_track(secrets):
        token = _install(secrets)
        installed.append(token)
        return token

    yield _install_and_track

    for token in installed:
        secret_scope.reset_secret_scope(token)
    secret_scope.set_multiplex_active(False)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _group_source(chat_id="grp1", chat_type="group"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=None,
        user_name=None,
        is_bot=False,
    )


def _bot_source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="dm1",
        chat_type="dm",
        user_id="bot-1",
        user_name="SomeBot",
        is_bot=True,
    )


def test_group_chat_allowlist_honors_profile_secret_scope(scoped_secrets):
    """TELEGRAM_GROUP_ALLOWED_CHATS set only via secret_scope must still authorize."""
    runner = _make_runner()
    scoped_secrets({"TELEGRAM_GROUP_ALLOWED_CHATS": "grp1"})

    assert runner._is_user_authorized(_group_source("grp1")) is True


def test_group_chat_allowlist_denies_when_not_in_profile_scope(scoped_secrets):
    """Same var, but the chat id isn't listed — must still deny (not fail open)."""
    runner = _make_runner()
    scoped_secrets({"TELEGRAM_GROUP_ALLOWED_CHATS": "some-other-chat"})

    assert runner._is_user_authorized(_group_source("grp1")) is False


def test_allow_bots_honors_profile_secret_scope(scoped_secrets):
    """TELEGRAM_ALLOW_BOTS set only via secret_scope must still admit bots."""
    runner = _make_runner()
    scoped_secrets({"TELEGRAM_ALLOW_BOTS": "all"})

    assert runner._is_user_authorized(_bot_source()) is True


def test_unauthorized_dm_behavior_honors_profile_secret_scope(scoped_secrets):
    """A profile-scoped GATEWAY_ALLOWED_USERS must switch the default from
    'pair' to 'ignore' — sending pairing codes to unknown senders when an
    allowlist IS configured (just not in os.environ) is exactly the info-leak
    the default is designed to avoid (see docstring on _get_unauthorized_dm_behavior)."""
    runner = _make_runner()
    runner.config = None
    scoped_secrets({"GATEWAY_ALLOWED_USERS": "owner1"})

    assert runner._get_unauthorized_dm_behavior(Platform.TELEGRAM) == "ignore"


def test_unauthorized_dm_behavior_defaults_to_pair_with_no_allowlist_anywhere(scoped_secrets):
    """Sanity check: with no allowlist configured (scope or env), default stays 'pair'."""
    runner = _make_runner()
    runner.config = None
    scoped_secrets({})

    assert runner._get_unauthorized_dm_behavior(Platform.TELEGRAM) == "pair"
