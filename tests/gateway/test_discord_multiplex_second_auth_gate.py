"""Regression for #80026: gateway second auth gate must not leak allowlists.

PR #75970 isolated Discord/Telegram adapter admission. The gateway's second
authorization gate still used a reader that fell through to process-global
``os.environ`` on a profile-scoped miss, so profile A's bridged
``DISCORD_ALLOWED_USERS`` could authorize (or block) traffic for profile B
and prevent B's adapter-local ``config.extra.allow_from`` from being
evaluated.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import secret_scope as ss
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


@pytest.fixture(autouse=True)
def _reset_scope_state(monkeypatch):
    for key in (
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


def _make_discord_runner(*, allow_from=None, profile="profile-b"):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    extra = {}
    if allow_from is not None:
        extra["allow_from"] = allow_from
    adapter = SimpleNamespace(
        send=AsyncMock(),
        config=SimpleNamespace(extra=extra),
        enforces_own_access_policy=False,
    )
    runner.adapters = {Platform.DISCORD: adapter}
    runner._profile_adapters = {profile: {Platform.DISCORD: adapter}}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_stores = {}
    return runner, adapter


def _discord_dm(user_id: str, *, profile="profile-b") -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        user_id=user_id,
        chat_id="test-dm",
        user_name=user_id,
        chat_type="dm",
        profile=profile,
    )


class TestDiscordMultiplexSecondAuthGate:
    def test_foreign_process_allowlist_does_not_authorize_secondary(
        self, monkeypatch
    ):
        # Profile A left DISCORD_ALLOWED_USERS=111 in process env. Profile B
        # has no env allowlist and only YAML allow_from=222.
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", "111")
        runner, _adapter = _make_discord_runner(allow_from="222")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({})  # profile B .env has no Discord keys
        try:
            assert runner._is_user_authorized(_discord_dm("111")) is False
            assert runner._is_user_authorized(_discord_dm("222")) is True
        finally:
            ss.reset_secret_scope(tok)

    def test_foreign_allow_all_does_not_open_secondary(self, monkeypatch):
        monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
        runner, _adapter = _make_discord_runner(allow_from="222")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({})
        try:
            assert runner._is_user_authorized(_discord_dm("111")) is False
            assert runner._is_user_authorized(_discord_dm("222")) is True
        finally:
            ss.reset_secret_scope(tok)

    def test_scoped_allowlist_still_authorizes(self, monkeypatch):
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", "111")
        runner, _adapter = _make_discord_runner(allow_from=None)
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"DISCORD_ALLOWED_USERS": "222"})
        try:
            assert runner._is_user_authorized(_discord_dm("111")) is False
            assert runner._is_user_authorized(_discord_dm("222")) is True
        finally:
            ss.reset_secret_scope(tok)

    def test_single_profile_environ_unchanged(self, monkeypatch):
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", "111")
        runner, _adapter = _make_discord_runner(allow_from="222")
        runner.config = GatewayConfig(multiplex_profiles=False)
        ss.set_multiplex_active(False)
        assert runner._is_user_authorized(_discord_dm("111", profile=None)) is True
        # Process env allowlist is authoritative; adapter allow_from is only
        # consulted when env-derived allowlists are empty.
        assert runner._is_user_authorized(_discord_dm("222", profile=None)) is False
