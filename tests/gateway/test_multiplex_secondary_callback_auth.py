"""RED test for #120639: secondary-profile inline-button callbacks are refused
because the adapter auth check never enters the owning profile's runtime scope.
"""

import pytest

from agent import secret_scope
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore


@pytest.fixture
def mux_home(tmp_path, monkeypatch):
    home = tmp_path / "hh"
    (home / "profiles" / "klubizz").mkdir(parents=True)
    (home / ".env").write_text("")
    # The secondary profile owns its own bot: allowlist lives ONLY in its .env.
    (home / "profiles" / "klubizz" / ".env").write_text("TELEGRAM_ALLOWED_USERS=4242\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_ALLOW_BOTS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(key, raising=False)
    prev = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    yield home
    secret_scope.set_multiplex_active(prev)


def _runner(home):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.platforms = {
        Platform.TELEGRAM: PlatformConfig(enabled=True, extra={})
    }
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {
        "default": runner.pairing_store,
        "klubizz": PairingStore(profile="klubizz"),
    }
    runner._primary_profile_name = "default"
    runner._profile_adapters = {"klubizz": {}}
    return runner


def _telegram(runner, home):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    tg = object.__new__(TelegramAdapter)
    tg.config = PlatformConfig(enabled=True, extra={})
    tg._authorization_check = None
    runner._profile_adapters["klubizz"][Platform.TELEGRAM] = tg
    tg.set_authorization_check(
        runner._make_adapter_auth_check(Platform.TELEGRAM, profile_name="klubizz")
    )
    return tg


def test_secondary_own_bot_callback_reads_own_allowlist(mux_home):
    """A secondary profile with its own bot token authorizes an inline-button
    caller against ITS allowlist, not the default profile's env."""
    runner = _runner(mux_home)
    tg = _telegram(runner, mux_home)

    # allowlisted in the secondary's .env; NOT in the default profile's env
    assert (
        tg._is_callback_user_authorized("4242", chat_id="4242", chat_type="dm") is True
    )
    assert (
        tg._is_callback_user_authorized("8888", chat_id="8888", chat_type="dm") is False
    )
