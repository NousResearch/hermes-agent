"""Tests for Telegram adapter fail-closed auth fallback (#24457).

The _is_callback_user_authorized fallback must deny users by default
when TELEGRAM_ALLOWED_USERS is empty, instead of allowing everyone.
"""

import sys
import types
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig, Platform


# -- Fake telegram modules (minimal stubs) --------------------------------

_fake_telegram_error = types.ModuleType("telegram.error")


class _TelegramError(Exception):
    pass


_fake_telegram_error.TelegramError = _TelegramError
_fake_telegram_error.BadRequest = type("BadRequest", (_TelegramError,), {})
_fake_telegram_error.NetworkError = type("NetworkError", (_TelegramError,), {})

_fake_telegram_constants = types.ModuleType("telegram.constants")
_fake_telegram_constants.ParseMode = SimpleNamespace(HTML="HTML")

_fake_telegram_request = types.ModuleType("telegram.request")
_fake_telegram_request.HTTPXRequest = type("HTTPXRequest", (), {"__init__": lambda *a, **kw: None})

_fake_telegram_ext = types.ModuleType("telegram.ext")
_fake_telegram_ext.ApplicationBuilder = type("ApplicationBuilder", (), {
    "token": lambda self, *a: self,
    "build": lambda self: None,
})

_fake_telegram = types.ModuleType("telegram")
_fake_telegram.error = _fake_telegram_error
_fake_telegram.constants = _fake_telegram_constants
_fake_telegram.ext = _fake_telegram_ext
_fake_telegram.request = _fake_telegram_request


@pytest.fixture(autouse=True)
def _inject_fake_telegram(monkeypatch):
    monkeypatch.setitem(sys.modules, "telegram", _fake_telegram)
    monkeypatch.setitem(sys.modules, "telegram.error", _fake_telegram_error)
    monkeypatch.setitem(sys.modules, "telegram.constants", _fake_telegram_constants)
    monkeypatch.setitem(sys.modules, "telegram.ext", _fake_telegram_ext)
    monkeypatch.setitem(sys.modules, "telegram.request", _fake_telegram_request)


def _make_adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
    adapter._config = config
    adapter._platform = Platform.TELEGRAM
    adapter._connected = True
    return adapter


class TestCallbackAuthFailClosed:
    """_is_callback_user_authorized fallback must be fail-closed."""

    def test_no_allowlist_no_allow_all_denies(self, monkeypatch):
        """No TELEGRAM_ALLOWED_USERS and no GATEWAY_ALLOW_ALL_USERS → deny."""
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        adapter = _make_adapter()
        # Force the fallback path (no runner auth)
        adapter._message_handler = None
        assert adapter._is_callback_user_authorized("12345") is False

    def test_no_allowlist_with_global_allow_all_permits(self, monkeypatch):
        """No TELEGRAM_ALLOWED_USERS but GATEWAY_ALLOW_ALL_USERS=true → allow."""
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = _make_adapter()
        adapter._message_handler = None
        assert adapter._is_callback_user_authorized("12345") is True

    def test_allowlist_with_matching_user_permits(self, monkeypatch):
        """TELEGRAM_ALLOWED_USERS contains the user → allow."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "12345,67890")
        adapter = _make_adapter()
        adapter._message_handler = None
        assert adapter._is_callback_user_authorized("12345") is True

    def test_allowlist_without_matching_user_denies(self, monkeypatch):
        """TELEGRAM_ALLOWED_USERS does not contain the user → deny."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "67890")
        adapter = _make_adapter()
        adapter._message_handler = None
        assert adapter._is_callback_user_authorized("12345") is False

    def test_allowlist_wildcard_permits(self, monkeypatch):
        """TELEGRAM_ALLOWED_USERS=* → allow everyone."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
        adapter = _make_adapter()
        adapter._message_handler = None
        assert adapter._is_callback_user_authorized("12345") is True


class TestCallbackAuthPrefersInjectedCheck:
    """_is_callback_user_authorized must use the auth callback GatewayRunner
    injects via set_authorization_check before the _message_handler.__self__
    introspection.

    A secondary multiplexed adapter's _message_handler is a profile closure
    (no __self__), so the introspection path resolves to nothing and the old
    code fell through to the env-only fallback — which knows nothing about
    profile config allowlists or the pairing store. The injected callback is
    registered for every gateway-connected adapter, including multiplexed
    secondaries, and delegates to the full _is_user_authorized chain.
    """

    def test_injected_check_used_when_handler_is_a_closure(self, monkeypatch):
        """Multiplexed shape: closure handler (no __self__) + injected check
        registered → the injected check decides, not the env fallback."""
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        adapter = _make_adapter()
        adapter._message_handler = lambda *a, **kw: None  # no __self__
        seen = {}

        def _check(user_id, chat_type=None, chat_id=None):
            seen.update(user_id=user_id, chat_type=chat_type, chat_id=chat_id)
            return user_id == "999"

        adapter._authorization_check = _check

        # Env fallback would deny (empty allowlist); the injected check allows.
        assert adapter._is_callback_user_authorized(
            "999", chat_id="777", chat_type="supergroup"
        ) is True
        assert seen == {"user_id": "999", "chat_type": "group", "chat_id": "777"}

    def test_injected_check_deny_wins_over_env_allowlist(self, monkeypatch):
        """The injected check is authoritative when registered — an env
        allowlist entry must not override its deny."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "12345")
        adapter = _make_adapter()
        adapter._message_handler = lambda *a, **kw: None
        adapter._authorization_check = lambda user_id, chat_type=None, chat_id=None: False

        assert adapter._is_callback_user_authorized("12345") is False

    def test_injected_check_error_falls_back_to_env(self, monkeypatch):
        """A raising injected check is treated as unregistered (base returns
        None), preserving the env fallback rather than crashing the button."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "12345")
        adapter = _make_adapter()
        adapter.platform = Platform.TELEGRAM  # for _is_sender_authorized's log line
        adapter._message_handler = None

        def _boom(user_id, chat_type=None, chat_id=None):
            raise RuntimeError("auth backend down")

        adapter._authorization_check = _boom

        assert adapter._is_callback_user_authorized("12345") is True
        assert adapter._is_callback_user_authorized("67890") is False
