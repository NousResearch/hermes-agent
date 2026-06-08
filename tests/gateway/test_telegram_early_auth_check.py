"""Tests for Telegram adapter early auth check (#40863).

The _is_user_auth_early method must reject messages from unauthorized
users at the adapter level, before text batching or event construction
can leak prompt content into the agent context.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
    from gateway.platforms.telegram import TelegramAdapter

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
    adapter._config = config
    adapter._platform = Platform.TELEGRAM
    adapter._connected = True
    return adapter


def _make_message(user_id="12345", username="testuser", chat_id=100, chat_type="private"):
    """Create a minimal mock Telegram Message."""
    user = SimpleNamespace(id=int(user_id), username=username, first_name=username)
    chat = SimpleNamespace(id=int(chat_id), type=chat_type)
    msg = SimpleNamespace(
        from_user=user,
        chat=chat,
        message_thread_id=None,
        text="hello",
    )
    return msg


def _make_message_no_user():
    """Create a message with no from_user (service message)."""
    chat = SimpleNamespace(id=100, type="private")
    return SimpleNamespace(from_user=None, chat=chat, message_thread_id=None, text="")


class TestEarlyAuthCheck:
    """_is_user_auth_early must reject unauthorized users before event construction."""

    def test_no_user_defers(self):
        """Service messages with no from_user → defer to cold path."""
        adapter = _make_adapter()
        adapter._message_handler = None
        msg = _make_message_no_user()
        assert adapter._is_user_auth_early(msg) is True

    def test_no_runner_no_env_fails_closed(self, monkeypatch):
        """No runner + no TELEGRAM_ALLOWED_USERS + no GATEWAY_ALLOW_ALL_USERS → deny (fail-closed)."""
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        adapter = _make_adapter()
        adapter._message_handler = None
        msg = _make_message()
        assert adapter._is_user_auth_early(msg) is False

    def test_no_runner_with_allow_all_permits(self, monkeypatch):
        """No runner but GATEWAY_ALLOW_ALL_USERS=true → allow."""
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = _make_adapter()
        adapter._message_handler = None
        msg = _make_message()
        assert adapter._is_user_auth_early(msg) is True

    def test_no_runner_user_in_allowlist_permits(self, monkeypatch):
        """No runner but user in TELEGRAM_ALLOWED_USERS → allow."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "12345,67890")
        adapter = _make_adapter()
        adapter._message_handler = None
        msg = _make_message(user_id="12345")
        assert adapter._is_user_auth_early(msg) is True

    def test_no_runner_user_not_in_allowlist_denies(self, monkeypatch):
        """No runner and user NOT in TELEGRAM_ALLOWED_USERS → deny."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "67890")
        adapter = _make_adapter()
        adapter._message_handler = None
        msg = _make_message(user_id="12345")
        assert adapter._is_user_auth_early(msg) is False

    def test_runner_authorized_user_passes(self, monkeypatch):
        """Runner says user is authorized → allowed."""
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        adapter = _make_adapter()

        class MockRunner:
            def _is_user_authorized(self, source):
                return source.user_id == "99999"

        handler = lambda self_ref, *a, **kw: None
        handler.__self__ = MockRunner()
        adapter._message_handler = handler

        msg = _make_message(user_id="99999")
        assert adapter._is_user_auth_early(msg) is True

    def test_runner_unauthorized_user_rejected(self, monkeypatch):
        """Runner says user is unauthorized → denied (no env fallback)."""
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        adapter = _make_adapter()

        class MockRunner:
            def _is_user_authorized(self, source):
                return source.user_id == "99999"

        handler = lambda self_ref, *a, **kw: None
        handler.__self__ = MockRunner()
        adapter._message_handler = handler

        msg = _make_message(user_id="12345")
        assert adapter._is_user_auth_early(msg) is False

    def test_runner_auth_exception_falls_back_to_env(self, monkeypatch):
        """Exception in runner auth → falls back to env-based check."""
        adapter = _make_adapter()

        class BadRunner:
            def _is_user_authorized(self, source):
                raise RuntimeError("boom")

        handler = lambda self_ref, *a, **kw: None
        handler.__self__ = BadRunner()
        adapter._message_handler = handler

        # With user in allowlist → permit via env fallback
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "12345")
        msg = _make_message(user_id="12345")
        assert adapter._is_user_auth_early(msg) is True

        # Without allowlist → deny via env fallback
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        assert adapter._is_user_auth_early(msg) is False

    def test_wildcard_permits(self, monkeypatch):
        """TELEGRAM_ALLOWED_USERS=* → allow everyone."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
        adapter = _make_adapter()
        adapter._message_handler = None
        msg = _make_message(user_id="99999")
        assert adapter._is_user_auth_early(msg) is True

class TestCallbackQueryAuthIntegration:
    """Callback queries from unauthorized users must be rejected.

    The _handle_callback_query handler has auth checks for approval
    buttons (ea:*) but model picker (mp:*) and gmail triage (gt:*)
    callbacks lack per-user authorization at the handler level.
    These tests document the current behavior.
    """

    @staticmethod
    def _make_callback_update(user_id="99999", data="mp:test", chat_id=123):
        """Build a minimal callback query update."""
        from types import SimpleNamespace

        user = SimpleNamespace(id=int(user_id), first_name="Test", username="testuser")
        chat = SimpleNamespace(id=chat_id, type="private")
        msg = SimpleNamespace(chat_id=chat_id, chat=chat, message_thread_id=None)
        query = SimpleNamespace(
            data=data, from_user=user, message=msg,
            answer=AsyncMock(), id="cb123",
        )
        update = SimpleNamespace(callback_query=query, message=None)
        return update

    def test_approval_callback_rejected_for_unauthorized(self, monkeypatch):
        """Approval button callbacks (ea:*) must be auth-gated."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "11111")
        adapter = _make_adapter()
        adapter._approval_state = {42: "session-key"}
        adapter._message_handler = None

        # The _is_callback_user_authorized check at line 3280 should deny
        assert adapter._is_callback_user_authorized("99999") is False

    def test_approval_callback_allowed_for_authorized(self, monkeypatch):
        """Approval button callbacks permit authorized users."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "99999")
        adapter = _make_adapter()
        adapter._message_handler = None

        assert adapter._is_callback_user_authorized("99999") is True

    def test_early_auth_check_passes_for_channel_post(self, monkeypatch):
        """Channel posts (no from_user) must pass the early auth gate."""
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        adapter = _make_adapter()

        msg = SimpleNamespace(from_user=None, text="broadcast")
        # Channel posts have no from_user → _is_user_auth_early returns True
        assert adapter._is_user_auth_early(msg) is True

    def test_early_auth_check_rejects_unauthorized_user(self, monkeypatch):
        """Regular messages from unauthorized users must be rejected early."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "11111")
        adapter = _make_adapter()

        user = SimpleNamespace(id=99999, first_name="Bad", username="bad")
        msg = SimpleNamespace(from_user=user, text="inject this")
        assert adapter._is_user_auth_early(msg) is False

    def test_early_auth_check_allows_authorized_user(self, monkeypatch):
        """Regular messages from authorized users must pass."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "99999")
        adapter = _make_adapter()

        user = SimpleNamespace(id=99999, first_name="Good", username="good")
        msg = SimpleNamespace(from_user=user, text="hello")
        assert adapter._is_user_auth_early(msg) is True
