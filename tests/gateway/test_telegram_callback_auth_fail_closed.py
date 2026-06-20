"""Tests for Telegram adapter fail-closed auth fallback (#24457).

The _is_callback_user_authorized fallback must deny users by default
when TELEGRAM_ALLOWED_USERS is empty, instead of allowing everyone.
"""

import asyncio
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
    from gateway.platforms.telegram import TelegramAdapter

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

    def test_credential_callback_approves_broker_request(self, monkeypatch, tmp_path):
        """cred:a callbacks should invoke the profile-local secure credential broker."""
        adapter = _make_adapter()
        adapter._is_callback_user_authorized = lambda *a, **kw: True
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr="")

        monkeypatch.setattr("subprocess.run", fake_run)

        class Query:
            data = "cred:a:cr_test123:654321"
            from_user = SimpleNamespace(id="12345", first_name="Stavros")
            message = SimpleNamespace(
                chat_id=42,
                chat=SimpleNamespace(type="private"),
                message_thread_id=None,
            )

            def __init__(self):
                self.answers = []
                self.edits = []

            async def answer(self, **kwargs):
                self.answers.append(kwargs)

            async def edit_message_text(self, **kwargs):
                self.edits.append(kwargs)

        query = Query()
        update = SimpleNamespace(callback_query=query)

        asyncio.run(adapter._handle_callback_query(update, SimpleNamespace()))

        assert calls
        assert calls[0][0][1] == str(tmp_path / "scripts" / "secure_credential_broker.py")
        assert calls[0][0][-3:] == ["approve", "cr_test123", "654321"]
        assert query.answers == [{"text": "✅ Credential approved"}]
        assert query.edits[0]["reply_markup"] is None

    def test_credential_callback_rejects_unauthorized_user(self, monkeypatch):
        """Unauthorized cred callbacks must not invoke the broker."""
        adapter = _make_adapter()
        adapter._is_callback_user_authorized = lambda *a, **kw: False

        calls = []
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: calls.append((a, kw)))

        class Query:
            data = "cred:a:cr_test123:654321"
            from_user = SimpleNamespace(id="99999", first_name="Mallory")
            message = SimpleNamespace(
                chat_id=42,
                chat=SimpleNamespace(type="private"),
                message_thread_id=None,
            )

            def __init__(self):
                self.answers = []

            async def answer(self, **kwargs):
                self.answers.append(kwargs)

        query = Query()
        update = SimpleNamespace(callback_query=query)

        asyncio.run(adapter._handle_callback_query(update, SimpleNamespace()))

        assert calls == []
        assert query.answers == [
            {"text": "⛔ You are not authorized to approve credentials."}
        ]
