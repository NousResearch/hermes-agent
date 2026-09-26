"""Tests for Telegram connect() non-retryable fatal error on missing credentials.

When Telegram has no bot token or no python-telegram-bot installed, connect()
must set a non-retryable fatal error so the gateway does not queue it for
background reconnection (#31049).
"""


import pytest

from gateway.config import PlatformConfig
import plugins.platforms.telegram.adapter as telegram_mod  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


class TestTelegramUnconfiguredNonRetryable:
    """Verify that missing dependency/token sets a non-retryable fatal error."""

    @pytest.mark.asyncio
    async def test_no_telegram_lib_sets_non_retryable_fatal(self, monkeypatch):
        """connect() with python-telegram-bot unavailable → non-retryable fatal error."""
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake"))
        monkeypatch.setattr(telegram_mod, "TELEGRAM_AVAILABLE", False)
        result = await adapter.connect()
        assert result is False
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_retryable is False
        assert adapter.fatal_error_code == "missing_dependency"


def test_lazy_install_rebinds_type_handler(monkeypatch):
    """After a first-import miss, TypeHandler must be rebound to the real PTB class.

    check_telegram_requirements() used to rebind Update/Application/HTTPXRequest
    but leave TypeHandler as typing.Any. connect() then crashed in
    _register_handlers with ``Any cannot be instantiated`` — the live
    telegram gateway failure after a lazy install.
    """
    from typing import Any as TypingAny

    monkeypatch.setattr(telegram_mod, "TELEGRAM_AVAILABLE", False)
    monkeypatch.setattr(telegram_mod, "TypeHandler", TypingAny)
    monkeypatch.setattr(telegram_mod, "Update", TypingAny)

    def _fake_ensure(name, prompt=False):
        assert name == "platform.telegram"
        return True

    monkeypatch.setattr(
        "tools.lazy_deps.ensure",
        _fake_ensure,
    )

    assert telegram_mod.check_telegram_requirements() is True
    assert telegram_mod.TELEGRAM_AVAILABLE is True
    assert telegram_mod.TypeHandler is not TypingAny

    handler = telegram_mod.TypeHandler(
        telegram_mod.Update, lambda *_args, **_kwargs: None
    )
    assert handler is not None

