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


class TestTelegramImportFailureReporting:
    """#87455: the ImportError that stubs the Telegram classes must be
    diagnosable at the stub site — present-but-broken logs the swallowed
    traceback; a clean not-installed stays quiet (debug only)."""

    def test_installed_but_broken_import_logs_warning_with_traceback(
        self, monkeypatch, caplog
    ):
        import importlib.util
        import logging

        monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

        with caplog.at_level(logging.DEBUG):
            try:
                raise ImportError("partial dist-info race")
            except ImportError:
                telegram_mod._report_telegram_import_failure()

        assert "failed to import" in caplog.text
        assert any(record.exc_info for record in caplog.records)

    def test_not_installed_stays_debug_only(self, monkeypatch, caplog):
        import importlib.util
        import logging

        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

        with caplog.at_level(logging.DEBUG):
            try:
                raise ImportError("No module named 'telegram'")
            except ImportError:
                telegram_mod._report_telegram_import_failure()

        assert "not installed" in caplog.text
        assert not [
            record for record in caplog.records if record.levelno >= logging.WARNING
        ]

    def test_unprobeable_install_is_treated_as_broken(self, monkeypatch, caplog):
        import importlib.util
        import logging

        def _explode(name):
            raise ValueError("finder misbehaving")

        monkeypatch.setattr(importlib.util, "find_spec", _explode)

        with caplog.at_level(logging.DEBUG):
            try:
                raise ImportError("original cause")
            except ImportError:
                telegram_mod._report_telegram_import_failure()

        assert "failed to import" in caplog.text

