"""
Tests for fd-leak fix: TelegramAdapter.connect() must clean up a
partially-built PTB Application when connect() fails mid-way.

Regression tests for the fix applied in:
  fix(gateway): release orphaned Telegram Application on failed connect (#14210)

Without the fix, httpx connection pools inside the Application are never
closed, leaving sockets in CLOSE_WAIT until the process dies and eventually
exhausting the fd limit (EMFILE) after repeated gateway supervisor retries.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def _no_auto_discovery(monkeypatch):
    async def _noop():
        return []

    monkeypatch.setattr("gateway.platforms.telegram.discover_fallback_ips", _noop)


def _make_adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))


def _make_mock_app(updater_running: bool = True):
    mock_updater = MagicMock()
    mock_updater.running = updater_running
    mock_updater.stop = AsyncMock()

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.running = True
    mock_app.stop = AsyncMock()
    mock_app.shutdown = AsyncMock()
    mock_app.initialize = AsyncMock(side_effect=Exception("initialize() failed"))

    return mock_app, mock_updater


def _patch_application(mock_app):
    """Patch gateway.platforms.telegram.Application so builder().…build() returns mock_app."""
    builder_chain = MagicMock()
    builder_chain.token.return_value = builder_chain
    builder_chain.base_url.return_value = builder_chain
    builder_chain.base_file_url.return_value = builder_chain
    builder_chain.local_mode.return_value = builder_chain
    builder_chain.request.return_value = builder_chain
    builder_chain.get_updates_request.return_value = builder_chain
    builder_chain.build.return_value = mock_app

    mock_application_cls = MagicMock()
    mock_application_cls.builder.return_value = builder_chain

    return patch("gateway.platforms.telegram.Application", mock_application_cls)


@pytest.mark.asyncio
async def test_connect_cleanup_runs_on_failure():
    """
    When connect() fails after self._app is set, the except handler must:
    - await updater.stop() (because updater.running is True)
    - await app.stop()    (because app.running is True)
    - await app.shutdown()
    - set self._app = None and self._bot = None
    - return False
    """
    adapter = _make_adapter()
    mock_app, mock_updater = _make_mock_app(updater_running=True)

    with _patch_application(mock_app):
        result = await adapter.connect()

    assert result is False, "connect() must return False on failure"
    assert adapter._app is None, "self._app must be None after failed connect()"
    assert adapter._bot is None, "self._bot must be None after failed connect()"

    mock_updater.stop.assert_awaited_once()
    mock_app.stop.assert_awaited_once()
    mock_app.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_connect_cleanup_skipped_when_app_not_built():
    """
    If connect() fails before self._app is assigned, the cleanup block must
    not raise — there is nothing to clean up.
    """
    adapter = _make_adapter()

    mock_application_cls = MagicMock()
    mock_application_cls.builder.side_effect = Exception("builder exploded before build()")

    with patch("gateway.platforms.telegram.Application", mock_application_cls):
        result = await adapter.connect()

    assert result is False
    assert adapter._app is None


@pytest.mark.asyncio
async def test_connect_cleanup_error_does_not_mask_original_exception():
    """
    When the inner cleanup itself raises, connect() must still:
    - return False
    - set self._app = None  (the finally block runs)
    - record the ORIGINAL error via _set_fatal_error, not the cleanup error
    """
    adapter = _make_adapter()
    mock_app, mock_updater = _make_mock_app(updater_running=True)

    mock_app.shutdown = AsyncMock(side_effect=Exception("shutdown() exploded during cleanup"))

    recorded_errors = []
    original_set_fatal = adapter._set_fatal_error

    def _capture_fatal(code, message, **kw):
        recorded_errors.append((code, message))
        original_set_fatal(code, message, **kw)

    adapter._set_fatal_error = _capture_fatal

    with _patch_application(mock_app):
        result = await adapter.connect()

    assert result is False, "connect() must return False even when cleanup raises"
    assert adapter._app is None, "self._app must be None (finally block must run)"

    assert len(recorded_errors) == 1, "Exactly one fatal error must be recorded"
    code, message = recorded_errors[0]
    assert code == "telegram_connect_error"
    assert "initialize() failed" in message, (
        f"Fatal error must record the ORIGINAL exception, got: {message!r}"
    )
    assert "shutdown() exploded" not in message, (
        "Cleanup error must NOT replace the original exception in _set_fatal_error"
    )
    assert "shutdown() exploded" not in message, (
        "Cleanup error must NOT replace the original exception in _set_fatal_error"
    )
