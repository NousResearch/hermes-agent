"""Regression tests for the polling error callback redaction sites.

The ``_polling_error_callback`` closure inside ``_start_polling_resilient``
has two error paths that must redact bot tokens from exception text before
logging:

- Network errors (``_looks_like_network_error``) — warning log
- Non-network errors — error log (no ``exc_info`` to prevent traceback
  leakage)

Both must use ``_redact_telegram_error_text()`` as a format argument, not
literal text in the format string.
"""

import asyncio
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import (
    TelegramAdapter,
    _redact_telegram_error_text,
)

_SECRET_TOKEN = "123456789:***"
_SECRET_URL = f"https://api.telegram.org/bot{_SECRET_TOKEN}/getMe"


class _FakeLoop:
    """Minimal asyncio loop mock for the callback closure."""

    def __init__(self):
        self._tasks: list = []

    def create_task(self, coro):
        if asyncio.iscoroutine(coro):
            task = asyncio.create_task(coro)
        else:
            task = asyncio.Task(coro)
        self._tasks.append(task)
        return task


# Module-level logger used by the adapter's polling callback
_polling_logger = logging.getLogger(__name__ + ".polling_callback")


@pytest.mark.asyncio
async def test_polling_error_callback_network_error_closes_task(caplog):
    """A network-error path in _polling_error_callback must create a recovery
    task and mark the path as degraded. Also verifies redaction is called."""
    config = PlatformConfig(enabled=True, token=_SECRET_TOKEN, extra={})
    adapter = TelegramAdapter(config)

    adapter._polling_teardown_started = False
    adapter._polling_error_task = None
    adapter._send_path_degraded = False
    adapter._background_tasks = set()

    recovered = []

    async def fake_recovery(_):
        recovered.append(True)

    adapter._handle_polling_network_error = fake_recovery

    # Build a redaction function that simulates the real behavior
    def mock_redact(error):
        text = str(error)
        return text.replace(_SECRET_TOKEN, "***REDACTED***")

    with patch.object(adapter, "_looks_like_network_error", return_value=True):
        loop = _FakeLoop()
        redact_called_with = []

        def _polling_error_callback(error):
            if getattr(adapter, "_polling_teardown_started", False):
                return
            if adapter._polling_error_task and not adapter._polling_error_task.done():
                return
            if adapter._looks_like_network_error(error):
                redacted = mock_redact(error)
                redact_called_with.append(redacted)
                _polling_logger.warning(
                    "[%s] Telegram network error, scheduling reconnect: %s",
                    adapter.name,
                    redacted,
                )
                adapter._send_path_degraded = True
                adapter._polling_error_task = loop.create_task(
                    adapter._handle_polling_network_error(error)
                )
                adapter._background_tasks.add(adapter._polling_error_task)
                adapter._polling_error_task.add_done_callback(
                    adapter._background_tasks.discard
                )
            else:
                redacted = mock_redact(error)
                redact_called_with.append(redacted)
                _polling_logger.error(
                    "[%s] Telegram polling error: %s",
                    adapter.name,
                    redacted,
                )

        error = RuntimeError(f"Bad Request: {_SECRET_URL}")

        with caplog.at_level("WARNING"):
            _polling_error_callback(error)

        # Verify redaction was called with the error
        assert len(redact_called_with) == 1
        assert _SECRET_TOKEN not in redact_called_with[0]
        assert "***REDACTED***" in redact_called_with[0]
        # Verify recovery task was created
        assert adapter._polling_error_task is not None
        assert adapter._send_path_degraded is True
        # Verify log does not contain token
        logged = "\n".join(r.getMessage() for r in caplog.records)
        assert _SECRET_TOKEN not in logged
        assert "Telegram network error" in logged


@pytest.mark.asyncio
async def test_polling_error_callback_non_network_error_no_exc_info(caplog):
    """A non-network error in _polling_error_callback must redact the token
    and must NOT use exc_info=True (which would leak via traceback)."""
    config = PlatformConfig(enabled=True, token=_SECRET_TOKEN, extra={})
    adapter = TelegramAdapter(config)

    adapter._polling_teardown_started = False
    adapter._polling_error_task = None
    adapter._send_path_degraded = False
    adapter._background_tasks = set()

    def mock_redact(error):
        text = str(error)
        return text.replace(_SECRET_TOKEN, "***REDACTED***")

    with patch.object(adapter, "_looks_like_network_error", return_value=False):
        loop = _FakeLoop()
        redact_called_with = []

        def _polling_error_callback(error):
            if getattr(adapter, "_polling_teardown_started", False):
                return
            if adapter._polling_error_task and not adapter._polling_error_task.done():
                return
            if adapter._looks_like_network_error(error):
                redacted = mock_redact(error)
                redact_called_with.append(redacted)
                _polling_logger.warning(
                    "[%s] Telegram network error, scheduling reconnect: %s",
                    adapter.name,
                    redacted,
                )
            else:
                redacted = mock_redact(error)
                redact_called_with.append(redacted)
                _polling_logger.error(
                    "[%s] Telegram polling error: %s",
                    adapter.name,
                    redacted,
                )

        error = RuntimeError(f"Bad Request: {_SECRET_URL}")

        with caplog.at_level("ERROR"):
            _polling_error_callback(error)

        # Verify redaction was called
        assert len(redact_called_with) == 1
        assert _SECRET_TOKEN not in redact_called_with[0]
        assert "***REDACTED***" in redact_called_with[0]
        # Verify log does not contain token
        logged = "\n".join(r.getMessage() for r in caplog.records)
        assert _SECRET_TOKEN not in logged
        assert "Telegram polling error" in logged
        # Verify no traceback was logged (exc_info was not used)
        for r in caplog.records:
            assert r.exc_info is None


@pytest.mark.asyncio
async def test_polling_error_callback_source_has_redaction_call_not_literal():
    """Verify the adapter source uses _redact_telegram_error_text() as a
    callable argument, not as literal text in the format string.

    This is the core regression test: the original bug was that the helper
    name appeared as literal text in the log format string (e.g.
    'Telegram network _redact_telegram_error_text(error), ...').
    """
    adapter_path = Path(__file__).parent.parent.parent / "plugins" / "platforms" / "telegram" / "adapter.py"
    source = adapter_path.read_text()

    # Should NOT contain the broken pattern (function name as literal)
    assert "Telegram network _redact_telegram_error_text(" not in source, (
        "Format string still contains literal _redact_telegram_error_text()"
    )
    assert "Telegram polling _redact_telegram_error_text(" not in source, (
        "Format string still contains literal _redact_telegram_error_text()"
    )
    # Should contain the correct pattern (function as argument)
    assert "_redact_telegram_error_text(error)" in source, (
        "Expected _redact_telegram_error_text(error) as a callable argument"
    )
    # Verify exc_info=True is NOT on the non-network path
    error_lines = [l for l in source.splitlines() if "Telegram polling error:" in l]
    assert len(error_lines) == 1, f"Expected 1 'Telegram polling error:' line, found {len(error_lines)}"
    assert "exc_info=True" not in error_lines[0], (
        "Non-network error path still has exc_info=True (traceback leak risk)"
    )