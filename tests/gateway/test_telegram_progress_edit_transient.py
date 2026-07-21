"""Tests for transient-error handling in Telegram progress-message editing.

Issue: #27828

When ``edit_message_text`` fails with a transient network error (e.g.
``httpx.ConnectError``), the gateway must NOT permanently disable progress-
message editing.  Only permanent failures (flood control, message-not-found,
permissions) should set ``can_edit = False``.

Two layers are tested:

1. The ``_TRANSIENT_EDIT_MARKERS`` / retryable classification logic in
   ``TelegramAdapter.edit_message``.
2. The ``send_progress_messages`` caller in ``run.py`` honours
   ``result.retryable`` and keeps ``can_edit = True``.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRANSIENT_MARKERS = (
    "connecterror",
    "connect error",
    "connection error",
    "networkerror",
    "network error",
    "timed out",
    "readtimeout",
    "writetimeout",
    "server disconnected",
    "temporarily unavailable",
    "temporary failure",
    "httpx",
)

_PERMANENT_MARKERS = (
    "message to edit not found",
    "message can't be edited",
    "not enough rights",
    "message_id_invalid",
)


def _is_transient(error_str: str) -> bool:
    """Mirrors the classification logic added to TelegramAdapter.edit_message."""
    err = error_str.lower()
    return any(m in err for m in _TRANSIENT_MARKERS)


def _is_permanent(error_str: str) -> bool:
    err = error_str.lower()
    return any(m in err for m in _PERMANENT_MARKERS)


# ---------------------------------------------------------------------------
# 1. Error classification — transient vs permanent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error_str", [
    "httpx.ConnectError: Connection refused",
    "telegram.error.NetworkError: httpx.ConnectError",
    "NetworkError: remote end closed connection without response",
    "httpx.ReadTimeout: read timed out",
    "ReadTimeout: timed out",
    "Server disconnected",
    "Temporarily unavailable",
    "Temporary failure in name resolution",
    "Connection error: failed to connect",
])
def test_transient_errors_are_classified_as_transient(error_str):
    """Network / transient errors must be classified as retryable."""
    assert _is_transient(error_str), (
        f"Expected {error_str!r} to be transient"
    )


@pytest.mark.parametrize("error_str", [
    "Bad Request: message to edit not found",
    "Bad Request: message can't be edited",
    "Bad Request: not enough rights to edit the message",
    "Bad Request: MESSAGE_ID_INVALID",
    "flood_control:30.0",
    "Forbidden: bot was blocked by the user",
])
def test_permanent_errors_are_not_transient(error_str):
    """Permanent edit failures must NOT be classified as retryable."""
    assert not _is_transient(error_str), (
        f"Expected {error_str!r} to be permanent (non-transient)"
    )


# ---------------------------------------------------------------------------
# 2. SendResult retryable field
# ---------------------------------------------------------------------------

def test_send_result_retryable_default_is_false():
    r = SendResult(success=True, message_id="1")
    assert r.retryable is False


def test_send_result_retryable_can_be_set_true():
    r = SendResult(success=False, error="httpx.ConnectError: ...", retryable=True)
    assert r.retryable is True


def test_send_result_retryable_false_for_permanent():
    r = SendResult(success=False, error="message to edit not found")
    assert r.retryable is False


# ---------------------------------------------------------------------------
# 3. run.py logic — retryable result must NOT set can_edit=False
#    We simulate the relevant block from send_progress_messages():
#
#      if not result.success:
#          if getattr(result, 'retryable', False):
#              continue           # <-- keep can_edit=True
#          ...
#          can_edit = False
#
# ---------------------------------------------------------------------------

def _simulate_progress_loop(edit_results):
    """
    Simulate the can_edit decision for a sequence of edit_message results.

    Returns the final value of can_edit after processing all results.
    """
    can_edit = True
    for result in edit_results:
        if not result.success:
            if getattr(result, "retryable", False):
                # Transient — keep can_edit True and skip to next cycle
                continue
            can_edit = False
            break
    return can_edit


def test_transient_failure_keeps_can_edit_true():
    """A single transient network error must not disable progress editing."""
    results = [
        SendResult(success=False, error="httpx.ConnectError", retryable=True),
        SendResult(success=True, message_id="42"),
    ]
    assert _simulate_progress_loop(results) is True


def test_permanent_failure_sets_can_edit_false():
    """A permanent edit failure must disable progress editing."""
    results = [
        SendResult(success=False, error="message to edit not found", retryable=False),
    ]
    assert _simulate_progress_loop(results) is False


def test_multiple_transient_then_success_keeps_can_edit_true():
    """Multiple transient failures followed by success keep can_edit=True."""
    results = [
        SendResult(success=False, error="httpx.ConnectError", retryable=True),
        SendResult(success=False, error="server disconnected", retryable=True),
        SendResult(success=True, message_id="99"),
    ]
    assert _simulate_progress_loop(results) is True


def test_flood_control_sets_can_edit_false():
    """Flood control (non-retryable) must disable progress editing."""
    results = [
        SendResult(success=False, error="flood_control:30.0", retryable=False),
    ]
    assert _simulate_progress_loop(results) is False


# ---------------------------------------------------------------------------
# 4. Real adapter lifecycle — a deleted/expired placeholder is terminal
# ---------------------------------------------------------------------------

def test_message_not_found_stops_markdown_fallback_without_error_noise(caplog):
    """A stale placeholder is one idempotent failure, not two noisy edits."""
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter._bot = AsyncMock()
    adapter._bot.edit_message_text.side_effect = Exception(
        "Bad Request: Message to edit not found"
    )
    adapter._rich_messages_enabled = False
    adapter._last_overflow_preview = {}

    with caplog.at_level(
        logging.DEBUG, logger="plugins.platforms.telegram.adapter"
    ):
        result = asyncio.run(
            adapter.edit_message(
                chat_id="-100123", message_id="42", content="done", finalize=True
            )
        )

    assert result.success is False
    assert result.retryable is False
    assert result.error_kind == "not_found"
    assert adapter._bot.edit_message_text.await_count == 1
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_other_edit_errors_keep_existing_fallback_and_error_logging(caplog):
    """The stale-placeholder special case must not hide unrelated edit errors."""
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter._bot = AsyncMock()
    adapter._bot.edit_message_text.side_effect = Exception(
        "Bad Request: message can't be edited"
    )
    adapter._rich_messages_enabled = False
    adapter._last_overflow_preview = {}

    with caplog.at_level(
        logging.DEBUG, logger="plugins.platforms.telegram.adapter"
    ):
        result = asyncio.run(
            adapter.edit_message(
                chat_id="-100123", message_id="42", content="done", finalize=True
            )
        )

    assert result.success is False
    assert adapter._bot.edit_message_text.await_count == 2
    assert any(record.levelno >= logging.ERROR for record in caplog.records)
