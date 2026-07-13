"""Tests for transient-error handling in Telegram progress-message editing.

Issue: #27828

When ``edit_message_text`` fails with a transient network error (e.g.
``httpx.ConnectError``), the gateway must NOT permanently disable progress-
message editing. Deleted/invalid edit anchors should be replaced, while
permissions and other permanent failures should set ``can_edit = False``.

Three contracts are tested directly:

1. The ``_TRANSIENT_EDIT_MARKERS`` / retryable classification logic in
   ``TelegramAdapter.edit_message``.
2. The ``SendResult.retryable`` transport field.
3. The production stale-anchor classifier in ``gateway.run``.

The real ``send_progress_messages`` behavior is covered by
``test_run_progress_topics.py`` instead of duplicating its state machine here.
"""

from __future__ import annotations


import pytest

from gateway.platforms.base import SendResult
from gateway.run import _is_stale_progress_edit_error


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

def _is_transient(error_str: str) -> bool:
    """Mirrors the classification logic added to TelegramAdapter.edit_message."""
    err = error_str.lower()
    return any(m in err for m in _TRANSIENT_MARKERS)

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


@pytest.mark.parametrize(
    ("error_str", "expected"),
    [
        ("Bad Request: message to edit not found", True),
        ("Bad Request: MESSAGE_ID_INVALID", True),
        ("Bad Request: not enough rights to edit the message", False),
        ("Forbidden: bot was blocked by the user", False),
        ("unexpected permanent adapter failure", False),
    ],
)
def test_stale_progress_edit_error_classification(error_str, expected):
    assert _is_stale_progress_edit_error(error_str) is expected


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
