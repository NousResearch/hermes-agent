"""Regression: a Telegram transport error whose str() is empty must still name its type.

Background (ai-hub incident, 2026-08-27, kanban t_31ab4b8c)
------------------------------------------------------------
``_redact_telegram_error_text()`` returns ``str(error)`` unchanged. Several
exception types that dominate the polling-recovery ladder stringify to the
EMPTY string -- most importantly ``asyncio.TimeoutError``, which is exactly
what ``_await_with_thread_deadline`` raises when the 30s ``telegram-init``
deadline expires.

The result is a log line that ends in a colon and nothing else::

    [Telegram] Telegram network error (attempt 10/10), reconnecting in 60s. Error:
    [Telegram] Telegram polling could not reconnect after 10 network error
        retries. Escalating to gateway recovery. Last error:

Every diagnostic signal is gone: the operator cannot tell a DNS failure from a
routing failure from a deadline expiry. On 2026-08-27 this cost a full
misdiagnosis -- the blank errors were read as "no route to host", an IPv6
theory was built on top, and the actual cause (a 6h Telegram per-chat flood
wait, plus a 30s init deadline expiring under load) was missed entirely.

Upstream fixed this exact class in 6cbb7b6115 ("preserve exception type when
error string is empty", #78183) for BlueBubbles, WhatsApp Cloud, QQ Bot and
Yuanbao -- using ``str(exc) or type(exc).__name__``. The Telegram adapter was
not included in that sweep even though it has the same latent bug, and it is
the adapter with the most aggressive automatic-recovery ladder.

These tests are falsifying: they fail against the pre-fix helper.
"""

import asyncio

import pytest

from plugins.platforms.telegram.adapter import _redact_telegram_error_text


# Exceptions that really do stringify to "" and really do occur on the
# Telegram polling/recovery paths.
EMPTY_STRINGIFYING_ERRORS = [
    asyncio.TimeoutError(),
    asyncio.CancelledError(),
    ConnectionResetError(),
    OSError(),
    RuntimeError(),
]


@pytest.mark.parametrize("error", EMPTY_STRINGIFYING_ERRORS)
def test_empty_error_string_falls_back_to_type_name(error):
    """An exception with no message must log its class name, never ''."""
    rendered = _redact_telegram_error_text(error)

    assert rendered, (
        f"{type(error).__name__} rendered as empty string; the log line would "
        f"read 'Error:' with nothing after it and the operator loses the only "
        f"signal about what failed"
    )
    assert rendered == f"<{type(error).__name__}>"


def test_timeout_error_names_itself_specifically():
    """The 'telegram-init' 30s deadline path is the one that burned us."""
    assert _redact_telegram_error_text(asyncio.TimeoutError()) == "<TimeoutError>"


def test_non_empty_error_text_is_preserved_unchanged():
    """The fallback must not shadow a real message."""
    assert (
        _redact_telegram_error_text(OSError(101, "Network is unreachable"))
        == "[Errno 101] Network is unreachable"
    )


def test_none_still_renders_empty():
    """None is 'no error', not 'an error we failed to describe'."""
    assert _redact_telegram_error_text(None) == ""


def test_bot_token_redaction_still_applies():
    """The fallback must not regress the security property this helper exists for."""
    # Must match agent.redact._TELEGRAM_RE: 8+ digits, colon, 30+ token chars.
    token = "123456789:AAHdqQABBBBCCCCDDDDEEEEFFFFGGGGHHHHIII"
    err = RuntimeError(f"failed calling https://api.telegram.org/bot{token}/getMe")

    rendered = _redact_telegram_error_text(err)

    assert token not in rendered
    assert rendered  # and it is still non-empty


def test_type_name_fallback_is_not_itself_secret_bearing():
    """A class name can never carry a token, so the fallback is redaction-safe."""

    class ConnectTimeout(Exception):
        pass

    assert _redact_telegram_error_text(ConnectTimeout()) == "<ConnectTimeout>"
