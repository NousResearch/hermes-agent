"""Telegram send idempotence for post-write NetworkErrors (#64238).

The legacy ``TelegramAdapter.send()`` retry loop used to handle only
``TimedOut`` carefully — a generic ``TimedOut`` may have reached Telegram, so
it is re-raised instead of blindly re-sent. But a *non-timeout* ``NetworkError``
(connection reset / ``RemoteProtocolError`` / ``ReadError``) raised **after the
request body was written** is exactly as ambiguous: Telegram may have already
accepted the message. The old code fell through to a blind in-loop resend (up to
3x) and additionally reported the failure as ``retryable=True``, so the
gateway's ``_send_with_retry`` re-sent it up to 2 more times — duplicating the
message in the chat.

``retryable=False`` alone cannot stop the gateway, because
``BasePlatformAdapter._send_with_retry`` computes
``is_network = result.retryable or self._is_retryable_error(error_str)`` and
``_RETRYABLE_ERROR_PATTERNS`` contains both ``"network"`` and
``"connectionreset"`` — the OR discards the adapter's answer. The adapter
therefore also sets the explicit ``SendResult.ambiguous_delivery`` flag, which
the base layer honors unconditionally (before ``is_network`` is computed, and
again inside the retry loop).

These tests pin the idempotence contract:

* a post-write ``NetworkError`` is NOT retried in-loop and surfaces as
  ``retryable=False`` **and** ``ambiguous_delivery=True`` (so neither the
  adapter loop nor the gateway re-sends);
* the same holds for the Bot API 10.1 ``sendRichMessage`` path;
* a *connect-phase* failure (ConnectError / connection refused / DNS) is still
  retried and stays retryable — the request demonstrably never left the
  process, so re-sending cannot duplicate;
* an httpx pool timeout still drains the pool and retries (unchanged);
* end-to-end through ``_send_with_retry``, an ambiguous failure produces
  exactly ONE underlying send.

The first case fails on ``main`` (3 in-loop sends + ``retryable=True`` + 2 more
gateway sends) and passes after the fix.  The base-layer unit coverage for the
flag itself lives in ``tests/gateway/test_send_retry.py``.
"""
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (mod.error.NetworkError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


def _ensure_real_telegram_constants():
    """Give ``telegram.constants`` real ``ChatType``/``ParseMode`` values.

    This module imports the Telegram adapter at module scope, and the adapter
    binds ``from telegram.constants import ParseMode, ChatType`` once, caching
    both as module globals for the whole session. When the real library is
    absent, ``tests/gateway/conftest.py`` registers a single ``MagicMock`` as
    *both* ``telegram`` and ``telegram.constants``, but configures the values
    on ``mod.constants.ChatType`` — a different attribute than the
    ``mod.ChatType`` that ``from telegram.constants import ChatType`` actually
    resolves. The adapter would therefore cache a bare mock, and sibling suites
    that compare a chat's ``type`` against ``adapter.ChatType.SUPERGROUP``
    (``test_telegram_thread_fallback.py``) would see every chat fall through to
    ``"dm"``.

    Set the attributes on whatever object is registered, so no module identity
    is swapped and the stubbed exception classes other suites already imported
    keep working.
    """
    constants = sys.modules.get("telegram.constants")
    if constants is None or getattr(constants, "__file__", None) is not None:
        return  # Not registered, or the real library — leave it alone.
    if isinstance(getattr(constants, "ChatType", None), SimpleNamespace):
        return  # Already normalized by an earlier import of this module.
    constants.ChatType = SimpleNamespace(
        PRIVATE="private",
        GROUP="group",
        SUPERGROUP="supergroup",
        CHANNEL="channel",
    )
    constants.ParseMode = SimpleNamespace(
        MARKDOWN="Markdown",
        MARKDOWN_V2="MarkdownV2",
        HTML="HTML",
    )


_ensure_telegram_mock()
_ensure_real_telegram_constants()

from telegram.error import NetworkError, TimedOut  # noqa: E402

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


class ConnectError(Exception):
    """Stand-in for ``httpx.ConnectError`` — matched by the class-name marker."""


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    return adapter


def _fail_with(exc: BaseException) -> AsyncMock:
    return AsyncMock(side_effect=exc)


# ---------------------------------------------------------------------------
# Post-write NetworkError: must NOT be re-sent (idempotence)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        NetworkError("Connection reset by peer"),
        NetworkError("Server disconnected mid-response (RemoteProtocolError)"),
        NetworkError("httpx.ReadError: connection broken while reading response"),
    ],
)
async def test_post_write_network_error_not_resent(exc):
    """A non-timeout NetworkError after the body was written propagates on the
    first attempt (no in-loop resend) and is reported non-retryable so the
    gateway layer does not re-send it either."""
    adapter = _make_adapter()
    adapter._bot.send_message = _fail_with(exc)

    with patch(
        "plugins.platforms.telegram.adapter.asyncio.sleep", new=AsyncMock()
    ):
        result = await adapter.send("123", "hello")

    assert result.success is False
    # No blind in-loop retry: exactly one underlying send attempt.
    assert adapter._bot.send_message.await_count == 1
    # Non-retryable → gateway _send_with_retry() will not re-send it.
    assert result.retryable is False
    # ...and `retryable=False` alone is not enough: _is_retryable_error()
    # matches "network"/"connectionreset" in the error text and ORs it away.
    # The explicit flag is what the base layer actually honors.
    assert result.ambiguous_delivery is True


@pytest.mark.asyncio
async def test_post_write_network_error_via_cause_chain_not_resent():
    """Even when the ambiguous failure is a wrapped cause (PTB wraps the httpx
    error), a non-connect-phase chain must not be treated as connect-phase."""
    adapter = _make_adapter()
    err = NetworkError("network error while sending")
    err.__cause__ = RuntimeError("Server disconnected without sending a response")
    adapter._bot.send_message = _fail_with(err)

    with patch(
        "plugins.platforms.telegram.adapter.asyncio.sleep", new=AsyncMock()
    ):
        result = await adapter.send("123", "hello")

    assert result.success is False
    assert adapter._bot.send_message.await_count == 1
    assert result.retryable is False
    assert result.ambiguous_delivery is True


# ---------------------------------------------------------------------------
# Connect-phase failures: still retried, still retryable (no over-broadening)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: NetworkError("Connect error: [Errno 111] Connection refused"),
        lambda: NetworkError("getaddrinfo failed: name or service not known"),
    ],
)
async def test_connect_phase_network_error_is_retried(exc_factory):
    """A pre-write connection failure never left the process, so re-sending is
    safe: the loop still retries (3 attempts) and the failure stays retryable."""
    adapter = _make_adapter()
    adapter._bot.send_message = _fail_with(exc_factory())

    with patch(
        "plugins.platforms.telegram.adapter.asyncio.sleep", new=AsyncMock()
    ):
        result = await adapter.send("123", "hello")

    assert result.success is False
    assert adapter._bot.send_message.await_count == 3
    assert result.retryable is True
    # Nothing left the process, so delivery is NOT ambiguous — the base layer
    # must stay free to retry.
    assert result.ambiguous_delivery is False


@pytest.mark.asyncio
async def test_connect_error_via_cause_chain_is_retried():
    """A NetworkError wrapping an httpx.ConnectError (matched by class name on
    the __cause__ chain) is connect-phase → retried and retryable."""
    adapter = _make_adapter()
    err = NetworkError("network error")
    # The connect-phase signal is only on the wrapped cause (class name
    # "ConnectError"), exactly as PTB wraps an httpx.ConnectError.
    err.__cause__ = ConnectError("[Errno 111] Connection refused")
    adapter._bot.send_message = _fail_with(err)

    with patch(
        "plugins.platforms.telegram.adapter.asyncio.sleep", new=AsyncMock()
    ):
        result = await adapter.send("123", "hello")

    assert result.success is False
    assert adapter._bot.send_message.await_count == 3
    assert result.retryable is True
    assert result.ambiguous_delivery is False


# ---------------------------------------------------------------------------
# Generic TimedOut: unchanged (already non-retryable, no resend)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generic_timed_out_still_not_resent():
    """Regression guard: the pre-existing TimedOut behavior is preserved — a
    plain TimedOut still raises on the first attempt and is non-retryable."""
    adapter = _make_adapter()
    adapter._bot.send_message = _fail_with(TimedOut("Timed out"))

    with patch(
        "plugins.platforms.telegram.adapter.asyncio.sleep", new=AsyncMock()
    ):
        result = await adapter.send("123", "hello")

    assert result.success is False
    assert adapter._bot.send_message.await_count == 1
    assert result.retryable is False
    # A generic read timeout is ambiguous for the same reason. Previously this
    # was only stopped by the base layer's `_is_timeout_error` substring scan of
    # the (redacted) error text; the flag makes it explicit and text-independent.
    assert result.ambiguous_delivery is True


# ---------------------------------------------------------------------------
# Pool timeout: unchanged (drains the pool and retries)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pool_timeout_still_drains_and_retries():
    """Regression guard: an httpx pool timeout is explicitly 'not sent to
    Telegram', so the loop still drains the pool and retries (retryable)."""
    adapter = _make_adapter()
    pool_err = TimedOut(
        "Pool timeout: All connections in the connection pool are occupied. "
        "Request was *not* sent to Telegram."
    )
    adapter._bot.send_message = _fail_with(pool_err)
    adapter._drain_general_connections_after_pool_timeout = AsyncMock()

    with patch(
        "plugins.platforms.telegram.adapter.asyncio.sleep", new=AsyncMock()
    ):
        result = await adapter.send("123", "hello")

    assert result.success is False
    assert adapter._bot.send_message.await_count == 3
    assert adapter._drain_general_connections_after_pool_timeout.await_count == 3
    assert result.retryable is True
    # PTB says the request was explicitly NOT sent, so delivery is unambiguous.
    assert result.ambiguous_delivery is False


# ---------------------------------------------------------------------------
# Cross-layer: the gateway retry wrapper must not re-send either
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_with_retry_does_not_resend_post_write_network_error():
    """The bug teknium1 flagged, end-to-end.

    Before the fix the adapter's ``retryable=False`` was ORed away by
    ``_send_with_retry``'s ``self._is_retryable_error(error_str)`` (the error
    text contains both "network" and "connection reset"), so the gateway
    re-sent the payload twice more and then posted a delivery-failure notice —
    up to three duplicate messages plus a bogus "delivery failed" notice for a
    message the user had already received.
    """
    adapter = _make_adapter()
    adapter._bot.send_message = _fail_with(
        NetworkError("NetworkError: connection reset by peer")
    )

    with patch(
        "plugins.platforms.telegram.adapter.asyncio.sleep", new=AsyncMock()
    ), patch("asyncio.sleep", new=AsyncMock()):
        result = await adapter._send_with_retry("123", "hello", max_retries=2, base_delay=0)

    assert result.success is False
    assert result.ambiguous_delivery is True
    # Exactly ONE underlying send: no gateway retry, no plain-text fallback,
    # and no "Message delivery failed after multiple attempts" notice (which
    # would itself be a fourth send).
    assert adapter._bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_send_with_retry_still_retries_connect_phase_failures():
    """Control: the flag must not turn every network failure into a no-retry.

    A connect-phase failure never left the process, so the gateway still gets
    its retries — 3 adapter-loop attempts per ``send()`` call, across the
    initial call plus 2 gateway retries plus the delivery-failure notice.
    """
    adapter = _make_adapter()
    adapter._bot.send_message = _fail_with(
        NetworkError("Connect error: [Errno 111] Connection refused")
    )

    with patch(
        "plugins.platforms.telegram.adapter.asyncio.sleep", new=AsyncMock()
    ), patch("asyncio.sleep", new=AsyncMock()):
        result = await adapter._send_with_retry("123", "hello", max_retries=2, base_delay=0)

    assert result.success is False
    assert result.ambiguous_delivery is False
    assert adapter._bot.send_message.await_count > 1


# ---------------------------------------------------------------------------
# Sibling site: the Bot API 10.1 sendRichMessage path has the same bug
# ---------------------------------------------------------------------------

def _make_rich_adapter() -> TelegramAdapter:
    """Adapter wired for the rich-send path (``sendRichMessage``)."""
    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="***", extra={"rich_messages": True})
    )
    bot = MagicMock()
    # An AsyncMock makes inspect.iscoroutinefunction() true, which is what
    # _bot_supports_rich() checks.
    bot.do_api_request = AsyncMock(return_value={"message_id": 1})
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    bot.send_chat_action = AsyncMock()
    adapter._bot = bot
    return adapter


RICH_CONTENT = "## Results\n\n| Case | Status |\n|---|---|\n| rich | ok |"


@pytest.mark.asyncio
async def test_rich_post_write_network_error_is_ambiguous():
    """``_send_rich_message`` had the identical ``not is_timeout`` computation:
    a post-write NetworkError came back ``retryable=True``, so the base layer
    re-sent it. It is now non-retryable and flagged ambiguous."""
    adapter = _make_rich_adapter()
    adapter._bot.do_api_request = _fail_with(NetworkError("Connection reset by peer"))

    result = await adapter.send("123", RICH_CONTENT)

    assert result.success is False
    assert result.retryable is False
    assert result.ambiguous_delivery is True
    # And no legacy resend of the same payload.
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_rich_send_with_retry_sends_exactly_once():
    adapter = _make_rich_adapter()
    adapter._bot.do_api_request = _fail_with(NetworkError("Connection reset by peer"))

    with patch("asyncio.sleep", new=AsyncMock()):
        result = await adapter._send_with_retry("123", RICH_CONTENT, max_retries=2, base_delay=0)

    assert result.success is False
    assert adapter._bot.do_api_request.await_count == 1
    adapter._bot.send_message.assert_not_called()
