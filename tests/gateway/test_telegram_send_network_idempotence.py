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

These tests pin the idempotence contract:

* a post-write ``NetworkError`` is NOT retried in-loop and surfaces as
  ``retryable=False`` (so neither the adapter loop nor the gateway re-sends);
* a *connect-phase* failure (ConnectError / connection refused / DNS) is still
  retried and stays retryable — the request demonstrably never left the
  process, so re-sending cannot duplicate;
* an httpx pool timeout still drains the pool and retries (unchanged).

The first case fails on ``main`` (3 in-loop sends + ``retryable=True``) and
passes after the fix.
"""
import sys
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


_ensure_telegram_mock()

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
