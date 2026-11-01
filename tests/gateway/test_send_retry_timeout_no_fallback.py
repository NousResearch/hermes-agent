"""A retry that ends in a timeout must not be reformatted into a second send (#147536).

``_send_with_retry`` already refuses to retry a *first-attempt* timeout ("may have delivered"),
but the retry loop reclassifies each attempt without ``_is_timeout_error``. A retry that ended in a
read timeout therefore broke the loop as "non-transient" and fell through to
``_send_plain_fallback``, which re-sends the whole payload — the user gets the message twice.
"""

from typing import Any

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


class _Adapter(BasePlatformAdapter):
    """Sends fail with a scripted list of results; records every call."""

    def __init__(self, results):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self._results = list(results)
        self.sent: list[str] = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        return self._results.pop(0) if self._results else SendResult(success=True, message_id="ok")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def _sleep(_delay):
        return None
    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", _sleep)


def _transient() -> SendResult:
    return SendResult(success=False, error="Connection reset by peer", retryable=True)


def _timeout() -> SendResult:
    return SendResult(success=False, error="ReadTimeout: the read operation timed out", retryable=False)


@pytest.mark.asyncio
async def test_a_timeout_on_retry_is_returned_instead_of_reformatted():
    retry_timeout = _timeout()
    adapter = _Adapter([_transient(), retry_timeout])

    result = await adapter._send_with_retry("c", "hello")

    assert result is retry_timeout
    # The retry itself is legitimate; the plain-text fallback would be the duplicate.
    assert adapter.sent == ["hello", "hello"]


@pytest.mark.asyncio
async def test_a_first_attempt_timeout_stays_single_send():
    """Pins the behaviour the retry path now mirrors — no retry, no fallback."""
    adapter = _Adapter([_timeout()])

    result = await adapter._send_with_retry("c", "hello")

    assert result.error and "timed out" in result.error
    assert adapter.sent == ["hello"]


@pytest.mark.asyncio
async def test_a_formatting_error_on_retry_still_falls_back_to_plain_text():
    """The timeout guard must not swallow the fallback the plain-text path exists for."""
    formatting = SendResult(success=False, error="Bad Request: can't parse entities", retryable=False)
    adapter = _Adapter([_transient(), formatting, SendResult(success=True, message_id="fallback")])

    result = await adapter._send_with_retry("c", "<b>hello")

    assert result.success is True
    assert len(adapter.sent) == 3  # attempt, retry, plain-text fallback
    assert adapter.sent[-1] != "<b>hello"


@pytest.mark.asyncio
async def test_a_rate_limited_retry_keeps_retrying_rather_than_returning_early():
    """A flood/rate-limit classification wins over the timeout guard: the ledger owns redelivery."""
    rate_limited = SendResult(success=False, error="Too Many Requests: retry after 3", retryable=False,
                              retry_after=3)
    adapter = _Adapter([_transient(), rate_limited, SendResult(success=True, message_id="ok2")])

    result = await adapter._send_with_retry("c", "hello")

    assert result.success is True
    assert adapter.sent == ["hello", "hello", "hello"]
