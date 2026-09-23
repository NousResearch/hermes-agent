"""
Tests for BasePlatformAdapter._send_with_retry and _is_retryable_error.

Verifies that:
- Transient network errors trigger retry with backoff
- Permanent errors fall back to plain-text immediately (no retry)
- User receives a delivery-failure notice when all retries are exhausted
- Successful sends on retry return success
- SendResult.retryable flag is respected
"""
import pytest
from unittest.mock import AsyncMock, patch

from gateway.platforms.base import BasePlatformAdapter, SendResult, _RETRYABLE_ERROR_PATTERNS
from gateway.platforms.base import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Minimal concrete adapter for testing (no real network)
# ---------------------------------------------------------------------------

class _StubAdapter(BasePlatformAdapter):
    def __init__(self):
        cfg = PlatformConfig()
        super().__init__(cfg, Platform.TELEGRAM)
        self._send_results = []   # queue of SendResult to return per call
        self._send_calls = []     # record of (chat_id, content) sent

    def _next_result(self) -> SendResult:
        if self._send_results:
            return self._send_results.pop(0)
        return SendResult(success=True, message_id="ok")

    async def send(self, chat_id, content, reply_to=None, metadata=None, **kwargs) -> SendResult:
        self._send_calls.append((chat_id, content))
        return self._next_result()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send_typing(self, chat_id, metadata=None) -> None:
        pass

    async def get_chat_info(self, chat_id):
        return {"name": "test", "type": "direct", "chat_id": chat_id}


# ---------------------------------------------------------------------------
# _is_retryable_error
# ---------------------------------------------------------------------------

class TestIsRetryableError:
    def test_none_is_not_retryable(self):
        assert not _StubAdapter._is_retryable_error(None)

    def test_empty_string_is_not_retryable(self):
        assert not _StubAdapter._is_retryable_error("")


    def test_permission_error_not_retryable(self):
        assert not _StubAdapter._is_retryable_error("Forbidden: bot was blocked by the user")


# ---------------------------------------------------------------------------
# _is_timeout_error
# ---------------------------------------------------------------------------

class TestIsTimeoutError:
    def test_none_is_not_timeout(self):
        assert not _StubAdapter._is_timeout_error(None)

    def test_empty_is_not_timeout(self):
        assert not _StubAdapter._is_timeout_error("")


# ---------------------------------------------------------------------------
# _is_rate_limited_error
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _send_with_retry — success on first attempt
# ---------------------------------------------------------------------------

class TestSendWithRetrySuccess:
    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        adapter = _StubAdapter()
        adapter._send_results = [SendResult(success=True, message_id="123")]
        result = await adapter._send_with_retry("chat1", "hello")
        assert result.success
        assert len(adapter._send_calls) == 1


# ---------------------------------------------------------------------------
# _send_with_retry — network error with successful retry
# ---------------------------------------------------------------------------

class TestSendWithRetryNetworkRetry:
    @pytest.mark.asyncio
    async def test_retries_on_connect_error_and_succeeds(self):
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="httpx.ConnectError: connection refused"),
            SendResult(success=True, message_id="ok"),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._send_with_retry("chat1", "hello", max_retries=2, base_delay=0)
        assert result.success
        assert len(adapter._send_calls) == 2  # initial + 1 retry

    @pytest.mark.asyncio
    async def test_timeout_not_retried_to_prevent_duplicates(self):
        """ReadTimeout is NOT retried because the request may have reached
        the server — retrying a non-idempotent send risks duplicate delivery.
        It also skips plain-text fallback (timeout is not a formatting issue)."""
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="ReadTimeout: request timed out"),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter._send_with_retry("chat1", "hello", max_retries=3, base_delay=0)
        # No retry, no fallback — timeout returns failure immediately
        mock_sleep.assert_not_called()
        assert not result.success
        assert len(adapter._send_calls) == 1


# ---------------------------------------------------------------------------
# _send_with_retry — all retries exhausted → user notification
# ---------------------------------------------------------------------------

class TestSendWithRetryExhausted:
    @pytest.mark.asyncio
    async def test_sends_user_notice_after_exhaustion(self):
        adapter = _StubAdapter()
        network_err = SendResult(success=False, error="httpx.ConnectError: host unreachable")
        # initial + 2 retries + notice attempt
        adapter._send_results = [network_err, network_err, network_err, SendResult(success=True)]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._send_with_retry("chat1", "hello", max_retries=2, base_delay=0)
        # Result is the last failed one (before notice)
        assert not result.success
        # 4 total calls: 1 initial + 2 retries + 1 notice
        assert len(adapter._send_calls) == 4
        # The notice content should mention delivery failure
        notice_content = adapter._send_calls[-1][1]
        assert "delivery failed" in notice_content.lower() or "Message delivery failed" in notice_content


# ---------------------------------------------------------------------------
# _send_with_retry — non-network failure → plain-text fallback (no retry)
# ---------------------------------------------------------------------------

class TestSendWithRetryFallback:
    @pytest.mark.asyncio
    async def test_non_network_error_falls_back_immediately(self):
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="Bad Request: can't parse entities"),
            SendResult(success=True, message_id="fallback_ok"),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter._send_with_retry("chat1", "**bold**", max_retries=2, base_delay=0)
        # No sleep — no retry loop for non-network errors
        mock_sleep.assert_not_called()
        assert result.success
        assert len(adapter._send_calls) == 2
        # Fallback content should be plain-text notice
        assert "plain text" in adapter._send_calls[1][1].lower()


# ---------------------------------------------------------------------------
# _send_with_retry — retry_after honor
# ---------------------------------------------------------------------------

class TestSendWithRetryAfter:
    @pytest.mark.asyncio
    async def test_retry_after_honored_on_first_retry(self):
        """When the initial result has retry_after, the first retry waits that long."""
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="Flood control exceeded. Retry in 37 seconds",
                       retryable=True, retry_after=37.0),
            SendResult(success=True, message_id="ok"),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter._send_with_retry("chat1", "hello", max_retries=2, base_delay=2.0)
        assert result.success
        # First sleep should use retry_after (~37s + jitter), not base_delay (~2s)
        first_sleep = mock_sleep.call_args_list[0][0][0]
        assert first_sleep >= 36.0  # 37 - 1 (max jitter)

    @pytest.mark.asyncio
    async def test_retry_after_from_subsequent_result(self):
        """If a retry itself returns retry_after, the next retry honors it."""
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="ConnectError", retryable=True),
            SendResult(success=False, error="Flood control exceeded. Retry in 30 seconds",
                       retryable=True, retry_after=30.0),
            SendResult(success=True, message_id="ok"),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter._send_with_retry("chat1", "hello", max_retries=3, base_delay=2.0)
        assert result.success
        # Second sleep should use the retry_after from the second result
        second_sleep = mock_sleep.call_args_list[1][0][0]
        assert second_sleep >= 29.0  # 30 - 1 (max jitter)


# ---------------------------------------------------------------------------
# _send_with_retry — rate-limited sends (flood control) without retry_after
# ---------------------------------------------------------------------------
# Platforms like Weixin surface a rate limit as a bare error with NO retry_after
# field.  These must still be treated as transient (back off / retry) rather than
# falling through to the truncating plain-text fallback, which would re-enter the
# server ban and drop the tail of the message.

class TestSendWithRetryRateLimited:

    @pytest.mark.asyncio
    async def test_long_server_retry_after_returns_typed_failure_without_sleeping(self):
        """A retry_after past the inline cap must not pin the coroutine (#91969): the
        typed failure goes back to the delivery ledger, no sleep, no fallback, no notice."""
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="flood_control:5820", retry_after=5820.0),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter._send_with_retry("c", "hello")
        assert result.success is False
        assert result.retry_after == 5820.0
        mock_sleep.assert_not_called()
        assert len(adapter._send_calls) == 1  # the original send only

    @pytest.mark.asyncio
    async def test_rate_limited_exhausted_returns_typed_failure_not_fallback(self):
        """When retries on a rate-limited send are exhausted, return the typed
        failure for ledger redelivery. Never the truncating plain-text fallback,
        and — because the final failure is still inside the flood penalty — never
        a delivery-failure notice send that would re-enter the ban. 1 initial + 2
        retries = 3 sends, no fourth notice send."""
        adapter = _StubAdapter()
        flood = SendResult(success=False, error="flood control exceeded, retry in 60 seconds")
        adapter._send_results = [flood, flood, flood]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter._send_with_retry("chat1", "hello", max_retries=2, base_delay=0)
        assert not result.success
        # 1 initial + 2 retries = 3 sends; NO delivery-failure notice (4th send)
        # is sent inside the active flood penalty
        assert len(adapter._send_calls) == 3
        # The only sends are the retries — no truncating "plain text" fallback and
        # no notice triggered inside the flood penalty
        for chat_id, content in adapter._send_calls:
            assert "plain text" not in content.lower(), \
                f"rate-limited send must not fall through to plain-text fallback, got: {content[:40]!r}"
            assert "delivery failed" not in content.lower() and \
                   "Message delivery failed" not in content, \
                "no notice send inside active flood penalty"


# ---------------------------------------------------------------------------
# _send_with_retry — failure-kind transitions between attempts
# ---------------------------------------------------------------------------
# is_rate_limited is recomputed from the refreshed error_str on every retry, so
# the loop's continue/break decision reflects the CURRENT attempt, not the
# initial send's classification. These cover the two transitions that a stale
# classification would get wrong.

class TestSendWithRetryFailureTypeTransitions:
    @pytest.mark.asyncio
    async def test_rate_limited_then_formatting_error_stops_retrying_and_falls_back(self):
        """A rate-limited first attempt followed by a PERMANENT formatting error
        on retry must stop retrying that permanent error and reach the existing
        plain-text fallback — not keep retrying it (as a stale, still-true
        is_rate_limited would)."""
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="flood control exceeded, retry in 60 seconds"),
            SendResult(success=False, error="Bad Request: can't parse entities"),
            # fallback send (auto-succeeds via _next_result)
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter._send_with_retry("chat1", "**bold**", max_retries=3, base_delay=0)
        # The formatting error was not retried further: exactly 1 retry then the
        # plain-text fallback. 1 initial + 1 retry + 1 fallback = 3 sends.
        assert len(adapter._send_calls) == 3
        # The permanent formatting error switched attempts to non-transient:
        # we fall through to (and return) the plain-text fallback.
        assert "plain text" in adapter._send_calls[-1][1].lower()
        assert result.success  # fallback succeeded
        # No delivery-failure notice was sent (this is a formatting fallback, not network exhaustion)
        assert "delivery failed" not in adapter._send_calls[-1][1].lower()
