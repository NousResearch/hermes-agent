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

    async def connect(self) -> bool:
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

    @pytest.mark.parametrize("pattern", _RETRYABLE_ERROR_PATTERNS)
    def test_known_pattern_is_retryable(self, pattern):
        assert _StubAdapter._is_retryable_error(f"httpx.{pattern.title()}: connection dropped")

    def test_permission_error_not_retryable(self):
        assert not _StubAdapter._is_retryable_error("Forbidden: bot was blocked by the user")

    def test_bad_request_not_retryable(self):
        assert not _StubAdapter._is_retryable_error("Bad Request: can't parse entities")

    def test_case_insensitive(self):
        assert _StubAdapter._is_retryable_error("CONNECTERROR: host unreachable")


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

    @pytest.mark.asyncio
    async def test_returns_message_id(self):
        adapter = _StubAdapter()
        adapter._send_results = [SendResult(success=True, message_id="abc")]
        result = await adapter._send_with_retry("chat1", "hi")
        assert result.message_id == "abc"


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
    async def test_retries_on_timeout_and_succeeds(self):
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="ReadTimeout: request timed out"),
            SendResult(success=False, error="ReadTimeout: request timed out"),
            SendResult(success=True, message_id="ok"),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._send_with_retry("chat1", "hello", max_retries=3, base_delay=0)
        assert result.success
        assert len(adapter._send_calls) == 3

    @pytest.mark.asyncio
    async def test_retryable_flag_respected(self):
        """SendResult.retryable=True should trigger retry even if error string doesn't match."""
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="internal platform error", retryable=True),
            SendResult(success=True, message_id="ok"),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._send_with_retry("chat1", "hello", max_retries=2, base_delay=0)
        assert result.success
        assert len(adapter._send_calls) == 2

    @pytest.mark.asyncio
    async def test_network_to_nonnetwork_transition_falls_back_to_plaintext(self):
        """If error switches from network to formatting mid-retry, fall through to plain-text fallback."""
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="httpx.ConnectError: host unreachable"),
            SendResult(success=False, error="Bad Request: can't parse entities"),
            SendResult(success=True, message_id="fallback_ok"),  # plain-text fallback
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._send_with_retry("chat1", "**bold**", max_retries=2, base_delay=0)
        assert result.success
        # 3 calls: initial (network) + 1 retry (non-network, breaks loop) + plain-text fallback
        assert len(adapter._send_calls) == 3
        assert "plain text" in adapter._send_calls[-1][1].lower()


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

    @pytest.mark.asyncio
    async def test_notice_send_exception_doesnt_propagate(self):
        """If the notice itself throws, _send_with_retry should not raise."""
        adapter = _StubAdapter()
        network_err = SendResult(success=False, error="ConnectError")
        adapter._send_results = [network_err, network_err, network_err]

        original_send = adapter.send
        call_count = [0]

        async def send_with_notice_failure(chat_id, content, **kwargs):
            call_count[0] += 1
            if call_count[0] > 3:
                raise RuntimeError("notice send also failed")
            return network_err

        adapter.send = send_with_notice_failure
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._send_with_retry("chat1", "hello", max_retries=2, base_delay=0)
        assert not result.success  # still failed, but no exception raised


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

    @pytest.mark.asyncio
    async def test_fallback_failure_logged_but_not_raised(self):
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="Forbidden: bot blocked"),
            SendResult(success=False, error="Forbidden: bot blocked"),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._send_with_retry("chat1", "hello", max_retries=2)
        assert not result.success
        assert len(adapter._send_calls) == 2  # original + fallback only
