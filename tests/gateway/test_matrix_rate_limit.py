"""Matrix M_LIMIT_EXCEEDED (429) backoff — issue #126493.

Outgoing send/redact calls must honor the homeserver's ``retry_after_ms``
and retry once instead of failing immediately.
"""
import asyncio

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from gateway.config import PlatformConfig


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter
    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"},
    )
    return MatrixAdapter(config)


class _MLimitExceeded(Exception):
    """Duck-typed mautrix MLimitExceeded."""

    def __init__(self, retry_after_ms=1500):
        super().__init__("M_LIMIT_EXCEEDED: Too Many Requests")
        self.errcode = "M_LIMIT_EXCEEDED"
        self.retry_after_ms = retry_after_ms
        self.status_code = 429


def _sleep_recorder(sleeps):
    async def fake_sleep(d):
        sleeps.append(d)
    return fake_sleep


class TestMatrixRateLimitBackoff:
    @pytest.mark.asyncio
    async def test_send_retries_on_429_honoring_retry_after_ms(self):
        adapter = _make_adapter()
        fake_client = MagicMock()
        fake_client.send_message_event = AsyncMock(side_effect=[_MLimitExceeded(1500), "$evt2"])
        adapter._client = fake_client
        sleeps = []
        with patch("asyncio.sleep", side_effect=_sleep_recorder(sleeps)):
            result = await adapter.send("!room:example.org", "hello")
        assert result.success is True
        assert result.message_id == "$evt2"
        assert fake_client.send_message_event.await_count == 2
        assert sleeps and sleeps[0] >= 1.5

    @pytest.mark.asyncio
    async def test_send_marks_persistent_rate_limit_distinctly(self):
        adapter = _make_adapter()
        fake_client = MagicMock()
        fake_client.send_message_event = AsyncMock(side_effect=[_MLimitExceeded(100), _MLimitExceeded(100)])
        adapter._client = fake_client
        with patch("asyncio.sleep", side_effect=_sleep_recorder([])):
            result = await adapter.send("!room:example.org", "hello")
        assert result.success is False
        assert result.error_kind == "rate_limited"
        assert result.retry_after is not None and result.retry_after >= 0.1
        assert result.retryable is True

    @pytest.mark.asyncio
    async def test_send_does_not_retry_ordinary_errors(self):
        adapter = _make_adapter()
        fake_client = MagicMock()
        fake_client.send_message_event = AsyncMock(side_effect=Exception("boom"))
        adapter._client = fake_client
        sleeps = []
        with patch("asyncio.sleep", side_effect=_sleep_recorder(sleeps)):
            result = await adapter.send("!room:example.org", "hello")
        assert result.success is False
        assert fake_client.send_message_event.await_count == 1
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_redact_retries_on_429(self):
        adapter = _make_adapter()
        fake_client = MagicMock()
        fake_client.redact = AsyncMock(side_effect=[_MLimitExceeded(500), "$redact"])
        adapter._client = fake_client
        sleeps = []
        with patch("asyncio.sleep", side_effect=_sleep_recorder(sleeps)):
            assert await adapter.redact_message("!room:example.org", "$ev1") is True
        assert fake_client.redact.await_count == 2
        assert sleeps and sleeps[0] >= 0.5

    @pytest.mark.asyncio
    async def test_redact_does_not_retry_ordinary_errors(self):
        adapter = _make_adapter()
        fake_client = MagicMock()
        fake_client.redact = AsyncMock(side_effect=Exception("boom"))
        adapter._client = fake_client
        with patch("asyncio.sleep", side_effect=_sleep_recorder([])):
            assert await adapter.redact_message("!room:example.org", "$ev1") is False
        assert fake_client.redact.await_count == 1

    @pytest.mark.asyncio
    async def test_send_retry_failing_permanently_is_not_rate_limited(self):
        """1st attempt 429s, the retry raises a permanent error: report THAT, not the 429."""
        adapter = _make_adapter()
        fake_client = MagicMock()
        fake_client.send_message_event = AsyncMock(side_effect=[
            _MLimitExceeded(5000),
            Exception("M_FORBIDDEN: forbidden — you do not have permission to post in this room")])
        adapter._client = fake_client
        with patch("asyncio.sleep", side_effect=_sleep_recorder([])):
            result = await adapter.send("!room:example.org", "hello")
        assert result.success is False
        assert result.error_kind == "forbidden"
        assert result.retryable is False
        assert result.retry_after is None

    @pytest.mark.asyncio
    async def test_long_penalty_stays_bounded_and_reaches_the_ledger(self):
        """A 97-minute penalty (Synapse's 429 ceiling) must not pin the send coroutine: the
        inline wait is capped below base's 60s guard, and the unclamped delay rides the
        failure out so the delivery ledger owns the wait."""
        adapter = _make_adapter()
        fake_client = MagicMock()
        fake_client.send_message_event = AsyncMock(
            side_effect=[_MLimitExceeded(5_820_000), _MLimitExceeded(5_820_000)])
        adapter._client = fake_client
        sleeps = []
        with patch("asyncio.sleep", side_effect=_sleep_recorder(sleeps)):
            result = await adapter._send_with_retry("!room:example.org", "hello")
        assert result.success is False
        assert result.error_kind == "rate_limited"
        assert result.retry_after == 5_820.0  # > base's 60s inline cap, so the guard fires
        assert fake_client.send_message_event.await_count == 2  # one inline retry, then hand off
        assert sleeps and max(sleeps) <= 46.0

    @pytest.mark.asyncio
    async def test_redact_long_penalty_is_inline_bounded(self):
        adapter = _make_adapter()
        fake_client = MagicMock()
        fake_client.redact = AsyncMock(side_effect=[_MLimitExceeded(5_820_000), None])
        adapter._client = fake_client
        sleeps = []
        with patch("asyncio.sleep", side_effect=_sleep_recorder(sleeps)):
            assert await adapter.redact_message("!room:example.org", "$ev1") is True
        assert sleeps and max(sleeps) <= 46.0

    @pytest.mark.asyncio
    async def test_backoff_waits_are_jittered(self):
        """A shared limit must not resume every client in lockstep: both inline waits carry
        the same jitter base.py adds (``random.uniform(0, 1)``)."""
        adapter = _make_adapter()
        fake_client = MagicMock()
        fake_client.send_message_event = AsyncMock(side_effect=[_MLimitExceeded(1500), "$evt2"])
        adapter._client = fake_client
        sends, redacts = [], []
        with patch("asyncio.sleep", side_effect=_sleep_recorder(sends)), \
                patch("plugins.platforms.matrix.adapter.random.uniform", return_value=0.25):
            await adapter.send("!room:example.org", "hello")
        fake_client.redact = AsyncMock(side_effect=[_MLimitExceeded(500), None])
        with patch("asyncio.sleep", side_effect=_sleep_recorder(redacts)), \
                patch("plugins.platforms.matrix.adapter.random.uniform", return_value=0.25):
            assert await adapter.redact_message("!room:example.org", "$ev1") is True
        assert sends == [1.75]
        assert redacts == [0.75]


class TestMatrixRateLimitDelayHelper:
    def test_ordinary_error_is_none(self):
        from plugins.platforms.matrix.adapter import _matrix_rate_limit_delay_seconds
        assert _matrix_rate_limit_delay_seconds(Exception("boom")) is None

    def test_retry_after_ms_scaled_unclamped(self):
        from plugins.platforms.matrix.adapter import _matrix_rate_limit_delay_seconds
        assert _matrix_rate_limit_delay_seconds(_MLimitExceeded(1500)) == 1.5
        # A 97-minute penalty must survive the helper: callers bound the sleep, not the value,
        # so base's inline-wait guard can still see the real delay.
        assert _matrix_rate_limit_delay_seconds(_MLimitExceeded(10**9)) == 1_000_000.0

    def test_limit_without_value_defaults_to_1s(self):
        from plugins.platforms.matrix.adapter import _matrix_rate_limit_delay_seconds
        assert _matrix_rate_limit_delay_seconds(_MLimitExceeded(None)) == 1.0

    def test_status_429_without_errcode(self):
        from plugins.platforms.matrix.adapter import _matrix_rate_limit_delay_seconds
        exc = Exception("Too Many Requests")
        exc.status_code = 429
        assert _matrix_rate_limit_delay_seconds(exc) == 1.0
