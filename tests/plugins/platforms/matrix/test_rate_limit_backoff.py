"""Matrix homeserver rate limiting (429/M_LIMIT_EXCEEDED) must be retried with backoff (#126493).

Approval-gated commands burst ``send``/``redact``/reaction calls; on a public homeserver
(matrix.org) that trip-wires ``M_LIMIT_EXCEEDED`` and the adapter used to fail the send
outright, so the reaction-based approval prompt never reached the user and the fallback
plain-text message raced the 60s approval timeout. Every outbound path used by a prompt
(``_send_room_message``, ``_send_reaction``, ``redact_message``) now retries 429s with
capped exponential backoff; other Matrix errors still fail immediately."""

import asyncio

import pytest

pytest.importorskip("mautrix.errors.request")  # matrix extra is linux-only; stubs cover CI runs
from mautrix.errors.request import MForbidden, MLimitExceeded, MatrixStandardRequestError

from gateway.config import PlatformConfig


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter
    adapter = MatrixAdapter(PlatformConfig(enabled=True, extra={}))
    return adapter


class _FlakyClient:
    """Fail the first ``failures`` calls of each method with ``exc_factory()``; then succeed."""

    def __init__(self, failures, exc_factory):
        self.failures = failures
        self.exc_factory = exc_factory
        self.send_calls = 0
        self.redact_calls = 0

    async def send_message_event(self, room_id, event_type, content):
        self.send_calls += 1
        if self.send_calls <= self.failures:
            raise self.exc_factory()
        return "$sent"

    async def redact(self, room_id, event_id, reason=None):
        self.redact_calls += 1
        if self.redact_calls <= self.failures:
            raise self.exc_factory()
        return "$redacted"


def _limit_exceeded():
    return MLimitExceeded(http_status=429, message="Too Many Requests")


@pytest.fixture
def backoff_delays(monkeypatch):
    delays = []

    async def _fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    return delays


@pytest.mark.asyncio
async def test_send_retries_rate_limit_and_succeeds(backoff_delays):
    adapter = _make_adapter()
    adapter._client = _FlakyClient(2, _limit_exceeded)
    result = await adapter.send("!room:example.org", "hello")
    assert result.success is True
    assert result.message_id == "$sent"
    assert adapter._client.send_calls == 3
    assert backoff_delays == [1.5, 3.0]


@pytest.mark.asyncio
async def test_send_gives_up_after_final_retry(backoff_delays):
    adapter = _make_adapter()
    adapter._client = _FlakyClient(99, _limit_exceeded)
    result = await adapter.send("!room:example.org", "hello")
    assert result.success is False
    assert "Too Many Requests" in result.error
    assert adapter._client.send_calls == 4  # initial attempt + 3 retries
    assert backoff_delays == [1.5, 3.0, 6.0]


@pytest.mark.asyncio
async def test_send_does_not_retry_other_matrix_errors(backoff_delays):
    adapter = _make_adapter()
    adapter._client = _FlakyClient(99, lambda: MForbidden(http_status=403, message="not for you"))
    result = await adapter.send("!room:example.org", "hello")
    assert result.success is False
    assert adapter._client.send_calls == 1
    assert backoff_delays == []


@pytest.mark.asyncio
async def test_plain_429_status_without_errcode_is_retried(backoff_delays):
    """Homeservers may answer 429 with an errcode mautrix does not map to MLimitExceeded."""
    adapter = _make_adapter()
    adapter._client = _FlakyClient(1, lambda: MatrixStandardRequestError(429, "Too Many Requests"))
    result = await adapter.send("!room:example.org", "hello")
    assert result.success is True
    assert adapter._client.send_calls == 2


@pytest.mark.asyncio
async def test_redact_retries_rate_limit(backoff_delays):
    adapter = _make_adapter()
    adapter._client = _FlakyClient(1, _limit_exceeded)
    assert await adapter.redact_message("!room:example.org", "$evt") is True
    assert adapter._client.redact_calls == 2
    assert backoff_delays == [1.5]


@pytest.mark.asyncio
async def test_reaction_send_retries_rate_limit(backoff_delays):
    adapter = _make_adapter()
    adapter._client = _FlakyClient(1, _limit_exceeded)
    assert await adapter._send_reaction("!room:example.org", "$evt", "\u2705") == "$sent"
    assert adapter._client.send_calls == 2
    assert backoff_delays == [1.5]
