"""Matrix homeserver rate limiting (429/M_LIMIT_EXCEEDED) must be retried with backoff (#126493).

Approval-gated commands burst ``send``/``redact``/reaction calls; on a public homeserver
(matrix.org) that trip-wires ``M_LIMIT_EXCEEDED`` and the adapter used to fail the send
outright, so the reaction-based approval prompt never reached the user and the fallback
plain-text message raced the 60s approval timeout. Every outbound path used by a prompt
(``_send_room_message``, ``_send_reaction``, ``redact_message``) now retries 429s with
capped exponential backoff; other Matrix errors still fail immediately."""

import asyncio

import pytest
from mautrix.errors.request import MForbidden, MLimitExceeded, MatrixStandardRequestError

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult


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
@pytest.mark.parametrize("operation, failures, error, expected", [
    ("send", 2, "limit", (True, "$sent", 3, 0, [1.5, 3.0])),
    ("send", 99, "limit", (False, None, 4, 0, [1.5, 3.0, 6.0])),
    ("send", 99, "forbidden", (False, None, 1, 0, [])),
    ("send", 1, "plain429", (True, "$sent", 2, 0, [1.5])),
    ("redact", 1, "limit", (True, None, 0, 2, [1.5])),
    ("reaction", 1, "limit", (True, "$sent", 2, 0, [1.5])),
])
async def test_rate_limited_outbound_paths(operation, failures, error, expected, backoff_delays):
    errors = {
        "limit": _limit_exceeded,
        "forbidden": lambda: MForbidden(http_status=403, message="not for you"),
        "plain429": lambda: MatrixStandardRequestError(429, "Too Many Requests"),
    }
    adapter = _make_adapter()
    adapter._client = _FlakyClient(failures, errors[error])
    if operation == "send":
        result = await adapter.send("!room:example.org", "hello")
        expected_result = SendResult(
            success=expected[0], message_id=expected[1],
            error=f"Matrix target '!room:example.org': {errors[error]()}" if not expected[0] else None,
        )
    elif operation == "redact":
        result = await adapter.redact_message("!room:example.org", "$evt")
        expected_result = expected[0]
    else:
        result = await adapter._send_reaction("!room:example.org", "$evt", "\u2705")
        expected_result = expected[1]
    assert (result, adapter._client.send_calls, adapter._client.redact_calls, backoff_delays) == (
        expected_result, *expected[2:],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("msgtype", ["m.image", "m.file", "m.audio", "m.video"])
@pytest.mark.parametrize("error, failures, attempts, delays, success", [
    ("limit", 1, 2, [1.5], True),
    ("plain429", 1, 2, [1.5], True),
    ("limit", 99, 4, [1.5, 3.0, 6.0], False),
    ("forbidden", 1, 1, [], False),
    ("unknown", 1, 1, [], False),
])
async def test_media_upload_retries_only_rate_limited_content(msgtype, error, failures, attempts, delays, success, backoff_delays):
    from copy import deepcopy
    from unittest.mock import AsyncMock

    errors = {
        "limit": _limit_exceeded,
        "plain429": lambda: MatrixStandardRequestError(429, "Too Many Requests"),
        "forbidden": lambda: MForbidden(http_status=403, message="not for you"),
        "unknown": lambda: RuntimeError("connection lost"),
    }
    adapter = _make_adapter()
    adapter._client = _FlakyClient(failures, errors[error])
    adapter._client.upload_media = AsyncMock(return_value="mxc://example.org/upload")
    payloads = []
    original_send = adapter._client.send_message_event

    async def record_send(room_id, event_type, content):
        payloads.append((str(room_id), str(event_type), deepcopy(content)))
        return await original_send(room_id, event_type, content)

    adapter._client.send_message_event = record_send
    result = await adapter._upload_and_send(
        "!room:example.org", b"content", "original.dat", "application/octet-stream", msgtype,
        caption="Caption", reply_to="$reply", metadata={"thread_id": "$thread"},
    )
    expected_payload = ("!room:example.org", "m.room.message", {
        "msgtype": msgtype, "body": "Caption", "filename": "original.dat",
        "info": {"mimetype": "application/octet-stream", "size": 7},
        "url": "mxc://example.org/upload", "m.relates_to": {
            "rel_type": "m.thread", "event_id": "$thread", "is_falling_back": False,
            "m.in_reply_to": {"event_id": "$reply"},
        },
    })
    assert (result, adapter._client.send_calls, backoff_delays, payloads) == (
        SendResult(success=success, message_id="$sent" if success else None,
                   error=f"Matrix target '!room:example.org': {errors[error]()}" if not success else None,
                   error_kind={"limit": "rate_limited", "forbidden": "forbidden", "unknown": "unknown"}.get(error) if not success else None),
        attempts, delays, [expected_payload] * attempts,
    )
    adapter._client.upload_media.assert_awaited_once_with(
        b"content", mime_type="application/octet-stream", filename="original.dat", size=7,
    )
