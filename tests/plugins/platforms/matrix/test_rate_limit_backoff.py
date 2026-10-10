"""Rate-limited Matrix writes retry without changing payloads (#126493)."""

import asyncio
from copy import deepcopy

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
        self.send_payloads = []
        self.redact_payloads = []

    async def send_message_event(self, room_id, event_type, content):
        self.send_calls += 1
        self.send_payloads.append((str(room_id), str(event_type), deepcopy(content)))
        if self.send_calls <= self.failures:
            raise self.exc_factory()
        return "$sent"

    async def redact(self, room_id, event_id, reason=None):
        self.redact_calls += 1
        self.redact_payloads.append((str(room_id), str(event_id), reason))
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
@pytest.mark.parametrize("operation", ["send", "reaction", "redact"])
@pytest.mark.parametrize("error, failures, attempts, delays, success", [
    ("limit", 2, 3, [1.5, 3.0], True),
    ("plain429", 1, 2, [1.5], True),
    ("limit", 99, 4, [1.5, 3.0, 6.0], False),
    ("forbidden", 1, 1, [], False),
    ("unknown", 1, 1, [], False),
    ("cancelled", 1, 1, [], False),
])
async def test_rate_limited_outbound_paths(operation, error, failures, attempts, delays, success, backoff_delays):
    errors = {
        "limit": _limit_exceeded,
        "plain429": lambda: MatrixStandardRequestError(429, "Too Many Requests"),
        "forbidden": lambda: MForbidden(http_status=403, message="not for you"),
        "unknown": lambda: RuntimeError("connection lost"),
        "cancelled": asyncio.CancelledError,
    }
    adapter = _make_adapter()
    adapter._client = _FlakyClient(failures, errors[error])
    calls = {
        "send": lambda: adapter.send("!room:example.org", "hello"),
        "reaction": lambda: adapter._send_reaction("!room:example.org", "$evt", "\u2705"),
        "redact": lambda: adapter.redact_message("!room:example.org", "$evt"),
    }
    if error == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await calls[operation]()
        result = None
    else:
        result = await calls[operation]()
    expected_results = {
        "send": SendResult(success=success, message_id="$sent" if success else None,
                           error=f"Matrix target '!room:example.org': {errors[error]()}" if not success else None),
        "reaction": "$sent" if success else None,
        "redact": success,
    }
    expected_payloads = {
        "send": ("!room:example.org", "m.room.message", {"msgtype": "m.text", "body": "hello"}),
        "reaction": ("!room:example.org", "m.reaction", {
            "m.relates_to": {"rel_type": "m.annotation", "event_id": "$evt", "key": "\u2705"},
        }),
    }
    assert (
        result, adapter._client.send_calls, adapter._client.redact_calls, backoff_delays,
        adapter._client.send_payloads, adapter._client.redact_payloads,
    ) == (
        None if error == "cancelled" else expected_results[operation],
        0 if operation == "redact" else attempts,
        attempts if operation == "redact" else 0,
        delays,
        [] if operation == "redact" else [expected_payloads[operation]] * attempts,
        [("!room:example.org", "$evt", None)] * attempts if operation == "redact" else [],
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
