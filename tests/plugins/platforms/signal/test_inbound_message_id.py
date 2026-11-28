"""A Signal ``MessageEvent`` carries the message's send timestamp as its ``message_id``.

Signal identifies a message by its send timestamp; a quote names the message it replies to by
that same value (``quote.id``). The adapter kept it only in ``raw_message``, so every gateway path
keyed on ``event.message_id`` (dedupe of a retried turn's user row, the turn's owner identity,
busy-path tracking) treated each Signal message as id-less.
"""
import pytest

from gateway.config import PlatformConfig

SENT_MS = 1793738280588


@pytest.fixture(autouse=True)
def _reset_signal_scheduler():
    from plugins.platforms.signal.signal_rate_limit import _reset_scheduler
    _reset_scheduler()
    yield
    _reset_scheduler()


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
    from plugins.platforms.signal.adapter import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {"http_url": "http://localhost:8080", "account": "+155****4567"}
    result = SignalAdapter(config)
    result.set_authorization_check(lambda *args, **kwargs: True)
    return result


async def _event_for(adapter, **envelope):
    captured = []

    async def fake_handle(event):
        captured.append(event)

    adapter.handle_message = fake_handle
    await adapter._handle_envelope({"envelope": {
        "sourceNumber": "+155****1111", "sourceUuid": "uuid-sender", "sourceName": "Tester",
        **envelope,
    }})
    assert len(captured) == 1
    return captured[0]


@pytest.mark.asyncio
async def test_the_event_carries_its_send_time_as_its_message_id(adapter):
    event = await _event_for(adapter, timestamp=SENT_MS, dataMessage={"message": "hello"})
    assert event.message_id == str(SENT_MS)
    assert event.raw_message["timestamp_ms"] == SENT_MS


@pytest.mark.asyncio
async def test_a_reply_names_the_quoted_message_by_the_same_id(adapter):
    first = await _event_for(adapter, timestamp=SENT_MS, dataMessage={"message": "lunch?"})
    reply = await _event_for(adapter, timestamp=SENT_MS + 5000, dataMessage={
        "message": "yes", "quote": {"id": SENT_MS, "text": "lunch?", "author": "+155****1111"}})
    assert reply.message_id == str(SENT_MS + 5000)
    assert reply.reply_to_message_id == first.message_id


@pytest.mark.asyncio
async def test_an_envelope_without_a_timestamp_has_no_message_id(adapter):
    event = await _event_for(adapter, dataMessage={"message": "no send time on this one"})
    assert event.message_id is None
