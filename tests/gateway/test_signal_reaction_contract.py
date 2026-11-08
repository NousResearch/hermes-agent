"""Signal's native reaction-target contract.

Signal has no opaque message id: a reaction names ``(author, ms timestamp)``. An inbound event must
carry that as the composite ``"<author>:<timestamp_ms>"`` so ``add_reaction``/``remove_reaction``
round-trip it into a ``sendReaction`` RPC naming ``targetAuthor``/``targetTimestamp`` (plus
``remove`` when retracting).
"""

import pytest

from gateway.config import PlatformConfig

SENDER = "+15559876543"
TS_MS = 1700000000000


@pytest.fixture
def adapter(monkeypatch):
    """A SignalAdapter with the minimal config the envelope path reads."""
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
    from plugins.platforms.signal.adapter import SignalAdapter

    config = PlatformConfig()
    config.enabled = True
    config.extra = {"http_url": "http://localhost:8080", "account": "+15551234567"}
    return SignalAdapter(config)


def _envelope(ts=TS_MS, text="hello"):
    """A minimal text DM envelope; ``ts=None`` omits the timestamp field entirely."""
    data = {
        "sourceNumber": SENDER,
        "sourceName": "Test User",
        "sourceUuid": "aaaaaaaa-0000-0000-0000-000000000001",
        "dataMessage": {"message": text, "attachments": []},
    }
    if ts is not None:
        data["timestamp"] = ts
        data["dataMessage"]["timestamp"] = ts
    return {"envelope": data}


async def _dispatch(adapter, envelope):
    """Run ``_handle_envelope`` and return the MessageEvent it handed to handle_message."""
    events = []

    async def _capture(event):
        events.append(event)

    adapter.handle_message = _capture
    await adapter._handle_envelope(envelope)
    assert events, "no event dispatched"
    return events[0]


def _record_rpc(adapter):
    """Swap ``_rpc`` for a recorder of ``(method, params)`` calls; a non-None result reads as ok."""
    calls = []

    async def _rpc(method, params, rpc_id=None, **kwargs):
        calls.append((method, dict(params)))
        return {"timestamp": 1}

    adapter._rpc = _rpc
    return calls


@pytest.mark.asyncio
async def test_inbound_event_exposes_composite_reaction_target(adapter):
    event = await _dispatch(adapter, _envelope())
    assert event.message_id == f"{SENDER}:{TS_MS}"
    assert event.source.message_id == f"{SENDER}:{TS_MS}"


@pytest.mark.asyncio
@pytest.mark.parametrize("ts", [0, None])
async def test_inbound_event_without_usable_timestamp_has_no_target(adapter, ts):
    event = await _dispatch(adapter, _envelope(ts=ts))
    assert event.message_id is None
    assert event.source.message_id is None


@pytest.mark.asyncio
async def test_add_reaction_round_trips_event_target_into_rpc(adapter):
    event = await _dispatch(adapter, _envelope())
    calls = _record_rpc(adapter)

    res = await adapter.add_reaction(event.source.chat_id, "👍", message_id=event.message_id)

    assert res["success"] is True
    method, params = calls[-1]
    assert method == "sendReaction"
    assert params["targetAuthor"] == SENDER
    assert params["targetTimestamp"] == TS_MS
    assert params["emoji"] == "👍"
    assert not params.get("remove")


@pytest.mark.asyncio
async def test_remove_reaction_round_trips_event_target_into_rpc(adapter):
    event = await _dispatch(adapter, _envelope())
    calls = _record_rpc(adapter)

    res = await adapter.remove_reaction(event.source.chat_id, message_id=event.message_id)

    assert res["success"] is True
    method, params = calls[-1]
    assert method == "sendReaction"
    assert params["targetAuthor"] == SENDER
    assert params["targetTimestamp"] == TS_MS
    assert params["remove"] is True
    assert params["emoji"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message_id",
    ["author:notanumber", f"{SENDER}:0", f"{SENDER}:-5", "no-colon", None],
)
async def test_add_reaction_rejects_unusable_target_without_rpc(adapter, message_id):
    calls = _record_rpc(adapter)

    res = await adapter.add_reaction(SENDER, "👍", message_id=message_id)

    assert res["success"] is False
    assert "message_id" in res["error"]
    assert calls == []
