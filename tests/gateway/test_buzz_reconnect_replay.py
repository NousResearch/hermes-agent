"""Reconnect replay, bounded cursor migration and keyset exhaustion regressions."""
import base64
import hashlib
import json
from collections import OrderedDict
from unittest.mock import AsyncMock

import pytest
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

mod = load_plugin_adapter("buzz")
CHANNEL = "ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd"
OTHER = "ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7ce"


def adapter():
    from gateway.config import PlatformConfig
    a = mod.BuzzAdapter(PlatformConfig(enabled=True, extra={"relay_url": "https://test.relay"}))
    a._self_pubkey = "11" * 32
    a._private_key = "00" * 31 + "03"
    a._resolve_user_name = AsyncMock(return_value="sender")
    a._dispatch_message = AsyncMock()
    a._channel_state[CHANNEL] = a._new_channel_state("dm")
    a._channel_state[CHANNEL]["_ws_eose"] = True
    return a


def event(i, stamp=10000, channel=CHANNEL):
    return {"id": f"{i:064x}", "created_at": stamp, "kind": 9, "pubkey": "22" * 32,
            "content": "hello", "tags": [["h", channel]]}


def paged(events, clamp):
    async def query(q):
        out = [e for e in events if q["since"] <= e["created_at"] <= q["until"]]
        if q.get("before_id"):
            out = [e for e in out if e["created_at"] < q["until"] or e["id"] > q["before_id"]]
        return sorted(out, key=lambda e: (-e["created_at"], e["id"]))[:clamp]
    return AsyncMock(side_effect=query)


@pytest.mark.asyncio
async def test_skewed_reconnect_recovers_behind_future_high_water():
    a = adapter(); state = a._channel_state[CHANNEL]
    await a._handle_events(CHANNEL, state, [event(1, 10900)])
    ws = type("WS", (), {"send": AsyncMock()})()
    await a._send_channel_subscription(ws, "live", CHANNEL)
    q = json.loads(ws.send.call_args.args[0])[2]
    assert q["since"] <= 9401  # subsequent valid T+301-900 arrival
    a._query_replay_page = paged([event(1, 10900), event(2, 9401)], 1000)
    await a._replay_channel(CHANNEL, state)
    assert {c.kwargs["message_id"] for c in a._dispatch_message.call_args_list} == {event(1)["id"], event(2)["id"]}


@pytest.mark.asyncio
async def test_more_than_500_seen_ids_survive_replay_and_restart():
    a = adapter(); state = a._channel_state[CHANNEL]
    events = [event(i) for i in range(1, 751)]
    await a._handle_events(CHANNEL, state, events)
    state["replay_floor"] = 8199
    a._save_cursors()
    b = adapter(); b._load_cursors(); assert b._restore_channel_state(CHANNEL, "dm")
    await b._handle_events(CHANNEL, b._channel_state[CHANNEL], events + [event(751)])
    assert b._dispatch_message.call_count == 1
    assert b._dispatch_message.call_args.kwargs["message_id"] == event(751)["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("clamp", [1000, 117])
async def test_pages_exhaust_even_with_smaller_clamp_and_same_second_ties(clamp):
    a = adapter(); state = a._channel_state[CHANNEL]
    state.update(last_ts=10000, replay_floor=8199)
    events = [event(i) for i in range(1, 1202)]
    a._query_replay_page = paged(events, clamp)
    await a._replay_channel(CHANNEL, state)
    assert a._dispatch_message.call_count == len(events)
    assert state.get("replay_floor") is None
    assert a._query_replay_page.call_count > len(events) // clamp


@pytest.mark.asyncio
async def test_midpage_failure_keeps_floor_and_resume_deduplicates():
    a = adapter(); state = a._channel_state[CHANNEL]
    state.update(last_ts=10000, replay_floor=8199)
    events = [event(i) for i in range(1, 601)]
    a._query_replay_page = AsyncMock(side_effect=[events[:300], OSError("disconnect")])
    with pytest.raises(OSError): await a._replay_channel(CHANNEL, state)
    assert state["replay_floor"] == 8199
    a._save_cursors()
    b = adapter(); b._load_cursors(); b._restore_channel_state(CHANNEL, "dm")
    b._query_replay_page = paged(events, 200)
    await b._replay_channel(CHANNEL, b._channel_state[CHANNEL])
    assert b._dispatch_message.call_count == 300


@pytest.mark.asyncio
async def test_ignored_pagination_is_detected_without_advancing_floor():
    a = adapter(); state = a._channel_state[CHANNEL]
    state.update(last_ts=10000, replay_floor=8199)
    a._query_replay_page = AsyncMock(return_value=[event(1)])
    with pytest.raises(ValueError, match="pagination"):
        await a._replay_channel(CHANNEL, state)
    assert state["replay_floor"] == 8199
    assert a._dispatch_message.call_count == 1


@pytest.mark.asyncio
async def test_capacity_quarantines_only_affected_channel(monkeypatch):
    monkeypatch.setattr(mod, "_REPLAY_SEEN_LIMIT", 2)
    a = adapter(); state = a._channel_state[CHANNEL]
    state["replay_floor"] = 8199
    await a._handle_events(CHANNEL, state, [event(1), event(2), event(3)])
    assert state["replay_blocked"] and state["replay_floor"] == 8199
    assert event(3)["id"] not in state["seen"]
    a._channel_state[OTHER] = a._new_channel_state("dm")
    await a._handle_events(OTHER, a._channel_state[OTHER], [event(4, channel=OTHER)])
    assert a._dispatch_message.call_count == 3
    a._save_cursors()
    b = adapter(); b._load_cursors(); b._restore_channel_state(CHANNEL, "dm")
    assert b._channel_state[CHANNEL]["replay_blocked"]


@pytest.mark.asyncio
async def test_page_budget_quarantines_with_floor_retained(monkeypatch):
    monkeypatch.setattr(mod, "_REPLAY_MAX_PAGES", 1)
    a = adapter(); state = a._channel_state[CHANNEL]
    state.update(last_ts=10000, replay_floor=8199)
    a._query_replay_page = paged([event(1), event(2)], 1)
    await a._replay_channel(CHANNEL, state)
    assert state["replay_blocked"] and state["replay_floor"] == 8199


def test_pruning_uses_event_time_and_keeps_entire_pinned_window():
    a = adapter(); s = a._channel_state[CHANNEL]
    s.update(last_ts=12000, replay_floor=9000, seen=OrderedDict([("old", 8999), ("inside", 9100), ("new", 12000)]))
    a._trim_seen(s); assert list(s["seen"]) == ["inside", "new"]
    s.pop("replay_floor"); a._trim_seen(s); assert list(s["seen"]) == ["new"]


@pytest.mark.asyncio
async def test_http_page_signs_exact_body(monkeypatch):
    import httpx
    a = adapter(); a._auth_tag = '["auth","owner","","signature"]'
    real_client = httpx.AsyncClient
    async def serve(request):
        raw = base64.b64decode(request.headers["authorization"].removeprefix("Nostr "))
        auth = json.loads(raw)
        assert auth["kind"] == 27235
        assert ["u", str(request.url)] in auth["tags"]
        assert ["payload", hashlib.sha256(request.content).hexdigest()] in auth["tags"]
        assert request.headers["x-auth-tag"] == a._auth_tag
        expected = hashlib.sha256(json.dumps([0, auth["pubkey"], auth["created_at"], auth["kind"], auth["tags"], ""], separators=(",", ":")).encode()).hexdigest()
        assert auth["id"] == expected and len(auth["sig"]) == 128
        return httpx.Response(200, json=[event(1)])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(serve), **kwargs))
    assert await a._query_replay_page({"#h": [CHANNEL]}) == [event(1)]

@pytest.mark.asyncio
async def test_http_exhaustion_waits_for_ws_eose_before_pruning():
    a = adapter(); state = a._channel_state[CHANNEL]
    state.update(last_ts=10000, replay_floor=8199, _ws_eose=False)
    a._query_replay_page = paged([event(1, 8200), event(2, 12000)], 1000)
    await a._replay_channel(CHANNEL, state)
    assert state["replay_floor"] == 8199 and event(1)["id"] in state["seen"]
    # A buffered WS historical event cannot become a second delivery.
    await a._handle_events(CHANNEL, state, [event(1, 8200)])
    assert a._dispatch_message.call_count == 2
    await a._handle_ws_message(None, {"live": CHANNEL}, ["EOSE", "live"])
    assert state.get("replay_floor") is None and event(1)["id"] not in state["seen"]


@pytest.mark.asyncio
async def test_http_redirect_is_not_followed(monkeypatch):
    import httpx
    a = adapter()
    real_client = httpx.AsyncClient
    requests = []
    async def serve(request):
        requests.append(str(request.url))
        return httpx.Response(307, headers={"Location": "https://other.invalid/query"})
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(serve), **kwargs))
    with pytest.raises(httpx.HTTPStatusError):
        await a._query_replay_page({"#h": [CHANNEL]})
    assert requests == ["https://test.relay/query"]


@pytest.mark.asyncio
async def test_membership_removal_during_http_wait_prevents_dispatch():
    a = adapter(); state = a._channel_state[CHANNEL]
    state.update(last_ts=10000, replay_floor=8199)
    async def query(q):
        a._channel_state.pop(CHANNEL)
        a._restricted_channels.add(CHANNEL)
        return [event(1)]
    a._query_replay_page = query
    await a._replay_channel(CHANNEL, state)
    assert a._dispatch_message.call_count == 0
    assert state["replay_floor"] == 8199


@pytest.mark.asyncio
async def test_replay_accepts_extended_h_tag():
    a = adapter(); state = a._channel_state[CHANNEL]
    state.update(last_ts=10000, replay_floor=8199)
    e = event(1); e["tags"] = [["h", CHANNEL, "relay hint"]]
    a._query_replay_page = paged([e], 1000)
    await a._replay_channel(CHANNEL, state)
    assert a._dispatch_message.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [None, {}, {"id": "00" * 32, "created_at": "oops"}])
async def test_invalid_page_event_raises_value_error(bad):
    a = adapter(); state = a._channel_state[CHANNEL]
    state.update(last_ts=10000, replay_floor=8199)
    a._query_replay_page = AsyncMock(return_value=[bad])
    with pytest.raises(ValueError, match="invalid replay event"):
        await a._replay_channel(CHANNEL, state)
    assert state["replay_floor"] == 8199


@pytest.mark.asyncio
async def test_retry_budget_survives_replay_task_restart(monkeypatch):
    import asyncio
    a = adapter(); state = a._channel_state[CHANNEL]
    state.update(last_ts=10000, replay_floor=8199)
    a._query_replay_page = AsyncMock(side_effect=OSError("network"))
    monkeypatch.setattr(mod.asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError))
    for attempt in range(3):
        with pytest.raises(asyncio.CancelledError):
            await a._ws_replay_loop()
        assert state["_replay_failures"] == attempt + 1
    assert state["replay_blocked"] and state["replay_floor"] == 8199


def test_replay_requires_explicit_ws_eose():
    a = adapter(); state = a._channel_state[CHANNEL]
    state.pop("_ws_eose")
    state.update(replay_floor=8199, _replay_http_complete=True)
    a._finish_replay(state)
    assert state["replay_floor"] == 8199
