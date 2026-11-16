"""``client.hello`` — the post-upgrade handshake folded into ONE round trip (``gateway/session_hello``).

The authority serves it because the replay it batches is owner state: every window comes from the
connection's own ``session.events.since``, so ``truncated`` / ``snapshot_required`` / epoch rules and
the reconnect bookkeeping behind them are unchanged. These tests pin the FOLD: one answer carrying the
capabilities result, the skin etag the ready frame carried, one window per watermark and the deduped
open requests — and a per-session refusal that costs only that session's window.
"""
from types import SimpleNamespace

import pytest

from gateway import session_hello
from hermes_state_runtime import RuntimeStoreError
from tui_gateway.contracts.registry import CANONICAL_METHODS


class _Transport:
    """What the fold reads and advertises on: the skin etag the ready frame carried."""

    def __init__(self, skin_etag="etag-1"):
        self.skin_etag = skin_etag


class _Connection:
    """An authority connection with a canned ``events_since``: the only replay the fold may use."""

    def __init__(self, windows):
        self.transport = _Transport()
        self.actor = SimpleNamespace(profile_id="owned")
        self.calls = []
        self._windows = windows

    async def events_since(self, ref, params):
        self.calls.append((ref.session_id, dict(params)))
        window = self._windows.get(ref.session_id)
        if window is None:
            raise RuntimeStoreError("not_found")
        return window


def _window(events=(), latest=0, truncated=False, open_requests=()):
    return {"events": list(events), "latest_seq": latest, "last_sequence": latest, "epoch": "proc-1",
            "replay_epoch": "proc-1", "truncated": truncated, "snapshot_required": truncated,
            "count": len(events), "open_requests": list(open_requests)}


@pytest.mark.asyncio
async def test_the_fold_answers_every_watermark_capabilities_and_the_skin_etag_in_one_result():
    frame = {"type": "message.complete", "session_id": "s1", "seq": 6, "payload": {}, "replay_epoch": "proc-1"}
    connection = _Connection({
        "s1": _window([frame], latest=6,
                      open_requests=[{"id": "srq-1", "method": "approval", "params": {"session_id": "s1"}}]),
        "s2": _window(latest=7, open_requests=[{"id": "srq-1", "method": "approval", "params": {"session_id": "s2"}},
                                               {"id": "srq-2", "method": "clarify", "params": {"session_id": "s2"}}]),
    })
    result = await session_hello.hello(connection, None, {
        "capabilities": {"server_requests": True},
        "replay_epoch": "proc-1",
        "watermarks": {"s1": 4, "s2": 7},
        "session_epochs": {"s2": "sid-epoch"},
    })

    # One cursor per session, with that session's own epoch as the qualifier (never the process one).
    assert connection.calls == [("s1", {"last_seen": 4}), ("s2", {"last_seen": 7, "replay_epoch": "sid-epoch"})]
    assert [window["session_id"] for window in result["per_session_replay"]] == ["s1", "s2"]
    assert result["per_session_replay"][0]["events"] == [frame]
    assert result["per_session_replay"][0]["open_requests"] is None  # the union is the answer's own field
    assert result["skin_etag"] == "etag-1"
    assert result["capabilities_result"]["declines_not_shown"] is True
    assert result["capabilities_result"]["canonical_clarify_batch"] is True
    # Every open request, once, across the replayed sessions.
    assert [request["id"] for request in result["open_requests"]] == ["srq-1", "srq-2"]
    CANONICAL_METHODS["client.hello"].result.model_validate(result)


@pytest.mark.asyncio
async def test_a_refused_session_costs_only_its_own_window():
    connection = _Connection({"s2": _window(latest=7)})
    result = await session_hello.hello(connection, None, {"watermarks": {"s1": 4, "s2": 7}})

    refused, answered = (result["per_session_replay"][0], result["per_session_replay"][1])
    assert (refused["session_id"], refused["error"]) == ("s1", "not_found")
    assert refused["events"] == [] and refused["truncated"] is False
    assert (answered["session_id"], answered["error"]) == ("s2", None)
    assert answered["latest_seq"] == 7
    CANONICAL_METHODS["client.hello"].result.model_validate(result)


@pytest.mark.asyncio
async def test_a_bad_cursor_is_refused_per_session_without_asking_the_owner():
    connection = _Connection({})
    result = await session_hello.hello(connection, None, {"watermarks": {"s1": -1}})

    assert connection.calls == []
    assert (result["per_session_replay"][0]["session_id"], result["per_session_replay"][0]["error"]) == \
        ("s1", "invalid_params")
    CANONICAL_METHODS["client.hello"].result.model_validate(result)


@pytest.mark.asyncio
async def test_a_cold_body_watermarkless_hello_still_answers():
    """No watermarks (a first connect): the fold is still the connection's ONE handshake round trip."""
    connection = _Connection({})
    result = await session_hello.hello(connection, None, {})

    assert result["per_session_replay"] == [] and result["open_requests"] == []
    assert result["capabilities_result"]["server_requests"]  # the answer clients gate on
    CANONICAL_METHODS["client.hello"].result.model_validate(result)
