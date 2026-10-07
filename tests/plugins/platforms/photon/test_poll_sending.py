"""Votes can answer only the poll returned by their clarify's send."""
import asyncio
import json
import logging

import pytest

from gateway.platforms.base import SendResult
from plugins.platforms.photon.adapter import _poll_state_path
from tests.plugins.platforms.photon.test_poll_clarify import (
    _capture, _make_adapter, _poll_option_event, _stub_sidecar_poll, _stub_sidecar_text,
)
from tools import clarify_gateway as cg

CHAT = "+155****4567"


@pytest.fixture(autouse=True)
def isolate_clarify(monkeypatch):
    monkeypatch.setattr(cg, "_entries", {})
    monkeypatch.setattr(cg, "_session_index", {})
    monkeypatch.setattr(cg, "_notify_cbs", {})


@pytest.fixture
def adapter(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    _capture(adapter, monkeypatch)
    _stub_sidecar_text(adapter, monkeypatch)
    return adapter


async def send(adapter, clarify_id):
    entry = cg.register(clarify_id, clarify_id, "Which?", ["Route", "Calendar"])
    await adapter.send_clarify(CHAT, entry.question, entry.choices, clarify_id, clarify_id)
    return entry


@pytest.mark.asyncio
async def test_evicted_old_poll_cannot_answer_during_new_send(adapter, monkeypatch):
    _stub_sidecar_poll(adapter, monkeypatch, poll_id="old-poll")
    old = await send(adapter, "old")
    cg.resolve_gateway_clarify(old.clarify_id, "Calendar")
    for i in range(200):
        _stub_sidecar_poll(adapter, monkeypatch, poll_id=f"poll-{i}")
        entry = await send(adapter, f"clarify-{i}")
        cg.resolve_gateway_clarify(entry.clarify_id, "Calendar")
    assert "old-poll" not in adapter._clarify_polls

    async def poll(*_):
        await adapter._dispatch_inbound(_poll_option_event(title="Route", poll_id="old-poll"))
        assert not adapter._early_poll_votes
        return SendResult(success=True, message_id="new-poll")

    monkeypatch.setattr(adapter, "_sidecar_send_poll", poll)
    new = await send(adapter, "new")
    assert not new.event.is_set()


@pytest.mark.asyncio
async def test_persisted_sent_id_is_stale_during_send(adapter, monkeypatch):
    path = _poll_state_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"polls": [], "sentPollIds": ["old-poll"]}))

    async def poll(*_):
        await adapter._dispatch_inbound(_poll_option_event(title="Route", poll_id="old-poll"))
        assert not adapter._early_poll_votes
        return SendResult(success=True, message_id="new-poll")

    monkeypatch.setattr(adapter, "_sidecar_send_poll", poll)
    assert not (await send(adapter, "new")).event.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("returned_id", ["new-poll", None])
async def test_unrelated_early_vote_never_answers(adapter, monkeypatch, caplog, returned_id):
    async def poll(*_):
        await adapter._dispatch_inbound(_poll_option_event(title="Route", poll_id="old-poll"))
        assert not cg._entries["new"].event.is_set()
        return SendResult(success=True, message_id=returned_id)

    monkeypatch.setattr(adapter, "_sidecar_send_poll", poll)
    with caplog.at_level(logging.WARNING):
        entry = await send(adapter, "new")
    assert not entry.event.is_set()
    assert not adapter._early_poll_votes
    if returned_id is None:
        assert "no messageId" in caplog.text
        assert cg.resolve_text_response_for_session("new", "2")
        assert entry.response == "Calendar"


@pytest.mark.asyncio
async def test_vote_without_poll_id_warns_and_never_answers(adapter, monkeypatch, caplog):
    _stub_sidecar_poll(adapter, monkeypatch)
    entry = await send(adapter, "new")
    with caplog.at_level(logging.WARNING):
        await adapter._dispatch_inbound(_poll_option_event(title="Route", poll_id=None))
    assert not entry.event.is_set()
    assert "without pollId" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("events,answer", [
    ([("Route", True), ("Route", False)], None),
    ([("Route", True), ("Route", False), ("Calendar", True)], "Calendar"),
    ([("Route", True), ("Calendar", True), ("Route", False)], "Calendar"),
])
async def test_buffer_replays_final_voter_selection(adapter, monkeypatch, events, answer):
    async def poll(*_):
        for i, (choice, selected) in enumerate(events):
            await adapter._dispatch_inbound(_poll_option_event(
                title=choice, selected=selected, msg_id=str(i), poll_id="new-poll"))
        assert not cg._entries["new"].event.is_set()
        return SendResult(success=True, message_id="new-poll")

    monkeypatch.setattr(adapter, "_sidecar_send_poll", poll)
    entry = await send(adapter, "new")
    assert entry.response == answer
    assert not adapter._early_poll_votes


@pytest.mark.asyncio
async def test_buffer_bounds_latest_voter_states_and_total(adapter, monkeypatch, caplog):
    caplog.set_level(logging.CRITICAL)

    async def poll(*_):
        for i in range(3000):
            await adapter._dispatch_inbound(_poll_option_event(title="Route", poll_id="never-bound", msg_id=str(i)))
        assert len(adapter._early_poll_votes["never-bound"]) == 1
        for p in range(80):
            for v in range(40):
                event = _poll_option_event(title="Route", poll_id=f"unbound-{p}")
                event["sender"]["id"] = f"voter-{v}"
                await adapter._dispatch_inbound(event)
        assert len(adapter._early_poll_votes) == 50
        assert all(len(states) == 32 for states in adapter._early_poll_votes.values())
        return SendResult(success=True, message_id="new-poll")

    monkeypatch.setattr(adapter, "_sidecar_send_poll", poll)
    entry = await send(adapter, "new")
    assert not entry.event.is_set()
    assert not adapter._early_poll_votes


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ["typed", "expired", "cancelled", "send-error"])
async def test_buffer_cleans_up_when_clarify_or_send_ends(adapter, monkeypatch, end):
    async def poll(*_):
        await adapter._dispatch_inbound(_poll_option_event(title="Route", poll_id="never-bound"))
        assert adapter._early_poll_votes
        if end == "typed":
            cg.resolve_text_response_for_session("new", "Calendar")
        elif end == "expired":
            cg.wait_for_response("new", 0.001)
        elif end == "cancelled":
            cg.clear_session("new")
        else:
            raise RuntimeError("send failed")
        # Wait on the production watcher, with a generous bound for busy test hosts.
        await asyncio.wait_for(adapter._watch_poll_send(adapter._open_poll_clarifies["new"]), 5)
        assert not adapter._early_poll_votes
        return SendResult(success=True, message_id="new-poll")

    monkeypatch.setattr(adapter, "_sidecar_send_poll", poll)
    if end == "send-error":
        with pytest.raises(RuntimeError, match="send failed"):
            await send(adapter, "new")
    else:
        await send(adapter, "new")
    assert not adapter._early_poll_votes
