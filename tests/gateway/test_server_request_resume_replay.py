"""A viewer that reconnects while the owner's server→client request waits gets that request again.

The canonical side of the sidecar's ``open_requests`` replay (``tui_gateway/server_requests.py``):
a Desktop reload re-renders a waiting clarification card, and answering it there
unblocks the turn instead of leaving it parked until the request timeout.
"""
from __future__ import annotations

import asyncio

import pytest

from tests.gateway.test_onboarding_on_the_authority import _desktop, _request, _Transport
from tests.waiting import async_wait_until, wait_until


async def _reload(desktop):
    """The renderer reloads: its socket closes and a new one connects and resumes."""
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local_migration import bind_native_transport
    await desktop.connection.close()
    identity = {"user_id": "uid:1", "provider": "local", "profile_id": desktop.authority.profile_id,
                "instance_id": "owner", "native_bootstrap": True}
    desktop.transport = _Transport()
    desktop.transport.desktop_local = True
    desktop.connection = AuthorityConnection(desktop.authority, desktop.transport, identity)
    bind_native_transport(desktop.authority, desktop.connection.actor, identity)


@pytest.mark.asyncio
async def test_a_resumed_viewer_gets_the_waiting_card_and_its_answer_unblocks_the_turn(tmp_path, monkeypatch):
    from gateway.session_server_requests import ask_viewer_result, waiting
    desktop = await _desktop(tmp_path, monkeypatch, [])
    try:
        created = await _request(desktop, 1, "session.create", {"source": "gui", "request_id": "replay"})
        assert "result" in created, created
        stored = created["result"]["stored_session_id"]
        assert "result" in await _request(desktop, 2, "session.resume", {"session_id": stored})
        live = desktop.authority.sessions[stored]
        params = {"questions": [{"qid": "q0", "question": "Name?", "choices": None, "multi_select": False}]}
        answer = asyncio.get_running_loop().run_in_executor(None, lambda: ask_viewer_result(
            live.event_stream, stored, "clarify", params, timeout=20, qids=["q0"]))
        await async_wait_until(lambda: waiting(live.event_stream, stored, "clarify"), timeout=5)

        await _reload(desktop)
        resumed = await _request(desktop, 3, "session.resume", {"session_id": stored})

        assert "result" in resumed, resumed
        [entry] = resumed["result"]["open_requests"]
        assert entry["method"] == "clarify" and entry["params"] == {"session_id": stored, **params}
        reply = await desktop.connection.dispatch({"jsonrpc": "2.0", "id": entry["id"],
                                                   "result": {"answers": {"q0": "Sid"}}})
        assert reply is None
        assert await asyncio.wait_for(answer, 10) == {"answers": {"q0": "Sid"}, "outcome": "submitted"}
        assert not waiting(live.event_stream, stored, "clarify")
    finally:
        await desktop.connection.close()


def test_an_automatic_window_request_is_not_handed_to_a_reconnecting_viewer():
    """A preview click still in flight at a reconnect would run twice if replayed; a card is replayed."""
    import threading
    from gateway.session_events import SessionEvents
    from gateway.session_server_requests import ask_viewer, ask_viewer_result, open_requests, waiting, withdraw
    events = SessionEvents()
    events.fanout.attach(_Transport())
    asks = [threading.Thread(target=ask_viewer, args=(events, "s1", "preview.act", {"action": "click"}),
                             kwargs={"timeout": 10, "replay": False}),
            threading.Thread(target=ask_viewer_result, args=(events, "s1", "clarify", {"questions": []}),
                             kwargs={"timeout": 10})]
    for ask in asks:
        ask.start()
    try:
        wait_until(lambda: waiting(events, "s1", "preview.act") and waiting(events, "s1", "clarify"), timeout=5)
        assert [entry["method"] for entry in open_requests(events, "s1")] == ["clarify"]
    finally:
        withdraw(events, "s1", "preview.act")
        withdraw(events, "s1", "clarify")
        for ask in asks:
            ask.join(5)
