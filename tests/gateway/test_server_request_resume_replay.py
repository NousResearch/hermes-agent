"""A viewer that reconnects while the owner's server→client request waits gets that request again.

The canonical side of the sidecar's ``open_requests`` replay (``tui_gateway/server_requests.py``):
a Desktop reload mid-onboarding re-renders the waiting ``setup_beat`` card, and answering it there
unblocks the turn instead of leaving it parked until ``CARD_TIMEOUT_S``.
"""
from __future__ import annotations

import asyncio

import pytest

from tests.gateway.test_onboarding_on_the_authority import _desktop, _request, _Transport
from tests.waiting import async_wait_until


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
async def test_a_resumed_viewer_gets_the_waiting_setup_card_and_its_answer_unblocks_the_turn(tmp_path, monkeypatch):
    from gateway.session_server_requests import ask_viewer_result, waiting
    desktop = await _desktop(tmp_path, monkeypatch, [])
    try:
        stored = (await _request(desktop, 1, "onboarding.start", {"rerun": True}))["result"]["stored_session_id"]
        assert "result" in await _request(desktop, 2, "session.resume", {"session_id": stored})
        live = desktop.authority.sessions[stored]
        params = {"beat": "name", "tool_call_id": "call-1", "card": {"suggested": "Sid"}}
        answer = asyncio.get_running_loop().run_in_executor(None, lambda: ask_viewer_result(
            live.event_stream, stored, "setup_beat", params, timeout=20))
        await async_wait_until(lambda: waiting(live.event_stream, stored, "setup_beat"), timeout=5)

        await _reload(desktop)
        resumed = await _request(desktop, 3, "session.resume", {"session_id": stored})

        assert "result" in resumed, resumed
        [entry] = resumed["result"]["open_requests"]
        assert entry["method"] == "setup_beat" and entry["params"] == {"session_id": stored, **params}
        reply = await desktop.connection.dispatch({"jsonrpc": "2.0", "id": entry["id"],
                                                   "result": {"answer": {"name": "Sid"}}})
        assert reply is None
        assert await asyncio.wait_for(answer, 10) == {"answer": {"name": "Sid"}}
        assert not waiting(live.event_stream, stored, "setup_beat")
    finally:
        await desktop.connection.close()
