"""``session.active_list`` reads ``waiting`` while a viewer request holds the turn.

A TUI kept a session working in the background (``/new`` mid-turn) and polls this verb to tell the
user when that session needs an answer. Clarify batches, sudo, secret and vault prompts are viewer
requests (``hermes_agent.gateway.session_server_requests``), not shared controls, so a status read from the
controls alone said ``working`` while the turn sat blocked on the question.
"""
import asyncio
import threading

import pytest

from tests.gateway.test_authority_active_list import owner as listing_owner

listing = listing_owner  # the active-list fixture, under a name the gateway conftest's `owner` does not shadow


class _Peer:
    def __init__(self):
        self.frames = []

    def write(self, frame, **_):
        self.frames.append(frame)
        return True

    def close(self):
        pass


@pytest.mark.asyncio
async def test_a_turn_blocked_on_a_viewer_request_reads_waiting(listing):
    from hermes_agent.gateway.session_server_requests import ask_viewer, withdraw

    await listing.submit('hi')
    events = listing.authority.sessions['chat'].event_stream
    peer = _Peer()
    events.fanout.attach(peer)
    assert (await listing.active())['chat']['status'] == 'working'

    asked = threading.Thread(target=ask_viewer, args=(events, 'chat', 'sudo', {'command': 'id'}),
                             kwargs={'timeout': 10}, daemon=True)
    asked.start()
    for _ in range(200):
        if peer.frames:
            break
        await asyncio.sleep(0.01)
    assert peer.frames, 'the sudo request never reached the viewer'

    try:
        assert (await listing.active())['chat']['status'] == 'waiting'
    finally:
        withdraw(events, 'chat', 'sudo')
        asked.join(5)
    assert (await listing.active())['chat']['status'] == 'working'
