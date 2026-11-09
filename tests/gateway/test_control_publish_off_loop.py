"""Control-card publishes read their snapshot off the loop and still land in the order they were scheduled.

The snapshot is three state.db meta reads; on the loop, the post-turn publish was a measured stall
with many turns running. Deferring the read must not let a slow, older read repaint the cards over
a newer state that a later publish already painted.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway import session_control_actions as actions


def _session(published):
    live = SimpleNamespace(control_seq=0, control_published_seq=0, control_revision='',
                           event_stream=SimpleNamespace(
                               publish=lambda sid, payload, event_type: published.append(payload['control']['revision'])))
    return SimpleNamespace(sessions={'s1': live}, profile_id='fixture'), live


@pytest.mark.asyncio
async def test_a_slow_snapshot_read_never_paints_over_a_newer_publish(monkeypatch):
    published = []
    authority, live = _session(published)
    old_read_entered, release_old_read = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    read_threads = []

    def read(_authority, _sid, *, scoped=False):
        read_threads.append(threading.get_ident())
        if not old_read_entered.is_set():
            old_read_entered.set()
            assert release_old_read.wait(5)
            return {'revision': 'old'}
        return {'revision': 'new'}

    monkeypatch.setattr(actions, '_read_control_snapshot', read)
    slow = asyncio.ensure_future(actions.publish_async(authority, 's1'))
    assert await asyncio.to_thread(old_read_entered.wait, 5)
    await actions.publish_async(authority, 's1')  # scheduled later, read finishes first
    release_old_read.set()
    await slow
    assert published == ['new']
    assert live.control_revision == 'new'
    assert loop_thread not in read_threads, 'the snapshot was read on the event loop'


@pytest.mark.asyncio
async def test_the_per_tool_republish_reads_its_snapshot_off_the_loop(monkeypatch):
    """Every tool of every running session republishes the control card. Its snapshot read takes
    state.db's connection lock, which the agents' own writes hold, so on the loop it was a large
    share of loop time with many sessions running (gateway disconnects under load)."""
    from gateway.local_viewer_events import publish_control_if_changed

    published = []
    authority, live = _session(published)
    loop_thread = threading.get_ident()
    read_threads = []

    def read(_authority, _sid, *, scoped=False):
        read_threads.append(threading.get_ident())
        return {'revision': 'r1'}

    monkeypatch.setattr(actions, '_read_control_snapshot', read)
    scheduled = []
    turn = SimpleNamespace(_approval_owner=(authority, 's1', 1),
                           _schedule=lambda coro, _message: scheduled.append(asyncio.ensure_future(coro)))
    publish_control_if_changed(turn)
    await asyncio.gather(*scheduled)
    assert published == ['r1'] and live.control_revision == 'r1'
    assert read_threads and loop_thread not in read_threads, 'the snapshot was read on the event loop'

    publish_control_if_changed(turn)  # nothing changed: revision-gated, nothing published
    await asyncio.gather(*scheduled)
    assert published == ['r1']
