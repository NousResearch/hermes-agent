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
