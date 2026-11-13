"""A gateway session's replay frames live only as long as the session does.

Each live session's ``SessionEvents`` stamps its frames into ``hermes_agent.tui_gateway.event_replay`` under a private
lookup key. Before this, nothing released that key when the session ended: a closed or deleted session's
ring stayed until the shared 64-session cap evicted it, and its counters stayed for the life of the process.
A session that is only detached (resumable) keeps its ring, since a reconnect replays from it.
"""
import gc

import pytest

from tests.gateway.test_viewer_close_lifecycle import _call, _open, _owner


def _recorded(key):
    from hermes_agent.tui_gateway import event_replay
    return (key in event_replay._replay_buffers or key in event_replay._replay_next_seq
            or key in event_replay._replay_evicted_through)


@pytest.mark.asyncio
async def test_closing_the_last_viewer_frees_the_ring_and_detach_keeps_it(tmp_path, monkeypatch):
    db, authority, _calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    stream = authority.sessions['chat'].event_stream
    stream.publish('chat', {'text': 'hello'})
    epoch, sequence = stream.watermark()

    # Detached but resumable: the reconnect contract still replays from the ring.
    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription)
    again = await _call(viewer, 'session.resume', session_id='chat')
    replay = await _call(viewer, 'session.events.since', session_id='chat', replay_epoch=epoch, last_sequence=0)
    assert not replay['snapshot_required'] and replay['count'] == sequence

    await _call(viewer, 'session.detach', session_id='chat', subscription_id=again['subscription_id'],
                end_reason='cli_close')
    assert db.get_session('chat')['end_reason'] == 'cli_close'
    assert not _recorded(stream._key)
    # A later resume of the ended session refetches history instead of trusting a stale cursor.
    await _call(viewer, 'session.resume', session_id='chat')
    replay = await _call(viewer, 'session.events.since', session_id='chat', replay_epoch=epoch,
                         last_sequence=sequence)
    assert replay['snapshot_required']


def test_a_dropped_session_stream_leaves_nothing_in_the_shared_cache():
    from hermes_agent.gateway.session_events import SessionEvents
    from hermes_agent.tui_gateway import event_replay

    kept, dropped = SessionEvents(), SessionEvents()
    for stream in (kept, dropped):
        for n in range(3):
            stream.publish('s', {'n': n})
    key = dropped._key
    del dropped, stream
    gc.collect()
    stats = event_replay.replay_stats()  # the next lock holder purges a finalizer's queued key
    assert not _recorded(key)
    assert stats['sessions'] >= 1
    assert [frame['payload'] for frame in kept.since(kept.epoch, 0)['events']] == [{'n': 0}, {'n': 1}, {'n': 2}]


def test_a_finalizer_that_finds_the_lock_held_defers_the_release_without_blocking():
    from hermes_agent.gateway.session_events import SessionEvents
    from hermes_agent.tui_gateway import event_replay

    dropped = SessionEvents()
    dropped.publish('s', {'n': 0})
    key = dropped._key
    with event_replay._replay_lock:  # GC can run the finalizer on a thread already holding the lock
        del dropped
        gc.collect()
        assert _recorded(key)  # queued, not purged: the finalizer never waited on the lock
    event_replay.replay_stats()  # the next lock holder purges it
    assert not _recorded(key)
