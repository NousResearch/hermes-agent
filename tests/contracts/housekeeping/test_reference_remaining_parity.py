"""Maintenance parity at lease boundaries: the orphan sweep checks and reclaims the compression lock before the turn lease."""

import pytest

from persistence import SessionId


@pytest.mark.parametrize("compaction,turn", [(None, None), (None, 10), (None, -1), (None, 0), (10, None), (-1, None), (0, None), (-1, 10), (10, -1), (-1, -1)])
def test_orphan_sweep_checks_and_reclaims_leases_in_order(backend, scope_a, monkeypatch, compaction, turn):
    store_a = backend.open(scope_a)
    now = 1800000000.0
    clock = [now - 100]
    monkeypatch.setattr("time.time", lambda: clock[0])
    store_a.maintenance.import_sessions([{"id": "orphan", "source": "tui", "started_at": now - 1000, "messages": []}])
    if compaction is not None:
        store_a.coordination.try_acquire_compression_lock(session_id=SessionId("orphan"), holder="opaque", ttl_seconds=100 + compaction)
    if turn is not None:
        store_a.coordination.try_acquire_turn_lease(session_id=SessionId("orphan"), holder="opaque", ttl_seconds=100 + turn)
    clock[0] = now
    result = store_a.maintenance.sweep_orphaned_sessions(max_idle_seconds=100, sources=("tui",), respect_gateway_heartbeats=False)
    compaction_live = compaction is not None and compaction > 0
    turn_live = turn is not None and turn > 0
    assert result == (() if compaction_live or turn_live else ("orphan",))
    assert (store_a.sessions.get(SessionId("orphan")).ended_at is None) == (compaction_live or turn_live)
    # A live compression lock stops the check before the turn lease, so an expired turn lease behind it is not reclaimed.
    assert store_a.coordination.refresh_compression_lock(session_id=SessionId("orphan"), holder="opaque", ttl_seconds=100) == compaction_live
    assert store_a.coordination.refresh_turn_lease(session_id=SessionId("orphan"), holder="opaque", ttl_seconds=100) == (turn is not None and (compaction_live or turn_live))
