"""Recovery and pruning distinguish touched sessions from unused rows."""

import json
from datetime import UTC, datetime

import pytest

from persistence import SessionId
from persistence.conversation.sessions import NewSession
from tests.contracts._reference.conversation import sessions
from tests.contracts._reference.gateway import routing
from tests.contracts._scopes import MEDIA_ROOT


@pytest.mark.parametrize("activity", [None, 0, 150, 400])
def test_routing_activity_watermark(backend, scope_a, monkeypatch, activity):
    store_a = backend.open(scope_a)
    def dt(epoch):
        return datetime.fromtimestamp(epoch, UTC)

    for sid, stamp in (("older", 100), ("newer", 200)):
        monkeypatch.setattr("time.time", lambda stamp=stamp: stamp)
        monkeypatch.setattr(sessions, "_now", lambda stamp=stamp: dt(stamp))
        store_a.sessions.create(NewSession(SessionId(sid), "telegram", session_key="peer"))
        store_a.routing.upsert_routing_entry(sid, json.dumps({"session_id": sid}), routing_scope="test")
    if activity is not None:
        store_a.sessions.touch_activity(SessionId("older"), activity)
    result = store_a.routing.latest_session_for_peer(session_key="peer", source="telegram")
    expected = "older" if activity is not None and activity > 200 else "newer"
    assert result.session.session_id == expected
    assert result.session.last_activity_at == (float(activity) if expected == "older" else None)
    monkeypatch.setattr("time.time", lambda: 1000000)
    monkeypatch.setattr(routing, "_now", lambda: dt(1000000))
    deleted = store_a.routing.prune_never_active(older_than_days=1, media_root=MEDIA_ROOT)
    expected_count = 2 if activity is None else 1
    assert (deleted.sessions_deleted, deleted.routing_entries_deleted) == (expected_count, expected_count)
    assert (store_a.sessions.get(SessionId("older")) is not None) is (activity is not None)


@pytest.mark.parametrize("days", [-1, -0.000001, 0, 1])
@pytest.mark.parametrize("operation", ["never_active_keyed_sessions", "prune_never_active"])
def test_never_active_retention_boundary(backend, scope_a, days, operation):
    call = getattr(backend.open(scope_a).routing, operation)
    media = {"media_root": MEDIA_ROOT} if operation == "prune_never_active" else {}
    if days < 0:
        with pytest.raises(ValueError, match="older_than_days must be >= 0"):
            call(older_than_days=days, **media)
    else:
        result = call(older_than_days=days, **media)
        if operation == "prune_never_active":
            assert (result.sessions_deleted, result.routing_entries_deleted, result.sessions_skipped, result.deleted_ids, result.removed_ids) == (0, 0, 0, (), ())
        else:
            assert result == ()
