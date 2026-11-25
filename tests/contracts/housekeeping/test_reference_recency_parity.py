"""Maintenance recency uses activity and transcript timestamps before creation."""

from datetime import datetime, timedelta, UTC

import pytest

from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.housekeeping.maintenance import PruneCriteria
from tests.contracts._reference.conversation import sessions
from tests.contracts._scopes import MEDIA_ROOT


@pytest.mark.parametrize("touch", [False, True])
def test_archive_and_prune_use_activity(backend, scope_a, monkeypatch, touch):
    store_a = backend.open(scope_a)
    now = datetime(2027, 1, 15, 8, tzinfo=UTC)
    old = now - timedelta(days=10)
    monkeypatch.setattr("time.time", lambda: old.timestamp())
    monkeypatch.setattr(sessions, "_now", lambda: old)
    store_a.sessions.create(NewSession("recency", "cli"))
    store_a.messages.append("recency", NewMessage("user", "old", timestamp=old.timestamp()))
    if touch:
        store_a.sessions.touch_activity("recency", (now - timedelta(seconds=10)).timestamp(), description="working", provenance="agent.compression")
    monkeypatch.setattr("time.time", lambda: now.timestamp())
    # The idle sweep and the prune both read recency as the freshest of the activity stamp and the newest message.
    assert len(store_a.maintenance.list_idle_archive_candidates(1)) == (0 if touch else 1)
    store_a.sessions.end("recency", "done")
    criteria = PruneCriteria(last_active_before=(now - timedelta(days=1)).timestamp())
    assert store_a.maintenance.count_prune_matches(criteria=criteria) == (0 if touch else 1)
    assert len(store_a.maintenance.list_prune_candidates(criteria=criteria)) == (0 if touch else 1)
    assert store_a.maintenance.prune_sessions(criteria=criteria, media_root=MEDIA_ROOT).count == (0 if touch else 1)


def test_started_at_is_only_the_fallback_when_no_activity_or_message_stamp_exists(backend, scope_a, monkeypatch):
    store_a = backend.open(scope_a)
    now = datetime(2027, 1, 15, 8, tzinfo=UTC)
    old, started = now - timedelta(days=10), now - timedelta(hours=1)
    monkeypatch.setattr("time.time", lambda: started.timestamp())
    monkeypatch.setattr(sessions, "_now", lambda: started)
    store_a.sessions.create(NewSession("imported-history", "cli"))
    store_a.sessions.create(NewSession("no-messages", "cli"))
    monkeypatch.setattr("time.time", lambda: old.timestamp())
    store_a.messages.append("imported-history", NewMessage("user", "old", timestamp=old.timestamp()))
    monkeypatch.setattr("time.time", lambda: now.timestamp())
    for sid in ("imported-history", "no-messages"):
        store_a.sessions.end(sid, "done")
    criteria = PruneCriteria(last_active_before=(now - timedelta(days=1)).timestamp())
    # CONTRACT: recency is the freshest of the activity stamp and the newest message stamp, and started_at only when neither exists: a row started an hour ago whose newest message is ten days old matches a one-day cutoff; a row with no message falls back to its recent start and does not.
    assert [row.session_id for row in store_a.maintenance.list_prune_candidates(criteria=criteria)] == ["imported-history"]
