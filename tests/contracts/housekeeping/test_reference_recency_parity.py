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
