"""A session row whose stored instant is corrupt reads degraded on SQLite instead of failing the read."""

import logging

from persistence.conversation.sessions import NewSession, RecentBrowseQuery
from persistence.core.conventions import UNTRUSTED_INSTANT


def _plant(store, session_id: str, **cells) -> None:
    """Write raw cells into the stored row, bypassing every typed write path (how a damaged page or a foreign writer
    lands them)."""
    assignments = ", ".join(f"{column} = ?" for column in cells)
    store.holder()._execute_write(
        lambda conn: conn.execute(f"UPDATE sessions SET {assignments} WHERE id = ?", (*cells.values(), session_id)))


def test_a_corrupt_stored_instant_reads_degraded(backend, scope_a, caplog):
    store = backend.open(scope_a)
    store.sessions.create(NewSession("si-bad", "cli"))
    store.sessions.create(NewSession("si-good", "cli"))
    store.sessions.end("si-good", "idle")
    _plant(store, "si-bad", started_at="not-a-timestamp", ended_at=8.4e252)
    with caplog.at_level(logging.WARNING, logger="persistence.core.conventions"):
        bad = store.sessions.get("si-bad")
        good = store.sessions.get("si-good")
    assert bad is not None and bad.started_at == UNTRUSTED_INSTANT.timestamp() and bad.ended_at is None
    assert good is not None and good.started_at > UNTRUSTED_INSTANT.timestamp() and good.ended_at is not None
    warned = [rec.getMessage() for rec in caplog.records if rec.name == "persistence.core.conventions"]
    assert len(warned) == 2 and all("on session si-bad" in message for message in warned)


def test_a_recent_browse_row_with_no_trusted_cell_has_no_recency(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession("good", "cli"))
    store.sessions.create(NewSession("bad", "cli"))
    _plant(store, "bad", started_at=8.4e252, last_activity_at=8.4e252)
    entries = store.sessions.list_recent_sessions(RecentBrowseQuery(limit=10))
    assert [(entry.session_id, entry.last_active is None) for entry in entries] == [("good", False), ("bad", True)]
