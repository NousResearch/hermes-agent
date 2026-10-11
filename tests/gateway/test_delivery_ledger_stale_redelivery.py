"""A reply owed by a process that died long ago must not be resurrected.

Incident: the gateway restarted mid-send, then the machine stayed off. On the next boot ~12 h
later the boot sweep claimed the stale obligation and delivered it, so the old reply landed on
top of a brand-new inbound message and read as its answer. ``STALE_AFTER_SECONDS`` (24 h) bounds
how long a row is *retained*; it is far too generous to bound *redelivery* — a reply is only
useful as recovery while the turn it answers is still the conversation's live edge.
"""

from __future__ import annotations

import sqlite3

import pytest

from gateway import delivery_ledger as dl

FRESH_AGE_SECONDS = 60
# Comfortably past any sane cap, and the real-world figure from the incident.
STALE_AGE_SECONDS = 12 * 60 * 60


@pytest.fixture
def ledger_db(tmp_path, monkeypatch):
    """Point the ledger at a throwaway state.db and hand back the module."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    return dl


def _record(ledger, *, content: str, chat_id: str, session_key: str) -> str:
    oid = ledger.compute_obligation_id(session_key, "msg-1", content)
    ledger.record_obligation(
        obligation_id=oid, session_key=session_key, platform="telegram",
        chat_id=chat_id, thread_id=None, content=content,
    )
    return oid


def _age_row(ledger, oid: str, *, seconds: float) -> None:
    """Backdate a row (and clear its owner) so the boot sweep sees it as crash-left."""
    with sqlite3.connect(ledger._db_path()) as conn:
        conn.execute(
            "UPDATE delivery_obligations SET created_at = created_at - ?,"
            " updated_at = updated_at - ?, owner_pid = NULL, owner_started_at = NULL"
            " WHERE obligation_id = ?",
            (seconds, seconds, oid),
        )


def _state_of(ledger, oid: str) -> str:
    with sqlite3.connect(ledger._db_path()) as conn:
        return conn.execute(
            "SELECT state FROM delivery_obligations WHERE obligation_id = ?", (oid,)
        ).fetchone()[0]


def test_stale_recovered_reply_is_not_redelivered(ledger_db):
    """The reported bug: a ~12 h old reply must not be sent onto a fresh inbound message."""
    oid = _record(ledger_db, content="the stale answer", chat_id="1",
                  session_key="telegram:1:1")
    _age_row(ledger_db, oid, seconds=STALE_AGE_SECONDS)

    claimed = ledger_db.sweep_recoverable(deliverable_platforms={"telegram"})

    assert claimed == [], "a half-day-old reply must not be resurrected"
    assert _state_of(ledger_db, oid) == "abandoned", "it must leave the retry set for good"


def test_fresh_recovered_reply_is_still_redelivered(ledger_db):
    """The age cap must not break the feature it guards: a crash seconds ago still recovers."""
    oid = _record(ledger_db, content="the fresh answer", chat_id="2",
                  session_key="telegram:2:1")
    _age_row(ledger_db, oid, seconds=FRESH_AGE_SECONDS)

    claimed = ledger_db.sweep_recoverable(deliverable_platforms={"telegram"})

    assert [row["content"] for row in claimed] == ["the fresh answer"]
    assert _state_of(ledger_db, oid) == "attempting"
