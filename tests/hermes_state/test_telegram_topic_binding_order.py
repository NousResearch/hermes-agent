"""Telegram DM topic bindings must come back in a *total* order.

``list_telegram_topic_bindings_for_chat`` documents "newest first", and
``GatewayRunner._recover_telegram_topic_thread_id`` relies on that: it walks
the rows and takes the first one belonging to the user as their most-recent
topic.  ``ORDER BY updated_at DESC`` alone does not give a total order --
``updated_at`` is a ``time.time()`` float, so two topics bound in the same
tick tie and SQLite may return them in either order.  A tie there sends a
lobby-shaped reply into an arbitrary lane.

These tests bind topics back-to-back (no sleep) so the tie is the normal
case, not a rare one.
"""
from __future__ import annotations

import pytest

from hermes_state import SessionDB


def _bind(db: SessionDB, *, chat_id: str, thread_id: str, user_id: str, session_id: str):
    db.create_session(session_id, "telegram", user_id=user_id)
    db.bind_telegram_topic(
        chat_id=chat_id,
        thread_id=thread_id,
        user_id=user_id,
        session_key=f"agent:main:telegram:dm:{chat_id}:{thread_id}",
        session_id=session_id,
    )


def test_same_tick_bindings_return_newest_first(tmp_path):
    """Two topics bound in the same tick: the later one must come first."""
    db = SessionDB(db_path=tmp_path / "state.db")
    for i, thread in enumerate(("111", "222")):
        _bind(db, chat_id="chat-1", thread_id=thread, user_id="u1", session_id=f"s{i}")

    rows = db.list_telegram_topic_bindings_for_chat(chat_id="chat-1")
    assert [r["thread_id"] for r in rows] == ["222", "111"], (
        "bindings written in the same tick came back in insertion order or "
        "arbitrary order; the newest must lead"
    )


def test_ordering_is_stable_across_repeated_reads(tmp_path):
    """The same rows must come back in the same order every time."""
    db = SessionDB(db_path=tmp_path / "state.db")
    for i in range(8):
        _bind(
            db, chat_id="chat-1", thread_id=str(100 + i),
            user_id="u1", session_id=f"s{i}",
        )

    orders = {
        tuple(r["thread_id"] for r in db.list_telegram_topic_bindings_for_chat(chat_id="chat-1"))
        for _ in range(25)
    }
    assert len(orders) == 1, f"ordering varied across reads: {orders}"
    # Newest binding first.
    assert next(iter(orders))[0] == "107"


def test_explicit_updated_at_still_wins_over_the_tiebreaker(tmp_path):
    """rowid only breaks ties -- a genuinely newer updated_at still leads."""
    db = SessionDB(db_path=tmp_path / "state.db")
    _bind(db, chat_id="chat-1", thread_id="111", user_id="u1", session_id="s0")
    _bind(db, chat_id="chat-1", thread_id="222", user_id="u1", session_id="s1")

    # Make the *older* rowid unambiguously the most recently updated.
    with db._lock:
        db._conn.execute(
            "UPDATE telegram_dm_topic_bindings SET updated_at = ? "
            "WHERE chat_id = ? AND thread_id = ?",
            (9_999_999_999.0, "chat-1", "111"),
        )
        db._conn.commit()

    rows = db.list_telegram_topic_bindings_for_chat(chat_id="chat-1")
    assert [r["thread_id"] for r in rows] == ["111", "222"], (
        "rowid must be a tiebreaker only, never override updated_at"
    )
