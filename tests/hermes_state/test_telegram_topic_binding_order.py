"""Telegram DM topic bindings must come back in a *total* order.

``list_telegram_topic_bindings_for_chat`` documents "newest first", and
``GatewayRunner._recover_telegram_topic_thread_id`` relies on that: it walks
the rows and takes the first one belonging to the user as their most-recent
topic.  ``ORDER BY updated_at DESC`` alone does not give a total order --
``updated_at`` is a ``time.time()`` float, so two topics bound in the same
tick tie and SQLite may return them in either order.  A tie there sends a
lobby-shaped reply into an arbitrary lane.

These tests force every binding to share one ``updated_at`` rather than
relying on back-to-back writes landing in the same tick: that depends on
the platform's clock resolution, and on a high-resolution clock the
tiebreaker would never be reached. Each test asserts the tie exists
before asserting the order.
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


def _force_tie(db: SessionDB, chat_id: str, ts: float = 1_760_000_000.0) -> None:
    """Give every binding in *chat_id* the same ``updated_at``.

    Binding back-to-back usually ties on its own, but that depends on the
    platform's ``time.time()`` resolution -- on a high-resolution clock the
    writes get distinct timestamps and the ``rowid`` tiebreaker is never
    reached, so the test would pass without exercising what it names.
    Forcing the tie makes these tests deterministic on every platform.
    """
    with db._lock:
        db._conn.execute(
            "UPDATE telegram_dm_topic_bindings SET updated_at = ? WHERE chat_id = ?",
            (ts, chat_id),
        )
        db._conn.commit()


def _assert_tied(db: SessionDB, chat_id: str) -> None:
    """Guard: the rows really do tie, so the tiebreaker is under test."""
    with db._lock:
        distinct = db._conn.execute(
            "SELECT COUNT(DISTINCT updated_at) FROM telegram_dm_topic_bindings "
            "WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()[0]
    assert distinct == 1, (
        f"expected all bindings to share one updated_at, saw {distinct} distinct "
        "values — the rowid tiebreaker would not be exercised"
    )


def test_tied_bindings_return_newest_first(tmp_path):
    """Two topics sharing an updated_at: the later-inserted one must lead."""
    db = SessionDB(db_path=tmp_path / "state.db")
    for i, thread in enumerate(("111", "222")):
        _bind(db, chat_id="chat-1", thread_id=thread, user_id="u1", session_id=f"s{i}")
    _force_tie(db, "chat-1")
    _assert_tied(db, "chat-1")

    rows = db.list_telegram_topic_bindings_for_chat(chat_id="chat-1")
    assert [r["thread_id"] for r in rows] == ["222", "111"], (
        "tied bindings came back in insertion order or arbitrary order; "
        "the newest must lead"
    )


def test_ordering_is_stable_across_repeated_reads(tmp_path):
    """The same tied rows must come back in the same order every time."""
    db = SessionDB(db_path=tmp_path / "state.db")
    for i in range(8):
        _bind(
            db, chat_id="chat-1", thread_id=str(100 + i),
            user_id="u1", session_id=f"s{i}",
        )
    _force_tie(db, "chat-1")
    _assert_tied(db, "chat-1")

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
