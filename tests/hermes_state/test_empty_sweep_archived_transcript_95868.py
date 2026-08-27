"""The empty-session sweep must not eat a rewound / compacted transcript (#95868).

``count_empty_sessions`` / ``delete_empty_sessions`` back the dashboard's
"Delete empty (N)" affordance. They used to define "empty" as
``sessions.message_count = 0``, which is a denormalized counter over the LIVE
(``active = 1``) rows only.

Two production transcript-rewrite paths reset that counter on purpose while
keeping every dropped turn on disk as ``active = 0``:

* ``replace_messages(..., archive_dropped=True)`` — the rewind / edit /
  regenerate mode added in #82756 precisely so a user can take a turn back
  without the rows being unrecoverable.
* ``archive_and_compact`` — in-place compaction, which archives the
  pre-compaction transcript under the same session id (#38763).

Rewind a chat back to its first turn, or compact it with an empty live set,
and the row reports ``message_count = 0`` while still holding its entire
recoverable history. Those soft-archived rows are the ONLY copy.

A gateway reload is what makes such a row *eligible*: every detached session
gets ``ended_at`` stamped (``end_reason='ws_orphan_reap'``), which satisfies
the sweep's ``ended_at IS NOT NULL`` gate. The next "Delete empty" click then
hard-deleted the session row AND ``DELETE FROM messages`` — silently, with no
log line to trace it by.

These tests pin the counter drift as real, and pin the sweep's refusal to act
on it.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    return SessionDB(db_path=tmp_path / "state.db")


def _seed(db, session_id, turns=4):
    """A populated, ended desktop chat — the shape #95868 lost."""
    db.create_session(session_id, source="desktop", model="test-model")
    for i in range(turns):
        db.append_message(
            session_id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"turn {i}",
        )
    return session_id


def _row_counts(db, session_id):
    """(message_count column, real rows on disk) for *session_id*."""
    with db._lock:
        counter = db._conn.execute(
            "SELECT message_count FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()["message_count"]
        rows = db._conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()["n"]
    return counter, rows


def test_rewind_to_empty_drifts_the_counter_to_zero(db):
    """The premise: an archived-drop rewrite zeroes the counter, keeps the rows."""
    _seed(db, "rewound")
    assert _row_counts(db, "rewound") == (4, 4)

    db.replace_messages("rewound", [], archive_dropped=True)

    counter, rows = _row_counts(db, "rewound")
    assert counter == 0, "message_count tracks the live set only"
    assert rows == 4, "the dropped turns stay on disk as the recoverable copy"
    assert len(db.get_messages("rewound", include_inactive=True)) == 4


def test_sweep_spares_a_rewound_session(db):
    """A rewound chat is not empty and must survive the sweep."""
    _seed(db, "rewound")
    db.replace_messages("rewound", [], archive_dropped=True)
    # A gateway reload stamps ended_at on every detached session, which is what
    # makes the row eligible for the sweep in the first place.
    db.end_session("rewound", end_reason="ws_orphan_reap")

    assert db.count_empty_sessions() == 0
    assert db.delete_empty_sessions() == 0
    assert db.get_session("rewound") is not None
    assert len(db.get_messages("rewound", include_inactive=True)) == 4


def test_sweep_spares_an_in_place_compacted_session(db):
    """``archive_and_compact`` leaves the same zero-counter/live-rows shape."""
    _seed(db, "compacted")
    db.archive_and_compact("compacted", [])
    assert _row_counts(db, "compacted") == (0, 4)

    db.end_session("compacted", end_reason="ws_orphan_reap")

    assert db.count_empty_sessions() == 0
    assert db.delete_empty_sessions() == 0
    assert db.get_session("compacted") is not None
    assert len(db.get_messages("compacted", include_inactive=True)) == 4


def test_genuinely_empty_sessions_are_still_swept(db):
    """The fix must not neuter the feature: no rows at all is still empty."""
    db.create_session("ghost", source="desktop")
    db.end_session("ghost", end_reason="tui_close")

    _seed(db, "populated")
    db.end_session("populated", end_reason="tui_close")

    assert db.count_empty_sessions() == 1
    assert db.delete_empty_sessions() == 1
    assert db.get_session("ghost") is None
    assert db.get_session("populated") is not None


def test_count_and_delete_agree_on_a_mixed_database(db):
    """The button's N and the sweep it triggers must never disagree.

    They read one shared selector; this pins that they still agree once
    drifted-counter rows are in the mix.
    """
    db.create_session("ghost", source="desktop")
    db.end_session("ghost", end_reason="tui_close")

    _seed(db, "rewound")
    db.replace_messages("rewound", [], archive_dropped=True)
    db.end_session("rewound", end_reason="ws_orphan_reap")

    _seed(db, "live")  # never ended — not a candidate either way

    counted = db.count_empty_sessions()
    assert counted == 1
    assert db.delete_empty_sessions() == counted
