"""The empty-session sweep must not eat a rewound / compacted transcript (#95868).

``count_empty_sessions`` / ``delete_empty_sessions`` back the dashboard's
"Delete empty (N)" affordance. They used to define "empty" as
``sessions.message_count = 0``, which is a denormalized counter over the LIVE
(``active = 1``) rows only.

Two production transcript-rewrite paths reset that counter on purpose while
keeping every dropped turn on disk as ``active = 0``:

* ``replace_messages(..., archive_dropped=True)`` — the rewind / edit /
  regenerate mode added in #82756 precisely so a user can take a turn back
  without the rows being unrecoverable. ``prompt.submit`` reaches it with an
  empty prefix on a confirmed ordinal-0 rewind (regenerating the very first
  user turn), which is the reachable production shape.
* ``archive_and_compact`` — in-place compaction, which archives the
  pre-compaction transcript under the same session id (#38763). It normally
  publishes at least a summary row, so it lands on the same shape only when
  the live set comes back empty; pinned here as defense in depth.

Either way the row reports ``message_count = 0`` while still holding its
entire recoverable history, and those soft-archived rows are the ONLY copy —
which is the whole point of the archive-instead-of-delete guarantee that
#70516 / #80763 / #82756 were fixed to provide.

A gateway reload is what makes such a row *eligible*: every detached session
gets ``ended_at`` stamped (``end_reason='ws_orphan_reap'``), which satisfies
the sweep's ``ended_at IS NOT NULL`` gate. The next "Delete empty" click then
hard-deleted the session row AND ``DELETE FROM messages`` — silently, with no
log line to trace it by.

These tests pin the counter drift as real, and pin the sweep's refusal to act
on it.
"""

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import (
    append_message,
    archive_and_compact,
    get_messages,
    replace_messages,
)
from persistence.sqlite.conversation._store.sessions import (
    count_empty_sessions,
    create_session,
    delete_empty_sessions,
    end_session,
    get_session,
)


@pytest.fixture()
def db(tmp_path):
    return StoreHolder(tmp_path / "state.db")


@pytest.fixture()
def media_root(tmp_path):
    return str(tmp_path / "media")


def _seed(db, session_id, turns=4):
    """A populated, ended desktop chat — the shape #95868 lost."""
    create_session(db, session_id, source="desktop", model="test-model")
    for i in range(turns):
        append_message(
            db,
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

    replace_messages(db, "rewound", [], archive_dropped=True)

    counter, rows = _row_counts(db, "rewound")
    assert counter == 0, "message_count tracks the live set only"
    assert rows == 4, "the dropped turns stay on disk as the recoverable copy"
    assert len(get_messages(db, "rewound", include_inactive=True)) == 4


def test_sweep_spares_a_rewound_session(db, media_root):
    """A rewound chat is not empty and must survive the sweep."""
    _seed(db, "rewound")
    replace_messages(db, "rewound", [], archive_dropped=True)
    # A gateway reload stamps ended_at on every detached session, which is what
    # makes the row eligible for the sweep in the first place.
    end_session(db, "rewound", end_reason="ws_orphan_reap")

    assert count_empty_sessions(db) == 0
    assert delete_empty_sessions(db, media_root=media_root) == ()
    assert get_session(db, "rewound") is not None
    assert len(get_messages(db, "rewound", include_inactive=True)) == 4


def test_sweep_spares_an_in_place_compacted_session(db, media_root):
    """``archive_and_compact`` leaves the same zero-counter/live-rows shape."""
    _seed(db, "compacted")
    archive_and_compact(db, "compacted", [])
    assert _row_counts(db, "compacted") == (0, 4)

    end_session(db, "compacted", end_reason="ws_orphan_reap")

    assert count_empty_sessions(db) == 0
    assert delete_empty_sessions(db, media_root=media_root) == ()
    assert get_session(db, "compacted") is not None
    assert len(get_messages(db, "compacted", include_inactive=True)) == 4


def test_genuinely_empty_sessions_are_still_swept(db, media_root):
    """The fix must not neuter the feature: no rows at all is still empty."""
    create_session(db, "ghost", source="desktop")
    end_session(db, "ghost", end_reason="tui_close")

    _seed(db, "populated")
    end_session(db, "populated", end_reason="tui_close")

    assert count_empty_sessions(db) == 1
    assert len(delete_empty_sessions(db, media_root=media_root)) == 1
    assert get_session(db, "ghost") is None
    assert get_session(db, "populated") is not None


def test_count_and_delete_agree_on_a_mixed_database(db, media_root):
    """The button's N and the sweep it triggers must never disagree.

    They read one shared selector; this pins that they still agree once
    drifted-counter rows are in the mix.
    """
    create_session(db, "ghost", source="desktop")
    end_session(db, "ghost", end_reason="tui_close")

    _seed(db, "rewound")
    replace_messages(db, "rewound", [], archive_dropped=True)
    end_session(db, "rewound", end_reason="ws_orphan_reap")

    _seed(db, "live")  # never ended — not a candidate either way

    counted = count_empty_sessions(db)
    assert counted == 1
    assert len(delete_empty_sessions(db, media_root=media_root)) == counted
