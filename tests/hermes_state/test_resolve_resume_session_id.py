"""Regression guard for #15000: --resume <id> after compression loses messages.

Context compression ends the current session and forks a new child session
(linked by ``parent_session_id``). The SQLite flush cursor is reset, so
only the latest descendant ends up with rows in the ``messages`` table —
the parent row has ``message_count = 0``. ``hermes --resume <parent_id>``
used to load zero rows and show a blank chat.

``SessionDB.resolve_resume_session_id()`` walks the parent → child chain
and redirects to the first descendant that actually has messages. These
tests pin that behaviour.
"""
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _make_chain(db: SessionDB, ids_with_parent):
    """Create sessions in order, forcing started_at so ordering is deterministic."""
    base = int(time.time()) - 10_000
    for i, (sid, parent) in enumerate(ids_with_parent):
        db.create_session(sid, source="cli", parent_session_id=parent)
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (base + i * 100, sid),
        )
    db._conn.commit()


def test_returns_self_when_only_parent_has_messages(db):
    # When a session already has messages AND no descendant has messages,
    # it should still be returned.  The chain walk finds no better candidate.
    _make_chain(db, [("root", None), ("child", "root")])
    db.append_message("root", role="user", content="hi")
    assert db.resolve_resume_session_id("root") == "root"










def test_walks_from_middle_of_chain(db):
    # If the user happens to know an intermediate ID, we still find the msg-bearing descendant.
    _make_chain(db, [("a", None), ("b", "a"), ("c", "b"), ("d", "c")])
    db.append_message("d", role="user", content="x")
    assert db.resolve_resume_session_id("b") == "d"
    assert db.resolve_resume_session_id("c") == "d"


def test_follows_compression_tip_when_parent_retains_messages(db):
    # The bug behind the desktop "I came back and the reply isn't there" report
    # on large sessions: auto-compression ends the live session and forks a
    # continuation child, but a long parent keeps its own flushed message rows.
    # The empty-head walk below never redirects a non-empty head, so resuming
    # the parent id reloaded the pre-compression transcript and the response
    # generated *after* compression (which lives in the continuation) was
    # missing. resolve_resume_session_id must follow the compression-tip chain
    # forward even when the parent still has messages.
    base = int(time.time()) - 10_000
    db.create_session("root", source="cli")
    db.append_message("root", role="user", content="pre-compression turn")
    db.end_session("root", "compression")
    db.create_session("cont", source="cli", parent_session_id="root")
    db.append_message("cont", role="assistant", content="post-compression reply")
    # Force deterministic ordering so the continuation's started_at is clearly
    # at/after the parent's ended_at (the get_compression_tip discriminator).
    conn = db._conn
    assert conn is not None
    conn.execute("UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = 'root'", (base, base + 50))
    conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'cont'", (base + 100,))
    conn.commit()

    assert db.resolve_resume_session_id("root") == "cont"




def test_prefers_most_recent_child_when_fork_exists(db):
    # If a session was somehow forked (two children), pick the latest one.
    # In practice, compression only produces single-chain shape, but the helper
    # should degrade gracefully.
    _make_chain(db, [
        ("parent", None),
        ("older_fork", "parent"),
        ("newer_fork", "parent"),
    ])
    db.append_message("newer_fork", role="user", content="x")
    assert db.resolve_resume_session_id("parent") == "newer_fork"




def test_compression_tip_prefers_continuation_over_retained_subagent_sibling(db):
    # Read-only delegate/subagent rows can be retained for audit/debugging under
    # the compressed parent, but they are not the user's focused continuation.
    # A newer subagent sibling must not consume the projected list/resume slot.
    base = int(time.time()) - 10_000
    db.create_session("root", source="cli")
    db.append_message("root", role="user", content="pre-compression turn")
    db.end_session("root", "compression")

    db.create_session("focused_cont", source="cli", parent_session_id="root")
    db.append_message("focused_cont", role="assistant", content="real continuation")

    # Shape produced by retained audit/debug child rows: linked to the parent,
    # source=subagent, no deletion required, and no reliable legacy marker.
    db.create_session("audit_subagent", source="subagent", parent_session_id="root")
    db.append_message("audit_subagent", role="assistant", content="read-only audit child")

    # Separate normal conversation: newer than the real continuation, older
    # than the audit child. If the audit child leaks into the compression-chain
    # CTE, the focused continuation incorrectly outranks this row.
    db.create_session("solo", source="cli")
    db.append_message("solo", role="user", content="separate normal chat")

    conn = db._conn
    assert conn is not None
    conn.execute("UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = 'root'", (base, base + 100))
    conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'focused_cont'", (base + 90,))
    conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'audit_subagent'", (base + 200,))
    conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'solo'", (base + 180,))
    conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = 'root'", (base + 10,))
    conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = 'focused_cont'", (base + 110,))
    conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = 'solo'", (base + 220,))
    conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = 'audit_subagent'", (base + 300,))
    conn.commit()

    assert db.get_compression_tip("root") == "focused_cont"
    assert db.resolve_resume_session_id("root") == "focused_cont"

    listed = db.list_sessions_rich(source="cli", limit=10, order_by_last_active=True)
    assert [row["id"] for row in listed] == ["solo", "focused_cont"]
    assert listed[1]["_lineage_root_id"] == "root"

