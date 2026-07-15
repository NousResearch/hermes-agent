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


def test_redirects_from_empty_head_to_descendant_with_messages(db):
    # Reproducer shape from #15000: 6 sessions, only the 5th holds messages.
    _make_chain(db, [
        ("head",   None),
        ("mid1",   "head"),
        ("mid2",   "mid1"),
        ("mid3",   "mid2"),
        ("bulk",   "mid3"),    # has messages
        ("tail",   "bulk"),    # empty tail after another compression
    ])
    for i in range(5):
        db.append_message("bulk", role="user", content=f"msg {i}")

    assert db.resolve_resume_session_id("head") == "bulk"
def test_returns_self_when_only_parent_has_messages(db):
    # When a session already has messages AND no descendant has messages,
    # it should still be returned.  The chain walk finds no better candidate.
    _make_chain(db, [("root", None), ("child", "root")])
    db.append_message("root", role="user", content="hi")
    assert db.resolve_resume_session_id("root") == "root"


def test_returns_self_when_no_descendant_has_messages(db):
    _make_chain(db, [("root", None), ("child1", "root"), ("child2", "child1")])
    assert db.resolve_resume_session_id("root") == "root"


def test_returns_self_for_isolated_session(db):
    db.create_session("isolated", source="cli")
    assert db.resolve_resume_session_id("isolated") == "isolated"


def test_returns_self_for_nonexistent_session(db):
    assert db.resolve_resume_session_id("does_not_exist") == "does_not_exist"


def test_empty_session_id_passthrough(db):
    assert db.resolve_resume_session_id("") == ""
    assert db.resolve_resume_session_id(None) is None


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


def test_compression_tip_not_confused_with_delegation_child(db):
    # A delegation/branch child is created while the parent is still live (the
    # parent is NOT ended with end_reason='compression'), so resuming the
    # parent must stay on the parent, not get hijacked into the subagent branch.
    base = int(time.time()) - 10_000
    db.create_session("conv", source="cli")
    db.append_message("conv", role="user", content="parent turn")
    # Real delegate/subagent sessions carry the `_delegate_from` marker
    # (set in delegate_tool.py) — that marker, not timing, is what
    # distinguishes them from a compression continuation.
    db.create_session(
        "subagent",
        source="subagent",
        parent_session_id="conv",
        model_config={"_delegate_from": "conv"},
    )
    db.append_message("subagent", role="assistant", content="delegated work")
    conn = db._conn
    assert conn is not None
    conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'conv'", (base,))
    conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'subagent'", (base + 100,))
    conn.commit()

    assert db.resolve_resume_session_id("conv") == "conv"


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


def test_redirects_from_message_bearing_parent_to_child(db):
    """Fix for problem 2: parent has messages AND child also has messages.

    After context compression the parent holds old messages but the child
    is the active continuation session.  resolve_resume_session_id should
    prefer the latest descendant with messages, not short-circuit on the
    parent.
    """
    _make_chain(db, [
        ("original", None),
        ("continued", "original"),
    ])
    # Both parent and child have messages
    db.append_message("original", role="user", content="old msg")
    db.append_message("original", role="assistant", content="old reply")
    db.append_message("continued", role="user", content="new msg")
    db.append_message("continued", role="assistant", content="new reply")

    assert db.resolve_resume_session_id("original") == "continued"


def test_compression_tip_handles_pre_ended_real_child_and_ws_orphan_sibling(db):
    # Real desktop repro shape from a long GUI session:
    #
    #   root --compression--> real continuation --compression--> live tip
    #     \
    #      `-- stale websocket sibling ended by ws_orphan_reap
    #
    # The real continuation row can be inserted before root.ended_at is written,
    # so the old child.started_at >= parent.ended_at discriminator rejects it and
    # follows the stale websocket sibling instead. That makes the GUI look like
    # the latest conversation was lost. Resuming root must land on live_tip.
    base = int(time.time()) - 10_000
    db.create_session("root", source="tui")
    db.append_message("root", role="user", content="pre-compression")
    db.end_session("root", "compression")

    db.create_session("real_cont", source="tui", parent_session_id="root")
    db.append_message("real_cont", role="user", content="real continuation")
    db.end_session("real_cont", "compression")

    db.create_session("ws_orphan", source="tui", parent_session_id="root")
    db.append_message("ws_orphan", role="user", content="stale websocket")
    db.end_session("ws_orphan", "ws_orphan_reap")

    db.create_session("live_tip", source="tui", parent_session_id="real_cont")
    db.append_message("live_tip", role="user", content="latest real turn")

    conn = db._conn
    assert conn is not None
    conn.execute("UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = 'root'", (base, base + 1000))
    # The real continuation starts before root.ended_at, exactly the race that
    # broke the old timestamp-based chain walk.
    conn.execute("UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = 'real_cont'", (base + 500, base + 2000))
    conn.execute("UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = 'ws_orphan'", (base + 1000, base + 3000))
    conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'live_tip'", (base + 2000,))
    conn.commit()

    assert db.get_compression_tip("root") == "live_tip"
    assert db.resolve_resume_session_id("root") == "live_tip"

    listed = db.list_sessions_rich(limit=10, order_by_last_active=True)
    ids = {row["id"] for row in listed}
    assert "live_tip" in ids
    assert "real_cont" not in ids
    assert "ws_orphan" not in ids

