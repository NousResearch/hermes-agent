"""Resume resolution follows compression continuations only.

Context compression ends the current session and forks a new child session
(linked by ``parent_session_id``). Resuming any point in that chain should
land on the latest compression tip, even when earlier rows already contain
messages.

``SessionDB.resolve_resume_session_id()`` must use the same continuation
rules as ``get_compression_tip()``: follow child rows only when the parent
ended with ``end_reason='compression'`` and the child started after that
end time. Delegate/subagent children and branch children must not be treated
as resume continuations.
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


def _set_started_at(db: SessionDB, session_id: str, started_at: float):
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (started_at, session_id),
    )
    db._conn.commit()


def _end_at(db: SessionDB, session_id: str, ended_at: float, end_reason: str):
    db._conn.execute(
        "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?",
        (ended_at, end_reason, session_id),
    )
    db._conn.commit()


def _make_compression_chain(db: SessionDB, include_mid_message: bool = False):
    """Create root -> mid -> tip using valid compression-continuation edges."""
    base = int(time.time()) - 10_000

    db.create_session("root", source="cli")
    _set_started_at(db, "root", base)
    db.append_message("root", role="user", content="root already flushed")
    _end_at(db, "root", base + 100, "compression")

    db.create_session("mid", source="cli", parent_session_id="root")
    _set_started_at(db, "mid", base + 101)
    if include_mid_message:
        db.append_message("mid", role="assistant", content="mid already flushed")
    _end_at(db, "mid", base + 200, "compression")

    db.create_session("tip", source="cli", parent_session_id="mid")
    _set_started_at(db, "tip", base + 201)
    db.append_message("tip", role="user", content="latest live message")


def test_compression_chain_resolves_to_tip_even_when_requested_session_has_messages(db):
    _make_compression_chain(db, include_mid_message=True)

    assert db.resolve_resume_session_id("root") == "tip"
    assert db.resolve_resume_session_id("mid") == "tip"


def test_returns_self_when_no_compression_tip_exists(db):
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


def test_does_not_follow_delegate_child_created_before_parent_ended(db):
    base = int(time.time()) - 10_000
    db.create_session("root", source="cli")
    _set_started_at(db, "root", base)
    db.append_message("root", role="user", content="parent conversation")

    db.create_session("delegate", source="cli", parent_session_id="root")
    _set_started_at(db, "delegate", base + 10)
    db.append_message("delegate", role="user", content="subagent work")

    _end_at(db, "root", base + 100, "compression")

    assert db.resolve_resume_session_id("root") == "root"


def test_does_not_follow_child_when_parent_was_not_compressed(db):
    base = int(time.time()) - 10_000
    db.create_session("root", source="cli")
    _set_started_at(db, "root", base)
    db.append_message("root", role="user", content="parent conversation")
    _end_at(db, "root", base + 100, "user_exit")

    db.create_session("child", source="cli", parent_session_id="root")
    _set_started_at(db, "child", base + 101)
    db.append_message("child", role="user", content="not a compression continuation")

    assert db.resolve_resume_session_id("root") == "root"


def test_does_not_follow_branch_child(db):
    base = int(time.time()) - 10_000
    db.create_session("parent", source="cli")
    _set_started_at(db, "parent", base)
    _end_at(db, "parent", base + 100, "branched")

    db.create_session("branch", source="cli", parent_session_id="parent")
    _set_started_at(db, "branch", base + 101)
    db.append_message("branch", role="user", content="branch conversation")

    assert db.resolve_resume_session_id("parent") == "parent"


def test_legacy_empty_compression_head_resolves_to_tip(db):
    _make_compression_chain(db)

    assert db.resolve_resume_session_id("root") == "tip"
    assert db.resolve_resume_session_id("mid") == "tip"
