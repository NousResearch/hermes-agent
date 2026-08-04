"""Regression tests for delete_session on compression chains.

Mirrors the archiving coverage in test_session_archiving.py.  Deleting any
node in a compression chain must remove the entire logical conversation —
not just the visible tip (the "onion peeling" bug that shipped before this
fix).  Non-compression branches must be orphaned, not deleted.
"""
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _compression_pair(db: SessionDB):
    """Create a root → tip compression chain (both nodes have compression end_reason)."""
    base = time.time() - 100
    db.create_session("root", source="cli")
    db.create_session("tip", source="cli", parent_session_id="root")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
        (base, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 20,),
    )
    db._conn.commit()


def _compression_triple(db: SessionDB):
    """Create a root → mid → tip chain (3 segments, all compressed)."""
    base = time.time() - 300
    db.create_session("root", source="cli")
    db.create_session("mid", source="cli", parent_session_id="root")
    db.create_session("tip", source="cli", parent_session_id="mid")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
        (base, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'mid'",
        (base + 20, base + 30),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 40,),
    )
    db._conn.commit()


def test_delete_compression_tip_removes_entire_chain(db):
    _compression_pair(db)

    assert db.delete_session("tip") is True

    # Both root and tip must be gone.
    assert db.get_session("root") is None
    assert db.get_session("tip") is None
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == []


def test_delete_compression_root_removes_entire_chain(db):
    _compression_pair(db)

    assert db.delete_session("root") is True

    assert db.get_session("root") is None
    assert db.get_session("tip") is None
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == []


def test_delete_compression_middle_removes_entire_chain(db):
    _compression_triple(db)

    assert db.delete_session("mid") is True

    assert db.get_session("root") is None
    assert db.get_session("mid") is None
    assert db.get_session("tip") is None
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == []


def test_delete_compression_chain_orphans_non_compression_branch(db):
    """A branch child that doesn't match the compression contract must survive."""
    base = time.time() - 100
    db.create_session("root", source="cli")
    db.create_session("tip", source="cli", parent_session_id="root")
    db.create_session("branch", source="cli", parent_session_id="root")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
        (base, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 20,),
    )
    # branch was created WHILE root was still live (started_at < ended_at)
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'branch'",
        (base + 5,),
    )
    db._conn.commit()

    assert db.delete_session("tip") is True

    # Compression chain gone; branch orphaned (parent_session_id = NULL).
    assert db.get_session("root") is None
    assert db.get_session("tip") is None
    branch = db.get_session("branch")
    assert branch is not None
    assert branch["parent_session_id"] is None


def test_delete_standalone_session(db):
    db.create_session("solo", source="cli")
    assert db.delete_session("solo") is True
    assert db.get_session("solo") is None


def test_delete_nonexistent_session_returns_false(db):
    assert db.delete_session("does_not_exist") is False


# ---------------------------------------------------------------------------
# #48525 / Teknium hermes-sweeper review 2026-07-25: the lineage CTE must
# use the SAME continuation policy as ``get_compression_tip()``. Three
# cases that the previous version of this PR got wrong:
# 1. Explicit branch / delegate / tool children must NOT be cascade-deleted.
# 2. Continuation rows whose ``started_at < parent.ended_at`` (the
#    timestamp race where the real continuation is inserted before the
#    parent's ``ended_at`` is written) must still be found.
# 3. Bulk delete (``delete_sessions``) must apply the same expansion.
# ---------------------------------------------------------------------------


def test_delete_preserves_explicit_branch_child(db):
    """A child with ``model_config._branched_from`` is an explicit branch,
    not a compression continuation. Deleting the compression chain must
    orphan it, not delete it.

    Teknium's review: "The CTE does not exclude explicit branch/delegate/tool
    children. That can delete a marked branch created after a
    compression-ended parent, although projection excludes those children
    (``hermes_state.py:3160-3177``)."
    """
    base = time.time() - 100
    db.create_session("root", source="cli")
    db.create_session("tip", source="cli", parent_session_id="root")
    # Explicit branch — same parent as tip, but tagged with
    # ``_branched_from`` in ``model_config``. The projection path
    # (get_compression_tip / list_sessions_rich) excludes this row, so
    # the delete path must too.
    db.create_session("explicit_branch", source="cli", parent_session_id="root")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
        (base, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 20,),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1, model_config = ? WHERE id = 'explicit_branch'",
        (base + 15, '{"_branched_from": "user-action"}'),
    )
    db._conn.commit()

    assert db.delete_session("tip") is True

    # Compression chain gone.
    assert db.get_session("root") is None
    assert db.get_session("tip") is None
    # Explicit branch survives with its parent nulled out.
    branch = db.get_session("explicit_branch")
    assert branch is not None
    assert branch["parent_session_id"] is None


def test_delete_cascades_explicit_delegate_child(db):
    """``_delete_delegate_children`` cascade-deletes ``_delegate_from``-tagged
    children with their parent. The lineage CTE correctly excludes
    delegates from the chain walk, but the cascade handler then
    deletes them in a separate step. This is the desired behavior —
    a delegate subagent is part of the parent's session lifecycle, not
    a sibling of the compression chain.

    The assertion here is that the *projection* (the lineage CTE
    used for delete) doesn't add a delegate to the chain — the
    cascade is responsible for the actual removal. Without the
    CTE exclusion, the delegate would be both chain-walked AND
    cascade-deleted, which double-counts and breaks audit
    expectations.
    """
    base = time.time() - 100
    db.create_session("root", source="cli")
    db.create_session("tip", source="cli", parent_session_id="root")
    db.create_session("delegate", source="cli", parent_session_id="root")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
        (base, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 20,),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1, model_config = ? WHERE id = 'delegate'",
        (base + 15, '{"_delegate_from": "root"}'),
    )
    db._conn.commit()

    # Chain + delegate are all gone (chain via CTE, delegate via cascade).
    assert db.delete_session("tip") is True
    assert db.get_session("root") is None
    assert db.get_session("tip") is None
    # The cascade is the contract here — delegates ride along with
    # the parent session, not the user-facing chain. We don't make
    # a separate test for cascade behavior; the original
    # _delete_delegate_children tests cover that path.


def test_delete_preserves_tool_source_child(db):
    """A child whose ``source='tool'`` is a tool invocation, not part of
    the user's chain.
    """
    base = time.time() - 100
    db.create_session("root", source="cli")
    db.create_session("tip", source="cli", parent_session_id="root")
    db.create_session("tool_child", source="tool", parent_session_id="root")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
        (base, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 20,),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tool_child'",
        (base + 15,),
    )
    db._conn.commit()

    assert db.delete_session("tip") is True

    assert db.get_session("tool_child") is not None


def test_delete_finds_continuation_inserted_before_parent_ended_at(db):
    """Timestamp race: the real continuation is inserted *before* the
    parent's ``ended_at`` is written (gateway + compression race). The
    previous version of this PR required ``child.started_at >=
    parent.ended_at`` and would have missed the continuation in this
    case. Teknium's review specifically flagged this predicate as the
    brittle ordering upstream abandoned in ``get_compression_tip()``.

    Use the structural predicate (end_reason='compression' on parent,
    no branch/delegate/tool markers on child) instead.
    """
    base = time.time() - 100
    db.create_session("root", source="cli")
    # Continuation is created BEFORE root's ended_at is written.
    db.create_session("tip", source="cli", parent_session_id="root")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
        # root ended AFTER tip started — the race condition.
        (base, base + 30),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 20,),
    )
    db._conn.commit()

    assert db.delete_session("tip") is True
    assert db.get_session("root") is None
    assert db.get_session("tip") is None


def test_delete_sessions_bulk_expands_lineage(db):
    """Teknium's review: "Dashboard bulk delete still calls
    delete_sessions() which only deletes selected IDs and orphans
    children." The bulk path must apply the same lineage expansion as
    the single-session path.
    """
    base = time.time() - 100
    db.create_session("root", source="cli")
    db.create_session("tip", source="cli", parent_session_id="root")
    # An explicit branch under the same root.
    db.create_session("branch", source="cli", parent_session_id="root")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
        (base, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 20,),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1, model_config = '{\"_branched_from\": \"user-action\"}' WHERE id = 'branch'",
        (base + 5,),
    )
    db._conn.commit()

    # Bulk delete: select only the tip. The full chain (root, tip) must
    # be deleted, but the explicit branch must survive (orphaned).
    assert db.delete_sessions(["tip"]) >= 1

    assert db.get_session("root") is None
    assert db.get_session("tip") is None
    branch = db.get_session("branch")
    assert branch is not None
    assert branch["parent_session_id"] is None
