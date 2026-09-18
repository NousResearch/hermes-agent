"""Deleting a session must delete its whole compression lineage.

Deleting only the visible compression *tip* orphans the compressed ancestors:
their ``parent_session_id`` is NULLed (FK safety) and the rows survive, so the
old snapshot resurfaces in session pickers on the next refresh — the session
the user just deleted comes back. A delete must walk the same
compression-continuation edge the projection uses (parent ended
``end_reason='compression'``, child is not a branch / delegate / tool row) and
remove the entire lineage. (#53684)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_state import SessionDB


def _mark_compression_parent(db: SessionDB, sid: str) -> None:
    db._conn.execute(
        "UPDATE sessions SET ended_at = 1.0, end_reason = 'compression' WHERE id = ?", (sid,)
    )
    db._conn.commit()


def _seed_compression_lineage(db: SessionDB) -> tuple[str, str, str]:
    """root -> mid -> tip linked by compression continuations; return (root, mid, tip)."""
    db.create_session("root-sid", source="test")
    db.create_session("mid-sid", parent_session_id="root-sid", source="test")
    db.create_session("tip-sid", parent_session_id="mid-sid", source="test")
    for sid in ("root-sid", "mid-sid", "tip-sid"):
        db.append_message(sid, "user", f"hello {sid}")
        db.append_message(sid, "assistant", f"answer {sid}")
    _mark_compression_parent(db, "root-sid")
    _mark_compression_parent(db, "mid-sid")
    return "root-sid", "mid-sid", "tip-sid"


def _session_ids(db: SessionDB) -> set[str]:
    return {row["id"] for row in db._conn.execute("SELECT id FROM sessions").fetchall()}


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    d.close()


class TestCompressionLineageDelete:
    def test_deleting_the_tip_removes_the_whole_lineage(self, db: SessionDB):
        root, mid, tip = _seed_compression_lineage(db)

        assert db.delete_session(tip) is True

        remaining = _session_ids(db)
        assert tip not in remaining
        assert mid not in remaining, "the compressed parent must not survive its tip"
        assert root not in remaining, "the compression root must not survive its tip"

    def test_deleting_the_root_removes_the_whole_lineage(self, db: SessionDB):
        _seed_compression_lineage(db)

        assert db.delete_session("root-sid") is True

        assert _session_ids(db) == set()

    def test_deleting_a_middle_node_removes_the_whole_lineage(self, db: SessionDB):
        _seed_compression_lineage(db)

        assert db.delete_session("mid-sid") is True

        assert _session_ids(db) == set()

    def test_messages_of_every_lineage_member_are_deleted(self, db: SessionDB):
        _seed_compression_lineage(db)

        db.delete_session("tip-sid")

        for sid in ("root-sid", "mid-sid", "tip-sid"):
            n = db._conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (sid,)
            ).fetchone()["n"]
            assert n == 0, f"messages leaked for {sid}"

    def test_a_branch_child_survives_deleting_its_compression_parent(self, db: SessionDB):
        """A branch (``_branched_from``) is NOT a compression continuation and must stay."""
        _seed_compression_lineage(db)
        db.create_session("branch-sid", parent_session_id="mid-sid", source="test")
        db._conn.execute(
            "UPDATE sessions SET model_config = ? WHERE id = ?",
            (json.dumps({"_branched_from": "mid-sid"}), "branch-sid"),
        )
        db._conn.commit()

        assert db.delete_session("tip-sid") is True

        remaining = _session_ids(db)
        assert "branch-sid" in remaining, "a branch must be orphaned, not deleted"
        # Orphaned, so it no longer points at the deleted row.
        row = db._conn.execute(
            "SELECT parent_session_id AS p FROM sessions WHERE id = ?", ("branch-sid",)
        ).fetchone()
        assert row["p"] is None

    def test_a_delegate_child_is_still_cascade_deleted(self, db: SessionDB):
        """The existing delegate cascade must keep working alongside lineage deletion."""
        _seed_compression_lineage(db)
        db.create_session("delegate-sid", parent_session_id="tip-sid", source="test")
        db._conn.execute(
            "UPDATE sessions SET model_config = ? WHERE id = ?",
            (json.dumps({"_delegate_from": "tip-sid"}), "delegate-sid"),
        )
        db._conn.commit()

        assert db.delete_session("tip-sid") is True

        assert "delegate-sid" not in _session_ids(db)

    def test_deleting_an_unrelated_session_leaves_the_lineage_alone(self, db: SessionDB):
        root, mid, tip = _seed_compression_lineage(db)
        db.create_session("other-sid", source="test")

        assert db.delete_session("other-sid") is True

        assert _session_ids(db) == {root, mid, tip}

    def test_bulk_delete_removes_whole_lineages(self, db: SessionDB):
        _seed_compression_lineage(db)
        db.create_session("solo-sid", source="test")

        assert db.delete_sessions(["tip-sid", "solo-sid"]) == 2

        assert _session_ids(db) == set()

    def test_bulk_delete_skips_unknown_ids_but_still_deletes_known_lineages(self, db: SessionDB):
        _seed_compression_lineage(db)

        # Unknown ids are skipped; the tip's lineage still dies in full.
        assert db.delete_sessions(["tip-sid", "does-not-exist"]) == 1

        assert _session_ids(db) == set()
