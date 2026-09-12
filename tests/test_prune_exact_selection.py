"""Tests for ``SessionDB.prune_exact_selection`` — the frozen-manifest Prune
primitive backing a future ``hermes sessions prune --selection-file``.

Unlike ``prune_sessions`` (filter-driven, may select a different set than a
caller previously reviewed), this takes an explicit, caller-supplied list of
physical session IDs and deletes exactly that set or none of it. It exists so
an external Store/Verify pipeline (e.g. a private cold-archive helper) can
review precisely which rows a Prune will touch, then hand back the same exact
ID list for deletion — with no filter re-evaluation gap between review and
delete.
"""

from __future__ import annotations

import json
import time

import pytest

import hermes_state_maintenance
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(db_path=tmp_path / "state.db")
    yield database
    database.close()


def _mk_ended(
    db,
    sid,
    *,
    source="cli",
    pinned=False,
    archived=True,
    parent_session_id=None,
):
    db.create_session(
        session_id=sid, source=source, parent_session_id=parent_session_id
    )
    db.end_session(sid, end_reason="done")
    if archived:
        db.set_session_archived(sid, True)
    if pinned:
        db.set_session_pinned(sid, True)


class TestPruneExactSelectionHappyPath:
    def test_deletes_exactly_the_listed_ended_sessions(self, db):
        _mk_ended(db, "a")
        _mk_ended(db, "b")
        _mk_ended(db, "c")

        result = db.prune_exact_selection(["a", "b"])

        assert result["count"] == 2
        assert set(result["deleted"]) == {"a", "b"}
        assert db.get_session("a") is None
        assert db.get_session("b") is None
        assert db.get_session("c") is not None

    def test_chunks_every_sql_id_probe_under_a_tight_parameter_budget(self, db, monkeypatch):
        """Exact Prune must retain its all-or-nothing contract when a frozen
        plan is larger than the SQLite parameter budget."""
        ids = ["a", "b", "c"]
        for session_id in ids:
            _mk_ended(db, session_id)

        original_placeholders = hermes_state_maintenance._placeholders

        def bounded_placeholders(items):
            count = items if isinstance(items, int) else len(items)
            if count > 1:
                raise AssertionError("exact Prune built an oversized SQL IN list")
            return original_placeholders(items)

        monkeypatch.setattr(
            hermes_state_maintenance,
            "_id_chunks",
            lambda values, size=900: ([value] for value in values),
        )
        monkeypatch.setattr(
            hermes_state_maintenance,
            "_placeholders",
            bounded_placeholders,
        )

        result = db.prune_exact_selection(ids)

        assert result == {"count": 3, "deleted": ids}
        assert all(db.get_session(session_id) is None for session_id in ids)

    def test_removes_messages_and_usage_rows_for_deleted_sessions(self, db):
        _mk_ended(db, "a")
        db.append_message("a", role="user", content="hi")

        db.prune_exact_selection(["a"])

        assert (
            db._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", ("a",)
            ).fetchone()[0]
            == 0
        )


class TestPruneExactSelectionFailsClosed:
    def test_missing_id_deletes_nothing(self, db):
        _mk_ended(db, "a")

        with pytest.raises(ValueError, match="does not exist"):
            db.prune_exact_selection(["a", "does-not-exist"])

        # Atomic: the existing, valid id must NOT have been deleted either.
        assert db.get_session("a") is not None

    def test_open_session_in_selection_is_refused(self, db):
        db.create_session(session_id="open", source="cli")
        _mk_ended(db, "a")

        with pytest.raises(ValueError, match="not ended"):
            db.prune_exact_selection(["a", "open"])

        assert db.get_session("a") is not None
        assert db.get_session("open") is not None

    def test_pinned_session_is_refused_by_default(self, db):
        _mk_ended(db, "a", pinned=True)

        with pytest.raises(ValueError, match="pinned"):
            db.prune_exact_selection(["a"])

        assert db.get_session("a") is not None

    def test_pinned_session_can_be_included_explicitly(self, db):
        _mk_ended(db, "a", pinned=True)

        result = db.prune_exact_selection(["a"], include_pinned=True)

        assert result["count"] == 1
        assert db.get_session("a") is None

    def test_unarchived_session_is_refused(self, db):
        _mk_ended(db, "a", archived=False)

        with pytest.raises(ValueError, match="not archived"):
            db.prune_exact_selection(["a"])

        assert db.get_session("a") is not None

    def test_empty_selection_is_rejected(self, db):
        with pytest.raises(ValueError, match="empty"):
            db.prune_exact_selection([])

    def test_duplicate_id_is_rejected_instead_of_normalized(self, db):
        _mk_ended(db, "a")

        with pytest.raises(ValueError, match="duplicate"):
            db.prune_exact_selection(["a", "a"])

        assert db.get_session("a") is not None

    def test_removes_transcript_files_only_for_pruned_ids(self, db, tmp_path):
        _mk_ended(db, "a")
        _mk_ended(db, "b")
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        pruned_transcript = sessions_dir / "a.jsonl"
        pruned_transcript.write_text("{}")
        kept_transcript = sessions_dir / "b.jsonl"
        kept_transcript.write_text("{}")

        db.prune_exact_selection(["a"], sessions_dir=sessions_dir)

        assert not pruned_transcript.exists()
        assert kept_transcript.exists()


class TestPruneExactSelectionReferenceSafety:
    def test_uncovered_child_session_blocks_the_whole_operation(self, db):
        """A child outside the selection still points at a row we would delete —
        deleting the parent would silently orphan a session the caller never
        reviewed. Refuse the whole batch rather than orphan it implicitly."""
        _mk_ended(db, "parent")
        _mk_ended(db, "child", parent_session_id="parent")
        _mk_ended(db, "other")

        with pytest.raises(ValueError, match="uncovered"):
            db.prune_exact_selection(["parent", "other"])

        assert db.get_session("parent") is not None
        assert db.get_session("other") is not None

    def test_child_included_in_the_same_selection_is_allowed(self, db):
        """Covering both parent and child in one selection is fine — nothing is
        orphaned because both rows go together."""
        _mk_ended(db, "parent")
        _mk_ended(db, "child", parent_session_id="parent")

        result = db.prune_exact_selection(["parent", "child"])

        assert result["count"] == 2
        assert db.get_session("parent") is None
        assert db.get_session("child") is None

    def test_active_compression_lock_blocks_deletion(self, db):
        """A live compression lock is a durable in-flight reference to the
        session; deleting the row out from under it would corrupt the
        compression pipeline."""
        _mk_ended(db, "locked")
        db._conn.execute(
            "INSERT INTO compression_locks (session_id, holder, acquired_at, expires_at) "
            "VALUES (?, 'worker-1', ?, ?)",
            ("locked", time.time(), time.time() + 60),
        )
        db._conn.commit()

        with pytest.raises(ValueError, match="compression_locks"):
            db.prune_exact_selection(["locked"])

        assert db.get_session("locked") is not None

    @pytest.mark.parametrize("namespace", ["goal", "loop", "heartbeat"])
    def test_session_state_meta_reference_blocks_deletion(self, db, namespace):
        _mk_ended(db, "stateful")
        db.set_meta(f"{namespace}:stateful", "persisted state")

        with pytest.raises(ValueError, match="state_meta"):
            db.prune_exact_selection(["stateful"])

        assert db.get_session("stateful") is not None

    def test_active_turn_lease_blocks_deletion(self, db):
        """A live turn lease means a conversation is mid-turn on this exact
        session id; deleting it would race the in-flight write."""
        _mk_ended(db, "leased")
        db._conn.execute(
            "INSERT INTO session_turn_leases (conversation_id, holder, acquired_at, expires_at) "
            "VALUES (?, 'worker-1', ?, ?)",
            ("leased", time.time(), time.time() + 60),
        )
        db._conn.commit()

        with pytest.raises(ValueError, match="session_turn_leases"):
            db.prune_exact_selection(["leased"])

        assert db.get_session("leased") is not None

    def test_async_delegation_origin_reference_blocks_deletion(self, db):
        """A delegation row naming this session as its origin is a durable
        cross-session reference the caller did not review."""
        _mk_ended(db, "origin")
        db._conn.execute(
            "INSERT INTO async_delegations "
            "(delegation_id, origin_session, state, dispatched_at, updated_at) "
            "VALUES ('deleg-1', ?, 'pending', ?, ?)",
            ("origin", time.time(), time.time()),
        )
        db._conn.commit()

        with pytest.raises(ValueError, match="async_delegations"):
            db.prune_exact_selection(["origin"])

        assert db.get_session("origin") is not None

    def test_async_delegation_parent_reference_blocks_deletion(self, db):
        _mk_ended(db, "target")
        db._conn.execute(
            "INSERT INTO async_delegations "
            "(delegation_id, origin_session, parent_session_id, state, dispatched_at, updated_at) "
            "VALUES ('deleg-2', 'unrelated-origin', ?, 'pending', ?, ?)",
            ("target", time.time(), time.time()),
        )
        db._conn.commit()

        with pytest.raises(ValueError, match="async_delegations"):
            db.prune_exact_selection(["target"])

        assert db.get_session("target") is not None

    def test_gateway_routing_reference_blocks_deletion(self, db):
        """A persisted gateway route can resume its embedded session id after
        restart, so exact Prune must refuse rather than leave a dangling route."""
        _mk_ended(db, "routed")
        db.save_gateway_routing_entry(
            "agent:main:telegram:dm:chat-1",
            json.dumps({"session_id": "routed"}),
            scope="/tmp/sessions",
        )

        with pytest.raises(ValueError, match="gateway_routing"):
            db.prune_exact_selection(["routed"])

        assert db.get_session("routed") is not None

    def test_unrelated_gateway_routing_entry_is_preserved(self, db):
        _mk_ended(db, "selected")
        db.save_gateway_routing_entry(
            "agent:main:telegram:dm:chat-2",
            json.dumps({"session_id": "unrelated"}),
            scope="/tmp/sessions",
        )

        result = db.prune_exact_selection(["selected"])

        assert result["count"] == 1
        assert db.load_gateway_routing_entries(scope="/tmp/sessions") == {
            "agent:main:telegram:dm:chat-2": json.dumps({"session_id": "unrelated"})
        }

    def test_gateway_routing_check_never_parses_json_in_python(self, db):
        """The reference-safety scan must push matching into SQLite (json1)
        rather than fetch every entry_json blob and json.loads() it in
        Python -- that would be O(total routes) CPU/memory under an
        exclusive write lock. The production module deliberately carries no
        top-level ``json`` import for this mixin; a reviewer reintroducing
        Python-side JSON parsing here should fail this test."""
        assert not hasattr(hermes_state_maintenance, "json"), (
            "hermes_state_maintenance must not import json: the "
            "gateway_routing reference-safety scan is SQL-side (json1) only"
        )

        _mk_ended(db, "selected")
        for n in range(50):
            db.save_gateway_routing_entry(
                f"agent:main:telegram:dm:chat-{n}",
                json.dumps({"session_id": f"unrelated-{n}"}),
                scope="/tmp/sessions",
            )

        result = db.prune_exact_selection(["selected"])

        assert result["count"] == 1

    def test_gateway_routing_check_still_blocks_amid_many_unrelated_rows(self, db):
        """A malformed row must still fail closed even when surrounded by many
        well-formed unrelated rows that the SQL-side scoped query would
        otherwise skip entirely."""
        _mk_ended(db, "selected")
        for n in range(50):
            db.save_gateway_routing_entry(
                f"agent:main:telegram:dm:chat-{n}",
                json.dumps({"session_id": f"unrelated-{n}"}),
                scope="/tmp/sessions",
            )
        db.save_gateway_routing_entry(
            "agent:main:telegram:dm:damaged",
            "not-json",
            scope="/tmp/sessions",
        )

        with pytest.raises(ValueError, match="cannot verify gateway_routing"):
            db.prune_exact_selection(["selected"])

        assert db.get_session("selected") is not None

    def test_unverifiable_non_object_gateway_routing_entry_blocks_deletion(self, db):
        _mk_ended(db, "selected")
        db.save_gateway_routing_entry(
            "agent:main:telegram:dm:wrong-shape",
            json.dumps(["not", "a", "routing", "object"]),
            scope="/tmp/sessions",
        )

        with pytest.raises(ValueError, match="cannot verify gateway_routing"):
            db.prune_exact_selection(["selected"])

        assert db.get_session("selected") is not None

    def test_unverifiable_gateway_routing_entry_blocks_deletion(self, db):
        _mk_ended(db, "selected")
        db.save_gateway_routing_entry(
            "agent:main:telegram:dm:damaged",
            "not-json",
            scope="/tmp/sessions",
        )

        with pytest.raises(ValueError, match="cannot verify gateway_routing"):
            db.prune_exact_selection(["selected"])

        assert db.get_session("selected") is not None
