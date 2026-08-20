"""Tests for the update_session_meta fix.

Verifies that:
1. SessionDB.update_session_meta() exists and works correctly via the
   public _execute_write path (not db._lock / db._conn directly).
2. session.py _persist() no longer touches db._lock or db._conn.
3. update_session_meta updates the correct columns atomically.
"""

import ast
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from hermes_state import SessionDB
from acp_adapter.session import SessionManager


def _tmp_db(tmp_path):
    return SessionDB(db_path=tmp_path / "state.db")


def _mock_agent():
    return MagicMock(name="MockAIAgent")


# ---------------------------------------------------------------------------
# hermes_state.SessionDB.update_session_meta — unit tests
# ---------------------------------------------------------------------------

class TestUpdateSessionMeta:
    """Direct unit tests for the new public method."""

    def test_method_exists(self, tmp_path):
        db = _tmp_db(tmp_path)
        assert hasattr(db, "update_session_meta"), (
            "SessionDB must have update_session_meta() public method"
        )
        assert callable(db.update_session_meta)

    def test_updates_model_config(self, tmp_path):
        db = _tmp_db(tmp_path)
        db.create_session("s1", source="acp", model="gpt-4")

        new_meta = json.dumps({"cwd": "/new/path", "provider": "openai"})
        db.update_session_meta("s1", new_meta, model=None)

        row = db.get_session("s1")
        stored = json.loads(row["model_config"])
        assert stored["cwd"] == "/new/path"
        assert stored["provider"] == "openai"



    def test_uses_execute_write_not_private_api(self, tmp_path):
        """update_session_meta must route through _execute_write, not _conn directly."""
        db = _tmp_db(tmp_path)
        db.create_session("s4", source="acp")

        call_count = [0]
        original = db._execute_write

        def patched(fn):
            call_count[0] += 1
            return original(fn)

        db._execute_write = patched
        db.update_session_meta("s4", json.dumps({"cwd": "."}), model="m")

        assert call_count[0] >= 1, (
            "update_session_meta must call _execute_write at least once"
        )



# ---------------------------------------------------------------------------
# AST check: session.py must not access db._lock or db._conn
# ---------------------------------------------------------------------------

class TestNoPrviateDBAccess:
    """_persist() in session.py must not access db._lock or db._conn."""

    def test_no_db_private_lock_access(self):
        with open("acp_adapter/session.py", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)

        violations = []
        for node in ast.walk(tree):
            # Looking for: db._lock  or  db._conn
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "db":
                    if node.attr in ("_lock", "_conn"):
                        violations.append(
                            f"db.{node.attr} at line {node.lineno}"
                        )

        assert violations == [], (
            "session.py accesses private SessionDB internals: "
            + ", ".join(violations)
            + " — use db.update_session_meta() instead"
        )

    def test_persist_calls_update_session_meta(self):
        """AST check: _persist must call db.update_session_meta()."""
        with open("acp_adapter/session.py", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_persist":
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        if isinstance(func, ast.Attribute):
                            if func.attr == "update_session_meta":
                                found = True
                                break
                break

        assert found, (
            "_persist() must call db.update_session_meta() "
            "instead of db._conn.execute() directly"
        )


# ---------------------------------------------------------------------------
# Integration: _persist round-trip via SessionManager
# ---------------------------------------------------------------------------

class TestPersistRoundTrip:
    """End-to-end: save a session and verify DB state is correct."""

    def test_cwd_persisted_via_update_session_meta(self, tmp_path):
        db = _tmp_db(tmp_path)
        manager = SessionManager(agent_factory=_mock_agent, db=db)

        state = manager.create_session(cwd="/original")
        assert db.get_session(state.session_id) is not None

        # Simulate cwd change and save
        state.cwd = "/updated"
        manager.save_session(state.session_id)

        row = db.get_session(state.session_id)
        mc = json.loads(row["model_config"])
        assert mc["cwd"] == "/updated"

    def test_model_persisted_via_update_session_meta(self, tmp_path):
        db = _tmp_db(tmp_path)
        manager = SessionManager(agent_factory=_mock_agent, db=db)

        state = manager.create_session()
        state.model = "new-model-xyz"
        manager.save_session(state.session_id)

        row = db.get_session(state.session_id)
        assert row["model"] == "new-model-xyz"

    def test_existing_model_not_cleared_on_save(self, tmp_path):
        """If state.model is empty, the DB model column must not be overwritten."""
        db = _tmp_db(tmp_path)
        manager = SessionManager(agent_factory=_mock_agent, db=db)

        state = manager.create_session()
        # Manually set a model in DB
        db.update_session_meta(state.session_id, json.dumps({"cwd": "."}), model="stored-model")

        # Now save with empty model
        state.model = ""
        manager.save_session(state.session_id)

        row = db.get_session(state.session_id)
        assert row["model"] == "stored-model", (
            "COALESCE must preserve the existing model when new value is NULL"
        )


# ---------------------------------------------------------------------------
# archive_and_compact — compression-loop dedup guard
# ---------------------------------------------------------------------------


class TestArchiveAndCompactDedup:
    """Repeated compression passes must not re-insert rows they just archived.

    Root cause (8/8-8/9 reports): preflight + post-response + next-turn
    preflight compression passes each soft-archive the live set and insert
    the rebuilt tail again, so the archived (active=0, compacted=1) layer
    accumulates duplicate copies of the same (role, content, timestamp)
    rows — ballooning the DB and polluting FTS search. The dedup probe
    skips rows whose key was archived within the last 60s and turns a
    fully-duplicate pass into a no-op.
    """

    @staticmethod
    def _rebuilt_tail():
        now = time.time()
        return [
            {"role": "user", "content": "执行第 1–4 步", "timestamp": now},
            {"role": "assistant", "content": "开始执行", "timestamp": now + 1},
        ]

    def test_repeated_compaction_does_not_duplicate_archived_tail(self, tmp_path):
        db = _tmp_db(tmp_path)
        db.create_session("s-dedup", source="acp", model="test")
        tail = self._rebuilt_tail()
        db.append_messages_batch("s-dedup", list(tail))

        db.archive_and_compact("s-dedup", list(tail))
        db.archive_and_compact("s-dedup", list(tail))

        # Second pass is a pure re-insert: nothing new may land.
        rows = db.get_messages("s-dedup", include_inactive=True)
        assert len(rows) == 4, f"expected 2 archived + 2 active, got {len(rows)}"
        active = [r for r in rows if r["active"] == 1]
        assert len(active) == 2

    def test_fully_duplicate_pass_is_a_noop_returning_active_count(self, tmp_path):
        db = _tmp_db(tmp_path)
        db.create_session("s-noop", source="acp", model="test")
        tail = self._rebuilt_tail()
        db.append_messages_batch("s-noop", list(tail))
        db.archive_and_compact("s-noop", list(tail))

        before = db.get_messages("s-noop", include_inactive=True)
        returned = db.archive_and_compact("s-noop", list(tail))

        after = db.get_messages("s-noop", include_inactive=True)
        assert len(after) == len(before), "fully duplicate pass must not add rows"
        assert returned == 2, f"no-op must return the current active count, got {returned}"

    def test_new_rows_still_inserted_beside_duplicates(self, tmp_path):
        db = _tmp_db(tmp_path)
        db.create_session("s-partial", source="acp", model="test")
        tail = self._rebuilt_tail()
        db.append_messages_batch("s-partial", list(tail))
        db.archive_and_compact("s-partial", list(tail))

        # Next pass re-inserts the same tail PLUS a genuinely new row.
        now = time.time()
        next_pass = list(tail) + [
            {"role": "user", "content": "新的一轮提问", "timestamp": now + 10}
        ]
        db.archive_and_compact("s-partial", next_pass)

        rows = db.get_messages("s-partial", include_inactive=True)
        active = [r for r in rows if r["active"] == 1]
        # Archived seed (2) + archived first tail (2) + active: only the NEW row.
        assert len(active) == 1, f"expected only the new row active, got {len(active)}"
        assert active[0]["content"] == "新的一轮提问"

    def test_rows_without_timestamp_are_never_deduped(self, tmp_path):
        db = _tmp_db(tmp_path)
        db.create_session("s-nots", source="acp", model="test")
        no_ts = [{"role": "user", "content": "无时间戳行"}]
        db.archive_and_compact("s-nots", list(no_ts))
        db.archive_and_compact("s-nots", list(no_ts))

        rows = db.get_messages("s-nots", include_inactive=True)
        # Conservative: no timestamp -> never deduped, both passes insert.
        assert len(rows) == 2, f"expected 2 rows (no dedup), got {len(rows)}"
