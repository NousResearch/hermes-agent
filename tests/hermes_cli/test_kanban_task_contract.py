"""HERMES-ORCH-001A: versioned task-contract persistence and migration.

Covers:
- version-1 contracts persist and round-trip exactly
- legacy tasks without a contract remain readable
- repeated initialization is idempotent
- existing-board migration preserves task rows
- invalid contract structures are rejected
- no dispatch/admission enforcement (storage only)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _valid_contract(**overrides):
    base = {
        "version": 1,
        "scope": "ORCH-001A contract persistence only",
        "allowed_files": ["hermes_cli/kanban_db.py", "tests/hermes_cli/test_kanban_task_contract.py"],
        "forbidden_files": ["hermes_cli/main.py"],
        "base_commit": "e9b8ae6be137abead6d19ed8a67c523f8c527096",
        "required_evidence": ["pytest output", "commit SHA"],
        "required_commands": ["scripts/run_tests.sh tests/hermes_cli/test_kanban_task_contract.py -q"],
        "allow_child_creation": False,
        "forbidden_git_actions": ["push", "merge", "amend", "reset", "clean", "restore", "stash"],
        "notification_verified": False,
    }
    base.update(overrides)
    return base


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _fresh_conn(board: str = "default"):
    path = kb.kanban_db_path(board=board)
    path.parent.mkdir(parents=True, exist_ok=True)
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))
    return kb.connect(board=board)


# ---------------------------------------------------------------------------
# Pure ser/de
# ---------------------------------------------------------------------------


class TestTaskContractSerde:
    def test_round_trip_exact(self):
        contract = _valid_contract()
        blob = kb.serialize_task_contract(contract)
        # Stable compact JSON (sort_keys + separators).
        expected = json.dumps(
            kb.normalize_task_contract(contract),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        assert blob == expected
        again = kb.deserialize_task_contract(blob)
        assert again == kb.normalize_task_contract(contract)
        assert kb.serialize_task_contract(again) == blob

    def test_strips_scope_and_base_commit(self):
        contract = _valid_contract(scope="  scoped  ", base_commit="  abc  ")
        normalized = kb.normalize_task_contract(contract)
        assert normalized["scope"] == "scoped"
        assert normalized["base_commit"] == "abc"

    def test_empty_blob_is_legacy_none(self):
        assert kb.deserialize_task_contract(None) is None
        assert kb.deserialize_task_contract("") is None
        assert kb.deserialize_task_contract("   ") is None

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            "not-a-dict",
            42,
            [],
            {"version": 2, **{k: _valid_contract()[k] for k in kb.TASK_CONTRACT_V1_KEYS if k != "version"}},
            {**_valid_contract(), "version": True},
            {**_valid_contract(), "scope": ""},
            {**_valid_contract(), "base_commit": "  "},
            {**_valid_contract(), "allowed_files": "a.py"},
            {**_valid_contract(), "allowed_files": [1]},
            {**_valid_contract(), "allow_child_creation": 1},
            {**_valid_contract(), "notification_verified": "yes"},
            {k: v for k, v in _valid_contract().items() if k != "required_evidence"},
            {**_valid_contract(), "extra_field": "nope"},
            "{not json",
        ],
    )
    def test_invalid_structures_rejected(self, bad):
        if isinstance(bad, str) and bad.startswith("{"):
            with pytest.raises(kb.TaskContractError):
                kb.deserialize_task_contract(bad)
        else:
            with pytest.raises(kb.TaskContractError):
                kb.normalize_task_contract(bad)


# ---------------------------------------------------------------------------
# DB persistence / migration
# ---------------------------------------------------------------------------


class TestTaskContractPersistence:
    def test_fresh_schema_has_contract_column(self, isolated_home):
        with _fresh_conn() as conn:
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
        assert "contract" in cols

    def test_create_and_round_trip(self, isolated_home):
        contract = _valid_contract()
        with _fresh_conn() as conn:
            tid = kb.create_task(conn, title="with contract", contract=contract)
            stored = kb.get_task_contract(conn, tid)
            task = kb.get_task(conn, tid)
            raw = conn.execute(
                "SELECT contract FROM tasks WHERE id = ?", (tid,)
            ).fetchone()["contract"]

        assert stored == kb.normalize_task_contract(contract)
        assert task is not None
        assert task.contract == stored
        assert raw == kb.serialize_task_contract(contract)
        # Exact blob round-trip through deserialize + re-serialize.
        assert kb.serialize_task_contract(kb.deserialize_task_contract(raw)) == raw

    def test_set_get_contract_on_existing_task(self, isolated_home):
        contract = _valid_contract(scope="post-create attach")
        with _fresh_conn() as conn:
            tid = kb.create_task(conn, title="legacy first")
            assert kb.get_task_contract(conn, tid) is None
            written = kb.set_task_contract(conn, tid, contract)
            assert written == kb.normalize_task_contract(contract)
            assert kb.get_task_contract(conn, tid) == written
            assert kb.get_task(conn, tid).contract == written

    def test_create_rejects_invalid_contract(self, isolated_home):
        with _fresh_conn() as conn:
            with pytest.raises(kb.TaskContractError):
                kb.create_task(
                    conn,
                    title="bad",
                    contract={"version": 1, "scope": "missing rest"},
                )
            # Nothing was inserted.
            n = conn.execute("SELECT COUNT(*) AS c FROM tasks").fetchone()["c"]
            assert n == 0

    def test_set_rejects_invalid_contract(self, isolated_home):
        with _fresh_conn() as conn:
            tid = kb.create_task(conn, title="ok")
            with pytest.raises(kb.TaskContractError):
                kb.set_task_contract(conn, tid, {"version": 99})
            assert kb.get_task_contract(conn, tid) is None

    def test_legacy_task_readable_without_contract(self, isolated_home):
        with _fresh_conn() as conn:
            tid = kb.create_task(conn, title="legacy", body="no contract")
            task = kb.get_task(conn, tid)
            assert task is not None
            assert task.id == tid
            assert task.title == "legacy"
            assert task.contract is None
            assert kb.get_task_contract(conn, tid) is None

    def test_repeated_init_idempotent(self, isolated_home):
        path = kb.kanban_db_path(board="default")
        path.parent.mkdir(parents=True, exist_ok=True)
        kb._INITIALIZED_PATHS.discard(str(path.resolve()))
        with kb.connect(board="default") as conn:
            tid = kb.create_task(conn, title="keep", contract=_valid_contract())
            blob = conn.execute(
                "SELECT contract FROM tasks WHERE id = ?", (tid,)
            ).fetchone()["contract"]

        for _ in range(3):
            kb._INITIALIZED_PATHS.discard(str(path.resolve()))
            with kb.connect(board="default") as conn:
                cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
                assert "contract" in cols
                row = conn.execute(
                    "SELECT id, title, contract FROM tasks WHERE id = ?", (tid,)
                ).fetchone()
                assert row["title"] == "keep"
                assert row["contract"] == blob
                assert kb.get_task_contract(conn, tid) == kb.normalize_task_contract(
                    _valid_contract()
                )

    def test_existing_board_migration_preserves_rows(self, isolated_home):
        """Simulate a pre-contract board: open SCHEMA without contract column
        is not feasible via SCHEMA_SQL now, so build a minimal legacy tasks
        table then run connect() migration."""
        path = kb.kanban_db_path(board="legacy-board")
        path.parent.mkdir(parents=True, exist_ok=True)
        kb._INITIALIZED_PATHS.discard(str(path.resolve()))

        # Hand-build a simplified pre-ORCH-001 tasks table + one row.
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                body TEXT,
                assignee TEXT,
                status TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                created_by TEXT,
                created_at INTEGER NOT NULL,
                started_at INTEGER,
                completed_at INTEGER,
                workspace_kind TEXT NOT NULL DEFAULT 'scratch',
                workspace_path TEXT,
                claim_lock TEXT,
                claim_expires INTEGER,
                tenant TEXT,
                result TEXT
            );
            CREATE TABLE task_links (
                parent_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                PRIMARY KEY (parent_id, child_id)
            );
            CREATE TABLE task_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                author TEXT NOT NULL,
                body TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE task_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT,
                created_at INTEGER NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at, workspace_kind) "
            "VALUES ('task-legacy-1', 'Preserve me', 'ready', 1000, 'scratch')"
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, created_at) "
            "VALUES ('task-legacy-1', 'created', 1000)"
        )
        conn.commit()
        conn.close()

        # Migration path: connect runs SCHEMA_SQL + additive column migration.
        with kb.connect(path) as migrated:
            cols = {row["name"] for row in migrated.execute("PRAGMA table_info(tasks)")}
            assert "contract" in cols
            row = migrated.execute(
                "SELECT id, title, status, contract FROM tasks WHERE id = ?",
                ("task-legacy-1",),
            ).fetchone()
            assert row is not None
            assert row["title"] == "Preserve me"
            assert row["status"] == "ready"
            assert row["contract"] is None
            task = kb.get_task(migrated, "task-legacy-1")
            assert task is not None
            assert task.title == "Preserve me"
            assert task.contract is None
            assert kb.get_task_contract(migrated, "task-legacy-1") is None
            # Can attach a contract after migration.
            kb.set_task_contract(migrated, "task-legacy-1", _valid_contract())
            assert kb.get_task_contract(migrated, "task-legacy-1")["scope"] == (
                "ORCH-001A contract persistence only"
            )
