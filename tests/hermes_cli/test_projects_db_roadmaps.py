"""Integration tests for the Roadmaps schema in the real projects store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import projects_db as pdb


TABLES = {
    "roadmaps",
    "roadmap_versions",
    "roadmap_nodes",
    "roadmap_relations",
    "roadmap_sessions",
    "roadmap_todos",
}

ROADMAP_COLUMNS = {
    "roadmaps": "profile_id TEXT, project_id TEXT, roadmap_id TEXT, title TEXT, purpose TEXT, lifecycle_state TEXT, active_version INTEGER, created_by TEXT, updated_by TEXT, created_at INTEGER, updated_at INTEGER",
    "roadmap_versions": "profile_id TEXT, project_id TEXT, roadmap_id TEXT, version INTEGER, state TEXT, source TEXT, reason TEXT, created_by TEXT, created_at INTEGER, content_hash TEXT",
    "roadmap_nodes": "profile_id TEXT, project_id TEXT, roadmap_id TEXT, version INTEGER, node_id TEXT, parent_node_id TEXT, kind TEXT, title TEXT, description TEXT, state TEXT, progress INTEGER, owner_agent TEXT, block_reason TEXT, created_at INTEGER, updated_at INTEGER",
    "roadmap_relations": "profile_id TEXT, project_id TEXT, roadmap_id TEXT, version INTEGER, relation_id TEXT, from_node_id TEXT, to_node_id TEXT, kind TEXT, state TEXT, reason TEXT",
    "roadmap_sessions": "profile_id TEXT, project_id TEXT, roadmap_id TEXT, stored_session_id TEXT, kind TEXT, node_id TEXT, plan_version INTEGER, state TEXT, actor TEXT, created_at INTEGER, updated_at INTEGER",
    "roadmap_todos": "profile_id TEXT, project_id TEXT, roadmap_id TEXT, version INTEGER, todo_id TEXT, node_id TEXT, title TEXT, state TEXT, position INTEGER, created_at INTEGER, updated_at INTEGER",
}


def _open(path: Path) -> sqlite3.Connection:
    return pdb.connect(db_path=path)


def _seed(conn: sqlite3.Connection, profile: str = "profile-a", project: str = "project-a") -> None:
    conn.execute("INSERT INTO projects(id, slug, name, created_at) VALUES (?, ?, ?, ?)", (project, project, project, 1))
    conn.execute(
        "INSERT INTO roadmaps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (profile, project, "roadmap-a", "Roadmap", None, "draft", None, "actor", "actor", 1, 1),
    )
    conn.execute(
        "INSERT INTO roadmap_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (profile, project, "roadmap-a", 1, "draft", "test", None, "actor", 1, None),
    )
    conn.execute(
        "INSERT INTO roadmap_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (profile, project, "roadmap-a", 1, "node-a", None, "objective", "A", None, "planned", 0, None, None, 1, 1),
    )
    conn.commit()


def test_real_store_creates_roadmaps_schema_idempotently_and_preserves_legacy(tmp_path: Path) -> None:
    path = tmp_path / "profile" / "projects.db"
    first = _open(path)
    assert first.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    first.execute("CREATE TABLE legacy (id INTEGER PRIMARY KEY, value TEXT)")
    first.execute("INSERT INTO legacy VALUES (1, 'keep')")
    first.execute("CREATE INDEX legacy_value ON legacy(value)")
    first.commit()
    first.close()

    second = _open(path)
    assert {row[0] for row in second.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'roadmap%'")} == TABLES
    assert second.execute("SELECT value FROM legacy WHERE id=1").fetchone()[0] == "keep"
    assert "legacy_value" in {row[1] for row in second.execute("PRAGMA index_list(legacy)")}
    second.close()

    third = _open(path)
    assert {row[0] for row in third.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'roadmap%'")} == TABLES
    third.close()


def test_real_store_enforces_active_version_and_qualified_node_relation_todo_scope(tmp_path: Path) -> None:
    conn = _open(tmp_path / "projects.db")
    _seed(conn)

    conn.execute("UPDATE roadmaps SET active_version=1")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE roadmaps SET active_version=99")
        conn.commit()
    conn.rollback()

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO roadmap_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("profile-b", "project-a", "roadmap-a", 1, "foreign", None, "step", "bad", None, "planned", 0, None, None, 1, 1),
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO roadmap_relations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("profile-a", "project-a", "roadmap-a", 1, "rel", "node-a", "missing", "depends_on", "active", None),
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO roadmap_todos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("profile-a", "project-a", "roadmap-a", 1, "todo", "missing", "T", "open", 0, 1, 1),
        )
    conn.close()


def test_roadmap_sessions_schema_enforces_durable_vision_contract(tmp_path: Path) -> None:
    conn = _open(tmp_path / "projects.db")
    _seed(conn)

    columns = {row[1] for row in conn.execute("PRAGMA table_info(roadmap_sessions)")}
    assert columns == {
        "profile_id", "project_id", "roadmap_id", "stored_session_id", "kind",
        "node_id", "plan_version", "state", "actor", "created_at", "updated_at",
    }
    assert "runtime_session_id" not in columns

    conn.execute(
        "INSERT INTO roadmap_sessions "
        "(profile_id, project_id, roadmap_id, stored_session_id, kind, state, actor, created_at, updated_at) "
        "VALUES ('profile-a', 'project-a', 'roadmap-a', 'stored-1', 'vision', 'active', 'actor', 1, 1)"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO roadmap_sessions "
            "(profile_id, project_id, roadmap_id, stored_session_id, kind, state, actor, created_at, updated_at) "
            "VALUES ('profile-a', 'project-a', 'roadmap-a', 'stored-2', 'vision', 'active', 'actor', 1, 1)"
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO roadmap_sessions "
            "(profile_id, project_id, roadmap_id, stored_session_id, kind, state, actor, created_at, updated_at) "
            "VALUES ('profile-a', 'project-a', 'roadmap-a', 'stored-x', 'node', 'closed', 'actor', 1, 1)"
        )
    conn.rollback()

    conn.execute("DELETE FROM roadmaps WHERE roadmap_id='roadmap-a'")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM roadmap_sessions").fetchone()[0] == 0
    conn.close()


def test_connect_additively_restores_missing_roadmap_sessions_table(tmp_path: Path) -> None:
    path = tmp_path / "projects.db"
    raw = sqlite3.connect(path)
    raw.executescript(pdb.SCHEMA_SQL)
    raw.execute("DROP TABLE roadmap_sessions")
    raw.commit()
    raw.close()

    conn = _open(path)
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='roadmap_sessions'"
    ).fetchone() is not None
    conn.close()


def test_real_store_preserves_project_cascade_and_isolates_db_paths(tmp_path: Path) -> None:
    first = _open(tmp_path / "a" / "projects.db")
    second = _open(tmp_path / "b" / "projects.db")
    _seed(first, "profile-a")
    _seed(second, "profile-b")
    assert first.execute("SELECT profile_id FROM roadmaps").fetchone()[0] == "profile-a"
    assert second.execute("SELECT profile_id FROM roadmaps").fetchone()[0] == "profile-b"

    first.execute("DELETE FROM projects WHERE id='project-a'")
    first.commit()
    for table in TABLES:
        assert first.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert second.execute("SELECT COUNT(*) FROM roadmaps").fetchone()[0] == 1
    first.close()
    second.close()


def test_real_store_keeps_deferred_runtime_tables_out_of_phase_one(tmp_path: Path) -> None:
    conn = _open(tmp_path / "projects.db")
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not names.intersection({"reports", "proofs", "events", "agent_projections"})
    conn.close()


@pytest.mark.parametrize("table", sorted(TABLES))
def test_real_store_rejects_same_columns_without_roadmap_constraints(
    tmp_path: Path, table: str
) -> None:
    path = tmp_path / f"{table}.db"
    raw = sqlite3.connect(path)
    raw.execute(f"CREATE TABLE {table} ({ROADMAP_COLUMNS[table]})")
    raw.commit()
    raw.close()

    with pytest.raises((sqlite3.DatabaseError, ValueError), match=table):
        _open(path)


def test_connect_rejects_weakened_core_projects_without_roadmaps_residue(tmp_path: Path) -> None:
    path = tmp_path / "projects.db"
    raw = sqlite3.connect(path)
    raw.execute(
        "CREATE TABLE projects ("
        "id TEXT, slug TEXT, name TEXT, description TEXT, icon TEXT, color TEXT, "
        "board_slug TEXT, primary_path TEXT, created_at INTEGER, archived INTEGER)"
    )
    raw.commit()
    raw.close()

    with pytest.raises((sqlite3.DatabaseError, ValueError), match="projects"):
        _open(path)

    check = sqlite3.connect(path)
    names = {
        row[0]
        for row in check.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert not names.intersection(TABLES)
    check.close()


def test_connect_rejects_projects_with_non_unique_slug_without_roadmaps_residue(tmp_path: Path) -> None:
    path = tmp_path / "projects.db"
    raw = sqlite3.connect(path)
    raw.execute(
        "CREATE TABLE projects ("
        "id TEXT PRIMARY KEY, slug TEXT NOT NULL, name TEXT NOT NULL, "
        "description TEXT, icon TEXT, color TEXT, board_slug TEXT, "
        "primary_path TEXT, created_at INTEGER NOT NULL, archived INTEGER NOT NULL)"
    )
    raw.commit()
    raw.close()

    with pytest.raises(sqlite3.DatabaseError, match="projects"):
        _open(path)

    check = sqlite3.connect(path)
    names = {row[0] for row in check.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not names.intersection(TABLES)
    check.close()


def test_connect_rejects_orphan_roadmaps_after_restoring_missing_projects(tmp_path: Path) -> None:
    path = tmp_path / "projects.db"
    raw = sqlite3.connect(path)
    raw.executescript(pdb.SCHEMA_SQL)
    raw.execute(
        "INSERT INTO roadmaps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("profile-a", "missing-project", "roadmap-a", "Roadmap", None, "draft", None, "actor", "actor", 1, 1),
    )
    raw.execute("DROP TABLE projects")
    raw.commit()
    raw.close()

    with pytest.raises(sqlite3.DatabaseError, match="foreign key"):
        _open(path)


def test_connect_retries_optional_project_column_migration_with_complete_roadmaps_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "projects.db"
    raw = sqlite3.connect(path)
    raw.executescript(pdb.SCHEMA_SQL.replace("    board_slug    TEXT,\n", ""))
    raw.commit()
    raw.close()

    conn = _open(path)
    assert "board_slug" in {
        row[1] for row in conn.execute("PRAGMA table_info(projects)")
    }
    conn.close()


def test_connect_rejects_roadmap_check_contract_tampering(tmp_path: Path) -> None:
    path = tmp_path / "projects.db"
    raw = sqlite3.connect(path)
    raw.executescript(pdb.SCHEMA_SQL)
    original = raw.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='roadmaps'"
    ).fetchone()[0]
    tampered = original.replace("'archived'))", "'archived','bogus'))")
    assert tampered != original
    raw.execute("PRAGMA writable_schema=ON")
    raw.execute(
        "UPDATE sqlite_master SET sql=? WHERE type='table' AND name='roadmaps'",
        (tampered,),
    )
    raw.commit()
    raw.close()

    with pytest.raises((sqlite3.DatabaseError, ValueError), match="roadmaps"):
        _open(path)


def test_connect_restores_missing_core_schema_with_complete_roadmaps(tmp_path: Path) -> None:
    path = tmp_path / "projects.db"
    raw = sqlite3.connect(path)
    raw.executescript(pdb.SCHEMA_SQL)
    raw.execute("DROP TABLE projects")
    raw.commit()
    raw.close()

    conn = _open(path)
    names = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"projects", "project_folders", "project_meta", "discovered_repos"}.issubset(names)
    assert TABLES.issubset(names)
    assert "idx_project_folders_path" in {
        row[1] for row in conn.execute("PRAGMA index_list(project_folders)")
    }
    conn.close()


def test_real_store_reinitializes_after_database_file_is_replaced(tmp_path: Path) -> None:
    path = tmp_path / "projects.db"
    first = _open(path)
    first.close()
    path.unlink()

    second = _open(path)
    names = {row[0] for row in second.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert TABLES.issubset(names)
    second.close()
