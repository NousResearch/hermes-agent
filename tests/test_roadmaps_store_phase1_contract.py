"""Phase 1 SQLite contract tests.

These tests are deliberately self-contained even though the runtime
``hermes_cli/projects_db.py`` is present in this worktree: porting the DDL to
that runtime is a separate task.  The DDL is a preparatory contract, not a
replacement backend implementation.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


ROADMAPS_SCHEMA = """
CREATE TABLE IF NOT EXISTS roadmaps (
    profile_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(profile_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    project_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(project_id, char(9), ''), char(10), ''), char(13), ''))) > 0) REFERENCES projects(id) ON DELETE CASCADE,
    roadmap_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(roadmap_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    title TEXT NOT NULL,
    purpose TEXT,
    lifecycle_state TEXT NOT NULL CHECK (lifecycle_state IN
        ('draft','proposed','validated','in_progress','blocked','completed','archived')),
    active_version INTEGER CHECK (active_version IS NULL OR active_version >= 1),
    created_by TEXT NOT NULL CHECK (length(trim(replace(replace(replace(created_by, char(9), ''), char(10), ''), char(13), ''))) > 0),
    updated_by TEXT NOT NULL CHECK (length(trim(replace(replace(replace(updated_by, char(9), ''), char(10), ''), char(13), ''))) > 0),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (profile_id, project_id, roadmap_id),
    FOREIGN KEY (profile_id, project_id, roadmap_id, active_version)
      REFERENCES roadmap_versions(profile_id, project_id, roadmap_id, version)
      DEFERRABLE INITIALLY DEFERRED
);
CREATE TABLE IF NOT EXISTS roadmap_versions (
    profile_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(profile_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    project_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(project_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    roadmap_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(roadmap_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    version INTEGER NOT NULL CHECK (version >= 1),
    state TEXT NOT NULL CHECK (state IN ('draft','proposed','validated','superseded','archived')),
    source TEXT,
    reason TEXT,
    created_by TEXT NOT NULL CHECK (length(trim(replace(replace(replace(created_by, char(9), ''), char(10), ''), char(13), ''))) > 0),
    created_at INTEGER NOT NULL,
    content_hash TEXT,
    PRIMARY KEY (profile_id, project_id, roadmap_id, version),
    FOREIGN KEY (profile_id, project_id, roadmap_id)
      REFERENCES roadmaps(profile_id, project_id, roadmap_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS roadmap_nodes (
    profile_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(profile_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    project_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(project_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    roadmap_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(roadmap_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    version INTEGER NOT NULL,
    node_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(node_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    parent_node_id TEXT,
    kind TEXT NOT NULL CHECK (kind IN ('objective','phase','milestone','step','decision')),
    title TEXT NOT NULL,
    description TEXT,
    state TEXT NOT NULL CHECK (state IN ('planned','ready','in_progress','blocked','completed','archived')),
    progress INTEGER NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    owner_agent TEXT CHECK (owner_agent IS NULL OR length(trim(replace(replace(replace(owner_agent, char(9), ''), char(10), ''), char(13), ''))) > 0),
    block_reason TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (profile_id, project_id, roadmap_id, version, node_id),
    FOREIGN KEY (profile_id, project_id, roadmap_id, version)
      REFERENCES roadmap_versions(profile_id, project_id, roadmap_id, version) ON DELETE CASCADE,
    FOREIGN KEY (profile_id, project_id, roadmap_id, version, parent_node_id)
      REFERENCES roadmap_nodes(profile_id, project_id, roadmap_id, version, node_id),
    CHECK (parent_node_id IS NULL OR (length(trim(replace(replace(replace(parent_node_id, char(9), ''), char(10), ''), char(13), ''))) > 0 AND parent_node_id <> node_id))
);
CREATE TABLE IF NOT EXISTS roadmap_relations (
    profile_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(profile_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    project_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(project_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    roadmap_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(roadmap_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    version INTEGER NOT NULL,
    relation_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(relation_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    from_node_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(from_node_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    to_node_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(to_node_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    kind TEXT NOT NULL CHECK (kind IN ('depends_on','blocks','enables','follows','validates','supersedes')),
    state TEXT NOT NULL DEFAULT 'active' CHECK (state IN ('active','superseded','invalid')),
    reason TEXT,
    PRIMARY KEY (profile_id, project_id, roadmap_id, version, relation_id),
    FOREIGN KEY (profile_id, project_id, roadmap_id, version, from_node_id)
      REFERENCES roadmap_nodes(profile_id, project_id, roadmap_id, version, node_id) ON DELETE CASCADE,
    FOREIGN KEY (profile_id, project_id, roadmap_id, version, to_node_id)
      REFERENCES roadmap_nodes(profile_id, project_id, roadmap_id, version, node_id) ON DELETE CASCADE,
    CHECK (from_node_id <> to_node_id)
);
CREATE TABLE IF NOT EXISTS roadmap_todos (
    profile_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(profile_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    project_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(project_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    roadmap_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(roadmap_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    version INTEGER NOT NULL,
    todo_id TEXT NOT NULL CHECK (length(trim(replace(replace(replace(todo_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    node_id TEXT CHECK (node_id IS NULL OR length(trim(replace(replace(replace(node_id, char(9), ''), char(10), ''), char(13), ''))) > 0),
    title TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('open','in_progress','done','cancelled')),
    position INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (profile_id, project_id, roadmap_id, version, todo_id),
    FOREIGN KEY (profile_id, project_id, roadmap_id, version)
      REFERENCES roadmap_versions(profile_id, project_id, roadmap_id, version) ON DELETE CASCADE,
    FOREIGN KEY (profile_id, project_id, roadmap_id, version, node_id)
      REFERENCES roadmap_nodes(profile_id, project_id, roadmap_id, version, node_id)
);
"""


# This is intentionally independent of ``ROADMAPS_SCHEMA``.  Checking only
# names would accept a table with the right API-shaped columns but no PK, FK,
# NOT NULL, or CHECK constraints.
_SCHEMA_CONTRACT = {
    "roadmaps": {
        "columns": (("profile_id", "TEXT", 1, 1), ("project_id", "TEXT", 1, 2),
                    ("roadmap_id", "TEXT", 1, 3), ("title", "TEXT", 1, 0),
                    ("purpose", "TEXT", 0, 0), ("lifecycle_state", "TEXT", 1, 0),
                    ("active_version", "INTEGER", 0, 0), ("created_by", "TEXT", 1, 0),
                    ("updated_by", "TEXT", 1, 0), ("created_at", "INTEGER", 1, 0),
                    ("updated_at", "INTEGER", 1, 0)),
        "fks": (("roadmap_versions", ("profile_id", "project_id", "roadmap_id", "active_version"),
                 ("profile_id", "project_id", "roadmap_id", "version"), "NO ACTION"),
                ("projects", ("project_id",), ("id",), "CASCADE")),
        "sql_markers": ("primary key", "foreign key", "check", "deferrable initially deferred",
                         "lifecycle_state text not null check", "active_version integer check"),
    },
    "roadmap_versions": {
        "columns": (("profile_id", "TEXT", 1, 1), ("project_id", "TEXT", 1, 2),
                    ("roadmap_id", "TEXT", 1, 3), ("version", "INTEGER", 1, 4),
                    ("state", "TEXT", 1, 0), ("source", "TEXT", 0, 0), ("reason", "TEXT", 0, 0),
                    ("created_by", "TEXT", 1, 0), ("created_at", "INTEGER", 1, 0),
                    ("content_hash", "TEXT", 0, 0)),
        "fks": (("roadmaps", ("profile_id", "project_id", "roadmap_id"),
                 ("profile_id", "project_id", "roadmap_id"), "CASCADE"),),
        "sql_markers": ("primary key", "foreign key", "check", "state text not null check", "version integer not null check"),
    },
    "roadmap_nodes": {
        "columns": (("profile_id", "TEXT", 1, 1), ("project_id", "TEXT", 1, 2),
                    ("roadmap_id", "TEXT", 1, 3), ("version", "INTEGER", 1, 4),
                    ("node_id", "TEXT", 1, 5), ("parent_node_id", "TEXT", 0, 0),
                    ("kind", "TEXT", 1, 0), ("title", "TEXT", 1, 0), ("description", "TEXT", 0, 0),
                    ("state", "TEXT", 1, 0), ("progress", "INTEGER", 1, 0), ("owner_agent", "TEXT", 0, 0),
                    ("block_reason", "TEXT", 0, 0),
                    ("created_at", "INTEGER", 1, 0), ("updated_at", "INTEGER", 1, 0)),
        "fks": (("roadmap_nodes", ("profile_id", "project_id", "roadmap_id", "version", "parent_node_id"),
                 ("profile_id", "project_id", "roadmap_id", "version", "node_id"), "NO ACTION"),
                ("roadmap_versions", ("profile_id", "project_id", "roadmap_id", "version"),
                 ("profile_id", "project_id", "roadmap_id", "version"), "CASCADE")),
        "sql_markers": ("primary key", "foreign key", "check", "progress integer not null default 0 check",
                         "parent_node_id is null"),
    },
    "roadmap_relations": {
        "columns": (("profile_id", "TEXT", 1, 1), ("project_id", "TEXT", 1, 2),
                    ("roadmap_id", "TEXT", 1, 3), ("version", "INTEGER", 1, 4),
                    ("relation_id", "TEXT", 1, 5), ("from_node_id", "TEXT", 1, 0),
                    ("to_node_id", "TEXT", 1, 0), ("kind", "TEXT", 1, 0),
                    ("state", "TEXT", 1, 0), ("reason", "TEXT", 0, 0)),
        "fks": (("roadmap_nodes", ("profile_id", "project_id", "roadmap_id", "version", "from_node_id"),
                 ("profile_id", "project_id", "roadmap_id", "version", "node_id"), "CASCADE"),
                ("roadmap_nodes", ("profile_id", "project_id", "roadmap_id", "version", "to_node_id"),
                 ("profile_id", "project_id", "roadmap_id", "version", "node_id"), "CASCADE")),
        "sql_markers": ("primary key", "foreign key", "check", "from_node_id <> to_node_id", "state text not null default 'active' check"),
    },
    "roadmap_todos": {
        "columns": (("profile_id", "TEXT", 1, 1), ("project_id", "TEXT", 1, 2),
                    ("roadmap_id", "TEXT", 1, 3), ("version", "INTEGER", 1, 4),
                    ("todo_id", "TEXT", 1, 5), ("node_id", "TEXT", 0, 0), ("title", "TEXT", 1, 0),
                    ("state", "TEXT", 1, 0), ("position", "INTEGER", 1, 0),
                    ("created_at", "INTEGER", 1, 0), ("updated_at", "INTEGER", 1, 0)),
        "fks": (("roadmap_nodes", ("profile_id", "project_id", "roadmap_id", "version", "node_id"),
                 ("profile_id", "project_id", "roadmap_id", "version", "node_id"), "NO ACTION"),
                ("roadmap_versions", ("profile_id", "project_id", "roadmap_id", "version"),
                 ("profile_id", "project_id", "roadmap_id", "version"), "CASCADE")),
        "sql_markers": ("primary key", "foreign key", "check", "state text not null check", "node_id text check"),
    },
}


def _validate_schema_contract(conn: sqlite3.Connection) -> None:
    for table, expected in _SCHEMA_CONTRACT.items():
        actual_columns = tuple(
            (row[1], row[2].upper(), row[3], row[5])
            for row in conn.execute(f"PRAGMA table_info({table})")
        )
        if actual_columns != expected["columns"]:
            raise RuntimeError(f"incompatible existing {table} table columns")

        grouped: dict[int, list[tuple]] = {}
        for row in conn.execute(f"PRAGMA foreign_key_list({table})"):
            grouped.setdefault(row[0], []).append(row)
        actual_fks = tuple(
            (rows[0][2], tuple(row[3] for row in rows), tuple(row[4] for row in rows), rows[0][6])
            for fk_id, rows in sorted(grouped.items())
        )
        if set(actual_fks) != set(expected["fks"]):
            raise RuntimeError(f"incompatible existing {table} table foreign keys")

        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()[0].lower()
        if any(marker not in " ".join(sql.split()) for marker in expected["sql_markers"]):
            raise RuntimeError(f"incompatible existing {table} table constraints")


def open_contract_db(path: Path, *, fail_after_statement: int | None = None) -> sqlite3.Connection:
    """Create the isolated contract DB, atomically, without runtime imports.

    ``projects_db.connect()`` has a separate porting obligation to make its
    existing initialization transaction atomic; this helper only proves the
    DDL contract and deliberately does not emulate that runtime.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    statements = [
        "CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, name TEXT NOT NULL)",
        *(statement.strip() for statement in ROADMAPS_SCHEMA.split(";") if statement.strip()),
    ]
    try:
        conn.execute("BEGIN IMMEDIATE")
        for index, statement in enumerate(statements):
            conn.execute(statement)
            if fail_after_statement is not None and index == fail_after_statement:
                raise RuntimeError("injected fixture failure")
        _validate_schema_contract(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    return conn


def seed_scope(conn: sqlite3.Connection, profile: str = "profile-a", project: str = "project-a") -> None:
    conn.execute("INSERT INTO projects(id, name) VALUES (?, ?)", (project, project))
    conn.execute(
        "INSERT INTO roadmaps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (profile, project, "roadmap-a", "Roadmap", None, "draft", None, "test", "test", 1, 1),
    )
    conn.execute(
        "INSERT INTO roadmap_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (profile, project, "roadmap-a", 1, "draft", "test", None, "test", 1, None),
    )
    conn.execute(
        "INSERT INTO roadmap_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (profile, project, "roadmap-a", 1, "node-a", None, "objective", "A", None, "planned", 0, None, None, 1, 1),
    )
    conn.commit()


def test_initialization_is_idempotent_and_uses_only_explicit_db_path(tmp_path: Path) -> None:
    path = tmp_path / "profile-a" / "projects.db"
    first = open_contract_db(path)
    first.close()
    second = open_contract_db(path)
    tables = {
        row[0] for row in second.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'roadmap%'")
    }
    assert tables == {"roadmaps", "roadmap_versions", "roadmap_nodes", "roadmap_relations", "roadmap_todos"}
    assert not (tmp_path / "roadmaps.db").exists()
    second.close()


def test_deferred_reports_proofs_events_and_agent_projections_are_absent(tmp_path: Path) -> None:
    conn = open_contract_db(tmp_path / "projects.db")
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not names.intersection({"reports", "proofs", "events", "agent_projections", "roadmap_reports"})
    conn.close()


def test_two_db_paths_are_isolated_profiles(tmp_path: Path) -> None:
    first = open_contract_db(tmp_path / "a" / "projects.db")
    second = open_contract_db(tmp_path / "b" / "projects.db")
    seed_scope(first, "profile-a")
    seed_scope(second, "profile-b")
    assert first.execute("SELECT profile_id FROM roadmaps").fetchone()[0] == "profile-a"
    assert second.execute("SELECT profile_id FROM roadmaps").fetchone()[0] == "profile-b"
    first.close()
    second.close()


def test_foreign_keys_enforce_project_and_qualified_scope(tmp_path: Path) -> None:
    conn = open_contract_db(tmp_path / "projects.db")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO roadmaps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("profile-a", "missing", "r", "R", None, "draft", None, "x", "x", 1, 1),
        )
    seed_scope(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO roadmap_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("profile-b", "project-a", "roadmap-a", 1, "foreign", None, "step", "bad", None, "planned", 0, None, None, 1, 1),
        )
    conn.close()


def test_project_delete_cascades_roadmap_rows(tmp_path: Path) -> None:
    conn = open_contract_db(tmp_path / "projects.db")
    seed_scope(conn)
    conn.execute("DELETE FROM projects WHERE id=?", ("project-a",))
    conn.commit()
    for table in ("roadmaps", "roadmap_versions", "roadmap_nodes"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    conn.close()


def test_duplicate_ids_are_rejected_within_the_qualified_scope(tmp_path: Path) -> None:
    conn = open_contract_db(tmp_path / "projects.db")
    seed_scope(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO roadmap_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("profile-a", "project-a", "roadmap-a", 1, "node-a", None, "step", "duplicate", None, "planned", 0, None, None, 1, 1),
        )
    conn.close()


def test_relations_cannot_cross_scope_or_reference_missing_nodes(tmp_path: Path) -> None:
    conn = open_contract_db(tmp_path / "projects.db")
    seed_scope(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO roadmap_relations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("profile-a", "project-a", "roadmap-a", 1, "rel", "node-a", "missing", "depends_on", "active", None),
        )
    conn.close()


def test_sqlite_schema_exposes_constraints_and_deferred_active_version_fk(tmp_path: Path) -> None:
    conn = open_contract_db(tmp_path / "projects.db")
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert tuple(
        (row[1], row[2].upper(), row[3], row[5])
        for row in conn.execute("PRAGMA table_info(roadmaps)")
    ) == _SCHEMA_CONTRACT["roadmaps"]["columns"]
    fks = conn.execute("PRAGMA foreign_key_list(roadmaps)").fetchall()
    assert any(row[2] == "roadmap_versions" and row[3] == "profile_id" for row in fks)
    assert list(conn.execute("PRAGMA index_list(roadmap_versions)"))
    assert list(conn.execute("PRAGMA index_list(roadmap_nodes)"))
    conn.close()


def test_active_version_must_reference_same_roadmap_version(tmp_path: Path) -> None:
    conn = open_contract_db(tmp_path / "projects.db")
    conn.execute("INSERT INTO projects VALUES (?, ?)", ("project-a", "A"))
    conn.execute(
        "INSERT INTO roadmaps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("profile-a", "project-a", "r", "R", None, "draft", 1, "actor", "actor", 1, 1),
    )
    # The pointer may precede its version inside one transaction.
    conn.execute(
        "INSERT INTO roadmap_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("profile-a", "project-a", "r", 1, "draft", "test", None, "actor", 1, None),
    )
    assert conn.execute("SELECT active_version FROM roadmaps").fetchone()[0] == 1
    conn.commit()
    conn.close()


def test_unresolved_active_version_fails_at_commit(tmp_path: Path) -> None:
    conn = open_contract_db(tmp_path / "projects.db")
    conn.execute("INSERT INTO projects VALUES (?, ?)", ("project-a", "A"))
    conn.execute(
        "INSERT INTO roadmaps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("profile-a", "project-a", "r", "R", None, "draft", 99, "actor", "actor", 1, 1),
    )
    assert conn.execute("SELECT active_version FROM roadmaps").fetchone()[0] == 99
    with pytest.raises(sqlite3.IntegrityError):
        conn.commit()
    conn.rollback()
    conn.close()


@pytest.mark.parametrize("other_scope", [
    ("profile-b", "project-a", "roadmap-a"),
    ("profile-a", "project-b", "roadmap-a"),
    ("profile-a", "project-a", "roadmap-b"),
])
def test_active_version_cannot_be_satisfied_by_other_scope(
    tmp_path: Path, other_scope: tuple[str, str, str]
) -> None:
    conn = open_contract_db(tmp_path / "projects.db")
    conn.executemany("INSERT INTO projects VALUES (?, ?)", [("project-a", "A"), ("project-b", "B")])
    for profile, project, roadmap in (("profile-a", "project-a", "roadmap-a"), other_scope):
        conn.execute(
            "INSERT INTO roadmaps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (profile, project, roadmap, "R", None, "draft", None, "actor", "actor", 1, 1),
        )
    conn.execute(
        "UPDATE roadmaps SET active_version=1 WHERE profile_id='profile-a' AND project_id='project-a' AND roadmap_id='roadmap-a'"
    )
    conn.execute(
        "INSERT INTO roadmap_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (*other_scope, 1, "draft", "test", None, "actor", 1, None),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.commit()
    conn.close()


def test_project_fk_is_not_profile_qualified_within_one_projects_db(tmp_path: Path) -> None:
    """profile_id is application validation; projects.db owns project identity."""
    conn = open_contract_db(tmp_path / "projects.db")
    conn.execute("INSERT INTO projects VALUES (?, ?)", ("project-a", "A"))
    for profile, roadmap in (("profile-a", "roadmap-a"), ("profile-b", "roadmap-b")):
        conn.execute(
            "INSERT INTO roadmaps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (profile, "project-a", roadmap, "R", None, "draft", None, "actor", "actor", 1, 1),
        )
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM roadmaps").fetchone()[0] == 2
    conn.close()


@pytest.mark.parametrize("table,column,value", [
    ("roadmaps", "profile_id", " "), ("roadmaps", "project_id", ""),
    ("roadmaps", "roadmap_id", "\t"), ("roadmaps", "created_by", " "),
    ("roadmap_nodes", "node_id", " "), ("roadmap_relations", "relation_id", ""),
    ("roadmap_todos", "todo_id", "\t"),
])
def test_empty_and_whitespace_identifiers_and_actors_are_rejected(
    tmp_path: Path, table: str, column: str, value: str
) -> None:
    conn = open_contract_db(tmp_path / "projects.db")
    seed_scope(conn)
    if table == "roadmaps":
        values = ["profile-a", "project-a", "bad", "R", None, "draft", None, "actor", "actor", 1, 1]
        values[{"profile_id": 0, "project_id": 1, "roadmap_id": 2, "created_by": 7}[column]] = value
        sql = "INSERT INTO roadmaps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    elif table == "roadmap_nodes":
        values = ["profile-a", "project-a", "roadmap-a", 1, "bad", None, "step", "B", None, "planned", 0, None, None, 1, 1]
        values[4] = value
        sql = "INSERT INTO roadmap_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    elif table == "roadmap_relations":
        values = ["profile-a", "project-a", "roadmap-a", 1, "bad", "node-a", "node-a", "depends_on", "active", None]
        values[4] = value
        sql = "INSERT INTO roadmap_relations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    else:
        values = ["profile-a", "project-a", "roadmap-a", 1, "bad", None, "T", "open", 0, 1, 1]
        values[4] = value
        sql = "INSERT INTO roadmap_todos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(sql, values)
    conn.close()


def test_nodes_reject_missing_and_self_parent(tmp_path: Path) -> None:
    conn = open_contract_db(tmp_path / "projects.db")
    seed_scope(conn)
    for parent in ("missing", "child"):
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO roadmap_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("profile-a", "project-a", "roadmap-a", 1, "child", parent, "step", "B", None, "planned", 0, None, None, 1, 1),
            )
    conn.close()


def test_relations_reject_self_and_cross_scope_endpoints(tmp_path: Path) -> None:
    conn = open_contract_db(tmp_path / "projects.db")
    seed_scope(conn)
    for from_node, to_node, profile in (("node-a", "node-a", "profile-a"), ("node-a", "node-a", "profile-b")):
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO roadmap_relations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (profile, "project-a", "roadmap-a", 1, "rel-" + profile, from_node, to_node, "depends_on", "active", None),
            )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO roadmap_relations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("profile-a", "project-a", "roadmap-a", 1, "bad-state", "node-a", "node-a", "depends_on", "bogus", None),
        )
    conn.close()


def test_todos_reject_missing_node_and_cross_scope_node(tmp_path: Path) -> None:
    conn = open_contract_db(tmp_path / "projects.db")
    seed_scope(conn)
    for node, profile in (("missing", "profile-a"), ("node-a", "profile-b")):
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO roadmap_todos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (profile, "project-a", "roadmap-a", 1, "todo-" + profile, node, "T", "open", 0, 1, 1),
            )
    conn.close()


def test_additive_migration_preserves_multiple_legacy_tables_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE legacy_settings (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO legacy_settings VALUES ('keep', 'yes')")
    conn.execute("CREATE TABLE legacy_audit (id INTEGER PRIMARY KEY, detail TEXT)")
    conn.execute("INSERT INTO legacy_audit VALUES (1, 'preserve')")
    conn.execute("CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT NOT NULL)")
    conn.execute("CREATE INDEX legacy_audit_detail ON legacy_audit(detail)")
    conn.execute("INSERT INTO projects VALUES ('legacy-project', 'Legacy')")
    conn.commit()
    conn.close()

    migrated = open_contract_db(path)
    assert migrated.execute("SELECT value FROM legacy_settings WHERE key='keep'").fetchone()[0] == "yes"
    assert migrated.execute("SELECT detail FROM legacy_audit WHERE id=1").fetchone()[0] == "preserve"
    assert migrated.execute("SELECT name FROM projects WHERE id='legacy-project'").fetchone()[0] == "Legacy"
    assert "legacy_audit_detail" in {row[1] for row in migrated.execute("PRAGMA index_list(legacy_audit)")}
    migrated.close()


def test_incompatible_existing_roadmaps_table_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "incompatible.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE roadmaps (id INTEGER PRIMARY KEY, note TEXT)")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="incompatible existing roadmaps table"):
        open_contract_db(path)
    check = sqlite3.connect(path)
    assert {row[0] for row in check.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {"roadmaps"}
    check.close()


def test_same_columns_without_constraints_are_rejected_without_residue(tmp_path: Path) -> None:
    path = tmp_path / "weakened.db"
    conn = sqlite3.connect(path)
    # Deliberately preserve every column name/type while removing NOT NULL,
    # PK, FK, and CHECK constraints. A name-only compatibility check would
    # incorrectly accept this table.
    conn.execute("""
        CREATE TABLE roadmaps (
            profile_id TEXT, project_id TEXT, roadmap_id TEXT, title TEXT,
            purpose TEXT, lifecycle_state TEXT, active_version INTEGER,
            created_by TEXT, updated_by TEXT, created_at INTEGER, updated_at INTEGER
        )
    """)
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="incompatible existing roadmaps table"):
        open_contract_db(path)

    check = sqlite3.connect(path)
    assert {
        row[0] for row in check.execute("SELECT name FROM sqlite_master WHERE type='table'")
    } == {"roadmaps"}
    check.close()


def test_fixture_failure_rolls_back_new_roadmap_tables_and_preserves_legacy(tmp_path: Path) -> None:
    path = tmp_path / "injected-failure.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE legacy (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO legacy VALUES (1, 'keep')")
    conn.execute("CREATE INDEX legacy_value ON legacy(value)")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="injected fixture failure"):
        open_contract_db(path, fail_after_statement=2)

    check = sqlite3.connect(path)
    assert check.execute("SELECT value FROM legacy WHERE id=1").fetchone()[0] == "keep"
    assert "legacy_value" in {row[1] for row in check.execute("PRAGMA index_list(legacy)")}
    assert not {
        row[0] for row in check.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }.intersection({"roadmaps", "roadmap_versions", "roadmap_nodes", "roadmap_relations", "roadmap_todos"})
    check.close()
