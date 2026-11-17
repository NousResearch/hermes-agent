"""Offline, deterministic source-to-target reconciliation."""
from __future__ import annotations

import hashlib
import json
import shutil
import secrets
import sqlite3
import tempfile
import os
from pathlib import Path
from typing import Any

from hermes_cli.kanban_generation import (
    GenerationError, build_ready, generation_id, initialize_generation_db,
    manifest_digest, row_manifest, sha256_file, _atomic_json,
)

CHILD_TABLES = ("task_comments", "task_runs", "task_events", "task_attachments")
SUPPORTED_TABLES = {
    "tasks", "task_links", "task_comments", "task_runs", "task_events",
    "task_attachments", "kanban_notify_subs", "kanban_generation_meta",
    "task_holds", "projects", "workspaces", "claims",
    "recovery_imports", "recovery_row_provenance", "recovery_attachment_blobs",
    "recovery_attachment_resolution",
}
SUPPORTED_INDEXES = {
    "task_links_parent", "recovery_provenance_lookup",
}
MIGRATED_INDEX_SQL = {
    "task_links_parent": "CREATE INDEX task_links_parent ON task_links(parent_id)",
    "idx_tasks_assignee_status": "CREATE INDEX idx_tasks_assignee_status ON tasks(assignee, status)",
    "idx_events_run": "CREATE INDEX idx_events_run ON task_events(run_id, id)",
    "idx_tasks_idempotency": "CREATE INDEX idx_tasks_idempotency ON tasks(idempotency_key)",
    "idx_tasks_session_id": "CREATE INDEX idx_tasks_session_id ON tasks(session_id)",
    "idx_tasks_tenant": "CREATE INDEX idx_tasks_tenant ON tasks(tenant)",
}
RUNTIME_FIELDS = {
    "claim_lock", "claim_expires", "last_heartbeat_at", "worker_pid",
    "worker_started_at", "current_run_id", "session_id", "workspace_path",
}
RECOVERY_SQL = (
    "CREATE TABLE recovery_imports (import_id TEXT PRIMARY KEY, source_db_sha256 TEXT NOT NULL, source_snapshot_sha256 TEXT NOT NULL, target_snapshot_sha256 TEXT NOT NULL, requested_task_ids_json TEXT NOT NULL, source_max_event_created_at INTEGER, imported_at INTEGER NOT NULL, mode TEXT NOT NULL)",
    "CREATE TABLE recovery_row_provenance (import_id TEXT NOT NULL, table_name TEXT NOT NULL, source_pk TEXT NOT NULL, target_pk TEXT NOT NULL, disposition TEXT NOT NULL, source_row_sha256 TEXT NOT NULL, source_row_json TEXT NOT NULL, PRIMARY KEY(import_id, table_name, source_pk))",
    "CREATE TABLE recovery_attachment_blobs (import_id TEXT NOT NULL, source_attachment_id INTEGER NOT NULL, source_path TEXT NOT NULL, source_sha256 TEXT NOT NULL, target_path TEXT NOT NULL, size INTEGER NOT NULL, PRIMARY KEY(import_id, source_attachment_id))",
    "CREATE TABLE recovery_attachment_resolution (generation_id TEXT NOT NULL, attachment_id INTEGER PRIMARY KEY, original_stored_path TEXT NOT NULL, source_row_sha256 TEXT NOT NULL, blob_sha256 TEXT NOT NULL, size INTEGER NOT NULL, custody_relpath TEXT NOT NULL, disposition TEXT NOT NULL)",
)


def _read_seeds(path: Path) -> list[str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise GenerationError("seed file must contain task_ids") from exc
    seeds = value.get("task_ids", value) if isinstance(value, dict) else value
    if not isinstance(seeds, list) or not all(isinstance(seed, str) and seed for seed in seeds):
        raise GenerationError("seed file must contain task_ids")
    return sorted(set(seeds))


def _closure(conn: sqlite3.Connection, seeds: list[str]) -> list[str]:
    if not seeds:
        raise GenerationError("seed file must contain task_ids")
    found = set(seeds)
    marks = ','.join('?' * len(seeds))
    rows = conn.execute(f"SELECT parent_id, child_id FROM task_links WHERE parent_id IN ({marks}) OR child_id IN ({marks})", seeds + seeds).fetchall()
    for parent, child in rows:
        found.update((parent, child))
    marks = ','.join('?' * len(found))
    missing = sorted(found - {row[0] for row in conn.execute(f"SELECT id FROM tasks WHERE id IN ({marks})", tuple(found))})
    if missing:
        raise GenerationError("missing seed task: " + ",".join(missing))
    return sorted(found)


def _child_ids(conn: sqlite3.Connection, table: str, tasks: list[str]) -> list[tuple[str, int]]:
    if not tasks:
        return []
    return [(row[0], int(row[1])) for row in conn.execute(f"SELECT task_id, id FROM {table} WHERE task_id IN ({','.join('?' * len(tasks))}) ORDER BY task_id, id", tasks)]


def _floor(conn: sqlite3.Connection, table: str) -> int:
    maximum = int(conn.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table}").fetchone()[0] or 0)
    try:
        row = conn.execute("SELECT COALESCE(seq, 0) FROM sqlite_sequence WHERE name = ?", (table,)).fetchone()
        sequence = int(row[0] if row else 0)
    except sqlite3.OperationalError:
        sequence = 0
    return max(maximum, sequence)


def _snapshot_db(source: Path, destination: Path) -> None:
    """Take one SQLite-consistent snapshot, including WAL pages."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst:
        src.backup(dst)
        dst.execute("PRAGMA wal_checkpoint(FULL)")
        dst.commit()
    with destination.open("rb") as stream:
        os.fsync(stream.fileno())


def _snapshot_pair(source: Path, target: Path) -> tuple[Path, Path, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory(prefix="hermes-kanban-snapshot-")
    source_snapshot, target_snapshot = Path(tmp.name) / "source.db", Path(tmp.name) / "target.db"
    _snapshot_db(source, source_snapshot)
    _snapshot_db(target, target_snapshot)
    return source_snapshot, target_snapshot, tmp


def _analyze_snapshots(source: Path, target: Path, seed_file: Path, output: Path) -> dict[str, Any]:
    with sqlite3.connect(source) as conn:
        _validate_source_schema(conn)
        seeds = _read_seeds(seed_file)
        closure = _closure(conn, seeds)
        remaps: dict[str, dict[str, int]] = {}
        with sqlite3.connect(target) as dest:
            _validate_source_schema(dest)
            for table in CHILD_TABLES:
                floor = _floor(dest, table)
                remaps[table] = {str(old): floor + offset for offset, (_, old) in enumerate(_child_ids(conn, table, closure), 1)}
    return {"dry_run": True, "source_db_sha256": sha256_file(source), "target_db_sha256": sha256_file(target), "seeds": seeds, "closure": closure, "remaps": remaps, "output_generation": str(output)}


def _normalized_schema(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return the complete SQLite schema contract, including implicit constraints.

    Comparing only ``sqlite_master.sql`` misses column defaults, NOT NULL/PK
    flags and foreign-key actions.  The admission contract therefore combines
    normalized SQL with every PRAGMA definition that SQLite exposes.
    """
    objects = conn.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY type, name"
    ).fetchall()
    tables: dict[str, Any] = {}
    for kind, name, sql in objects:
        if kind != "table":
            continue
        quoted = '"' + name.replace('"', '""') + '"'
        columns = [tuple(row) for row in conn.execute(f"PRAGMA table_info({quoted})")]
        foreign_keys = [tuple(row) for row in conn.execute(f"PRAGMA foreign_key_list({quoted})")]
        tables[name] = {
            "sql": " ".join((sql or "").split()).lower(),
            "columns": columns,
            "foreign_keys": foreign_keys,
        }
    return {
        "tables": tables,
        "indexes": {
            name: " ".join((sql or "").split()).lower()
            for kind, name, sql in objects if kind == "index"
        },
        "views": {name: " ".join((sql or "").split()).lower()
                  for kind, name, sql in objects if kind == "view"},
        "triggers": {name: " ".join((sql or "").split()).lower()
                     for kind, name, sql in objects if kind == "trigger"},
    }


def _validate_source_schema(conn: sqlite3.Connection) -> None:
    actual = _normalized_schema(conn)
    unknown = sorted(set(actual["tables"]) - SUPPORTED_TABLES)
    if unknown:
        raise GenerationError("unsupported persistent table: " + ",".join(unknown))
    canonical = _canonical_schema_objects()
    expected = canonical["schema"]
    # Recovery tables are deliberately absent from source/target inputs; they
    # are added only after both inputs have passed this gate.
    optional = {"task_links_parent": MIGRATED_INDEX_SQL["task_links_parent"].lower()}
    actual_indexes = dict(actual["indexes"])
    for name, sql in list(actual_indexes.items()):
        if name in optional and " ".join(sql.split()).lower() != " ".join(optional[name].split()).lower():
            raise GenerationError(f"canonical schema mismatch: index {name}")
    required_indexes = set(expected["indexes"])
    if set(actual_indexes) - required_indexes - set(optional) or not required_indexes <= set(actual_indexes):
        missing = sorted(required_indexes - set(actual_indexes))
        extra = sorted(set(actual_indexes) - required_indexes - set(optional))
        name = (missing or extra or ["unknown"])[0]
        raise GenerationError(f"canonical schema mismatch: index {name}")
    for name, expected_sql in expected["indexes"].items():
        if actual_indexes[name] != expected_sql:
            raise GenerationError(f"canonical schema mismatch: index {name}")
    if set(actual["tables"]) != set(expected["tables"]):
        raise GenerationError("canonical schema mismatch: table presence")
    for name, definition in expected["tables"].items():
        if actual["tables"][name] != definition:
            raise GenerationError(f"canonical schema mismatch: table {name}")
    if actual["views"] != expected["views"] or actual["triggers"] != expected["triggers"]:
        raise GenerationError("canonical schema mismatch: views or triggers")


def _canonical_schema_objects() -> dict[str, set[str]]:
    """Return the object names from a fresh native schema, never a hand copy.

    Reconciliation accepts the documented recovery extensions and legacy
    compatibility indexes separately, but the native baseline is generated by
    the production schema itself.  This keeps migrations and this admission
    gate from silently drifting apart.
    """
    from hermes_cli import kanban_db as native
    from hermes_cli import kanban_db_connect as native_connect
    with sqlite3.connect(":memory:") as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(native.SCHEMA_SQL)
        # The admission contract is the post-migration schema, not merely the
        # base script.  Reuse the production migration path so new additive
        # columns/definitions cannot silently diverge from reconciliation.
        native_connect._migrate_add_optional_columns(conn)
    # Apply the same post-SCHEMA_SQL additive index set used by init_db.  The
    # fresh schema is the source of truth; this list only describes migration
    # artifacts that are intentionally created after the base script.
    for name, sql in MIGRATED_INDEX_SQL.items():
        if name == "task_links_parent":
            # Older exports carried this duplicate of the native parent index;
            # it is accepted only when its definition is authenticated below,
            # but it is not required in the post-migration native schema.
            continue
        conn.execute(sql.replace("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS ", 1))
    return {"schema": _normalized_schema(conn)}


def _capture_blob(path: Path) -> tuple[bytes, str]:
    """Capture one regular, non-symlink file and detect concurrent changes."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise GenerationError("attachment path is not a regular file") from exc
    try:
        before = os.fstat(fd)
        if not __import__("stat").S_ISREG(before.st_mode):
            raise GenerationError("attachment path is not a regular file")
        chunks = []
        while chunk := os.read(fd, 1024 * 1024):
            chunks.append(chunk)
        data = b"".join(chunks)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns) or len(data) != before.st_size:
        raise GenerationError("attachment changed during capture")
    return data, hashlib.sha256(data).hexdigest()


def _install_blob(destination: Path, data: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != data:
            raise GenerationError("content-addressed blob conflict")
        return
    with destination.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def dry_run(source: Path, target: Path, seed_file: Path, output: Path) -> dict[str, Any]:
    if source.resolve() == target.resolve():
        raise GenerationError("source and target must be different")
    source_snapshot, target_snapshot, tmp = _snapshot_pair(source, target)
    try:
        return _analyze_snapshots(source_snapshot, target_snapshot, seed_file, output)
    finally:
        tmp.cleanup()


def _json_row(row: sqlite3.Row) -> str:
    return json.dumps({key: row[key] for key in row.keys()}, sort_keys=True, separators=(",", ":"), default=str)


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _add_recovery_tables(conn: sqlite3.Connection) -> None:
    for statement in RECOVERY_SQL:
        conn.execute(statement)
    conn.execute("CREATE INDEX recovery_provenance_lookup ON recovery_row_provenance(table_name, source_pk, source_row_sha256)")


def reconcile(source: Path, target: Path, seed_file: Path, output: Path, *, apply: bool = False) -> dict[str, Any]:  # health: allow CC -- one transaction is the reviewed reconciliation boundary; splitting it would obscure the source/target mapping invariants
    if not apply:
        return dry_run(source, target, seed_file, output)
    resolved_output = output.resolve()
    if output.exists() or target.resolve() == resolved_output or target.resolve() in resolved_output.parents:
        raise GenerationError("output generation already exists or contains active target")
    output.mkdir(parents=True)
    source_snap, target_snap, tmp_owner = _snapshot_pair(source, target)
    report = _analyze_snapshots(source_snap, target_snap, seed_file, output)
    source, target = source_snap, target_snap
    shutil.copy2(target, output / "kanban.db")
    # The output directory is the caller-owned generation identity.  The
    # pointer resolver addresses ``generations/<generation_id>``; using a
    # separate random id here would publish a pointer to a directory that was
    # never created.  Callers that need uniqueness choose a unique output
    # directory, while the logical manifest remains stable across runs.
    gid = output.name or generation_id()
    nonce = secrets.token_hex(16)
    # Capture the two independent input snapshots before adding generation or
    # recovery tables.  These digests are the replay/idempotency identity.
    with sqlite3.connect(source) as source_conn, sqlite3.connect(target) as target_conn:
        source_conn.row_factory = sqlite3.Row
        target_conn.row_factory = sqlite3.Row
        _validate_source_schema(source_conn)
        _validate_source_schema(target_conn)
        source_snapshot = row_manifest(source_conn)
        target_snapshot = row_manifest(target_conn)
    source_snapshot_sha = manifest_digest(source_snapshot)
    target_snapshot_sha = manifest_digest(target_snapshot)
    initialize_generation_db(output / "kanban.db", gid=gid, nonce=nonce, control_root=output.parent)
    import_id = "ri_" + hashlib.sha256((report["source_db_sha256"] + report["target_db_sha256"]).encode()).hexdigest()[:16]
    with sqlite3.connect(output / "kanban.db") as dest, sqlite3.connect(source) as src:
        dest.row_factory = sqlite3.Row
        src.row_factory = sqlite3.Row
        _add_recovery_tables(dest)
        dest.execute("BEGIN")
        try:
            max_event = src.execute("SELECT MAX(created_at) FROM task_events").fetchone()[0]
        except sqlite3.OperationalError:
            max_event = None
        dest.execute("INSERT INTO recovery_imports VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (import_id, report["source_db_sha256"], source_snapshot_sha, target_snapshot_sha, json.dumps(report["seeds"], separators=(",", ":")), max_event, 0, "reconcile"))
        # Replaying a source snapshot into the same target is deterministic; a
        # different snapshot with the same import identity is a hard conflict.
        prior = dest.execute("SELECT source_snapshot_sha256, target_snapshot_sha256 FROM recovery_imports WHERE import_id = ?", (import_id,)).fetchone()
        if prior is not None and tuple(prior) != (source_snapshot_sha, target_snapshot_sha):
            raise GenerationError("recovery idempotency conflict")
        # Preserve target attachment row bytes exactly while moving their file
        # custody into the sealed generation.  The original absolute path is
        # metadata only and is never used as a read fallback.
        for existing in dest.execute("SELECT * FROM task_attachments ORDER BY id").fetchall():
            source_path = Path(str(existing["stored_path"]))
            if not source_path.is_file():
                raise GenerationError("uncustodied target attachment")
            blob, digest = _capture_blob(source_path)
            relative = Path("blobs") / "sha256" / digest[:2] / digest
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            _install_blob(destination, blob)
            dest.execute("INSERT INTO recovery_attachment_resolution VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (gid, int(existing["id"]), str(existing["stored_path"]), manifest_digest(_row_dict(existing)), digest, len(blob), str(relative), "legacy-custodied"))
        tasks = report["closure"]
        target_wins: set[str] = set()
        for task_id in tasks:
            source_row = src.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            target_row = dest.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if source_row is None:
                raise GenerationError("missing source task: " + task_id)
            source_keys = set(source_row.keys())
            if "idempotency_key" in source_keys and source_row["idempotency_key"]:
                collision = dest.execute("SELECT id FROM tasks WHERE idempotency_key = ? AND id != ?", (source_row["idempotency_key"], task_id)).fetchone()
                if collision is not None:
                    raise GenerationError("idempotency collision")
            if target_row is not None:
                if task_id in report["seeds"]:
                    raise GenerationError("requested task collision")
                disposition = "target-wins"
                target_id = task_id
                target_wins.add(task_id)
            else:
                values = [source_row[column] for column in source_row.keys()]
                safety = {"status": "triage", "assignee": None, "project_id": None,
                          "branch_name": None}
                safety.update({column: None for column in RUNTIME_FIELDS})
                values = [safety.get(column, value) for column, value in zip(source_row.keys(), values)]
                dest.execute(f"INSERT INTO tasks ({','.join(source_row.keys())}) VALUES ({','.join('?' * len(values))})", values)
                disposition, target_id = "imported", task_id
            dest.execute("INSERT INTO recovery_row_provenance VALUES (?, 'tasks', ?, ?, ?, ?, ?)", (import_id, task_id, str(target_id), disposition, manifest_digest(_row_dict(source_row)), _json_row(source_row)))
        # Preserve the selected incident/context edges, but never recursively
        # widen the closure beyond the one-hop admission decision.
        for link in src.execute(
            "SELECT parent_id, child_id FROM task_links WHERE parent_id IN ({0}) AND child_id IN ({0}) ORDER BY parent_id, child_id".format(','.join('?' * len(tasks))),
            tasks * 2,
        ).fetchall():
            if dest.execute("SELECT 1 FROM task_links WHERE parent_id=? AND child_id=?", tuple(link)).fetchone() is None:
                dest.execute("INSERT INTO task_links(parent_id, child_id) VALUES (?, ?)", tuple(link))
        maps: dict[str, dict[int, int]] = {}
        for table in CHILD_TABLES:
            maps[table] = {}
            imported_tasks = [task for task in tasks if task not in target_wins]
            rows = src.execute(f"SELECT * FROM {table} WHERE task_id IN ({','.join('?' * len(imported_tasks))}) ORDER BY task_id, id", imported_tasks).fetchall() if imported_tasks else []
            floor = _floor(dest, table)
            for offset, row in enumerate(rows, 1):
                maps[table][int(row["id"])] = floor + offset
            for row in rows:
                old_id, new_id = int(row["id"]), maps[table][int(row["id"])]
                values = [row[column] for column in row.keys()]
                values[list(row.keys()).index("id")] = new_id
                if table == "task_runs":
                    columns = list(row.keys())
                    if "status" in columns and row["status"] == "running":
                        values[columns.index("status")] = "released"
                    for column in RUNTIME_FIELDS:
                        if column in columns:
                            values[columns.index(column)] = None
                if table == "task_events" and row["run_id"] is not None:
                    if int(row["run_id"]) not in maps["task_runs"]:
                        raise GenerationError("dangling run reference")
                    values[list(row.keys()).index("run_id")] = maps["task_runs"][int(row["run_id"])]
                if table == "task_attachments":
                    source_path = Path(str(row["stored_path"]))
                    if not source_path.is_file():
                        raise GenerationError("missing source attachment")
                    blob, digest = _capture_blob(source_path)
                    relative = Path("blobs") / "sha256" / digest[:2] / digest
                    destination = output / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    _install_blob(destination, blob)
                    values[list(row.keys()).index("stored_path")] = str(relative)
                    rewritten = _row_dict(row)
                    rewritten["stored_path"] = str(relative)
                    # The resolver authenticates the rewritten target row.  The
                    # source absolute path remains in recovery_attachment_blobs
                    # as provenance and is never used as custody.
                    dest.execute("INSERT INTO recovery_attachment_resolution VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (gid, new_id, str(relative), manifest_digest(rewritten), digest, len(blob), str(relative), "imported"))
                    source_key = old_id
                    if dest.execute("SELECT 1 FROM recovery_attachment_blobs WHERE import_id=? AND source_attachment_id=?", (import_id, source_key)).fetchone():
                        source_key = -old_id
                    dest.execute("INSERT INTO recovery_attachment_blobs VALUES (?, ?, ?, ?, ?, ?)", (import_id, source_key, str(source_path), digest, str(relative), len(blob)))
                dest.execute(f"INSERT INTO {table} ({','.join(row.keys())}) VALUES ({','.join('?' * len(values))})", values)
                dest.execute("INSERT INTO recovery_row_provenance VALUES (?, ?, ?, ?, 'imported', ?, ?)", (import_id, table, str(old_id), str(new_id), manifest_digest(_row_dict(row)), _json_row(row)))
            for task_id in sorted(target_wins):
                for row in src.execute(f"SELECT * FROM {table} WHERE task_id = ? ORDER BY id", (task_id,)).fetchall():
                    dest.execute("INSERT INTO recovery_row_provenance VALUES (?, ?, ?, ?, 'target-wins', ?, ?)", (import_id, table, str(row["id"]), "", manifest_digest(_row_dict(row)), _json_row(row)))
            try:
                dest.execute("UPDATE sqlite_sequence SET seq = ? WHERE name = ?", (max([_floor(dest, table)] + list(maps[table].values())), table))
            except sqlite3.OperationalError:
                pass
        dest.commit()
    ready = build_ready(output, gid=gid, nonce=nonce)
    report.update({"dry_run": False, "generation_id": gid, "generation_nonce": nonce, "ready_sha256": sha256_file(output / "READY.json"), "manifest_digest": ready["manifest_digest"]})
    _atomic_json(output / "import-journal.json", {"import_id": import_id, "generation_id": gid, "state": "prepared"})
    tmp_owner.cleanup()
    return report
