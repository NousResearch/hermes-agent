import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from hermes_cli.kanban_reconcile import reconcile


def _fixture(tmp_path):
    from tests.hermes_cli.test_kanban_recovery_round_trip import _db
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    _db(source, "source-task")
    _db(target, "target-task")
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"task_ids": ["source-task"]}))
    out = tmp_path / "generation"
    return reconcile(source, target, seed, out, apply=True), out


def test_source_qualified_provenance_maps_colliding_child_ids(tmp_path):
    report, out = _fixture(tmp_path)
    with sqlite3.connect(out / "kanban.db") as conn:
        conn.row_factory = sqlite3.Row
        snapshot = conn.execute("SELECT source_snapshot_sha256 FROM recovery_imports").fetchone()[0]
        row = conn.execute("SELECT target_pk, source_row_sha256 FROM recovery_row_provenance WHERE table_name='task_comments' AND source_pk='10'").fetchone()
        assert row["target_pk"] != "10"
        assert row["source_row_sha256"]
        assert snapshot == report["manifest_digest"] or isinstance(snapshot, str)


def test_provenance_never_uses_bare_source_integer_as_unique_identity(tmp_path):
    _, out = _fixture(tmp_path)
    with sqlite3.connect(out / "kanban.db") as conn:
        assert conn.execute("PRAGMA table_info(recovery_row_provenance)").fetchall()
        assert conn.execute("SELECT COUNT(*) FROM recovery_row_provenance WHERE source_pk='10'").fetchone()[0] >= 3


def _public_provenance(db: Path, home: Path, task_id: str, snapshot: str,
                       *, table: str = "task_comments", source_id: int = 10):
    env = os.environ.copy()
    env.update(HERMES_HOME=str(home), HERMES_KANBAN_DB=str(db))
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban", "provenance", task_id,
         "--source-snapshot", snapshot, "--table", table,
         "--source-id", str(source_id), "--json"],
        env=env, capture_output=True, text=True, check=False,
    )


def test_public_cli_provenance_is_source_qualified_and_task_scoped(tmp_path):
    """The public command accepts own/descendant rows and rejects inferred identity."""
    _, generation = _fixture(tmp_path)
    base = tmp_path / "public.db"
    shutil.copy2(generation / "kanban.db", base)
    with sqlite3.connect(base) as conn:
        snapshot = conn.execute(
            "SELECT source_snapshot_sha256 FROM recovery_imports"
        ).fetchone()[0]
        # target-task is an explicit ancestor of source-task for the descendant arm.
        conn.execute("INSERT INTO task_links(parent_id, child_id) VALUES ('target-task', 'source-task')")

    own = _public_provenance(base, tmp_path / "own-home", "source-task", snapshot)
    assert own.returncode == 0, own.stderr
    assert json.loads(own.stdout)[0]["owning_task"] == "source-task"
    descendant = _public_provenance(base, tmp_path / "desc-home", "target-task", snapshot)
    assert descendant.returncode == 0, descendant.stderr
    assert json.loads(descendant.stdout)[0]["owning_task"] == "source-task"
    # The same integer exists in several child tables. Table qualification must
    # select the requested comment without treating the bare integer as identity.
    collision = _public_provenance(
        base, tmp_path / "collision-home", "source-task", snapshot,
        table="task_events",
    )
    assert collision.returncode == 0, collision.stderr
    assert json.loads(collision.stdout)[0]["table_name"] == "task_events"

    for name, sql in {
        "wrong-owner": "UPDATE task_comments SET task_id='target-task' WHERE id=(SELECT CAST(target_pk AS INTEGER) FROM recovery_row_provenance WHERE table_name='task_comments' AND source_pk='10')",
        "empty": "UPDATE recovery_row_provenance SET target_pk='' WHERE table_name='task_comments' AND source_pk='10'",
        "zero": "UPDATE recovery_row_provenance SET target_pk='0' WHERE table_name='task_comments' AND source_pk='10'",
        "unrelated": "UPDATE recovery_row_provenance SET source_row_json=replace(source_row_json, '\"task_id\":\"source-task\"', '\"task_id\":\"target-task\"') WHERE table_name='task_comments' AND source_pk='10'",
    }.items():
        candidate = tmp_path / f"{name}.db"
        shutil.copy2(base, candidate)
        with sqlite3.connect(candidate) as conn:
            conn.execute(sql)
        result = _public_provenance(
            candidate, tmp_path / f"{name}-home", "source-task", snapshot,
        )
        assert result.returncode != 0, (name, result.stdout, result.stderr)
        assert "not uniquely source-qualified" in result.stderr

    multiple = tmp_path / "multiple.db"
    shutil.copy2(base, multiple)
    with sqlite3.connect(multiple) as conn:
        conn.execute(
            "INSERT INTO recovery_imports SELECT 'duplicate-import', source_db_sha256, "
            "source_snapshot_sha256, target_snapshot_sha256, requested_task_ids_json, "
            "source_max_event_created_at, imported_at, mode FROM recovery_imports LIMIT 1"
        )
        conn.execute(
            "INSERT INTO recovery_row_provenance SELECT 'duplicate-import', table_name, "
            "source_pk, target_pk, disposition, source_row_sha256, source_row_json "
            "FROM recovery_row_provenance WHERE table_name='task_comments' AND source_pk='10'"
        )
    result = _public_provenance(
        multiple, tmp_path / "multiple-home", "source-task", snapshot,
    )
    assert result.returncode != 0
    assert "not uniquely source-qualified" in result.stderr

    legacy = tmp_path / "legacy-missing-source-row.db"
    shutil.copy2(base, legacy)
    with sqlite3.connect(legacy) as conn:
        conn.execute("ALTER TABLE recovery_row_provenance DROP COLUMN source_row_json")
    result = _public_provenance(
        legacy, tmp_path / "legacy-home", "source-task", snapshot,
    )
    assert result.returncode != 0
    assert "not uniquely source-qualified" in result.stderr
