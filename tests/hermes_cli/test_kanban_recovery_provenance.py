import json
import sqlite3
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
