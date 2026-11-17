import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli.kanban_generation import (
    GenerationError, resolve_attachment_blob, validate_generation,
)
from hermes_cli.kanban_reconcile import reconcile


def test_preserved_absolute_attachment_is_custodied_and_old_path_is_not_read(tmp_path):
    from tests.hermes_cli.test_kanban_recovery_round_trip import _db
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    old = tmp_path / "absolute-old.txt"
    old.write_bytes(b"custody")
    _db(source, "source-task")
    _db(target, "target-task", old)
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps(["source-task"]))
    generation = tmp_path / "generation"
    reconcile(source, target, seed, generation, apply=True)
    old.unlink()
    with sqlite3.connect(generation / "kanban.db") as conn:
        conn.row_factory = sqlite3.Row
        attachment = conn.execute("SELECT * FROM task_attachments WHERE id=10").fetchone()
        resolved = resolve_attachment_blob(conn, attachment, board=generation)
        assert resolved.read_bytes() == b"custody"
        assert str(resolved) != str(old)


def test_attachment_row_digest_mismatch_fails_closed(tmp_path):
    from tests.hermes_cli.test_kanban_recovery_round_trip import _db
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    old = tmp_path / "old.txt"
    old.write_bytes(b"x")
    _db(source, "source-task")
    _db(target, "target-task", old)
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps(["source-task"]))
    generation = tmp_path / "generation"
    reconcile(source, target, seed, generation, apply=True)
    with sqlite3.connect(generation / "kanban.db") as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("UPDATE task_attachments SET stored_path='/changed' WHERE id=10")
        conn.commit()
        with pytest.raises(GenerationError):
            resolve_attachment_blob(conn, conn.execute("SELECT * FROM task_attachments WHERE id=10").fetchone(), board=generation)


@pytest.mark.parametrize(
    "tamper", ["missing", "changed", "wrong-size", "wrong-digest", "symlink", "escape", "unexpected"]
)
def test_prepared_generation_binds_complete_attachment_blob_set(tmp_path, tamper):
    from tests.hermes_cli.test_kanban_recovery_round_trip import _db
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    old = tmp_path / "old.txt"
    old.write_bytes(b"prepared custody")
    _db(source, "source-task")
    _db(target, "target-task", old)
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps(["source-task"]))
    generation = tmp_path / "generation"
    reconcile(source, target, seed, generation, apply=True)
    with sqlite3.connect(generation / "kanban.db") as conn:
        relative = conn.execute(
            "SELECT custody_relpath FROM recovery_attachment_resolution"
        ).fetchone()[0]
        blob = generation / relative
        match tamper:
            case "missing":
                blob.unlink()
            case "changed":
                blob.write_bytes(b"changed")
            case "wrong-size":
                conn.execute("UPDATE recovery_attachment_resolution SET size = size + 1")
            case "wrong-digest":
                conn.execute("UPDATE recovery_attachment_resolution SET blob_sha256 = ?", ("f" * 64,))
            case "symlink":
                blob.unlink()
                blob.symlink_to(old)
            case "escape":
                conn.execute("UPDATE recovery_attachment_resolution SET custody_relpath = '../old.txt'")
            case "unexpected":
                unexpected = generation / "blobs" / "sha256" / "ee" / ("e" * 64)
                unexpected.parent.mkdir(parents=True)
                unexpected.write_bytes(b"unexpected")
        conn.commit()
    with pytest.raises(GenerationError):
        validate_generation(generation)
