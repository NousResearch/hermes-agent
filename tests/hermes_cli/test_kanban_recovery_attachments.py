import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli.kanban_generation import GenerationError, resolve_attachment_blob
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
