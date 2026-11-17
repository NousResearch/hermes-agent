import sqlite3

from hermes_cli.kanban_generation import resolve_attachment_blob


def test_dashboard_style_attachment_read_uses_generation_custody(tmp_path):
    from tests.hermes_cli.test_kanban_recovery_round_trip import _db
    from hermes_cli.kanban_reconcile import reconcile

    source, target = tmp_path / "source.db", tmp_path / "target.db"
    old = tmp_path / "legacy.txt"
    old.write_bytes(b"dashboard custody")
    _db(source, "source-task")
    _db(target, "target-task", old)
    seed = tmp_path / "seed.json"
    seed.write_text('["source-task"]')
    generation = tmp_path / "generation"
    reconcile(source, target, seed, generation, apply=True)
    old.unlink()
    with sqlite3.connect(generation / "kanban.db") as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM task_attachments WHERE id=10").fetchone()
        resolved = resolve_attachment_blob(conn, row, board=generation)
        assert resolved.read_bytes() == b"dashboard custody"
        assert resolved != old
