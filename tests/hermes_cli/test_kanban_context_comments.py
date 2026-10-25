"""Worker handoff comment limits (regression for #138372)."""

from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def test_worker_context_keeps_latest_checkpoint_longer_than_older_comments(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="resume from checkpoint")
        older = "A" * 2100 + "older-tail"
        checkpoint = "B" * 2100 + "checkpoint-tail"
        for stamp, body in enumerate((older, checkpoint), start=1):
            comment_id = kb.add_comment(conn, task_id, "worker", body)
            conn.execute("UPDATE task_comments SET created_at = ? WHERE id = ?", (stamp, comment_id))
        conn.commit()

        context = kb.build_worker_context(conn, task_id)
        assert "older-tail" not in context
        assert "checkpoint-tail" in context
        assert f"[truncated, {len(older) - 2048} chars omitted]" in context

        oversized = "C" * 8200 + "oversized-tail"
        comment_id = kb.add_comment(conn, task_id, "worker", oversized)
        conn.execute("UPDATE task_comments SET created_at = 3 WHERE id = ?", (comment_id,))
        conn.commit()

        context = kb.build_worker_context(conn, task_id)
        assert "checkpoint-tail" not in context
        assert "oversized-tail" not in context
        assert "C" * 8192 in context
        assert f"[truncated, {len(oversized) - 8192} chars omitted]" in context
    finally:
        conn.close()
