"""Comment-watermark queries in kanban_db.

``list_comments_after`` backs the live worker bridge: it returns only comments
newer than a cursor so a running worker folds in new operator notes without
re-reading the whole thread.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_HOME", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    return home


def test_list_comments_after_cursor(fresh_home):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="chat")
        c1 = kb.add_comment(conn, tid, author="alice", body="first")
        c2 = kb.add_comment(conn, tid, author="bob", body="second")

        assert [c.id for c in kb.list_comments_after(conn, tid, after_id=0)] == [c1, c2]

        newer = kb.list_comments_after(conn, tid, after_id=c1)
        assert [c.id for c in newer] == [c2]
        assert newer[0].body == "second"

        assert kb.list_comments_after(conn, tid, after_id=c2) == []
    finally:
        conn.close()


def test_worker_context_renders_recent_comments(fresh_home, monkeypatch):
    """``recent_comments`` is the one bounded read: count cap, body cap, omitted
    count; ``build_worker_context`` renders exactly that set."""
    monkeypatch.setattr(kb, "_CTX_MAX_COMMENTS", 2)
    monkeypatch.setattr(kb, "_CTX_MAX_COMMENT_BYTES", 5)
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="chat")
        for body in ("first", "second", "third"):
            kb.add_comment(conn, tid, author="alice", body=body)
        shown, omitted = kb.recent_comments(conn, tid)
        assert omitted == 1
        assert [c.body for c in shown] == ["secon… [truncated, 1 chars omitted]", "third"]
        assert [c.body for c in kb.list_comments(conn, tid)] == ["first", "second", "third"]
        thread = kb.build_worker_context(conn, tid).split("## Comment thread\n", 1)[1].splitlines()
        assert thread[0] == "_(1 earlier comment omitted; showing most recent 2)_"
        assert [thread[2], thread[5]] == [c.body for c in shown]
        assert "first" not in "\n".join(thread)
    finally:
        conn.close()
