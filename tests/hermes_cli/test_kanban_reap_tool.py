"""Tests: kanban_reap — the worker-facing "retire a card you don't own" tool
(Problem B, kanban task t_a95da99a).

Unlike the automatic dispatcher reaper (test_kanban_reap_stale_siblings.py),
this is invoked by a WORKER exercising judgment — e.g. a coordinator that
notices a sibling card's premise went false (its blocking PR already merged
under a different card). It is deliberately NOT lineage-gated: the whole
point is retiring a card that is NOT a same-title/creator/assignee clone.

Safety instead comes from three mechanical limits enforced by
``kb.reap_task`` regardless of whether the caller's judgment is right:
action in {archive, unblock} only, source status in {blocked, scheduled}
only (never a live claim, never a resolved outcome), and a mandatory
``reason`` for the audit trail. The tool layer (``_handle_reap``) adds:
delegated-child rejection, same-board-only (no ``board`` override), and a
refusal to target the caller's own task (that's what kanban_complete/
kanban_block are for).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import delegation_context as dctx
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from tools import kanban_tools as kt


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def _mk(conn, *, status="blocked", title="target-card", assignee="w"):
    kwargs = {"initial_status": "blocked"} if status == "blocked" else {}
    tid = kb.create_task(conn, title=title, assignee=assignee, created_by="tester", **kwargs)
    if status not in ("blocked", "ready"):
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, tid))
        conn.commit()
    return tid


def _call_reap(conn, **args):
    """Invoke the tool handler exactly as the MCP surface would, closing over
    the already-open test ``conn`` isn't possible (the handler opens its own
    via _board) — so this just wraps args into the handler's expected shape."""
    return json.loads(kt._handle_reap(args))


class TestReapTaskPrimitive:
    """Unit tests for hermes_cli.kanban_db.reap_task."""

    def test_archives_blocked_task(self, conn):
        tid = _mk(conn, status="blocked")
        assert kb.reap_task(conn, tid, action="archive", reason="premise went false") is True
        assert conn.execute(
            "SELECT status FROM tasks WHERE id=?", (tid,)
        ).fetchone()["status"] == "archived"

    def test_unblocks_blocked_task(self, conn):
        tid = _mk(conn, status="blocked")
        assert kb.reap_task(conn, tid, action="unblock", reason="dependency actually resolved") is True
        assert conn.execute(
            "SELECT status FROM tasks WHERE id=?", (tid,)
        ).fetchone()["status"] in ("ready", "todo")

    def test_archives_scheduled_task(self, conn):
        tid = _mk(conn, status="scheduled")
        assert kb.reap_task(conn, tid, action="archive", reason="superseded") is True

    def test_rejects_running_task(self, conn):
        """Never touch a live claim — this is the core safety property since
        there is no lineage check to fall back on."""
        tid = _mk(conn, status="running")
        with pytest.raises(ValueError, match="running"):
            kb.reap_task(conn, tid, action="archive", reason="x")
        assert conn.execute(
            "SELECT status FROM tasks WHERE id=?", (tid,)
        ).fetchone()["status"] == "running"

    def test_rejects_review_task(self, conn):
        tid = _mk(conn, status="review")
        with pytest.raises(ValueError, match="review"):
            kb.reap_task(conn, tid, action="archive", reason="x")

    def test_rejects_done_task(self, conn):
        tid = _mk(conn, status="done")
        with pytest.raises(ValueError, match="done"):
            kb.reap_task(conn, tid, action="archive", reason="x")

    def test_rejects_already_archived_task(self, conn):
        tid = _mk(conn, status="archived")
        with pytest.raises(ValueError, match="archived"):
            kb.reap_task(conn, tid, action="archive", reason="x")

    def test_rejects_empty_reason(self, conn):
        tid = _mk(conn, status="blocked")
        with pytest.raises(ValueError, match="reason"):
            kb.reap_task(conn, tid, action="archive", reason="")

    def test_rejects_whitespace_only_reason(self, conn):
        tid = _mk(conn, status="blocked")
        with pytest.raises(ValueError, match="reason"):
            kb.reap_task(conn, tid, action="archive", reason="   ")

    def test_rejects_bad_action(self, conn):
        tid = _mk(conn, status="blocked")
        with pytest.raises(ValueError, match="action"):
            kb.reap_task(conn, tid, action="complete", reason="x")  # type: ignore[arg-type]

    def test_rejects_unknown_task(self, conn):
        with pytest.raises(ValueError, match="unknown"):
            kb.reap_task(conn, "t_doesnotexist", action="archive", reason="x")

    def test_no_lineage_requirement(self, conn):
        """The defining property vs reap_stale_sibling: totally unrelated
        title/creator/assignee must still be reapable."""
        tid = _mk(conn, status="blocked", title="wildly different title", assignee="nobody-related")
        assert kb.reap_task(conn, tid, action="archive", reason="its blocking PR merged elsewhere") is True

    def test_archive_then_unblock_is_rejected_not_reversible(self, conn):
        """Archiving is system-wide terminal — confirms the docstring claim
        against real unblock_task SQL rather than trusting the comment."""
        tid = _mk(conn, status="blocked")
        kb.reap_task(conn, tid, action="archive", reason="x")
        with pytest.raises(ValueError, match="archived"):
            kb.reap_task(conn, tid, action="unblock", reason="oops, undo")

    def test_event_and_attributed_comment_logged(self, conn):
        tid = _mk(conn, status="blocked")
        kb.reap_task(conn, tid, action="archive", reason="blocking PR #42 already merged",
                     actor_task_id="t_caller123", author="coordinator-profile")

        events = kb.list_events(conn, tid)
        reaped = [e for e in events if e.kind == "reaped_by_worker"]
        assert len(reaped) == 1
        assert reaped[0].payload["action"] == "archive"
        assert reaped[0].payload["actor_task_id"] == "t_caller123"
        assert "blocking PR #42" in reaped[0].payload["reason"]

        comments = kb.list_comments(conn, tid)
        assert any(c.author == "coordinator-profile" for c in comments)
        assert any("t_caller123" in (c.body or "") for c in comments)

    def test_default_author_is_worker(self, conn):
        tid = _mk(conn, status="blocked")
        kb.reap_task(conn, tid, action="archive", reason="x")
        comments = kb.list_comments(conn, tid)
        assert any(c.author == "worker" for c in comments)


class TestHandleReapTool:
    """Integration tests through the actual MCP tool handler."""

    def test_archive_via_tool(self, conn):
        tid = _mk(conn, status="blocked")
        result = _call_reap(conn, task_id=tid, action="archive", reason="premise false")
        assert result["ok"] is True
        assert result["status"] == "archived"

    def test_unblock_via_tool(self, conn):
        tid = _mk(conn, status="blocked")
        result = _call_reap(conn, task_id=tid, action="unblock", reason="resolved")
        assert result["ok"] is True
        assert result["status"] in ("ready", "todo")

    def test_missing_task_id_rejected(self, conn):
        result = _call_reap(conn, action="archive", reason="x")
        assert "error" in result
        assert "task_id" in result["error"]

    def test_missing_action_rejected(self, conn):
        tid = _mk(conn, status="blocked")
        result = _call_reap(conn, task_id=tid, reason="x")
        assert "error" in result
        assert "action" in result["error"]

    def test_missing_reason_rejected(self, conn):
        tid = _mk(conn, status="blocked")
        result = _call_reap(conn, task_id=tid, action="archive")
        assert "error" in result
        assert "reason" in result["error"]

    def test_bad_status_surfaces_as_tool_error_not_traceback(self, conn):
        tid = _mk(conn, status="running")
        result = _call_reap(conn, task_id=tid, action="archive", reason="x")
        assert "error" in result
        assert "running" in result["error"]

    def test_board_arg_is_ignored_not_honoured(self, conn, monkeypatch):
        """Passing board=<anything> must not let a worker reach outside its
        own board — the handler must never read args['board']."""
        tid = _mk(conn, status="blocked")
        result = _call_reap(conn, task_id=tid, action="archive", reason="x",
                             board="some-other-board-entirely")
        assert result["ok"] is True  # succeeded against the CALLER's own board, not the arg

    def test_delegated_child_rejected(self, conn):
        tid = _mk(conn, status="blocked")
        with dctx.delegated_child_context():
            result = _call_reap(conn, task_id=tid, action="archive", reason="x")
        assert "error" in result
        assert "delegate_task child" in result["error"]
        # and the task must be genuinely untouched
        assert conn.execute(
            "SELECT status FROM tasks WHERE id=?", (tid,)
        ).fetchone()["status"] == "blocked"

    def test_cannot_target_own_task(self, conn, monkeypatch):
        tid = _mk(conn, status="blocked")
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        result = _call_reap(conn, task_id=tid, action="archive", reason="x")
        assert "error" in result
        assert "own" in result["error"].lower()
        assert conn.execute(
            "SELECT status FROM tasks WHERE id=?", (tid,)
        ).fetchone()["status"] == "blocked"

    def test_can_target_other_task_when_own_task_env_set(self, conn, monkeypatch):
        """Only the EXACT own-task id is refused; a sibling is fair game."""
        own = _mk(conn, status="ready", title="my-own-card")
        other = _mk(conn, status="blocked", title="someone-elses-stale-card")
        monkeypatch.setenv("HERMES_KANBAN_TASK", own)
        result = _call_reap(conn, task_id=other, action="archive", reason="x")
        assert result["ok"] is True

    def test_no_lineage_requirement_through_tool(self, conn):
        """Confirms the tool layer doesn't add back a lineage check that the
        DB layer deliberately omits."""
        tid = _mk(conn, status="blocked", title="completely unrelated card")
        result = _call_reap(conn, task_id=tid, action="archive",
                             reason="blocking dependency shipped under a different card")
        assert result["ok"] is True

    def test_invalid_action_value_rejected(self, conn):
        tid = _mk(conn, status="blocked")
        result = _call_reap(conn, task_id=tid, action="complete", reason="x")
        assert "error" in result
        assert "action" in result["error"]
