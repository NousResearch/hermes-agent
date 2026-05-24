"""Tests for kanban git worktree provisioning."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_worktree as kwt


class KanbanWorktreeTests(unittest.TestCase):
    def _init_repo(self, root: Path) -> None:
        subprocess.run(["git", "init"], cwd=str(root), check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=str(root),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=str(root),
            check=True,
            capture_output=True,
        )
        (root / "README.md").write_text("hello\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=str(root), check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=str(root),
            check=True,
            capture_output=True,
        )

    def test_ensure_worktree_workspace_creates_git_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            self._init_repo(repo)
            wt_path = repo / ".worktrees" / "t_test"
            task = kb.Task(
                id="t_test",
                title="x",
                body=None,
                assignee="worker",
                status="ready",
                priority=0,
                created_by=None,
                created_at=0,
                started_at=None,
                completed_at=None,
                workspace_kind="worktree",
                workspace_path=str(wt_path),
                claim_lock=None,
                claim_expires=None,
                tenant=None,
                branch_name="wt/t_test",
            )
            created = kwt.ensure_worktree_workspace(task, wt_path, repo_root=repo)
            self.assertEqual(created, wt_path.resolve())
            self.assertTrue((wt_path / ".git").exists())
            self.assertTrue((wt_path / "README.md").exists())

    def test_apply_kanban_worker_workspace_sets_cwd_and_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            original_cwd = os.getcwd()
            try:
                os.environ["HERMES_KANBAN_WORKSPACE"] = str(workspace)
                agent = type("Agent", (), {})()
                applied = kwt.apply_kanban_worker_workspace(agent)
                self.assertEqual(applied, str(workspace))
                self.assertEqual(agent.session_cwd, str(workspace))
                self.assertEqual(Path(os.getcwd()), workspace.resolve())
            finally:
                os.chdir(original_cwd)
                os.environ.pop("HERMES_KANBAN_WORKSPACE", None)
                os.environ.pop("TERMINAL_CWD", None)
                os.environ.pop("HERMES_CURSOR_AUX_CWD", None)

    def test_complete_task_adds_completion_comment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / ".hermes"
            home.mkdir()
            with patch.dict(os.environ, {"HERMES_HOME": str(home)}, clear=False), patch.object(
                Path, "home", lambda: Path(tmp)
            ):
                kb.init_db()
                conn = kb.connect()
                try:
                    tid = kb.create_task(conn, title="x", assignee="worker")
                    kb.claim_task(conn, tid)
                    ok = kb.complete_task(
                        conn,
                        tid,
                        summary="shipped the fix",
                        metadata={"changed_files": ["a.py"]},
                    )
                    self.assertTrue(ok)
                    comments = kb.list_comments(conn, tid)
                finally:
                    conn.close()
            self.assertEqual(len(comments), 1)
            self.assertIn("shipped the fix", comments[0].body)
            self.assertIn("changed_files", comments[0].body)
            self.assertEqual(comments[0].author, "worker")


if __name__ == "__main__":
    unittest.main()
