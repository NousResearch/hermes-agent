"""Respawn-guard ``active_pr`` re-queue bypass tests.

Mirrors the ``recent_success`` exception: an explicit re-queue after a PR-URL
comment is a deliberate re-run and must not be held by the 24h PR window.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


PR_COMMENT = "Opened https://github.com/example/repo/pull/456 — follow-up needed."


def test_active_pr_guard_defers_ready_lane_without_requeue(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="already PRed", assignee="worker")
        kb.add_comment(conn, tid, author="worker", body=PR_COMMENT)
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_active_pr_guard_honors_manual_promote_after_pr_comment(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="PR then manual promote", assignee="worker")
        kb.add_comment(conn, tid, author="worker", body=PR_COMMENT)
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"

        kb.block_task(conn, tid, reason="waiting on parent", kind="dependency")
        assert kb.get_task(conn, tid).status == "todo"
        ok, _err = kb.promote_task(conn, tid, actor="dispatcher", force=True)
        assert ok is True

        assert kbd.check_respawn_guard(conn, tid) is None


def test_active_pr_guard_honors_unblock_after_pr_comment(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="PR then human unblock", assignee="worker")
        kb.add_comment(conn, tid, author="worker", body=PR_COMMENT)
        kb.block_task(conn, tid, reason="needs merge", kind="needs_input")
        assert kb.get_task(conn, tid).status == "blocked"

        kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"
        assert kbd.check_respawn_guard(conn, tid) is None
