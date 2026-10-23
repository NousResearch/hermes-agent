"""Respawn-guard ``blocker_auth`` exemptions (hermes_cli.kanban_db_dispatch)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_respawn_guard_ignores_auth_words_in_crashed_or_spawn_failed_runs(kanban_home):
    """Crashed and spawn_failed error text is never a provider auth diagnosis.

    ``_classify_dead_worker`` appends the worker's last output to the persisted
    failure text.  A benign command such as ``claude auth status`` must not turn
    an unrelated crash into a permanent auth guard on the next dispatch.  A
    spawn failure is a local host/workspace error raised before any model call:
    a branch named ``billing-dw/...`` in a git worktree error must not park the
    card.  Other outcomes (``timed_out``) still trip ``blocker_auth``.
    """
    with kbc.connect() as conn:
        crashed_id = kb.create_task(conn, title="crashed", assignee="a")
        kb.claim_task(conn, crashed_id)
        crashed_run_id = kb.get_task(conn, crashed_id).current_run_id
        conn.execute(
            "UPDATE task_runs SET outcome='crashed', status='failed', ended_at=? "
            "WHERE id=?",
            (5_000_000, crashed_run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
            "last_failure_error=? WHERE id=?",
            (
                "pid 1 killed by signal 9. Worker's last output: "
                "'env -u ANTHROPIC_API_KEY claude auth status --text'",
                crashed_id,
            ),
        )

        spawn_failed_id = kb.create_task(conn, title="spawn failed", assignee="a")
        kb.claim_task(conn, spawn_failed_id)
        spawn_run_id = kb.get_task(conn, spawn_failed_id).current_run_id
        conn.execute(
            "UPDATE task_runs SET outcome='spawn_failed', status='failed', ended_at=? "
            "WHERE id=?",
            (5_000_000, spawn_run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
            "last_failure_error=? WHERE id=?",
            (
                "workspace: git worktree add failed for /x/.worktrees/t_1 on "
                "branch billing-dw/integration: fatal: 'billing-dw/integration' "
                "is already used by worktree at '/x/.worktrees/t_0'",
                spawn_failed_id,
            ),
        )

        timed_out_id = kb.create_task(conn, title="timed out", assignee="a")
        kb.claim_task(conn, timed_out_id)
        timed_out_run_id = kb.get_task(conn, timed_out_id).current_run_id
        conn.execute(
            "UPDATE task_runs SET outcome='timed_out', status='failed', ended_at=? "
            "WHERE id=?",
            (5_000_000, timed_out_run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
            "last_failure_error=? WHERE id=?",
            ("provider authentication failed", timed_out_id),
        )
        conn.commit()

        assert kbd.check_respawn_guard(conn, crashed_id) is None
        assert kbd.check_respawn_guard(conn, spawn_failed_id) is None
        assert kbd.check_respawn_guard(conn, timed_out_id) == "blocker_auth"
