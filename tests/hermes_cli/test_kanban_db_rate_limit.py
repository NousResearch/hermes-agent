"""Respawn backoff for consecutive rate-limited runs (hermes_cli.kanban_db_rate_limit)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_respawn_guard_rate_limit_backoff_grows_5_15_45(kanban_home, monkeypatch):
    """Consecutive rate-limited runs back off 5 -> 15 -> 45 min and stay at 45; none counts as a crash."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    now = 5_000_000
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rl-backoff", assignee="a")
        for streak, wait in [(1, 300), (2, 900), (3, 2700), (4, 2700)]:
            kb.claim_task(conn, tid)
            run_id = kb.get_task(conn, tid).current_run_id
            conn.execute(
                "UPDATE task_runs SET outcome='rate_limited', status='rate_limited', "
                "ended_at=? WHERE id=?", (now + streak, run_id),
            )
            conn.execute(
                "UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, "
                "claim_expires=NULL, worker_pid=NULL WHERE id=?", (tid,),
            )
            conn.commit()
            monkeypatch.setattr(kb.time, "time", lambda: now + streak + wait - 1)
            assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown", streak
            monkeypatch.setattr(kb.time, "time", lambda: now + streak + wait + 1)
            assert kbd.check_respawn_guard(conn, tid) is None, streak
