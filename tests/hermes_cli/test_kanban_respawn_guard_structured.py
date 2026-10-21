"""The respawn guard decides from structured run state, never from prose.

``check_respawn_guard`` used to regex ``tasks.last_failure_error`` for
quota/auth words and park the card on a match (``blocker_auth``). That
column mixes our own templates, exception text, provider ids and the
worker's captured stdout, so ordinary words parked healthy cards with no
escalation: "author" (#117009), ``claude auth status`` in crash output
(#117097), and the provider id ``claude-subscription-directsdk-experimental``
surviving on the task row after a later run had already succeeded. Provider
conditions arrive structurally — exit 75 → ``rate_limited`` run →
``rate_limit_cooldown``; exit 78 → sticky ``blocked``; anything else is a
bounded crash — so the guard reads only the latest run's outcome.
"""

from __future__ import annotations

import sqlite3
import time
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


def _fake_spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42
    return fake_spawn


def _review_dispatch_on(monkeypatch):
    import hermes_cli.config as cfgmod
    monkeypatch.setattr(
        cfgmod, "load_config", lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )


_PROVIDER_TIMEOUT_OUTPUT = (
    "claude-subscription-directsdk-experimental did not answer in time. "
    "Send /retry, or /model to switch."
)


def test_crash_text_outliving_a_successful_run_does_not_park_the_review(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Crash (provider id in the worker's last output) → the next worker run
    succeeds and requests review → the reviewer must spawn on the next tick.
    Drives the real path: ``detect_crashed_workers`` stamps the text,
    ``request_review`` ends the run, ``dispatch_once`` spawns."""
    import hermes_cli.kanban_db as _kb

    _review_dispatch_on(monkeypatch)
    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kbd, "_worker_final_output", lambda *a, **k: _PROVIDER_TIMEOUT_OUTPUT)

    with kbc.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="cull plumbing", assignee="worker")

        # Run 1: worker dies with exit 1 after printing the provider timeout.
        kb.claim_task(conn, tid, claimer=f"{host}:w0")
        pid = 71001
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, tid))
        conn.commit()
        kbd._record_worker_exit(pid, 1 << 8)
        assert tid in kbd.detect_crashed_workers(conn)
        stamped = kb.get_task(conn, tid).last_failure_error or ""
        assert "subscription" in stamped  # the trap, as recorded by the real sweep
        # The crash happened a minute before the retry (a same-second tie on
        # ``ended_at`` would let the crash row shadow the later success).
        conn.execute(
            "UPDATE task_runs SET ended_at = ended_at - 60 WHERE task_id=? AND outcome='crashed'",
            (tid,),
        )
        conn.commit()

        # Run 2: worker succeeds and hands off to review.
        claimed = kb.claim_task(conn, tid, claimer=f"{host}:w1")
        assert kb.request_review(
            conn, tid, summary="done", reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        )
        assert kb.get_task(conn, tid).status == "review"

        assert kbd.check_respawn_guard(conn, tid, lane="review") is None

        spawns: list = []
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2)

    assert [task_id for task_id, *_ in res.spawned] == [tid]
    assert res.respawn_guarded == []


def test_spawn_failure_exception_text_does_not_park_the_card(kanban_home):
    """A workspace ``PermissionError`` is a bounded crash for the breaker, not a
    provider auth verdict: the guard must not read "Permission denied"."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ws", assignee="worker")
        kb.claim_task(conn, tid)
        tripped = kbd._record_task_failure(
            conn, tid, "workspace: [Errno 13] Permission denied: '/srv/repo'",
            outcome="spawn_failed", failure_limit=3, release_claim=True, end_run=True,
        )
        assert tripped is False
        assert kb.get_task(conn, tid).status == "ready"
        assert kbd.check_respawn_guard(conn, tid) is None


@pytest.mark.parametrize(
    "error_text",
    [
        "401 auth failed",
        "authentication error from provider",
        "429 quota exceeded",
        "billing hard limit reached",
        "pid 1 exited with code 1 Worker's last output: 'claude auth status'",
        "Workstream C items C-3 and C-4: author t  (90.59s)",
    ],
)
def test_last_failure_error_prose_never_yields_a_guard_reason(kanban_home, error_text):
    """``last_failure_error`` is operator context; no word in it is a dispatch input."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="prose", assignee="worker")
        conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (error_text, tid))
        conn.commit()
        assert kbd.check_respawn_guard(conn, tid) is None
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None


def test_structured_provider_outcomes_still_guard(kanban_home, monkeypatch):
    """Deleting the prose guard must leave the structured paths intact: a
    ``rate_limited`` latest run cools down, and a terminal provider exit (78)
    parks the card ``blocked`` on the first death."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    now = int(time.time())
    with kbc.connect() as conn:
        rl = kb.create_task(conn, title="rl", assignee="worker")
        kb.claim_task(conn, rl)
        run_id = kb.get_task(conn, rl).current_run_id
        conn.execute(
            "UPDATE task_runs SET outcome='rate_limited', status='rate_limited', ended_at=? WHERE id=?",
            (now, run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL WHERE id=?", (rl,),
        )
        conn.commit()
        assert kbd.check_respawn_guard(conn, rl) == "rate_limit_cooldown"

        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
        monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
        host = _kb._claimer_id().split(":", 1)[0]
        tp = kb.create_task(conn, title="terminal", assignee="worker", max_retries=5)
        kb.claim_task(conn, tp, claimer=f"{host}:w0")
        pid = 71002
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, tp))
        conn.commit()
        kbd._record_worker_exit(pid, _kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE << 8)
        assert tp in kbd.detect_crashed_workers(conn)
        assert kb.get_task(conn, tp).status == "blocked"
