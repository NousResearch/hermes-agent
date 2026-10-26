"""Respawn-guard attribution: a superseded failure text must not park a card.

``tasks.last_failure_error`` caches the last run that FAILED, not the latest run:
a rate-limited requeue stamps quota-flavoured text and a later handoff run never
clears it. The guard therefore has to attribute that text to the latest run before
parking a card as ``blocker_auth`` — otherwise every review card that ever hit a
quota wall stays parked before its lane exemption is ever reached.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
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


def _latest_run_id(conn, tid):
    """Id of the run just claimed for ``tid`` (``kb.get_task(...).current_run_id`` is Optional)."""
    run_id = conn.execute(
        "SELECT MAX(id) FROM task_runs WHERE task_id = ?", (tid,),
    ).fetchone()[0]
    assert run_id is not None
    return int(run_id)


def _close_run(conn, run_id, *, outcome, status, ended_at):
    conn.execute(
        "UPDATE task_runs SET outcome=?, status=?, ended_at=? WHERE id=?",
        (outcome, status, ended_at, run_id),
    )


def _seed_superseded_blocker_run(conn, *, title, latest_outcome, latest_status=None, ended_at=5_000_000):
    """A card whose ``last_failure_error`` was stamped by an OLDER, superseded run.

    Mirrors the real writers: the older run is closed ``rate_limited`` (the requeue
    that stamps the quota text), the newer run is closed with ``latest_outcome`` the
    way ``_end_run`` records it. The newer run ends LAST so the guard's
    ``ORDER BY ended_at DESC`` picks it.
    """
    tid = kb.create_task(conn, title=title, assignee="a")
    kb.claim_task(conn, tid)
    _close_run(conn, _latest_run_id(conn, tid), outcome="rate_limited", status="rate_limited", ended_at=ended_at)
    conn.execute(
        "UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, "
        "claim_expires=NULL, worker_pid=NULL, last_failure_error=? WHERE id=?",
        ("pid 69679 exited rate-limited (quota wall) — requeued without counting a failure", tid),
    )
    kb.claim_task(conn, tid)
    _close_run(
        conn, _latest_run_id(conn, tid),
        outcome=latest_outcome, status=latest_status or latest_outcome, ended_at=ended_at + 600,
    )
    conn.execute(
        "UPDATE tasks SET status='review', current_run_id=NULL, claim_lock=NULL, "
        "claim_expires=NULL, worker_pid=NULL WHERE id=?",
        (tid,),
    )
    conn.commit()
    return tid


def test_dispatcher_facade_imports_first_without_a_cycle():
    """The facade must import cleanly when NOTHING has imported ``kanban_db`` yet.

    ``kanban_db`` imports names BACK from the dispatcher facade at the bottom of its
    own module, while the facade imports its topical siblings at the bottom of its
    module. A name the facade only defines after those sibling imports therefore
    raises ``ImportError: cannot import name ... from partially initialized module``
    - but only for the import order that starts at the dispatcher, which is the order
    ``gateway/kanban_watchers_dispatcher.py`` uses (``from hermes_cli import
    kanban_db_dispatch`` inside ``_kbd()``). Every other test imports ``kanban_db``
    first and never sees it. A fresh interpreter is the only honest way to pin it.
    """
    repo_root = Path(__file__).resolve().parents[2]
    probe = "import hermes_cli.kanban_db_dispatch as kbd; print(callable(kbd.check_respawn_guard))"
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(repo_root), env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True, text=True, encoding="utf-8", timeout=180, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "True", proc.stdout


def test_respawn_guard_ignores_failure_text_a_later_handoff_superseded(kanban_home, monkeypatch):
    """A stale quota text must not park a card after a newer run handed it on.

    ``last_failure_error`` caches the last run that FAILED; nothing clears it when a
    later run ends in a handoff. A rate-limited requeue stamps a quota-flavored
    message, the ``review_requested`` run supersedes it, and the column keeps
    matching ``_RESPAWN_BLOCKER_RE`` — which parked every review card that had ever
    hit a quota wall before the review-lane exemption could be reached.
    """
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")

    with kbc.connect() as conn:
        review_id = _seed_superseded_blocker_run(
            conn, title="review superseded", latest_outcome="review_requested", latest_status="review",
        )
        ready_id = _seed_superseded_blocker_run(
            conn, title="ready superseded", latest_outcome="changes_requested", latest_status="ready",
        )

        # The review lane spawns its reviewer; the attribution rule is lane-independent.
        assert kbd.check_respawn_guard(conn, review_id, lane="review") is None
        assert kbd.check_respawn_guard(conn, ready_id) is None


def test_respawn_guard_still_parks_a_blocker_on_the_latest_run(kanban_home, monkeypatch):
    """Attribution must not weaken a CURRENT blocker: both lanes stay parked.

    The review-lane exemption deliberately sits AFTER ``blocker_auth``, so a reviewer
    spawn refused on a genuine quota/auth wall still parks instead of burning a
    worker slot every tick.
    """
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")

    with kbc.connect() as conn:
        # Latest run IS the rate-limited requeue the text came from: the cooldown
        # applies in both lanes, so the review exemption cannot outrun it.
        rate_limited_id = kb.create_task(conn, title="rate limited latest", assignee="a")
        kb.claim_task(conn, rate_limited_id)
        _close_run(conn, _latest_run_id(conn, rate_limited_id), outcome="rate_limited",
                   status="rate_limited", ended_at=int(time.time()))
        conn.execute(
            "UPDATE tasks SET status='review', current_run_id=NULL, claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL, last_failure_error=? WHERE id=?",
            ("pid 69679 exited rate-limited (quota wall) — requeued without counting a failure",
             rate_limited_id),
        )

        # Latest run failed on a genuine auth wall: blocker_auth, not the review exemption.
        spawn_failed_id = _seed_superseded_blocker_run(
            conn, title="spawn failed latest", latest_outcome="spawn_failed", latest_status="failed",
        )
        conn.execute(
            "UPDATE tasks SET last_failure_error='provider authentication failed' WHERE id=?",
            (spawn_failed_id,),
        )
        conn.commit()

        assert kbd.check_respawn_guard(conn, rate_limited_id, lane="review") == "rate_limit_cooldown"
        assert kbd.check_respawn_guard(conn, rate_limited_id) == "rate_limit_cooldown"
        assert kbd.check_respawn_guard(conn, spawn_failed_id, lane="review") == "blocker_auth"
        assert kbd.check_respawn_guard(conn, spawn_failed_id) == "blocker_auth"
