"""Tests: ``kanban.failure_limit`` reaches the crash-accounting path.

``dispatch_once(failure_limit=...)`` is threaded into every spawn/seal path, but
``detect_crashed_workers`` -> ``_account_crashes`` -> ``_record_task_failure``
was called without it, so an ordinary worker crash was always tripped at the
hardcoded ``DEFAULT_FAILURE_LIMIT`` no matter what ``config.yaml`` said
(issue #140426). The protocol-violation, terminal-provider and systemic
(>= 3 identical fingerprints in one tick) budgets keep their own thresholds.
"""

from __future__ import annotations

import json

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
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def _crash(conn, tid: str, pid: int, host: str, claimer_tag: str) -> None:
    """Claim ``tid`` with a live-looking claim, point it at a reaped dead pid."""
    kb.claim_task(conn, tid, claimer=f"{host}:{claimer_tag}")
    conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, tid))
    conn.commit()
    # WIFEXITED with code 1: an ordinary worker crash (not a clean exit, which
    # would be a protocol violation with its own bounded budget).
    kbd._record_worker_exit(pid, 1 << 8)


def test_two_ordinary_crashes_stay_under_a_configured_limit(conn, monkeypatch):
    """``failure_limit=5`` must buy five ordinary crashes, not two.

    Before the fix the second crash parked the card ``blocked``/``gave_up``
    with ``effective_limit: 2`` in the event payload.
    """
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    host = _kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="budget", assignee="a")

    for i in range(2):
        _crash(conn, tid, 73000 + i, host, f"w{i}")
        kbd.detect_crashed_workers(conn, failure_limit=5)
        task = kb.get_task(conn, tid)
        assert task.status == "ready", f"crash {i + 1} under a limit of 5 parked the card"
        assert task.consecutive_failures == i + 1

    gave_up = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='gave_up'", (tid,),
    ).fetchone()[0]
    assert gave_up == 0


def test_default_limit_still_trips_at_two_for_a_caller_that_passes_none(conn, monkeypatch):
    """The old behaviour stays the default: no limit threaded = trip at 2."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    host = _kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="default", assignee="a")

    for i in range(2):
        _crash(conn, tid, 73100 + i, host, f"w{i}")
        kbd.detect_crashed_workers(conn)
        if i == 0:
            assert kb.get_task(conn, tid).status == "ready"

    task = kb.get_task(conn, tid)
    assert task.status == "blocked"
    gave_up = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='gave_up'", (tid,),
    ).fetchone()
    assert gave_up is not None
    payload = json.loads(gave_up["payload"])
    assert payload["effective_limit"] == kbd.DEFAULT_FAILURE_LIMIT
    assert payload["limit_source"] == "dispatcher"


def test_systemic_crashes_trip_at_one_even_with_a_wide_limit(conn, monkeypatch):
    """Three identical fingerprints in one tick are still held for an operator.

    A wide ``failure_limit`` must not turn the systemic guard into a retry loop.
    """
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    host = _kb._claimer_id().split(":", 1)[0]
    tids = [kb.create_task(conn, title=f"systemic-{i}", assignee="a") for i in range(3)]

    for i, tid in enumerate(tids):
        _crash(conn, tid, 73200 + i, host, f"w{i}")

    kbd.detect_crashed_workers(conn, failure_limit=5)

    for tid in tids:
        task = kb.get_task(conn, tid)
        assert task.status == "blocked", f"{tid} should be held for an operator"
        assert task.consecutive_failures == 1
