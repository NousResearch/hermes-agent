"""Regression tests for the failure_accounting cluster (s4-w1b extraction).

Covers the circuit-breaker bookkeeping moved verbatim from
``hermes_cli.kanban_db`` (cluster c10 / failure_accounting) into
``hermes_cli.failure_accounting``: ``_record_task_failure``,
``_record_spawn_failure`` (back-compat alias), ``_set_worker_pid``,
``_clear_failure_counter``, and the ``_clear_spawn_failures`` alias.
"""

from __future__ import annotations

import pytest

import hermes_cli.kanban_db as kb
from hermes_cli.failure_accounting import (
    _clear_failure_counter,
    _clear_spawn_failures,
    _record_spawn_failure,
    _record_task_failure,
    _set_worker_pid,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(__import__("pathlib").Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Re-export parity
# ---------------------------------------------------------------------------


def test_moved_names_reexported_on_kanban_db_module():
    for name in ("_record_task_failure", "_record_spawn_failure",
                 "_set_worker_pid", "_clear_failure_counter",
                 "_clear_spawn_failures"):
        assert getattr(kb, name) is globals()[name], name


def test_legacy_alias_preserved():
    assert _clear_spawn_failures is _clear_failure_counter
    assert kb._clear_spawn_failures is kb._clear_failure_counter


def test_direct_module_import_works():
    import hermes_cli.failure_accounting as fa
    assert fa._record_task_failure is _record_task_failure


# ---------------------------------------------------------------------------
# _record_task_failure
# ---------------------------------------------------------------------------


def test_record_task_failure_below_limit_does_not_block(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="below")
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (t,))
        conn.commit()
        blocked = _record_task_failure(
            conn, t, error="boom", outcome="crashed",
        )
        assert blocked is False
        row = conn.execute(
            "SELECT consecutive_failures, status FROM tasks WHERE id = ?", (t,),
        ).fetchone()
        assert row["consecutive_failures"] == 1
        assert row["status"] == "ready"


def test_record_task_failure_trips_breaker_at_limit(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="trip")
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (t,))
        conn.commit()
        # First hit: 1 < DEFAULT_FAILURE_LIMIT(2) -> not blocked.
        assert _record_task_failure(conn, t, error="e1", outcome="crashed") is False
        # Second hit: 2 >= 2 -> blocked.
        assert _record_task_failure(conn, t, error="e2", outcome="crashed") is True
        row = conn.execute(
            "SELECT consecutive_failures, status FROM tasks WHERE id = ?", (t,),
        ).fetchone()
        assert row["consecutive_failures"] == 2
        assert row["status"] == "blocked"
        ev = conn.execute(
            "SELECT 1 FROM task_events "
            "WHERE task_id = ? AND kind = 'gave_up' ORDER BY id DESC LIMIT 1",
            (t,),
        ).fetchone()
        assert ev is not None


def test_record_task_failure_respects_caller_failure_limit(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="limit1")
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (t,))
        conn.commit()
        blocked = _record_task_failure(
            conn, t, error="x", outcome="spawn_failed", failure_limit=1,
            release_claim=False, end_run=False,
        )
        assert blocked is True
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (t,),
        ).fetchone()
        assert row["status"] == "blocked"


def test_record_task_failure_missing_task_returns_false(kanban_home):
    with kb.connect() as conn:
        assert _record_task_failure(conn, "nope", error="x", outcome="crashed") is False


def test_record_task_failure_force_trip(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="force")
        blocked = _record_task_failure(
            conn, t, error="x", outcome="crashed", force_trip=True,
        )
        assert blocked is True


# ---------------------------------------------------------------------------
# _record_spawn_failure (back-compat wrapper)
# ---------------------------------------------------------------------------


def test_record_spawn_failure_wraps_record_task_failure(kanban_home, monkeypatch):
    import hermes_cli.failure_accounting as fa
    calls = []

    def _spy(conn, task_id, error, *, outcome, failure_limit=None,
             force_trip=False, release_claim=False, end_run=False,
             event_payload_extra=None):
        calls.append((task_id, error, outcome, release_claim, end_run))
        return False

    monkeypatch.setattr(fa, "_record_task_failure", _spy)
    with kb.connect() as conn:
        t = kb.create_task(conn, title="spawnfail")
        _record_spawn_failure(conn, t, "spawn blew up")
    assert calls == [(t, "spawn blew up", "spawn_failed", True, True)]


# ---------------------------------------------------------------------------
# _set_worker_pid
# ---------------------------------------------------------------------------


def test_set_worker_pid_persists_and_emits_event(kanban_home):
    import json
    with kb.connect() as conn:
        t = kb.create_task(conn, title="pid")
        kb.claim_task(conn, t)
        _set_worker_pid(conn, t, 12345)
        row = conn.execute(
            "SELECT worker_pid FROM tasks WHERE id = ?", (t,),
        ).fetchone()
        assert row["worker_pid"] == 12345
        ev = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'spawned' ORDER BY id DESC LIMIT 1",
            (t,),
        ).fetchone()
        assert ev is not None
        assert json.loads(ev["payload"])["pid"] == 12345


# ---------------------------------------------------------------------------
# _clear_failure_counter
# ---------------------------------------------------------------------------


def test_clear_failure_counter_resets(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="clear")
        _record_task_failure(conn, t, error="e1", outcome="crashed")
        _clear_failure_counter(conn, t)
        row = conn.execute(
            "SELECT consecutive_failures, last_failure_error "
            "FROM tasks WHERE id = ?", (t,),
        ).fetchone()
        assert row["consecutive_failures"] == 0
        assert row["last_failure_error"] is None
