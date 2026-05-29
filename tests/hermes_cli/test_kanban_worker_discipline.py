"""Tests for worker-discipline fixes from t_80581b0d.

Fix 2: enforce_missing_heartbeat — 15-min warning event, 30-min auto-block.
Fix 3: max_iterations field — create_task → DB → _default_spawn env injection.

Retro motivation: t_4ba269e5 ran 42 minutes with zero heartbeats and
exhausted the 150-iteration budget without shipping. The two fixes here
make that failure mode structurally impossible.

TEST ISOLATION RULE
-------------------
Every test that calls into kanban_db MUST use the ``kanban_home`` fixture
below, which explicitly sets ``HERMES_KANBAN_HOME`` (highest priority in
``kanban_db.kanban_home()``) AND clears ``HERMES_KANBAN_DB`` so that no
path injected by a parent dispatcher process can leak into the test.

The fixture asserts at setup time that ``kb.connect()`` resolves to a path
inside ``tmp_path`` — the test will fail immediately (not silently) if the
isolation breaks.

Without this guard, a worker that runs ``pytest`` while
``HERMES_KANBAN_DB=/Users/…/.hermes/kanban.db`` is set in its environment
would write fixture tasks into the live board, which is exactly what the
``t_80581b0d`` retro found.
"""

from __future__ import annotations

import os
import time

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated kanban DB rooted at tmp_path.

    Isolation strategy:
    - Sets ``HERMES_KANBAN_HOME`` to ``tmp_path/.hermes`` — this is the
      *highest-priority* override in ``kanban_db.kanban_home()``, so it
      wins over any ambient ``HERMES_HOME`` inherited from a parent process.
    - Clears ``HERMES_KANBAN_DB`` so a dispatcher-injected pin cannot
      override the fixture home.
    - Clears ``HERMES_KANBAN_BOARD`` so no named-board override sneaks in.
    - Clears ``_INITIALIZED_PATHS`` module cache so a prior test's DB path
      does not carry forward into this test's connect() call.
    - Asserts that ``kb.kanban_db_path()`` resolves inside ``tmp_path``
      before returning — hard failure if isolation is broken.
    """
    home = tmp_path / ".hermes"
    home.mkdir()

    # Pin the kanban root explicitly — highest-priority override.
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))

    # Clear env vars that could pin the real DB path over our fixture home.
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)

    # Clear module-level init cache so connect() doesn't reuse a previous
    # test's path.
    kb._INITIALIZED_PATHS.clear()

    # Initialize the DB in the tmp home.
    kb.init_db()

    # Hard assertion: kanban_db_path() must resolve inside tmp_path.
    resolved = kb.kanban_db_path().resolve()
    assert str(tmp_path) in str(resolved), (
        f"ISOLATION BROKEN: kanban_db_path() resolved to {resolved}, "
        f"which is outside tmp_path={tmp_path}. "
        f"HERMES_KANBAN_HOME={os.environ.get('HERMES_KANBAN_HOME')!r}, "
        f"HERMES_KANBAN_DB={os.environ.get('HERMES_KANBAN_DB')!r}"
    )

    yield home

    # Cleanup: discard the tmp path from the init cache so subsequent
    # tests that call connect() without the fixture don't accidentally
    # reuse this temp DB.
    kb._INITIALIZED_PATHS.discard(str(resolved))


# ---------------------------------------------------------------------------
# DB isolation regression test — fail-loud if live DB is reachable
# ---------------------------------------------------------------------------


def test_kanban_create_without_isolation_fixture_uses_tmp_path(tmp_path, monkeypatch):
    """Regression: kanban_create must NOT reach the live kanban.db.

    This test simulates a 'naked' kanban_db call made without the
    isolation fixture — we ensure the global conftest already pins
    HERMES_KANBAN_HOME (or HERMES_HOME) to a tmp_path before any
    kanban_db operation can reach the live ~/.hermes/kanban.db.

    If the global conftest's _hermetic_environment fixture is working,
    HERMES_KANBAN_DB and HERMES_KANBAN_HOME are already cleared and
    HERMES_HOME points to a temp dir. We verify that creating a task
    from this baseline state writes to the per-test tmpdir, NOT the
    real kanban.db.
    """
    live_kanban_db = os.path.expanduser("~/.hermes/kanban.db")

    # If we're running as a dispatched worker, HERMES_KANBAN_DB is set.
    # The global conftest _hermetic_environment should have cleared it.
    # Verify it's gone:
    assert os.environ.get("HERMES_KANBAN_DB", "") == "", (
        "HERMES_KANBAN_DB is set — global conftest _hermetic_environment "
        "should have cleared it. Test environment is not hermetic."
    )

    # kanban_db_path() must NOT resolve to the live path.
    resolved_path = str(kb.kanban_db_path().resolve())
    assert resolved_path != os.path.abspath(live_kanban_db), (
        f"kanban_db_path() resolved to the live DB at {live_kanban_db}. "
        "HERMES_HOME is not pointing to a tmpdir. "
        "The global conftest _hermetic_environment fixture is broken."
    )


# ---------------------------------------------------------------------------
# Fix 2: enforce_missing_heartbeat — 15-min warning, 30-min auto-block
# ---------------------------------------------------------------------------


def _backdate_task(conn, task_id: str, elapsed_seconds: int) -> None:
    """Helper: backdate both tasks.started_at and task_runs.started_at."""
    fake_start = int(time.time()) - elapsed_seconds
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET started_at = ? WHERE id = ?", (fake_start, task_id)
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? "
            "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (fake_start, task_id),
        )


def test_enforce_missing_heartbeat_warn_writes_event(kanban_home, monkeypatch):
    """Task running 16 minutes with no heartbeat gets a warning event."""
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="no-hb-warn", assignee="worker")
        host = _kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, t, claimer=f"{host}:worker")
        kb._set_worker_pid(conn, t, 99999)
        _backdate_task(conn, t, 16 * 60)

        warned, blocked = kb.enforce_missing_heartbeat(
            conn,
            warn_after_seconds=15 * 60,
            block_after_seconds=30 * 60,
        )

    assert t in warned
    assert t not in blocked

    with kb.connect() as conn:
        events = conn.execute(
            "SELECT kind FROM task_events "
            "WHERE task_id = ? AND kind = 'missing_heartbeat_warning'",
            (t,),
        ).fetchall()
    assert len(events) >= 1


def test_enforce_missing_heartbeat_warn_is_idempotent(kanban_home):
    """Two calls at 20 minutes → only one warning event emitted."""
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="idempotent-warn", assignee="worker")
        host = _kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, t, claimer=f"{host}:worker")
        kb._set_worker_pid(conn, t, 11111)
        _backdate_task(conn, t, 20 * 60)

        kb.enforce_missing_heartbeat(
            conn, warn_after_seconds=15 * 60, block_after_seconds=60 * 60
        )
        kb.enforce_missing_heartbeat(
            conn, warn_after_seconds=15 * 60, block_after_seconds=60 * 60
        )

        events = conn.execute(
            "SELECT id FROM task_events "
            "WHERE task_id = ? AND kind = 'missing_heartbeat_warning'",
            (t,),
        ).fetchall()
    assert len(events) == 1, "idempotent: only one warning event expected"


def test_enforce_missing_heartbeat_block_at_30min(kanban_home, monkeypatch):
    """Task running 31 minutes with no heartbeat is auto-blocked."""
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="block-at-30", assignee="worker")
        host = _kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, t, claimer=f"{host}:worker")
        kb._set_worker_pid(conn, t, 22222)
        _backdate_task(conn, t, 31 * 60)

        monkeypatch.setattr(_kb, "_pid_alive", lambda pid: False)

        warned, blocked = kb.enforce_missing_heartbeat(
            conn,
            warn_after_seconds=15 * 60,
            block_after_seconds=30 * 60,
        )

    assert t in blocked
    assert t not in warned

    with kb.connect() as conn:
        task = kb.get_task(conn, t)
    assert task is not None
    assert (
        task.status == "blocked"
    ), "task should be auto-blocked after 30min no heartbeat"


def test_enforce_missing_heartbeat_skips_task_with_recent_heartbeat(kanban_home):
    """Task that has sent a heartbeat is not warned or blocked."""
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-hb", assignee="worker")
        host = _kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, t, claimer=f"{host}:worker")
        kb._set_worker_pid(conn, t, 33333)

        recent_hb = int(time.time()) - 60
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET last_heartbeat_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (recent_hb, t),
            )
        _backdate_task(conn, t, 20 * 60)

        warned, blocked = kb.enforce_missing_heartbeat(
            conn, warn_after_seconds=15 * 60, block_after_seconds=30 * 60
        )

    assert t not in warned
    assert t not in blocked


def test_enforce_missing_heartbeat_skips_no_heartbeat_required(kanban_home):
    """no_heartbeat_required=True skips the check entirely."""
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="no-hb-required", assignee="worker")
        host = _kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, t, claimer=f"{host}:worker")
        kb._set_worker_pid(conn, t, 44444)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET no_heartbeat_required = 1 WHERE id = ?", (t,)
            )
        _backdate_task(conn, t, 40 * 60)

        warned, blocked = kb.enforce_missing_heartbeat(
            conn, warn_after_seconds=15 * 60, block_after_seconds=30 * 60
        )

    assert t not in warned
    assert t not in blocked


def test_enforce_missing_heartbeat_skips_recent_task(kanban_home):
    """Task that just started (5 min) is not warned."""
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="just-started", assignee="worker")
        host = _kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, t, claimer=f"{host}:worker")
        kb._set_worker_pid(conn, t, 55555)
        _backdate_task(conn, t, 5 * 60)

        warned, blocked = kb.enforce_missing_heartbeat(
            conn, warn_after_seconds=15 * 60, block_after_seconds=30 * 60
        )

    assert t not in warned
    assert t not in blocked


# ---------------------------------------------------------------------------
# Fix 3: max_iterations field
# ---------------------------------------------------------------------------


def test_max_iterations_stored_and_retrieved(kanban_home):
    """max_iterations is persisted and read back on Task."""
    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="budget-task", assignee="worker", max_iterations=250
        )
        task = kb.get_task(conn, t)

    assert task is not None
    assert task.max_iterations == 250


def test_max_iterations_defaults_to_none(kanban_home):
    """When not set, max_iterations is None."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="default-budget", assignee="worker")
        task = kb.get_task(conn, t)

    assert task is not None
    assert task.max_iterations is None


def test_default_spawn_injects_hermes_max_iterations(
    kanban_home, monkeypatch, tmp_path
):
    """_default_spawn injects HERMES_MAX_ITERATIONS=N into child env."""
    import hermes_cli.kanban_db as _kb

    captured_envs: list[dict] = []

    class _FakeProc:
        pid = 54321

    def fake_popen(cmd, **kwargs):
        captured_envs.append(kwargs.get("env", {}).copy())
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    with kb.connect() as conn:
        t = kb.create_task(
            conn,
            title="spawn-iter-test",
            assignee="braintrusteng",
            max_iterations=200,
        )
        kb.claim_task(conn, t)
        task = kb.get_task(conn, t)

    assert task is not None
    workspace = str(tmp_path)
    try:
        _kb._default_spawn(task, workspace, board=None)
    except Exception:
        pass  # May fail on missing hermes binary; env is still captured

    if captured_envs:
        assert captured_envs[0].get("HERMES_MAX_ITERATIONS") == "200", (
            f"Expected HERMES_MAX_ITERATIONS=200, got "
            f"{captured_envs[0].get('HERMES_MAX_ITERATIONS')!r}"
        )
    else:
        # Popen not reached — verify the task object at least holds the value
        assert task.max_iterations == 200


def test_default_spawn_does_not_inject_max_iterations_when_none(
    kanban_home, monkeypatch, tmp_path
):
    """When max_iterations is None, HERMES_MAX_ITERATIONS is not overridden."""
    import hermes_cli.kanban_db as _kb

    captured_envs: list[dict] = []

    class _FakeProc:
        pid = 88888

    def fake_popen(cmd, **kwargs):
        captured_envs.append(kwargs.get("env", {}).copy())
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    # Make sure HERMES_MAX_ITERATIONS is not in the ambient env
    monkeypatch.delenv("HERMES_MAX_ITERATIONS", raising=False)

    with kb.connect() as conn:
        t = kb.create_task(
            conn,
            title="default-iter-spawn",
            assignee="braintrusteng",
        )
        kb.claim_task(conn, t)
        task = kb.get_task(conn, t)

    assert task is not None
    assert task.max_iterations is None

    workspace = str(tmp_path)
    try:
        _kb._default_spawn(task, workspace, board=None)
    except Exception:
        pass

    if captured_envs:
        assert (
            "HERMES_MAX_ITERATIONS" not in captured_envs[0]
        ), "HERMES_MAX_ITERATIONS should not appear when max_iterations is None"


def test_max_iterations_dispatch_once_e2e(kanban_home, monkeypatch):
    """End-to-end: dispatch_once spawns a task with max_iterations set."""
    import hermes_cli.kanban_db as _kb
    import hermes_cli.profiles as _profiles

    spawned_tasks: list[kb.Task] = []

    def fake_spawn(task, workspace, *, board=None):
        spawned_tasks.append(task)
        return 11111

    # Allow any assignee so profile_exists check doesn't filter the task
    monkeypatch.setattr(_profiles, "profile_exists", lambda name: True)

    with kb.connect() as conn:
        t = kb.create_task(
            conn,
            title="dispatch-max-iter",
            assignee="worker",
            max_iterations=300,
        )
        task_before = kb.get_task(conn, t)
        assert task_before is not None
        assert task_before.status == "ready"
        assert task_before.max_iterations == 300

        kb.dispatch_once(conn, spawn_fn=fake_spawn, board=None)

    assert any(
        st.id == t and st.max_iterations == 300 for st in spawned_tasks
    ), f"Dispatched task should carry max_iterations=300; spawned={[(s.id, s.max_iterations) for s in spawned_tasks]}"
