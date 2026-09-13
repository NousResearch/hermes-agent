from __future__ import annotations

import hermes_cli.kanban_db_boards as _owner_kanban_db_boards
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
import pytest
from hermes_cli import kanban_db as kb
from gateway import kanban_watchers as _kw
from tests.hermes_cli.kanban_scope_support import (
    _stable_module_identity,
    _SYSTEMD_RUN_SHIM,
    _SYSTEMCTL_SHIM,
    _CHILD_WAIT_PROGRAM,
    _STUBBORN_CHILD_WAIT_PROGRAM,
    Shims,
    shims,
    kanban_home,
    conn,
    _make_task,
    _patch_systemd_available,
    _patch_systemd_run_binary,
    _fake_popen_capture,
    _write_kanban_config,
    _capture_worker_argv,
    _assert_plain_argv_shape,
    _scoped_task_row,
    _patch_managed_gateway,
    _fake_refused_launch_popen,
    _refused_launch_setup,
    _spawnable_profile,
    _running_row,
    _deferred_handoff_row,
    _breaker_shaped_row,
    _max_runtime_row,
    _timed_out_payload,
    _load_dashboard_plugin,
    _untracked_running_row,
    _PRE_CHANGE_TASKS_SQL,
    _PRE_CHANGE_TASK_RUNS_SQL,
)


def test_shutdown_cancels_cleanup_thread_and_reports_unstopped(
    shims,
    conn,
    kanban_home,
    monkeypatch,
    caplog,
):
    """Q, wiring half: when the shutdown budget expires, the cleanup
    daemon thread is CANCELLED (stop event + same deadline) instead of
    scanning and stopping scopes after the dispatcher lock is released,
    and the caller's warning names what was left un-stopped."""
    import asyncio
    import logging
    import threading

    import gateway.kanban_watchers as kw
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    _write_kanban_config(
        Path(kanban_home), "  worker_isolation_stop_on_shutdown: true\n"
    )
    _kanban_db_connect._INITIALIZED_PATHS.clear()
    u1 = _kanban_worker_scope._kanban_worker_scope_unit("t_slowstop1", 1)
    u2 = _kanban_worker_scope._kanban_worker_scope_unit("t_slowstop2", 2)
    p1, p2 = shims.sleeper(), shims.sleeper()
    shims.write_unit(u1, [p1])
    shims.write_unit(u2, [p2])
    _scoped_task_row(conn, scope=u1, pid=p1)
    _scoped_task_row(conn, scope=u2, pid=p2)

    monkeypatch.setattr(_shutdown, "_SHUTDOWN_STOP_BASE_SECONDS", 0.1)
    monkeypatch.setattr(
        "tools.process_registry_scope.SCOPE_STOP_VERIFY_BOUND_SECONDS",
        0.1,
    )

    attempts: list[str] = []
    gate = threading.Event()

    def slow_verified_stop(unit, **kwargs):
        attempts.append(unit)
        if unit == u1:
            gate.wait(timeout=15.0)  # slow fake stop: outlives the budget
        return True

    monkeypatch.setattr(
        _kanban_worker_scope, "_stop_kanban_worker_scope", slow_verified_stop
    )

    class Harness(GatewayKanbanWatchersMixin):
        def __init__(self):
            self._running = False
            self._kanban_dispatcher_lock_handle = None

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        # budget = 0.1 base + 2 units x (0.1 bound + 2.0 margin) = 4.3 s.
        # u1's stop blocks past it: the caller times out, cancels the
        # cleanup thread, and reports BOTH units (u1 unconfirmed, u2 not
        # reached) while u1's stop is still blocked.
        asyncio.run(Harness()._kanban_dispatcher_watcher())
        gate.set()  # let u1's stop return so the thread hits the check

    warnings = [r for r in caplog.records if "still stopping" in r.message]
    assert warnings, "expected the leftover-units warning"
    assert u1 in warnings[0].message and u2 in warnings[0].message

    # Bounded wait for the daemon thread to observe the cancellation and
    # stand down BEFORE u2.
    deadline = time.monotonic() + 5.0
    stood_down = []
    while time.monotonic() < deadline:
        stood_down = [r for r in caplog.records if "stood down before" in r.message]
        if stood_down:
            break
        time.sleep(0.05)
    assert stood_down, "cleanup thread never logged its stand-down"
    assert u2 in stood_down[0].message
    assert attempts == [u1]  # u2 was never signalled after cancellation


def test_shutdown_prescan_timeout_reports_incomplete_not_zero(
    shims,
    conn,
    kanban_home,
    monkeypatch,
    caplog,
):
    """Y: when the shutdown pre-scan outlives its base budget the summary
    must say the scan is incomplete — never a confident "0 unit(s) still
    stopping" for boards it never enumerated."""
    import asyncio
    import logging
    import threading

    import gateway.kanban_watchers as kw
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    _write_kanban_config(
        Path(kanban_home), "  worker_isolation_stop_on_shutdown: true\n"
    )
    _kanban_db_connect._INITIALIZED_PATHS.clear()
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_scanx", 1)
    shims.write_unit(unit, [pid])
    _scoped_task_row(conn, scope=unit, pid=pid)

    monkeypatch.setattr(_shutdown, "_SHUTDOWN_STOP_BASE_SECONDS", 0.1)
    monkeypatch.setattr(
        "tools.process_registry_scope.SCOPE_STOP_VERIFY_BOUND_SECONDS",
        0.1,
    )

    gate = threading.Event()
    real_list_boards = _owner_kanban_db_boards.list_boards

    def stalled_list_boards(**kwargs):
        gate.wait(timeout=15.0)  # board listing wedged past any budget
        return real_list_boards(**kwargs)

    monkeypatch.setattr(_owner_kanban_db_boards, "list_boards", stalled_list_boards)

    class Harness(GatewayKanbanWatchersMixin):
        def __init__(self):
            self._running = False
            self._kanban_dispatcher_lock_handle = None

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        asyncio.run(Harness()._kanban_dispatcher_watcher())
        gate.set()  # let the daemon thread finish for teardown

    warnings = [r for r in caplog.records if "still stopping" in r.message]
    assert warnings, "expected the leftover-units warning"
    msg = warnings[0].message
    assert "scan incomplete" in msg
    assert "not enumerated" in msg
    assert "0 unit(s) still stopping" not in msg, (
        "an incomplete scan must not present an enumerated zero"
    )


def test_shutdown_prescan_partial_names_unscanned_boards(
    shims,
    conn,
    kanban_home,
    monkeypatch,
    caplog,
):
    """Y, partial-scan branch: the board listing returned (so the board
    count is known) but a board's connect stalls — the summary names the
    unscanned boards instead of a zero."""
    import asyncio
    import logging
    import threading

    import gateway.kanban_watchers as kw
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    _write_kanban_config(
        Path(kanban_home), "  worker_isolation_stop_on_shutdown: true\n"
    )
    _kanban_db_connect._INITIALIZED_PATHS.clear()
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_scanp", 1)
    shims.write_unit(unit, [pid])
    _scoped_task_row(conn, scope=unit, pid=pid)

    monkeypatch.setattr(_shutdown, "_SHUTDOWN_STOP_BASE_SECONDS", 0.1)
    monkeypatch.setattr(
        "tools.process_registry_scope.SCOPE_STOP_VERIFY_BOUND_SECONDS",
        0.1,
    )

    gate = threading.Event()
    real_connect = _kanban_db_connect.connect

    def stalled_connect(*args, **kwargs):
        gate.wait(timeout=15.0)  # per-board scan wedged past any budget
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(_kanban_db_connect, "connect", stalled_connect)

    class Harness(GatewayKanbanWatchersMixin):
        def __init__(self):
            self._running = False
            self._kanban_dispatcher_lock_handle = None

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        asyncio.run(Harness()._kanban_dispatcher_watcher())
        gate.set()  # let the daemon thread finish for teardown

    warnings = [r for r in caplog.records if "still stopping" in r.message]
    assert warnings, "expected the leftover-units warning"
    msg = warnings[0].message
    assert "scan incomplete" in msg
    assert "unscanned board(s) not enumerated" in msg
    assert "0 unit(s) still stopping" not in msg
    pre = [r for r in caplog.records if "pre-scan incomplete" in r.message]
    assert pre, "expected the pre-scan timeout warning"


from hermes_cli import kanban_boards as _kanban_boards
from hermes_cli import kanban_db_connect as _kanban_db_connect
from hermes_cli import kanban_worker_scope as _kanban_worker_scope

from gateway import kanban_watchers_shutdown as _shutdown
