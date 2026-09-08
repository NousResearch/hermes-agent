import time

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from tests.hermes_cli.test_kanban_core_functionality import kanban_home  # noqa: F401


def test_dispatch_tick_supervisor_restart_is_activity_not_idle(kanban_home, monkeypatch):
    """A supervisor-restart collateral release must surface in DispatchResult
    and count as tick activity — a gateway restart releasing N pre-boot claims
    must not read as an idle tick to on_kanban_dispatch_tick subscribers."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="pre-boot idle check", assignee="worker")
        kb.claim_task(conn, tid)
        kbd._set_worker_pid(conn, tid, 234568)
        monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
        monkeypatch.setattr(
            kbd, "_supervisor_restart_boot_ts", lambda: time.time() + 3600.0
        )

        result = kbd.dispatch_once(conn)

        assert result.supervisor_restarts == [tid]
        assert result.crashed == []
        assert kbd.detect_crashed_workers._last_supervisor_restart == [tid]
    finally:
        conn.close()


def test_tick_activity_fields_include_supervisor_restarts():
    """The tick-activity whitelist must count supervisor-restart releases as
    activity, else a restart-heavy tick reports outcome='idle' to subscribers."""
    assert "supervisor_restarts" in kb._TICK_ACTIVITY_FIELDS
