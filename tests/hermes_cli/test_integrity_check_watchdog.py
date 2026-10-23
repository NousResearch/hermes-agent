"""The post-update integrity guard must survive slow I/O, not a stalled scan."""

import sqlite3

import hermes_startup_watchdog as watchdog
from hermes_cli.backup import verify_sqlite_integrity


def test_slow_integrity_check_renews_bounded_startup_lease(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE messages (content TEXT)")
        conn.executemany("INSERT INTO messages VALUES (?)", [("message",)] * 5_000)

    now = [0.0]
    monkeypatch.setattr(watchdog.time, "monotonic", lambda: now[0])
    handle = watchdog.StartupWatchdogHandle(timeout_s=300.0, exit_code=75)
    monkeypatch.setattr(watchdog, "_handle", handle)
    monkeypatch.setattr(handle, "_process_cpu_seconds", lambda: 0.0)
    # Stop the synchronous watchdog loop at its next wait, without a real sleep.
    monkeypatch.setattr(handle._disarmed_event, "wait", lambda timeout: True)
    fired = []
    monkeypatch.setattr(handle, "_fire", lambda: fired.append(now[0]))

    report = watchdog.report_startup_progress

    def slow_progress(seconds, phase=""):
        if handle._lease_count:
            now[0] += 120.0
            handle._run()
            assert not fired, "slow SQLite progress must outlive the boot deadline"
        report(seconds, phase=phase)

    monkeypatch.setattr(watchdog, "report_startup_progress", slow_progress)
    connect = sqlite3.connect

    class SlowConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "PRAGMA integrity_check":
                now[0] += 400.0
                handle._run()
                assert not fired, "the entry lease must cover slow initial disk reads"
            return super().execute(sql, *args, **kwargs)

    monkeypatch.setattr(
        sqlite3, "connect", lambda *args, **kwargs: connect(*args, factory=SlowConnection, **kwargs)
    )
    assert verify_sqlite_integrity(path)["valid"]
    assert handle._lease_count > 1
    assert handle._lease_phase == "state_db_integrity_check"
    assert not fired

    # No more SQLite callbacks: even this lease must eventually expire.
    now[0] = handle._lease_until + 1.0
    handle._run()
    assert fired == [now[0]]
