"""#103339 item 2: ``hermes doctor --fix`` never checkpoints state.db through a bare writable ``sqlite3.connect``
— the checkpoint runs on the exclusive repair guard, so an opener arriving after the holder scan is refused."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import hermes_state_repair
from hermes_cli.doctor_report import Finding
from hermes_cli.doctor_state import _state_db_wal


def _large_wal_db(tmp_path):
    db = tmp_path / "state.db"
    setup = sqlite3.connect(str(db))
    setup.execute("CREATE TABLE t(x)")
    setup.execute("PRAGMA journal_mode=WAL")
    setup.execute("INSERT INTO t VALUES (1)")
    setup.commit()
    setup.close()
    with open(Path(f"{db}-wal"), "ab") as handle:
        handle.truncate(51 * 1024 * 1024)
    return db


def test_doctor_checkpoint_runs_only_on_the_exclusive_repair_guard(tmp_path, monkeypatch):
    db = _large_wal_db(tmp_path)

    bare_connects: list[str] = []
    real_connect = sqlite3.connect

    def _spy(database, *args, **kwargs):
        if not str(database).startswith("file:") or "mode=ro" not in str(database):
            bare_connects.append(str(database))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", _spy)
    guard_connects: list[Path] = []
    real_durable = hermes_state_repair._connect_repair_durable

    def _durable(path, **kwargs):
        guard_connects.append(Path(path))
        return real_durable(path, **kwargs)

    monkeypatch.setattr(hermes_state_repair, "_connect_repair_durable", _durable)

    finding = Finding()
    _state_db_wal(finding, True, db)

    assert finding.fixed == 1 and not finding.issues
    # Every writable open went through the repair connector (probe + exclusive guard); none was a bare connect.
    assert bare_connects and len(bare_connects) == len(guard_connects) >= 2


def test_session_count_reads_a_home_with_uri_reserved_characters(tmp_path):
    """`file:` URIs treat '?' and '#' as delimiters; a home named `profile?blue` must still count."""
    import sqlite3

    from hermes_cli.doctor_state import _session_count

    home = tmp_path / "profile?blue#x"
    home.mkdir()
    db = home / "state.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE sessions(id TEXT)")
    conn.execute("INSERT INTO sessions VALUES ('a'), ('b')")
    conn.commit()
    conn.close()
    assert _session_count(db) == 2


def test_large_wal_warning_under_a_live_writer_never_suggests_a_bare_fix(tmp_path, monkeypatch, capsys):
    """`hermes doctor` (no --fix) on a large WAL while Desktop/gateway hold the DB must say it is normal and
    order "stop" before any `--fix` — the bare "run 'hermes doctor --fix'" nudge is how users became the
    second writer (#110054)."""
    import hermes_state_holders

    monkeypatch.setattr(hermes_state_holders, "live_writer_holds_db", lambda *a, **k: True)
    finding = Finding()
    _state_db_wal(finding, False, _large_wal_db(tmp_path))
    assert len(finding.issues) == 1 and not finding.fixed
    assert "normal while Desktop or the gateway is running, or state.db cannot be inspected" in capsys.readouterr().out
    assert finding.issues[0].index("stop the profile's gateway") < finding.issues[0].index("hermes doctor --fix")


def test_large_wal_warning_without_a_holder_still_orders_stop_before_fix(tmp_path, monkeypatch):
    import hermes_state_holders

    monkeypatch.setattr(hermes_state_holders, "live_writer_holds_db", lambda *a, **k: False)
    finding = Finding()
    _state_db_wal(finding, False, _large_wal_db(tmp_path))
    assert len(finding.issues) == 1 and not finding.fixed
    assert finding.issues[0].index("stop the profile's gateway") < finding.issues[0].index("hermes doctor --fix")
