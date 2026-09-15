"""#103339 item 2: ``hermes doctor --fix`` never checkpoints state.db through a bare writable ``sqlite3.connect``
— the checkpoint runs on the exclusive repair guard, so an opener arriving after the holder scan is refused."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import hermes_state_repair
from hermes_cli.doctor_report import Finding
from hermes_cli.doctor_state import _state_db_wal


def test_doctor_checkpoint_runs_only_on_the_exclusive_repair_guard(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    setup = sqlite3.connect(str(db))
    setup.execute("CREATE TABLE t(x)")
    setup.execute("PRAGMA journal_mode=WAL")
    setup.execute("INSERT INTO t VALUES (1)")
    setup.commit()
    setup.close()
    wal = Path(f"{db}-wal")
    with open(wal, "ab") as handle:
        handle.truncate(51 * 1024 * 1024)

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
