"""Regression: ``hermes doctor --fix`` must not checkpoint the live WAL under a running gateway.

Checkpoint-lock premise (#40177): a bare ``sqlite3.connect`` runs WAL recovery and
``PRAGMA wal_checkpoint(PASSIVE)`` joins the live WAL — that second-writer handling
on a gateway-held state.db is the corruption class #103339 tracks. The check must
skip with an actionable finding while a live writer holds the database.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

from hermes_cli.doctor_report import Finding
from hermes_cli.doctor_state import _state_db_wal


# NOTE: no ``requires_wal`` marker here on purpose. That gate exists for tests
# that depend on Hermes *choosing* WAL mode (declined on vulnerable SQLite
# builds). This test forces WAL explicitly through raw SQL and asserts only on
# the holder-guard skip, so the probe mechanics work on any build.
def test_wal_checkpoint_skipped_while_live_writer_holds_db(tmp_path):
    """A held database skips the checkpoint; nothing is checkpointed or fixed."""
    db = tmp_path / "state.db"
    setup = sqlite3.connect(str(db))
    try:
        setup.execute("CREATE TABLE t(x)")
        setup.execute("PRAGMA journal_mode=WAL")
        setup.execute("INSERT INTO t VALUES (1)")
        setup.commit()
    finally:
        setup.close()
    holder = sqlite3.connect(str(db))
    holder.execute("SELECT count(*) FROM t").fetchone()
    try:
        wal = Path(f"{db}-wal")
        assert wal.exists()
        # Push past the 50 MB fix threshold without 50 MB of real frames: the
        # guard runs before any WAL byte is parsed, so padding is never read.
        with open(wal, "ab") as handle:
            handle.truncate(51 * 1024 * 1024)
        finding = Finding()
        _state_db_wal(finding, True, db)
    finally:
        try:
            holder.close()
        except Exception:
            pass

    assert finding.fixed == 0
    assert any("gateway" in issue for issue in finding.issues)


def _make_large_wal_db(tmp_path):
    db = tmp_path / "state.db"
    setup = sqlite3.connect(str(db))
    try:
        setup.execute("CREATE TABLE t(x)")
        setup.execute("PRAGMA journal_mode=WAL")
        setup.execute("INSERT INTO t VALUES (1)")
        setup.commit()
    finally:
        setup.close()
    wal = Path(f"{db}-wal")
    if not wal.exists():
        # A closed connection checkpoints and may remove the sidecar; recreate it as a plain file — the
        # warn path only stats it.
        wal.touch()
    with open(wal, "ab") as handle:
        handle.truncate(51 * 1024 * 1024)
    return db


def _hold_in_another_process(db):
    """A foreign holder for the scan: another process keeps the database open until killed."""
    code = ("import sqlite3, sys, time\n"
            "c = sqlite3.connect(sys.argv[1])\n"
            "c.execute('SELECT count(*) FROM t').fetchone()\n"
            "print('ready', flush=True)\n"
            "time.sleep(120)\n")
    proc = subprocess.Popen([sys.executable, "-c", code, str(db)], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"
    return proc


def test_large_wal_advice_never_says_bare_fix_while_another_process_holds_db(tmp_path):
    """#110054 ask 2: the warn path used to print "run 'hermes doctor --fix'" with no qualifier; a Desktop
    user who followed it while the gateway held the file walked into the second-writer trap. While
    something holds the database the advice says what has to stop first."""
    db = _make_large_wal_db(tmp_path)
    holder = _hold_in_another_process(db)
    try:
        finding = Finding()
        _state_db_wal(finding, False, db)
    finally:
        holder.kill()
        holder.wait()

    assert finding.fixed == 0
    assert len(finding.issues) == 1
    assert "stop the profile's gateway / Desktop / cron first" in finding.issues[0]
    assert "— run 'hermes doctor --fix' to checkpoint" not in finding.issues[0]


def test_large_wal_advice_suggests_fix_when_nothing_holds_db(tmp_path, monkeypatch):
    """Nobody holds the file and the scan can prove it: the plain suggestion stays as it was."""
    import hermes_state_holders

    db = _make_large_wal_db(tmp_path)
    monkeypatch.setattr(hermes_state_holders, "foreign_state_db_holders", lambda path, **kw: [])
    finding = Finding()
    _state_db_wal(finding, False, db)

    assert finding.issues == ["Large WAL file — run 'hermes doctor --fix' to checkpoint"]


def test_large_wal_advice_fails_closed_when_the_scan_breaks(tmp_path, monkeypatch):
    """A scan failure is "cannot prove quiet", never permission to hand out the bare --fix."""
    import hermes_state_holders

    db = _make_large_wal_db(tmp_path)

    def _boom(path, **kw):
        raise RuntimeError("no /proc here")

    monkeypatch.setattr(hermes_state_holders, "foreign_state_db_holders", _boom)
    finding = Finding()
    _state_db_wal(finding, False, db)

    assert len(finding.issues) == 1
    assert "stop the profile's gateway / Desktop / cron first" in finding.issues[0]
