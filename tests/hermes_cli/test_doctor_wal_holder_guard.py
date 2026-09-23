"""Regression: ``hermes doctor --fix`` must not checkpoint the live WAL under a running gateway.

Checkpoint-lock premise (#40177): a bare ``sqlite3.connect`` runs WAL recovery and
``PRAGMA wal_checkpoint(PASSIVE)`` joins the live WAL — that second-writer handling
on a gateway-held state.db is the corruption class #103339 tracks. The check must
skip with an actionable finding while a live writer holds the database.
"""

from __future__ import annotations

import sqlite3
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
    # A live writer holding the WAL is expected; doctor must not put it on the issue list.
    assert finding.issues == []


def test_doctor_names_retired_wal_holders_instead_of_healthy_state_db(tmp_path, monkeypatch, capsys):
    """After the deleted-WAL guard fires (#110054), doctor must name the PIDs holding the retired
    generation, must not print a healthy state.db line, and must not open the store itself (the
    health probe is another opener) nor checkpoint under --fix."""
    import hermes_cli.doctor as doctor
    import hermes_cli.doctor_state as doctor_state
    import hermes_state_dbfile

    db = tmp_path / "state.db"
    db.write_bytes(b"")
    monkeypatch.setattr(doctor, "HERMES_HOME", tmp_path)
    monkeypatch.setattr(hermes_state_dbfile, "iter_deleted_sqlite_sidecar_holders",
                        lambda path: [(4242, f"{path}-wal"), (4242, f"{path}-shm")])
    probed = []
    monkeypatch.setattr(doctor_state, "_state_db_health", lambda *a, **k: probed.append(a))
    monkeypatch.setattr(doctor_state, "_state_db_stats", lambda *a, **k: probed.append(a))
    monkeypatch.setattr(doctor_state, "_state_db_wal", lambda *a, **k: probed.append(a))

    finding = doctor_state._check_state_db(True)
    out = capsys.readouterr().out

    assert "4242" in out and "retired WAL" in out
    assert "✓" not in out
    assert probed == [] and finding.fixed == 0
    assert any("4242" in issue and "gateway stop" in issue for issue in finding.issues)
