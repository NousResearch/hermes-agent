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
    assert any("gateway" in issue for issue in finding.issues)
