"""Separate-output recovery preserves complete adoption evidence without rebinding it."""

import sqlite3
from contextlib import closing

from hermes_cli.session_recovery import recover_session_database
from hermes_state import SessionDB
from tests.hermes_state.test_adoption_transfer_records import _seed_compression_lineage


def test_recovery_keeps_the_complete_pending_adoption_journal(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    profile = home / "profiles" / "science"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    donor, target = SessionDB(home / "state.db"), SessionDB(profile / "state.db")
    try:
        _seed_compression_lineage(donor, "recovery-root", "recovery-tip")
        pending = target.adopt_session_lineage_from(donor, "recovery-tip", retire_donor=False)
        assert pending["adopted"]
        original = [tuple(row) for row in target._read_all("SELECT * FROM adoption_transfers ORDER BY member_order")]
        source_path, donor_path = target.db_path, donor.db_path
    finally:
        donor.close()
        target.close()
    output = tmp_path / "recovered.sqlite"
    work = tmp_path / "recovery-work"
    work.mkdir()
    recover_session_database(source_path, output, work_dir=work)
    with closing(sqlite3.connect(output)) as recovered:
        exists = recovered.execute("SELECT 1 FROM sqlite_master WHERE name='adoption_transfers'").fetchone()
        assert exists, "recovery silently dropped the pending transfer's membership authority"
        rows = recovered.execute("SELECT * FROM adoption_transfers ORDER BY member_order").fetchall()
    assert rows == original, "recovery must preserve evidence, not rebind its store identities"
    with closing(sqlite3.connect(donor_path.as_uri() + "?mode=ro", uri=True)) as unchanged:
        rows = unchanged.execute("SELECT id,end_reason FROM sessions ORDER BY id").fetchall()
    assert rows == [("recovery-root", "compression"), ("recovery-tip", None)]
