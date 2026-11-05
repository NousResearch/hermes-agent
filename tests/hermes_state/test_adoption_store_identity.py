"""A replaced donor handle cannot mint authority for the replacement file."""

from hermes_state import SessionDB


def test_stale_donor_is_refused_before_any_target_import(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    profile = home / "profiles" / "science"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    donor, target = SessionDB(home / "state.db"), SessionDB(profile / "state.db")
    try:
        donor.create_session("stale-donor", "tui")
        donor.append_message("stale-donor", "user", "old handle history")
        monkeypatch.setattr(donor, "_db_file_was_replaced", lambda: True)
        refused = target.adopt_session_lineage_from(donor, "stale-donor")
        assert refused.get("adopted") is False
        assert target.get_session("stale-donor") is None
        assert target._read_one("SELECT 1 FROM sqlite_master WHERE name='adoption_transfers'") is None
    finally:
        donor.close()
        target.close()


def test_changed_database_generation_cannot_reuse_an_old_record(tmp_path, monkeypatch):
    import sqlite3
    from contextlib import closing
    home = tmp_path / "hermes"
    profile = home / "profiles" / "science"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    donor_path = home / "state.db"
    donor, target = SessionDB(donor_path), SessionDB(profile / "state.db")
    try:
        donor.create_session("generation-bound", "tui")
        donor.append_message("generation-bound", "user", "original generation")
        assert target.adopt_session_lineage_from(donor, "generation-bound", retire_donor=False)["adopted"]
        generation = donor._read_one("PRAGMA application_id")[0]
        donor.close()
        inode = donor_path.stat().st_ino
        with closing(sqlite3.connect(donor_path)) as replacement:
            replacement.execute(f"PRAGMA application_id={generation % 2147483646 + 1}")
            replacement.commit()
        assert donor_path.stat().st_ino == inode
        donor = SessionDB(donor_path)
        refused = target.adopt_session_lineage_from(donor, "generation-bound")
        assert refused.get("adopted") is False
        assert donor.get_session("generation-bound")["end_reason"] is None
        assert target._read_one("SELECT status FROM adoption_transfers")[0] == "pending"
    finally:
        donor.close()
        target.close()
