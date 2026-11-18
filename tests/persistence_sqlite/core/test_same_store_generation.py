"""``serves_same_store`` on SQLite is holder identity: a handle opened after ``state.db`` was replaced serves the new
generation, and the handle that held the old one does not serve the same store as it."""

import os

from persistence.sqlite._holder import StoreHolder
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder


def _replace_state_db(home, owner):
    donor = StoreHolder(home / "donor" / "state.db")
    donor.set_meta("who", "replacement")
    donor.close()
    owner.state_meta.set("who", "original")
    held = raw_holder(owner)
    # Quiesce the WAL so a new connection reads the replacement's pages, not the old generation's frames.
    with held._lock:
        held._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    os.replace(home / "donor" / "state.db", home / "state.db")


def test_a_handle_opened_after_the_file_was_replaced_serves_another_generation(tmp_path):
    with open_persistence(profile_scope(home=tmp_path), eager=True) as owner:
        _replace_state_db(tmp_path, owner)
        with open_persistence(profile_scope(home=tmp_path), eager=True) as late:
            assert late.state_meta.get("who") == "replacement"
            assert not owner.serves_same_store(late) and not late.serves_same_store(owner)


def test_handles_on_an_unreplaced_file_serve_the_same_store(tmp_path):
    with open_persistence(profile_scope(home=tmp_path), eager=True) as owner:
        owner.state_meta.set("who", "original")
        with open_persistence(profile_scope(home=tmp_path), eager=True) as late:
            assert owner.serves_same_store(late) and late.serves_same_store(owner)
