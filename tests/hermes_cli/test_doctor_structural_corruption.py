"""#88587 — doctor must name STRUCTURAL state.db corruption honestly.

The write-health probe's failure used to be reported as "FTS write
corruption" unconditionally, routing operators to `--fix` / `sessions
repair` (FTS rebuilds that cannot repair canonical-table damage) and to
the .malformed-backup beside the DB (a snapshot of the same corrupt
file). The discriminator maps integrity_check damage through
sqlite_master and only keeps the FTS wording when every damaged object
is an FTS shadow.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from hermes_cli.doctor import (
    _integrity_damage_is_non_fts,
    _state_db_has_non_fts_damage,
)


def _master(rows):
    return [(rp, t, name) for rp, t, name in rows]


class TestClassifier:
    def test_canonical_table_tree_is_structural(self):
        # The reporter's exact mapping: tree 5 -> sessions table.
        integrity = [("Tree 5  page 421385: btreeInitPage() returns error code 11",)]
        master = _master([(5, "table", "sessions")])
        assert _integrity_damage_is_non_fts(integrity, master) is True

    def test_canonical_index_tree_is_structural(self):
        integrity = [("Tree 45 page 7701: btreeInitPage() returns error code 11",)]
        master = _master([(45, "index", "idx_sessions_gateway_peer")])
        assert _integrity_damage_is_non_fts(integrity, master) is True

    def test_fts_shadow_tree_only_is_not_structural(self):
        integrity = [("Tree 12 page 9: btreeInitPage() returns error code 11",)]
        master = _master([
            (12, "table", "messages_fts_data"),
            (3, "table", "sessions"),
        ])
        assert _integrity_damage_is_non_fts(integrity, master) is False

    def test_fts_virtual_table_tree_only_is_not_structural(self):
        integrity = [("Tree 7 page 3: btreeInitPage() returns error code 11",)]
        master = _master([
            (7, "table", "messages_fts_trigram"),
        ])
        assert _integrity_damage_is_non_fts(integrity, master) is False

    def test_missing_rows_from_canonical_index_is_structural(self):
        integrity = [("row 1 missing from index sqlite_autoindex_delivery_obligations_1",)]
        master = _master([])
        assert _integrity_damage_is_non_fts(integrity, master) is True

    def test_missing_rows_from_fts_shadow_is_not_structural(self):
        integrity = [("row 4 missing from index messages_fts_idx",)]
        master = _master([])
        assert _integrity_damage_is_non_fts(integrity, master) is False

    def test_mixed_damage_is_structural(self):
        integrity = [
            ("Tree 12 page 9: btreeInitPage() returns error code 11",),
            ("Tree 5  page 421385: btreeInitPage() returns error code 11",),
        ]
        master = _master([
            (12, "table", "messages_fts_data"),
            (5, "table", "sessions"),
        ])
        assert _integrity_damage_is_non_fts(integrity, master) is True

    def test_unparseable_line_keeps_fts_wording(self):
        # Fail-closed toward the pre-existing reporting: an integrity line
        # we cannot map is NOT classified as structural.
        integrity = [("Page 99 is never used",)]
        assert _integrity_damage_is_non_fts(integrity, _master([])) is False

    def test_unknown_rootpage_keeps_fts_wording(self):
        integrity = [("Tree 999 page 1: btreeInitPage() returns error code 11",)]
        assert _integrity_damage_is_non_fts(integrity, _master([])) is False


class TestProbeAgainstRealDb:
    def test_healthy_db_is_not_structural(self, tmp_path):
        db = tmp_path / "state.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, k TEXT)")
        conn.execute("CREATE TABLE messages_fts_data (a)")
        conn.commit()
        conn.close()
        assert _state_db_has_non_fts_damage(db) is False

    def test_missing_file_is_not_structural(self, tmp_path):
        assert _state_db_has_non_fts_damage(tmp_path / "absent.db") is False

    def test_garbage_file_fails_safely(self, tmp_path):
        db = tmp_path / "state.db"
        db.write_bytes(b"this is not a database" * 100)
        # An unreadable file must classify as False so the caller keeps the
        # existing reporting path rather than crashing.
        assert _state_db_has_non_fts_damage(db) is False
