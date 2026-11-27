"""``MaintenanceAdmin`` on a store file that does not exist: the page statistics read as ``None``, the FTS storage notice raises, and neither creates the file."""

from __future__ import annotations

import pytest

from persistence.sqlite.admin import SQLiteStoreAdmin


# ── page statistics ────────────────────────────────────────────────────────


def test_page_statistics_of_a_missing_file_are_none_and_create_nothing(tmp_path):
    path = tmp_path / "absent" / "state.db"
    admin = SQLiteStoreAdmin(path, read_only=True)
    assert (admin.logical_size_bytes(), admin.freelist_ratio()) == (None, None)
    assert not path.exists()


# ── FTS storage notice ─────────────────────────────────────────────────────


def test_fts_storage_notice_of_a_missing_file_raises_and_creates_nothing(tmp_path):
    from persistence import PersistenceError

    path = tmp_path / "absent" / "state.db"
    with pytest.raises(PersistenceError):
        SQLiteStoreAdmin(path).fts_storage_notice()
    assert not path.exists()
