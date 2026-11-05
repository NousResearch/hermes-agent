"""``write_health_reason`` raises the typed store error where the legacy probe raised SQLite's, so the doctor's
malformed-store branch still applies to a held store whose snapshot copy fails."""

from __future__ import annotations

import sqlite3

import pytest

from persistence.sqlite._state.errors import is_malformed_db_error
from persistence import CorruptStoreError
from persistence.sqlite.admin import SQLiteStoreAdmin
from tests.persistence_sqlite.admin.test_doctor_parity import _held, _seed

# The markers the doctor's malformed-store branch tests on the typed error's text.
_MALFORMED_DB_MARKERS = ("malformed database schema", "database disk image is malformed")


def _legacy_write_health_reason(state_db_path, *, should_fix):
    """The legacy doctor's probe of a held store: a read-only snapshot copy, then the probe on the copy."""
    import tempfile

    from persistence.sqlite._state.holders import read_only_db_uri
    from persistence.sqlite._state.repair import _db_opens_cleanly
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = tmp + "/state.db"
        src = sqlite3.connect(read_only_db_uri(state_db_path), uri=True, timeout=1.0)
        try:
            dest = sqlite3.connect(snapshot)
            try:
                src.backup(dest)
            finally:
                dest.close()
        finally:
            src.close()
        return _db_opens_cleanly(snapshot)


def _truncate(path, *, keep: float = 0.5) -> None:
    with open(path, "r+b") as f:
        f.truncate(int(path.stat().st_size * keep) // 4096 * 4096)


def test_a_malformed_held_store_raises_the_typed_corruption(tmp_path):
    legacy, admin = _seed(tmp_path / "legacy" / "state.db"), _seed(tmp_path / "admin" / "state.db")
    with _held(legacy, admin):
        _truncate(legacy)
        _truncate(admin)
        with pytest.raises(sqlite3.DatabaseError) as raw:
            _legacy_write_health_reason(legacy, should_fix=True)
        with pytest.raises(CorruptStoreError) as typed:
            SQLiteStoreAdmin(admin).write_health_reason(should_fix=True)
    assert is_malformed_db_error(raw.value)
    assert str(typed.value) == str(raw.value)
    assert isinstance(typed.value.__cause__, sqlite3.DatabaseError)
    # The doctor takes the schema-repair branch for this error, as it did for the raw one.
    assert any(marker in str(typed.value).lower() for marker in _MALFORMED_DB_MARKERS)
