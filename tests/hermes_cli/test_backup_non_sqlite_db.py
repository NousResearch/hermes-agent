"""Regression tests for full-zip backups containing non-SQLite ``.db`` files (#75724).

Before the fix, ``_write_full_zip_backup_locked`` treated every ``.db`` file as SQLite;
a file with a ``.db`` suffix but no SQLite magic header failed the safe-copy snapshot
and aborted the entire archive (``_SQLiteSnapshotError``), so pre-update backups
silently produced nothing.
"""

import sqlite3
import tempfile
import zipfile
from pathlib import Path

import hermes_cli.backup as backup_mod


def _real_sqlite_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()


def _read_zip_member(zf_path: Path, name: str) -> bytes:
    with zipfile.ZipFile(zf_path) as zf:
        return zf.read(name)


def test_non_sqlite_db_is_included_as_plain_file(tmp_path):
    """A ``.db`` without the SQLite magic header no longer aborts the backup (#75724)."""
    root = tmp_path / ".hermes"
    root.mkdir()
    payload = b"not a sqlite database\x00\xff\xfe\xfd" * 64
    (root / "notes.db").write_bytes(payload)
    (root / "config.yaml").write_text("x: 1\n")
    out_path = tmp_path / "pre-update-test.zip"

    result = backup_mod._write_full_zip_backup_locked(out_path, root)

    assert result == out_path
    assert _read_zip_member(out_path, "notes.db") == payload


def test_real_sqlite_db_still_gets_wal_safe_snapshot(tmp_path):
    """Real SQLite databases keep going through the safe-copy snapshot path."""
    root = tmp_path / ".hermes"
    root.mkdir()
    _real_sqlite_db(root / "state.db")
    out_path = tmp_path / "pre-update-test.zip"

    result = backup_mod._write_full_zip_backup_locked(out_path, root)

    assert result == out_path
    raw = _read_zip_member(out_path, "state.db")
    assert raw.startswith(b"SQLite format 3\x00")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp.write(raw)
        probe = Path(tmp.name)
    try:
        conn = sqlite3.connect(f"file:{probe}?mode=ro", uri=True)
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone() == (1,)
        conn.close()
    finally:
        probe.unlink(missing_ok=True)
