"""Retired WAL generations must stay outside portable backups (regression for #139784)."""

import sqlite3
import zipfile


def test_prunes_retired_wal_generation_directories(tmp_path):
    """Prune the recovery directory together with its captured DB and manifest."""
    from hermes_cli.backup import _iter_backup_files

    root = tmp_path / ".hermes"
    root.mkdir()
    retired = root / "state.db.retired-wal-20261011-205050-4478"
    retired.mkdir()
    (retired / "state.db").write_bytes(b"not a live SQLite database")
    (retired / "manifest.json").write_text("{}")
    (root / "notes.txt").write_text("keep")

    skipped: set = set()
    selected = {str(rel) for _, rel in _iter_backup_files(root, tmp_path / "out.zip", skipped)}

    assert selected == {"notes.txt"}
    assert retired.name in skipped


def test_pre_update_backup_skips_retired_wal_generation(tmp_path):
    """An unreadable captured DB must not abort a backup of the live SQLite DB."""
    from hermes_cli.backup import create_pre_update_backup

    root = tmp_path / ".hermes"
    root.mkdir()
    (root / "config.yaml").write_text("model:\n  provider: openrouter\n")
    with sqlite3.connect(root / "state.db") as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO sessions VALUES ('live-session')")
    retired = root / "state.db.retired-wal-20261011-205050-4478"
    retired.mkdir()
    (retired / "state.db").write_bytes(b"not a live SQLite database")
    (retired / "manifest.json").write_text("{}")

    out = create_pre_update_backup(hermes_home=root)

    assert out is not None
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "config.yaml" in names
        assert "state.db" in names
        assert not any(name.startswith(f"{retired.name}/") for name in names)
        restored = tmp_path / "restored.db"
        restored.write_bytes(zf.read("state.db"))
    with sqlite3.connect(restored) as conn:
        assert conn.execute("SELECT id FROM sessions").fetchall() == [("live-session",)]
