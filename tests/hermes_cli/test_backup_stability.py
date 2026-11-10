from __future__ import annotations

import json
import os
import sqlite3
import stat
import zipfile
from pathlib import Path

import pytest

from hermes_cli import backup as backup_mod
from hermes_cli import backup_zip as backup_zip_mod
from hermes_cli.backup import (
    BackupInProgressError,
    _atomic_output_path,
    _backup_operation_lock,
    _write_full_zip_backup,
    create_quick_snapshot,
    list_quick_snapshots,
)


def test_backup_lock_rejects_a_second_operation(tmp_path) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()

    with _backup_operation_lock(home):
        with pytest.raises(BackupInProgressError):
            with _backup_operation_lock(home, timeout_seconds=0):
                raise AssertionError("second backup unexpectedly acquired the lock")


def test_atomic_output_publishes_only_after_clean_close(tmp_path) -> None:
    final = tmp_path / "backup.zip"
    final.write_bytes(b"previous")

    with _atomic_output_path(final) as partial:
        partial.write_bytes(b"complete")
        assert final.read_bytes() == b"previous"

    assert final.read_bytes() == b"complete"
    assert not partial.exists()


def test_atomic_output_keeps_previous_file_after_failure(tmp_path) -> None:
    final = tmp_path / "backup.zip"
    final.write_bytes(b"previous")

    with pytest.raises(RuntimeError):
        with _atomic_output_path(final) as partial:
            partial.write_bytes(b"incomplete")
            raise RuntimeError("compression failed")

    assert final.read_bytes() == b"previous"
    assert not partial.exists()


def test_quick_snapshot_is_published_with_manifest(tmp_path, monkeypatch) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_bytes(b"model: {}\n")
    published: list[tuple[Path, Path]] = []

    from hermes_cli import backup

    real_replace = backup.os.replace

    def replace(source, destination) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path.parent == home / "state-snapshots":
            assert source_path.name.endswith(".partial")
            assert (source_path / "manifest.json").is_file()
            assert not destination_path.exists()
            published.append((source_path, destination_path))
        real_replace(source, destination)

    monkeypatch.setattr(backup.os, "replace", replace)
    snapshot_id = create_quick_snapshot(hermes_home=home)

    assert snapshot_id is not None
    assert len(published) == 1
    manifest = json.loads(
        (home / "state-snapshots" / snapshot_id / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["id"] == snapshot_id
    assert manifest["files"] == {"config.yaml": 10}


@pytest.mark.platforms("posix")  # POSIX permission bits
def test_quick_snapshot_tree_is_owner_only_under_permissive_umask(tmp_path) -> None:
    """Recovery snapshots must never inherit world-readable default modes.

    A normal 0022 umask creates SQLite databases and JSON files as 0644 and
    directories as 0755.  Quick snapshots contain session state, credentials,
    pairing records, and cron data, so every published file must be 0600 and
    every directory 0700 regardless of the caller's umask or source modes.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    with sqlite3.connect(home / "state.db") as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")

    old_umask = os.umask(0o022)
    try:
        snapshot_id = create_quick_snapshot(hermes_home=home)
    finally:
        os.umask(old_umask)

    assert snapshot_id is not None
    root = home / "state-snapshots"
    snapshot = root / snapshot_id
    directories = [root, snapshot, *(p for p in snapshot.rglob("*") if p.is_dir())]
    files = [p for p in snapshot.rglob("*") if p.is_file()]

    assert directories
    assert files
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in directories)
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in files)


def test_quick_snapshot_listing_ignores_partial_directories(tmp_path) -> None:
    home = tmp_path / ".hermes"
    partial = home / "state-snapshots" / ".unfinished.1.partial"
    partial.mkdir(parents=True)
    (partial / "manifest.json").write_text('{"id":"unfinished"}', encoding="utf-8")

    assert list_quick_snapshots(hermes_home=home) == []


def test_failed_automatic_backup_preserves_previous_archive(tmp_path, monkeypatch) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "state.db").write_bytes(b"not-a-database")
    archive = tmp_path / "automatic.zip"
    archive.write_bytes(b"previous-valid-backup")

    monkeypatch.setattr("hermes_cli.backup._safe_copy_db", lambda _src, _dst: False)

    assert _write_full_zip_backup(archive, home) is None
    assert archive.read_bytes() == b"previous-valid-backup"
    assert list(tmp_path.glob(".*.partial")) == []


def test_automatic_backup_tolerates_a_file_vanishing_between_scan_and_read(
        tmp_path, monkeypatch) -> None:
    """A rotated log that disappears mid-archive must not demote the whole backup.

    The tree is still coherent: everything that exists was archived. Demoting on a single
    FileNotFoundError published a multi-GB salvage and froze the last complete backup.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    for name in ("alpha.txt", "beta.txt", "gamma.txt", "delta.txt"):
        (home / name).write_text(name, encoding="utf-8")
    (home / "rotated.log").write_text("rotated", encoding="utf-8")
    archive = tmp_path / "automatic.zip"

    real_write = backup_zip_mod._write_zip_file

    def vanish_rotated_log(zf, path, arcname):
        if Path(path).name == "rotated.log":
            Path(path).unlink()
            raise FileNotFoundError(2, "No such file or directory", str(path))
        return real_write(zf, path, arcname)

    monkeypatch.setattr(backup_zip_mod, "_write_zip_file", vanish_rotated_log)

    assert backup_mod._write_full_zip_backup(archive, home) == archive
    names = zipfile.ZipFile(archive).namelist()
    assert len([name for name in names if name.endswith(".txt")]) == 4
    assert not (tmp_path / "automatic.incomplete.zip").exists()


def test_automatic_backup_salvages_when_too_many_files_vanish(tmp_path, monkeypatch) -> None:
    """The allowance is bounded: a flood of missing entries still demotes the archive."""
    home = tmp_path / ".hermes"
    home.mkdir()
    for i in range(20):
        (home / f"gone-{i}.txt").write_text("x", encoding="utf-8")
    for i in range(10):
        (home / f"keep-{i}.txt").write_text("y", encoding="utf-8")
    archive = tmp_path / "automatic.zip"

    real_write = backup_zip_mod._write_zip_file

    def vanish_doomed(zf, path, arcname):
        if Path(path).name.startswith("gone-"):
            raise FileNotFoundError(2, "No such file or directory", str(path))
        return real_write(zf, path, arcname)

    monkeypatch.setattr(backup_zip_mod, "_write_zip_file", vanish_doomed)

    assert backup_mod._write_full_zip_backup(archive, home) is None
    assert not archive.exists()
    assert (tmp_path / "automatic.incomplete.zip").exists()
