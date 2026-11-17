from __future__ import annotations

import json
import os
import sqlite3
import stat
from pathlib import Path

import pytest

from hermes_cli.backup import (
    BackupInProgressError,
    _atomic_output_path,
    _backup_operation_lock,
    _write_full_zip_backup,
    create_quick_snapshot,
    list_quick_snapshots,
)
from tests.fixtures.storage_holder import store_copy_seam


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


@pytest.mark.platforms("posix")
def test_atomic_output_syncs_archive_data_before_publishing(tmp_path, monkeypatch) -> None:
    """A crash after publish must not leave a zero-tailed zip (no EOCD) under the final name."""
    final = tmp_path / "out" / "backup.zip"
    final.parent.mkdir()
    events: list[tuple[str, object]] = []
    real_fsync, real_replace = os.fsync, os.replace

    def spy_fsync(fd):
        events.append(("fsync", os.fstat(fd).st_ino))
        return real_fsync(fd)

    def spy_replace(src, dst):
        events.append(("replace", os.stat(src).st_ino))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(os, "replace", spy_replace)
    with _atomic_output_path(final) as partial:
        partial.write_bytes(b"complete")

    published_inode = final.stat().st_ino
    replace_at = events.index(("replace", published_inode))
    assert ("fsync", published_inode) in events[:replace_at]
    dir_inode = final.parent.stat().st_ino
    assert ("fsync", dir_inode) in events[replace_at:]


def test_atomic_output_publishes_nothing_when_sync_fails(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    final = out_dir / "backup.zip"
    final.write_bytes(b"previous")

    def failing_fsync(fd):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(OSError):
        with _atomic_output_path(final) as partial:
            partial.write_bytes(b"unsynced")

    assert final.read_bytes() == b"previous"
    assert list(out_dir.iterdir()) == [final]


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
    monkeypatch.setattr(*store_copy_seam(), lambda _src, _dst, **_kwargs: False)

    assert _write_full_zip_backup(archive, home) is None
    assert archive.read_bytes() == b"previous-valid-backup"
    assert list(tmp_path.glob(".*.partial")) == []


@pytest.mark.platforms("posix")  # directory fsync is a no-op on Windows
def test_quick_snapshot_files_are_synced_before_publish(tmp_path, monkeypatch) -> None:
    """Update auto-restore trusts a published snapshot: its files must be durable before it appears."""
    home = tmp_path / ".hermes"
    (home / "cron").mkdir(parents=True)
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (home / "cron" / "jobs.json").write_text('{"jobs": []}\n', encoding="utf-8")
    synced: set[int] = set()
    unsynced_at_publish: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def spy_fsync(fd):
        synced.add(os.fstat(fd).st_ino)
        return real_fsync(fd)

    def spy_replace(src, dst):
        src_path = Path(src)
        if src_path.name.endswith(".partial"):
            for path in (src_path, *src_path.rglob("*")):
                if path.stat().st_ino not in synced:
                    unsynced_at_publish.append(path.relative_to(src_path).as_posix() or ".")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(os, "replace", spy_replace)
    snapshot_id = create_quick_snapshot(hermes_home=home)

    assert snapshot_id is not None
    assert unsynced_at_publish == []
    assert (home / "state-snapshots").stat().st_ino in synced


def test_quick_snapshot_is_not_published_when_sync_fails(tmp_path, monkeypatch) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")

    def failing_fsync(fd):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(OSError):
        create_quick_snapshot(hermes_home=home)

    assert list_quick_snapshots(hermes_home=home) == []
    assert list((home / "state-snapshots").iterdir()) == []


@pytest.mark.platforms("windows")  # copy2 keeps the read-only attribute; the sync opens O_RDWR
def test_quick_snapshot_captures_read_only_source_files(tmp_path) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text("model: {}\n", encoding="utf-8")
    os.chmod(config, stat.S_IREAD)
    try:
        snapshot_id = create_quick_snapshot(hermes_home=home)
    finally:
        os.chmod(config, stat.S_IREAD | stat.S_IWRITE)

    assert snapshot_id is not None
    copied = home / "state-snapshots" / snapshot_id / "config.yaml"
    assert copied.read_text(encoding="utf-8") == "model: {}\n"
