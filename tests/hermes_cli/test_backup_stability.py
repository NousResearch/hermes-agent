from __future__ import annotations

import json
import os
import sqlite3
import stat
import zipfile
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


def test_vanished_file_keeps_the_pre_update_archive_complete(tmp_path, monkeypatch, caplog) -> None:
    """A file rotated away between the scan and the write must not demote a complete archive.

    Regression for the pre-update path: one ``cron/output`` file vanished while the archive was
    written, ``errors`` became non-empty, the finished archive was published as
    ``*.incomplete.zip`` and ``hermes update`` printed "Backup skipped" — so a 2 GB backup that
    held ``state.db`` looked like a missing rollback anchor.
    """
    import logging

    from hermes_cli import backup as backup_mod

    caplog.set_level(logging.INFO, logger="hermes_cli.backup")
    home = tmp_path / ".hermes"
    home.mkdir()
    kept = home / "config.yaml"
    kept.write_text("model: {}\n", encoding="utf-8")
    rotated = home / "cron" / "output" / "job" / "2026-11-16_07-48-15.md"
    rotated.parent.mkdir(parents=True)
    rotated.write_text("# run\n", encoding="utf-8")
    publish = tmp_path / "pre-update-2026-11-17-083748.zip"

    def scan(_root, _out_path, *_args, **_kwargs):
        entries = [(kept, Path("config.yaml")),
                   (rotated, Path("cron/output/job/2026-11-16_07-48-15.md"))]
        rotated.unlink()  # the rotation happens after the walk, before the write
        return entries

    monkeypatch.setattr(backup_mod, "_iter_backup_files", scan)

    assert backup_mod._write_full_zip_backup(publish, home) == publish
    assert publish.is_file()
    assert list(tmp_path.glob("*.incomplete.zip")) == []
    with zipfile.ZipFile(publish) as zf:
        assert zf.namelist() == ["config.yaml"]
    assert "vanished=1" in caplog.text, "a skipped entry must still be visible in the log"


def test_unreadable_member_still_publishes_an_incomplete_archive(tmp_path, monkeypatch) -> None:
    """The vanished-file classification must not swallow real gaps.

    A member that still exists but cannot be read keeps the archive out of retention, exactly as
    before: only "the path is gone" is a race, "the path is unreadable" is damage.
    """
    from hermes_cli import backup as backup_mod

    home = tmp_path / ".hermes"
    home.mkdir()
    good = home / "config.yaml"
    good.write_text("model: {}\n", encoding="utf-8")
    broken = home / "notes.md"
    broken.write_text("x", encoding="utf-8")
    real_write = backup_mod._write_zip_file

    def scan(_root, _out_path, *_args, **_kwargs):
        return [(good, Path("config.yaml")), (broken, Path("notes.md"))]

    def write(zf, path, arcname):
        if Path(path).name == "notes.md":
            raise PermissionError(13, "Permission denied")
        return real_write(zf, path, arcname)

    monkeypatch.setattr(backup_mod, "_iter_backup_files", scan)
    monkeypatch.setattr(backup_mod, "_write_zip_file", write)

    publish = tmp_path / "pre-update-2026-11-17-090000.zip"
    assert backup_mod._write_full_zip_backup(publish, home) is None
    assert not publish.exists()
    salvage = tmp_path / "pre-update-2026-11-17-090000.incomplete.zip"
    assert salvage.is_file()
    with zipfile.ZipFile(salvage) as zf:
        assert zf.namelist() == ["config.yaml"]


def test_run_backup_reports_a_vanished_file_and_stays_complete(tmp_path, monkeypatch, capsys) -> None:
    """``hermes backup`` keeps exit status 0 and says which file rotated away."""
    from argparse import Namespace

    from hermes_cli import backup as backup_mod

    home = tmp_path / ".hermes"
    home.mkdir()
    kept = home / "config.yaml"
    kept.write_text("model: {}\n", encoding="utf-8")
    rotated = home / "cron" / "output" / "job" / "2026-11-16_07-48-15.md"
    rotated.parent.mkdir(parents=True)
    rotated.write_text("# run\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    def scan(_root, _out_path, *_args, **_kwargs):
        entries = [(kept, Path("config.yaml")),
                   (rotated, Path("cron/output/job/2026-11-16_07-48-15.md"))]
        rotated.unlink()
        return entries

    monkeypatch.setattr(backup_mod, "_iter_backup_files", scan)
    monkeypatch.setattr(backup_mod, "_collect_external_entries", lambda: ([], set()))

    out = tmp_path / "run-backup.zip"
    assert backup_mod.run_backup(Namespace(output=str(out))) is True
    printed = capsys.readouterr().out
    assert "Backup complete" in printed
    assert "disappeared after the scan" in printed
    assert "2026-11-16_07-48-15.md" in printed
