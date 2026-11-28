"""Regression tests for ZIP timestamp handling in backups."""

import os
import time
import zipfile
from pathlib import Path

from hermes_cli.backup import _write_zip_entries


def test_backup_archives_member_with_pre_1980_mtime(tmp_path: Path) -> None:
    source = tmp_path / "old.txt"
    source.write_text("preserve this member", encoding="utf-8")
    os.utime(source, (1.0, 1.0))
    archive_path = tmp_path / "backup.zip"
    errors: list[str] = []

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        _write_zip_entries(
            archive,
            [(source, Path("old.txt"))],
            archive_path,
            on_db_failure=lambda rel_path: errors.append(f"SQLite: {rel_path}"),
            on_error=lambda rel_path, exc: errors.append(f"{rel_path}: {exc}"),
            on_progress=lambda _count: None,
            track_bytes=False,
        )

    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert "old.txt" in archive.namelist()
    assert not errors


def test_backup_keeps_valid_mtime_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "ordinary.txt"
    source.write_text("ordinary timestamp", encoding="utf-8")
    timestamp = time.mktime((2024, 2, 3, 4, 5, 6, 0, 0, -1))
    os.utime(source, (timestamp, timestamp))
    archive_path = tmp_path / "backup.zip"
    errors: list[str] = []

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        _write_zip_entries(
            archive,
            [(source, Path("ordinary.txt"))],
            archive_path,
            on_db_failure=lambda rel_path: errors.append(f"SQLite: {rel_path}"),
            on_error=lambda rel_path, exc: errors.append(f"{rel_path}: {exc}"),
            on_progress=lambda _count: None,
            track_bytes=False,
        )

    with zipfile.ZipFile(archive_path) as archive:
        assert archive.getinfo("ordinary.txt").date_time == time.localtime(timestamp)[:6]
        assert archive.testzip() is None
    assert not errors
