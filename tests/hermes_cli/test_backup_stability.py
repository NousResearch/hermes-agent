from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.backup import (
    BackupInProgressError,
    _atomic_output_path,
    _backup_operation_lock,
    _write_full_zip_backup,
    create_pre_update_backup,
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
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
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


def test_busy_slot_is_distinguishable_from_a_failed_write(tmp_path) -> None:
    """A held lock and an unwritable archive both returned None, so callers
    could not tell "another backup is running" from "the backup is broken".
    ``raise_if_busy`` separates them."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    archive = tmp_path / "automatic.zip"

    with _backup_operation_lock(home):
        # Default: still swallowed, so existing callers are unaffected.
        assert _write_full_zip_backup(archive, home, lock_timeout=0) is None
        assert not archive.exists()

        with pytest.raises(BackupInProgressError):
            _write_full_zip_backup(archive, home, lock_timeout=0, raise_if_busy=True)


def test_pre_update_backup_reports_a_busy_slot_when_asked(tmp_path) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")

    with _backup_operation_lock(home):
        assert create_pre_update_backup(hermes_home=home, lock_timeout=0) is None

        with pytest.raises(BackupInProgressError):
            create_pre_update_backup(
                hermes_home=home, lock_timeout=0, raise_if_busy=True
            )

    # Slot free again: the same call now produces a real archive, which is the
    # point — the skip was never about the backup being broken.
    out = create_pre_update_backup(hermes_home=home, lock_timeout=0)
    assert out is not None and out.exists()


def test_pre_update_backup_waits_for_the_slot(tmp_path, monkeypatch) -> None:
    """The update path must wait for a busy slot rather than lose a 0.25s race
    and skip the only rollback point the update has."""
    import hermes_cli.backup as backup_mod

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")

    waited: list[float] = []
    real_lock = backup_mod._backup_operation_lock

    def recording_lock(hermes_home, timeout_seconds=backup_mod._BACKUP_LOCK_DEFAULT_TIMEOUT):
        waited.append(timeout_seconds)
        return real_lock(hermes_home, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(backup_mod, "_backup_operation_lock", recording_lock)

    assert create_pre_update_backup(hermes_home=home) is not None
    assert waited == [backup_mod._PRE_UPDATE_LOCK_TIMEOUT]
    assert backup_mod._PRE_UPDATE_LOCK_TIMEOUT > backup_mod._BACKUP_LOCK_DEFAULT_TIMEOUT
