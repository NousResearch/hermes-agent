"""Zip-member writing shared by backup and import, plus the read-error accounting a full backup uses.

Split out of ``hermes_cli/backup.py`` so the facade can keep shrinking under its code-health cap
while the member-level behaviour stays in one topical place.
"""

import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from hermes_cli.backup_sqlite import _safe_copy_db

# Entries are written after the tree was scanned, so ordinary churn — cron output rotation, cache
# and audio pruning, a gateway unlinking its own temp file — can delete one in between. A bounded
# trickle of that is not damage, and demoting the whole archive for it used to freeze the last
# good backup at an older date while every run published another multi-GB salvage.
VANISH_ALLOWANCE_MIN = 5
VANISH_ALLOWANCE_DIVISOR = 200


def vanish_allowance(total_files: int) -> int:
    """How many entries may vanish between the scan and the read without demoting the archive."""
    return max(VANISH_ALLOWANCE_MIN, total_files // VANISH_ALLOWANCE_DIVISOR)


def record_entry_error(
    errors: list[str], vanished: list[str], rel_path: Path, exc: BaseException) -> None:
    """Sort one read failure: a file that vanished is churn, anything else is damage."""
    if isinstance(exc, FileNotFoundError):
        vanished.append(f"{rel_path}: {exc}")
    else:
        errors.append(f"{rel_path}: {exc}")


def _discard_failed_zip_members(zf: zipfile.ZipFile, filelist_len: int) -> None:
    """Drop the member(s) created by a failed write, both from the central directory and the file.

    ZipFile.write finalizes its destination member while unwinding a source-read
    failure, so the partial bytes can otherwise become a CRC-valid archive member.
    This runs immediately after that failed write, so the dropped bytes are the tail
    of the file: truncate at the first dropped local header and rewind start_dir so
    later members overwrite it. Leaving the bytes (with a valid local header) would
    still expose a ghost member to streaming readers. Rebuilding NameToInfo from the
    surviving file list also restores the previous entry when a duplicate name failed.
    """
    if len(zf.filelist) <= filelist_len:
        return
    offset = zf.filelist[filelist_len].header_offset
    zf.fp.seek(offset)
    zf.fp.truncate()
    zf.start_dir = offset
    del zf.filelist[filelist_len:]
    zf.NameToInfo.clear()
    zf.NameToInfo.update((info.filename, info) for info in zf.filelist)


def _write_zip_file(zf: zipfile.ZipFile, path: Path, arcname: str) -> None:
    """Write one member while keeping a failed partial write out of the central directory."""
    filelist_len = len(zf.filelist)
    try:
        zf.write(path, arcname=arcname)
    except Exception:
        _discard_failed_zip_members(zf, filelist_len)
        raise


def _zip_sqlite_snapshot(zf: zipfile.ZipFile, abs_path: Path, rel_path: Path, out_path: Path) -> Optional[int]:
    """Add a WAL-safe snapshot of *abs_path* to *zf*; return its byte size, or None on failure.

    Staged beside the output zip: /tmp may be a small tmpfs that cannot hold large databases.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False, dir=str(out_path.parent)) as tmp:
        tmp_db = Path(tmp.name)
    try:
        if not _safe_copy_db(abs_path, tmp_db):
            return None
        _write_zip_file(zf, tmp_db, str(rel_path))
        return tmp_db.stat().st_size
    finally:
        tmp_db.unlink(missing_ok=True)


def _write_zip_entries(
    zf: zipfile.ZipFile, files_to_add: list[tuple[Path, Path]], out_path: Path,
    *, on_db_failure, on_error, on_progress, track_bytes: bool) -> int:
    """Add every ``(abs_path, rel_path)`` to *zf*, WAL-safe for ``*.db``; return bytes archived.

    ``on_db_failure(rel_path)`` runs when a SQLite snapshot fails (may raise to abort);
    ``on_error(rel_path, exc)`` records a read failure; ``on_progress(i)`` fires every 500 files;
    ``track_bytes`` stats plain files for the size total.
    """
    total_bytes = 0
    for i, (abs_path, rel_path) in enumerate(files_to_add, 1):
        try:
            if abs_path.suffix == ".db":
                size = _zip_sqlite_snapshot(zf, abs_path, rel_path, out_path)
                if size is None:
                    on_db_failure(rel_path)
                    continue
                total_bytes += size
            else:
                _write_zip_file(zf, abs_path, str(rel_path))
                if track_bytes:
                    total_bytes += abs_path.stat().st_size
        except (PermissionError, OSError, ValueError) as exc:
            on_error(rel_path, exc)
            continue
        if i % 500 == 0:
            on_progress(i)
    return total_bytes
