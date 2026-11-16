"""Owned, locked staging beside a ZIP, recoverable after an uncatchable process exit.

Names alone cannot establish ownership of old tmp*.db files. Only this protocol's
marked directories are reclaimed, and the kernel lock, not a PID or age estimate,
decides whether another backup still owns one.
"""

from __future__ import annotations

import logging
import os
import re
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Generator

from hermes_cli.termination_guard import unwind_on_termination

logger = logging.getLogger(__name__)
_PREFIX = ".hermes-backup-stage-"
_NAME = re.compile(r"\.hermes-backup-stage-[0-9a-f]{32}\Z")
_LEGACY_NAME = re.compile(r"(?:tmp.+\.db|\..+\.zip\.\d+-\d+\.partial)\Z")
_LEASE = "owner.lock"
_PAYLOADS = frozenset(
    {
        "archive.partial",
        "snapshot.db",
        "snapshot.db-journal",
        "snapshot.db-wal",
        "snapshot.db-shm",
    }
)


def _plain_path(path: Path, *, directory: bool = False) -> bool:
    info = path.lstat()
    if getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        return False
    return (
        stat.S_ISDIR(info.st_mode)
        if directory
        else stat.S_ISREG(info.st_mode) and info.st_nlink == 1
    )


def _marker(stage: Path) -> bytes:
    return f"hermes-backup-stage-v1:{stage.name}\n".encode("ascii")


def is_backup_staging(stage: Path) -> bool:
    """Recognize a marked directory without reading the lock's reserved first byte."""
    if not _NAME.fullmatch(stage.name):
        return False
    try:
        if not _plain_path(stage, directory=True) or not _plain_path(stage / _LEASE):
            return False
        with (stage / _LEASE).open("rb") as lease:
            lease.seek(1)
            return (
                lease.read(128) == _marker(stage) and _stage_entries(stage) is not None
            )
    except OSError:
        return False


def _lock(lease: BinaryIO) -> None:
    lease.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _stage_entries(stage: Path) -> list[Path] | None:
    entries = list(stage.iterdir())
    if any(p.name not in _PAYLOADS | {_LEASE} or not _plain_path(p) for p in entries):
        return None
    return entries


def _remove_payload(stage: Path) -> bool:
    entries = _stage_entries(stage)
    if entries is None:
        logger.warning("Leaving backup staging with unexpected contents: %s", stage)
        return False
    for path in entries:
        if path.name != _LEASE:
            path.unlink()
    return True


def _remove_shell(stage: Path) -> None:
    # The lease must be closed before unlink on Windows. Reapers open it without
    # O_CREAT and never reuse a directory, so no new writer can enter this gap.
    (stage / _LEASE).unlink(missing_ok=True)
    stage.rmdir()


def _reclaim_one(stage: Path) -> None:
    if not is_backup_staging(stage):
        return
    try:
        with (stage / _LEASE).open("r+b") as lease:
            try:
                _lock(lease)
            except OSError:
                return  # Live owner, competing reaper, or uncertain lock support.
            if not is_backup_staging(stage) or not _remove_payload(stage):
                return
        _remove_shell(stage)
    except OSError as exc:
        logger.warning("Could not reclaim backup staging %s: %s", stage, exc)


def reclaim_backup_staging(directory: Path) -> None:
    """Reclaim only inactive, marked siblings. Never recurse into user directories."""
    try:
        if not _plain_path(directory, directory=True):
            return
    except OSError:
        return
    for stage in directory.glob(f"{_PREFIX}*"):
        _reclaim_one(stage)


def warn_legacy_backup_file(path: Path) -> None:
    """Identify possible pre-protocol residue without inferring ownership."""
    if _LEGACY_NAME.fullmatch(path.name):
        logger.warning(
            "Possible leftover from an interrupted backup: %s. "
            "Inspect this file before removing it; its name cannot prove ownership.",
            path,
        )


def prune_backup_staging(directory: Path, names: list[str]) -> list[str]:
    """Prune owned stages from a backup walk, reclaiming inactive ones as found.

    Discovering them during the existing walk also handles a retry that selects
    a different output directory, without a second recursive scan of the home.
    """
    kept = []
    for name in names:
        stage = directory / name
        if is_backup_staging(stage):
            _reclaim_one(stage)
        else:
            kept.append(name)
    return kept


@contextmanager
def staged_backup_file(directory: Path, name: str) -> Generator[Path]:
    """Stage on the destination filesystem, retaining ownership through publication."""
    if name not in {"archive.partial", "snapshot.db"}:
        raise ValueError(f"unsupported backup staging payload: {name}")
    reclaim_backup_staging(directory)
    with unwind_on_termination():
        stage = directory / f"{_PREFIX}{uuid.uuid4().hex}"
        remove_shell = True
        try:
            try:
                stage.mkdir(mode=0o700)
            except OSError:
                remove_shell = False  # Creation failed, so this path is not ours.
                raise
            with (stage / _LEASE).open("x+b") as lease:
                try:
                    lease.write(b" ")
                    lease.flush()
                    try:
                        _lock(lease)
                    except OSError as exc:
                        logger.warning(
                            "Could not lock backup staging %s: %s. "
                            "Continuing without a recovery marker.",
                            stage,
                            exc,
                        )
                    else:
                        lease.seek(1)
                        lease.write(_marker(stage))
                        lease.flush()
                    yield stage / name
                finally:
                    remove_shell = False
                    try:
                        remove_shell = _remove_payload(stage)
                    except OSError as exc:
                        # A Windows SQLite worker may still hold the file during
                        # unwinding. The next run reclaims it after process exit.
                        logger.warning(
                            "Could not clean backup staging %s: %s", stage, exc
                        )
        finally:
            if remove_shell:
                try:
                    _remove_shell(stage)
                except OSError as exc:
                    logger.warning(
                        "Could not remove backup staging directory %s: %s", stage, exc
                    )
