"""Owned, locked staging beside a ZIP, recoverable after an uncatchable process exit.

Names alone cannot establish ownership of old tmp*.db files. Only this protocol's
marked directories are reclaimed, and the kernel lock, not a PID or age estimate,
decides whether another backup still owns one. Everything else that looks like
residue is named for the user and left where it is.
"""

from __future__ import annotations

import logging
import os
import re
import stat
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
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
# Stages this process is writing right now. A run that could not lock its stage
# leaves it unmarked, and its own walk must not report that as residue.
_ACTIVE: set[Path] = set()
# Residue named during the current backup, in the order found. One backup starts
# one sweep, which starts one list, so nothing is named twice in a run.
_NAMED: ContextVar[list[Path] | None] = ContextVar("backup_residue", default=None)


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


def _stage_marked(stage: Path) -> bool | None:
    """True for a marked stage, False for one whose run never marked it, None for anything else.

    Only the lease tells a stage from a user folder that shares the name. The
    lock's reserved first byte is never read.
    """
    if not _NAME.fullmatch(stage.name):
        return None
    try:
        if not _plain_path(stage, directory=True) or not _plain_path(stage / _LEASE):
            return None
        if _stage_entries(stage) is None:
            return None
        with (stage / _LEASE).open("rb") as lease:
            lease.seek(1)
            return lease.read(128) == _marker(stage)
    except OSError:
        return None


def is_backup_staging(stage: Path) -> bool:
    """Recognize a marked directory, the only kind that is ever reclaimed."""
    return _stage_marked(stage) is True


def _name_residue(path: Path, message: str) -> None:
    named = _NAMED.get()
    if named is None:
        logger.warning(message, path)
    elif path not in named:
        named.append(path)
        logger.warning(message, path)


def _name_unmarked_stage(stage: Path) -> bool:
    """Name a stage no backup can reclaim; True when *stage* is one, reported or not."""
    if _stage_marked(stage) is not False:
        return False
    if stage not in _ACTIVE:
        _name_residue(
            stage,
            "Possible leftover from an interrupted backup: %s. "
            "This staging folder was never locked, so it cannot be told from one a "
            "running backup still uses. Inspect it before removing it.",
        )
    return True


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


def _plain_directory(directory: Path) -> bool:
    try:
        return _plain_path(directory, directory=True)
    except OSError:
        return False


def reclaim_backup_staging(directory: Path) -> None:
    """Reclaim only inactive, marked siblings. Never recurse into user directories."""
    if _plain_directory(directory):
        for stage in directory.glob(f"{_PREFIX}*"):
            _reclaim_one(stage)


def warn_legacy_backup_file(path: Path) -> None:
    """Identify possible pre-protocol residue without inferring ownership."""
    if _LEGACY_NAME.fullmatch(path.name):
        _name_residue(
            path,
            "Possible leftover from an interrupted backup: %s. "
            "Inspect this file before removing it; its name cannot prove ownership.",
        )


def _sweep_folder(directory: Path) -> None:
    """Reclaim what is provably abandoned in one folder and name the rest. Never recurses."""
    reclaim_backup_staging(directory)
    if not _plain_directory(directory):
        return
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        if _name_unmarked_stage(entry):
            continue
        try:
            if _plain_path(entry):
                warn_legacy_backup_file(entry)
        except OSError:
            continue


def sweep_backup_folders(hermes_root: Path, output_dir: Path, backups_dir: str) -> None:
    """Start a backup: sweep the folders its walk never enters.

    The walk skips every ``backups`` folder, and the output folder may lie outside the
    home altogether, so residue in the default folder, in each named profile's, and
    beside the archive about to be written is otherwise never seen.
    """
    _NAMED.set([])
    folders = [hermes_root / backups_dir, output_dir]
    profiles = hermes_root / "profiles"
    if _plain_directory(profiles):
        try:
            folders += [p / backups_dir for p in sorted(profiles.iterdir()) if _plain_directory(p)]
        except OSError:
            pass
    for folder in dict.fromkeys(folders):
        _sweep_folder(folder)


def print_backup_residue(result: bool) -> bool:
    """Show the terminal what the log already names; hand *result* back unchanged."""
    named = _NAMED.get()
    if named:
        print(
            "\n  Possible leftovers from an interrupted backup. Inspect each before removing it; "
            "a name cannot prove ownership:\n" + "\n".join(f"    {path}" for path in named)
        )
    return result


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
        elif not _name_unmarked_stage(stage):
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
        _ACTIVE.add(stage)
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
            _ACTIVE.discard(stage)
            if remove_shell:
                try:
                    _remove_shell(stage)
                except OSError as exc:
                    logger.warning(
                        "Could not remove backup staging directory %s: %s", stage, exc
                    )
