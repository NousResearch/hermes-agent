"""Atomic, owner-only publication of backup archives."""
import os
import shutil
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

from hermes_cli.backup_staging import staged_backup_file
from utils import fsync_directory


def _fsync_file(path: Path) -> None:
    """Flush *path*'s data to stable storage (O_RDWR: Windows ``_commit`` rejects a read-only fd)."""
    fd = os.open(path, os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sync_tree(root: Path) -> None:
    """Flush every regular file under *root*, then every directory including *root*."""
    entries = [p for p in root.rglob("*") if not p.is_symlink()]
    for path in entries:
        if path.is_file():
            _fsync_file(path)
    for directory in (*(p for p in entries if p.is_dir()), root):
        fsync_directory(directory)


def _normalize_private_modes(root: Path, staging_dir: Path) -> None:
    """Owner-only on POSIX; on Windows only clear read-only (``copy2`` keeps it), which O_RDWR needs."""
    if os.name == "nt":
        for path in staging_dir.rglob("*"):
            if path.is_file() and not path.is_symlink():
                os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
        return
    os.chmod(root, 0o700)
    os.chmod(staging_dir, 0o700)
    for path in staging_dir.rglob("*"):
        if path.is_dir():
            os.chmod(path, 0o700)
        elif path.is_file():
            os.chmod(path, 0o600)


def publish_private_tree(staging_dir: Path, final_dir: Path) -> None:
    """Make *staging_dir* owner-only and durable, then rename it to *final_dir* and sync the parent.

    The staging directory is private from creation, so copied source modes can be normalized
    before the rename exposes it. A chmod or sync failure is fatal (publishing a readable or
    possibly-zeroed recovery bundle is worse than reporting a failed snapshot); a failed sync
    removes the unpublished staging. The grandparent is synced too: the first snapshot also
    creates ``final_dir.parent``.
    """
    root = final_dir.parent
    _normalize_private_modes(root, staging_dir)
    try:
        _sync_tree(staging_dir)
    except OSError:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    os.replace(staging_dir, final_dir)
    fsync_directory(root)
    fsync_directory(root.parent)


@contextmanager
def _atomic_output_path(final_path: Path, publish_path: Optional[Callable[[], Optional[Path]]] = None):
    """Yield a hidden sibling path and publish it only after a clean close.

    ``publish_path`` picks the destination at publish time (default ``final_path``) so a caller
    can divert an incomplete archive elsewhere without ever touching ``final_path``; returning
    ``None`` discards the partial instead of publishing it.

    The partial is fsynced before the rename and its new directory entry after it. Otherwise
    the rename can reach disk before the data: a crash or power loss soon after "Backup complete"
    leaves a published archive whose tail (the central directory and the last members) reads as
    zeros, and it may already have replaced the previous good backup of the same name.
    """
    with staged_backup_file(final_path.parent, "archive.partial") as partial_path:
        # Archives carry credentials: create the partial owner-only before any byte is written so
        # neither it nor the published file ever has umask-derived group/other bits. O_EXCL refuses
        # a file or symlink planted in the staging directory; the writer reopens and truncates
        # this inode, which keeps its mode.
        os.close(os.open(partial_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        yield partial_path
        destination = publish_path() if publish_path else final_path
        if destination is not None:
            _fsync_file(partial_path)
            os.replace(partial_path, destination)
            fsync_directory(destination.parent)
