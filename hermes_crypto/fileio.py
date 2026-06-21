"""Atomic, owner-only file writes for the encryption layer.

Mirrors the ``os.open(O_EXCL) + fdopen + fsync + atomic_replace`` sequence
already used by ``hermes_cli/auth.py`` and ``agent/google_oauth.py`` so the
keystore, keyfile, and re-encrypted credential files are never left in a
partially-written state and never briefly world-readable.
"""

from __future__ import annotations

import os
import shutil
import stat
import uuid
from pathlib import Path
from typing import IO, Union

from utils import atomic_replace


def atomic_write_private(path: Union[str, Path], data: bytes, mode: int = 0o600) -> Path:
    """Atomically write *data* to *path* with owner-only permissions.

    The temp file is created with :data:`os.O_EXCL` and *mode* so there is no
    window where the file exists with default (world-readable) permissions.
    Returns the resolved real path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        real_path = atomic_replace(tmp_path, path)
        # Best-effort directory fsync so the rename is durable. Opening a
        # directory fails on Windows — that is fine, the rename still lands.
        try:
            dir_fd = os.open(str(Path(real_path).parent), os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
    try:
        os.chmod(real_path, mode)
    except OSError:
        pass
    return Path(real_path)


def harden_dir(path: Union[str, Path], *, group_readable: bool = False) -> None:
    """Create *path* if needed and chmod it to ``0o700`` (or ``0o770`` when
    ``group_readable=True``). No-op chmod on Windows.

    ``group_readable=True`` is for the managed-mode log/audit directory pattern
    where the live file is chmod 0o660 so an interactive operator in the same
    group can ``tail`` it — see ``log_handler.EncryptingRotatingFileHandler``.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IRWXU | (stat.S_IRWXG if group_readable else 0)
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def open_private_append(
    path: Union[str, Path], *, encoding: str = "utf-8"
) -> "IO[str]":
    """Open *path* in text append mode with owner-only (``0o600``) permissions.

    Drop-in replacement for ``open(path, "a", encoding=...)`` that avoids the
    umask-default world-readable window for newly-created files. If *path*
    already exists, post-chmods to ``0o600`` so a previously looser-permissioned
    file is tightened on open. Mirrors the ``os.open(..., O_APPEND, 0o600)``
    idiom already used by ``audit.py``.

    Stdlib-only — preserves the import-safety invariant (AGENTS.md §3.1).
    """
    path = Path(path)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    handle = os.fdopen(fd, "a", encoding=encoding)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return handle


def atomic_copy(
    src: Union[str, Path], dest: Union[str, Path], *, mode: int = 0o600
) -> Path:
    """Stream-copy *src* to *dest* via a tmp file + atomic ``os.replace``.

    Combines ``shutil.copy2``'s streaming behaviour (no whole-file memory
    load) with ``atomic_write_private``'s atomicity guarantee — a crash
    mid-copy leaves either nothing or the full final file at *dest*, never
    a truncated intermediate. Sets *mode* on the destination so backups /
    re-encrypted credentials are owner-only.

    Closes hardening gaps: ``migrate._backup`` was a non-atomic
    ``shutil.copy2`` where a crash mid-copy left a truncated ``.bak``, and
    ``_encrypt_file`` / ``_rollback`` restore paths used
    ``atomic_write_private(Path(backup).read_bytes())`` which loads the whole
    backup into memory — fine for credentials but bad for any large artifact
    that ended up under the credential dir.

    Returns the resolved real *dest* path.
    """
    src = Path(src)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        shutil.copy2(src, tmp)
        try:
            os.chmod(tmp, mode)
        except OSError:
            pass
        # Best-effort fsync of the copied data so a power loss between the
        # copy and the rename still lands a complete file at *dest*.
        try:
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
        except OSError:
            pass
        real_dest = atomic_replace(tmp, dest)
        try:
            dir_fd = os.open(str(Path(real_dest).parent), os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    try:
        os.chmod(real_dest, mode)
    except OSError:
        pass
    return Path(real_dest)
