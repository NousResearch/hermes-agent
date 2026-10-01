"""Shared filesystem primitives for agent-owned package trees.

Descriptor-relative, symlink-refusing helpers used by the responsibility
engine (and formerly the retired skills engine): atomic writes, directory
walks, and safe tree removal, plus the reserved archive directory name.
"""

from __future__ import annotations

import os
import stat
from typing import Sequence
import uuid


WORKSPACE_ARCHIVE_DIRNAME = ".archive"

MAX_SUPPORT_FILE_BYTES = 1024 * 1024


class PackageFilesystemError(ValueError):
    """A package filesystem operation was refused or failed."""



def _read_regular_file_fd(directory_fd: int, name: str, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    file_fd = os.open(name, flags, dir_fd=directory_fd)
    try:
        info = os.fstat(file_fd)
        if not stat.S_ISREG(info.st_mode):
            # Neutral noun: this primitive is shared by the skill and
            # responsibility engines, so "skill file" would misname
            # responsibility-package errors on model-visible paths.
            raise PackageFilesystemError(f"{name} is not a regular file")
        if info.st_size > max_bytes:
            raise PackageFilesystemError(f"{name} exceeds {max_bytes} bytes")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(file_fd, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        result = b"".join(chunks)
        if len(result) > max_bytes:
            raise PackageFilesystemError(f"{name} exceeds {max_bytes} bytes")
        return result
    finally:
        os.close(file_fd)


def _open_directories(root_fd: int, parts: Sequence[str]) -> int:
    current = os.dup(root_fd)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        for part in parts:
            next_fd = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = next_fd
        return current
    except Exception:
        os.close(current)
        raise


def _open_or_create_directories(root_fd: int, parts: Sequence[str]) -> int:
    current = os.dup(root_fd)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        for part in parts:
            try:
                os.mkdir(part, mode=0o755, dir_fd=current)
            except FileExistsError:
                pass
            next_fd = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = next_fd
        return current
    except Exception:
        os.close(current)
        raise


def _atomic_write_fd(directory_fd: int, filename: str, content: str) -> None:
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_SUPPORT_FILE_BYTES:
        raise PackageFilesystemError(f"file exceeds {MAX_SUPPORT_FILE_BYTES} bytes")
    temporary = f".{filename}.{uuid.uuid4().hex}.tmp"
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    except Exception:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except OSError:
            pass
        raise
    finally:
        os.close(fd)
    os.replace(temporary, filename, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    os.fsync(directory_fd)


def _remove_tree_contents(directory_fd: int) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    for entry in list(os.scandir(directory_fd)):
        info = entry.stat(follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            child = os.open(entry.name, flags, dir_fd=directory_fd)
            try:
                _remove_tree_contents(child)
            finally:
                os.close(child)
            os.rmdir(entry.name, dir_fd=directory_fd)
        else:
            os.unlink(entry.name, dir_fd=directory_fd)
    os.fsync(directory_fd)

def _open_or_create_workspace_archive_fd(root_fd: int) -> int:
    """Open the shared hidden archive namespace without following symlinks."""

    created = False
    try:
        os.mkdir(WORKSPACE_ARCHIVE_DIRNAME, mode=0o700, dir_fd=root_fd)
        created = True
    except FileExistsError:
        pass
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    archive_fd = os.open(WORKSPACE_ARCHIVE_DIRNAME, flags, dir_fd=root_fd)
    try:
        if created:
            os.fsync(root_fd)
    except OSError:
        os.close(archive_fd)
        raise
    return archive_fd


def ensure_workspace_archive(root) -> None:
    """Create and verify a root's recoverable archive directory."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(os.fspath(root), flags)
    archive_fd = -1
    try:
        archive_fd = _open_or_create_workspace_archive_fd(root_fd)
    finally:
        if archive_fd >= 0:
            os.close(archive_fd)
        os.close(root_fd)
