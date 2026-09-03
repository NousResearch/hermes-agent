"""Secure control-path operations for Task 28 Phase 28A."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any, NoReturn

from htr.bounded_action_strict_json import parse_strict_json_bytes
from htr.state import BoundedActionValidationError

_O_RDONLY = os.O_RDONLY
_O_WRONLY = os.O_WRONLY
_O_DIRECTORY = os.O_DIRECTORY
_O_CREAT = os.O_CREAT
_O_EXCL = os.O_EXCL
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)

CONTROL_DIR_MODE = 0o700
CONTROL_FILE_MODE = 0o600


def _raise_unsafe_path(context: str, exc: BaseException) -> NoReturn:
    raise BoundedActionValidationError(f"unsafe bounded-action control path ({context})") from exc


def open_dir_no_follow(path: Path, *, context: str) -> int:
    try:
        return os.open(str(path), _O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC)
    except OSError as exc:
        _raise_unsafe_path(context, exc)


def openat_dir_no_follow(dir_fd: int, name: str, *, context: str) -> int:
    try:
        return os.open(name, _O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC, dir_fd=dir_fd)
    except OSError as exc:
        _raise_unsafe_path(f"{context}/{name}", exc)


def openat_file_no_follow(
    dir_fd: int,
    name: str,
    flags: int,
    mode: int = 0,
    *,
    context: str,
) -> int:
    try:
        if mode:
            return os.open(name, flags, mode, dir_fd=dir_fd)
        return os.open(name, flags, dir_fd=dir_fd)
    except FileExistsError:
        raise
    except OSError as exc:
        _raise_unsafe_path(f"{context}/{name}", exc)


def mkdirat(dir_fd: int, name: str, mode: int, *, context: str) -> bool:
    try:
        os.mkdir(name, mode, dir_fd=dir_fd)
        return True
    except FileExistsError:
        return False
    except OSError as exc:
        _raise_unsafe_path(f"{context}/{name}", exc)


def fstat_identity(fd: int) -> tuple[int, int]:
    st = os.fstat(fd)
    return st.st_dev, st.st_ino


def stat_entry_identity(dir_fd: int, name: str) -> tuple[int, int]:
    st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    return st.st_dev, st.st_ino


def stat_entry_mode(dir_fd: int, name: str) -> int:
    st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    return st.st_mode


def is_regular_file_mode(st_mode: int) -> bool:
    return stat.S_ISREG(st_mode)


def is_directory_mode(st_mode: int) -> bool:
    return stat.S_ISDIR(st_mode)


def validate_preexisting_control_dir(fd: int, *, context: str) -> None:
    """Shared ``.control`` — mode/type only; no retroactive uid/gid check."""
    st = os.fstat(fd)
    if not stat.S_ISDIR(st.st_mode):
        raise BoundedActionValidationError(f"{context}: not a directory")
    if stat.S_IMODE(st.st_mode) != CONTROL_DIR_MODE:
        raise BoundedActionValidationError(f"{context}: directory mode must be 0700")


def bind_regular_file(dir_fd: int, name: str, *, context: str) -> int:
    """Open a regular file via parent directory fd (O_NOFOLLOW)."""
    flags = _O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC
    file_fd = openat_file_no_follow(dir_fd, name, flags, context=context)
    validate_file_mode_0600_link_count(file_fd, context=f"{context}/{name}")
    return file_fd


def validate_dir_mode_0700(fd: int, *, context: str, require_ownership: bool = True) -> None:
    st = os.fstat(fd)
    if not stat.S_ISDIR(st.st_mode):
        raise BoundedActionValidationError(f"{context}: not a directory")
    if stat.S_IMODE(st.st_mode) != CONTROL_DIR_MODE:
        raise BoundedActionValidationError(f"{context}: directory mode must be 0700")
    if require_ownership:
        validate_new_task28_ownership(fd, context=context)


def read_regular_file_at(dir_fd: int, name: str, *, context: str) -> tuple[dict[str, Any], bytes]:
    """Read and strictly parse a regular JSON record bound via *dir_fd*."""
    file_fd = bind_regular_file(dir_fd, name, context=context)
    try:
        os.lseek(file_fd, 0, os.SEEK_SET)
        return read_json_record_fd(file_fd)
    finally:
        os.close(file_fd)


def validate_new_task28_ownership(fd: int, *, context: str) -> None:
    st = os.fstat(fd)
    if st.st_uid != os.geteuid() or st.st_gid != os.getegid():
        raise BoundedActionValidationError(f"{context}: ownership mismatch")


def validate_file_mode_0600_link_count(fd: int, *, context: str) -> None:
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        raise BoundedActionValidationError(f"{context}: not a regular file")
    if stat.S_IMODE(st.st_mode) != CONTROL_FILE_MODE:
        raise BoundedActionValidationError(f"{context}: file mode must be 0600")
    if st.st_nlink != 1:
        raise BoundedActionValidationError(f"{context}: link count must be 1")


def read_all_bytes_fd(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        part = os.read(fd, 65536)
        if not part:
            break
        chunks.append(part)
    return b"".join(chunks)


def read_json_record_fd(fd: int) -> tuple[dict[str, Any], bytes]:
    raw = read_all_bytes_fd(fd)
    return parse_strict_json_bytes(raw), raw


def list_dir_names(dir_fd: int) -> frozenset[str]:
    return frozenset(os.listdir(dir_fd))


def list_dir_names_sorted(dir_fd: int) -> list[str]:
    return sorted(os.listdir(dir_fd))


def format_mode(st_mode: int) -> str:
    return format(stat.S_IMODE(st_mode), "04o")


def fsync_dir_fd(dir_fd: int) -> None:
    os.fsync(dir_fd)


def fsync_file_fd(file_fd: int) -> None:
    os.fsync(file_fd)


def write_all_fd(file_fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(file_fd, view[offset:])
        if written <= 0:
            raise BoundedActionValidationError("short write")
        offset += written
