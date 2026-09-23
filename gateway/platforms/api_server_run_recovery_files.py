"""Handle-bound point-in-time snapshots for recovery-owned local files.

``inspect_file`` validates and hashes one already-open regular file.  The result
is evidence about that object at inspection time, not a lock or a promise that
the pathname stays bound after the function returns.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any

from hermes_platform.host.facts import os_family

_OPEN_REPARSE_POINT = 0x00200000
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_FILE_ATTRIBUTE_DIRECTORY = 0x00000010
_READ_CHUNK = 1024 * 1024


def _validate_request(path: str, expected_size: int) -> Path:
    if not isinstance(path, str) or not path:
        raise ValueError("local file path must be a non-empty string")
    if type(expected_size) is not int or expected_size < 0:
        raise ValueError("expected size must be a non-negative integer")
    target = Path(path)
    if not target.is_absolute() or (os_family() == "win32" and str(target).startswith("\\\\")):
        raise ValueError("file snapshot requires an absolute local path")
    return target


def _path_chain(target: Path) -> tuple[tuple[str, int, int, int, int], ...]:
    """Return non-link identities for every resolved spelling component."""
    entries = []
    for item in (*reversed(target.parents), target):
        item_stat = os.lstat(item)
        attributes = getattr(item_stat, "st_file_attributes", 0)
        if stat.S_ISLNK(item_stat.st_mode) or attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise OSError("local file snapshot rejects links and reparse points")
        entries.append((str(item), item_stat.st_dev, item_stat.st_ino, item_stat.st_ctime_ns, attributes))
    return tuple(entries)


def _snapshot_from_stat(item_stat: os.stat_result, digest: str) -> dict[str, Any]:
    return {
        "sha256": digest,
        "identity": {"device": item_stat.st_dev, "file_id": item_stat.st_ino},
        "size": item_stat.st_size,
        "mtime_ns": item_stat.st_mtime_ns,
        "ctime_ns": item_stat.st_ctime_ns,
    }


def _read_hash_fd(fd: int, expected_size: int) -> str:
    digest = hashlib.sha256()
    total = 0
    while total <= expected_size:
        remaining = expected_size + 1 - total
        data = os.read(fd, min(_READ_CHUNK, remaining))
        if not data:
            break
        digest.update(data)
        total += len(data)
    if total != expected_size:
        raise OSError("local file changed while its snapshot was read")
    return digest.hexdigest()


def _inspect_posix(target: Path, expected_size: int, before: tuple[tuple[str, int, int, int, int], ...]) -> dict[str, Any]:
    # A replaced FIFO must not block admission before fstat can reject it.
    flags = (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0))
    fd = os.open(target, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise OSError("local file snapshot requires a regular file")
        if opened.st_size != expected_size:
            raise ValueError("local file is not the expected regular-file size")
        digest = _read_hash_fd(fd, expected_size)
        # A same-size write can fall in one filesystem timestamp tick. Check
        # content coherence as well as metadata, without an unbounded retry.
        os.lseek(fd, 0, os.SEEK_SET)
        if _read_hash_fd(fd, expected_size) != digest:
            raise OSError("local file content changed during snapshot")
        after = os.fstat(fd)
        # Reading may update atime. Compare identity and mutation metadata at
        # nanosecond precision instead of stat_result's atime/float tuple.
        changed = any(getattr(after, name) != getattr(opened, name) for name in (
            "st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns",
        ))
        if changed or _path_chain(target) != before:
            raise OSError("local file path or object changed during snapshot")
        current = os.lstat(target)
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise OSError("local file path changed during snapshot")
        return _snapshot_from_stat(opened, digest)
    finally:
        os.close(fd)


def _windows_info(handle: Any, win32file: Any) -> tuple[int, int, int, int, int, int, int]:
    info = win32file.GetFileInformationByHandle(handle)
    attributes, created, _accessed, modified, volume, size_high, size_low, _links, index_high, index_low = info
    identity = (volume, index_high, index_low)
    size = (size_high << 32) | size_low
    return identity + (attributes, size, int(created.timestamp() * 1_000_000_000),
                       int(modified.timestamp() * 1_000_000_000))


def _read_hash_handle(handle: Any, win32file: Any, expected_size: int) -> str:
    digest = hashlib.sha256()
    total = 0
    while total <= expected_size:
        remaining = expected_size + 1 - total
        _error, data = win32file.ReadFile(handle, min(_READ_CHUNK, remaining))
        if not data:
            break
        digest.update(data)
        total += len(data)
    if total != expected_size:
        raise OSError("local file changed while its snapshot was read")
    return digest.hexdigest()


def _inspect_windows(target: Path, expected_size: int, before: tuple[tuple[str, int, int, int, int], ...]) -> dict[str, Any]:
    import win32con
    import win32file

    handle = None
    try:
        handle = win32file.CreateFile(
            str(target), win32con.GENERIC_READ,
            win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE | win32con.FILE_SHARE_DELETE,
            None, win32con.OPEN_EXISTING, win32con.FILE_ATTRIBUTE_NORMAL | _OPEN_REPARSE_POINT, None,
        )
        actual = win32file.GetFinalPathNameByHandle(int(handle), 0).removeprefix("\\\\?\\")
        expected = os.path.abspath(str(target))
        if os.path.normcase(actual) != os.path.normcase(expected):
            raise OSError("local file handle escaped its expected path")
        opened = _windows_info(handle, win32file)
        _volume, index_high, index_low, attributes, size, created_ns, modified_ns = opened
        if attributes & (_FILE_ATTRIBUTE_REPARSE_POINT | _FILE_ATTRIBUTE_DIRECTORY):
            raise OSError("local file snapshot requires a non-reparse regular file")
        if size != expected_size:
            raise ValueError("local file is not the expected regular-file size")
        digest = _read_hash_handle(handle, win32file, expected_size)
        win32file.SetFilePointer(int(handle), 0, win32con.FILE_BEGIN)
        if _read_hash_handle(handle, win32file, expected_size) != digest:
            raise OSError("local file content changed during snapshot")
        if _windows_info(handle, win32file) != opened or _path_chain(target) != before:
            raise OSError("local file path or object changed during snapshot")
        current = os.lstat(target)
        if current.st_ino != ((index_high << 32) | index_low):
            raise OSError("local file path changed during snapshot")
        return {
            "sha256": digest,
            "identity": {"device": opened[0], "file_id": current.st_ino},
            "size": size,
            "mtime_ns": modified_ns,
            "ctime_ns": created_ns,
        }
    except ValueError:
        raise
    except OSError:
        raise
    except Exception as exc:
        raise OSError("cannot inspect local file safely") from exc
    finally:
        if handle is not None:
            handle.Close()


def inspect_file(path: str, expected_size: int) -> dict[str, Any]:
    """Return a JSON-safe point-in-time digest and identity for one local file.

    The path and every ancestor must stay physical regular filesystem objects
    from check through hash.  A returned identity binds the digest to the opened
    object only at this instant; callers must re-inspect before later use.
    """
    target = _validate_request(path, expected_size)
    before = _path_chain(target)
    if os_family() == "win32":
        return _inspect_windows(target, expected_size, before)
    return _inspect_posix(target, expected_size, before)
