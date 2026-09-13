"""Read a distribution source while native Windows handles pin its ancestry."""

from __future__ import annotations

import ntpath
import os
import stat
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path


def _normal_path(value: str) -> str:
    value = value.replace("/", "\\")
    if value[:8].lower() == "\\\\?\\unc\\":
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    value = ntpath.normcase(ntpath.normpath(value))
    drive, tail = ntpath.splitdrive(value)
    # UNC share roots denote the same directory with or without the final slash.
    # Keep drive-root slashes: C: and C:\\ have different meanings.
    return drive if drive.startswith("\\\\") and tail == "\\" else value


def _checked_stat(path: Path):
    info = os.lstat(path)
    if info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise OSError(f"Distribution source contains a symlink or reparse point: {path}")
    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
        raise OSError(f"Distribution source is not a regular file or directory: {path}")
    return info


def _identity(info) -> tuple:
    return info.st_dev, info.st_ino, info.st_mode, info.st_file_attributes


def _contents(info) -> tuple:
    return _identity(info), info.st_size, info.st_mtime_ns


class _Source:
    def __init__(self, path: Path, handle, expected, win32file):
        self._path = path
        self._handle = handle
        self._expected = expected
        self._win32file = win32file
        self._entries = None
        self._handle_contents = None
        self.is_dir = stat.S_ISDIR(expected.st_mode)
        self.mode = expected.st_mode
        self._verify()

    def _verify(self) -> None:
        win32file = self._win32file
        if win32file.GetFileType(self._handle) != win32file.FILE_TYPE_DISK:
            raise OSError(f"Distribution source is not a disk file: {self._path}")
        info = win32file.GetFileInformationByHandle(self._handle)
        if info[0] & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise OSError(f"Distribution source became a symlink or reparse point: {self._path}")
        identity = info[4], (info[8] << 32) | info[9]
        if identity != (self._expected.st_dev, self._expected.st_ino):
            raise OSError(f"Distribution source identity changed: {self._path}")
        if bool(info[0] & stat.FILE_ATTRIBUTE_DIRECTORY) != self.is_dir:
            raise OSError(f"Distribution source type changed: {self._path}")
        contents = info[3], (info[5] << 32) | info[6]
        if self._handle_contents is None:
            self._handle_contents = contents
        if not self.is_dir and (
            contents != self._handle_contents or contents[1] != self._expected.st_size
        ):
            raise OSError(f"Distribution source handle changed while reading: {self._path}")
        actual = win32file.GetFinalPathNameByHandle(self._handle, 0)
        expected_path = _normal_path(str(self._path))
        if _normal_path(actual) != expected_path:
            import win32api

            # TEMP can contain 8.3 aliases, while the handle reports long names.
            # Expand only after validating the pinned handle's identity and type.
            expected_path = _normal_path(win32api.GetLongPathNameW(str(self._path)))
        if _normal_path(actual) != expected_path:
            raise OSError(
                f"Distribution source handle escaped its path: {self._path} "
                f"(expected {expected_path!r}, handle {actual!r})"
            )
        current = _checked_stat(self._path)
        if _identity(current) != _identity(self._expected):
            raise OSError(f"Distribution source path changed: {self._path}")
        # Ancestors outside the exported tree may have unrelated child activity.
        # Only an enumerated source directory needs content-stability checks.
        if (not self.is_dir or self._entries is not None) and (
            _contents(current) != _contents(self._expected)
        ):
            raise OSError(f"Distribution source changed while reading: {self._path}")

    def _scan(self) -> dict:
        with os.scandir(self._path) as entries:
            names = sorted(entry.name for entry in entries)
        return {name: _checked_stat(self._path / name) for name in names}

    def names(self) -> list[str]:
        if not self.is_dir:
            raise OSError(f"Distribution source is not a directory: {self._path}")
        self._verify()
        entries = self._scan()
        if self._entries is None:
            self._entries = entries
        elif {name: _contents(info) for name, info in entries.items()} != {
            name: _contents(info) for name, info in self._entries.items()
        }:
            raise OSError(f"Distribution source entries changed: {self._path}")
        self._verify()
        return list(entries)

    def read(self, size: int) -> bytes:
        if self.is_dir or size < 0:
            raise OSError("Distribution source read requires a file and a bounded size")
        self._verify()
        error, data = self._win32file.ReadFile(self._handle, size)
        if error:
            raise OSError(error, f"Could not read distribution source: {self._path}")
        self._verify()
        return bytes(data)

    @contextmanager
    def child(self, name: str):
        if not self.is_dir or name in ("", ".", "..") or any(c in name for c in "\\/:\0"):
            raise OSError("Distribution source child must be one path component")
        self._verify()
        if self._entries is not None and name not in self._entries:
            raise OSError(f"Distribution source entry was not in the snapshot: {name}")
        expected = self._entries[name] if self._entries is not None else None
        with _open_one(self._path / name, self._win32file, expected) as source:
            yield source
        self._verify()


@contextmanager
def _open_one(path: Path, win32file, expected=None):
    # OPEN_REPARSE_POINT applies to the final component only. Callers keep every
    # ancestor open without WRITE/DELETE sharing before arriving at this path.
    expected = expected if expected is not None else _checked_stat(path)
    handle = win32file.CreateFile(
        str(path), win32file.GENERIC_READ, win32file.FILE_SHARE_READ, None,
        win32file.OPEN_EXISTING,
        win32file.FILE_FLAG_OPEN_REPARSE_POINT | win32file.FILE_FLAG_BACKUP_SEMANTICS,
        None,
    )
    try:
        source = _Source(path, handle, expected, win32file)
        yield source
        source._verify()
        if source._entries is not None:
            source.names()
    finally:
        win32file.CloseHandle(handle)


def _filesystem_path(path: Path) -> Path:
    raw = str(path)
    if "\0" in raw or raw.startswith("\\\\.\\"):
        raise OSError("Distribution sources cannot use the Windows device namespace")
    if raw[:8].lower() == "\\\\?\\unc\\":
        raw = "\\\\" + raw[8:]
    elif raw.startswith("\\\\?\\"):
        raw = raw[4:]
        if not ntpath.splitdrive(raw)[0].endswith(":"):
            raise OSError("Distribution source must use a DOS drive or UNC filesystem path")
    path = Path(os.path.abspath(raw))
    if not path.anchor or any(":" in part for part in path.parts[1:]):
        raise OSError("Distribution source must use a filesystem path without alternate streams")
    return path


@contextmanager
def open_source(path: Path):
    """Open without following any reparse component, retaining all ancestor handles."""
    if sys.platform != "win32":
        raise OSError("Native Windows distribution sources require Windows")
    import pywintypes
    import win32file

    path = _filesystem_path(path)
    try:
        with ExitStack() as stack:
            source = stack.enter_context(_open_one(Path(path.anchor), win32file))
            for name in path.parts[1:]:
                source = stack.enter_context(source.child(name))
            yield source
    except pywintypes.error as exc:
        raise OSError(exc.winerror, exc.strerror, str(path)) from exc
