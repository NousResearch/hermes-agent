"""Anchor publication parents with native handles that deny rename and deletion."""

from __future__ import annotations

import os
import stat
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

from hermes_cli.profile_distribution_source_windows import _Source, _checked_stat, _filesystem_path


@contextmanager
def _native_errors(path: Path):
    import pywintypes

    try:
        yield
    except pywintypes.error as exc:
        raise OSError(exc.winerror, exc.strerror, str(path)) from exc


@contextmanager
def _open_directory_handle(path: Path):
    import win32file

    with _native_errors(path):
        expected = _checked_stat(path)
        if not stat.S_ISDIR(expected.st_mode):
            raise OSError(f"Distribution destination parent is not a directory: {path}")
        # Read access makes the no-delete sharing rule protect the directory.
        # Write sharing permits child publication; source readers stay stricter.
        handle = win32file.CreateFile(
            str(path), win32file.GENERIC_READ,
            win32file.FILE_SHARE_READ | win32file.FILE_SHARE_WRITE, None,
            win32file.OPEN_EXISTING,
            win32file.FILE_FLAG_OPEN_REPARSE_POINT | win32file.FILE_FLAG_BACKUP_SEMANTICS, None,
        )
        try:
            source = _Source(path, handle, expected, win32file)
            yield source
            source._verify()
        finally:
            win32file.CloseHandle(handle)


@contextmanager
def _directory_chain(path: Path):
    with ExitStack() as parents:
        source = parents.enter_context(_open_directory_handle(Path(path.anchor)))
        for name in path.parts[1:]:
            source._verify()
            source = parents.enter_context(_open_directory_handle(source._path / name))
        yield source


class DirAnchor:
    def __init__(self, source, manager):
        self.path = source._path
        self.identity = source._expected.st_dev, source._expected.st_ino
        self._source = source
        self._manager = manager
        self._children: dict[str, DirAnchor] = {}
        self._closed = False

    def _entry_path(self, name: str) -> Path:
        if self._closed:
            raise OSError("Distribution destination directory handle is closed")
        if name.rstrip(" .") in ("", ".", "..") or any(char in name for char in "\\/:\0"):
            raise OSError("Distribution destination entry must be one path component")
        return self.path / name

    def child(self, name: str) -> DirAnchor:
        path = self._entry_path(name)
        key = os.path.normcase(name)
        existing = self._children.get(key)
        if existing is not None and not existing._closed:
            return existing
        # child() retains its parent. Never enumerate it: sibling publications may
        # change directory contents while identity and ancestry remain protected.
        with _native_errors(path):
            self.verify()
            manager = _open_directory_handle(path)
            source = manager.__enter__()
            if not source.is_dir:
                manager.__exit__(None, None, None)
                raise OSError(f"Distribution destination parent is not a directory: {path}")
        anchor = DirAnchor(source, manager)
        self._children[key] = anchor
        return anchor

    def stat(self, name: str):
        return self._entry_path(name).lstat()

    def mkdir(self, name: str, mode: int = 0o777) -> None:
        self._entry_path(name).mkdir(mode=mode)

    def replace(self, name: str, destination_anchor: DirAnchor, destination_name: str) -> None:
        # Only parents are pinned. The leaf itself must remain renameable both
        # during publication and when the transaction restores its originals.
        os.replace(self._entry_path(name), destination_anchor._entry_path(destination_name))

    def rmdir(self, name: str) -> None:
        self._entry_path(name).rmdir()

    def _clear_readonly(self, name: str) -> bool:
        import win32file

        path = self._entry_path(name)
        with _native_errors(path):
            # Pin the leaf as well while changing attributes. A replacement link
            # must never redirect this cleanup write outside the private backup.
            handle = win32file.CreateFile(
                str(path), win32file.GENERIC_READ,
                win32file.FILE_SHARE_READ | win32file.FILE_SHARE_WRITE, None,
                win32file.OPEN_EXISTING,
                win32file.FILE_FLAG_OPEN_REPARSE_POINT | win32file.FILE_FLAG_BACKUP_SEMANTICS, None,
            )
            try:
                attributes = win32file.GetFileInformationByHandle(handle)[0]
                if (
                    win32file.GetFileType(handle) != win32file.FILE_TYPE_DISK
                    or attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
                    or not attributes & stat.FILE_ATTRIBUTE_READONLY
                ):
                    return False
                win32file.SetFileAttributes(
                    str(path), (attributes & ~stat.FILE_ATTRIBUTE_READONLY) or stat.FILE_ATTRIBUTE_NORMAL,
                )
                return True
            finally:
                win32file.CloseHandle(handle)

    def _remove_leaf(self, name: str, is_dir: bool) -> None:
        path = self._entry_path(name)
        remove = path.rmdir if is_dir else path.unlink
        try:
            remove()
        except PermissionError:
            if not self._clear_readonly(name):
                raise
            remove()

    def remove_tree(self, name: str) -> None:
        try:
            info = self.stat(name)
        except FileNotFoundError:
            return
        is_dir = stat.S_ISDIR(info.st_mode)
        if is_dir and not info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            child = self.child(name)
            try:
                with os.scandir(child.path) as entries:
                    names = [entry.name for entry in entries]
                for entry_name in names:
                    child.remove_tree(entry_name)
            finally:
                child.close()
        # A junction/symlink is removed as a leaf; its target is never traversed.
        self._remove_leaf(name, is_dir)

    def verify(self) -> None:
        if self._closed:
            raise OSError("Distribution destination directory handle is closed")
        with _native_errors(self.path):
            self._source._verify()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with _native_errors(self.path), ExitStack() as pending:
            pending.callback(self._manager.__exit__, None, None, None)
            for child in self._children.values():
                pending.callback(child.close)


def open_directory(path: Path) -> DirAnchor:
    """Keep every ancestor open without delete sharing until the anchor closes.

    Reuse the source reader's reparse, identity, UNC and short-name handling.
    Directory content checks remain disabled because no names() call is made.
    """
    if sys.platform != "win32":
        raise OSError("Native Windows distribution destinations require Windows")
    manager = _directory_chain(_filesystem_path(path))
    source = manager.__enter__()
    if not source.is_dir:
        manager.__exit__(None, None, None)
        raise OSError(f"Distribution destination parent is not a directory: {path}")
    return DirAnchor(source, manager)
