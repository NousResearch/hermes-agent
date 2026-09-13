"""Pin destination directories so publication never re-resolves their parents."""

from __future__ import annotations

import os
from pathlib import Path
import stat


def _component(name: str) -> None:
    if not name or name in (".", "..") or "/" in name or "\0" in name:
        raise OSError("Distribution destination requires one path component")


class _PosixDirectory:
    def __init__(self, path: Path, fd: int, parent=None, name: str = ""):
        self.path = path
        self.fd = fd
        info = os.fstat(fd)
        self.identity = info.st_dev, info.st_ino
        self._parent = parent
        self._name = name
        self._children = {}
        self._closed = False
        self._owner = None

    def stat(self, name: str):
        _component(name)
        return os.stat(name, dir_fd=self.fd, follow_symlinks=False)

    def child(self, name: str):
        _component(name)
        cached = self._children.get(name)
        if cached is not None and not cached._closed:
            return cached
        before = self.stat(name)
        if not stat.S_ISDIR(before.st_mode):
            raise OSError(f"Distribution destination parent is not a plain directory: {self.path / name}")
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.fd)
        try:
            child = _PosixDirectory(self.path / name, fd, self, name)
            if child.identity != (before.st_dev, before.st_ino):
                raise OSError(f"Distribution destination parent changed: {child.path}")
        except BaseException:
            os.close(fd)
            raise
        self._children[name] = child
        return child

    def verify(self) -> None:
        if self._parent is None:
            current = os.lstat(self.path)
        else:
            self._parent.verify()
            current = self._parent.stat(self._name)
        held = os.fstat(self.fd)
        if (
            self.identity != (held.st_dev, held.st_ino)
            or self.identity != (current.st_dev, current.st_ino)
            or not stat.S_ISDIR(current.st_mode)
        ):
            raise OSError(f"Distribution destination parent changed: {self.path}")

    def mkdir(self, name: str, mode: int = 0o777) -> None:
        _component(name)
        os.mkdir(name, mode, dir_fd=self.fd)

    def replace(self, name: str, destination, destination_name: str) -> None:
        _component(name)
        _component(destination_name)
        os.replace(name, destination_name, src_dir_fd=self.fd, dst_dir_fd=destination.fd)

    def rmdir(self, name: str) -> None:
        _component(name)
        os.rmdir(name, dir_fd=self.fd)

    def remove_tree(self, name: str) -> None:
        info = self.stat(name)
        if not stat.S_ISDIR(info.st_mode):
            os.unlink(name, dir_fd=self.fd)
            return
        child = self.child(name)
        # Only private scratch/backup trees use this removal path.
        os.fchmod(child.fd, stat.S_IMODE(info.st_mode) | stat.S_IRWXU)
        try:
            for entry in os.listdir(child.fd):
                child.remove_tree(entry)
        finally:
            child.close()
        self.rmdir(name)

    def close(self) -> None:
        if self._closed:
            return
        if self._owner is not None:
            owner, self._owner = self._owner, None
            owner.close()
            return
        for child in reversed(tuple(self._children.values())):
            child.close()
        os.close(self.fd)
        self._closed = True


def open_directory(path: Path):
    if os.name == "nt":
        from hermes_cli.profile_distribution_destination_windows import open_directory as open_native

        return open_native(path)
    path = Path(os.path.abspath(path))
    root = _PosixDirectory(Path(path.anchor), os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW))
    try:
        current = root
        for name in path.parts[1:]:
            current = current.child(name)
        current.verify()
        if current is not root:
            current._owner = root
        return current
    except BaseException:
        root.close()
        raise
