"""Read local distribution entries through handles that refuse symlinks."""

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import stat


def _snapshot(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@dataclass
class _PosixSource:
    path: Path
    fd: int
    mode: int

    @property
    def is_dir(self) -> bool:
        return stat.S_ISDIR(self.mode)

    def names(self) -> list[str]:
        with os.scandir(self.fd) as entries:
            return [entry.name for entry in entries]

    def read(self, size: int) -> bytes:
        return os.read(self.fd, size)

    def child(self, name: str):
        return _open_posix(self.path / name, parent_fd=self.fd)


@contextmanager
def _open_posix(path: Path, parent_fd: int | None = None):
    address = path.name if parent_fd is not None else path
    before = os.stat(address, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode):
        raise OSError(f"Profile distribution source contains a symlink: {path}")
    if not (stat.S_ISREG(before.st_mode) or stat.S_ISDIR(before.st_mode)):
        raise OSError(f"Profile distribution source is not a regular file or directory: {path}")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if stat.S_ISDIR(before.st_mode):
        flags |= os.O_DIRECTORY
    fd = os.open(address, flags, dir_fd=parent_fd)
    try:
        opened = os.fstat(fd)
        if _snapshot(before) != _snapshot(opened):
            raise OSError(f"Profile distribution source changed before reading: {path}")
        yield _PosixSource(path, fd, opened.st_mode)
        current = os.stat(address, dir_fd=parent_fd, follow_symlinks=False)
        if _snapshot(os.fstat(fd)) != _snapshot(opened) or _snapshot(current) != _snapshot(opened):
            raise OSError(f"Profile distribution source changed while reading: {path}")
    finally:
        os.close(fd)


@contextmanager
def open_source(path: Path):
    if os.name == "nt":
        from hermes_cli.profile_distribution_source_windows import open_source as open_native
    else:
        open_native = _open_posix
    with open_native(path) as source:
        yield source
