"""Shared exclusion and durable evidence primitives for offline operator entry points."""
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import TextIO


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_directory(path):
    """Persist every newly created directory entry before relying on its name."""
    path = Path(path)
    if path.is_dir():
        return
    ensure_directory(path.parent)
    path.mkdir(mode=0o700)
    sync_directory(path.parent)


def sync_file(path):
    with Path(path).open('rb') as stream:
        os.fsync(stream.fileno())


@dataclass
class StoreLock:
    path: Path
    stream: TextIO
    owner_pid: int
    active: bool = True

    def validate(self, database):
        expected = lock_path(database)
        if not self.active or self.owner_pid != os.getpid() or self.path != expected:
            raise RuntimeError('Invalid inherited maintenance lock')
        current = self.path.stat()
        held = os.fstat(self.stream.fileno())
        if (current.st_dev, current.st_ino) != (held.st_dev, held.st_ino):
            raise RuntimeError('Maintenance lock identity changed')


def lock_path(database):
    target = Path(database).resolve(strict=True)
    return target.with_name(target.name + '.maintenance.lock')


@contextmanager
def maintenance_lock(database, inherited=None):
    """One stable store lock, retained by a supervisor through recovery readiness."""
    if inherited is not None:
        if not isinstance(inherited, StoreLock):
            raise RuntimeError('Invalid inherited maintenance lock')
        inherited.validate(database)
        yield inherited
        return
    import fcntl
    path = lock_path(database)
    with path.open('a', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sync_directory(path.parent)
        token = StoreLock(path, stream, os.getpid())
        try:
            yield token
        finally:
            token.active = False
