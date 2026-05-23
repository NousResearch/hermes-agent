"""Private file helpers for sensitive Hermes artifacts."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


def ensure_private_dir(path: str | Path) -> Path:
    """Create missing directories as owner-only without chmodding existing ones."""
    target = Path(path).expanduser()
    if target.exists():
        if not target.is_dir():
            raise NotADirectoryError(str(target))
        return target

    missing: list[Path] = []
    cursor = target
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent

    for directory in reversed(missing):
        try:
            directory.mkdir(mode=PRIVATE_DIR_MODE)
        except FileExistsError:
            if not directory.is_dir():
                raise NotADirectoryError(str(directory))
        else:
            try:
                directory.chmod(PRIVATE_DIR_MODE)
            except OSError:
                pass
    return target


@contextmanager
def private_text_writer(path: str | Path, *, append: bool = False) -> Iterator[TextIO]:
    """Open a UTF-8 text file with owner-only permissions."""
    target = Path(path).expanduser()
    ensure_private_dir(target.parent)

    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_APPEND if append else os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    fd = os.open(str(target), flags, PRIVATE_FILE_MODE)
    try:
        try:
            os.fchmod(fd, PRIVATE_FILE_MODE)
        except OSError:
            pass
        mode = "a" if append else "w"
        with os.fdopen(fd, mode, encoding="utf-8") as handle:
            fd = -1
            yield handle
    finally:
        if fd >= 0:
            os.close(fd)

    try:
        target.chmod(PRIVATE_FILE_MODE)
    except OSError:
        pass


def write_private_text(path: str | Path, text: str) -> Path:
    """Write a UTF-8 text artifact as owner-only."""
    target = Path(path).expanduser()
    with private_text_writer(target) as handle:
        handle.write(text)
    return target
