#!/usr/bin/env python3
"""Shared utilities for autoresearch helper scripts."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TextIO


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hermes_home() -> str:
    """Return the active Hermes home directory, respecting HERMES_HOME."""
    return os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))


def atomic_write(path: str | Path, data: Any) -> None:
    """Atomically replace a JSON file in the same filesystem directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(destination.parent), suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty object when absent or invalid."""
    try:
        with Path(path).open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _lock_file(handle: TextIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        if handle.read(1) == "":
            handle.seek(0)
            handle.write("\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle: TextIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def exclusive_lock(path: str | Path) -> Iterator[None]:
    """Hold a cross-process exclusive lock for one state transaction."""
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        _lock_file(handle)
        try:
            yield
        finally:
            _unlock_file(handle)
