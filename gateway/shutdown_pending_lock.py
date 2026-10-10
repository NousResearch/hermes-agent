"""Serialise mutations of an existing pending snapshot across gateway owners."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path


@contextmanager
def pending_snapshot_lock(path: Path) -> Iterator[None]:
    from pm.filesystem import lock_fd

    # Keep the sidecar inode stable while another process waits for its lock.
    lock_path = path.with_name(f".{path.name}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if not lock_fd(descriptor, wait=True, timeout=5.0):
            raise BlockingIOError(f"pending snapshot is being changed: {path}")
        yield
    finally:
        os.close(descriptor)
