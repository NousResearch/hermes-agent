"""Cross-process serialization for source TUI dependency and bundle preparation."""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def tui_preparation_lock(workspace_root: Path) -> Iterator[None]:
    """Serialize install/build mutations for one source TUI workspace."""
    lock_path = (
        workspace_root
        / "node_modules"
        / ".cache"
        / "hermes"
        / "tui-preparation.lock"
    )
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+b")
    except OSError:
        yield
        return

    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except (ImportError, OSError):
        handle.close()
        yield
        return

    try:
        yield
    finally:
        if os.name == "nt":
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()
