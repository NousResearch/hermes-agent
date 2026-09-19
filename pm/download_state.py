"""Cross-process ownership of resumable partials and their GC boundary."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import time

from hermes_cli.runtime_state import _lock


@contextmanager
def partial_lock(root: Path, key: str, *, cancelled=None, wait: bool = True):
    # Lock inodes must survive release: removing one lets a new owner lock a
    # different inode while an older waiter still holds the original handle.
    locks = root / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    fd = os.open(locks / key, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        while not _lock(fd, wait=False):
            if not wait:
                yield False
                return
            if cancelled is not None and cancelled():
                yield False
                return
            time.sleep(0.05)
        yield True
    finally:
        os.close(fd)


GC_GRACE_SECONDS = 6 * 60 * 60


def collect_partials(root: Path, *, grace_seconds: float = GC_GRACE_SECONDS) -> None:
    """Do not let a GC snapshot race a downloader acquiring ownership."""
    if not root.is_dir():
        return
    keys = {path.stem for path in root.iterdir() if path.suffix in {".part", ".ranges"}}
    now = time.time()
    for key in sorted(keys):
        with partial_lock(root, key, wait=False) as acquired:
            if not acquired:
                continue
            pair = [root / f"{key}{suffix}" for suffix in (".part", ".ranges")]
            try:
                if any(path.exists() and grace_seconds > 0 and now - path.stat().st_mtime < grace_seconds for path in pair):
                    continue
                for path in pair:
                    path.unlink(missing_ok=True)
            except OSError:
                continue  # An inaccessible partial is not evidence it is unused.
