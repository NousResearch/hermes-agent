"""Cross-process ownership of resumable partials and their GC boundary."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

from pm.filesystem import lock_fd


@contextmanager
def partial_lock(root: Path, key: str, *, cancelled=None, wait: bool = True, on_wait=None):
    # Lock inodes must survive release: removing one lets a new owner lock a
    # different inode while an older waiter still holds the original handle.
    locks = root / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    fd = os.open(locks / key, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        while not lock_fd(fd, wait=False):
            if not wait:
                yield False
                return
            if cancelled is not None and cancelled():
                yield False
                return
            if on_wait is not None:
                on_wait()
                on_wait = None
            time.sleep(0.05)
        yield True
    finally:
        os.close(fd)


# A large model can run for hours and be resumed after an overnight outage.
GC_GRACE_SECONDS = 7 * 24 * 60 * 60


def collect_download_plans(root: Path, *, grace_seconds: float = GC_GRACE_SECONDS) -> None:
    """Expire abandoned recipes against their latest durable checkpoint."""
    plans = root / ".model-plans"
    now = time.time()
    for path in plans.glob("*.json"):
        if len(path.stem) != 64 or any(char not in "0123456789abcdef" for char in path.stem):
            continue
        with partial_lock(plans, path.stem, wait=False) as acquired:
            if not acquired:
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8-sig"))
                checkpoints = [path]
                for url, _, _ in record["files"]:
                    key = hashlib.sha256(url.encode()).hexdigest()
                    checkpoints.extend(root / f"{key}{suffix}" for suffix in (".part", ".ranges"))
                if any(item.exists() and now - item.stat().st_mtime < grace_seconds for item in checkpoints):
                    continue
                path.unlink(missing_ok=True)
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                continue  # Unreadable metadata cannot establish safe expiry.


def collect_partials(root: Path, *, grace_seconds: float = GC_GRACE_SECONDS) -> None:
    """Do not let a GC snapshot race a downloader acquiring ownership."""
    if not root.is_dir():
        return
    collect_download_plans(root, grace_seconds=grace_seconds)
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
