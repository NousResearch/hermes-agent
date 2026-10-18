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


GC_GRACE_SECONDS = 6 * 60 * 60
MODEL_PLAN_GRACE_SECONDS = 7 * 24 * 60 * 60


def partial_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def is_state_key(key: str) -> bool:
    return len(key) == 64 and all(char in "0123456789abcdef" for char in key)


def _referenced_by_plan(root: Path, url: str, excluding_plan: str | None) -> bool:
    for path in (root / ".model-plans").glob("*.json"):
        if path.stem == excluding_plan:
            continue
        try:
            record = json.loads(path.read_bytes())
            if url in [item[0] for item in record["files"]] + record.get("partial_urls", []):
                return True
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            return True  # An unreadable plan cannot establish safe deletion.
    return False


def discard_partial(root: Path, url: str, *, excluding_plan: str | None = None) -> bool:
    key = partial_key(url)
    with partial_lock(root, key, wait=False) as acquired:
        if not acquired or _referenced_by_plan(root, url, excluding_plan):
            return False
        for suffix in (".part", ".ranges"):
            (root / f"{key}{suffix}").unlink(missing_ok=True)
        return True


def collect_download_plans(root: Path, *, grace_seconds: float = MODEL_PLAN_GRACE_SECONDS) -> set[str]:
    """Expire model plans and return partial keys their retention still protects."""
    plans = root / ".model-plans"
    now = time.time()
    retained = set()
    for path in plans.glob("*.json"):
        if not is_state_key(path.stem):
            continue
        with partial_lock(plans, path.stem, wait=False) as acquired:
            try:
                record = json.loads(path.read_bytes())
                checkpoints = [path]
                keys = set()
                for url in [item[0] for item in record["files"]] + record.get("partial_urls", []):
                    key = partial_key(url)
                    keys.add(key)
                    checkpoints.extend(root / f"{key}{suffix}" for suffix in (".part", ".ranges"))
                if not acquired or any(item.exists() and now - item.stat().st_mtime < grace_seconds for item in checkpoints):
                    retained.update(keys)
                    continue
                path.unlink(missing_ok=True)
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                continue  # Unreadable metadata cannot establish safe expiry.
    return retained


def collect_partials(root: Path, *, grace_seconds: float = GC_GRACE_SECONDS) -> None:
    """Do not let a GC snapshot race a downloader acquiring ownership."""
    if not root.is_dir():
        return
    retained = collect_download_plans(root)
    keys = {path.stem for path in root.iterdir() if path.suffix in {".part", ".ranges"}}
    now = time.time()
    for key in sorted(keys):
        if key in retained:
            continue
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
