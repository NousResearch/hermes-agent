"""Trajectory saving + scratchpad helpers (``_convert_to_trajectory_format`` stays an AIAgent method — batch_runner.py calls it)."""

import json
import gzip
import io
import logging
import os
import shutil
import stat
import tempfile
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def convert_scratchpad_to_think(content: str) -> str:
    """Convert <REASONING_SCRATCHPAD> tags to <think> tags."""
    if not content or "<REASONING_SCRATCHPAD>" not in content:
        return content
    return content.replace("<REASONING_SCRATCHPAD>", "<think>").replace("</REASONING_SCRATCHPAD>", "</think>")


def has_incomplete_scratchpad(content: str) -> bool:
    """Whether content has an opening <REASONING_SCRATCHPAD> without a closing tag."""
    return bool(content) and "<REASONING_SCRATCHPAD>" in content and "</REASONING_SCRATCHPAD>" not in content


def _lock_append_handle(f, acquire: bool) -> None:
    """Exclusive whole-file lock on an append handle: ``flock`` on POSIX, a 1-byte
    ``msvcrt.locking`` range at offset 0 on Windows (append position is restored by the OS)."""
    if os.name == "nt":
        import msvcrt
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK if acquire else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if acquire else fcntl.LOCK_UN)


def _build_gzip_member(line: str) -> bytes:
    """Return one complete gzip member without touching the destination file."""
    member = io.BytesIO()
    with gzip.GzipFile(fileobj=member, mode="wb") as compressed:
        compressed.write(line.encode("utf-8"))
    return member.getvalue()


def _append_gzip_member_atomically(filename: str, payload: bytes) -> None:
    """Append a complete member with a stable lock and atomic destination replace.

    A process can be killed at any point during a regular-file write, including
    between short writes. Building the new file beside the destination keeps a
    killed writer from ever publishing a partial gzip member. The sidecar lock
    remains stable across ``os.replace`` so concurrent writers cannot split the
    critical section when the destination inode changes.
    """
    directory = os.path.dirname(os.path.abspath(filename)) or "."
    lock_name = f"{filename}.lock"
    with open(lock_name, "a+b") as lock_file:
        locked = False
        try:
            _lock_append_handle(lock_file, True)
            locked = True
            existing_mode = None
            if os.path.exists(filename):
                existing_mode = stat.S_IMODE(os.stat(filename).st_mode)
            fd, staged_name = tempfile.mkstemp(
                prefix=f".{os.path.basename(filename)}.", suffix=".tmp", dir=directory
            )
            try:
                with os.fdopen(fd, "wb") as staged:
                    if os.path.exists(filename):
                        with open(filename, "rb") as existing:
                            shutil.copyfileobj(existing, staged)
                    staged.write(payload)
                    staged.flush()
                    os.fsync(staged.fileno())
                if existing_mode is not None:
                    os.chmod(staged_name, existing_mode)
                os.replace(staged_name, filename)
            finally:
                if os.path.exists(staged_name):
                    os.unlink(staged_name)
        finally:
            if locked:
                try:
                    _lock_append_handle(lock_file, False)
                except (OSError, ValueError):
                    pass


def save_trajectory(trajectory: List[Dict[str, Any]], model: str, completed: bool, filename: str = None):
    """Append a ShareGPT-format entry, gzip-compressed by default."""
    if filename is None:
        filename = "trajectory_samples.jsonl.gz" if completed else "failed_trajectories.jsonl.gz"
    entry = {"conversations": trajectory, "timestamp": datetime.now().isoformat(), "model": model, "completed": completed}
    try:
        line = json.dumps(entry, ensure_ascii=False) + "\n"  # serialize before taking the lock
        is_gzip = str(filename).endswith(".gz")
        if is_gzip:
            payload = _build_gzip_member(line)
            _append_gzip_member_atomically(filename, payload)
        else:
            with open(filename, "a", encoding="utf-8") as text_file:
                locked = False
                try:
                    _lock_append_handle(text_file, True)
                    locked = True
                    text_file.write(line)
                    text_file.flush()
                    locked = False
                finally:
                    if locked:
                        try:
                            _lock_append_handle(text_file, False)
                        except (OSError, ValueError):
                            pass
        logger.info("Trajectory saved to %s", filename)
    except Exception as e:
        logger.warning("Failed to save trajectory: %s", e)
