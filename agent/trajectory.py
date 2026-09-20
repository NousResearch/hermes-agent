"""Trajectory saving + scratchpad helpers (``_convert_to_trajectory_format`` stays an AIAgent method — batch_runner.py calls it)."""

import json
import gzip
import io
import logging
import os
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
    """Append with a durable rollback offset, writing only the new member.

    The stable lock serializes writers. After a killed writer the next append
    rolls back the incomplete member before proceeding. Ordinary gzip readers
    must wait for that recovery if a writer died during its destination write.
    """
    directory = os.path.dirname(os.path.abspath(filename)) or "."
    journal = f"{filename}.pending"
    with open(f"{filename}.lock", "a+b") as lock_file:
        _lock_append_handle(lock_file, True)
        try:
            with open(filename, "a+b") as destination:
                if os.path.exists(journal):
                    with open(journal, encoding="ascii") as pending:
                        offset = int(pending.read())
                    if offset < 0 or offset > os.fstat(destination.fileno()).st_size:
                        raise ValueError("invalid trajectory recovery offset")
                    destination.truncate(offset)
                    destination.flush()
                    os.fsync(destination.fileno())
                    os.unlink(journal)
                destination.seek(0, os.SEEK_END)
                offset = destination.tell()
                fd, staged_name = tempfile.mkstemp(prefix=".trajectory-", dir=directory)
                try:
                    os.chmod(staged_name, os.fstat(destination.fileno()).st_mode & 0o777)
                    with os.fdopen(fd, "w", encoding="ascii") as pending:
                        pending.write(str(offset))
                        pending.flush()
                        os.fsync(pending.fileno())
                    os.replace(staged_name, journal)
                    _sync_trajectory_directory(directory)
                    try:
                        destination.write(payload)
                        destination.flush()
                        os.fsync(destination.fileno())
                    except BaseException:
                        destination.truncate(offset)
                        destination.flush()
                        os.fsync(destination.fileno())
                        raise
                    os.unlink(journal)
                    _sync_trajectory_directory(directory)
                finally:
                    if os.path.exists(staged_name):
                        os.unlink(staged_name)
        finally:
            _lock_append_handle(lock_file, False)


def _sync_trajectory_directory(directory: str) -> None:
    if os.name != "nt":
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


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
