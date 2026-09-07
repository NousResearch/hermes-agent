"""Trajectory saving utilities and static helpers.

_convert_to_trajectory_format stays as an AIAgent method (batch_runner.py
calls agent._convert_to_trajectory_format). Only the static helpers and
the file-write logic live here.
"""

import contextlib
import errno
import gzip
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX
    msvcrt = None


_trajectory_lock_guard = threading.Lock()
_trajectory_locks: dict[tuple[int, int], threading.Lock] = {}
_TRAJECTORY_LOCK_TIMEOUT_SECONDS = 10.0
_TRAJECTORY_LOCK_POLL_SECONDS = 0.05
_GZIP_MAGIC = b"\x1f\x8b"


def _acquire_os_lock(lock_file, deadline: float) -> None:
    while True:
        if fcntl is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                pass
        elif msvcrt is not None:
            try:
                # Windows byte-range locks may extend beyond EOF, so locking
                # byte zero also works for a newly created empty data file
                # without writing a marker into the trajectory.
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
        else:  # pragma: no cover - all supported hosts provide one
            raise RuntimeError("cross-process trajectory locking unavailable")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for trajectory append lock")
        time.sleep(min(_TRAJECTORY_LOCK_POLL_SECONDS, remaining))


def _release_os_lock(lock_file) -> None:
    if fcntl is not None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    elif msvcrt is not None:
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


@contextlib.contextmanager
def _trajectory_append_lock(filename):
    """Open and lock the trajectory inode across path aliases."""
    with open(filename, "a+b", buffering=0) as stream:
        stat_result = os.fstat(stream.fileno())
        lock_key = (stat_result.st_dev, stat_result.st_ino)
        with _trajectory_lock_guard:
            thread_lock = _trajectory_locks.setdefault(lock_key, threading.Lock())

        deadline = time.monotonic() + _TRAJECTORY_LOCK_TIMEOUT_SECONDS
        if not thread_lock.acquire(timeout=_TRAJECTORY_LOCK_TIMEOUT_SECONDS):
            raise TimeoutError("timed out waiting for in-process trajectory append lock")
        try:
            _acquire_os_lock(stream, deadline)
            try:
                yield stream
            finally:
                _release_os_lock(stream)
        finally:
            thread_lock.release()


def _write_all(stream, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = stream.write(remaining)
        if not written:
            raise OSError("short trajectory append write")
        remaining = remaining[written:]


def _validate_existing_gzip(stream) -> None:
    """Fail closed unless every existing gzip member reaches a clean EOF."""
    if os.fstat(stream.fileno()).st_size == 0:
        return

    stream.seek(0)
    try:
        with gzip.GzipFile(fileobj=stream, mode="rb") as reader:
            while reader.read(1024 * 1024):
                pass
    except (EOFError, gzip.BadGzipFile, OSError) as exc:
        raise ValueError(
            "existing gzip trajectory has an incomplete or invalid final member"
        ) from exc
    finally:
        stream.seek(0, os.SEEK_END)


def _existing_trajectory_format(stream) -> str | None:
    """Detect a nonempty trajectory as gzip or plain from its bytes."""
    if os.fstat(stream.fileno()).st_size == 0:
        return None
    stream.seek(0)
    magic = stream.read(len(_GZIP_MAGIC))
    stream.seek(0, os.SEEK_END)
    return "gzip" if magic == _GZIP_MAGIC else "plain"


def _validate_trajectory_format(stream, expected_format: str) -> None:
    actual_format = _existing_trajectory_format(stream)
    if actual_format is not None and actual_format != expected_format:
        raise ValueError(
            f"trajectory format conflict: path requests {expected_format}, "
            f"existing file is {actual_format}"
        )


def _append_payload(stream, payload: bytes) -> None:
    """Durably append one complete member/line or restore the old length."""
    original_size = os.fstat(stream.fileno()).st_size
    stream.seek(0, os.SEEK_END)
    try:
        _write_all(stream, payload)
        os.fsync(stream.fileno())
    except Exception:
        try:
            os.ftruncate(stream.fileno(), original_size)
            os.fsync(stream.fileno())
        except Exception:
            logger.error(
                "Failed to roll back partial trajectory append in %s",
                stream.name,
                exc_info=True,
            )
        raise


def convert_scratchpad_to_think(content: str) -> str:
    """Convert <REASONING_SCRATCHPAD> tags to <think> tags."""
    if not content or "<REASONING_SCRATCHPAD>" not in content:
        return content
    return content.replace("<REASONING_SCRATCHPAD>", "<think>").replace("</REASONING_SCRATCHPAD>", "</think>")


def has_incomplete_scratchpad(content: str) -> bool:
    """Check if content has an opening <REASONING_SCRATCHPAD> without a closing tag."""
    if not content:
        return False
    return "<REASONING_SCRATCHPAD>" in content and "</REASONING_SCRATCHPAD>" not in content


def save_trajectory(trajectory: List[Dict[str, Any]], model: str,
                    completed: bool, filename: str = None):
    """Append a trajectory entry to a lossless JSONL file.

    Args:
        trajectory: The ShareGPT-format conversation list.
        model: Model name for metadata.
        completed: Whether the conversation completed successfully.
        filename: Override output filename. Defaults to a gzip-compressed
                  ``.jsonl.gz`` file based on ``completed``. Explicit paths
                  ending in ``.jsonl`` retain the legacy plain-text format.

    Returns:
        ``True`` only after the complete append has been flushed and synced.
        Returns ``False`` after lock, serialization, write, or sync failure.
    """
    if filename is None:
        filename = (
            "trajectory_samples.jsonl.gz"
            if completed
            else "failed_trajectories.jsonl.gz"
        )

    entry = {
        "conversations": trajectory,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "completed": completed,
    }

    try:
        line = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
        expected_format = "gzip" if str(filename).endswith(".gz") else "plain"
        payload = gzip.compress(line) if expected_format == "gzip" else line
        with _trajectory_append_lock(filename) as stream:
            _validate_trajectory_format(stream, expected_format)
            if expected_format == "gzip":
                _validate_existing_gzip(stream)
            _append_payload(stream, payload)
        logger.info("Trajectory saved to %s", filename)
        return True
    except Exception as e:
        logger.warning("Failed to save trajectory: %s", e)
        return False
