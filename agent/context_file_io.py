"""Bounded context-file reads whose bytes and filesystem identity share one handle."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import queue
import stat
from dataclasses import dataclass

from agent.memory_provider import spawn_context_thread

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextFileRead:
    content: str = ""
    identity: tuple | None = None
    status: str = "unreadable"


def _file_identity(path: Path, file_stat: os.stat_result) -> tuple:
    if file_stat.st_ino:
        return ("inode", file_stat.st_dev, file_stat.st_ino)
    return ("path", os.path.normcase(os.path.realpath(path)))


def _read_once(path: Path, *, nofollow: bool = False) -> ContextFileRead:
    # O_NONBLOCK prevents opening a configured FIFO from leaving a blocked reader forever.
    # fstat is authoritative: a replaced path cannot associate another file's identity with these bytes.
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if nofollow:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            return ContextFileRead()
        identity = _file_identity(path, file_stat)
        with os.fdopen(fd, "r", encoding="utf-8-sig", closefd=False) as handle:
            content = handle.read().strip()
        return ContextFileRead(content, identity, "loaded" if content else "empty")
    finally:
        os.close(fd)


def _read_guarded_once(path: Path) -> ContextFileRead:
    from agent.file_safety import get_read_block_error
    # The guard rejects raw NT/device namespaces before resolving anything. Resolving and checking the
    # final target also run inside the deadline: network filesystems can stall before open/read begins.
    if get_read_block_error(str(path)):
        return ContextFileRead(status="blocked")
    target = path.resolve()
    if get_read_block_error(str(target)):
        return ContextFileRead(status="blocked")
    return _read_once(target, nofollow=True)


def read_context_file(path: Path, timeout: float, *, guarded: bool = False) -> ContextFileRead:
    """Read UTF-8/BOM text or skip an unavailable/non-file/slow source without blocking startup.

    A timed-out worker owns only its queue and descriptor; a late result cannot mutate the caller's
    loaded-identity set. Its profile ContextVars are copied by spawn_context_thread.
    """
    result: queue.Queue[ContextFileRead] = queue.Queue(maxsize=1)

    def read() -> None:
        try:
            loaded = _read_guarded_once(path) if guarded else _read_once(path)
        except (OSError, UnicodeError, ValueError, RuntimeError):
            logger.debug("Could not read context file %s", path, exc_info=True)
            loaded = ContextFileRead()
        result.put(loaded)

    try:
        spawn_context_thread(read, name=f"context-read:{path.name}").start()
    except (OSError, RuntimeError):
        logger.warning("Could not start context reader for %s; skipping", path, exc_info=True)
        return ContextFileRead()
    try:
        return result.get(timeout=timeout)
    except queue.Empty:
        logger.warning("Context file %s read timed out after %.1fs; skipping", path, timeout)
        return ContextFileRead()
