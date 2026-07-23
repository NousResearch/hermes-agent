"""Robust asyncio stdio streams for the ACP adapter.

``agent-client-protocol`` uses ``loop.connect_{read,write}_pipe`` on POSIX.
Those only accept pipes, sockets, and (for write) character devices. When a
host hands Hermes non-pipe stdio — regular files, some Desktop/launcher
wrappers, certain macOS launch paths — asyncio raises:

    ValueError: Pipe transport is only for pipes, sockets and character devices

``os.dup()`` does **not** change the descriptor type, so it cannot fix this.

This module mirrors the Windows path already present in ACP 0.9+:
thread-fed stdin + a custom stdout transport that writes to
``sys.stdout.buffer``. On POSIX we still prefer the native pipe transport
(lower overhead, correct for the normal IDE/parent-with-pipes case) and only
fall back when the pipe transport rejects the fd.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import stat
import sys
import threading
from asyncio import transports as aio_transports
from typing import Optional, cast

logger = logging.getLogger(__name__)


class _WritePipeProtocol(asyncio.BaseProtocol):
    """Minimal protocol exposing ``_drain_helper`` for StreamWriter."""

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._paused = False
        self._drain_waiter: Optional[asyncio.Future[None]] = None

    def pause_writing(self) -> None:  # type: ignore[override]
        self._paused = True
        if self._drain_waiter is None:
            self._drain_waiter = self._loop.create_future()

    def resume_writing(self) -> None:  # type: ignore[override]
        self._paused = False
        if self._drain_waiter is not None and not self._drain_waiter.done():
            self._drain_waiter.set_result(None)
        self._drain_waiter = None

    async def _drain_helper(self) -> None:
        if self._paused and self._drain_waiter is not None:
            await self._drain_waiter


class _BufferStdoutTransport(asyncio.BaseTransport):
    """Write-only transport that pushes bytes to ``sys.stdout.buffer``."""

    def __init__(self) -> None:
        self._is_closing = False

    def write(self, data: bytes) -> None:  # type: ignore[override]
        if self._is_closing:
            return
        try:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        except Exception:
            logger.exception("Error writing ACP frame to stdout")

    def can_write_eof(self) -> bool:  # type: ignore[override]
        return False

    def is_closing(self) -> bool:  # type: ignore[override]
        return self._is_closing

    def close(self) -> None:  # type: ignore[override]
        self._is_closing = True
        with contextlib.suppress(Exception):
            sys.stdout.flush()

    def abort(self) -> None:  # type: ignore[override]
        self.close()

    def get_extra_info(self, name: str, default=None):  # type: ignore[override]
        return default


def _start_stdin_feeder(
    loop: asyncio.AbstractEventLoop,
    reader: asyncio.StreamReader,
) -> None:
    """Block-read stdin on a daemon thread and feed the asyncio reader."""

    def blocking_read() -> None:
        try:
            while True:
                data = sys.stdin.buffer.readline()
                if not data:
                    break
                loop.call_soon_threadsafe(reader.feed_data, data)
        finally:
            loop.call_soon_threadsafe(reader.feed_eof)

    threading.Thread(target=blocking_read, daemon=True, name="acp-stdin-feeder").start()


def _is_pipe_transport_error(exc: BaseException) -> bool:
    if not isinstance(exc, ValueError):
        return False
    msg = str(exc).lower()
    # CPython variants:
    #   "Pipe transport is only for pipes, sockets and character devices"
    #   "Pipe transport is for pipes/sockets only."
    return "pipe transport" in msg


def _stdio_fds_support_pipe_transport() -> tuple[bool, str]:
    """Return whether stdin/stdout look safe for asyncio pipe transports.

    CPython's POSIX pipe transports accept:
      - read:  FIFO / socket  (TTY/CHR often rejected on macOS)
      - write: FIFO / socket / character device

    Checking up front avoids a partial ``connect_read_pipe`` that steals
    stdin before ``connect_write_pipe`` fails on a bad stdout.
    """
    try:
        in_mode = os.fstat(sys.stdin.fileno()).st_mode
        out_mode = os.fstat(sys.stdout.fileno()).st_mode
    except Exception as exc:  # noqa: BLE001 — any fileno failure → fallback
        return False, f"stdio fileno/stat failed: {exc}"

    in_ok = stat.S_ISFIFO(in_mode) or stat.S_ISSOCK(in_mode)
    out_ok = (
        stat.S_ISFIFO(out_mode)
        or stat.S_ISSOCK(out_mode)
        or stat.S_ISCHR(out_mode)
    )
    if in_ok and out_ok:
        return True, "ok"

    def _kind(mode: int) -> str:
        if stat.S_ISFIFO(mode):
            return "fifo"
        if stat.S_ISSOCK(mode):
            return "sock"
        if stat.S_ISCHR(mode):
            return "chr"
        if stat.S_ISREG(mode):
            return "reg"
        return f"mode={oct(mode)}"

    return False, f"stdin={_kind(in_mode)} stdout={_kind(out_mode)}"


async def _fallback_stdio_streams(
    loop: asyncio.AbstractEventLoop,
    limit: Optional[int] = None,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Thread-fed stdin + buffer-backed stdout (ACP Windows strategy)."""
    reader = (
        asyncio.StreamReader(limit=limit) if limit is not None else asyncio.StreamReader()
    )
    _ = asyncio.StreamReaderProtocol(reader)
    _start_stdin_feeder(loop, reader)

    write_protocol = _WritePipeProtocol()
    transport = _BufferStdoutTransport()
    writer = asyncio.StreamWriter(
        cast(aio_transports.WriteTransport, transport),
        write_protocol,
        None,
        loop,
    )
    return reader, writer


async def _posix_pipe_stdio_streams(
    loop: asyncio.AbstractEventLoop,
    limit: Optional[int] = None,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Native asyncio pipe transports (preferred on POSIX when fds allow)."""
    reader = (
        asyncio.StreamReader(limit=limit) if limit is not None else asyncio.StreamReader()
    )
    reader_protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: reader_protocol, sys.stdin)

    write_protocol = _WritePipeProtocol()
    transport, _ = await loop.connect_write_pipe(lambda: write_protocol, sys.stdout)
    writer = asyncio.StreamWriter(transport, write_protocol, None, loop)
    return reader, writer


async def open_acp_stdio_streams(
    limit: Optional[int] = None,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open stdio streams for ACP JSON-RPC.

    Returns ``(reader, writer)`` in the same order as
    ``acp.stdio.stdio_streams`` (reader from client, writer to client).

    Strategy:
    - Windows → always thread/buffer fallback (pipe transports unsupported).
    - POSIX → if stdin/stdout look pipe-compatible, try native transports;
      otherwise (or on pipe-transport ``ValueError``) fall back to
      thread/buffer so the adapter still starts.
    """
    loop = asyncio.get_running_loop()
    if platform.system() == "Windows":
        return await _fallback_stdio_streams(loop, limit=limit)

    ok, reason = _stdio_fds_support_pipe_transport()
    if not ok:
        logger.warning(
            "ACP stdio not pipe-compatible (%s); "
            "using thread/buffer fallback transport",
            reason,
        )
        return await _fallback_stdio_streams(loop, limit=limit)

    try:
        return await _posix_pipe_stdio_streams(loop, limit=limit)
    except ValueError as exc:
        if not _is_pipe_transport_error(exc):
            raise
        logger.warning(
            "ACP stdio pipe transport unavailable (%s); "
            "using thread/buffer fallback transport",
            exc,
        )
        return await _fallback_stdio_streams(loop, limit=limit)
