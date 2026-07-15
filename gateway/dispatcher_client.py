"""Async Unix-socket client for an external dispatcher service.

Wraps ``asyncio.open_unix_connection`` with a retry loop, a per-call
timeout, and a single connection per client instance. The client is
long-lived: construct once at gateway startup, call dispatch() many
times, close() at shutdown. Reconnect is automatic on transient
failures (ConnectionResetError, BrokenPipeError); a hard failure
(dispatcher down, refused connection, timeout) raises
DispatcherConnectionError so the caller can fall back.

Wire shape and Envelope dataclass live in dispatcher_protocol.py;
this module is the transport layer only.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from .dispatcher_protocol import (
    OP_PING,
    Envelope,
    STATUS_OK,
    make_request,
)


_LOG = logging.getLogger(__name__)


def _resolve_default_socket() -> str:
    """Build the default socket path using the current user's UID.

    Follows the XDG runtime directory convention:
    ``/run/user/<uid>/dispatcher/dispatcher.sock``.
    """
    if not hasattr(os, "getuid"):  # windows-footgun: ok
        raise RuntimeError(
            "AF_UNIX dispatcher requires a Unix platform; "
            "set DISPATCHER_SOCKET_PATH explicitly or disable the dispatcher"
        )
    uid = os.getuid()
    return f"/run/user/{uid}/dispatcher/dispatcher.sock"


# Per-call timeout for one round-trip.
DEFAULT_DISPATCHER_TIMEOUT_S = 5.0

# Retry count for transient connection failures. 2 retries with a
# fresh connection each time = 3 total attempts.
DEFAULT_MAX_RETRIES = 2


class DispatcherConnectionError(Exception):
    """Raised when the dispatcher is unreachable, refused the
    connection, or timed out after exhausting retries."""


class DispatcherClient:
    """Async Unix-socket client for an external dispatcher.

    One client per process. Lazy-connects on first dispatch().
    Reconnects on transient failure up to max_retries times. Closes
    cleanly on close() or context-manager exit.

    Thread safety: not thread-safe. Single asyncio loop only.
    """

    def __init__(
        self,
        socket_path: str | None = None,
        timeout_s: float = DEFAULT_DISPATCHER_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._path = (
            socket_path
            or os.environ.get("DISPATCHER_SOCKET_PATH")
            or _resolve_default_socket()
        )
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def socket_path(self) -> str:
        return self._path

    @property
    def is_connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def connect(self) -> None:
        """Open the Unix socket (lock-safe). Idempotent. Raises
        DispatcherConnectionError if the socket cannot be opened.

        Acquires ``_lock`` so concurrent callers cannot race into
        ``open_unix_connection`` simultaneously. Internal methods
        that already hold the lock call ``_connect_unlocked``.
        """
        async with self._lock:
            await self._connect_unlocked()

    async def _connect_unlocked(self) -> None:
        """Open the Unix socket. Caller must hold ``_lock``."""
        if self._closed:
            raise DispatcherConnectionError("client is closed")
        if self.is_connected:
            return
        try:
            self._reader, self._writer = await asyncio.open_unix_connection(
                self._path
            )
            _LOG.debug("dispatcher client connected to %s", self._path)
        except (OSError, ConnectionError) as e:
            self._reader = None
            self._writer = None
            raise DispatcherConnectionError(
                f"failed to connect to dispatcher at {self._path}: {e}"
            ) from e

    async def close(self) -> None:
        """Close the connection (lock-safe). Idempotent.

        Acquires ``_lock`` so an in-flight ``dispatch()`` finishes
        before the transport is torn down.
        """
        async with self._lock:
            await self._close_unlocked()

    async def _close_unlocked(self) -> None:
        """Close the connection. Caller must hold ``_lock``.

        Calls ``_writer.close()`` only (no ``wait_closed``). The
        caller in run.py wraps ``close()`` in ``wait_for(timeout=3)``;
        if ``wait_closed`` blocked on an unresponsive peer, the
        timeout cancellation would leave the fd in an indeterminate
        state. A plain ``close()`` schedules the teardown without
        blocking.
        """
        self._closed = True
        if self._writer is not None:
            try:
                self._writer.close()
            except (OSError, ConnectionError):
                pass
        self._reader = None
        self._writer = None

    async def __aenter__(self) -> "DispatcherClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def dispatch(self, envelope: Envelope) -> Envelope:
        """Send an envelope and return the response envelope.

        Lazy-connects on first call. Reconnects transparently on
        transient connection failure, retrying up to max_retries
        times. Raises DispatcherConnectionError on hard failure.
        """
        async with self._lock:
            return await self._dispatch_locked(envelope)

    async def _dispatch_locked(self, envelope: Envelope) -> Envelope:
        if self._closed:
            raise DispatcherConnectionError("client is closed")

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._dispatch_once(envelope)
            except (
                ConnectionResetError,
                BrokenPipeError,
                ConnectionError,
            ) as e:
                last_error = e
                _LOG.warning(
                    "dispatcher connection lost (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    e,
                )
                await self._drop_connection()
            except asyncio.TimeoutError as e:
                last_error = e
                _LOG.warning(
                    "dispatcher dispatch timed out after %.1fs "
                    "(attempt %d/%d)",
                    self._timeout_s,
                    attempt + 1,
                    self._max_retries + 1,
                )
                await self._drop_connection()
            except OSError as e:
                last_error = e
                _LOG.warning(
                    "dispatcher dispatch failed (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    e,
                )
                await self._drop_connection()

        raise DispatcherConnectionError(
            f"dispatcher unreachable after {self._max_retries + 1} "
            f"attempts: {last_error}"
        )

    # Maximum bytes to read for a single response line. Prevents a
    # misbehaving dispatcher from consuming unbounded memory.
    _MAX_LINE_BYTES = 1024 * 1024  # 1 MiB

    async def _dispatch_once(self, envelope: Envelope) -> Envelope:
        if not self.is_connected:
            await self._connect_unlocked()
        if self._writer is None or self._reader is None:
            raise DispatcherConnectionError(
                "connection not established after connect"
            )
        writer = self._writer
        reader = self._reader

        try:
            writer.write(envelope.to_jsonl())
            await writer.drain()
            line = await asyncio.wait_for(
                reader.readuntil(b"\n"), timeout=self._timeout_s
            )
            if len(line) > self._MAX_LINE_BYTES:
                raise ConnectionError(
                    f"dispatcher response too large ({len(line)} bytes)"
                )
        except (ConnectionResetError, BrokenPipeError):
            raise
        except asyncio.IncompleteReadError as e:
            raise ConnectionError(
                f"dispatcher closed before sending response: {e}"
            ) from e
        except asyncio.LimitOverrunError:
            raise DispatcherConnectionError(
                "dispatcher response exceeded line buffer limit"
            )

        return Envelope.from_jsonl(line)

    async def _drop_connection(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except (OSError, ConnectionError):
                pass
        self._reader = None
        self._writer = None

    async def ping(self) -> bool:
        """Send a ping. Returns True on STATUS_OK, False on any
        failure (no exception raised)."""
        try:
            req = make_request(OP_PING, {})
            resp = await self.dispatch(req)
            return resp.status == STATUS_OK
        except (DispatcherConnectionError, ValueError) as e:
            _LOG.debug("ping failed: %s", e)
            return False
