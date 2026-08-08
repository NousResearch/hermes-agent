"""Async Unix-socket client for the harness dispatcher (Phase 2.6).

Wraps `asyncio.open_unix_connection` with a retry loop, a per-call
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
from typing import Optional

from .dispatcher_protocol import (
    OP_PING,
    Envelope,
    STATUS_OK,
    make_request,
)


_LOG = logging.getLogger(__name__)


# Default socket path. The harness systemd unit places the dispatcher
# under /run/harness/ (RuntimeDirectory=harness). Overridable for tests
# and for non-systemd installs.
DEFAULT_DISPATCHER_SOCKET = "/run/harness/dispatcher.sock"

# Per-call timeout for one round-trip. The dispatcher is a single-
# Python-process server doing one envelope at a time per connection
# (synchronous dispatch path), so a few hundred ms is plenty under
# normal load. Long-running handlers (async / background mode) are
# NOT supported via this client: they return immediately with
# `accepted=True` and the actual response is delivered through a
# separate mechanism (callback, not yet implemented -- Phase 2.7
# is deferred to Phase 4.5b per PLAN.md).
DEFAULT_DISPATCHER_TIMEOUT_S = 5.0

# How many times to retry a transient connection failure (e.g. the
# dispatcher crashed and restarted between two calls). 2 retries
# with a fresh connection each time = 3 total attempts.
DEFAULT_MAX_RETRIES = 2


class DispatcherConnectionError(Exception):
    """Raised when the dispatcher is unreachable, refused the
    connection, or timed out after exhausting retries. Callers
    should catch this and fall back (e.g. surface "dispatcher down"
    to the user, or route the message through the agent instead).
    """


class DispatcherClient:
    """Async Unix-socket client for the harness dispatcher.

    One client per process. Lazy-connects on first dispatch().
    Reconnects on transient failure up to max_retries times. Closes
    cleanly on close() or context-manager exit.

    Thread safety: not thread-safe. Single asyncio loop only.
    """

    def __init__(
        self,
        socket_path: str = DEFAULT_DISPATCHER_SOCKET,
        timeout_s: float = DEFAULT_DISPATCHER_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._path = socket_path
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        # Lazy: opened on first dispatch() call.
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        # Set to True after close() so a stale client cannot dispatch.
        self._closed = False
        # Serializes concurrent dispatch() calls so writes/reads
        # on the shared socket don't interleave.
        self._lock = asyncio.Lock()

    @property
    def socket_path(self) -> str:
        return self._path

    @property
    def is_connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def connect(self) -> None:
        """Open the Unix socket. Idempotent: if already connected,
        no-op. Raises DispatcherConnectionError if the socket cannot
        be opened (dispatcher not running, wrong path, permissions).
        """
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
        """Close the connection. Idempotent."""
        self._closed = True
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except (OSError, ConnectionError):
                # Best-effort: the peer may have already hung up.
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
        transient connection failure (ConnectionResetError,
        BrokenPipeError), retrying up to max_retries times. A hard
        failure (timeout, refused, exhausted retries) raises
        DispatcherConnectionError so the caller can fall back.

        Concurrent calls are serialized via an asyncio.Lock so
        writes/reads on the shared socket don't interleave.
        """
        async with self._lock:
            return await self._dispatch_locked(envelope)

    async def _dispatch_locked(self, envelope: Envelope) -> Envelope:
        """dispatch() implementation. Caller must hold self._lock."""
        if self._closed:
            raise DispatcherConnectionError("client is closed")

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._dispatch_once(envelope)
            except (ConnectionResetError, BrokenPipeError) as e:
                # Peer hung up. Drop the stale connection, reconnect,
                # retry. This handles the case where the dispatcher
                # restarted between two calls.
                last_error = e
                _LOG.warning(
                    "dispatcher connection lost (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    e,
                )
                await self._drop_connection()
            except asyncio.TimeoutError as e:
                # No response within the timeout. The connection may
                # still be usable for a follow-up, but we don't trust
                # it -- a half-sent envelope could be in flight. Drop
                # and reconnect.
                last_error = e
                _LOG.warning(
                    "dispatcher dispatch timed out after %.1fs "
                    "(attempt %d/%d)",
                    self._timeout_s,
                    attempt + 1,
                    self._max_retries + 1,
                )
                await self._drop_connection()
            except (OSError, ConnectionError) as e:
                # Refused connection or similar. Drop and reconnect.
                last_error = e
                _LOG.warning(
                    "dispatcher dispatch failed (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    e,
                )
                await self._drop_connection()

        raise DispatcherConnectionError(
            f"dispatcher unreachable after {self._max_retries + 1} attempts: "
            f"{last_error}"
        )

    async def _dispatch_once(self, envelope: Envelope) -> Envelope:
        """One round-trip. Caller handles retries on transient errors."""
        if not self.is_connected:
            await self.connect()
        assert self._writer is not None and self._reader is not None
        writer = self._writer
        reader = self._reader

        try:
            writer.write(envelope.to_jsonl())
            await writer.drain()
            line = await asyncio.wait_for(
                reader.readuntil(b"\n"), timeout=self._timeout_s
            )
        except (ConnectionResetError, BrokenPipeError):
            # Propagate to the dispatch() loop for reconnect.
            raise
        except asyncio.IncompleteReadError as e:
            # Peer closed without sending a full line. Treat as a
            # connection failure so the loop reconnects.
            raise ConnectionError(
                f"dispatcher closed before sending response: {e}"
            ) from e
        except asyncio.LimitOverrunError as e:
            # Response line exceeded the stream reader limit.
            # Treat as a non-fatal dispatcher error so the caller
            # falls through to normal message handling.
            raise DispatcherConnectionError(
                f"dispatcher response too large: {e}"
            ) from e

        return Envelope.from_jsonl(line)

    async def _drop_connection(self) -> None:
        """Close the current connection (if any) without affecting
        the closed flag, so the next dispatch() reopens it."""
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except (OSError, ConnectionError):
                pass
        self._reader = None
        self._writer = None

    async def ping(self) -> bool:
        """Send a ping and return True if the dispatcher responds
        with STATUS_OK. Returns False on any failure (no exception
        raised) so callers can use this for liveness probes.
        """
        try:
            req = make_request(OP_PING, {})
            resp = await self.dispatch(req)
            return resp.status == STATUS_OK
        except (DispatcherConnectionError, ValueError) as e:
            _LOG.debug("ping failed: %s", e)
            return False