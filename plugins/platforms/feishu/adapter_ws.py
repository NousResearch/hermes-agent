"""Receive/open timeouts for the lark_oapi WebSocket client (#156356).

``lark_oapi.ws.client`` (pinned 1.7.5) has no timeout anywhere in its receive
path: ``_receive_message_loop`` awaits ``self._conn.recv()`` with no bound, the
bootstrap ``_get_conn_url`` calls ``requests.post`` with no timeout, and a
failed ping is only logged. On a half-dead TCP connection (VPN tunnel flap) the
receive loop blocks forever in ``SSL_read`` and the SDK's reconnect ladder is
never reached: the gateway stays up but silently deaf for hours or days.

The recovery ownership stays with the SDK. ``_receive_message_loop`` already
does ``except Exception: _disconnect(); _reconnect()`` — so all Hermes has to
do is make a stalled ``recv()`` *raise* after a budget, and let the ladder take
over. This module provides the two seams that do exactly that:

* :class:`TimedRecvConnection` — a delegating proxy installed over
  ``client._conn`` that bounds each ``recv()`` with ``asyncio.wait_for`` and
  aborts the stalled transport when the budget expires. The raised
  ``TimeoutError`` propagates into the SDK's own exception handler.
* :class:`TimedRequestsFacade` — a module-local ``requests`` replacement for
  the SDK's synchronous bootstrap POST, giving it a finite ``(connect, read)``
  budget it never had.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: Minimum automatic receive budget (seconds) when no explicit value is set.
MIN_AUTO_READ_TIMEOUT = 300.0


def effective_read_timeout(
    explicit: Optional[float],
    ping_interval: Optional[int],
) -> Optional[float]:
    """Resolve the per-``recv()`` budget.

    An explicit positive ``ws_read_timeout`` wins. An explicit ``0`` opts out
    entirely (``None`` result, the pre-fix behavior). ``None``/null means auto:
    the budget is derived from the live SDK ping interval so a healthy
    heartbeat cadence never trips it — ``max(300, 3 * ping_interval)``. With no
    ping interval to derive from, auto degrades to no budget rather than
    guessing.
    """
    if explicit is not None:
        return explicit if explicit > 0 else None
    if ping_interval is not None and ping_interval > 0:
        return max(MIN_AUTO_READ_TIMEOUT, 3.0 * ping_interval)
    return None


class TimedRecvConnection:
    """Delegate everything to the real connection except ``recv()``, which is bounded.

    The SDK touches its connection through ``recv()``, ``send()``, ``close()``
    and attribute probes only (``lark_oapi.ws.client`` 1.7.5), so a plain
    ``__getattr__`` delegate with one overridden coroutine is enough. When the
    budget expires the underlying transport is aborted first: a socket stuck
    in ``SSL_read`` can also hold a blocked ``send()`` that the SDK's
    ``_disconnect`` (which awaits ``close()`` under its lock) would otherwise
    wait on forever.
    """

    def __init__(self, connection: Any, read_timeout: Optional[float]) -> None:
        self._connection = connection
        self._read_timeout = read_timeout

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_connection", "_read_timeout"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._connection, name, value)

    async def recv(self, *args: Any, **kwargs: Any) -> Any:
        timeout = self._read_timeout
        if timeout is None:
            return await self._connection.recv(*args, **kwargs)
        try:
            return await asyncio.wait_for(
                self._connection.recv(*args, **kwargs), timeout=timeout
            )
        except TimeoutError as exc:
            # Abort the stalled transport so blocked writes and the pending
            # read all fail now instead of parking the SDK's disconnect lock.
            transport = getattr(self._connection, "transport", None)
            abort = getattr(transport, "abort", None)
            if abort is not None:
                try:
                    abort()
                except Exception:  # pragma: no cover - transport already gone
                    logger.debug(
                        "Feishu WS transport abort after read timeout failed",
                        exc_info=True,
                    )
            raise TimeoutError(f"Feishu WS receive timed out after {timeout}s") from exc


class TimedRequestsFacade:
    """Bound the SDK's synchronous bootstrap POST without touching shared ``requests``.

    ``lark_oapi.ws.client`` calls ``requests.post(url, headers=..., json=...)``
    inside ``_get_conn_url`` with no timeout, on the WS worker thread, *before*
    ``websockets.connect`` runs. A half-dead tunnel wedges that POST the same
    way it wedges ``recv`` — and no connection-level timeout can help because
    no connection exists yet. This facade is rebound onto the SDK module only
    (``ws_client_module.requests``), so the process-wide ``requests`` module
    other plugins rely on keeps its exact behavior. Calls from threads that
    never registered a Hermes WS budget pass straight through.
    """

    def __init__(
        self,
        real_requests: Any,
        budget_for_thread: Callable[[], Optional[tuple[float, float]]],
    ) -> None:
        self._real = real_requests
        self._budget_for_thread = budget_for_thread

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def post(self, *args: Any, **kwargs: Any) -> Any:
        budget = self._budget_for_thread()
        if budget is None or kwargs.get("timeout") is not None:
            return self._real.post(*args, **kwargs)
        kwargs["timeout"] = budget
        return self._real.post(*args, **kwargs)


def wrap_client_connection(client: Any, read_timeout: Optional[float]) -> None:
    """Install :class:`TimedRecvConnection` over ``client._conn`` (idempotent).

    Called from the receive-entry wrapper on every (re)connected socket: the
    SDK replaces ``_conn`` on each reconnect, so the proxy must be re-attached
    per link. Never unwraps in a ``finally`` — after a successful reconnect
    ``client._conn`` is already the *new* connection and must keep its proxy.
    """
    conn = getattr(client, "_conn", None)
    if conn is None or isinstance(conn, TimedRecvConnection):
        return
    client._conn = TimedRecvConnection(conn, read_timeout)
