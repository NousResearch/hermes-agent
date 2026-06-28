"""Regression test for PTY bridge FD leak when the reader task exits.

Issue #54028: when ``pump_pty_to_ws`` exits (PTY EOF or WS send failure)
the writer loop blocked forever on ``ws.receive()`` because nothing
unblocked it.  The ``finally`` block (which calls ``bridge.close()``)
never ran, leaking the PTY file descriptor.

The fix races ``ws.receive()`` against ``reader_task`` via
``asyncio.wait(FIRST_COMPLETED)`` so the loop exits when the reader
dies, allowing cleanup to proceed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_bridge_closed_when_reader_exits_on_pty_eof():
    """bridge.close() must run even when ws.receive() would block forever.

    Simulates a PTY that immediately returns EOF (``read() -> None``).
    Without the race-based fix the writer loop hangs on ``ws.receive()``
    and ``bridge.close()`` is never called.
    """
    # --- mock bridge: read() returns None (EOF) immediately -----------
    bridge = MagicMock()
    bridge.read = MagicMock(return_value=None)  # EOF on first call
    bridge.close = MagicMock()
    bridge.resize = MagicMock()

    # --- mock WebSocket: receive() blocks forever ----------------------
    ws = AsyncMock()

    async def _blocking_receive():
        # Block until cancelled (reader should exit first, causing the
        # race to cancel this future).
        ev = asyncio.Event()
        try:
            await ev.wait()
        except asyncio.CancelledError:
            raise

    ws.receive = _blocking_receive
    ws.send_bytes = AsyncMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    ws.client = MagicMock()
    ws.client.host = "127.0.0.1"
    ws.query_params = {}

    # --- patch dependencies so pty_ws() reaches the bridge spawn -------
    exc = None
    with (
        patch("hermes_cli.web_server._DASHBOARD_EMBEDDED_CHAT_ENABLED", True),
        patch("hermes_cli.web_server._ws_auth_reason", return_value=(None, "ok")),
        patch("hermes_cli.web_server._ws_auth_mode", return_value="token"),
        patch("hermes_cli.web_server._ws_host_origin_reason", return_value=None),
        patch("hermes_cli.web_server._ws_client_reason", return_value=None),
        patch("hermes_cli.web_server._PTY_BRIDGE_AVAILABLE", True),
        patch("hermes_cli.web_server._channel_or_close_code", return_value=None),
        patch("hermes_cli.web_server._build_sidecar_url", return_value=None),
        patch(
            "hermes_cli.web_server.PtyBridge",
            type("FakePtyBridge", (), {"spawn": staticmethod(lambda argv, **kw: bridge)}),
        ),
        patch(
            "hermes_cli.web_server._resolve_chat_argv_async",
            new_callable=AsyncMock,
            return_value=(["echo"], "/tmp", {}),
        ),
    ):
        from hermes_cli.web_server import pty_ws

        # pty_ws should complete within a few seconds (not hang).
        try:
            await asyncio.wait_for(pty_ws(ws), timeout=5.0)
        except Exception as e:
            exc = e

    # --- the critical assertion: bridge.close() must have been called ---
    print(f"Exception: {exc}")
    print(f"bridge.close called: {bridge.close.called}")
    print(f"bridge.close call_count: {bridge.close.call_count}")
    bridge.close.assert_called_once()
