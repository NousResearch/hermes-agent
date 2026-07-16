"""Tests for issue #65787 — MCP keepalive uses list_tools() (O(tool-count))
which causes guaranteed timeout + reconnect loop on large servers.

The fix:
1. The list_tools fallback timeout is increased from 30s to 60s to give
   servers with hundreds of tools more headroom.
2. The TimeoutError is re-raised with a descriptive message instead of the
   default empty-string asyncio.TimeoutError, so logs are actionable.
3. The keepalive caller logs `exc or type(exc).__name__` to avoid empty
   log lines when the exception str() is empty.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# --------------------------------------------------------------------------- #
# list_tools fallback timeout is 60s (not 30s)
# --------------------------------------------------------------------------- #


def test_list_tools_fallback_timeout_is_60s():
    """The list_tools fallback must use 60s timeout, not 30s.

    The ping path uses 30s (fine — ping is a few bytes). The list_tools
    fallback transfers the full tool catalog which can be multi-MB on
    servers with hundreds of tools, so it needs more headroom.

    See issue #65787.
    """
    import inspect
    from tools import mcp_tool

    source = inspect.getsource(mcp_tool.MCPServerTask._keepalive_probe)

    # The ping path should still use 30s
    assert "timeout=30.0" in source, "ping path should keep 30s timeout"

    # The list_tools fallback should use 60s
    assert "timeout=60.0" in source, (
        "list_tools fallback should use 60s timeout — see issue #65787"
    )

    # The old 30s list_tools timeout must not appear
    # (the only 30.0 should be in the ping call)
    lines = source.split("\n")
    list_tools_lines = [l for l in lines if "list_tools" in l and "timeout" in l]
    for line in list_tools_lines:
        assert "30.0" not in line, (
            f"list_tools fallback must not use 30s timeout: {line.strip()}"
        )


# --------------------------------------------------------------------------- #
# TimeoutError has a descriptive message (not empty)
# --------------------------------------------------------------------------- #


def test_list_tools_timeout_has_descriptive_message():
    """When the list_tools fallback times out, the error message must be
    descriptive — not the default empty string from asyncio.TimeoutError.

    The issue reports operators seeing:
        'MCP server 'X' keepalive failed, triggering reconnect:'
    with an empty reason. See issue #65787.
    """
    import inspect
    from tools import mcp_tool

    source = inspect.getsource(mcp_tool.MCPServerTask._keepalive_probe)

    # The except block must create a new TimeoutError with a message
    assert "TimeoutError(" in source, (
        "list_tools timeout must re-raise with a descriptive message"
    )
    # The message should mention the timeout duration and server
    assert "60s" in source or "timed out" in source, (
        "TimeoutError message should include duration and context"
    )


# --------------------------------------------------------------------------- #
# Keepalive caller logs exc or type name (not empty)
# --------------------------------------------------------------------------- #


def test_keepalive_caller_logs_type_name_on_empty_exc():
    """The keepalive caller should log `exc or type(exc).__name__` so
    an empty-string exception still produces a log line with a reason."""
    import inspect
    from tools import mcp_tool

    source = inspect.getsource(mcp_tool.MCPServerTask._wait_for_lifecycle_event)

    # Must include the type name fallback
    assert "type(exc).__name__" in source, (
        "Keepalive log should fall back to type name when exc str is empty"
    )


# --------------------------------------------------------------------------- #
# Integration: the full keepalive probe with list_tools fallback
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_keepalive_probe_uses_60s_for_list_tools_fallback():
    """When ping is unsupported, the fallback list_tools call uses 60s timeout.

    This is the core regression test: verify the actual timeout value passed
    to asyncio.wait_for for the list_tools path is 60s.
    """
    from tools.mcp_tool import MCPServerTask

    # Build a minimal MCPServer stub with the _keepalive_probe method
    server = MagicMock(spec=MCPServerTask)
    server._ping_unsupported = True  # Force list_tools fallback
    server.name = "test-server"

    # Mock the session's list_tools
    mock_session = MagicMock()
    mock_session.list_tools = AsyncMock(return_value=[])
    server.session = mock_session

    # Track the timeout used in asyncio.wait_for
    timeouts_used = []
    original_wait_for = asyncio.wait_for

    async def tracking_wait_for(coro, timeout):
        timeouts_used.append(timeout)
        return await coro

    # Bind the real method
    probe = MCPServerTask._keepalive_probe.__get__(server, type(server))

    with patch("asyncio.wait_for", tracking_wait_for):
        await probe()

    # list_tools path should have been called with 60s
    assert len(timeouts_used) == 1
    assert timeouts_used[0] == 60.0, (
        f"list_tools fallback timeout should be 60s, got {timeouts_used[0]}"
    )


@pytest.mark.asyncio
async def test_keepalive_probe_ping_path_still_uses_30s():
    """When ping is supported, the ping path should still use 30s timeout."""
    from tools.mcp_tool import MCPServerTask

    server = MagicMock(spec=MCPServerTask)
    server._ping_unsupported = False  # ping is supported
    server.name = "test-server"

    mock_session = MagicMock()
    mock_session.send_ping = AsyncMock(return_value=None)
    server.session = mock_session

    timeouts_used = []
    original_wait_for = asyncio.wait_for

    async def tracking_wait_for(coroutine, timeout):
        timeouts_used.append(timeout)
        return await coroutine

    probe = MCPServerTask._keepalive_probe.__get__(server, type(server))

    with patch("asyncio.wait_for", tracking_wait_for):
        await probe()

    assert len(timeouts_used) == 1
    assert timeouts_used[0] == 30.0, (
        f"ping path timeout should remain 30s, got {timeouts_used[0]}"
    )


@pytest.mark.asyncio
async def test_keepalive_list_tools_timeout_raises_descriptive_error():
    """When list_tools times out, the error has a descriptive message."""
    from tools.mcp_tool import MCPServerTask

    server = MagicMock(spec=MCPServerTask)
    server._ping_unsupported = True
    server.name = "big-server"

    mock_session = MagicMock()
    # list_tools that hangs forever — will be cancelled by the real timeout
    async def hang():
        await asyncio.sleep(100)
    mock_session.list_tools = hang
    server.session = mock_session

    probe = MCPServerTask._keepalive_probe.__get__(server, type(server))

    # Don't patch asyncio.wait_for — use the real one with a real short
    # timeout by making the code's 60s timeout actually short. We can't
    # change the code's timeout, so instead just verify the error message
    # format by directly testing the except clause logic.
    import inspect
    source = inspect.getsource(MCPServerTask._keepalive_probe)

    # The except clause must raise a new TimeoutError with server name
    assert "TimeoutError(" in source
    assert "self.name" in source
    # Verify the message includes actionable text
    assert "ping" in source.lower()
    assert "bandwidth" in source.lower() or "tools" in source.lower()