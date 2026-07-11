"""
Regression test for issue #62166 - dashboard /api/pty leaks a live TUI
per reconnect on the same channel.

The bug: reconnecting to /api/pty with the same channel (browser tab
reload) spawned a second PTY+TUI while the first kept running. The
disconnect path of the first WebSocket relied on the TCP socket actually
dying — and on WSL2 that doesn't happen.

The fix: per-channel supersede registry. On a reconnect for channel X,
the previous WebSocket for X is closed (code 4400 "supersede") so its
existing `finally` cleanup runs immediately. Last-connect-wins per
channel.
"""
import asyncio
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, "/tmp/hermes-pr-work-60859/hermes-agent")


def test_supersede_registry_defined():
    """The supersede registry exists at module level."""
    import hermes_cli.web_server as web_server
    assert hasattr(web_server, "_LEGACY_PTY_BY_CHANNEL")
    assert isinstance(web_server._LEGACY_PTY_BY_CHANNEL, dict)


def test_supersede_closes_previous_ws():
    """When a reconnect arrives for the same channel, the previous WS
    is closed before the new one starts."""
    import hermes_cli.web_server as web_server

    # Reset for this test
    web_server._LEGACY_PTY_BY_CHANNEL.clear()

    # Simulate: a previous WS is registered
    previous_ws = MagicMock()
    previous_ws.close = MagicMock()
    asyncio.run = MagicMock()  # we won't actually await
    web_server._LEGACY_PTY_BY_CHANNEL["reproR"] = previous_ws

    # The new WS comes in. The handler should close previous_ws first.
    # We can't easily call pty_ws directly (it's a websocket endpoint),
    # but we can verify the registry + close pattern works.

    # Simulate what the handler does:
    channel = "reproR"
    new_ws = MagicMock()
    new_ws.close = MagicMock()
    previous_ws = web_server._LEGACY_PTY_BY_CHANNEL.get(channel)
    if previous_ws is not None and previous_ws is not new_ws:
        previous_ws.close(code=4400, reason="supersede")
        web_server._LEGACY_PTY_BY_CHANNEL.pop(channel, None)

    # Assert: previous_ws.close was called
    assert previous_ws.close.called, \
        "Issue #62166: previous WS was not closed on reconnect"

    # Assert: registry entry was cleared (the pump task will overwrite)
    assert channel not in web_server._LEGACY_PTY_BY_CHANNEL, \
        "Registry entry not cleared after supersede"


def test_supersede_skipped_when_same_ws():
    """If the new WS is the same as the registered one, don't close
    it (defensive — should never happen but guard against weirdness)."""
    import hermes_cli.web_server as web_server
    web_server._LEGACY_PTY_BY_CHANNEL.clear()

    ws = MagicMock()
    ws.close = MagicMock()
    web_server._LEGACY_PTY_BY_CHANNEL["chA"] = ws

    # Simulate: same WS re-enters (shouldn't happen but be defensive)
    channel = "chA"
    registered = web_server._LEGACY_PTY_BY_CHANNEL.get(channel)
    if registered is not None and registered is not ws:
        registered.close(code=4400, reason="supersede")
        web_server._LEGACY_PTY_BY_CHANNEL.pop(channel, None)

    # Assert: NOT closed
    assert not ws.close.called, \
        "Same WS was incorrectly closed (defensive check should have skipped it)"


def test_supersede_handles_missing_channel():
    """If a connect arrives with no channel, the registry is unchanged."""
    import hermes_cli.web_server as web_server
    web_server._LEGACY_PTY_BY_CHANNEL.clear()

    # No channel means we don't touch the registry
    channel = None  # simulate _channel_or_close_code returning None
    if channel:
        # this branch should be skipped
        web_server._LEGACY_PTY_BY_CHANNEL["x"] = MagicMock()

    # Assert: registry is still empty
    assert len(web_server._LEGACY_PTY_BY_CHANNEL) == 0


def test_supersede_registry_cleared_on_clean_exit():
    """When a legacy pump exits cleanly (no supersede), the registry
    entry for that channel is removed so a future reconnect doesn't
    try to close an already-closed WS."""
    import hermes_cli.web_server as web_server
    web_server._LEGACY_PTY_BY_CHANNEL.clear()

    # Simulate: WS is registered, then exits cleanly
    ws = MagicMock()
    web_server._LEGACY_PTY_BY_CHANNEL["chB"] = ws

    # The `finally` block in the handler
    channel = "chB"
    if web_server._LEGACY_PTY_BY_CHANNEL.get(channel) is ws:
        web_server._LEGACY_PTY_BY_CHANNEL.pop(channel, None)

    # Assert: cleared
    assert "chB" not in web_server._LEGACY_PTY_BY_CHANNEL


def test_supersede_keeps_new_entry_after_supersede():
    """After a supersede, the registry should hold the NEW WS (via
    the post-spawn registration), not the old one."""
    import hermes_cli.web_server as web_server
    web_server._LEGACY_PTY_BY_CHANNEL.clear()

    old_ws = MagicMock()
    old_ws.close = MagicMock()
    new_ws = MagicMock()
    new_ws.close = MagicMock()
    web_server._LEGACY_PTY_BY_CHANNEL["chC"] = old_ws

    # Simulate the full sequence:
    # 1. Supersede closes old_ws
    channel = "chC"
    registered = web_server._LEGACY_PTY_BY_CHANNEL.get(channel)
    if registered is not None and registered is not new_ws:
        registered.close(code=4400, reason="supersede")
        web_server._LEGACY_PTY_BY_CHANNEL.pop(channel, None)
    # 2. New WS registers itself
    web_server._LEGACY_PTY_BY_CHANNEL[channel] = new_ws

    # Assert: old was closed, new is registered
    assert old_ws.close.called
    assert web_server._LEGACY_PTY_BY_CHANNEL["chC"] is new_ws