"""A PTY capture fault must not masquerade as a normal EOF."""

import errno
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.mark.platforms("posix")
def test_wait_failure_is_not_eof(monkeypatch):
    from hermes_cli.pty_bridge import PtyBridge

    bridge = PtyBridge.spawn(["/bin/sh", "-c", "exec sleep 30"])
    try:
        def fail(timeout):
            raise OSError(errno.EIO, "private-token-must-not-leak")

        monkeypatch.setattr(bridge._waiter, "wait", fail)
        with pytest.raises(OSError):
            bridge.read()
    finally:
        bridge.close()


@pytest.mark.asyncio
async def test_session_read_failure_closes_socket_as_error():
    from hermes_cli.pty_session import PtySession

    bridge = Mock()
    bridge.read.side_effect = OSError(errno.EIO, "private-token-must-not-leak")
    session = PtySession("capture-test", bridge, buffer_cap=1024, read_timeout=0.1)
    ws = AsyncMock()
    await session.attach(ws)
    # The drain owns transport errors; it must finish rather than leave a live
    # session whose background task has silently died.
    await session._drain()
    assert session.alive is False
    ws.close.assert_awaited_once_with(code=1011, reason="PTY output capture failed")
    bridge.close.assert_called_once()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("read_errno, expected", [(errno.EIO, None), (errno.EAGAIN, b"")])
def test_pty_normal_eof_and_idle_remain_distinct(monkeypatch, read_errno, expected):
    from hermes_cli.pty_bridge import PtyBridge
    bridge = PtyBridge.spawn(["/bin/sh", "-c", "exec sleep 30"])
    try:
        monkeypatch.setattr(bridge._waiter, "wait", lambda timeout: True)
        def read(fd, size):
            raise OSError(read_errno, "injected")
        with monkeypatch.context() as patcher:
            patcher.setattr("hermes_cli.pty_bridge.os.read", read)
            assert bridge.read() == expected
    finally:
        bridge.close()
    assert bridge.read() is None


@pytest.mark.platforms("windows")
def test_conpty_unexpected_error_is_not_eof():
    from hermes_cli.win_pty_bridge import WinPtyBridge
    proc = Mock()
    bridge = WinPtyBridge(proc)
    proc.read.side_effect = EOFError()
    assert bridge.read() is None
    proc.read.side_effect = RuntimeError("private-token")
    with pytest.raises(OSError, match="capture") as raised:
        bridge.read()
    assert "private-token" not in str(raised.value)
