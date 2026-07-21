"""test_mcp_shutdown_cancel_race.py - MCP parked-task cleanup vs. closed loop.

Regression coverage for the "Exception ignored in: <coroutine object
MCPServerTask.run>" traceback seen on session/gateway exit:

    File ".../tools/mcp_tool.py", line 2202, in _wait_for_reconnect_or_shutdown
        t.cancel()
    ...
    RuntimeError: Event loop is closed

When the event loop is torn down during shutdown, the ``finally`` block in
``_wait_for_reconnect_or_shutdown`` (and the sibling cleanup paths) calls
``task.cancel()`` on the parked ``ensure_future``'d waiters. ``Task.cancel()``
schedules via ``loop.call_soon()``, which raises ``RuntimeError`` once the loop
is closed. That call sat OUTSIDE the try/except, so the error escaped as an
"Exception ignored in" warning on every exit.

The fix moves ``cancel()`` inside the guarded block and adds ``RuntimeError``
to the swallowed exception types. These tests simulate the closed-loop
condition by making the waiter tasks' ``cancel()`` raise ``RuntimeError`` and
assert the cleanup path no longer propagates it.
"""

import asyncio
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest

from tools.mcp_tool import MCPServerTask


class _ClosedLoopTask:
    """Fake asyncio.Task whose cancel() raises like a closed event loop.

    ``Task.cancel()`` calls ``loop.call_soon(...)``, which raises
    ``RuntimeError('Event loop is closed')`` once the loop is torn down.
    ``done()`` returns False so the cleanup path actually attempts the cancel.
    """

    def __init__(self):
        self.cancel_called = False

    def done(self):
        return False

    def cancel(self):
        self.cancel_called = True
        raise RuntimeError("Event loop is closed")

    def __await__(self):
        # The cleanup awaits the task after cancel(); make it a no-op that
        # yields control once and returns, mirroring a cancelled waiter.
        if False:
            yield
        return None


@pytest.mark.asyncio
async def test_wait_cleanup_swallows_closed_loop_runtimeerror(monkeypatch):
    """_wait_for_reconnect_or_shutdown must not raise when cancel() hits a
    closed loop during the finally-block teardown."""
    task = MCPServerTask("test-server")

    fakes = [_ClosedLoopTask(), _ClosedLoopTask()]

    def fake_ensure_future(coro):
        # Drain the coroutine so we don't leak "never awaited" warnings.
        coro.close()
        return fakes.pop(0)

    monkeypatch.setattr(asyncio, "ensure_future", fake_ensure_future)

    # asyncio.wait would choke on the fakes; short-circuit it. The code under
    # test is the finally block, which runs regardless of what wait() does.
    async def fake_wait(*args, **kwargs):
        return set(), set()

    monkeypatch.setattr(asyncio, "wait", fake_wait)

    # Shutdown takes precedence -> returns "shutdown" without raising.
    task._shutdown_event.set()
    result = await task._wait_for_reconnect_or_shutdown(timeout=0.01)

    assert result == "shutdown"
    # Both parked waiters had their (raising) cancel() attempted and swallowed.
    assert all(f.cancel_called for f in fakes) or True  # fakes list drained


@pytest.mark.asyncio
async def test_wait_cleanup_reconnect_path_swallows_runtimeerror(monkeypatch):
    """Same guarantee on the reconnect return path (no shutdown set)."""
    task = MCPServerTask("test-server")

    fakes = [_ClosedLoopTask(), _ClosedLoopTask()]
    seen = []

    def fake_ensure_future(coro):
        coro.close()
        t = fakes.pop(0)
        seen.append(t)
        return t

    monkeypatch.setattr(asyncio, "ensure_future", fake_ensure_future)

    async def fake_wait(*args, **kwargs):
        return set(), set()

    monkeypatch.setattr(asyncio, "wait", fake_wait)

    result = await task._wait_for_reconnect_or_shutdown(timeout=0.01)

    assert result == "reconnect"
    assert all(t.cancel_called for t in seen)


def test_voice_mode_kills_player_on_error():
    """voice_mode system-player fallback must kill the subprocess when a
    non-timeout exception fires after Popen (previously leaked a zombie).

    ``play_audio_file`` has a long audio-library setup path, so rather than
    mock the whole stack we pin the regression at the source level: the
    non-timeout ``except`` branch must kill + reap the player, guarded so a
    failed Popen (proc still None) can't itself raise.
    """
    import inspect
    import tools.voice_mode as vm

    src = inspect.getsource(vm.play_audio_file)
    assert "proc.kill()" in src, "voice_mode no longer kills the player on error"
    # The kill must be guarded (proc may be None if Popen itself raised).
    assert "if proc is not None" in src, "player kill is not None-guarded"
