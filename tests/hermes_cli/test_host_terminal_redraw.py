"""The resize pair that redraws a host terminal's program reaches it as two observed sizes."""
import asyncio
import queue
import shutil
import sys
from types import SimpleNamespace

import pytest

from hermes_cli import web_host_terminal_sessions
from hermes_cli.pty_session import PtySession
from hermes_cli.web_host_terminal_sessions import HostOwner, _pump_input, _resize
from tests.waiting import async_wait_until

# Programs that, like Node's stdout 'resize' event (Hermes' TUI redraws on it), report a size only
# when the one they read differs from the last, and read it after a busy event loop gets to it.
_PYTHON_SIZE_WATCHER = """
import os, signal, time
pending = []
signal.signal(signal.SIGWINCH, lambda *_: pending.append(1))
last = os.get_terminal_size(1)
print('READY', flush=True)
while True:
    time.sleep(0.2)
    if pending:
        pending.clear()
        size = os.get_terminal_size(1)
        if size != last:
            last = size
            print('RESIZE', size.lines, flush=True)
        print('WINCH', flush=True)
    print('TICK', flush=True)
"""
_NODE_SIZE_WATCHER = (
    "process.stdout.on('resize', () => process.stdout.write('RESIZE ' + process.stdout.rows + '\\n'));"
    "process.on('SIGWINCH', () => process.stdout.write('WINCH\\n'));"
    "process.stdout.write('READY\\n'); setInterval(() => process.stdout.write('TICK\\n'), 200)"
)


class Tab:
    """The attached browser tab: it keeps what the program prints and sends ``send``'s frames."""

    query_params: dict = {}

    def __init__(self):
        self.output = b""
        self.inbound = asyncio.Queue()

    def send(self, *frames):
        for frame in frames:
            self.inbound.put_nowait({"type": "websocket.receive", "text": frame})

    def close_tab(self):
        self.inbound.put_nowait({"type": "websocket.disconnect"})

    async def receive(self):
        return await self.inbound.get()

    async def send_text(self, _text):
        pass

    async def send_bytes(self, data):
        self.output += data

    async def close(self, code=1000, reason=None):
        pass


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
@pytest.mark.parametrize("watcher", ["python", "node"])
async def test_a_resize_back_waits_until_the_program_answered_the_step(tmp_path, watcher):
    """After a trimmed replay the browser steps the size one row away and back (browser-terminal.ts)
    so the foreground program redraws. Applied back to back, both SIGWINCHes can reach the program
    after the size is back where it was: it sees no change and does not redraw. The step back waits
    until the program has answered the first step, so a real program sees both sizes."""
    from hermes_cli.pty_bridge import PtyBridge

    if watcher == "node":
        node = shutil.which("node")
        if node is None:
            pytest.skip("node is not installed")
        argv = [node, "-e", _NODE_SIZE_WATCHER]
    else:
        argv = [sys.executable, "-u", "-c", _PYTHON_SIZE_WATCHER]

    async def remove(_token):
        pass

    bridge = PtyBridge.spawn(argv, cwd=str(tmp_path), cols=80, rows=24)
    session = PtySession("terminal", bridge, buffer_cap=1 << 16, read_timeout=0.01)
    owner = HostOwner(("loopback", "test"), tmp_path, "sh", str(tmp_path))
    tab = Tab()
    pump = None
    await session.start()
    try:
        assert await session.attach(tab)
        await async_wait_until(lambda: b"READY" in tab.output, timeout=10)
        start = len(tab.output)
        # The redraw's two frames, arriving together as they leave the browser.
        tab.send("\x1b[RESIZE:80;23]", "\x1b[RESIZE:80;24]")
        pump = asyncio.create_task(_pump_input(tab, SimpleNamespace(remove=remove), "terminal", session, owner))
        # The program prints TICK five times a second: give it three seconds.
        await async_wait_until(
            lambda: b"RESIZE 24" in tab.output[start:] or tab.output[start:].count(b"TICK") >= 15, timeout=10)
        seen = tab.output[start:]
        tab.close_tab()
        await asyncio.wait_for(pump, 5)
    finally:
        if pump is not None:
            pump.cancel()
        await session.close()
        await asyncio.to_thread(bridge.close)
    assert b"WINCH" in seen
    reported = [line for line in seen.split(b"\r\n") if line.startswith(b"RESIZE ")]
    assert reported == [b"RESIZE 23", b"RESIZE 24"]


class ScriptedBridge:
    """A PTY whose program prints only when the test says so; it records every size set on it."""

    def __init__(self):
        self.sizes = [(80, 24)]
        self.printed = queue.SimpleQueue()

    def read(self, timeout):
        try:
            return self.printed.get(timeout=timeout)
        except queue.Empty:
            return b""

    def size(self):
        return self.sizes[-1]

    def resize(self, cols, rows):
        self.sizes.append((cols, rows))

    def close(self):
        self.printed.put(None)

    def is_alive(self):
        return True


@pytest.mark.asyncio
async def test_the_step_back_is_applied_only_after_output_that_follows_the_step(monkeypatch):
    """Whatever the scheduler does, the size goes back only once the program has printed
    something after the step was applied; output read before the step does not count."""
    monkeypatch.setattr(web_host_terminal_sessions, "_RESIZE_ANSWER_SECONDS", 60)
    bridge = ScriptedBridge()
    session = PtySession("terminal", bridge, buffer_cap=1 << 16, read_timeout=0.01)
    tab = Tab()
    await session.start()
    try:
        assert await session.attach(tab)
        bridge.printed.put(b"before the step")
        await async_wait_until(lambda: b"before the step" in tab.output, timeout=10)
        undo = await _resize(session, tab, (80, 23), None)
        back = asyncio.create_task(_resize(session, tab, (80, 24), undo))
        # Nothing is printed, so the step back cannot be applied however long this waits.
        done, _ = await asyncio.wait({back}, timeout=0.2)
        assert not done
        assert bridge.sizes == [(80, 24), (80, 23)]
        bridge.printed.put(b"redrawn at 23 rows")
        await asyncio.wait_for(back, 10)
        assert bridge.sizes == [(80, 24), (80, 23), (80, 24)]
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("knows_size", [True, False], ids=["posix", "conpty"])
async def test_only_a_resize_that_undoes_the_previous_one_waits(knows_size):
    """A drag's resizes are applied at once; so are repeats of the same size and every resize on a
    bridge that cannot report its size."""
    current = [(80, 24)]
    waits = []

    async def wait_for_output(since, timeout):
        waits.append((since, timeout))

    session = SimpleNamespace(
        size=lambda: current[0] if knows_size else None, outputs=7, wait_for_output=wait_for_output,
        resize=lambda _ws, *, cols, rows: current.__setitem__(0, (cols, rows)))
    undo = None
    for step in [(80, 23), (80, 24), (100, 30), (100, 30), (80, 24), (90, 24)]:
        undo = await _resize(session, None, step, undo)
        assert current[0] == step
    assert waits == ([(7, web_host_terminal_sessions._RESIZE_ANSWER_SECONDS)] if knows_size else [])
