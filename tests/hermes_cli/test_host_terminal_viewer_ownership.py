"""Viewer takeover linearizes at the PTY sink, including off-loop writes."""
import asyncio
from contextlib import contextmanager, suppress
import os
from pathlib import Path
import select
import socket
import threading
import time

import pytest

from hermes_cli.pty_session import PtySession, WS_CLOSE_SUPERSEDED
from hermes_cli.web_host_terminal_sessions import HostOwner, HostTerminalRegistry
from tests.waiting import async_wait_until, wait_until


class Socket:
    def __init__(self):
        self.closed = []

    async def close(self, code, reason=None):
        self.closed.append(code)


def read_pty_through(fd, marker, timeout=3.0):
    """What a pseudo-terminal's slave reads up to and including ``marker``, written to the master
    after everything else. A write to the master returns before the slave can read it: the kernel
    hands the bytes over later (Linux from a worker thread), so earlier writes can still be on
    their way when the first read returns. They arrive in order, so the marker comes last."""
    got = bytearray()
    deadline = time.monotonic() + timeout
    while not got.endswith(marker):
        readable, _, _ = select.select([fd], [], [], max(0.0, deadline - time.monotonic()))
        if not readable:
            break
        got += os.read(fd, 1024)
    return bytes(got)


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
@pytest.mark.parametrize("pause_at_sink", [False, True])
async def test_takeover_revokes_waiting_workers_and_waits_for_an_admitted_write(tmp_path, monkeypatch, pause_at_sink):
    import pty
    import tty
    from hermes_cli import pty_bridge

    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    owner = HostOwner(("loopback", "test"), profile, "sh", str(tmp_path))
    master, slave = pty.openpty()
    tty.setraw(slave)
    os.set_blocking(master, False)
    bridge = pty_bridge.PtyBridge.__new__(pty_bridge.PtyBridge)
    bridge._fd, bridge._closed, bridge._fd_lock = master, False, threading.Lock()
    session = PtySession("terminal", bridge, buffer_cap=32, read_timeout=0.01)
    old, new = Socket(), Socket()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    real_write = os.write

    def pause():
        entered.set()
        assert release.wait(5), "write barrier was not released"

    def sink(fd, data):
        if pause_at_sink and fd == master and bytes(data) == b"OLD\n":
            pause()
        return real_write(fd, data)

    def old_fence(write):
        try:
            if not pause_at_sink:
                pause()
            return owner.admit(write)
        finally:
            finished.set()

    monkeypatch.setattr(pty_bridge.os, "write", sink)
    writing = takeover = None
    try:
        assert await session.attach(old)
        writing = asyncio.create_task(session.write(old, b"OLD\n", fence=old_fence))
        assert await asyncio.to_thread(entered.wait, 3)
        takeover = asyncio.create_task(session.attach(new))
        if pause_at_sink:
            # The sink already admitted OLD: takeover waits, with the event loop
            # still free to run the bridge timeout/cancellation cleanup.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(takeover), 0.05)
            release.set()
        assert await asyncio.wait_for(takeover, 3)
        assert await asyncio.wait_for(writing, 3) is False
        assert old.closed == [WS_CLOSE_SUPERSEDED]
        assert await session.write(new, b"NEW\n", fence=owner.admit)
        release.set()
        assert await asyncio.to_thread(finished.wait, 3)
        real_write(master, b"END\n")
        expected = b"OLD\nNEW\n" if pause_at_sink else b"NEW\n"
        assert await asyncio.to_thread(read_pty_through, slave, b"END\n") == expected + b"END\n"
        assert session.alive
    finally:
        release.set()
        for task in (writing, takeover):
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(finished.wait, 5)
        os.close(master)
        os.close(slave)


class InputSocket(Socket):
    """A viewer that sends ``frames`` and then reads until it is closed."""

    def __init__(self, *frames):
        super().__init__()
        self.inbound = asyncio.Queue()
        for frame in frames:
            self.inbound.put_nowait({"type": "websocket.receive", "bytes": frame})

    async def receive(self):
        return await self.inbound.get()

    async def close(self, code, reason=None):
        await super().close(code, reason)
        self.inbound.put_nowait({"type": "websocket.disconnect"})


class ConsoleInput:
    """The console behind the Windows bridge: keeps the text it is given, after ``on_write`` sees it."""

    pid = 1

    def __init__(self, on_write):
        self.on_write, self.written = on_write, []

    def write(self, text):
        self.on_write(text.encode())
        self.written.append(text)

    def terminate(self, force=False):
        pass


@contextmanager
def posix_pty(monkeypatch, on_write):
    """The POSIX bridge on a real pseudo-terminal. It returns a cancelled write at once, while its
    worker may still be inside the sink."""
    import pty
    import tty
    from hermes_cli import pty_bridge

    master, slave = pty.openpty()
    tty.setraw(slave)
    os.set_blocking(master, False)
    bridge = pty_bridge.PtyBridge.__new__(pty_bridge.PtyBridge)
    bridge._fd, bridge._closed, bridge._fd_lock = master, False, threading.Lock()
    real_write = os.write

    def sink(fd, data):
        if fd == master:
            on_write(bytes(data))
        return real_write(fd, data)

    monkeypatch.setattr(pty_bridge.os, "write", sink)
    try:
        yield bridge, lambda write: write is None
    finally:
        os.close(master)
        os.close(slave)


@contextmanager
def conpty(_monkeypatch, on_write):
    """The Windows bridge over a stand-in console. It gives a cancelled write a grace window to land,
    so the write is still running, cancelled, while the sink holds it."""
    from hermes_cli.win_pty_bridge import WinPtyBridge

    yield WinPtyBridge(ConsoleInput(on_write)), lambda write: write is not None and write.cancelling() > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [
    # The POSIX bridge imports fcntl and termios and writes to a real pseudo-terminal.
    pytest.param(posix_pty, id="posix-pty", marks=pytest.mark.platforms("posix")),
    pytest.param(conpty, id="conpty", marks=pytest.mark.platforms("any")),
])
async def test_takeover_during_an_admitted_input_write_keeps_the_shell(tmp_path, monkeypatch, terminal):
    """A takeover cancels the old socket's input write while the sink holds it and waits for the
    sink. The cancelled write returns False; the pump must neither close the shell nor let the old
    tab reconnect."""
    from types import SimpleNamespace
    from hermes_cli.web_host_terminal_sessions import _pump_input

    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    owner = HostOwner(("loopback", "test"), profile, "sh", str(tmp_path))
    entered, release = threading.Event(), threading.Event()

    def hold_old_input(data):
        if data == b"OLD\n":
            entered.set()
            assert release.wait(5), "write barrier was not released"

    removed = []

    async def remove(token):
        removed.append(token)

    old, new = InputSocket(b"OLD\n"), Socket()
    pump = takeover = None
    with terminal(monkeypatch, hold_old_input) as (bridge, write_cancelled):
        session = PtySession("terminal", bridge, buffer_cap=32, read_timeout=0.01)
        try:
            assert await session.attach(old)
            pump = asyncio.create_task(_pump_input(old, SimpleNamespace(remove=remove), "terminal", session, owner))
            assert await asyncio.to_thread(entered.wait, 3)
            takeover = asyncio.create_task(session.attach(new))
            # The takeover cancelled the write and waits for the sink; the old socket is still the viewer.
            await async_wait_until(lambda: write_cancelled(session._input_task), timeout=3)
            assert session._ws is old
            release.set()
            assert await asyncio.wait_for(takeover, 3)
            async with asyncio.timeout(3):
                await pump
            assert removed == [] and session.alive and not bridge._closed
            assert old.closed == [WS_CLOSE_SUPERSEDED]
            assert await session.write(new, b"NEW\n", fence=owner.admit)
        finally:
            release.set()
            for task in (pump, takeover):
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_host_reaper_releases_dead_shell_before_drain_observes_eof(tmp_path):
    import signal
    from hermes_cli.pty_bridge import PtyBridge

    bridge = PtyBridge.spawn(["/bin/sh", "-c", "trap '' HUP; sleep 60 & echo ready; exec sleep 60"], cwd=str(tmp_path))
    registry = HostTerminalRegistry()
    session = PtySession("terminal", bridge, buffer_cap=32, read_timeout=0.01)
    registry._sessions[session.key] = session
    registry.owners[session.key] = HostOwner(("loopback", "test"), tmp_path, "sh", str(tmp_path))
    viewer = Socket()
    try:
        async with asyncio.timeout(5):
            while b"ready" not in (await asyncio.to_thread(bridge.read, 0.05) or b""):
                await asyncio.sleep(0)
        assert await session.attach(viewer)
        os.kill(bridge.pid, signal.SIGKILL)
        await async_wait_until(lambda: not bridge.is_alive(), timeout=5)
        # The drain has not observed EOF. A helper can delay it indefinitely
        # on Linux; macOS can report EOF as soon as the leader exits.
        assert session.alive and session.attached
        await registry.reap_idle()
        assert not registry._sessions and not registry.owners
        assert bridge._closed
    finally:
        await registry.close_all()
        await asyncio.to_thread(bridge.close)


class ClosingTab:
    """A browser tab on the host terminal: it sends ``frames``, then whatever ``type`` adds, and its
    socket ends when ``close`` is called. Closing never cancels the route: like uvicorn, it only
    queues the disconnect for the next ``receive``. ``inbound`` holds what the server has not read
    yet; ``sent`` is every binary frame the tab received."""

    app = None
    query_params: dict = {}

    def __init__(self, *frames):
        self.inbound = asyncio.Queue()
        self.sent = []
        self.type(*frames)

    def type(self, *frames):
        for frame in frames:
            kind = "text" if isinstance(frame, str) else "bytes"
            self.inbound.put_nowait({"type": "websocket.receive", kind: frame})

    def close_tab(self):
        self.inbound.put_nowait({"type": "websocket.disconnect"})

    async def receive(self):
        return await self.inbound.get()

    async def send_text(self, _text):
        pass

    async def send_bytes(self, data):
        self.sent.append(data)

    async def close(self, code=1000, reason=None):
        pass


class UnreadInput:
    """The POSIX bridge while the foreground program is not reading: its input queue is full, so a
    write takes 0 bytes and then, with no deadline, waits for room until it is cancelled."""

    def __init__(self):
        self._closed = False
        self.waiting, self.cancelled = threading.Event(), threading.Event()

    async def write(self, data, *, timeout=10.0, fence=None):
        assert timeout is None, "host terminal input has no deadline"
        assert await asyncio.to_thread(fence, lambda: 0) == 0
        self.waiting.set()
        try:
            await asyncio.get_running_loop().create_future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    def resize(self, cols, rows):
        pass

    def close(self):
        self._closed = True

    def is_alive(self):
        return not self._closed


@contextmanager
def unread_input(_monkeypatch):
    bridge = UnreadInput()
    yield bridge, bridge.waiting, bridge.cancelled


@contextmanager
def unread_posix_pty(_monkeypatch):
    """The POSIX bridge on a real pseudo-terminal that nobody reads: a paste fills its input buffers
    and then waits for room."""
    import pty
    import tty
    from hermes_cli import pty_bridge

    master, slave = pty.openpty()
    tty.setraw(slave)
    os.set_blocking(master, False)
    bridge = pty_bridge.PtyBridge.__new__(pty_bridge.PtyBridge)
    bridge._fd, bridge._closed, bridge._fd_lock = master, False, threading.Lock()
    waiting, cancelled = threading.Event(), threading.Event()
    wait_writable = bridge._wait_writable

    async def watched(timeout):
        waiting.set()
        try:
            return await wait_writable(timeout)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    bridge._wait_writable = watched
    try:
        yield bridge, waiting, cancelled
    finally:
        os.close(master)
        os.close(slave)


def keys_filling_the_input_queue(extra=0, *, text=False):
    """``(frames, filled)``: keystroke frames whose first ``filled`` fill the host terminal's input
    queue, its frame bound of frames that together reach its byte bound, then ``extra`` more. Each
    frame repeats one letter; the next frame has the next letter. With ``text``, the frames are text
    frames, as a browser sends them, of 4-byte emoji: each has a quarter as many characters as UTF-8
    bytes."""
    from hermes_cli import web_host_terminal_sessions as host_sessions

    filled = host_sessions._INPUT_QUEUE_FRAMES
    size = host_sessions._INPUT_QUEUE_BYTES // filled
    assert filled * (size + host_sessions._FRAME_OVERHEAD_BYTES) >= host_sessions._INPUT_QUEUE_BYTES
    if text:
        return [chr(0x1F600 + n % 26) * (size // 4) for n in range(filled + extra)], filled
    return [bytes([65 + n % 26]) * size for n in range(filled + extra)], filled


def as_written(frame) -> bytes:
    """What the program reads for a frame: a text frame's UTF-8, a binary frame's bytes."""
    return frame.encode() if isinstance(frame, str) else frame


def test_text_frames_count_their_utf8_bytes_against_the_input_queue():
    """A text frame is written as UTF-8, so it counts its UTF-8 bytes: a non-ASCII paste fills the
    input queue at the same bound as the bytes it will write, not at a multiple of it."""
    from hermes_cli import web_host_terminal_sessions as host_sessions
    from hermes_cli.web_server_chat import _RESIZE_RE

    keys, _ = keys_filling_the_input_queue(text=True)
    inbox = host_sessions._InputQueue()
    for key in keys:
        assert not inbox.full(), "the input queue filled before its byte bound"
        inbox.put_nowait({"type": "websocket.receive", "text": key})
    assert inbox.full(), "text frames filled the input queue by characters, not bytes"
    assert inbox.queued == sum(len(as_written(key)) + host_sessions._FRAME_OVERHEAD_BYTES for key in keys)
    while not inbox.empty():
        inbox.get_nowait()
    assert inbox.queued == 0
    # A frame counts exactly what ``_input_runs`` writes for it.
    frame = {"type": "websocket.receive", "text": keys[0]}
    written, = host_sessions._input_runs([frame], _RESIZE_RE)
    assert host_sessions._frame_cost(frame) == len(written) + host_sessions._FRAME_OVERHEAD_BYTES


def test_a_paste_or_keys_alone_do_not_fill_the_input_queue():
    """The reader stops taking frames only once the input queue holds both its byte bound and 64
    frames. One paste over the byte bound, or many keys, leave it reading the socket, where the
    server's keepalive pongs arrive."""
    from hermes_cli import web_host_terminal_sessions as host_sessions

    inbox = host_sessions._InputQueue()
    for frame in [b"P" * 2 * host_sessions._INPUT_QUEUE_BYTES, *[b"k"] * 48]:
        inbox.put_nowait({"type": "websocket.receive", "bytes": frame})
        assert not inbox.full(), "a paste over the byte bound filled the input queue"
    while not inbox.empty():
        inbox.get_nowait()
    for _ in range(1024):
        inbox.put_nowait({"type": "websocket.receive", "bytes": b"k"})
        assert not inbox.full(), "keys filled the input queue by count alone"


@pytest.mark.asyncio
@pytest.mark.parametrize("fill_inbox", [False, True], ids=["quiet", "full-inbox"])
@pytest.mark.parametrize("terminal", [
    pytest.param(unread_posix_pty, id="posix-pty", marks=pytest.mark.platforms("posix")),
    pytest.param(unread_input, id="unread-input", marks=pytest.mark.platforms("any")),
])
async def test_closing_the_tab_during_a_waiting_paste_detaches_and_keeps_the_shell(
    tmp_path, monkeypatch, terminal, fill_inbox,
):
    """A paste into a program that is not reading (``sleep``, ``npm install``) waits for it with no
    deadline. When the tab closes meanwhile, the wait is cancelled and the shell is detached, as on
    any disconnect: it keeps running, retained for a reattach, and the reaper retires it once its
    retention runs out. A full input queue behind the paste (keys typed while it waited) does not
    hide the close."""
    from hermes_cli import web_host_terminal_sessions as host_sessions

    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    owner = HostOwner(("loopback", "test"), profile, "sh", str(tmp_path))
    registry = HostTerminalRegistry()
    # 1 MiB: more than a pseudo-terminal buffers anywhere between its master and the program.
    paste = b"".join(b"%05d %s\n" % (n, b"x" * 57) for n in range(16 * 1024))
    handler = None
    with terminal(monkeypatch) as (bridge, waiting, cancelled):
        session = PtySession("terminal", bridge, buffer_cap=32, read_timeout=0.01)
        registry._sessions[session.key] = session
        registry.owners[session.key] = owner

        async def admit(_ws, _registry):
            return session, owner

        monkeypatch.setattr(host_sessions, "get_host_terminals", lambda _app: registry)
        monkeypatch.setattr(host_sessions, "_admit", admit)
        tab = ClosingTab(paste)
        try:
            handler = asyncio.create_task(host_sessions.host_terminal(tab))
            assert await asyncio.to_thread(waiting.wait, 5), "the paste never waited for the program"
            tab.type(*(keys_filling_the_input_queue()[0] if fill_inbox else ()))
            tab.close_tab()
            async with asyncio.timeout(5):
                await handler
            assert cancelled.is_set()
            assert session.alive and not bridge._closed
            assert registry._sessions.get(session.key) is session
            assert not session.attached and session._ws is None
            assert session.last_detached_at is not None
            assert registry.expired(session, session.last_detached_at + registry._ttl + 1)
        finally:
            if handler is not None:
                handler.cancel()
                await asyncio.gather(handler, return_exceptions=True)
            registry._sessions.clear()
            registry.owners.clear()
            await registry.close_all()


class BrowserTab:
    """A browser's end of the host terminal socket: a real WebSocket client on a plain TCP socket."""

    def __init__(self, port):
        from websockets.client import ClientProtocol
        from websockets.protocol import State
        from websockets.uri import parse_uri

        self.protocol = ClientProtocol(parse_uri(f"ws://127.0.0.1:{port}/terminal"))
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.protocol.send_request(self.protocol.connect())
        self._flush()
        while self.protocol.state is State.CONNECTING:
            self.protocol.receive_data(self.sock.recv(65536))
        assert self.protocol.state is State.OPEN, self.protocol.handshake_exc

    def _flush(self):
        self.sock.sendall(b"".join(self.protocol.data_to_send()))

    def send(self, *frames):
        for frame in frames:
            self.protocol.send_binary(frame)
        self._flush()

    def close(self):
        """Go away as a closing tab does: a close frame, then the TCP connection. A browser has read
        everything the server sent, so the connection ends with a FIN, not a reset."""
        self.protocol.send_close(1001)
        self._flush()
        self.sock.setblocking(False)
        with suppress(BlockingIOError):
            while self.sock.recv(65536):
                pass
        self.sock.close()


class AnsweringTab(BrowserTab):
    """A tab whose browser answers the server's WebSocket pings as they arrive, on its own thread, as a
    browser does whatever the page is doing. ``pings`` counts the pings it answered."""

    def __init__(self, port):
        super().__init__(port)
        self.protocol.events_received()  # the handshake response
        self.pings, self._lock, self._stop = 0, threading.Lock(), threading.Event()
        self.sock.settimeout(0.05)
        self._thread = threading.Thread(target=self._answer, daemon=True)
        self._thread.start()

    def _answer(self):
        from websockets.frames import Opcode

        while not self._stop.is_set():
            try:
                data = self.sock.recv(65536)
            except TimeoutError:
                continue
            except OSError:
                return
            with self._lock:
                if data:
                    self.protocol.receive_data(data)
                else:
                    self.protocol.receive_eof()
                self.pings += sum(getattr(event, "opcode", None) is Opcode.PING
                                  for event in self.protocol.events_received())
                with suppress(OSError):
                    self._flush()  # the pongs
            if not data:
                return

    def send(self, *frames):
        with self._lock:
            super().send(*frames)

    def _flush(self):
        # The socket's short timeout is for the answering thread's reads; a paste can take longer
        # than that to send.
        data = memoryview(b"".join(self.protocol.data_to_send()))
        deadline = time.monotonic() + 10
        while data:
            try:
                data = data[self.sock.send(data):]
            except TimeoutError:
                if time.monotonic() > deadline:
                    raise

    @property
    def closed_by_server(self):
        """Whether the server's close frame has arrived, as a browser sees it before the route returns."""
        return self.protocol.close_rcvd is not None

    def close(self):
        from websockets.protocol import State

        self._stop.set()
        self._thread.join(5)
        if self.protocol.state is State.OPEN:
            super().close()
        else:
            self.sock.close()


@contextmanager
def served_host_terminal(monkeypatch, session, owner, ws_ping=(None, None)):
    """``host_terminal`` for ``session`` behind real uvicorn and Starlette on a loopback port, with
    uvicorn's default WebSocket protocol and the loopback Webapp's settings (no WebSocket pings), or
    a non-loopback bind's ``ws_ping`` (interval, timeout). Yields the port, the frames the route has
    taken from the socket, and an Event set once the route returns."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import WebSocketRoute
    from hermes_cli import web_host_terminal_sessions as host_sessions

    registry = HostTerminalRegistry()
    taken, finished = [], threading.Event()

    async def admit(_ws, _registry):
        return session, owner

    monkeypatch.setattr(host_sessions, "get_host_terminals", lambda _app: registry)
    monkeypatch.setattr(host_sessions, "_admit", admit)

    async def route(ws):
        await ws.accept()
        try:
            await host_sessions.host_terminal(ws)
        finally:
            finished.set()

    app = Starlette(routes=[WebSocketRoute("/terminal", route)])

    async def counting(scope, receive, send):
        async def counted():
            message = await receive()
            if message["type"] == "websocket.receive":
                taken.append(message)
            return message

        await app(scope, counted, send)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(
        counting, lifespan="off", log_level="error", ws_ping_interval=ws_ping[0], ws_ping_timeout=ws_ping[1],
        # A route that never returns (the bug) is cancelled instead of holding shutdown forever.
        timeout_graceful_shutdown=1,
    ))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        wait_until(lambda: server.started or not thread.is_alive(), what="uvicorn to start")
        assert server.started
        yield sock.getsockname()[1], taken, finished
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        asyncio.run(registry.close_all())
        assert not thread.is_alive()


@pytest.mark.parametrize("terminal", [
    pytest.param(unread_posix_pty, id="posix-pty", marks=pytest.mark.platforms("posix")),
    pytest.param(unread_input, id="unread-input", marks=pytest.mark.platforms("any")),
])
def test_a_tab_closed_behind_more_keys_than_the_input_queue_holds_is_detached(tmp_path, monkeypatch, terminal):
    """Keys typed while a paste waits on a program that is not reading fill the input queue and
    then some, so the tab's close is unread behind them when it comes, and the browser drops the
    connection. Over a real socket the shell is still detached within a few seconds: it keeps
    running, retained for a reattach."""
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    owner = HostOwner(("loopback", "test"), profile, "sh", str(tmp_path))
    # 1 MiB: more than a pseudo-terminal buffers anywhere between its master and the program.
    paste = b"".join(b"%05d %s\n" % (n, b"x" * 57) for n in range(16 * 1024))
    keys, fill = keys_filling_the_input_queue(17)
    with terminal(monkeypatch) as (bridge, waiting, cancelled):
        session = PtySession("terminal", bridge, buffer_cap=32, read_timeout=0.01)
        with served_host_terminal(monkeypatch, session, owner) as (port, taken, finished):
            tab = BrowserTab(port)
            tab.send(paste)
            assert waiting.wait(5), "the paste never waited for the program"
            tab.send(*keys)
            # The paste, a full queue, and one more frame the reader holds with no room for it.
            wait_until(lambda: len(taken) >= fill + 2, timeout=5, what="the reader to fill the input queue")
            tab.close()
            assert finished.wait(10), "the shell stayed attached after its tab closed"
        assert cancelled.is_set()
        assert session.alive and not bridge._closed
        assert not session.attached and session._ws is None


@pytest.mark.parametrize("typed", [
    # Far more frames than the WebSocket library buffers unread (32) and than 64 queued frames.
    [bytes([65 + n % 26]) for n in range(1024)],
    # A paste over the input queue's byte bound (one text frame from the browser), then more keys
    # than the WebSocket library buffers unread, but fewer frames than 64.
    [b"P" * 2 * 1024 * 1024, *(bytes([97 + n % 26]) for n in range(48))],
], ids=["keys", "paste-then-keys"])
def test_a_live_tab_typing_behind_a_waiting_write_answers_keepalive_pings(tmp_path, monkeypatch, typed):
    """A non-loopback Webapp pings its sockets (``web_server._build_uvicorn_server``) and closes one
    whose pong does not come back in time. Keys or a paste typed while a write waits on a slow
    program must not stop the server reading the socket, where the pongs arrive: the tab, answering
    every ping, stays attached through several ping timeouts, and everything reaches the program in
    order once it reads."""
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    owner = HostOwner(("loopback", "test"), profile, "sh", str(tmp_path))
    bridge = SlowAccept()
    session = PtySession("terminal", bridge, buffer_cap=32, read_timeout=0.01)
    with served_host_terminal(monkeypatch, session, owner, ws_ping=(1.0, 2.0)) as (port, taken, finished):
        tab = AnsweringTab(port)
        try:
            tab.send(b"A")
            assert bridge.entered.wait(5), "the first write never started"
            tab.send(*typed)
            # Five pings a second apart, each given two seconds for its pong, all while the program
            # holds its first write: the socket stays unread across several ping timeouts.
            wait_until(lambda: tab.pings >= 5 or tab.closed_by_server or finished.is_set(), timeout=30,
                       what="five keepalive pings")
            assert not tab.closed_by_server and not finished.is_set() and session.attached, \
                "the keepalive closed a live tab"
            assert len(taken) == 1 + len(typed), "the reader stopped reading the socket"
            bridge.release.set()
            expected = b"A" + b"".join(typed)
            wait_until(lambda: bytes(bridge.written) == expected, timeout=5, what="every key to reach the program")
            assert session.attached and not finished.is_set()
        finally:
            bridge.release.set()
            tab.close()
        assert finished.wait(10), "the shell stayed attached after its tab closed"
    assert session.alive and not session.attached


class SlowAccept:
    """A program that reads every byte, but not until ``release``: slow, not stopped. The first write
    takes its bytes and then waits; later writes land at once."""

    def __init__(self):
        self._closed = False
        self.written = bytearray()
        self.entered, self.release = threading.Event(), threading.Event()
        self._gated = True

    async def write(self, data, *, timeout=10.0, fence=None):
        assert timeout is None, "host terminal input has no deadline"
        assert await asyncio.to_thread(fence, lambda: len(data)) == len(data)
        if self._gated:
            self._gated = False
            self.entered.set()
            assert await asyncio.to_thread(self.release.wait, 30), "the slow program was not released"
        self.written.extend(data)
        return True

    def resize(self, cols, rows):
        pass

    def close(self):
        self._closed = True

    def is_alive(self):
        return not self._closed


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [False, True], ids=["binary", "non-ascii-text"])
async def test_a_live_tab_typing_past_a_full_input_queue_is_held_back_and_loses_nothing(tmp_path, monkeypatch, text):
    """Keys typed while a write waits on a slow program queue up to the input bound. Past it the
    reader takes no more frames, however many the tab sends (they wait in the socket, which pushes
    back on the browser), and checks only that the tab is still there, which leaves a live tab
    attached and writes nothing to its terminal. Once the program reads, every key reaches it in
    order. Text frames count the UTF-8 bytes they are written as."""
    from hermes_cli import web_host_terminal_sessions as host_sessions

    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    owner = HostOwner(("loopback", "test"), profile, "sh", str(tmp_path))
    registry = HostTerminalRegistry()
    bridge = SlowAccept()
    session = PtySession("terminal", bridge, buffer_cap=32, read_timeout=0.01)
    registry._sessions[session.key] = session
    registry.owners[session.key] = owner

    async def admit(_ws, _registry):
        return session, owner

    monkeypatch.setattr(host_sessions, "get_host_terminals", lambda _app: registry)
    monkeypatch.setattr(host_sessions, "_admit", admit)
    typed, fill = keys_filling_the_input_queue(3 * 16, text=text)
    # The first write holds b"A"; the reader queues ``fill`` frames and holds one more with no room.
    unread = len(typed) - fill - 1
    tab = ClosingTab(b"A")
    handler = None
    try:
        handler = asyncio.create_task(host_sessions.host_terminal(tab))
        assert await asyncio.to_thread(bridge.entered.wait, 5), "the first write never started"
        tab.type(*typed)
        await async_wait_until(lambda: tab.inbound.qsize() <= unread, timeout=5,
                               what="the reader to fill the input queue")
        assert tab.inbound.qsize() == unread, "the reader took frames past the input queue"
        replayed = len(tab.sent)
        await async_wait_until(lambda: len(tab.sent) > replayed, timeout=5,
                               what="a check on the tab behind the full input queue")
        assert tab.inbound.qsize() == unread, "the reader took frames while it waited for room"
        assert session.attached and session._ws is tab, "a live tab was detached"
        assert b"".join(tab.sent[replayed:]) == b"", "the check wrote to the tab's terminal"
        bridge.release.set()
        expected = b"A" + b"".join(map(as_written, typed))
        await async_wait_until(lambda: bytes(bridge.written) == expected, timeout=5,
                               what="every keystroke to reach the slow program")
        tab.close_tab()
        async with asyncio.timeout(5):
            await handler
        assert session.alive and not bridge._closed
        assert registry._sessions.get(session.key) is session
        assert not session.attached and session._ws is None
    finally:
        bridge.release.set()
        if handler is not None:
            handler.cancel()
            await asyncio.gather(handler, return_exceptions=True)
        registry._sessions.clear()
        registry.owners.clear()
        await registry.close_all()
