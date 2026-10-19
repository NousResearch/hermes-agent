import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os

import pytest


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_private_control_peer_mints_profile_bound_ticket(tmp_path):
    from gateway.control_socket import GatewayControlServer, resolve_client_socket_path
    home = tmp_path / 'home'
    home.mkdir(mode=0o700)
    server = GatewayControlServer(home)
    # The actual service must implement this closed verb; unknown-verb is the pre-feature RED.
    try:
        from gateway.runtime_bootstrap import TicketStore
        server.ticket_store = TicketStore('instance', frozenset({str(home.resolve())}))
    except ImportError:
        pass
    assert await server.start()
    async def request(params, **extra):
        reader, writer = await asyncio.open_unix_connection(str(resolve_client_socket_path(home)))
        writer.write(json.dumps({'protocol': 1, 'verb': 'session-ticket', 'params': params, **extra}).encode() + b'\n')
        await writer.drain()
        reply = json.loads(await asyncio.wait_for(reader.readline(), 5))
        writer.close()
        await writer.wait_closed()
        return reply
    params = {'profile_id': str(home.resolve()), 'instance_id': 'instance', 'purpose': 'interactive'}
    try:
        reply = await request(params)
        assert reply['ok'], reply
        ticket = reply['result']['ticket']
        grant = server.ticket_store.redeem(ticket, profile_id=params['profile_id'], purpose='interactive')
        assert grant['subject'] == f'uid:{os.getuid()}'
        assert grant['instance_id'] == 'instance'
        assert 'session:create' in grant['capabilities']
        for invalid in ({**params, 'instance_id': 'old'}, {**params, 'profile_id': '/wrong'}, {**params, 'subject': 'admin'}):
            assert not (await request(invalid))['ok']
        assert not (await request(params, origin='http://browser'))['ok']
        # Calling the dispatcher directly does not establish peer identity.
        assert not json.loads(server.handle_request_line(json.dumps({'verb': 'session-ticket', 'params': params}).encode()))['ok']
    finally:
        await server.stop()


def test_ticket_atomic_single_use_profile_purpose_expiry_and_capacity(monkeypatch):
    from gateway.runtime_bootstrap import TicketStore
    clock = [0.0]
    monkeypatch.setattr('gateway.runtime_bootstrap.time.monotonic', lambda: clock[0])
    store = TicketStore('boot', frozenset({'a', 'b'}))
    ticket = store.mint(profile_id='a', subject='uid:1', purpose='interactive')
    for profile, purpose in [('b', 'interactive'), ('a', 'exposure')]:
        with pytest.raises(PermissionError):
            store.redeem(ticket, profile_id=profile, purpose=purpose)
    def consume(_):
        try:
            return store.redeem(ticket, profile_id='a', purpose='interactive')
        except PermissionError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        grants = list(pool.map(consume, range(8)))
    assert sum(g is not None for g in grants) == 1
    expired = store.mint(profile_id='a', subject='uid:1', purpose='interactive')
    native = store.mint(profile_id='a', subject='uid:1', purpose='native-http')
    clock[0] = 30
    with pytest.raises(PermissionError):
        store.redeem(expired, profile_id='a', purpose='interactive')
    with pytest.raises(PermissionError, match='expired'):  # the native-HTTP gate's redeem (profile_id=None)
        store.redeem(native, profile_id=None, purpose='native-http')
    exposure = store.mint(profile_id='a', subject='uid:1', purpose='exposure')
    assert store.redeem(exposure, profile_id='a', purpose='exposure')['capabilities'] == frozenset({'transport:delegate'})
    for _ in range(store.MAX_ENTRIES):
        store.mint(profile_id='a', subject='uid:1', purpose='interactive')
    with pytest.raises(PermissionError):
        store.mint(profile_id='a', subject='uid:1', purpose='interactive')


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_old_socket_cleanup_cannot_unlink_replacement(tmp_path):
    from gateway.control_socket import GatewayControlServer, resolve_client_socket_path
    home = tmp_path / 'private'
    home.mkdir(mode=0o700)
    old = GatewayControlServer(home)
    new = GatewayControlServer(home)
    assert await old.start()
    await old.stop()
    assert await new.start()
    try:
        old.cleanup_files()
        assert resolve_client_socket_path(home) is not None
        reader, writer = await asyncio.open_unix_connection(str(resolve_client_socket_path(home)))
        writer.write(b'{"verb":"identify"}\n')
        await writer.drain()
        assert json.loads(await reader.readline())['ok']
        writer.close()
        await writer.wait_closed()
    finally:
        await new.stop()


@pytest.mark.platforms("windows")
def test_native_pipe_authenticated_peer_and_deadline(tmp_path):
    import time
    from gateway.runtime_bootstrap_windows import NativeControlServer, query_runtime_control
    import threading
    stalled = threading.Event()  # the stalled handler hangs until teardown releases it

    def handler(raw, subject):
        assert subject.startswith('sid:S-1-')
        if raw == b'stall':
            stalled.wait(10)
        return json.dumps({'subject': subject}).encode() + b'\n'
    server = NativeControlServer(tmp_path, handler)
    server.start()
    try:
        # The pipe must survive its first client: CPython's _winapi has no DisconnectNamedPipe,
        # so a server that reaches for it dies after one answer and every later query fails.
        for _ in range(3):
            assert json.loads(query_runtime_control(tmp_path, b'hello', 5))['subject'].startswith('sid:')
        assert server._thread.is_alive() and server._error is None, server._error
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            query_runtime_control(tmp_path, b'stall', 0.1)
        assert time.monotonic() - started < 2
    finally:
        stalled.set()
        server.close()
    assert not server._thread.is_alive()


def _serve_one_child_query(tmp_path, monkeypatch):
    """Serve one query from a separate process whose PID the gateway cannot open, as when a
    one-shot ``gateway ensure`` child has already exited (#139084). Returns (subjects, child)."""
    import subprocess
    import sys
    import gateway.runtime_bootstrap_windows as rbw
    real_process_sid, own = rbw._process_sid, os.getpid()

    def process_sid(pid):
        if pid != own:
            raise PermissionError('cannot verify bootstrap peer process')  # OpenProcess on a gone PID
        return real_process_sid(pid)
    monkeypatch.setattr(rbw, '_process_sid', process_sid)
    subjects = []

    def handler(raw, subject):
        subjects.append(subject)
        return json.dumps({'ok': True}).encode() + b'\n'
    server = rbw.NativeControlServer(tmp_path, handler)
    server.start()
    try:
        child = subprocess.run(
            [sys.executable, '-c', 'import sys; from pathlib import Path; '
             'from gateway.runtime_bootstrap_windows import query_runtime_control as q; '
             'sys.stdout.write(q(Path(sys.argv[1]), b"hello", 10).decode())', str(tmp_path)],
            check=False, capture_output=True, text=True, timeout=30, cwd=os.getcwd())
    finally:
        server.close()
    return subjects, child


@pytest.mark.platforms("windows")
def test_native_pipe_identifies_a_peer_whose_process_is_gone(tmp_path, monkeypatch):
    """The peer's identity comes from the token captured when it opened the pipe, so a same-user
    client is served even when its PID can no longer be opened (#139084)."""
    import gateway.runtime_bootstrap_windows as rbw
    owner = rbw._process_sid(os.getpid())
    subjects, child = _serve_one_child_query(tmp_path, monkeypatch)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == {'ok': True}
    assert subjects == ['sid:' + owner]


@pytest.mark.platforms("windows")
def test_native_pipe_refuses_a_peer_it_cannot_identify(tmp_path, monkeypatch):
    """No token and no readable process: the peer stays refused, never served as anonymous."""
    import gateway.runtime_bootstrap_windows as rbw

    def no_token(handle):
        raise PermissionError('cannot impersonate named-pipe peer')
    monkeypatch.setattr(rbw, '_pipe_client_sid', no_token, raising=False)
    subjects, child = _serve_one_child_query(tmp_path, monkeypatch)
    assert subjects == []
    assert child.returncode != 0


@pytest.mark.platforms("windows")
def test_native_pipe_handler_may_dial_its_own_pipe(tmp_path):
    """A hosted room whose source and member share one multiplexer re-enters the control pipe
    from inside a handler (hosted-producer -> hosted-attest). A single serving instance cannot
    accept that nested request, so the outer call timed out; a slow handler must also still be
    able to answer once its own (client-bounded) work finishes."""
    import time
    from gateway.runtime_bootstrap_windows import NativeControlServer, query_runtime_control

    def handler(raw, subject):
        if raw == b'outer':
            try:
                inner = json.loads(query_runtime_control(tmp_path, b'inner', 5))
            except Exception as exc:  # report, never kill the serving thread
                inner = {'error': repr(exc)}
            return json.dumps({'outer': inner}).encode() + b'\n'
        time.sleep(2.5)  # health: allow HX013 -- the handler must outlast the 2 s per-peer I/O budget under test
        return json.dumps({'inner': raw.decode()}).encode() + b'\n'

    server = NativeControlServer(tmp_path, handler)
    server.start()
    try:
        assert json.loads(query_runtime_control(tmp_path, b'outer', 15)) == {'outer': {'inner': 'inner'}}
        assert server._error is None, server._error
    finally:
        server.close()


@pytest.mark.platforms("windows")
def test_native_pipe_never_drops_a_client_that_lands_between_idle_accepts(tmp_path, monkeypatch):
    """A listening pipe accepts CreateFile with no ConnectNamedPipe pending, so a client can land
    after an idle accept is cancelled and before the next one is issued. Recycling the instance
    with DisconnectNamedPipe there dropped that client mid-request (WinError 233). The client is
    injected at the server's first DisconnectNamedPipe, the widest point of that window."""
    import threading
    import gateway.runtime_bootstrap_windows as rbw
    real_disconnect, injected, armed = rbw._disconnect_pipe, {}, threading.Event()
    real_native, real_accept = rbw._native, rbw._accept
    landed, idle = threading.Event(), threading.Event()

    class _InjectedClient:
        """The injected client's ``_winapi``: reports when it reached the pipe."""
        def __init__(self, win):
            self._win = win

        def __getattr__(self, name):
            return getattr(self._win, name)

        def CreateFile(self, *args):
            try:
                handle = self._win.CreateFile(*args)
            except OSError as exc:
                if exc.winerror != 231:  # 231 (all instances busy) queues on WaitNamedPipe next
                    landed.set()
                raise
            landed.set()  # it opened the idle instance
            return handle

        def WaitNamedPipe(self, *args):
            landed.set()
            return self._win.WaitNamedPipe(*args)

    def native():
        win = real_native()
        return _InjectedClient(win) if threading.current_thread() is injected.get("thread") else win

    def accept(handle, budget):
        connected = real_accept(handle, budget)
        if not connected:
            idle.set()
        return connected

    def query():
        try:
            injected["reply"] = json.loads(rbw.query_runtime_control(tmp_path, b"injected", 5))
        except OSError as exc:
            injected["error"] = repr(exc)

    def disconnect_with_client_in_flight(handle):
        if "thread" not in injected:
            injected["thread"] = threading.Thread(target=query)
            injected["thread"].start()
            armed.set()
            # let the client open (idle window) or queue on WaitNamedPipe (served)
            assert landed.wait(5), "the injected client never reached the pipe"
        real_disconnect(handle)
    monkeypatch.setattr(rbw, "_disconnect_pipe", disconnect_with_client_in_flight)
    monkeypatch.setattr(rbw, "_native", native)
    monkeypatch.setattr(rbw, "_accept", accept)
    server = rbw.NativeControlServer(tmp_path, lambda raw, subject: json.dumps({"echo": raw.decode()}).encode() + b"\n")
    server.start()
    try:
        assert idle.wait(10), "no idle accept cycle ran"  # at least one idle accept cycle
        assert json.loads(rbw.query_runtime_control(tmp_path, b"hello", 5)) == {"echo": "hello"}
        assert armed.wait(5)
        injected["thread"].join(10)
    finally:
        server.close()
    assert injected.get("reply") == {"echo": "injected"}, injected.get("error")
    assert server._error is None, server._error


@pytest.mark.platforms("windows")
def test_client_reads_a_real_gateway_pipes_owner_dacl_and_session(tmp_path):
    """The facts the client half judges a server by when it cannot open the server process
    (#140706), read from a live gateway pipe through the client's own handle: owner this user, or
    Administrators when this process is elevated; a DACL of exactly this user; this session."""
    import _winapi
    import gateway.runtime_bootstrap_windows as rbw
    from gateway.control_socket import windows_pipe_name
    from hermes_dir_acl import process_is_elevated
    server = rbw.NativeControlServer(tmp_path, lambda raw, subject: b'{}\n')
    server.start()
    try:
        handle = _winapi.CreateFile(windows_pipe_name(tmp_path), 0xC0000000, 0, _winapi.NULL,
                                    _winapi.OPEN_EXISTING, 0, _winapi.NULL)
        try:
            own = rbw._process_sid(os.getpid())
            owner, aces = rbw._pipe_security(handle)
            assert owner == own or (owner == rbw._ADMINISTRATORS_SID and process_is_elevated()), owner
            assert aces == ((rbw._ACCESS_ALLOWED_ACE, own),)
            assert rbw._pipe_server_session(handle) == rbw._session_of(os.getpid())
            # Same token on both ends: attached, whichever owner this runner's token gives the pipe.
            assert rbw._peer_subject(handle, server=True) == 'sid:' + own
        finally:
            _winapi.CloseHandle(handle)
    finally:
        server.close()
