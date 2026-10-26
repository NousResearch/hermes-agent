"""A shared socket's frame naming a profile that hot-serve is still starting.

``serve_profile_runtime`` reserves the profile's registry slot (``SessionAuthorities.add(home, None)``)
before ``initialize_session_authority`` has built its authority. On main a host-scoped frame naming
that profile walked into the reserved slot and raised AttributeError, which the socket answered with
JSON-RPC -32603 "internal error". A frame for a profile that was already running matched its own slot
first and never reached the reserved one.

The real ``handle_ws`` lanes and a real multiplexing runtime; only the starting profile's
``initialize_session_authority`` is held, or failed, to keep the start in progress.
"""
import asyncio
import json
from pathlib import Path

import pytest
import pytest_asyncio

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


class _Socket:
    """The browser end of one WebSocket: frames in, replies collected by id and in written order."""

    def __init__(self, authority):
        self.scope = {'hermes.session_authority': authority}
        self.inbound = asyncio.Queue()
        self.replies = {}
        self.order = []
        self.arrived = asyncio.Event()
        self.task: asyncio.Task | None = None  # the handle_ws task serving this socket

    async def accept(self, **_kw):
        pass

    async def send_text(self, line):
        for part in line.splitlines():
            frame = json.loads(part)
            if 'id' in frame:
                self.replies[frame['id']] = frame
                self.order.append(frame['id'])
                self.arrived.set()

    async def receive_text(self):
        frame = await self.inbound.get()
        if frame is None:
            from tui_gateway.ws import _WebSocketDisconnect
            raise _WebSocketDisconnect()
        return json.dumps(frame)

    async def close(self, **_kw):
        pass

    async def reply(self, request_id, timeout=30):
        async def arrived():
            while request_id not in self.replies:
                self.arrived.clear()
                await self.arrived.wait()
            return self.replies[request_id]
        return await asyncio.wait_for(arrived(), timeout)


@pytest_asyncio.fixture
async def rig(tmp_path, monkeypatch):
    """Default and ``alpha`` served, ``beta`` held mid-start; ``connect`` opens a ticketed socket."""
    from types import SimpleNamespace

    from gateway import session_authority
    from gateway.run_runtime import initialize_gateway_runtime, serve_profile_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import authority_for_profile_id
    from tui_gateway import server
    from tui_gateway import ws as ws_mod

    monkeypatch.setattr(ws_mod, '_start_backend_liveness', lambda: None)
    monkeypatch.setattr(server, '_ensure_skin_watcher', lambda: None)
    monkeypatch.setattr(server, '_ensure_lease_watcher', lambda: None)
    monkeypatch.setattr(server, '_WS_ORPHAN_REAP_GRACE_S', 0)
    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('alpha',))
    beta_home = root / 'profiles' / 'beta'
    beta_home.mkdir(parents=True, mode=0o700)
    (beta_home / 'config.yaml').write_text('{}')
    all_homes = [*(home for _, home in homes), beta_home]
    process_ownership.reserve(all_homes)
    entered, release, fail = asyncio.Event(), asyncio.Event(), []
    sockets, tasks, runner = [], [], None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        real_initialize = session_authority.initialize_session_authority

        async def held_initialize(*args, **kwargs):
            entered.set()
            await release.wait()
            if fail:
                raise RuntimeError('beta store unusable')
            return await real_initialize(*args, **kwargs)

        monkeypatch.setattr(session_authority, 'initialize_session_authority', held_initialize)
        serving = asyncio.create_task(serve_profile_runtime(runner, 'beta', beta_home))
        tasks.append(serving)
        await asyncio.wait_for(entered.wait(), timeout=30)
        assert runner.session_authorities.for_home(beta_home) is None  # reserved, not built yet

        async def connect(scope='host'):
            store = runner.session_ticket_store
            ticket = store.mint(profile_id=str(root.resolve()), subject='uid:desk', purpose='interactive', scope=scope)
            grant = store.redeem(ticket, profile_id=None, purpose='interactive')
            identity = {'user_id': grant['subject'], 'provider': 'local', 'profile_id': grant['profile_id'],
                        'instance_id': grant['instance_id'], 'capabilities': grant['capabilities'],
                        'native_bootstrap': True, 'profile_scope': grant.get('scope', 'profile')}
            sock = _Socket(authority_for_profile_id(runner, grant['profile_id']))
            sockets.append(sock)
            sock.task = asyncio.create_task(ws_mod.handle_ws(sock, auth_identity=identity, operator=True))
            tasks.append(sock.task)
            return sock

        def create(sock, profile, request_id):
            sock.inbound.put_nowait({'jsonrpc': '2.0', 'id': request_id, 'method': 'session.create', 'params': {
                'profile': profile, 'request_id': request_id, 'source': 'cli', 'cwd': str(tmp_path),
                'model': 'm', 'toolsets': []}})

        yield SimpleNamespace(runner=runner, alpha=homes[1][1], beta=beta_home, serving=serving,
                              release=release, fail=fail, connect=connect, create=create)
    finally:
        release.set()
        for sock in sockets:
            sock.inbound.put_nowait(None)
        if tasks:
            await asyncio.wait(tasks, timeout=30)
        for task in tasks:
            if task.done() and not task.cancelled():
                task.exception()
        for authority in list(getattr(runner, 'session_authorities', None) or ()):
            authority.db.close()
        for home in all_homes:
            process_ownership.release(Path(home))


@pytest.mark.asyncio
async def test_a_running_profile_is_answered_while_a_frame_waits_for_another_profile_to_start(rig):
    """Head-of-line: the frame for the starting profile waits off the socket's serial lane, so the
    next frame, for a profile already running, is answered without waiting for that start."""
    sock = await rig.connect()
    rig.create(sock, 'beta', 'beta-while-starting')
    rig.create(sock, 'alpha', 'alpha-behind-it')
    alpha = await sock.reply('alpha-behind-it', timeout=10)
    assert 'result' in alpha, alpha
    assert alpha['result']['session_id'] in rig.runner.session_authorities.for_home(rig.alpha).sessions
    assert 'beta-while-starting' not in sock.replies  # still waiting for beta, not refused

    rig.release.set()
    served = await asyncio.wait_for(rig.serving, 30)
    beta = await sock.reply('beta-while-starting')
    assert 'result' in beta, beta
    assert beta['result']['session_id'] in served.sessions


@pytest.mark.asyncio
async def test_a_failed_start_answers_the_waiting_frame_with_profile_mismatch(rig):
    """A start that fails drops the slot; the waiting frame is answered then, as for any profile this
    process does not serve, not when its wait runs out (``_WS_PROFILE_START_WAIT_S``, 20 s)."""
    sock = await rig.connect()
    rig.create(sock, 'beta', 'beta-start-fails')
    sock.inbound.put_nowait({'jsonrpc': '2.0', 'id': 'after-beta', 'method': 'ping', 'params': {}})
    assert 'result' in await sock.reply('after-beta', timeout=10)  # the beta frame was read first
    rig.fail.append(True)
    rig.release.set()
    with pytest.raises(RuntimeError, match='beta store unusable'):
        await asyncio.wait_for(rig.serving, 30)
    reply = await sock.reply('beta-start-fails', timeout=10)
    assert reply.get('error', {}).get('message') == 'profile_mismatch', reply


@pytest.mark.asyncio
async def test_a_profile_still_starting_when_the_wait_runs_out_is_refused_retryably(rig, monkeypatch):
    from tui_gateway import ws as ws_mod
    monkeypatch.setattr(ws_mod, '_WS_PROFILE_START_WAIT_S', 0.5)
    sock = await rig.connect()
    rig.create(sock, 'beta', 'beta-start-slow')
    reply = await sock.reply('beta-start-slow', timeout=30)
    assert rig.runner.session_authorities.for_home(rig.beta) is None  # beta is still starting
    assert reply.get('error', {}).get('data', {}).get('reason') == 'runtime_draining', reply
    assert reply['error']['data'].get('field') == 'profile'


@pytest.mark.asyncio
async def test_a_profile_bound_grant_naming_a_starting_profile_is_refused_at_once(rig):
    """Only a host-scoped grant reaches another profile; a profile-bound one never waits for it."""
    sock = await rig.connect(scope='profile')
    rig.create(sock, 'beta', 'bound-names-beta')
    reply = await sock.reply('bound-names-beta', timeout=10)
    assert reply.get('error', {}).get('message') == 'profile_mismatch', reply


@pytest.mark.asyncio
async def test_frames_held_for_a_starting_profile_still_park_the_reader(rig, monkeypatch):
    """Held frames count against the socket's dispatch bound: a flood naming the starting profile stops
    being read off the socket once that many wait, instead of piling up in memory, and every frame is
    still answered once the start ends."""
    from tui_gateway import ws as ws_mod

    bound = 4
    monkeypatch.setattr(ws_mod, '_WS_DISPATCH_QUEUE_MAX', bound)
    sock = await rig.connect()
    flood = 10 * bound
    for n in range(flood):
        sock.inbound.put_nowait({'jsonrpc': '2.0', 'id': f'held-{n}', 'method': 'ping', 'params': {'profile': 'beta'}})

    # The reader parks once it took the held bound, the queue bound and one frame each in the
    # dispatcher's and the reader's hands. Wait until it has taken that many, so the reader and the
    # dispatcher really ran, then give it room to read on before checking it did not.
    from tests.waiting import async_wait_until

    parked = 2 * bound + 2
    await async_wait_until(lambda: flood - sock.inbound.qsize() >= parked, timeout=10,
                           what='the reader took the frames that park it')
    await asyncio.sleep(0.2)  # health: allow HX013 -- a parked reader sets no event; reading no further needs a delay
    assert flood - sock.inbound.qsize() == parked, 'held frames escaped the dispatch bound'
    assert not sock.replies  # all still waiting for beta

    rig.release.set()
    await asyncio.wait_for(rig.serving, 30)
    for n in range(flood):
        await sock.reply(f'held-{n}', timeout=10)
    # Written in arrival order, the frames the reader took only after the start ended included.
    assert sock.order == [f'held-{n}' for n in range(flood)]


@pytest.mark.asyncio
async def test_a_frame_read_after_the_start_ends_still_queues_behind_the_held_frames(rig, monkeypatch):
    """The start ended but its lane is still answering the frames held for it: a frame for that profile
    read now joins the end of that queue instead of running on the serial lane ahead of them."""
    from tui_gateway import ws as ws_mod

    real_dispatch = ws_mod._dispatch_request
    ran, first_held_running, first_held_may_finish = [], asyncio.Event(), asyncio.Event()
    gate_expired = []  # held-0's gate ran out: its reply is then an internal error, not the hold

    async def recording_dispatch(connection, req, method, transport):
        ran.append(req.get('id'))
        if req.get('id') == 'held-0':
            first_held_running.set()
            try:
                await asyncio.wait_for(first_held_may_finish.wait(), timeout=10)
            except TimeoutError:
                gate_expired.append(req.get('id'))
                raise
        return await real_dispatch(connection, req, method, transport)

    monkeypatch.setattr(ws_mod, '_dispatch_request', recording_dispatch)
    sock = await rig.connect()

    def ping(request_id, **params):
        sock.inbound.put_nowait({'jsonrpc': '2.0', 'id': request_id, 'method': 'ping', 'params': params})

    ping('held-0', profile='beta')
    ping('held-1', profile='beta')
    ping('probe-held')  # the serial lane answers it only after it set both beta frames aside
    await sock.reply('probe-held', timeout=10)
    assert ran == ['probe-held']

    rig.release.set()
    await asyncio.wait_for(rig.serving, 30)
    await asyncio.wait_for(first_held_running.wait(), 10)  # the lane drains: held-0 runs, held-1 waits
    ping('late', profile='beta')
    ping('probe-late')  # answered only after the serial lane took 'late'
    await sock.reply('probe-late', timeout=10)
    # 'late' was read while held-0 was still at its gate and held-1 behind it, so the order checked
    # below is that interleaving, not frames that had all finished before 'late' arrived.
    assert not gate_expired, "held-0's gate ran out before 'late' was read"
    assert 'held-0' not in sock.replies and 'held-1' not in sock.replies, sock.order
    first_held_may_finish.set()
    await sock.reply('late', timeout=10)

    beta = ['held-0', 'held-1', 'late']
    assert [request_id for request_id in ran if request_id in beta] == beta, ran
    assert [request_id for request_id in sock.order if request_id in beta] == beta, sock.order
    assert not gate_expired
    # Each ran on beta once it was served, not refused as a profile still starting or not served,
    # nor answered with the internal error a held frame's crash (or its gate running out) becomes.
    for request_id in beta:
        error = sock.replies[request_id].get('error') or {}
        assert error.get('message') not in ('profile_mismatch', 'internal error'), sock.replies[request_id]
        assert (error.get('data') or {}).get('reason') != 'runtime_draining', sock.replies[request_id]


def _record_dispatch(monkeypatch):
    from tui_gateway import ws as ws_mod

    real_dispatch = ws_mod._dispatch_request
    ran = []

    async def recording_dispatch(connection, req, method, transport):
        ran.append(req.get('id'))
        return await real_dispatch(connection, req, method, transport)

    monkeypatch.setattr(ws_mod, '_dispatch_request', recording_dispatch)
    return ran


@pytest.mark.asyncio
async def test_a_disconnect_drops_the_frames_held_for_a_start_that_has_not_ended(rig, monkeypatch):
    """Nobody reads the replies once the client is gone: the socket's handler ends without waiting
    out the start, and the held frames are never run, not even once the start ends."""
    from tui_gateway import ws as ws_mod

    monkeypatch.setattr(ws_mod, '_WS_PROFILE_START_WAIT_S', 30.0)
    ran = _record_dispatch(monkeypatch)
    sock = await rig.connect()
    for n in range(2):
        sock.inbound.put_nowait({'jsonrpc': '2.0', 'id': f'held-{n}', 'method': 'ping', 'params': {'profile': 'beta'}})
    sock.inbound.put_nowait({'jsonrpc': '2.0', 'id': 'probe', 'method': 'ping', 'params': {}})
    await sock.reply('probe', timeout=10)  # both beta frames were set aside before it
    sock.inbound.put_nowait(None)  # the client disconnects

    done, _ = await asyncio.wait({sock.task}, timeout=5)
    assert sock.task in done, 'the handler waited out the start of a profile for a client that is gone'
    rig.release.set()
    await asyncio.wait_for(rig.serving, 30)
    assert ran == ['probe'], ran


@pytest.mark.asyncio
async def test_frames_held_for_a_start_that_ended_before_the_disconnect_still_run(rig, monkeypatch):
    """Frames whose start ended while the client was connected finish as the serial lane's queue does."""
    from tui_gateway import ws as ws_mod

    real_dispatch = ws_mod._dispatch_request
    ran, first_held_running, first_held_may_finish = [], asyncio.Event(), asyncio.Event()

    async def gated_dispatch(connection, req, method, transport):
        ran.append(req.get('id'))
        if req.get('id') == 'held-0':
            first_held_running.set()
            await asyncio.wait_for(first_held_may_finish.wait(), timeout=10)
        return await real_dispatch(connection, req, method, transport)

    monkeypatch.setattr(ws_mod, '_dispatch_request', gated_dispatch)
    sock = await rig.connect()
    for n in range(2):
        sock.inbound.put_nowait({'jsonrpc': '2.0', 'id': f'held-{n}', 'method': 'ping', 'params': {'profile': 'beta'}})
    sock.inbound.put_nowait({'jsonrpc': '2.0', 'id': 'probe', 'method': 'ping', 'params': {}})
    await sock.reply('probe', timeout=10)
    rig.release.set()
    await asyncio.wait_for(rig.serving, 30)
    await asyncio.wait_for(first_held_running.wait(), 10)
    sock.inbound.put_nowait(None)  # the client disconnects while the lane runs held-0
    first_held_may_finish.set()

    await asyncio.wait_for(sock.task, 10)
    assert ran == ['probe', 'held-0', 'held-1'], ran
