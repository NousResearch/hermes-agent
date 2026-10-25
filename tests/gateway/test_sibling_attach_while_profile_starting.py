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
    """The browser end of one WebSocket: frames in, replies collected by id."""

    def __init__(self, authority):
        self.scope = {'hermes.session_authority': authority}
        self.inbound = asyncio.Queue()
        self.replies = {}
        self.arrived = asyncio.Event()

    async def accept(self, **_kw):
        pass

    async def send_text(self, line):
        for part in line.splitlines():
            frame = json.loads(part)
            if 'id' in frame:
                self.replies[frame['id']] = frame
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
            tasks.append(asyncio.create_task(ws_mod.handle_ws(sock, auth_identity=identity, operator=True)))
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
