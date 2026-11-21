"""Which attached clients were recently used (``gateway/session_presence.py``).

Driven through the native socket's real dispatch: owner-seen input on a connection, the client's
own ``client.activity`` reports and the hello, read back with ``viewers``. The clock is the presence
module's ``time`` (no sleeping).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.gateway.test_local_desktop_identity import _connect
from tests.waiting import async_wait_until

DESKTOP = {'name': 'hermes-desktop', 'version': '2.1.0', 'kind': 'desktop'}


def _peer(identity):
    """A real ``WSTransport`` (its own attributes) whose frames are collected instead of sent."""
    from tui_gateway.ws import WSTransport

    class Peer(WSTransport):
        def write(self, obj):
            self.frames.append(obj)
            return True

        async def write_many_if_current_async(self, frames, is_current):  # a session's fanned-out frames
            self.frames.extend(frames)
            return True

    peer = Peer(object(), asyncio.get_running_loop(), auth_identity=identity)
    peer.frames = []
    return peer


class _Client:
    def __init__(self, conn):
        from gateway.session_controls import AuthorityConnection
        identity = dict(conn.connection._identity)
        self.transport = _peer(identity)
        self.connection = AuthorityConnection(conn.authority, self.transport, identity)
        self._rid = 0

    async def call(self, method, **params):
        from tui_gateway.ws import _dispatch_request
        self._rid += 1
        rid = self._rid
        reply = await _dispatch_request(self.connection, {'jsonrpc': '2.0', 'id': rid, 'method': method,
                                                          'params': params}, method, self.transport)
        if reply is None:
            def written():
                return next((f for f in self.transport.frames if isinstance(f, dict) and f.get('id') == rid), None)
            await async_wait_until(written, timeout=20)
            reply = written()
        return reply


@pytest.fixture
def clock(monkeypatch):
    from gateway import session_presence
    now = [1000.0]
    monkeypatch.setattr(session_presence, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    return now


async def _session(client, request_id):
    created = await client.call('session.create', source='tui', request_id=request_id)
    assert 'result' in created, created
    return created['result']['session_id']


async def _attach(client, session_id):
    resumed = await client.call('session.resume', session_id=session_id)
    assert 'result' in resumed, resumed
    return resumed['result']


async def _type(client, session_id):
    """Input the owner sees: an interrupt (refused here as stale, which is still the user's hand)."""
    reply = await client.call('session.interrupt', session_id=session_id, execution_generation=10 ** 6)
    assert 'error' in reply, reply


def _viewers(conn, session_id):
    from gateway.session_presence import viewers
    return viewers(conn.authority, session_id)


def _report(client, sessions, idle, **extra):
    return client.call('client.activity', sessions=sessions, idle_seconds=idle, basis='system', **extra)


@pytest.mark.asyncio
async def test_input_and_reports_move_only_the_sessions_they_name(tmp_path, monkeypatch, clock):
    conn = await _connect(tmp_path, monkeypatch)
    client = _Client(conn)
    try:
        first, second = await _session(client, 'a'), await _session(client, 'b')
        for sid in (first, second):
            await _attach(client, sid)
        assert (await _report(client, [first, second], 100))['result'] == {}
        clock[0] += 10
        await _type(client, first)
        assert [(v['idle_seconds'], v['basis']) for v in _viewers(conn, first)] == [(0.0, 'input')]
        assert [(v['idle_seconds'], v['basis'], v['can_alert']) for v in _viewers(conn, second)] == [
            (110.0, 'system', True)]
        clock[0] += 20
        await _report(client, [first, second], 5, can_alert=False)
        assert [(v['idle_seconds'], v['basis'], v['can_alert']) for v in _viewers(conn, first)] == [
            (5.0, 'system', False)]
        clock[0] += 30
        await _type(client, second)
        assert [(v['idle_seconds'], v['basis'], v['can_alert']) for v in _viewers(conn, second)] == [
            (0.0, 'input', False)]
    finally:
        await client.connection.close()
        await conn.connection.close()


@pytest.mark.asyncio
async def test_a_report_is_the_connection_s_whole_statement(tmp_path, monkeypatch, clock):
    """Each report replaces the last: a session it does not list has no evidence any more, even input
    the owner saw itself; ``sessions: []`` withdraws everything. Input after a report counts again."""
    conn = await _connect(tmp_path, monkeypatch)
    client = _Client(conn)
    try:
        first, second = await _session(client, 'a'), await _session(client, 'b')
        for sid in (first, second):
            await _attach(client, sid)
        await _report(client, [first], 3)
        await _report(client, [second], 4)
        assert [(v['idle_seconds'], v['basis'], v['can_alert']) for v in _viewers(conn, first)] == [(None, None, None)]
        assert [(v['idle_seconds'], v['basis']) for v in _viewers(conn, second)] == [(4.0, 'system')]
        await _type(client, first)
        assert [v['basis'] for v in _viewers(conn, first)] == ['input']
        await _report(client, [second], 4)
        assert [(v['idle_seconds'], v['basis']) for v in _viewers(conn, first)] == [(None, None)]
        await _type(client, first)
        assert (await _report(client, [], 0))['result'] == {}
        assert [v['basis'] for sid in (first, second) for v in _viewers(conn, sid)] == [None, None]
        clock[0] += 2
        await _type(client, second)
        assert [(v['idle_seconds'], v['basis']) for v in _viewers(conn, second)] == [(0.0, 'input')]
    finally:
        await client.connection.close()
        await conn.connection.close()


@pytest.mark.asyncio
async def test_an_older_report_does_not_undo_newer_input_and_away_clears(tmp_path, monkeypatch, clock):
    conn = await _connect(tmp_path, monkeypatch)
    client = _Client(conn)
    try:
        sid = await _session(client, 'a')
        await _attach(client, sid)
        await _type(client, sid)
        clock[0] += 3
        await client.call('client.activity', sessions=[sid], idle_seconds=50, basis='page', can_alert=False)
        assert _viewers(conn, sid) == [{'name': None, 'kind': None, 'idle_seconds': 3.0, 'basis': 'input',
                                        'can_alert': False}]
        assert (await client.call('client.activity', sessions=[], idle_seconds=0, basis='system',
                                  away=True))['result'] == {}
        assert [(v['idle_seconds'], v['basis'], v['can_alert']) for v in _viewers(conn, sid)] == [(None, None, None)]
    finally:
        await client.connection.close()
        await conn.connection.close()


@pytest.mark.asyncio
async def test_viewers_are_the_attached_connections_with_their_hello(tmp_path, monkeypatch, clock):
    conn = await _connect(tmp_path, monkeypatch)
    desktop, other = _Client(conn), _Client(conn)
    try:
        sid = await _session(desktop, 'a')
        await desktop.call('client.capabilities', client=dict(DESKTOP, kind='watch'))
        await desktop.call('client.capabilities', client=DESKTOP)
        await _attach(desktop, sid)
        subscription = (await _attach(other, sid))['subscription_id']
        await _report(desktop, [sid], 7)
        seen = sorted(_viewers(conn, sid), key=lambda v: v['name'] or '')
        assert seen == [
            {'name': None, 'kind': None, 'idle_seconds': None, 'basis': None, 'can_alert': None},
            {'name': 'hermes-desktop', 'kind': 'desktop', 'idle_seconds': 7.0, 'basis': 'system', 'can_alert': True},
        ]
        await _report(other, [sid], 1)
        await other.call('session.detach', session_id=sid, subscription_id=subscription)
        assert [v['name'] for v in _viewers(conn, sid)] == ['hermes-desktop']
        await desktop.connection.close()
        assert _viewers(conn, sid) == []
        assert _viewers(conn, 'no-such-session') == []
    finally:
        await other.connection.close()
        await desktop.connection.close()
        await conn.connection.close()


@pytest.mark.asyncio
async def test_a_report_for_a_session_not_attached_is_never_a_viewer(tmp_path, monkeypatch, clock):
    conn = await _connect(tmp_path, monkeypatch)
    owner, stranger = _Client(conn), _Client(conn)
    try:
        sid = await _session(owner, 'a')
        await _attach(owner, sid)
        await _report(stranger, [sid], 0)
        await _type(stranger, sid)
        assert len(_viewers(conn, sid)) == 1
        assert _viewers(conn, sid)[0]['idle_seconds'] is None
    finally:
        await stranger.connection.close()
        await owner.connection.close()
        await conn.connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('params', [
    {'sessions': 'a', 'idle_seconds': 1, 'basis': 'system'},
    {'sessions': [f's{i}' for i in range(17)], 'idle_seconds': 1, 'basis': 'system'},
    {'sessions': ['a'], 'idle_seconds': -1, 'basis': 'system'},
    {'sessions': ['a'], 'idle_seconds': 10 ** 400, 'basis': 'system'},
    {'sessions': ['a'], 'idle_seconds': 1, 'basis': 'focus'},
    {'sessions': ['a'], 'idle_seconds': 1, 'basis': 'page', 'away': 'yes'},
])
async def test_a_malformed_report_is_refused_and_changes_nothing(tmp_path, monkeypatch, clock, params):
    conn = await _connect(tmp_path, monkeypatch)
    client = _Client(conn)
    try:
        sid = await _session(client, 'a')
        await _attach(client, sid)
        await _report(client, [sid], 9)
        refused = await client.call('client.activity', **params)
        assert refused['error']['code'] == 4001, refused
        assert _viewers(conn, sid)[0]['idle_seconds'] == 9.0
    finally:
        await client.connection.close()
        await conn.connection.close()


@pytest.mark.asyncio
async def test_input_on_a_refused_or_malformed_request_keeps_its_answer(tmp_path, monkeypatch, clock):
    conn = await _connect(tmp_path, monkeypatch)
    client = _Client(conn)
    try:
        sid = await _session(client, 'a')
        await _attach(client, sid)
        missing = await client.call('prompt.submit', session_id=sid)
        assert missing['error']['message'] == 'invalid_params', missing
        assert _viewers(conn, sid)[0]['basis'] == 'input'
        clock[0] += 4
        typed = await client.call('prompt.submit', session_id=7, text='hi')
        assert typed['error']['message'] == 'invalid_params', typed
        assert _viewers(conn, sid)[0]['idle_seconds'] == 4.0
    finally:
        await client.connection.close()
        await conn.connection.close()


def test_a_transport_without_presence_accepts_a_report_and_keeps_nothing():
    from gateway.session_presence import report
    transport = SimpleNamespace()
    assert report(transport, {'sessions': ['a'], 'idle_seconds': 1, 'basis': 'page'}) is None
    assert vars(transport) == {}


def test_a_connection_keeps_at_most_sixteen_sessions(clock):
    from gateway.session_presence import ClientPresence, touch, viewers
    transport = SimpleNamespace(client_presence=ClientPresence())
    connection = SimpleNamespace(transport=transport, authority=SimpleNamespace(profile_id='home'))
    authority = SimpleNamespace(profile_id='home', events={'t': transport},
                                sessions={f's{i}': SimpleNamespace(subscribers={'a': SimpleNamespace(transport_id='t')})
                                          for i in range(17)})
    for i in range(17):
        clock[0] += 1
        touch(connection, 'prompt.submit', {'session_id': f's{i}'})
    assert viewers(authority, 's0')[0]['basis'] is None
    assert viewers(authority, 's16')[0]['basis'] == 'input'


def test_a_full_report_and_input_on_its_sessions_do_not_push_each_other_out(clock):
    """Sixteen is the bound for each source by itself. A report may name sixteen sessions; input the
    owner then sees on them must not push those reports out (a lost report forgets ``can_alert``),
    and input on a seventeenth session drops the oldest input only."""
    from gateway.session_presence import ClientPresence, report, touch, viewers
    transport = SimpleNamespace(client_presence=ClientPresence())
    connection = SimpleNamespace(transport=transport, authority=SimpleNamespace(profile_id='home'))
    names = [f's{i}' for i in range(17)]
    authority = SimpleNamespace(profile_id='home', events={'t': transport},
                                sessions={name: SimpleNamespace(subscribers={'a': SimpleNamespace(transport_id='t')})
                                          for name in names})

    def read(name):
        viewer = viewers(authority, name)[0]
        return viewer['basis'], viewer['can_alert']

    assert report(transport, {'sessions': names[:16], 'idle_seconds': 30, 'basis': 'system', 'can_alert': False}) is None
    for name in names[:16]:
        clock[0] += 1
        touch(connection, 'prompt.submit', {'session_id': name})
    assert [read(name) for name in names[:16]] == [('input', False)] * 16
    clock[0] += 1
    touch(connection, 'prompt.submit', {'session_id': names[16]})
    # s0's input was the oldest: its report (older evidence, same can_alert) is what is left.
    assert [read(name) for name in names] == [('system', False), *[('input', False)] * 15, ('input', None)]


async def _profiles(tmp_path, monkeypatch):
    """The launch profile and a ``beta`` sibling served by one process, each with its own live ``s``
    (session ids are unique only per profile), and a host-scoped socket that reaches both."""
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_authority import LiveSession, initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from tests.fixtures.storage_seed import seed_session
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    authorities = []
    for home in (tmp_path / '.hermes' / 'profiles' / 'beta', tmp_path / '.hermes'):
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv('HERMES_HOME', str(home))
        store = SessionStore(home / 'sessions', GatewayConfig())
        runner = SimpleNamespace(_session_db=store.persistence, session_store=store, _draining=False,
                                 _adapter_for_source=lambda source: None)
        authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='owner')
        seed_session(authority.persistence, 's', 'telegram')
        authority.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='c'), 's')
        authorities.append(authority)
    beta, launch = authorities
    launch.runner.session_authorities = beta.runner.session_authorities = [launch, beta]  # the process's registry
    client = _Client.__new__(_Client)
    client.transport = _peer({})
    client.connection = AuthorityConnection(launch, client.transport, {'user_id': 'owner', 'profile_scope': 'host'})
    client._rid = 0
    return launch, beta, client


def _bases(launch, beta):
    from gateway.session_presence import viewers
    return [[v['basis'] for v in viewers(authority, 's')] for authority in (launch, beta)]


@pytest.mark.asyncio
async def test_input_on_a_shared_socket_is_evidence_only_for_the_profile_it_was_for(tmp_path, monkeypatch, clock):
    launch, beta, client = await _profiles(tmp_path, monkeypatch)
    try:
        for params in ({}, {'profile': 'beta'}):
            assert 'result' in await client.call('session.resume', session_id='s', **params)
        await client.call('session.interrupt', session_id='s', execution_generation=10 ** 6, profile='beta')
        assert _bases(launch, beta) == [[None], ['input']]
    finally:
        await client.connection.close()


@pytest.mark.asyncio
async def test_a_report_counts_only_where_its_session_id_names_one_attached_session(tmp_path, monkeypatch, clock):
    """A report names sessions by id alone: it counts for the one profile this connection is attached
    to that id in, also when it came just before the attach, and for neither when the id is ambiguous."""
    launch, beta, client = await _profiles(tmp_path, monkeypatch)
    try:
        assert (await _report(client, ['s'], 2))['result'] == {}
        assert 'result' in await client.call('session.resume', session_id='s', profile='beta')
        assert _bases(launch, beta) == [[], ['system']]
        assert 'result' in await client.call('session.resume', session_id='s')
        assert _bases(launch, beta) == [[None], [None]]
    finally:
        await client.connection.close()


@pytest.mark.asyncio
async def test_a_person_s_answer_to_a_server_request_counts_as_input(tmp_path, monkeypatch, clock):
    """An accepted answer to a card (here a clarify) is the user's hand on that connection; a refused
    one (an error frame), a repeat of a settled one and an automatic window request are not."""
    from gateway.session_server_requests import ask_viewer, ask_viewer_result, open_requests
    conn = await _connect(tmp_path, monkeypatch)
    client = _Client(conn)
    try:
        sid = await _session(client, 'a')
        await _attach(client, sid)
        events = conn.authority.sessions[sid].event_stream

        async def answer(ask, reply):
            asking = asyncio.create_task(asyncio.to_thread(ask))

            def frame():
                return next((f for f in client.transport.frames if isinstance(f, dict)
                             and str(f.get('id', '')).startswith('srq-') and f['id'] not in answered), None)
            sent = await async_wait_until(frame, timeout=10)
            answered.add(sent['id'])
            await client.connection.dispatch({'jsonrpc': '2.0', 'id': sent['id'], **reply})
            return sent['id'], await asking

        answered = set()
        clarify = {'questions': [{'qid': 'q', 'question': 'Proceed?'}]}
        await answer(lambda: ask_viewer_result(events, sid, 'clarify', clarify, qids=['q'], timeout=10),
                                      {'error': {'code': 4404, 'message': 'not shown'}})
        assert _viewers(conn, sid)[0]['basis'] is None
        _rid, value = await answer(lambda: ask_viewer(events, sid, 'preview.read', {}, timeout=10, replay=False),
                                   {'result': {'value': 'page text'}})
        assert value == 'page text' and _viewers(conn, sid)[0]['basis'] is None
        rid, result = await answer(lambda: ask_viewer_result(events, sid, 'clarify', clarify, qids=['q'], timeout=10),
                                   {'result': {'answers': {'q': 'yes'}}})
        assert result['answers'] == {'q': 'yes'} and open_requests(events, sid) == []
        assert [(v['idle_seconds'], v['basis']) for v in _viewers(conn, sid)] == [(0.0, 'input')]
        clock[0] += 5
        await client.connection.dispatch({'jsonrpc': '2.0', 'id': rid, 'result': {'answers': {'q': 'no'}}})
        assert [(v['idle_seconds'], v['basis']) for v in _viewers(conn, sid)] == [(5.0, 'input')]
    finally:
        await client.connection.close()
        await conn.connection.close()
