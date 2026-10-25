"""The classic CLI re-supplies its launch-only key on every resume (refs #134889, RV-14).

An owner restart revokes launch keys and takes the same key back on ``session.resume {api_key}``
(``gateway/session_policy.py::rebind_launch_key``; the thin client sends it). The classic CLI resumed with
the session id only, at startup (``--resume`` / ``-c``), on ``/resume`` and on a reconnect, so the first
turn after any gateway restart failed with ``launch_credentials_unavailable``.

Real owners (``initialize_session_authority`` again on the same DB is the restart), real
``AuthorityConnection.dispatch``, real ``GatewayConnection`` and ``GatewaySessionStore``: each dial of the
CLI's connection reaches the current owner through a bridge client."""
import asyncio
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError, rpc_error
from hermes_cli.gateway_connection import GatewayConnection
from hermes_cli.gateway_session_store import GatewaySessionStore

RAW = 'UNIQUE-LAUNCH-ONLY-KEY'
LAUNCH = {'model': 'm', 'provider': None, 'api_key': RAW, 'base_url': 'http://127.0.0.1:9/v1'}


class _Life:
    """One socket to one owner: requests run on the owner's loop (the test's)."""

    def __init__(self, connection, loop):
        self.connection, self.loop = connection, loop
        self.events = asyncio.Queue()

    async def rpc(self, method, *, _timeout=None, **params):
        frame = {'id': 1, 'method': method, 'params': params}
        reply = await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self.connection.dispatch(frame), self.loop))
        if 'error' in reply:
            raise rpc_error(reply['error'])
        return reply['result']


class _Owners:
    """The profile's gateway across a restart: the first owner made a keyed and a keyless CLI session."""

    async def start(self, tmp_path, monkeypatch):
        from gateway import run
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore
        from gateway.session_controls import AuthorityConnection
        from gateway.session_local import create_local_session
        monkeypatch.setattr(run, '_load_gateway_config', lambda *a, **k: {'platform_toolsets': {'cli': []}})
        self.profile, self.loop, self.lives = str(tmp_path), asyncio.get_running_loop(), []
        self.identity = {'user_id': 'owner'}  # the CLI's principal on each dial
        self.store = SessionStore(tmp_path / 'sessions', GatewayConfig())
        first = await self.boot('first')
        actor = AuthorityConnection(first, object(), {'user_id': 'owner'}).actor
        base = {'cwd': self.profile, 'model': 'm', 'toolsets': []}
        self.keyed = create_local_session(first, actor, {**base, 'request_id': 'keyed', 'api_key': RAW,
                                                         'base_url': LAUNCH['base_url']}).session_id
        self.plain = create_local_session(first, actor, {**base, 'request_id': 'plain'}).session_id
        return self

    async def boot(self, instance_id):
        from gateway.session_authority import initialize_session_authority
        owner = SimpleNamespace(session_store=self.store, _session_db=self.store._db, adapters={}, _draining=False,
                                _cached_agent_for=lambda route: None, _evict_cached_agent=lambda route: None)
        owner._adapter_for_source = lambda source: owner.adapters.get(source.platform)
        self.current = await initialize_session_authority(owner, profile_id=self.profile, instance_id=instance_id)
        return self.current

    async def _open(self):
        from gateway.session_controls import AuthorityConnection
        self.lives.append(_Life(AuthorityConnection(self.current, object(), self.identity), self.loop))
        return self.lives[-1]

    @asynccontextmanager
    async def dial(self):
        yield await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self._open(), self.loop))

    async def hang_up(self, conn):
        thread = conn._thread
        conn.loop.call_soon_threadsafe(self.lives[-1].events.put_nowait, GatewayClientError('Gateway disconnected'))
        await asyncio.to_thread(thread.join, 5)
        await self.lives[-1].connection.close()

    def key_of(self, session_id):
        """What the next turn's agent build reads (``launch_key``); ``launch_credentials_unavailable`` on main."""
        from gateway.session_policy import launch_key, policy_for_source
        return launch_key(self.current, policy_for_source(self.current.runner, self.current.sessions[session_id].source))

    async def close(self, conn):
        await asyncio.to_thread(conn.close)
        for life in self.lives:
            await life.connection.close()
        self.store._db.close()


@pytest.mark.asyncio
async def test_resume_after_a_restart_resupplies_the_launch_key(tmp_path, monkeypatch):
    """``hermes --resume <id> --api-key K`` (or ``-c`` with a keyed alias) once the gateway restarted; then
    ``/resume`` of a session launched without that key still attaches."""
    owners = await _Owners().start(tmp_path, monkeypatch)
    await owners.boot('restarted')
    conn = GatewayConnection(connect=owners.dial)
    try:
        store = GatewaySessionStore(conn)
        cli = SimpleNamespace(_resumed=True, session_id=owners.keyed)
        assert await asyncio.to_thread(store.open_session, cli, LAUNCH) == owners.keyed
        assert owners.key_of(owners.keyed) == RAW
        assert (await asyncio.to_thread(store.attach, owners.plain))['stored_session_id'] == owners.plain
    finally:
        await owners.close(conn)


@pytest.mark.asyncio
async def test_a_reconnect_after_a_restart_resupplies_the_launch_key(tmp_path, monkeypatch):
    """A classic CLI left open across ``hermes gateway restart``: its redial resubscribes with the key, and
    a session launched without it is still resubscribed."""
    owners = await _Owners().start(tmp_path, monkeypatch)
    conn = GatewayConnection(connect=owners.dial)
    try:
        store = GatewaySessionStore(conn)
        await asyncio.to_thread(store.open_session, SimpleNamespace(_resumed=True, session_id=owners.keyed), LAUNCH)
        await asyncio.to_thread(store.attach, owners.plain)
        before = conn.states[owners.plain].subscription_id
        await owners.hang_up(conn)
        await owners.boot('restarted')
        await asyncio.to_thread(conn.start, 10)  # the redial resubscribes before it counts as ready
        assert owners.key_of(owners.keyed) == RAW
        assert conn.states[owners.plain].subscription_id not in (None, before)
    finally:
        await owners.close(conn)


def _refuse_the_rebind(monkeypatch, code):
    """The owner refuses the keyed resume with ``code`` where it takes the key back (a receipt read that
    fails is ``storage_unavailable``); a plain resume never gets there."""
    from gateway import session_policy
    from hermes_state_runtime import RuntimeStoreError

    def refuse(*_a):
        raise RuntimeStoreError(code)
    monkeypatch.setattr(session_policy, 'rebind_launch_key', refuse)


def _plain_retries(caplog):
    return [r.getMessage() for r in caplog.records if r.name == 'hermes_cli.gateway_connection']


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['attach', 'reconnect'])
@pytest.mark.parametrize('code, plain', [('admission_conflict', True), ('permission_denied', True),
                                         ('storage_unavailable', False)])
async def test_a_refused_keyed_resume_retries_plain_only_when_the_key_does_not_apply(
        tmp_path, monkeypatch, caplog, path, code, plain):
    """A key the session does not take (``admission_conflict``) or a viewer without ``session:control``
    (``permission_denied``) resumes plain as main did, with a debug line naming the code, never the key. Any
    other refusal fails the resume, on ``/resume`` and on a reconnect alike (one helper): a plain retry would
    hide it, and the first turn would fail ``launch_credentials_unavailable`` with nothing in the log."""
    caplog.set_level(logging.DEBUG, logger='hermes_cli.gateway_connection')
    owners = await _Owners().start(tmp_path, monkeypatch)
    conn = GatewayConnection(connect=owners.dial)
    store = GatewaySessionStore(conn)
    cli = SimpleNamespace(_resumed=True, session_id=owners.keyed)
    try:
        before = None
        if path == 'reconnect':
            await asyncio.to_thread(store.open_session, cli, LAUNCH)
            before = conn.states[owners.keyed].subscription_id
            await owners.hang_up(conn)
        await owners.boot('restarted')
        _refuse_the_rebind(monkeypatch, code)
        resume = (lambda: conn.start(10)) if path == 'reconnect' else (lambda: store.open_session(cli, LAUNCH))
        if plain:
            await asyncio.to_thread(resume)
            assert conn.states[owners.keyed].subscription_id not in (None, before)
            assert any(code in line for line in _plain_retries(caplog))
        else:
            with pytest.raises(GatewayRPCError, match=code):
                await asyncio.to_thread(resume)
        assert RAW not in caplog.text
    finally:
        await owners.close(conn)


@pytest.mark.asyncio
async def test_a_read_only_viewer_still_gets_the_plain_attach(tmp_path, monkeypatch, caplog):
    """The owner checks ``session:control`` before it takes a key back: a read-only viewer's keyed resume is
    ``permission_denied``, and it still gets main's ``session:read`` attach, binding nothing."""
    from hermes_state_runtime import RuntimeStoreError
    caplog.set_level(logging.DEBUG, logger='hermes_cli.gateway_connection')
    owners = await _Owners().start(tmp_path, monkeypatch)
    await owners.boot('restarted')
    owners.identity = {'user_id': 'owner', 'capabilities': ['session:read']}
    conn = GatewayConnection(connect=owners.dial)
    try:
        store = GatewaySessionStore(conn)
        cli = SimpleNamespace(_resumed=True, session_id=owners.keyed)
        assert await asyncio.to_thread(store.open_session, cli, LAUNCH) == owners.keyed
        assert any('permission_denied' in line for line in _plain_retries(caplog))
        with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable'):
            owners.key_of(owners.keyed)
    finally:
        await owners.close(conn)


def _typed(session_id, key):
    """``hermes --resume <id> --api-key <key>``: the key the user typed (``HermesCLI._typed_api_key``)."""
    return SimpleNamespace(_resumed=True, session_id=session_id, _typed_api_key=key)


@pytest.mark.asyncio
async def test_resume_with_a_different_explicit_key_is_refused_not_dropped(tmp_path, monkeypatch):
    """Review probe (#141481, CLI-34): a typed ``--api-key`` the session was not launched with is refused with
    the thin client's words, never attached plain on the session's own key. Same owner life: K1 still held."""
    from hermes_cli.gateway_client import RESUME_KEY_MISMATCH
    owners = await _Owners().start(tmp_path, monkeypatch)
    conn = GatewayConnection(connect=owners.dial)
    try:
        store = GatewaySessionStore(conn)
        wrong = {**LAUNCH, 'api_key': 'A-DIFFERENT-KEY'}
        with pytest.raises(GatewayClientError) as refused:
            await asyncio.to_thread(store.open_session, _typed(owners.keyed, 'A-DIFFERENT-KEY'), wrong)
        assert str(refused.value) == RESUME_KEY_MISMATCH
        assert owners.keyed not in store.snapshots  # nothing attached
        # A typed key on a session launched without one: the same refusal (the thin client's too).
        with pytest.raises(GatewayClientError, match='must be the key this session was launched with'):
            await asyncio.to_thread(store.open_session, _typed(owners.plain, RAW), LAUNCH)
        # The right typed key still resumes (after a restart: test_resume_after_a_restart_resupplies_the_launch_key).
        assert await asyncio.to_thread(store.open_session, _typed(owners.keyed, RAW), LAUNCH) == owners.keyed
    finally:
        assert owners.key_of(owners.keyed) == RAW  # the session runs on K1 regardless
        await owners.close(conn)


@pytest.mark.asyncio
async def test_a_keyed_alias_still_opens_a_keyless_session(tmp_path, monkeypatch, caplog):
    """The key came from a keyed startup alias (nothing typed): ``-c`` / ``--resume`` of a session launched
    without a key keeps the plain-resume fallback, and a reconnect resubscribes it plain too."""
    caplog.set_level(logging.DEBUG, logger='hermes_cli.gateway_connection')
    owners = await _Owners().start(tmp_path, monkeypatch)
    conn = GatewayConnection(connect=owners.dial)
    try:
        store = GatewaySessionStore(conn)
        alias = SimpleNamespace(_resumed=True, session_id=owners.plain, _typed_api_key=None)
        launch = {**LAUNCH, 'base_url': None}  # a changed --base-url on resume is refused on its own
        assert await asyncio.to_thread(store.open_session, alias, launch) == owners.plain
        assert conn.launch_api_key == RAW and not conn.launch_api_key_typed
        assert any('admission_conflict' in line for line in _plain_retries(caplog))
        before = conn.states[owners.plain].subscription_id
        await owners.hang_up(conn)
        await owners.boot('restarted')
        await asyncio.to_thread(conn.start, 10)
        assert conn.states[owners.plain].subscription_id not in (None, before)
    finally:
        await owners.close(conn)


@pytest.mark.asyncio
async def test_a_typed_key_reconnect_keeps_the_fallback(tmp_path, monkeypatch):
    """A reconnect types nothing: a typed-key CLI that also viewed a keyless session resubscribes it plain."""
    owners = await _Owners().start(tmp_path, monkeypatch)
    conn = GatewayConnection(connect=owners.dial)
    try:
        store = GatewaySessionStore(conn)
        await asyncio.to_thread(store.open_session, _typed(owners.keyed, RAW), LAUNCH)
        assert conn.launch_api_key_typed
        await asyncio.to_thread(conn.resume, store.call, owners.plain)  # e.g. a session.resume a view made plain
        conn.state(owners.plain).subscription_id = 'viewed'
        await owners.hang_up(conn)
        await owners.boot('restarted')
        await asyncio.to_thread(conn.start, 10)
        assert owners.key_of(owners.keyed) == RAW
        assert conn.states[owners.plain].subscription_id not in (None, 'viewed')
    finally:
        await owners.close(conn)


@pytest.mark.asyncio
async def test_slash_resume_in_a_typed_key_cli_switches_to_a_keyless_session_on_its_own_route(
        tmp_path, monkeypatch):
    """Review (#141481, CLI-34): only the startup resume holds a typed ``--api-key``. ``/resume <keyless
    session>`` in a ``--api-key`` CLI switches to it as on main, and the owner does not bind the key to it:
    the session keeps its own frozen route."""
    from unittest.mock import patch
    from cli import HermesCLI
    from hermes_state import SessionDB
    owners = await _Owners().start(tmp_path, monkeypatch)
    conn = GatewayConnection(connect=owners.dial)
    try:
        store = GatewaySessionStore(conn)
        store._reader = SessionDB(db_path=owners.store._db.db_path, read_only=True)  # the CLI's state.db reads
        cli = HermesCLI.__new__(HermesCLI)
        cli._session_db, cli._resumed, cli.session_id, cli._typed_api_key = store, True, owners.keyed, RAW
        cli.agent, cli.conversation_history, cli._pending_resume_sessions = None, [], None
        cli._restore_session_cwd = cli._restore_session_yolo = cli._restore_session_model = lambda *a, **k: None
        await asyncio.to_thread(store.open_session, cli, LAUNCH)  # hermes --resume <keyed> --api-key K
        printed = []
        with patch('hermes_cli.main._resolve_session_by_name_or_id', return_value=None), \
                patch('cli._cprint', printed.append):
            await asyncio.to_thread(cli._handle_resume_command, f'/resume {owners.plain}')
        assert cli.session_id == owners.plain, printed
        assert store.current['stored_session_id'] == owners.plain
        assert owners.key_of(owners.plain) is None  # not rebound to K
    finally:
        await owners.close(conn)


@pytest.mark.asyncio
async def test_branch_after_new_in_a_typed_key_cli_switches_to_the_child(tmp_path, monkeypatch):
    """Review probe (#141481, CLI-34): ``/new`` in a ``--api-key`` CLI opens a session on the profile's default
    route, without the key, and ``/branch`` of it attaches the child the owner just made with the plain
    fallback. Held to the typed key, the attach raised the ``--api-key`` mismatch after the owner had
    committed the child: the CLI stayed on the parent, and each retry made another child."""
    from datetime import datetime
    from gateway import run
    from hermes_cli.cli_init_mixin import CLIInitMixin
    owners = await _Owners().start(tmp_path, monkeypatch)
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a, **k: {'platform_toolsets': {'cli': []}, 'model': 'm'})
    conn = GatewayConnection(connect=owners.dial)
    try:
        store = GatewaySessionStore(conn)
        cli = _typed(owners.keyed, RAW)
        cli._session_db, cli._gateway_session_open = store, True
        await asyncio.to_thread(store.open_session, cli, LAUNCH)
        fresh = await asyncio.to_thread(CLIInitMixin._new_session_id, cli, datetime.now())  # /new
        assert owners.key_of(fresh) is None
        owners.store._db.append_message(fresh, role='user', content='hello')
        child = await asyncio.to_thread(CLIInitMixin._new_session_id, cli, datetime.now(), branch_of=fresh)  # /branch
        assert child != fresh and store.current['stored_session_id'] == child
    finally:
        await owners.close(conn)


@pytest.mark.asyncio
async def test_probe_slash_resume_keyless_session_from_api_key_cli_switches(tmp_path, monkeypatch):
    """Review 2.1 probe (#141481, CLI-34): in a CLI started with ``--resume <keyed> --api-key K`` after a
    restart, ``/resume``'s id lookup (``_gateway_owned_id``) of a session launched without K answers that
    session's id and attaches it, as on main; only the startup resume is held to the typed key."""
    from hermes_cli.cli_agent_setup_mixin import _gateway_owned_id
    owners = await _Owners().start(tmp_path, monkeypatch)
    await owners.boot('restarted')
    conn = GatewayConnection(connect=owners.dial)
    try:
        store = GatewaySessionStore(conn)
        assert await asyncio.to_thread(store.open_session, _typed(owners.keyed, RAW), LAUNCH) == owners.keyed
        assert await asyncio.to_thread(_gateway_owned_id, store, owners.plain) == owners.plain
        assert store.current['stored_session_id'] == owners.plain
        assert owners.key_of(owners.plain) is None  # K was not bound to it
        assert owners.key_of(owners.keyed) == RAW
    finally:
        await owners.close(conn)
