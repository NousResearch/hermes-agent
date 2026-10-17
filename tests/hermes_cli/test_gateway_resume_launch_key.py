"""The classic CLI re-supplies its launch-only key on every resume (refs #134889, RV-14).

An owner restart revokes launch keys and takes the same key back on ``session.resume {api_key}``
(``gateway/session_policy.py::rebind_launch_key``; the thin client sends it). The classic CLI resumed with
the session id only, at startup (``--resume`` / ``-c``), on ``/resume`` and on a reconnect, so the first
turn after any gateway restart failed with ``launch_credentials_unavailable``.

Real owners (``initialize_session_authority`` again on the same DB is the restart), real
``AuthorityConnection.dispatch``, real ``GatewayConnection`` and ``GatewaySessionStore``: each dial of the
CLI's connection reaches the current owner through a bridge client."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from hermes_cli.gateway_client import GatewayClientError, rpc_error
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
        self.lives.append(_Life(AuthorityConnection(self.current, object(), {'user_id': 'owner'}), self.loop))
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
