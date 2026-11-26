"""A profile's gateway owner in this process, for tests that drive the classic CLI's ``GatewaySessionStore``.

``GatewayOwner`` runs a real session authority (``initialize_session_authority``) on its own loop thread; each store
:meth:`GatewayOwner.client` hands out is a ``GatewaySessionStore`` on a real ``GatewayConnection`` whose requests run
through ``AuthorityConnection.dispatch`` on that loop, so the CLI's synchronous calls reach the owner as they do in
production. ``persistence`` is the owner's storage interface handle, for seeding and reading rows.
"""
import asyncio
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

from hermes_cli.gateway_client import rpc_error
from hermes_cli.gateway_connection import GatewayConnection
from hermes_cli.gateway_session_store import GatewaySessionStore


class _Life:
    """One socket to the owner: each request runs on the owner's loop."""

    def __init__(self, connection, loop):
        self.connection, self.loop = connection, loop
        self.events = asyncio.Queue()

    async def rpc(self, method, *, _timeout=None, **params):
        frame = {"id": 1, "method": method, "params": params}
        reply = await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self.connection.dispatch(frame), self.loop))
        if "error" in reply:
            raise rpc_error(reply["error"])
        return reply["result"]


class GatewayOwner:
    """The profile at *home*'s gateway; ``close`` stops it and every client it handed out."""

    def __init__(self, home, monkeypatch):
        from gateway import run
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore
        monkeypatch.setattr(run, "_load_gateway_config",
                            lambda *a, **k: {"platform_toolsets": {"cli": []}, "model": "m"})
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.store = SessionStore(home / "sessions", GatewayConfig())
        self.persistence = self.store.persistence
        self.lives, self.clients = [], []
        self.authority = self._run(self._boot(str(home)))

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(30)

    async def _boot(self, profile):
        from gateway.session_authority import initialize_session_authority
        runner = SimpleNamespace(session_store=self.store, _session_db=self.persistence, adapters={}, _draining=False,
                                 _cached_agent_for=lambda route: None, _evict_cached_agent=lambda route: None)
        runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
        runner._adapters_for_profile = lambda profile: runner.adapters  # /new and /clear send command.observe
        return await initialize_session_authority(runner, profile_id=profile, instance_id="owner")

    async def _open(self):
        from gateway.session_controls import AuthorityConnection
        self.lives.append(_Life(AuthorityConnection(self.authority, object(), {"user_id": "owner"}), self.loop))
        return self.lives[-1]

    @asynccontextmanager
    async def _dial(self):
        yield await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self._open(), self.loop))

    def client(self):
        """A classic CLI's session store on its own connection to this owner."""
        self.clients.append(GatewaySessionStore(GatewayConnection(connect=self._dial)))
        return self.clients[-1]

    def close(self):
        for client in self.clients:
            client.connection.close()
            client.close_reader()
        for life in self.lives:
            self._run(life.connection.close())
        self.store.close_all_db_handles()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
