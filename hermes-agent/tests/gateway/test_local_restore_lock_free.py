"""Restoring a live local session never waits on the SessionStore lock when its route is already current.

Every queued Desktop turn's claim runs ``restore_local_session`` on the gateway loop, and worker
threads hold ``store._lock`` across state.db writes; the claim used to take that lock even when it
had nothing to write, so the loop waited on other turns' writes.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_restoring_a_current_live_route_does_not_wait_on_the_store_lock(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import restore_local_session

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path / 'fixture'), instance_id='owner')
    viewer = AuthorityConnection(authority, object(), {'user_id': 'human', 'profile_id': str(tmp_path / 'fixture')})
    ref = create_local_session(authority, viewer.actor, {'request_id': 'r', 'source': 'cli',
                                                         'cwd': str(tmp_path), 'model': 'm', 'toolsets': []})
    restore_local_session(authority, ref.session_id)  # live and current from here on
    assert ref.session_id in authority.sessions

    held, release = threading.Event(), threading.Event()

    def writer_holding_the_lock():  # another turn's worker inside a state.db write
        with store._lock:
            held.set()
            release.wait(10)

    holder = threading.Thread(target=writer_holding_the_lock)
    holder.start()
    try:
        assert held.wait(5)
        started = time.monotonic()
        restored = await asyncio.wait_for(asyncio.to_thread(restore_local_session, authority, ref.session_id), 3)
        assert restored.session_id == ref.session_id
        assert time.monotonic() - started < 2
    finally:
        release.set()
        holder.join(10)
