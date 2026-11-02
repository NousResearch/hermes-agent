"""Startup native-session recovery reads the store in a worker thread (#148023).

On a host with many/large profile stores the recovery's synchronous SQLite reads parked the
event loop past the shutdown watchdog's liveness budget, so the gateway was exit-75'd mid-startup
and the supervisor restarted into the same blocking pass -- an unbreakable restart loop. The
admission scan, the route list and every ``logical_owner`` walk are store reads, so they must run
off the loop; the scheduling half (``recover_detached_sessions``) still runs on it.
"""
import threading

import pytest


@pytest.mark.asyncio
async def test_startup_recovery_reads_the_store_off_the_event_loop(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.run_runtime import recover_profile_native_sessions
    from gateway.session_authority import initialize_session_authority
    from gateway.session_ingress_context import native_callback, register_transport_home
    from hermes_constants import get_hermes_home
    from tests.gateway.test_command_bypass_active_session import _StubAdapter

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', '42')
    monkeypatch.setattr('gateway.run._hermes_home', tmp_path)
    monkeypatch.setattr('hermes_state.DEFAULT_DB_PATH', tmp_path / 'state.db')  # conftest repoints it per test
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=False, sessions_dir=tmp_path / 'sessions'))
    adapter = _StubAdapter(PlatformConfig(enabled=True, token='fixture', typing_indicator=False), Platform.TELEGRAM)

    async def send(chat_id, content, **kwargs):
        return SendResult(success=True, message_id='sent')
    monkeypatch.setattr(adapter, 'send', send)
    adapter.gateway_runner = runner
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    first = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='old')
    first._schedule = lambda ref: None
    first.register(source)
    head = MessageEvent(text='queued while the owner went away', source=source, message_id='head')
    with native_callback(runner, head, get_hermes_home()):
        await first.admit_native(head)

    # The restart's owner: the queued row is in its store, the route is in this chat's index.
    recovered = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='new')
    recovered._schedule = lambda ref: None
    loop_thread = threading.get_ident()
    reads = {'admissions': set(), 'routes': set(), 'lineage': set()}

    def recorder(kind, bound):
        def spy(*args, **kwargs):
            reads[kind].add(threading.get_ident())
            return bound(*args, **kwargs)
        return spy

    monkeypatch.setattr(recovered.db, '_read_all', recorder('admissions', recovered.db._read_all))
    monkeypatch.setattr(runner.session_store, 'list_sessions',
                        recorder('routes', runner.session_store.list_sessions))
    monkeypatch.setattr(recovered, 'logical_owner', recorder('lineage', recovered.logical_owner))
    try:
        outcome = await recover_profile_native_sessions(runner, recovered)
        assert outcome, 'startup recovery found no queued native turn to recover'
        for kind, threads in reads.items():
            assert threads, f'the {kind} read never happened, so this test proves nothing'
            assert loop_thread not in threads, f'the {kind} read ran on the event loop thread'
    finally:
        runner.session_store.close_all_db_handles()
