"""Native startup recovery's verified-route read stays off the loop; egress priming stays on it.

Main rebuilds each queued route from the session's newest verified native admission
(``recovered_native_source``, a store read per route) and warms the receiving adapter's egress
caches before any recovered turn replies (``_prime_recovered_egress``). The read belongs with the
rest of the binding scan in the worker thread (#148023: a slow store must not park the loop past the
startup watchdog); priming mutates adapter state, so it must run on the owner loop, before the
authority schedules any recovered turn.
"""
import threading

import pytest


@pytest.mark.asyncio
async def test_recovered_route_read_off_loop_and_egress_primed_on_loop(tmp_path, monkeypatch):
    from gateway import session_native_recovery_routes as routes
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.run_runtime import recover_profile_native_sessions
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_ingress_context import native_callback, register_transport_home
    from hermes_constants import get_hermes_home
    from tests.gateway.test_command_bypass_active_session import _StubAdapter

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', '42')
    monkeypatch.setattr('gateway.run._hermes_home', tmp_path)
    monkeypatch.setattr('hermes_state.DEFAULT_DB_PATH', tmp_path / 'state.db')
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=False, sessions_dir=tmp_path / 'sessions'))
    adapter = _StubAdapter(PlatformConfig(enabled=True, token='fixture', typing_indicator=False), Platform.TELEGRAM)
    loop_thread = threading.get_ident()
    events = []

    async def send(chat_id, content, **kwargs):
        return SendResult(success=True, message_id='sent')

    def prime_routing_cache(event):
        events.append(('prime', event.source.chat_id, threading.get_ident()))

    monkeypatch.setattr(adapter, 'send', send)
    monkeypatch.setattr(adapter, 'prime_routing_cache', prime_routing_cache, raising=False)
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

    recovered = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='new')
    recovered._schedule = lambda ref: events.append(('schedule', ref.session_id, threading.get_ident()))
    real_latest = routes.latest_session_admission

    def latest(*args, **kwargs):
        events.append(('route_read', kwargs.get('session_id'), threading.get_ident()))
        return real_latest(*args, **kwargs)

    monkeypatch.setattr(routes, 'latest_session_admission', latest)
    original_recover = SessionAuthority.recover_native_sessions

    async def recover(self, bindings):
        events.append(('validate', None, threading.get_ident()))
        return await original_recover(self, bindings)

    monkeypatch.setattr(SessionAuthority, 'recover_native_sessions', recover)
    try:
        outcome = await recover_profile_native_sessions(runner, recovered)
        assert list(outcome.values()) == ['ready'], outcome
        kinds = [kind for kind, _, _ in events]
        assert kinds == ['route_read', 'prime', 'validate', 'schedule'], events
        threads = {kind: thread for kind, _, thread in events}
        assert threads['route_read'] != loop_thread, 'the verified-route admission read ran on the event loop'
        assert threads['prime'] == loop_thread, 'adapter egress caches were primed off the owner loop'
        assert events[1][1] == '42', 'priming used a source other than the recovered route'
    finally:
        runner.session_store.close_all_db_handles()
