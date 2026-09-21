"""Selected-route preparation uses real selection; credentials alone are inert."""
from pathlib import Path
from types import SimpleNamespace
import json
import socket
import subprocess

import pytest


@pytest.fixture
def selection(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.session import SessionStore, SessionSource
    from gateway.session_authority import SessionAuthority
    from gateway.session_authorities import SessionAuthorities
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch
    from hermes_cli import runtime_provider as rp
    from gateway.run import _profile_runtime_scope
    import run_agent
    home = tmp_path / 'a'; home.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text('model:\n  provider: openai-codex\n  default: fixture-model\n  openai_runtime: codex_app_server\nfallback_providers:\n  - provider: anthropic\n    model: fallback-model\n')
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner._sessions = {}
    runner.session_store = SessionStore(config=runner.config, sessions_dir=home / 'sessions')
    db = SessionDB(home / 'state.db')
    authority = SessionAuthority(runner, profile_id=str(home), instance_id='fixture-owner', db=db,
        epoch=begin_runtime_epoch(db, instance_id='fixture-owner'))
    runner.session_authority = authority
    runner.session_authorities = SessionAuthorities(home)
    runner.session_authorities.add(home, authority)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': 'inert-api-key'}))
    adapter.gateway_runner = runner
    runner.adapters = {Platform.API_SERVER: adapter}
    source = SessionSource(platform=Platform.API_SERVER, chat_id='prospective', user_id='api', chat_type='dm')
    key = runner.session_store._generate_session_key(source)
    calls = []
    boundary = SimpleNamespace(fail=False, callback=None, expiry=None)
    def forbidden(*a, **k):
        pytest.fail('network/process/agent boundary called')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(run_agent.AIAgent, '__init__', forbidden)
    def resolve(requested, **kwargs):
        from hermes_cli.auth import AuthError
        if boundary.fail and requested == 'openai-codex':
            raise AuthError('synthetic secret diagnostic', provider=requested, code='missing_api_key')
        return requested
    monkeypatch.setattr(rp, 'resolve_provider', resolve)
    def pool(provider):
        entry = SimpleNamespace(runtime_api_key='synthetic-secret',
            base_url='https://api.anthropic.com' if provider == 'anthropic' else 'https://chatgpt.com/backend-api/codex',
            source='inert', expires_at_ms=boundary.expiry * 1000 if boundary.expiry else None)
        def select(**kwargs):
            calls.append(provider)
            if boundary.callback:
                boundary.callback()
            return entry
        return SimpleNamespace(provider=provider, has_credentials=lambda: True, select=select, current=lambda: entry)
    monkeypatch.setattr(rp, 'load_pool', pool)
    with _profile_runtime_scope(home, prepared_secret_scope={}):
        yield SimpleNamespace(home=home, runner=runner, authority=authority, db=db, adapter=adapter,
            source=source, key=key, calls=calls, boundary=boundary)
    adapter._run_idempotency_store._conn.close()
    db.close()


def test_preparation_defers_rehydration_and_recovery_publication(selection):
    t = selection
    # The baseline's only selection entrypoint eagerly publishes recovery state.
    prepare = getattr(t.runner, '_prepare_session_agent_runtime', t.runner._resolve_session_agent_runtime)
    selected = prepare(source=t.source, session_key=t.key)
    assert t.runner._sessions == {}, 'selection must not create session/recovery caches before publication'
    assert t.db.get_session('prospective') is None
    selected.publish()
    assert t.runner._peek_session_state(t.key).conversation.last_resolved_model == 'fixture-model'


def scoped(t, request=None, purpose='execute'):
    from gateway.session_selected_route import selection_scope
    return selection_scope(t.adapter, source=t.source, session_key=t.key, session_id='prospective',
        request_identity=request if request is not None else object(), purpose=purpose)


def test_real_native_and_same_config_sdk_fallback_and_readonly_projection(selection):
    from gateway.session_selected_route import prepare_selected_route, peek_selected_route, hold_selected_route, consume_selected_route
    t = selection
    frozen = (t.home / 'config.yaml').read_bytes()
    scope = scoped(t)
    native = prepare_selected_route(scope)
    p = peek_selected_route(scope, native)
    assert not p.supports_prepared_files and p.reason == 'unsupported_transport'
    assert t.runner._sessions == {} and t.adapter._last_resolved_model == {}
    t.boundary.fail = True
    sdk = prepare_selected_route(scope)
    assert (t.home / 'config.yaml').read_bytes() == frozen
    before = list(t.calls)
    for _ in range(3):
        p = peek_selected_route(scope, sdk)
        assert p.supports_prepared_files
    assert t.calls == before
    encoded = json.dumps(p.__dict__)
    for secret in ('synthetic-secret', 'anthropic', 'fixture-model', 'https:', str(t.home)):
        assert secret not in encoded and secret not in repr(sdk)
    with hold_selected_route(scope, sdk):
        with consume_selected_route(scope, sdk) as material:
            assert material.model == 'fallback-model'
            assert material.runtime['api_mode'] == 'anthropic_messages'
            assert material.runtime['api_key'] == 'synthetic-secret'
    assert t.calls == before
    assert not peek_selected_route(scope, sdk).supports_prepared_files
    assert sdk._material is None


def test_independent_requests_do_not_overwrite_newer_recovery_cache(selection):
    from gateway import session_selected_route as sr
    from gateway.session import SessionSource
    from gateway.config import Platform
    t = selection; t.boundary.fail = True
    first = sr.prepare_selected_route(scoped(t), api_settings={'requested_model': 'first-model'})
    second_source = SessionSource(platform=Platform.API_SERVER, chat_id='second', user_id='api', chat_type='dm')
    second_scope = sr.selection_scope(t.adapter, source=second_source,
        session_key=t.runner.session_store._generate_session_key(second_source), session_id='second',
        request_identity=object(), purpose='execute')
    second = sr.prepare_selected_route(second_scope, api_settings={'requested_model': 'second-model'})
    with sr.hold_selected_route(second_scope, second):
        with sr.consume_selected_route(second_scope, second) as m:
            assert m.model == 'second-model'
    with sr.hold_selected_route(first._scope, first):
        with sr.consume_selected_route(first._scope, first) as m:
            assert m.model == 'first-model'
    assert t.adapter._last_resolved_model['*'] == 'second-model'


@pytest.mark.parametrize('drift', ['request', 'purpose', 'epoch', 'adapter', 'foreign', 'expiry', 'cancel'])
def test_scope_lifetime_and_foreign_peek_do_not_clear_owner(selection, monkeypatch, drift):
    from gateway import session_selected_route as sr
    from gateway.run import _profile_runtime_scope
    t = selection; t.boundary.fail = True
    scope = scoped(t); binding = sr.prepare_selected_route(scope)
    if drift in {'request', 'purpose'}:
        other = scoped(t, scope.request_identity if drift == 'purpose' else object(),
            'stage' if drift == 'purpose' else 'execute')
        assert not sr.peek_selected_route(other, binding).supports_prepared_files
        assert sr.peek_selected_route(scope, binding).supports_prepared_files
    elif drift == 'foreign':
        other = t.home.parent / 'b'; other.mkdir()
        with _profile_runtime_scope(other, prepared_secret_scope={}):
            assert not sr.peek_selected_route(scope, binding).supports_prepared_files
        assert sr.peek_selected_route(scope, binding).supports_prepared_files
    else:
        if drift == 'epoch': t.authority.epoch += 1
        if drift == 'adapter': t.runner.adapters.clear()
        if drift == 'expiry': monkeypatch.setattr(sr.time, 'monotonic', lambda: binding.deadline + 1)
        if drift == 'cancel': binding.close()
        assert not sr.peek_selected_route(scope, binding).supports_prepared_files
        with pytest.raises(sr.SelectedRouteUnavailable):
            with sr.hold_selected_route(scope, binding): pass


def test_config_drift_during_selection_does_not_publish(selection):
    from gateway import session_selected_route as sr
    t = selection; scope = scoped(t)
    t.boundary.callback = lambda: (t.home / 'config.yaml').write_text('model:\n  provider: anthropic\n  default: new-model\n')
    with pytest.raises(sr.SelectedRouteUnavailable, match='selection_changed'):
        sr.prepare_selected_route(scope)
    assert t.runner._sessions == {} and t.adapter._last_resolved_model == {}


def test_override_drift_cannot_be_overwritten_by_pending_publication(selection):
    t = selection
    selected = t.runner._prepare_session_agent_runtime(source=t.source, session_key=t.key)
    newer = {'model': 'newer', 'provider': 'anthropic', 'api_key': 'new-secret'}
    t.runner._session_state(t.key).conversation.model_override = newer
    from gateway.session_selected_route import SelectedRouteUnavailable
    with pytest.raises(SelectedRouteUnavailable, match='selection_changed'):
        selected.publish()
    assert t.runner._session_model_override(t.key) is newer
    assert t.runner._peek_session_state('*') is None


def test_canonical_adapter_arguments_are_none_and_consumed_once(selection, monkeypatch):
    from gateway import session_selected_route as sr
    t = selection; t.boundary.fail = True
    original = t.adapter._select_agent_runtime
    calls = []
    def record(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(t.adapter, '_select_agent_runtime', record)
    scope = scoped(t)
    b = sr.prepare_selected_route(scope, api_settings={'requested_model': 'request-model'})
    assert calls[0]['gateway_session_key'] is None and calls[0]['session_id'] is None
    with sr.hold_selected_route(scope, b):
        with sr.consume_selected_route(scope, b) as m:
            assert m.model == 'request-model'
    with pytest.raises(sr.SelectedRouteUnavailable):
        with sr.consume_selected_route(scope, b): pass
    assert len(calls) == 1


@pytest.mark.parametrize('files,native', [(True, True), (True, False), (False, True)])
def test_actual_turnrunner_consumes_selection_before_agent_boundary(selection, monkeypatch, files, native):
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext
    from gateway.session_api_turn import api_execution
    t = selection
    t.boundary.fail = not native
    t.runner._provider_routing = {}
    ctx = TurnContext(source=t.source, session_key=t.key, session_id='prospective',
                      message='ordinary text', user_config={})
    turn = TurnRunner(t.runner, ctx)
    monkeypatch.setattr(turn, '_combined_ephemeral_prompt', lambda: '')
    monkeypatch.setattr(turn, '_setup_stream_consumer', lambda _: (None, None, None, False))
    received = []
    class ReachedAgentBoundary(Exception): pass
    def record(route, *a):
        received.append(route)
        raise ReachedAgentBoundary()
    monkeypatch.setattr(turn, '_resolve_turn_agent', record)
    settings = {'room_dispatch': {'attachment_manifest_digest': 'fixture'}} if files else {}
    token = api_execution.set({'adapter': t.adapter, 'settings': settings})
    try:
        if native and files:
            result = turn._run_sync_scoped()
            assert result['failure_reason'] == 'prepared_files_unsupported'
            assert result['api_calls'] == 0 and not received
            assert t.runner._sessions == {}
        else:
            with pytest.raises(ReachedAgentBoundary):
                turn._run_sync_scoped()
            assert received[0]['runtime']['api_mode'] == ('codex_app_server' if native else 'anthropic_messages')
            assert t.calls == (['openai-codex'] if native else ['anthropic'])
            assert t.runner._peek_session_state(t.key) is not None
    finally:
        api_execution.reset(token)


def test_known_credential_deadline_caps_snapshot_and_peek_never_reads_pool(selection):
    import time
    from gateway import session_selected_route as sr
    t = selection; t.boundary.fail = True
    t.boundary.expiry = time.time() + 4
    scope = scoped(t)
    binding = sr.prepare_selected_route(scope)
    assert 0 < binding.deadline - time.monotonic() <= 4
    t.boundary.callback = lambda: pytest.fail('peek touched credentials')
    before = list(t.calls)
    with sr.hold_selected_route(scope, binding):
        t.db._execute_write(lambda conn: (
            sr.peek_selected_route(scope, binding, connection=conn).supports_prepared_files
            or pytest.fail('held SQL projection unavailable')))
    assert t.calls == before and binding._material is None


def test_late_override_invalidates_held_publication(selection):
    from gateway import session_selected_route as sr
    t = selection; t.boundary.fail = True
    scope = scoped(t); binding = sr.prepare_selected_route(scope)
    with sr.hold_selected_route(scope, binding):
        t.runner._session_state(t.key).conversation.model_override = {'model': 'newer'}
        assert not sr.peek_selected_route(scope, binding).supports_prepared_files
    assert binding._material is None


def test_concurrent_old_selection_cannot_publish_over_new_input(selection):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    from threading import Event, current_thread
    from gateway import session_selected_route as sr
    t = selection; t.boundary.fail = True
    scope = scoped(t); entered, release = Event(), Event()
    def pause():
        if current_thread().name.startswith('selected-old'):
            entered.set()
            assert release.wait(10)
    t.boundary.callback = pause
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='selected-old') as workers:
        old = workers.submit(copy_context().run, sr.prepare_selected_route, scope)
        assert entered.wait(10)
        (t.home / 'config.yaml').write_text('model:\n  provider: anthropic\n  default: newest-model\n')
        new = sr.prepare_selected_route(scoped(t))
        release.set()
        with pytest.raises(sr.SelectedRouteUnavailable, match='selection_changed'):
            old.result(10)
    assert t.runner._sessions == {}
    with sr.hold_selected_route(new._scope, new):
        with sr.consume_selected_route(new._scope, new) as m:
            assert m.model == 'newest-model'
    assert t.runner._last_resolved_model[t.key] == 'newest-model'


def test_same_operation_repreparation_retires_previous_positive(selection):
    from gateway import session_selected_route as sr
    t = selection; t.boundary.fail = True
    scope = scoped(t); old = sr.prepare_selected_route(scope)
    assert sr.peek_selected_route(scope, old).supports_prepared_files
    t.boundary.fail = False
    native = sr.prepare_selected_route(scope)
    assert not sr.peek_selected_route(scope, native).supports_prepared_files
    assert not sr.peek_selected_route(scope, old).supports_prepared_files
    with pytest.raises(sr.SelectedRouteUnavailable):
        with sr.hold_selected_route(scope, old): pass
    t.boundary.fail = True
    positive = sr.prepare_selected_route(scope)
    with pytest.raises(sr.SelectedRouteUnavailable, match='selection_cancelled'):
        sr.prepare_selected_route(scope, cancelled=lambda: True)
    assert not sr.peek_selected_route(scope, positive).supports_prepared_files
    old.close(); native.close(); positive.close()


def test_preparing_cancellation_retires_without_cache_publication(selection):
    from gateway import session_selected_route as sr
    t = selection
    with pytest.raises(sr.SelectedRouteUnavailable, match='selection_cancelled'):
        sr.prepare_selected_route(scoped(t), cancelled=lambda: True)
    assert t.runner._sessions == {} and t.adapter._last_resolved_model == {}
