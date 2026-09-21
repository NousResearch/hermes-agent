"""Real generic/LOCAL precedence controls for the extracted selection chain."""
import json
from types import SimpleNamespace

import pytest

from tests.gateway.test_selected_route import selection  # noqa: F401


@pytest.mark.parametrize('branch,expected', [
    ('override', 'session-override'), ('persisted', 'session-persisted'),
    ('alias', 'alias-model'), ('request', 'request-model'),
    ('default', 'fixture-model'), ('locked', 'alias-model'),
])
def test_generic_adapter_precedence_retains_actual_session_arguments(selection, monkeypatch, branch, expected):
    from gateway import run
    t = selection
    monkeypatch.setattr(run, '_gateway_runner_ref', lambda: t.runner)
    if branch in {'override', 'locked'}:
        t.runner._session_state(t.key).conversation.model_override = {
            'model': 'session-override', 'provider': 'anthropic', 'api_key': 'override-private'}
    runtime = t.runner._prepare_session_agent_runtime(source=t.source, session_key=t.key).runtime
    opts = dict(requested_model='request-model', requested_provider='anthropic',
        route={'model': 'alias-model', 'provider': 'anthropic', 'api_key': 'alias-private'},
        session_model='session-persisted' if branch in {'persisted', 'override', 'locked'} else None,
        confirmed_runtime_lock=branch == 'locked', gateway_session_key=t.key, session_id='prospective')
    if branch in {'request', 'default'}:
        opts['route'] = None
    if branch == 'default':
        opts.update(requested_model=None, requested_provider=None)
    pending = {}
    model, override, _, _ = t.adapter._select_agent_runtime(runtime, 'fixture-model', **opts, pending_models=pending)
    assert model == expected
    assert (override is not None) is (branch == 'override')
    assert runtime['api_key'] == ('override-private' if branch == 'override' else
                                 'alias-private' if branch in {'alias', 'locked'} else 'synthetic-secret')
    assert t.adapter._last_resolved_model == {}
    assert pending['*'] == expected


@pytest.mark.parametrize('failure', [False, True])
def test_local_policy_does_not_enter_api_or_global_auth_fallback(selection, monkeypatch, failure):
    from gateway.session_local import LocalSessionAdapter
    from gateway.session_policy import LocalSessionPolicy
    from gateway.session import SessionSource
    from gateway.config import Platform
    from hermes_cli.auth import AuthError
    t = selection
    source = SessionSource(platform=Platform.LOCAL, chat_id='local', user_id='owner', chat_type='dm')
    adapter = LocalSessionAdapter(t.authority)
    t.runner.adapters[Platform.LOCAL] = adapter
    t.authority.sessions['local'] = SimpleNamespace(source=source)
    adapter.register_source(source)
    adapter.policies['local'] = LocalSessionPolicy(source='cli', platform='cli', cwd=str(t.home),
        model='local-model', toolsets=(), config_json=json.dumps({'model': {'provider': 'openai-codex'}}),
        request_json='{}', terminal_json='{}')
    def forbidden(*a, **k): pytest.fail('LOCAL branch read ordinary session persistence or API resolver')
    monkeypatch.setattr(t.runner.session_store, 'get_model_override', forbidden)
    monkeypatch.setattr(t.adapter, '_select_agent_runtime', forbidden)
    t.boundary.fail = failure
    if failure:
        with pytest.raises(AuthError):
            t.runner._prepare_session_agent_runtime(source=source)
        assert 'anthropic' not in t.calls
    else:
        selected = t.runner._prepare_session_agent_runtime(source=source)
        assert selected.model == 'local-model'
        assert selected.runtime['api_mode'] == 'codex_app_server'
        selected.publish()
    assert t.runner._sessions == {}


def test_secret_and_terminal_scope_mismatch_are_readonly_refusals(selection):
    from agent.secret_scope import set_secret_scope, reset_secret_scope
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope
    from gateway import session_selected_route as sr
    from tests.gateway.test_selected_route import scoped
    t = selection; t.boundary.fail = True
    scope = scoped(t); b = sr.prepare_selected_route(scope)
    before = list(t.calls)
    for setter, reset, value in [(set_secret_scope, reset_secret_scope, {'OTHER_KEY': 'private'}),
                                  (set_terminal_scope, reset_terminal_scope, {'TERMINAL_CWD': '/other'})]:
        token = setter(value)
        try:
            assert not sr.peek_selected_route(scope, b).supports_prepared_files
            with pytest.raises(sr.SelectedRouteUnavailable):
                with sr.hold_selected_route(scope, b): pass
        finally:
            reset(token)
        assert sr.peek_selected_route(scope, b).supports_prepared_files
    assert t.calls == before
    b.close()


def test_foreign_sql_connection_cannot_supply_owner_revision(selection):
    from gateway import session_selected_route as sr
    from tests.gateway.test_selected_route import scoped
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch
    t = selection; t.boundary.fail = True
    scope = scoped(t); b = sr.prepare_selected_route(scope)
    foreign = SessionDB(t.home.parent / 'foreign.db')
    try:
        begin_runtime_epoch(foreign, instance_id='foreign')
        assert not sr.peek_selected_route(scope, b, connection=foreign._conn).supports_prepared_files
        assert sr.peek_selected_route(scope, b).supports_prepared_files
    finally:
        foreign.close(); b.close()


def test_copied_context_cannot_consume_other_threads_held_binding(selection):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    from gateway import session_selected_route as sr
    from tests.gateway.test_selected_route import scoped
    t = selection; t.boundary.fail = True
    scope = scoped(t); b = sr.prepare_selected_route(scope)
    def steal():
        with sr.consume_selected_route(scope, b): pass
    with sr.hold_selected_route(scope, b):
        with ThreadPoolExecutor(max_workers=1) as worker:
            f = worker.submit(copy_context().run, steal)
            with pytest.raises(sr.SelectedRouteUnavailable): f.result(10)
        with sr.consume_selected_route(scope, b) as m:
            assert m.runtime['api_mode'] == 'anthropic_messages'


def test_probes_cannot_initialize_and_closed_adapter_cannot_revive(selection, monkeypatch):
    from gateway import session_selected_route as sr
    from tests.gateway.test_selected_route import scoped
    from hermes_state_runtime import begin_runtime_epoch
    t = selection; t.boundary.fail = True
    scope = scoped(t); b = sr.prepare_selected_route(scope)
    def forbidden(*a, **k): pytest.fail('probe initialized/read config/resolved credentials')
    monkeypatch.setattr(t.runner, '_prepare_session_agent_runtime', forbidden)
    monkeypatch.setattr(t.adapter, '_ensure_session_db', forbidden)
    import gateway.run
    import hermes_cli.runtime_provider
    monkeypatch.setattr(gateway.run, '_load_gateway_config', forbidden)
    monkeypatch.setattr(hermes_cli.runtime_provider, 'load_config', forbidden)
    for _ in range(3): assert sr.peek_selected_route(scope, b).supports_prepared_files
    t.adapter._close_cached_session_dbs()
    assert not sr.peek_selected_route(scope, b).supports_prepared_files
    t.adapter._session_db_cache_closed = False
    begin_runtime_epoch(t.db, instance_id='replacement')
    assert not sr.peek_selected_route(scope, b).supports_prepared_files
    b.close()


def test_canonical_alias_secret_override_preserves_selection(selection):
    from gateway import session_selected_route as sr
    from tests.gateway.test_selected_route import scoped
    t = selection; t.boundary.fail = True
    t.adapter._model_routes['fixture-alias'] = {'model': 'aliased', 'provider': 'anthropic', 'api_key': 'route-key'}
    scope = scoped(t)
    settings = {'requested_model': 'fixture-alias', 'route': {'model': 'aliased', 'provider': 'anthropic'}}
    b = sr.prepare_selected_route(scope, api_settings=settings)
    with sr.hold_selected_route(scope, b):
        with sr.consume_selected_route(scope, b) as material:
            assert material.model == 'aliased'
            assert material.runtime['api_key'] == 'route-key'


def test_postconstruction_native_drift_refuses_before_dispatch(selection, monkeypatch):
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext
    from gateway.session_api_turn import api_execution
    t = selection; t.boundary.fail = True
    t.runner._provider_routing = {}
    ctx = TurnContext(source=t.source, session_key=t.key, session_id='prospective', message='text', user_config={})
    turn = TurnRunner(t.runner, ctx)
    monkeypatch.setattr(turn, '_combined_ephemeral_prompt', lambda: '')
    monkeypatch.setattr(turn, '_setup_stream_consumer', lambda _: (None, None, None, False))
    monkeypatch.setattr(turn, '_resolve_turn_agent', lambda *a: (SimpleNamespace(api_mode='codex_app_server'), True))
    def forbidden(*a, **k): pytest.fail('unsupported resulting runtime reached dispatch')
    monkeypatch.setattr(turn, '_wire_turn_agent_callbacks', forbidden)
    token = api_execution.set({'adapter': t.adapter, 'settings': {'room_dispatch': {'attachment_manifest_digest': 'fixture'}}})
    try:
        result = turn._run_sync_scoped()
        assert result['failure_reason'] == 'prepared_files_unsupported'
        assert result['api_calls'] == 0
    finally:
        api_execution.reset(token)
