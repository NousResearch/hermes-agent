"""Lower selected-route review regressions; all external boundaries stay inert."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Event, RLock, current_thread
from types import SimpleNamespace

import pytest

from tests.gateway.test_selected_route import selection, scoped  # noqa: F401


def test_real_moa_resolution_is_not_files_sdk(selection):
    from gateway import session_selected_route as sr
    t = selection
    (t.home / 'config.yaml').write_text('model:\n  provider: moa\n  default: default\n')
    scope = scoped(t)
    b = sr.prepare_selected_route(scope)
    try:
        assert b._material.runtime['provider'] == 'moa'
        assert b._material.runtime['api_mode'] == 'chat_completions'
        assert not sr.peek_selected_route(scope, b).supports_prepared_files
        assert not t.calls
    finally:
        b.close()


@pytest.mark.parametrize('kind', ['moa', 'acp', 'unknown', 'openai', 'anthropic', 'codex'])
def test_post_setup_uses_actual_execution_client(selection, monkeypatch, kind):
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext
    from gateway.session_api_turn import api_execution
    from openai import OpenAI
    from anthropic import Anthropic
    t = selection; t.boundary.fail = True
    t.runner._provider_routing = {}
    agent = SimpleNamespace(api_mode='chat_completions', provider='openai', acp_command=None,
                            client=object(), _anthropic_client=None)
    if kind == 'moa': agent.provider = 'moa'
    if kind == 'acp': agent.acp_command = 'inert-external'; agent.provider = 'copilot-acp'
    if kind in {'openai', 'codex'}:
        agent.client = OpenAI(api_key='inert-fixture')
        if kind == 'codex': agent.api_mode = 'codex_responses'; agent.provider = 'openai-codex'
    if kind == 'anthropic':
        agent.provider = 'anthropic'; agent.api_mode = 'anthropic_messages'
        agent._anthropic_client = Anthropic(api_key='inert-fixture')
    turn = TurnRunner(t.runner, TurnContext(source=t.source, session_key=t.key,
        session_id='prospective', message='text', user_config={}))
    monkeypatch.setattr(turn, '_combined_ephemeral_prompt', lambda: '')
    monkeypatch.setattr(turn, '_setup_stream_consumer', lambda _: (None, None, None, False))
    monkeypatch.setattr(turn, '_resolve_turn_agent', lambda *a: (agent, True))
    class DispatchBoundary(Exception): pass
    def dispatch(*a): raise DispatchBoundary()
    monkeypatch.setattr(turn, '_wire_turn_agent_callbacks', dispatch)
    token = api_execution.set({'adapter': t.adapter,
        'settings': {'room_dispatch': {'attachment_manifest_digest': 'fixture'}}})
    try:
        if kind in {'openai', 'anthropic', 'codex'}:
            with pytest.raises(DispatchBoundary): turn._run_sync_scoped()
        else:
            result = turn._run_sync_scoped()
            assert result['failure_reason'] == 'prepared_files_unsupported'
            assert result['api_calls'] == 0
    finally:
        api_execution.reset(token)
        if isinstance(agent.client, OpenAI): agent.client.close()
        if agent._anthropic_client is not None: agent._anthropic_client.close()


class ObservedLock:
    """Signal an actual writer's acquisition attempt, not a timing guess."""
    def __init__(self, progress):
        self.lock = RLock(); self.progress = progress; self.blocked = False
    def __enter__(self):
        if current_thread().name.startswith('mutator'):
            acquired = self.lock.acquire(blocking=False)
            self.blocked = not acquired
            self.progress.set()
            if not acquired: self.lock.acquire()
        else: self.lock.acquire()
        return self
    def __exit__(self, *a): self.lock.release()


@pytest.mark.parametrize('writer', ['model', 'restore', 'reset', 'reasoning', 'tier', 'recovery'])
def test_real_mutator_cannot_enter_check_to_publish_gap(selection, monkeypatch, writer):
    from gateway import session_selected_route as sr
    t = selection
    t.runner._session_state(t.key)
    t.runner.session_store.get_or_create_session(t.source)
    t.runner.session_store.set_model_override(t.key, {'model': 'old-pending', 'provider': 'anthropic'})
    selected = t.runner._prepare_session_agent_runtime(source=t.source, session_key=t.key)
    assert selected.pending_override is not None  # real persisted rehydration, inert credentials
    checked, release, progress = Event(), Event(), Event()
    # The production owner lock is observed without replacing any mutator.
    lock = ObservedLock(progress)
    t.runner.__dict__['_selected_route_publication_lock'] = lock
    original = selected.current
    def gate(*a, **kw):
        valid = original(*a, **kw); checked.set(); assert release.wait(10); return valid
    monkeypatch.setattr(selected, 'current', gate)
    monkeypatch.setattr(t.runner, '_evict_cached_agent', lambda *a: None)
    monkeypatch.setattr(t.runner, '_clear_session_boundary_security_state', lambda *a: None)
    order = []
    def publish(): selected.publish(); order.append('published')
    def mutate():
        if writer == 'model': t.runner._session_model_overrides[t.key] = {'model': 'new'}
        if writer == 'restore': t.runner._restore_session_model_override(t.key, {'had_override': True, 'override': {'model': 'new'}})
        if writer == 'reset': t.runner._clear_conversation_scope(t.key, reason='test')
        if writer == 'reasoning': t.runner._set_session_reasoning_override(t.key, {'effort': 'low'})
        if writer == 'tier': t.runner._set_session_service_tier_override(t.key, 'priority')
        if writer == 'recovery': t.runner._last_resolved_model[t.key] = 'new'
        order.append(('mutated', selected._published)); progress.set()
    with ThreadPoolExecutor(1, thread_name_prefix='publisher') as pub, ThreadPoolExecutor(1, thread_name_prefix='mutator') as mut:
        f = pub.submit(copy_context().run, publish)
        assert checked.wait(10)
        g = mut.submit(copy_context().run, mutate)
        try: assert progress.wait(10)
        finally: release.set()
        f.result(10); g.result(10)
    assert ('mutated', True) in order
    if writer in {'model', 'restore'}: assert t.runner._session_model_overrides[t.key]['model'] == 'new'
    if writer == 'reset': assert t.runner._session_model_override(t.key) is None


@pytest.mark.parametrize('consumer', ['wrapper', 'turnrunner'])
def test_ordinary_consumer_refuses_effective_input_drift(selection, monkeypatch, consumer):
    from gateway import session_selected_route as sr
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext
    t = selection
    t.runner._provider_routing = {}
    entered, release = Event(), Event()
    def pause(): entered.set(); assert release.wait(10)
    t.boundary.callback = pause
    if consumer == 'wrapper':
        run = lambda: t.runner._resolve_session_agent_runtime(source=t.source, session_key=t.key)
    else:
        turn = TurnRunner(t.runner, TurnContext(source=t.source, session_key=t.key,
            session_id='prospective', message='text', user_config={}))
        monkeypatch.setattr(turn, '_combined_ephemeral_prompt', lambda: '')
        def forbidden(*a): pytest.fail('stale ordinary turn reached agent setup')
        monkeypatch.setattr(turn, '_setup_stream_consumer', forbidden)
        run = turn._run_sync_scoped
    with ThreadPoolExecutor(1) as worker:
        f = worker.submit(copy_context().run, run)
        try:
            assert entered.wait(10)
            (t.home / 'config.yaml').write_text('model:\n  provider: anthropic\n  default: changed\n')
        finally: release.set()
        if consumer == 'wrapper':
            with pytest.raises(sr.SelectedRouteUnavailable, match='selection_changed'): f.result(10)
        else: assert f.result(10)['api_calls'] == 0
    assert t.runner._sessions == {}


@pytest.mark.parametrize('interleave', [False, True])
def test_peek_never_reopens_closed_owner_database(selection, monkeypatch, interleave):
    from gateway import session_selected_route as sr
    t = selection; t.boundary.fail = True
    scope = scoped(t); b = sr.prepare_selected_route(scope)
    entered, release = Event(), Event()
    original = scope._revision
    def gate(connection=None): entered.set(); assert release.wait(10); return original(connection)
    opens = []
    reopen = t.db._reopen_after_close_locked
    def record(*a, **k): opens.append(True); return reopen(*a, **k)
    monkeypatch.setattr(t.db, '_reopen_after_close_locked', record)
    if interleave:
        monkeypatch.setattr(scope, '_revision', gate)
        with ThreadPoolExecutor(1) as worker:
            f = worker.submit(copy_context().run, sr.peek_selected_route, scope, b)
            try:
                assert entered.wait(10); t.db.close()
            finally: release.set()
            result = f.result(10)
    else:
        t.db.close(); result = sr.peek_selected_route(scope, b)
    b.close()
    assert not opens, 'readiness reopened the owning writer'
    assert not result.supports_prepared_files
    assert t.db._conn is None


def test_ready_peek_close_has_safe_linearization(selection, monkeypatch):
    from gateway import session_selected_route as sr
    t = selection; t.boundary.fail = True
    scope = scoped(t); b = sr.prepare_selected_route(scope)
    entered, release = Event(), Event()
    original = b._material.memory_current
    def gate():
        result = original(); entered.set(); assert release.wait(10); return result
    monkeypatch.setattr(b._material, 'memory_current', gate)
    with ThreadPoolExecutor(1) as worker:
        f = worker.submit(copy_context().run, sr.peek_selected_route, scope, b)
        try: assert entered.wait(10); b.close()
        finally: release.set()
        result = f.result(10)
    assert not result.supports_prepared_files
    assert result.state == 'RETIRED'


@pytest.mark.parametrize('ending', ['close', 'hold-error', 'consume', 'cancel', 'prepare-error', 'superseded'])
def test_private_snapshots_release_without_retiring_successor(selection, monkeypatch, ending):
    from gateway import session_selected_route as sr
    from agent.secret_scope import set_secret_scope, reset_secret_scope
    t = selection; t.boundary.fail = True
    token = set_secret_scope({'UNRELATED_PROFILE_SECRET': 'private-fixture'})
    created = []
    original = sr.PreparedSelectedRoute.__init__
    def capture(self, *a): original(self, *a); created.append(self)
    monkeypatch.setattr(sr.PreparedSelectedRoute, '__init__', capture)
    try:
        scope = scoped(t)
        if ending in {'cancel', 'prepare-error'}:
            if ending == 'prepare-error':
                def fail(): raise RuntimeError('inert preparation error')
                t.boundary.callback = fail
            with pytest.raises(Exception): sr.prepare_selected_route(scope, cancelled=lambda: ending == 'cancel')
            b = created[-1]
        else:
            b = sr.prepare_selected_route(scope)
            if ending == 'hold-error':
                with pytest.raises(RuntimeError):
                    with sr.hold_selected_route(scope, b): raise RuntimeError('operation aborted')
            elif ending == 'consume':
                with sr.hold_selected_route(scope, b):
                    with sr.consume_selected_route(scope, b): pass
            elif ending == 'superseded':
                material = b._material
                successor = sr.prepare_selected_route(scope)
                assert b._material is None and b._inputs is None
                assert b._secret_snapshot is None and b._terminal_snapshot is None
                assert material._effective_inputs is None
                assert sr.peek_selected_route(scope, successor).supports_prepared_files
                b.close()  # Repeated close cannot retire the replacement.
                assert sr.peek_selected_route(scope, successor).supports_prepared_files
                successor.close()
            else: b.close()
        assert b._material is None and b._inputs is None
        # Check only private value snapshots, not borrowed runner/authority owners.
        snapshots = [v for obj in (scope, b) for k, v in vars(obj).items()
                     if k in {'secret_scope', '_secret_snapshot', '_inputs', '_user_config', '_settings', '_adapter_models', '_adapter_before'}]
        def contains(value):
            if isinstance(value, dict): return 'UNRELATED_PROFILE_SECRET' in value or any(contains(v) for v in value.values())
            if isinstance(value, (tuple, list)): return any(contains(v) for v in value)
            return False
        assert not any(contains(v) for v in snapshots), 'closed binding retained a private secret snapshot'
    finally: reset_secret_scope(token)


def test_replacement_releases_old_binding_after_active_hold(selection, monkeypatch):
    from gateway import session_selected_route as sr
    t = selection; t.boundary.fail = True
    scope = scoped(t)
    old = sr.prepare_selected_route(scope)
    material = old._material
    replacing, completed = Event(), Event()
    close = old.close

    def observed_close():
        replacing.set()
        close()

    monkeypatch.setattr(old, 'close', observed_close)

    def replace():
        result = sr.prepare_selected_route(scope)
        completed.set()
        return result

    with ThreadPoolExecutor(1) as pool:
        with sr.hold_selected_route(scope, old):
            future = pool.submit(copy_context().run, replace)
            assert replacing.wait(10)
            assert not completed.is_set()
            assert sr.peek_selected_route(scope, old).supports_prepared_files
        successor = future.result(10)
    assert old._material is None and old._secret_snapshot is None
    assert material._effective_inputs is None
    assert sr.peek_selected_route(scope, successor).supports_prepared_files
    successor.close()


def test_reentrant_replacement_does_not_close_own_active_hold(selection):
    from gateway import session_selected_route as sr
    t = selection; t.boundary.fail = True
    scope = scoped(t)
    old = sr.prepare_selected_route(scope)
    with sr.hold_selected_route(scope, old):
        with pytest.raises(sr.SelectedRouteUnavailable, match='selection_in_use'):
            sr.prepare_selected_route(scope)
        assert old._material is not None
        assert sr.peek_selected_route(scope, old).supports_prepared_files


@pytest.mark.parametrize('writer', ['model', 'reset', 'adapter', 'persisted'])
def test_operation_hold_coordinates_actual_writers(selection, monkeypatch, writer):
    from gateway import session_selected_route as sr
    t = selection; t.boundary.fail = True
    t.runner.session_store.get_or_create_session(t.source)
    scope = scoped(t); b = sr.prepare_selected_route(scope)
    progress, done = Event(), Event()
    lock = ObservedLock(progress)
    if writer == 'persisted':
        lock.lock = t.runner.session_store._lock
        monkeypatch.setattr(t.runner.session_store, '_lock', lock)
    else:
        owner = t.adapter if writer == 'adapter' else t.runner
        monkeypatch.setitem(owner.__dict__, '_selected_route_publication_lock', lock)
    monkeypatch.setattr(t.runner, '_clear_session_boundary_security_state', lambda *a: None)
    def mutate():
        if writer == 'model': t.runner._session_model_overrides[t.key] = {'model': 'new'}
        if writer == 'reset': t.runner._clear_conversation_scope(t.key, reason='test')
        if writer == 'adapter': t.adapter._recover_or_record_model('new', {}, None)
        if writer == 'persisted': t.runner.session_store.set_model_override(t.key, {'model': 'new', 'provider': 'anthropic'})
        done.set(); progress.set()
    with ThreadPoolExecutor(1, thread_name_prefix='mutator') as worker:
        with sr.hold_selected_route(scope, b):
            f = worker.submit(copy_context().run, mutate)
            assert progress.wait(10)
            excluded = lock.blocked
            # This is the actual SQL-held observation, not a private lock probe.
            projected = t.db._execute_write(lambda conn: sr.peek_selected_route(scope, b, connection=conn))
        f.result(10)
    assert excluded, 'actual writer changed selected inputs during the bound operation'
    assert projected.supports_prepared_files


@pytest.mark.parametrize('ending', ['close', 'consume', 'cancel'])
def test_material_releases_effective_input_copies(selection, monkeypatch, ending):
    from gateway import session_selected_route as sr
    t = selection; t.boundary.fail = True
    materials = []
    original = sr.SessionRuntimeSelection._capture
    def capture(self): original(self); materials.append(self)
    monkeypatch.setattr(sr.SessionRuntimeSelection, '_capture', capture)
    scope = scoped(t)
    if ending == 'cancel':
        with pytest.raises(sr.SelectedRouteUnavailable): sr.prepare_selected_route(scope, cancelled=lambda: True)
    else:
        b = sr.prepare_selected_route(scope)
        if ending == 'consume':
            with sr.hold_selected_route(scope, b):
                with sr.consume_selected_route(scope, b): pass
        else: b.close()
    assert materials and all(m._effective_inputs is None for m in materials)


@pytest.mark.parametrize('drift', ['user-config', 'environment'])
def test_ordinary_shared_inputs_do_not_miss_effective_values(selection, monkeypatch, drift):
    from gateway import session_selected_route as sr
    t = selection
    config = {'model': {'default': 'old'}}
    def change():
        if drift == 'user-config': config['model']['default'] = 'new'
        else: monkeypatch.setenv('LLM_MODEL', 'changed-environment')
    t.boundary.callback = change
    with pytest.raises(sr.SelectedRouteUnavailable, match='selection_changed'):
        t.runner._resolve_session_agent_runtime(source=t.source, session_key=t.key, user_config=config)
    assert t.runner._sessions == {}


def test_close_during_live_sql_read_never_closes_borrowed_connection(selection, monkeypatch):
    from gateway import session_selected_route as sr
    t = selection; t.boundary.fail = True
    scope = scoped(t); b = sr.prepare_selected_route(scope)
    reading, closing, release = Event(), Event(), Event()
    def trace(sql):
        if sql.startswith('SELECT runtime_revision'):
            reading.set(); assert release.wait(10)
    t.db._conn.set_trace_callback(trace)
    evict = t.db._evict_one_idle_read_conn
    def closing_readers():
        if t.db._read_conns_closed: closing.set()
        return evict()
    monkeypatch.setattr(t.db, '_evict_one_idle_read_conn', closing_readers)
    def forbidden(*a, **kw): pytest.fail('peek reopened or initialized the database')
    monkeypatch.setattr(t.db, '_open_writer_conn', forbidden)
    monkeypatch.setattr(t.db, '_reopen_after_close_locked', forbidden)
    monkeypatch.setattr(t.db, '_init_schema', forbidden)
    with ThreadPoolExecutor(2) as worker:
        observation = worker.submit(copy_context().run, sr.peek_selected_route, scope, b)
        try:
            assert reading.wait(10)
            close = worker.submit(copy_context().run, t.db.close)
            assert closing.wait(10)
            assert t.db._conn is not None
        finally: release.set()
        result = observation.result(10); close.result(10)
    assert not result.supports_prepared_files and t.db._conn is None
    before = {p.name: (p.stat().st_mode, p.read_bytes()) for p in t.home.glob('state.db*')}
    assert not sr.peek_selected_route(scope, b).supports_prepared_files
    after = {p.name: (p.stat().st_mode, p.read_bytes()) for p in t.home.glob('state.db*')}
    assert before == after
    b.close()


def test_local_custom_launch_key_ignores_live_config_drift(selection):
    import json
    from gateway.session_local import LocalSessionAdapter
    from gateway.session_policy import LocalSessionPolicy, bind_launch_key
    from gateway.session import SessionSource
    from gateway.config import Platform
    t = selection
    source = SessionSource(platform=Platform.LOCAL, chat_id='local', user_id='owner', chat_type='dm')
    adapter = LocalSessionAdapter(t.authority)
    t.runner.adapters[Platform.LOCAL] = adapter
    t.authority.sessions['local'] = SimpleNamespace(source=source)
    adapter.register_source(source)
    config = {'model': {'provider': 'custom:fixture', 'base_url': 'https://fixture.invalid/v1'},
              'custom_providers': [{'name': 'fixture', 'base_url': 'https://fixture.invalid/v1'}]}
    policy = LocalSessionPolicy(source='cli', platform='cli', cwd=str(t.home), model='frozen-model',
        toolsets=(), config_json=json.dumps(config), request_json='{}', terminal_json='{}')
    adapter.policies['local'] = bind_launch_key(t.authority, 'local', policy, 'inert-launch-key')
    selected = t.runner._prepare_session_agent_runtime(source=source)
    (t.home / 'config.yaml').write_text('model:\n  provider: moa\n  default: changed\n')
    try:
        selected.publish()
        assert selected.model == 'frozen-model'
        assert selected.runtime['api_key'] == 'inert-launch-key'
        assert selected.runtime['base_url'] == 'https://fixture.invalid/v1'
        assert not t.calls
    finally: selected.release_snapshots()


def test_ordinary_unavailable_persistence_keeps_existing_fallback(selection, monkeypatch):
    t = selection; t.boundary.fail = True
    def failed_read(*a): raise OSError('inert session-index read failure')
    monkeypatch.setattr(t.runner.session_store, 'get_model_override', failed_read)
    model, runtime = t.runner._resolve_session_agent_runtime(source=t.source, session_key=t.key)
    assert model == 'fallback-model' and runtime['provider'] == 'anthropic'
