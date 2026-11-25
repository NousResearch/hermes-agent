"""Cold automatic speech reviews retain their API route and fence delayed idle dispatch."""
from contextlib import contextmanager, nullcontext
import json
import threading
from types import SimpleNamespace

import pytest

from gateway.session_contract import SessionRef
from tests.waiting import wait_until


def deferred_review(monkeypatch, *, before_spawn=None):
    from agent import background_review, review_idle_queue
    from gateway import session_authorities, session_cold_review, session_external_review, session_local_recovery
    from run_agent import AIAgent
    import run_agent

    queue = review_idle_queue.ReviewIdleQueue()
    monkeypatch.setattr(queue, '_ensure_thread', lambda: None)
    monkeypatch.setattr(review_idle_queue, 'QUEUE', queue)
    monkeypatch.setattr(run_agent, '_review_should_defer', lambda *a: True)
    monkeypatch.setattr(background_review, 'load_background_review_settings', lambda: (True, {'defer': 'auto'}))
    monkeypatch.setattr(session_authorities, 'owner_scope', lambda a: nullcontext())
    monkeypatch.setattr(session_local_recovery, 'transcript_target', lambda *a: 'speech')
    calls, parents = [], []
    live = SimpleNamespace(source=SimpleNamespace(platform='local'), route='route', review=None, task=None)
    owner = SimpleNamespace(retiring=False, sessions={'speech': live}, runner=object(),
        persistence=SimpleNamespace(messages=SimpleNamespace(read_conversation=lambda *a: [])))

    @contextmanager
    def cold_parent(*args, agent_type=None):
        parent = object.__new__(agent_type or AIAgent)
        parent.session_id = 'speech'
        parent.valid_tool_names = {'memory'}
        parents.append(parent)
        yield parent
    monkeypatch.setattr(session_cold_review, 'cold_review_parent', cold_parent)

    def spawn(parent, **kwargs):
        if before_spawn is not None:
            before_spawn()
        parent._background_review_run = SimpleNamespace(request_done=threading.Event(), cancel_requested=threading.Event())
        calls.append(bool(owner._side_agents))
    monkeypatch.setattr(AIAgent, '_spawn_background_review_now', spawn)
    stop = session_external_review._ReviewStop()
    session_external_review._review(owner, SessionRef('p', 'speech'), live, stop)
    assert queue.pending_count() == 1
    item = queue._pending.pop('speech')
    return owner, live, stop, queue, item, parents[0], calls


@pytest.mark.parametrize('retire', ['owner', 'session', 'replace', 'stop'])
def test_deferred_speech_review_never_dispatches_after_retirement(monkeypatch, retire):
    owner, _live, stop, queue, item, _parent, calls = deferred_review(monkeypatch)
    if retire == 'owner':
        owner.retiring = True
    elif retire == 'session':
        owner.sessions.clear()
    elif retire == 'replace':
        owner.sessions['speech'] = object()
    else:
        stop.stopped.set()
    queue._dispatch(item)
    wait_until(lambda: not owner._side_agents, what='obsolete deferred review discarded')
    assert calls == []


def test_deferred_dispatch_is_owned_until_its_request_finishes(monkeypatch):
    owner, _live, _stop, queue, item, parent, calls = deferred_review(monkeypatch)
    try:
        queue._dispatch(item)
        wait_until(lambda: calls, what='deferred fork dispatch')
        assert calls == [True], 'fork spawned outside owner work ledger'
        assert owner._side_agents
    finally:
        run = getattr(parent, '_background_review_run', None)
        if run:
            run.request_done.set()
        wait_until(lambda: not owner._side_agents, what='review released owner ledger')


@pytest.mark.asyncio
async def test_api_review_uses_persisted_route_and_confirmed_lock(api, owner, monkeypatch):
    from gateway import session_policy, session_worker_construct
    from gateway.session_api import bind_api_steps, run_steps
    from gateway.session_external_review import _ReviewStop, _review
    from gateway.session_persistence import authority_persistence
    from tests.fixtures.storage_seed import seed_session
    import run_agent

    seed_session(owner.persistence, 'speech', source='api_server')
    run_steps(bind_api_steps(owner, 'speech'))
    authority_persistence(owner).state_meta.set('gateway.api.settings.v1.speech', json.dumps({
        'requested_model': 'voice-model', 'route': {'provider': 'custom', 'base_url': 'http://127.0.0.1:9999/v1'},
        'confirmed_runtime_lock': True}))
    runner = owner.runner
    runner._provider_routing, runner._prefill_messages = {}, []
    runner._resolve_turn_toolsets = lambda *a: (['memory'], [])
    runner._resolve_session_service_tier = lambda **k: None
    runner._resolve_session_agent_runtime = lambda **k: ('profile-default', {'provider': 'custom'})
    runner._resolve_turn_agent_config = lambda _text, model, runtime, **k: {'model': model, 'runtime': runtime}
    runner._resolve_session_reasoning_config = lambda **k: None
    runner._refresh_fallback_model = lambda: 'profile-fallback'
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda *a: None)
    monkeypatch.setattr(session_worker_construct, 'session_prefill', lambda *a: [])
    calls = []

    def select(runtime, model, **options):
        runtime.update(options['route'])
        assert options['confirmed_runtime_lock']
        return options['requested_model'], None, None, None
    monkeypatch.setattr(api, '_select_agent_runtime', select)
    monkeypatch.setattr('gateway.session_api_turn.check_api_settings', lambda *a: None)

    class Parent:
        valid_tool_names = {'memory'}
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def _spawn_background_review(self, **kwargs):
            pass
        def release_clients(self):
            pass
    monkeypatch.setattr(run_agent, 'AIAgent', Parent)
    live = owner.sessions['speech']
    _review(owner, SessionRef(owner.profile_id, 'speech'), live, _ReviewStop())
    assert calls[0]['model'] == 'voice-model'
    assert calls[0]['base_url'] == 'http://127.0.0.1:9999/v1'
    assert calls[0]['fallback_model'] is None


@pytest.mark.parametrize('dispatch', ['immediate', 'deferred', 'requeued'])
def test_stop_before_run_token_cancels_it_after_spawn(monkeypatch, dispatch):
    from agent import background_review
    entered, release = threading.Event(), threading.Event()

    def before_spawn():
        entered.set()
        assert release.wait(10)

    owner, _live, stop, queue, item, parent, calls = deferred_review(monkeypatch, before_spawn=before_spawn)

    def cancel(agent, **kwargs):
        run = getattr(agent, '_background_review_run', None)
        if run is not None:
            run.cancel_requested.set()
    monkeypatch.setattr(background_review, 'cancel_background_review_for_live_turn', cancel)
    if dispatch == 'requeued':
        release.set()
        queue._dispatch(item)
        wait_until(lambda: calls, what='first deferred request token')
        parent._background_review_run.request_done.set()
        wait_until(lambda: not owner._side_agents, what='first deferred request finished')
        parent._background_review_run = None
        entered.clear()
        release.clear()
    try:
        if dispatch == 'immediate':
            parent._spawn_background_review_now(**item.kwargs)
        else:
            queue._dispatch(item)
        assert entered.wait(10)
        assert owner._side_agents
        stop.hard_interrupt('owner stopping before token creation')
        release.set()
        wait_until(lambda: getattr(parent, '_background_review_run', None), what='late request token installed')
        run = parent._background_review_run
        wait_until(run.cancel_requested.is_set, what='late request token canceled')
        assert owner._side_agents, 'shutdown lost the dispatched fork before its request acknowledged'
    finally:
        release.set()
        run = getattr(parent, '_background_review_run', None)
        if run is not None:
            run.request_done.set()
        wait_until(lambda: not owner._side_agents, what='canceled request released owner ledger')
