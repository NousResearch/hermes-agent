"""A real admitted Telegram turn invokes the model tool and settles only after its receipt is terminal."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import pytest_asyncio
import hermes_bootstrap

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_authority import initialize_session_authority
from hermes_state_runtime import list_session_admissions


class SyntheticTelegram(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {'id': chat_id, 'name': 'Synthetic topic', 'type': 'group'}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((content, metadata))
        return SendResult(success=True, message_id=str(len(self.sent)))


class Provider(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(body)
        if 'messages' not in body:
            self.send_error(400, 'Synthetic provider supports chat completions only')
            return
        messages = body['messages']
        if any(m['role'] == 'tool' for m in messages):
            self.server.at_final.set()
            if not self.server.release_final.wait(timeout=15):
                self.send_error(400, 'Synthetic barrier timed out')
                return
            if self.server.fail_final:
                self.send_error(400, 'Synthetic failure after tool call')
                return
        last_user = next(str(m.get('content', '')) for m in reversed(messages) if m['role'] == 'user')
        request_handoff = 'handoff please' in last_user and not any(m['role'] == 'tool' for m in messages)
        message = {'role': 'assistant', 'content': None if request_handoff else 'Provider turn complete.'}
        if request_handoff:
            message['tool_calls'] = [{'id': 'handoff-call', 'type': 'function', 'function': {
                'name': 'fresh_session', 'arguments': json.dumps({'handoff': 'Continue task: inspect synthetic.txt; verification pending.'})}}]
        choice = {'index': 0, 'message': message, 'finish_reason': 'tool_calls' if request_handoff else 'stop'}
        frame = {'id': 'provider', 'object': 'chat.completion', 'created': 1, 'model': body['model'], 'choices': [choice],
                 'usage': {'prompt_tokens': 200, 'completion_tokens': 20, 'total_tokens': 220}}
        if body.get('stream'):
            choice['delta'] = choice.pop('message')
            payload = ('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n').encode()
            content_type = 'text/event-stream'
        else:
            payload, content_type = json.dumps(frame).encode(), 'application/json'
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def provider():
    server = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
    server.requests = []
    server.at_final = threading.Event()
    server.release_final = threading.Event()
    server.release_final.set()
    server.fail_final = False
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server
    finally:
        server.release_final.set()
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()


@pytest_asyncio.fixture
async def admitted(tmp_path, monkeypatch, provider):
    from agent import title_generator
    import hermes_state
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', tmp_path / 'state.db')
    from hermes_cli import _early_recovery
    monkeypatch.setattr(_early_recovery, 'restore_interrupted_pull', lambda: False)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-key')
    monkeypatch.setattr(title_generator, '_auto_title_enabled', lambda: False)
    cfg = {'model': {'default': 'gpt-4.1', 'provider': 'openai', 'base_url': f'http://127.0.0.1:{provider.server_port}/v1'},
           'platform_toolsets': {'telegram': ['terminal']}, 'compression': {'enabled': False},
           'memory': {'memory_enabled': False, 'user_profile_enabled': False},
           'display': {'tool_progress': 'off', 'streaming': False, 'interim_assistant_messages': False},
           'approvals': {'destructive_slash_confirm': False}, 'terminal': {'cwd': str(tmp_path)}}
    (tmp_path / 'config.yaml').write_text(json.dumps(cfg))
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=False))
    adapter = SyntheticTelegram()
    runner.adapters[Platform.TELEGRAM] = adapter
    adapter.set_message_handler(runner._handle_message)
    adapter.gateway_runner = runner
    monkeypatch.setattr(runner, '_is_user_authorized_for_source', lambda source, **kwargs: True)
    monkeypatch.setattr(runner, '_resolve_session_agent_runtime', lambda **kwargs: ('gpt-4.1', {
        'provider': 'openai', 'api_mode': 'chat_completions', 'api_key': 'synthetic-key',
        'base_url': f'http://127.0.0.1:{provider.server_port}/v1'}))
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='synthetic-owner')
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='-100123', user_id='456', chat_type='group', thread_id='789')
    source._transport_adapter_ref = adapter
    try:
        yield SimpleNamespace(runner=runner, adapter=adapter, authority=authority, source=source, provider=provider)
    finally:
        for live in list(authority.sessions.values()):
            if live.task is not None and not live.task.done():
                live.task.cancel()
                await asyncio.gather(live.task, return_exceptions=True)
        for key in list(runner._agent_cache):
            runner._evict_cached_agent(key)


@pytest.mark.asyncio
async def test_admitted_model_choice_publishes_after_terminal_and_next_message_reads_handoff(admitted):
    a = admitted
    key = a.runner._session_key_for_source(a.source)
    parent = a.runner.session_store.get_or_create_session(a.source).session_id
    reply = await asyncio.wait_for(a.adapter._process_message_background(
        MessageEvent(text='handoff please', source=a.source, message_id='m1'), key), 30)
    rows = list_session_admissions(a.authority.db, session_id=parent, pending_only=False)
    assert len(rows) == 1 and rows[0]['status'] == 'terminal' and rows[0]['outcome'] == 'completed'
    child = a.runner.session_store.peek_session_id(key)
    assert child != parent, (reply, rows, a.adapter.sent)
    assert a.authority.db.get_session(parent)['end_reason'] == 'session_handoff'
    model_request = next(r for r in a.provider.requests if r.get('tools'))
    assert any(tool['function']['name'] == 'fresh_session' for tool in model_request['tools'])
    from tests.waiting import async_wait_until
    assert await async_wait_until(lambda: any('Continuing in a fresh session' in t for t, _ in a.adapter.sent))
    assert [text for text, _ in a.adapter.sent][-2:] == [
        'Provider turn complete.', 'Continuing in a fresh session in this topic.']
    assert sum('Continuing in a fresh session' in text for text, _ in a.adapter.sent) == 1
    await asyncio.wait_for(a.runner._handle_message(MessageEvent(text='next step', source=a.source, message_id='m2')), 30)
    assert a.runner.session_store.peek_session_id(key) == child
    request = next(r for r in reversed(a.provider.requests) if r.get('tools'))
    assert any('inspect synthetic.txt' in str(m.get('content', '')) for m in request['messages'])
    assert not any(m.get('tool_call_id') == 'handoff-call' for m in request['messages'])


@pytest.mark.asyncio
async def test_no_model_choice_keeps_the_admitted_conversation(admitted):
    a = admitted
    key = a.runner._session_key_for_source(a.source)
    parent = a.runner.session_store.get_or_create_session(a.source).session_id
    await asyncio.wait_for(a.runner._handle_message(MessageEvent(text='ordinary request', source=a.source, message_id='ordinary')), 30)
    assert a.runner.session_store.peek_session_id(key) == parent
    assert a.authority.db.get_session(parent)['ended_at'] is None
    assert not any('Continuing in a fresh session' in text for text, _ in a.adapter.sent)


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['publication', 'json', 'json_save', 'new', 'new_crash'])
async def test_admitted_faults_and_restart_keep_one_durable_route(admitted, monkeypatch, fault):
    from gateway.fresh_session import publication
    from gateway.session import SessionStore, AsyncSessionStore
    from tools import approval
    a = admitted
    key = a.runner._session_key_for_source(a.source)
    store = a.runner.session_store
    parent = store.get_or_create_session(a.source).session_id
    store.set_session_yolo(key, True)
    approval.approve_session(key, 'synthetic-dangerous-command')
    approval.enable_session_yolo(key)
    publish = publication.publish
    switch = store.switch_session
    observations = []

    def inspect_publish(db, **kwargs):
        rows = list_session_admissions(db, session_id=parent, pending_only=False)
        assert rows[0]['status'] == 'terminal' and rows[0]['outcome'] == 'completed'
        observations.append(kwargs['child_id'])
        if fault == 'publication':
            # Fail inside the real transaction, after the child and summary were inserted.
            insert = db._insert_message_rows
            def fail_insert(conn, sid, messages):
                insert(conn, sid, messages)
                raise OSError('synthetic transactional publication fault')
            with monkeypatch.context() as patch:
                patch.setattr(db, '_insert_message_rows', fail_insert)
                return publish(db, **kwargs)
        return publish(db, **kwargs)

    def fail_adoption(route, child, **kwargs):
        if fault == 'json_save':
            with monkeypatch.context() as patch:
                patch.setattr(store, '_save', lambda: (_ for _ in ()).throw(OSError('handoff JSON fault')))
                return switch(route, child, **kwargs)
        if fault == 'new_crash':
            a.runner._clear_session_boundary_security_state(route)
            with monkeypatch.context() as patch:
                patch.setattr(store, '_save', lambda: (_ for _ in ()).throw(OSError('new JSON fault')))
                store.reset_session(route)
        if fault == 'new':
            store.reset_session(route)
            a.runner._clear_session_boundary_security_state(route)
            return switch(route, child, **kwargs)
        raise OSError('synthetic JSON adoption fault')

    with monkeypatch.context() as patch:
        patch.setattr(publication, 'publish', inspect_publish)
        if fault != 'publication':
            patch.setattr(store, 'switch_session', fail_adoption)
        await asyncio.wait_for(a.runner._handle_message(MessageEvent(text='handoff please', source=a.source, message_id='fault')), 30)
    assert len(observations) == 1
    child = observations[0]
    assert a.authority.db.get_messages(parent)
    assert not any('Continuing in a fresh session' in text for text, _ in a.adapter.sent)
    if fault == 'publication':
        assert a.authority.db.get_session(child) is None
        assert a.authority.db.get_session(parent)['ended_at'] is None
        expected = parent
    elif fault in ('new', 'new_crash'):
        assert a.authority.db.get_session(child)['end_reason'] == 'superseded_by_new'
        expected = a.authority.db.get_telegram_topic_binding(chat_id=a.source.chat_id,
            thread_id=a.source.thread_id, profile_name=a.source.profile)['session_id']
        assert expected not in (parent, child)
    else:
        assert a.authority.db.get_session(parent)['end_reason'] == 'session_handoff'
        expected = child
    if fault == 'json_save':
        assert store.peek_session_id(key) == parent
    # Reconstruct the routing mirror and authority with a new owner epoch, using only test state.
    restored = SessionStore(store.sessions_dir, a.runner.config)
    restored._db = a.authority.db
    a.runner.session_store = restored
    a.runner._async_session_store = AsyncSessionStore(restored)
    new_authority = await initialize_session_authority(a.runner, profile_id=a.authority.profile_id, instance_id='synthetic-restart')
    a.authority = new_authority
    await asyncio.wait_for(a.runner._handle_message(MessageEvent(text='after restart', source=a.source, message_id='restart')), 30)
    assert restored.peek_session_id(key) == expected
    if fault != 'publication':
        assert restored.lookup_by_session_key(key).yolo is None
        assert not approval.is_session_yolo_enabled(key)
        assert not approval.is_approved(key, 'synthetic-dangerous-command')
    if fault in ('json', 'json_save'):
        request = next(r for r in reversed(a.provider.requests) if r.get('tools'))
        assert any('inspect synthetic.txt' in str(m.get('content', '')) for m in request['messages'])
        assert not any(m.get('tool_call_id') == 'handoff-call' for m in request['messages'])


@pytest.mark.asyncio
async def test_queued_admitted_input_prevents_handoff_and_is_delivered(admitted):
    from tests.waiting import async_wait_until
    a = admitted
    key = a.runner._session_key_for_source(a.source)
    parent = a.runner.session_store.get_or_create_session(a.source).session_id
    a.provider.release_final.clear()
    first = asyncio.create_task(a.runner._handle_message(MessageEvent(text='handoff please', source=a.source, message_id='q1')))
    second = None
    try:
        assert await async_wait_until(a.provider.at_final.is_set, timeout=10)
        second = asyncio.create_task(a.runner._handle_message(MessageEvent(text='following input', source=a.source, message_id='q2')))
        assert await async_wait_until(lambda: any(r['status'] == 'queued' for r in list_session_admissions(
            a.authority.db, session_id=parent, pending_only=False)), timeout=10)
    finally:
        a.provider.release_final.set()
        await asyncio.wait_for(asyncio.gather(first, *([second] if second else [])), 30)
    rows = list_session_admissions(a.authority.db, session_id=parent, pending_only=False)
    assert len(rows) == 2 and all(r['status'] == 'terminal' for r in rows)
    assert a.runner.session_store.peek_session_id(key) == parent
    assert a.authority.db.get_session(parent)['ended_at'] is None
    assert not any('Continuing in a fresh session' in text for text, _ in a.adapter.sent)
    assert any('following input' in str(r.get('messages')) for r in a.provider.requests)


@pytest.mark.asyncio
async def test_failed_provider_turn_after_intent_does_not_publish(admitted):
    a = admitted
    key = a.runner._session_key_for_source(a.source)
    parent = a.runner.session_store.get_or_create_session(a.source).session_id
    a.provider.fail_final = True
    await asyncio.wait_for(a.runner._handle_message(MessageEvent(text='handoff please', source=a.source, message_id='failed')), 30)
    assert a.provider.at_final.is_set()
    assert a.runner.session_store.peek_session_id(key) == parent
    assert a.authority.db.get_session(parent)['ended_at'] is None
    assert not any('Continuing in a fresh session' in text for text, _ in a.adapter.sent)


@pytest.mark.asyncio
async def test_goal_arriving_during_publication_resolves_the_successor(admitted, monkeypatch):
    from gateway.fresh_session import publication
    from hermes_cli import goals
    from tests.waiting import async_wait_until
    a = admitted
    key = a.runner._session_key_for_source(a.source)
    parent = a.runner.session_store.get_or_create_session(a.source).session_id
    publishing, release = threading.Event(), threading.Event()
    original = publication.publish
    def blocked(db, **kwargs):
        publishing.set()
        assert release.wait(timeout=15)
        return original(db, **kwargs)
    monkeypatch.setattr(publication, 'publish', blocked)
    handoff = asyncio.create_task(a.runner._handle_message(MessageEvent(text='handoff please', source=a.source, message_id='g1')))
    goal = None
    try:
        assert await async_wait_until(publishing.is_set, timeout=10)
        goal = asyncio.create_task(a.runner._handle_goal_command(MessageEvent(text='/goal inspect synthetic goal', source=a.source)))
        # Observe the command waiting at the same lock before it can resolve its manager.
        lock = a.runner._fresh_session_locks[key]
        assert await async_wait_until(lambda: bool(lock._waiters), timeout=10)
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(handoff, *([goal] if goal else [])), 30)
    child = a.runner.session_store.peek_session_id(key)
    assert child != parent
    assert goals.GoalManager(parent).state is None
    assert goals.GoalManager(child).state is not None
    assert a.authority.db.get_meta(f'goal:{child}')


@pytest.mark.asyncio
async def test_incoming_message_during_publication_runs_in_live_successor(admitted, monkeypatch):
    from gateway.fresh_session import publication
    from tests.waiting import async_wait_until
    a = admitted
    key = a.runner._session_key_for_source(a.source)
    parent = a.runner.session_store.get_or_create_session(a.source).session_id
    from tools import approval
    approval.approve_session(key, 'synthetic-dangerous-command')
    approval.enable_session_yolo(key)
    a.runner.session_store.set_session_yolo(key, True)
    publishing, release, resolving = threading.Event(), threading.Event(), threading.Event()
    original = publication.publish
    resolve = a.runner.session_store.get_or_create_session
    def blocked(db, **kwargs):
        publishing.set()
        assert release.wait(timeout=15)
        return original(db, **kwargs)
    def incoming(*args, **kwargs):
        resolving.set()
        entry = resolve(*args, **kwargs)
        assert not approval.is_session_yolo_enabled(key)
        assert not approval.is_approved(key, 'synthetic-dangerous-command')
        return entry
    monkeypatch.setattr(publication, 'publish', blocked)
    first = asyncio.create_task(a.runner._handle_message(MessageEvent(text='handoff please', source=a.source, message_id='p1')))
    second = None
    try:
        assert await async_wait_until(publishing.is_set, timeout=10)
        monkeypatch.setattr(a.runner.session_store, 'get_or_create_session', incoming)
        second = asyncio.create_task(a.runner._handle_message(MessageEvent(text='publication follower', source=a.source, message_id='p2')))
        assert await async_wait_until(resolving.is_set, timeout=10)
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(first, *([second] if second else [])), 30)
    child = a.runner.session_store.peek_session_id(key)
    assert child != parent
    assert a.authority.db.get_session(child)['ended_at'] is None
    assert a.authority.db.get_session(parent)['end_reason'] == 'session_handoff'
    request = next(r for r in reversed(a.provider.requests) if r.get('tools'))
    assert any('inspect synthetic.txt' in str(m.get('content', '')) for m in request['messages'])
    assert any('publication follower' in str(m.get('content', '')) for m in request['messages'])
    assert await async_wait_until(lambda: any('Continuing in a fresh session' in t for t, _ in a.adapter.sent))
    assert sum('Continuing in a fresh session' in text for text, _ in a.adapter.sent) == 1


@pytest.mark.asyncio
async def test_real_background_process_keeps_admitted_handoff_on_parent(admitted):
    import shlex
    import sys
    from tools.process_registry import process_registry
    a = admitted
    key = a.runner._session_key_for_source(a.source)
    parent = a.runner.session_store.get_or_create_session(a.source).session_id
    command = shlex.quote(sys.executable) + ' -c ' + shlex.quote('import threading; threading.Event().wait()')
    process = process_registry.spawn_local(command, session_key=key, task_id='synthetic-process', owner_task_id='synthetic-process')
    try:
        assert process_registry.has_active_for_session(key)
        await asyncio.wait_for(a.runner._handle_message(MessageEvent(text='handoff please', source=a.source, message_id='process')), 30)
        assert a.runner.session_store.peek_session_id(key) == parent
        assert a.authority.db.get_session(parent)['ended_at'] is None
        assert process_registry.has_active_for_session(key)
        assert not any('Continuing in a fresh session' in text for text, _ in a.adapter.sent)
    finally:
        process_registry.kill_process(process.id)


@pytest.mark.asyncio
async def test_publication_precedes_buffered_delivery_and_note_survives_following_admission(admitted, monkeypatch):
    from tests.waiting import async_wait_until
    a = admitted
    key = a.runner._session_key_for_source(a.source)
    parent = a.runner.session_store.get_or_create_session(a.source).session_id
    entered, release = asyncio.Event(), asyncio.Event()
    send = a.adapter.send
    async def buffered(chat_id, content, **kwargs):
        if content == 'Provider turn complete.' and not entered.is_set():
            entered.set()
            await release.wait()
        return await send(chat_id, content, **kwargs)
    monkeypatch.setattr(a.adapter, 'send', buffered)
    first = asyncio.create_task(a.adapter._process_message_background(
        MessageEvent(text='handoff please', source=a.source, message_id='buffered-1'), key))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 15)
        child = a.runner.session_store.peek_session_id(key)
        assert child != parent
        assert a.authority.db.get_session(parent)['end_reason'] == 'session_handoff'
        rows = list_session_admissions(a.authority.db, session_id=parent, pending_only=False)
        assert rows[0]['status'] == 'terminal'
        assert not any('Continuing in a fresh session' in text for text, _ in a.adapter.sent)
        second = asyncio.create_task(a.runner._handle_message(
            MessageEvent(text='Buffered follower input.', source=a.source, message_id='buffered-2')))
        assert await async_wait_until(lambda: bool(list_session_admissions(
            a.authority.db, session_id=child, pending_only=False)), timeout=10)
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(first, *([second] if second else [])), 30)
    assert await async_wait_until(lambda: any('Continuing in a fresh session' in text for text, _ in a.adapter.sent))
    assert [text for text, _ in a.adapter.sent][-2:] == [
        'Provider turn complete.', 'Continuing in a fresh session in this topic.']
    assert sum('Continuing in a fresh session' in text for text, _ in a.adapter.sent) == 1
    assert a.runner.session_store.peek_session_id(key) == child


@pytest.mark.asyncio
async def test_failed_final_delivery_does_not_send_a_handoff_note(admitted, monkeypatch):
    from tests.waiting import async_wait_until
    a = admitted
    key = a.runner._session_key_for_source(a.source)
    parent = a.runner.session_store.get_or_create_session(a.source).session_id
    attempts = []
    async def rejected_send(chat_id, content, **kwargs):
        attempts.append(content)
        return SendResult(success=False, error='Synthetic delivery refusal')
    monkeypatch.setattr(a.adapter, 'send', rejected_send)
    await asyncio.wait_for(a.adapter._process_message_background(
        MessageEvent(text='handoff please', source=a.source, message_id='delivery-failed'), key), 30)
    assert await async_wait_until(lambda: all(live.task is None or live.task.done() for live in a.authority.sessions.values()))
    assert a.runner.session_store.peek_session_id(key) != parent
    assert a.authority.db.get_session(parent)['end_reason'] == 'session_handoff'
    assert attempts and not any('Continuing in a fresh session' in text for text in attempts)
    assert not a.adapter.sent
