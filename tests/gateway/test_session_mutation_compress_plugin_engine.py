"""Gateway ``/compress`` runs through the session's configured plugin context engine (#143587).

The unified gateway summarized every manual ``/compress`` with a native ``ContextCompressor`` and
published the rewrite, so a plugin engine never compressed its own history nor saw the compaction
boundary; the next agent's engine then re-stored the retained replies as new rows.
"""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading

import pytest
from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket
from tests.gateway.test_session_mutation_compress import SummaryModel
from tests.waiting import async_wait_until

ENGINE = '''
import json, os
from agent.context_engine import ContextEngine

def _record(event, **fields):
    with open(os.environ["ENGINE_LOG"], "a") as log:
        log.write(json.dumps({"event": event, **fields}) + "\\n")

class RecordingEngine(ContextEngine):
    @property
    def name(self):
        return "recording_engine"

    def update_from_response(self, usage):
        pass

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False, memory_context=""):
        _record("compress", count=len(messages), focus_topic=focus_topic, force=force)
        return [messages[0], {"role": "assistant", "content": "ENGINE_SUMMARY"}, *messages[-2:]]

    def on_session_start(self, session_id, **kwargs):
        _record("start", session_id=session_id, boundary_reason=kwargs.get("boundary_reason"),
                old_session_id=kwargs.get("old_session_id"))

    def on_session_end(self, session_id, messages):
        _record("end", session_id=session_id)

def register(ctx):
    ctx.register_context_engine(RecordingEngine())
'''


@pytest.mark.parametrize('resident', [True, False], ids=['resident-agent', 'cold-session'])
def test_manual_compress_uses_configured_plugin_engine(tmp_path, resident):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    (home / 'plugins' / 'recording_engine').mkdir(parents=True)
    (home / 'plugins' / 'recording_engine' / '__init__.py').write_text(ENGINE)
    log = tmp_path / 'engine.jsonl'
    peer = ThreadingHTTPServer(('127.0.0.1', 0), SummaryModel)
    peer.requests = []
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    cfg = {'gateway': {'multiplex_profiles': False}, 'context': {'engine': 'recording_engine'},
           'model': {'provider': 'custom', 'default': 'original', 'base_url': url},
           'auxiliary': {'title_generation': {'enabled': False},
                         'compression': {'provider': 'custom', 'model': 'summary', 'base_url': url}},
           'platform_toolsets': {'cli': []}, 'compression': {'in_place': True}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    (home / 'models_dev_cache.json').write_text(json.dumps({'custom': {'id': 'custom', 'models': {
        name: {'id': name, 'name': name, 'limit': {'context': 1000000, 'output': 4000}}
        for name in ('original', 'summary')}}}))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1', ENGINE_LOG=str(log))

    async def call(ws, method, **params):
        response = await rpc(ws, method, **params)
        assert 'result' in response, response
        return response['result']

    async def history(desc):
        async with websocket(home, desc) as ws:
            sid = (await call(ws, 'session.create', source='cli', request_id='owner', toolsets=[],
                              model='original', provider='custom', base_url=url))['session_id']
            for i in range(3):
                await call(ws, 'prompt.submit', session_id=sid, input_id=f'H{i}', text=f'HISTORY_{i}')

                def terminal(text=f'H{i}'):
                    with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
                        row = db.execute('SELECT status FROM session_admissions WHERE request_id=?', (text,)).fetchone()
                    return row and row[0] == 'terminal'
                await async_wait_until(terminal, timeout=30, what=f'H{i} terminal')
            return sid

    async def compress(desc, sid):
        async with websocket(home, desc) as ws:
            before = await call(ws, 'session.resume', session_id=sid)
            result = await call(ws, 'session.mutate', session_id=sid, request_id='compress', operation='compress',
                                payload={'focus': 'the plan'}, expected_revision=before['revision'],
                                expected_generation=before['execution_generation'])
            return result, await call(ws, 'session.resume', session_id=sid)

    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            sid = asyncio.run(history(desc))
            if resident:
                result, after = asyncio.run(compress(desc, sid))
        if not resident:  # a fresh gateway holds no agent for the session
            with daemon(root, home, env, barrier=False) as (_, desc):
                result, after = asyncio.run(compress(desc, sid))
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)

    events = [json.loads(line) for line in log.read_text().splitlines()]
    compresses = [e for e in events if e['event'] == 'compress']
    assert compresses and compresses[-1]['focus_topic'] == 'the plan' and compresses[-1]['force'] is True
    assert 'ENGINE_SUMMARY' in json.dumps(after['messages'])
    assert not any('context checkpoint' in json.dumps(r.get('messages')) for r in peer.requests)  # no native summary
    boundary = [e for e in events if e['event'] == 'start' and e['boundary_reason'] == 'compression']
    assert boundary == [{'event': 'start', 'session_id': result['target_session_id'],
                         'boundary_reason': 'compression', 'old_session_id': result['previous_target_session_id']}]


@pytest.mark.parametrize('resident_mode', [True, False], ids=['resident-agent', 'cold-session'])
def test_prepare_compress_retains_context_compressor_subclass_engine(monkeypatch, tmp_path, resident_mode):
    from types import SimpleNamespace
    from agent.context_compressor import ContextCompressor
    import gateway.session_mutation_compress as smc

    events = []

    class SubclassEngine(ContextCompressor):
        @property
        def name(self):
            return 'subclass_engine'

        def compress(self, messages, current_tokens=None, focus_topic=None, force=False, memory_context=''):
            events.append(('compress', focus_topic, force))
            if focus_topic == 'noop':
                return list(messages)
            return [messages[0], {'role': 'assistant', 'content': 'SUBCLASS_SUMMARY', '_compressed_summary': True}]

        def on_session_start(self, session_id, **kwargs):
            events.append(('start', session_id, kwargs.get('boundary_reason')))

        def on_session_end(self, session_id, messages):
            events.append(('end', session_id))

    engine = SubclassEngine(model='test-model', quiet_mode=True)
    builtin = ContextCompressor(model='test-model', quiet_mode=True)

    # Exact ContextCompressor on resident is treated as built-in (None), whereas a subclass is retained.
    assert smc._plugin_engine(None, None, {}, 'test-model', {}, SimpleNamespace(context_compressor=builtin), 's1') is None
    assert smc._plugin_engine(None, None, {}, 'test-model', {}, SimpleNamespace(context_compressor=engine), 's1') is engine

    history = [
        {'role': 'user', 'content': 'u1 ' * 50},
        {'role': 'assistant', 'content': 'a1 ' * 50},
        {'role': 'user', 'content': 'u2 ' * 50},
        {'role': 'assistant', 'content': 'a2 ' * 50},
    ]
    resident = SimpleNamespace(context_compressor=engine, _cached_system_prompt='', tools=None) if resident_mode else None

    class FakePolicy:
        safe_mode = False

        def config(self, authority):
            return {'context': {'engine': 'subclass_engine'}}

    from contextlib import asynccontextmanager, contextmanager

    @contextmanager
    def fake_policy_scope(policy, authority=None):
        yield

    @asynccontextmanager
    async def fake_async_profile_scope(path):
        yield

    import gateway.session_policy as sp
    import agent.agent_init as ai
    import gateway.run as gr

    monkeypatch.setattr(sp, 'restore_policy', lambda raw: FakePolicy())
    monkeypatch.setattr(sp, 'policy_scope', fake_policy_scope)
    monkeypatch.setattr(ai, '_select_context_engine', lambda cfg: engine)
    monkeypatch.setattr(gr, '_async_profile_runtime_scope', fake_async_profile_scope)

    authority = SimpleNamespace(
        profile_id=str(tmp_path),
        runner=SimpleNamespace(
            _resolve_session_agent_runtime=lambda **kw: ('test-model', {}),
            _cached_agent_for=lambda route: resident,
        ),
        db=SimpleNamespace(
            get_messages_as_conversation=lambda target: list(history),
            get_session=lambda target: {},
        ),
    )
    live = SimpleNamespace(source='cli', route='cli:local')
    prepared = {'snapshot': {'target': 'sess-1', 'receipt': {'policy': {}}}}

    out = asyncio.run(smc.prepare_compress(authority, live, {'focus': 'topic'}, prepared))
    assert ('compress', 'topic', True) in events
    assert any(m.get('content') == 'SUBCLASS_SUMMARY' for m in out['messages'])
    assert out['engine_owner'].context_compressor is engine
    assert out['engine_owner'].owned is (not resident_mode)

    asyncio.run(smc.finish_compress_engine(
        authority, out, {'target_session_id': 'sess-2', 'previous_target_session_id': 'sess-1'}
    ))
    assert ('start', 'sess-2', 'compression') in events
    if not resident_mode:
        assert ('end', 'sess-2') in events
    else:
        assert ('end', 'sess-2') not in events

    # Also verify noop cleanup calls on_session_end on cold-session subclass engine
    events.clear()
    noop_out = asyncio.run(smc.prepare_compress(authority, live, {'focus': 'noop'}, prepared))
    assert noop_out['status'] == 'noop'
    if not resident_mode:
        assert ('end', 'sess-1') in events
    else:
        assert ('end', 'sess-1') not in events

