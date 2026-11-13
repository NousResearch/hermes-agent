"""A plugin's frontend reaches its own Python half through the session authority, and back."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace

import pytest

import hermes_yaml as yaml
from tests.gateway.test_surface_admission import _authority
from tests.waiting import async_wait_until

_PLUGIN = '''
import threading


async def _thread(params, *, session_id, **kw):
    return threading.get_ident()


def register(ctx):
    ctx.register_rpc("pane.state", lambda params, *, session_id, **kw: {"n": params["n"], "session_id": session_id})
    ctx.register_rpc("pane.thread", _thread)
    ctx.register_rpc("pane.nan", lambda params, *, session_id, **kw: {"ratio": float("nan")})
    ctx.register_rpc("pane.where", lambda params, *, session_id, **kw: threading.current_thread().name)
'''

def _documented_chart_plugin():
    """The plugin RPC handler ``plugin-llm-access.md`` teaches, taken verbatim from the page."""
    from pathlib import Path
    page = (Path(__file__).resolve().parents[2] / 'website' / 'docs' / 'developer-guide'
            / 'plugin-llm-access.md').read_text(encoding='utf-8')
    section = page[page.index("#### From the plugin's frontend"):]
    return section[section.index('```python\n') + len('```python\n'):section.index('\n```\n')] + '\n'


class _ChartModel(BaseHTTPRequestHandler):
    """An OpenAI-compatible stub model: records each request, answers ``{"kind": "line"}`` once released."""

    def do_POST(self):
        self.server.requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
        self.server.release.wait(30)  # the test holds the model to see who waits for it
        message = {'role': 'assistant', 'content': '{"kind": "line"}'}
        payload = json.dumps({'id': 'stub', 'object': 'chat.completion', 'model': 'chart-model',
                              'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}],
                              'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


async def _answer(event):
    return 'ok'


def _live(platform, chat_id, **source):
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession
    return LiveSession(SessionSource(platform=platform, chat_id=chat_id, **source), chat_id)


@pytest.mark.asyncio
async def test_plugin_method_is_gated_like_a_session_verb_and_leaves_with_its_plugin(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.session_controls import AuthorityConnection
    from hermes_cli import plugins

    authority = await _authority(tmp_path, monkeypatch, _answer)
    plugin = tmp_path / 'plugins' / 'pane-demo'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text(yaml.safe_dump({'name': 'pane-demo', 'version': '0.1.0'}))
    (plugin / '__init__.py').write_text(_PLUGIN)
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'plugins': {'enabled': ['pane-demo']}}))
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, '_plugin_manager', manager)
    authority.sessions['local-x'] = _live(Platform.LOCAL, 'local-x', user_id='someone-else')
    owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    reader = AuthorityConnection(authority, object(), {'user_id': 'owner', 'capabilities': ['session:read']})

    async def call(connection, **params):
        reply = await connection.dispatch({'id': 1, 'method': 'plugin.pane-demo.pane.state', 'params': params})
        return reply.get('result', reply.get('error'))
    try:
        assert await call(owner, session_id='s', n=3) == {'n': 3, 'session_id': 's'}
        assert (await call(owner, n=4))['n'] == 4
        assert (await call(reader, session_id='s', n=3))['data']['reason'] == 'permission_denied'
        assert (await call(owner, session_id='local-x', n=3))['data']['reason'] == 'permission_denied'
        assert (await call(owner, session_id='s'))['data']['reason'] == 'plugin_rpc_failed'
        # NaN is not JSON: an error, never a frame the client's JSON.parse would drop unanswered.
        nan = await owner.dispatch({'id': 3, 'method': 'plugin.pane-demo.pane.nan', 'params': {}})
        assert nan['error']['data']['reason'] == 'plugin_rpc_failed'
        # A gateway runner's turn pool runs plugin handlers, never the loop's shared default pool.
        import contextvars
        from concurrent.futures import ThreadPoolExecutor
        turn_pool = ThreadPoolExecutor(thread_name_prefix='runner-turn')

        async def on_turn_pool(func, *args):
            return await asyncio.get_running_loop().run_in_executor(
                turn_pool, contextvars.copy_context().run, func, *args)
        authority.runner._run_in_executor_with_context = on_turn_pool
        where = await owner.dispatch({'id': 4, 'method': 'plugin.pane-demo.pane.where', 'params': {}})
        assert where['result'].startswith('runner-turn')
        del authority.runner._run_in_executor_with_context
        turn_pool.shutdown()
        # An async handler's body runs in the worker too, so blocking work in it never stalls the loop.
        reply = await owner.dispatch({'id': 2, 'method': 'plugin.pane-demo.pane.thread', 'params': {}})
        assert reply['result'] != threading.get_ident()
        # A concurrent reload holds the discovery lock: this call waits for it, the loop does not.
        held, release = threading.Event(), threading.Event()

        def reloading():
            with manager._discovery_lock:
                held.set()
                release.wait(10)
        reload = threading.Thread(target=reloading)
        reload.start()
        held.wait(5)
        safety = threading.Timer(5, release.set)  # a blocked loop gets control back only from this
        safety.start()
        pending = asyncio.ensure_future(call(owner, n=5))
        for _ in range(3):
            await asyncio.sleep(0)
        assert not release.is_set() and not pending.done()
        release.set()
        safety.cancel()
        assert (await pending)['n'] == 5
        reload.join(5)
        manager.unload('pane-demo')
        assert (await call(owner, session_id='s', n=3))['code'] == -32601
    finally:
        await owner.close()
        await reader.close()


@pytest.mark.asyncio
async def test_plugin_session_event_reaches_only_that_sessions_viewers(tmp_path, monkeypatch):
    from gateway import run_api
    from gateway.config import Platform
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    authority = await _authority(tmp_path, monkeypatch, _answer)
    authority.sessions['t'] = _live(Platform.TELEGRAM, 't')
    frames = {key: [] for key in ('s', 't')}
    for key, seen in frames.items():
        authority.sessions[key].event_stream.observers.add(seen.append)
    # Compression gave the agent of conversation 's' a new id; its viewers stay on 's'.
    agent = SimpleNamespace(session_id='s-compressed')
    authority.runner._cached_agent_for = lambda route: agent if route == 's' else None
    monkeypatch.setattr(run_api, '_ACTIVE_RUNNER', authority.runner)
    ctx = PluginContext(PluginManifest(name='pane-demo'), PluginManager())

    assert ctx.emit_to_session('s-compressed', 'pane.changed', {'n': 1}) is True
    assert [(f['params']['type'], f['params']['session_id'], f['params']['payload']) for f in frames['s']] == [
        ('plugin.pane-demo.pane.changed', 's', {'n': 1})]
    assert frames['t'] == []
    assert ctx.emit_to_session('nobody', 'pane.changed') is False
    with pytest.raises(ValueError):
        ctx.emit_to_session('s-compressed', 'pane.changed', {'ratio': float('inf')})
    # Session ids are per profile: another profile's 's' never gets this profile's event.
    other = SimpleNamespace(sessions={'s': _live(Platform.TELEGRAM, 's')}, agent=lambda ref: None)
    seen_elsewhere = []
    other.sessions['s'].event_stream.observers.add(seen_elsewhere.append)

    class Registry:  # serves both profiles; this caller's scope is `authority`'s
        def active(self):
            return authority

        def __iter__(self):
            return iter([other, authority])
    authority.runner.session_authorities = Registry()
    assert ctx.emit_to_session('s', 'pane.changed', {'n': 2}) is True
    assert seen_elsewhere == [] and frames['s'][-1]['params']['payload'] == {'n': 2}


def two_profile_runner(launch, secondary):
    """A multiplexed gateway serving ``launch`` and ``secondary``, each with one live session whose
    viewers record their frames: ``(runner, {home: frames})``."""
    from gateway.config import Platform
    from gateway.session_authorities import SessionAuthorities

    registry, frames = SessionAuthorities(launch), {}
    for home, session_id in ((launch, 'in-launch'), (secondary, 'in-secondary')):
        live = _live(Platform.TELEGRAM, session_id)
        frames[home] = []
        live.event_stream.observers.add(frames[home].append)
        registry.add(home, SimpleNamespace(profile_id=str(home), sessions={session_id: live}, agent=lambda ref: None))
    return SimpleNamespace(session_authorities=registry), frames


@pytest.mark.asyncio
async def test_owner_advertises_plugin_rpc_and_the_contract_declares_its_namespace(tmp_path, monkeypatch):
    """A client gates ``plugin.*`` on ``plugin-rpc-v1``: without it, -32601 cannot tell "no such
    plugin method" from "this server has no plugin RPC"."""
    from gateway.session_controls import AuthorityConnection

    authority = await _authority(tmp_path, monkeypatch, _answer)
    owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    try:
        described = await owner.dispatch({'id': 1, 'method': 'runtime.describe', 'params': {}})
    finally:
        await owner.close()
    assert 'plugin-rpc-v1' in described['result']['capabilities']
    from hermes_agent.tui_gateway.contracts import registry
    namespace = registry.CANONICAL_NAMESPACES['plugin.{plugin_id}.{name}']
    assert (namespace.prefix, namespace.capability) == ('plugin.', 'plugin-rpc-v1')
    assert not any(name.startswith(namespace.prefix) for name in registry.CANONICAL_METHODS)


def test_emit_from_a_plugin_thread_reaches_its_own_profile_on_a_multiplexed_gateway(tmp_path, monkeypatch):
    """A thread the plugin started carries no profile scope; ``ctx.emit_to_session`` still looks the
    session up in the plugin's own profile, never the launch profile's."""
    from gateway import run_api
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from hermes_constants import get_hermes_home_override

    launch, secondary = tmp_path / 'launch', tmp_path / 'secondary'
    runner, frames = two_profile_runner(launch, secondary)
    monkeypatch.setattr(run_api, '_ACTIVE_RUNNER', runner)

    def from_thread(ctx, session_id):
        out = []
        worker = threading.Thread(target=lambda: out.append(ctx.emit_to_session(session_id, 'pane.changed', {'n': 1})))
        worker.start()
        worker.join(10)
        return out

    hosted_by_secondary = PluginContext(PluginManifest(name='pane-demo'), PluginManager(str(secondary)))
    assert from_thread(hosted_by_secondary, 'in-secondary') == [True]
    assert [f['params']['type'] for f in frames[secondary]] == ['plugin.pane-demo.pane.changed']
    assert from_thread(hosted_by_secondary, 'in-launch') == [False]  # ids are per profile
    launch_ctx = PluginContext(PluginManifest(name='pane-demo'), PluginManager(str(launch)))
    assert from_thread(launch_ctx, 'in-launch') == [True]
    assert len(frames[launch]) == 1 and len(frames[secondary]) == 1
    assert get_hermes_home_override() is None


@pytest.mark.asyncio
@pytest.mark.parametrize('isolation', ['in_process', pytest.param('host', marks=pytest.mark.platforms('posix'))])
async def test_plugin_frontend_gets_schema_checked_json_from_its_own_task_with_input_redacted(
        tmp_path, monkeypatch, isolation):
    """The route the docs teach for a plugin frontend's structured output, run as the page prints it.

    ``plugin.<id>.chart.write`` answers ``{"job"}`` while the model call is still running, so the
    client's request lane is free (an interrupt or approval answer is not stuck behind the model). Four
    calls run at once and a fifth is refused busy; each result arrives as ``plugin.<id>.chart.written``
    with its job and the parsed JSON. The handler redacts its input and calls its own task: the model,
    on ``auxiliary.<task>``, never sees the secret, even with ``security.redact_secrets: false``
    (``force=True``). A ``Thread.start`` failure gives its slot back. Under ``plugins.isolation: host``
    the plugin's code runs in the plugin host and the same calls answer the same way."""
    import contextvars
    import sys
    from gateway import run_api
    from gateway.session_controls import AuthorityConnection
    from hermes_cli import plugins

    model = ThreadingHTTPServer(('127.0.0.1', 0), _ChartModel)
    model.requests, model.release = [], threading.Event()
    threading.Thread(target=model.serve_forever, daemon=True).start()
    authority = await _authority(tmp_path, monkeypatch, _answer)
    monkeypatch.setattr(run_api, '_ACTIVE_RUNNER', authority.runner)
    written = []
    authority.sessions['s'].event_stream.observers.add(
        lambda frame: frame['params']['type'] == 'plugin.chart-demo.chart.written'
        and written.append(frame['params']['payload']))
    plugin = tmp_path / 'plugins' / 'chart-demo'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text(yaml.safe_dump({'name': 'chart-demo', 'version': '0.1.0'}))
    (plugin / '__init__.py').write_text(_documented_chart_plugin())
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({
        'plugins': {'enabled': ['chart-demo'], 'isolation': isolation},
        'auxiliary': {'chart_writer': {'provider': 'custom', 'model': 'chart-model', 'api_key': 'loopback-only',
                                       'base_url': f'http://127.0.0.1:{model.server_port}/v1'}}}))
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, '_plugin_manager', manager)
    # What `security.redact_secrets: false` sets at boot (the env var also reaches the plugin host).
    monkeypatch.setattr('agent.redact._REDACT_ENABLED', False)
    monkeypatch.setenv('HERMES_REDACT_SECRETS', 'false')
    owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    secret = 'sk-proj-' + 'A1b2C3d4' * 5

    async def write(**params):
        reply = await asyncio.wait_for(owner.dispatch(
            {'id': 1, 'method': 'plugin.chart-demo.chart.write', 'params': params}), 20)
        return reply.get('result', reply.get('error'))

    start = threading.Thread.start
    try:
        assert (await write(text='no session'))['data']['reason'] == 'plugin_rpc_failed'
        if isolation == 'in_process':  # the plugin's own thread cannot start: its slot comes back
            def refuse(thread):
                if isinstance(getattr(thread._target, '__self__', None), contextvars.Context):
                    raise RuntimeError("can't start new thread")
                return start(thread)
            monkeypatch.setattr(threading.Thread, 'start', refuse)
            for _ in range(4):
                assert (await write(session_id='s', text='x'))['data']['reason'] == 'plugin_rpc_failed'
            monkeypatch.setattr(threading.Thread, 'start', start)
        jobs = [(await write(session_id='s', text=f'revenue by month, key {secret}'))['job'] for _ in range(4)]
        # All four answered while their model calls are still held: none waits on the model.
        await async_wait_until(lambda: len(model.requests) == 4, timeout=20, what='the four held model calls')
        assert not model.release.is_set()
        assert (await write(session_id='s', text='fifth'))['data']['reason'] == 'plugin_rpc_failed'
        assert written == []
        model.release.set()
        await async_wait_until(lambda: len(written) == 4, timeout=20, what='four chart.written events')
        assert sorted(event['job'] for event in written) == sorted(jobs)
        assert all(event['chart'] == {'kind': 'line'} for event in written), written
        sixth = (await write(session_id='s', text='sixth'))['job']
        await async_wait_until(lambda: len(written) == 5, timeout=20, what='the sixth chart.written event')
        assert written[-1] == {'job': sixth, 'chart': {'kind': 'line'}}
        hosted = not any('chart' in name for name in sys.modules)
    finally:
        monkeypatch.setattr(threading.Thread, 'start', start)
        model.release.set()
        await owner.close()
        model.shutdown()
        if isolation == 'host':
            manager.unload()
            manager._plugin_host().shutdown()
    assert hosted is (isolation == 'host')
    assert len(model.requests) == 5
    for request in model.requests:
        assert request['model'] == 'chart-model'
        assert request['response_format']['type'] == 'json_schema'
    sent = json.dumps([request['messages'] for request in model.requests])
    assert 'revenue by month' in sent and secret not in sent
