"""Plugins read whether Hermes itself started the running turn (``turn_internal()``), which
``turn_client()`` cannot say: a connection that announced nothing reads ``None`` there too."""
import asyncio
import contextvars
import json
from pathlib import Path
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from tests.hermes_cli.test_plugin_host import _home_with_plugins
from tests.hermes_cli.test_plugin_session_client import DESKTOP, _PairedModel, _terminal
from tests.waiting import async_wait_until


def _bound(event):
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli.plugin_session import turn_internal
    tokens = set_session_vars(platform='local', chat_id='s', session_key='s', session_id='agent-s', turn_event=event)
    try:
        return turn_internal()
    finally:
        clear_session_vars(tokens)


def _source(platform):
    from gateway.config import Platform
    from gateway.session import SessionSource
    return SessionSource(platform=Platform(platform), chat_id='c', chat_type='dm', user_id='u')


def test_the_bound_turn_event_decides_and_no_turn_says_nothing():
    import dataclasses

    from gateway.platforms.event import MessageEvent
    from hermes_cli.plugin_session import turn_internal
    assert turn_internal() is None
    assert _bound(None) is None  # a context that bound only a route (a plugin's slash command)
    assert _bound(MessageEvent(text='typed', source=_source('local'))) is False
    assert _bound(MessageEvent(text='[process done]', source=_source('local'), internal=True)) is True
    # A goal continuation is synthetic but not marked internal: it may follow a person's /goal.
    assert _bound(MessageEvent(text='continue', source=_source('telegram'), inherit_channel_pin=True)) is None
    # An API turn is marked internal for trust when it executes; only admission's record counts there.
    api = MessageEvent(text='hi', source=_source('api_server'), internal=True)
    assert _bound(api) is None
    api._started_by_hermes = False
    assert _bound(api) is False
    rewritten = dataclasses.replace(api, text='rewritten by a plugin')
    rewritten._rewritten_from = api
    assert _bound(rewritten) is False


@pytest.mark.asyncio
async def test_a_typed_turn_reads_false_and_hermes_own_admissions_read_true(tmp_path, monkeypatch):
    """Turns through the real FIFO drain (``execute_admission``) of a local session; the handler binds
    the turn's session context the way the turn runner does (``_set_session_env``) and reads the fact."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session import build_session_context
    from gateway.session_authority import initialize_session_authority
    from gateway.session_automation import admit_route_wake
    from gateway.session_contract import SessionActor, Submission
    from gateway.session_local import create_local_session
    from hermes_cli.plugin_session import turn_client, turn_internal

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    seen = {}

    async def answer(event):
        entry = runner.session_store.lookup_by_session_key(authority.sessions[ref.session_id].route)
        tokens = runner._set_session_env(build_session_context(event.source, runner.config, entry, event))
        try:
            seen[event.text] = (turn_internal(), turn_client())
        finally:
            runner._clear_session_env(tokens)
        return 'ok'

    monkeypatch.setattr(runner, '_handle_message', answer)
    try:
        viewer = SessionActor('test-owner', 'default', frozenset(
            {'session:create', 'session:submit', 'session:read'}), 'viewer')
        ref = create_local_session(authority, viewer, dict(request_id='gui', source='gui', model='fixture', toolsets=[]))
        live = authority.sessions[ref.session_id]
        await authority.attach(viewer, ref)
        entry = runner.session_store.lookup_by_session_key(live.route)
        adapter = runner._adapter_for_source(live.source)
        await authority.submit(viewer, Submission('typed', ref, {'text': 'typed by a person'}, 'queue'))
        await async_wait_until(lambda: 'typed by a person' in seen, timeout=10, what="the person's turn")
        assert await runner._admit_plugin_injection(
            adapter, live.source, session_key=live.route, session_id=entry.session_id,
            content='plugin injection', plugin_id='desk') is True
        await async_wait_until(lambda: 'plugin injection' in seen, timeout=10, what='the injected turn')
        assert await admit_route_wake(runner, adapter, live.source, 'wake', 'process-notice') is True
        await async_wait_until(lambda: 'wake' in seen, timeout=10, what='the wake')
    finally:
        runner.session_store.close_all_db_handles()
    assert seen == {'typed by a person': (False, None), 'plugin injection': (True, None), 'wake': (True, None)}


PROBE_PLUGIN = '''
import json
import threading

from hermes_cli import plugin_session

OUT = {out!r}
_LOCK = threading.Lock()


def register(ctx):
    def record(hook):
        def callback(session_id=None, **kw):
            line = {{"hook": hook, "session_id": session_id, "internal": plugin_session.turn_internal(),
                     "client": plugin_session.turn_client(), "ctx": ctx.turn_internal()}}
            with _LOCK, open(OUT, "a", encoding="utf-8") as out:
                out.write(json.dumps(line) + "\\n")
        return callback
    ctx.register_hook("post_llm_call", record("post_llm_call"))
    ctx.register_hook("on_session_end", record("on_session_end"))
'''


def test_a_typed_turn_from_a_connection_that_announced_nothing_is_not_internal(tmp_path, monkeypatch):
    """A real gateway process: two connections, one announced as Desktop and one that announced nothing,
    each submit a turn. Both read ``turn_internal() is False`` in ``post_llm_call`` and
    ``on_session_end``; the unannounced one reads ``turn_client() is None`` all the same."""
    from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket
    from tests.waiting import wait_until

    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    out = tmp_path / 'hooks.jsonl'
    plugin = home / 'plugins' / 'internalprobe'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text(json.dumps({'name': 'internalprobe', 'version': '1.0'}))
    (plugin / '__init__.py').write_text(PROBE_PLUGIN.format(out=str(out)))
    model = ThreadingHTTPServer(('127.0.0.1', 0), _PairedModel)
    model.pair, model.release = threading.Barrier(2, timeout=60), threading.Event()
    thread = threading.Thread(target=model.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{model.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': base},
        'auxiliary': {'title_generation': {'enabled': False}},
        'platform_toolsets': {'gui': []}, 'terminal': {'cwd': str(home)},
        'plugins': {'enabled': ['internalprobe']},
    }))
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
        PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, PYTHONUNBUFFERED='1')

    async def drive(desc):
        async with websocket(home, desc) as ws_a, websocket(home, desc) as ws_b:
            sockets, sessions = {'announced': ws_a, 'unannounced': ws_b}, {}
            assert 'result' in await rpc(ws_a, 'client.capabilities', client=DESKTOP)
            for name, ws in sockets.items():
                created = await rpc(ws, 'session.create', request_id='s-' + name, source='gui', cwd=str(home), toolsets=[])
                sessions[name] = created['result']['session_id']
            for name, ws in sockets.items():
                reply = await rpc(ws, 'prompt.submit', session_id=sessions[name], submission_id='t-' + name,
                                  text='TURN_' + name.upper())
                assert reply.get('result', {}).get('status') == 'queued', reply
            model.release.set()
            await asyncio.to_thread(wait_until, lambda: _terminal(home, ('t-announced', 't-unannounced')), timeout=60)
            return sessions

    try:
        with daemon(root, home, env, barrier=False) as (_proc, desc):
            sessions = asyncio.run(drive(desc))
    finally:
        model.release.set()
        model.shutdown()
        model.server_close()
        thread.join(timeout=5)

    by_session = {sid: name for name, sid in sessions.items()}
    seen = {(line['hook'], by_session[line['session_id']], line['internal'], line['ctx'],
             (line['client'] or {}).get('name')) for line in map(json.loads, out.read_text().splitlines())}
    assert seen == {(hook, name, False, False, 'hermes-desktop' if name == 'announced' else None)
                    for hook in ('post_llm_call', 'on_session_end') for name in sessions}, seen


HOSTED_PLUGIN = '''
from hermes_cli import plugin_session


def register(ctx):
    ctx.register_hook("post_llm_call", lambda **kw: [plugin_session.turn_internal(), ctx.turn_internal()])
'''


@pytest.mark.platforms('any')  # the host is a child process
def test_a_hosted_plugin_reads_it_through_ctx_and_nothing_at_module_level(tmp_path, monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli.plugins import PluginManager
    _home_with_plugins(tmp_path, monkeypatch, {'hostedinternal': HOSTED_PLUGIN}, isolation='host')
    manager = PluginManager()
    manager.discover_and_load()

    def turn(internal):
        event = MessageEvent(text='x', source=_source('local'), internal=internal)
        tokens = set_session_vars(platform='local', chat_id='s', session_key='s', session_id='agent-s', turn_event=event)
        try:
            return manager.invoke_hook('post_llm_call', session_id='agent-s')
        finally:
            clear_session_vars(tokens)
    try:
        assert manager._plugins['hostedinternal'].error is None
        assert contextvars.Context().run(turn, True) == [[None, True]]
        assert contextvars.Context().run(turn, False) == [[None, False]]
        assert contextvars.Context().run(manager.invoke_hook, 'post_llm_call') == [[None, None]]
    finally:
        manager.unload()
        manager._plugin_host().shutdown()


def test_a_delegated_child_reads_nothing_of_its_parent_s_turn():
    from gateway.platforms.event import MessageEvent
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli.plugin_session import turn_internal
    from tools.delegate_tool_child_run import _ChildRun
    seen = []

    class Child:
        session_id, model, provider = 'child', 'some/model', 'openrouter'

        def run_conversation(self, **kwargs):
            seen.append(turn_internal())
            return {'completed': True, 'final_response': 'ok'}

    event = MessageEvent(text='[process done]', source=_source('local'), internal=True)
    tokens = set_session_vars(platform='local', chat_id='s', session_key='s', session_id='agent-s', turn_event=event)
    try:
        seen.append(turn_internal())
        _ChildRun(Child(), SimpleNamespace(_current_task_id=None), 0, 'Probe', None, None).await_child()
    finally:
        clear_session_vars(tokens)
    assert seen == [True, None]
