"""Plugins read the Hermes producer of the running turn (``turn_origin()``), which ``turn_client()``
cannot say: a connection that announced nothing reads ``None`` there too. Only a process
notification and a delegation result name one; every other turn reads ``None``."""
import asyncio
import contextvars
import json
from pathlib import Path
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from tests.hermes_cli.plugin_host_support import _home_with_plugins
from tests.hermes_cli.test_plugin_session_client import DESKTOP, _PairedModel, _terminal
from tests.waiting import async_wait_until


def _bound(event):
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli.plugin_session import turn_origin
    tokens = set_session_vars(platform='local', chat_id='s', session_key='s', session_id='agent-s', turn_event=event)
    try:
        return turn_origin()
    finally:
        clear_session_vars(tokens)


def _named(event, origin):
    event.set_turn_origin(origin)
    return event


def _source(platform):
    from gateway.config import Platform
    from gateway.session import SessionSource
    return SessionSource(platform=Platform(platform), chat_id='c', chat_type='dm', user_id='u')


def test_the_bound_turn_event_decides_and_no_turn_says_nothing():
    import dataclasses

    from gateway.platforms.event import MessageEvent
    from hermes_cli.plugin_session import turn_origin
    assert turn_origin() is None
    assert _bound(None) is None  # a context that bound only a route (a plugin's slash command)
    assert _bound(MessageEvent(text='typed', source=_source('local'))) is None
    # ``internal`` is execution trust, not an origin: a restart resume or Bot Chat delivery is internal too.
    assert _bound(MessageEvent(text='', source=_source('local'), internal=True)) is None
    for origin in ('process_notification', 'delegation_result'):
        assert _bound(_named(MessageEvent(text='[done]', source=_source('local'), internal=True), origin)) == origin
    for unknown in ('cron', 'kanban', '', True, 1):
        assert _bound(_named(MessageEvent(text='x', source=_source('local'), internal=True), unknown)) is None
    api = _named(MessageEvent(text='[done]', source=_source('api_server'), internal=True), 'process_notification')
    rewritten = dataclasses.replace(api, text='rewritten by a plugin')
    rewritten._rewritten_from = api
    assert _bound(rewritten) == 'process_notification'


@pytest.mark.asyncio
async def test_only_process_and_delegation_notices_name_their_producer(tmp_path, monkeypatch):
    """Turns through the real FIFO drain (``execute_admission``) of a local session; the handler binds
    the turn's session context the way the turn runner does (``_set_session_env``) and reads the fact.
    The notices enter through the producer (``_inject_watch_notification``) and come back out of
    their committed rows."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session import build_session_context
    from gateway.session_authority import initialize_session_authority
    from gateway.session_automation import admit_route_wake
    from gateway.session_contract import SessionActor, Submission
    from gateway.session_local import create_local_session
    from hermes_cli.plugin_session import turn_client, turn_origin

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    seen = {}

    def ran(word):
        return [answer for text, answer in seen.items() if text == word or text.startswith('[' + word + ']')]

    async def answer(event):
        entry = runner.session_store.lookup_by_session_key(authority.sessions[ref.session_id].route)
        tokens = runner._set_session_env(build_session_context(event.source, runner.config, entry, event))
        try:
            seen[event.text] = (turn_origin(), turn_client())
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
        notice = {'session_id': 'proc_1', 'started_at': 1.0, 'session_key': live.route,
                  'parent_session_id': entry.session_id, 'command': 'make test', 'output': 'ok'}
        assert await runner._inject_watch_notification(
            '[process exited]', dict(notice, type='completion', exit_code=0)) is True
        await async_wait_until(lambda: ran('process exited'), timeout=10, what='the completion')
        assert await runner._inject_watch_notification(
            '[delegation finished]', dict(notice, type='async_delegation', delegation_id='d1',
                                          results=[{'status': 'completed', 'task_index': 0}])) is True
        await async_wait_until(lambda: ran('delegation finished'), timeout=10, what='the delegation result')
    finally:
        runner.session_store.close_all_db_handles()
    # A notice's model text carries the producer's footer (``_mark_internal_notification``).
    assert {word: ran(word) for word in ('typed by a person', 'plugin injection', 'wake', 'process exited',
                                         'delegation finished')} == {
        'typed by a person': [(None, None)], 'plugin injection': [(None, None)], 'wake': [(None, None)],
        'process exited': [('process_notification', None)], 'delegation finished': [('delegation_result', None)]}


PROBE_PLUGIN = '''
import json
import threading

from hermes_cli import plugin_session

OUT = {out!r}
_LOCK = threading.Lock()


def register(ctx):
    def record(hook):
        def callback(session_id=None, **kw):
            line = {{"hook": hook, "session_id": session_id, "origin": plugin_session.turn_origin(),
                     "client": plugin_session.turn_client(), "ctx": ctx.turn_origin()}}
            with _LOCK, open(OUT, "a", encoding="utf-8") as out:
                out.write(json.dumps(line) + "\\n")
        return callback
    ctx.register_hook("post_llm_call", record("post_llm_call"))
    ctx.register_hook("on_session_end", record("on_session_end"))
'''


def test_a_typed_turn_from_a_connection_that_announced_nothing_names_no_origin(tmp_path, monkeypatch):
    """A real gateway process: two connections, one announced as Desktop and one that announced nothing,
    each submit a turn. Both read ``turn_origin() is None`` in ``post_llm_call`` and
    ``on_session_end``; the unannounced one reads ``turn_client() is None`` all the same."""
    from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket
    from tests.waiting import wait_until

    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    out = tmp_path / 'hooks.jsonl'
    plugin = home / 'plugins' / 'originprobe'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text(json.dumps({'name': 'originprobe', 'version': '1.0'}))
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
        'plugins': {'enabled': ['originprobe']},
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
    seen = {(line['hook'], by_session[line['session_id']], line['origin'], line['ctx'],
             (line['client'] or {}).get('name')) for line in map(json.loads, out.read_text().splitlines())}
    assert seen == {(hook, name, None, None, 'hermes-desktop' if name == 'announced' else None)
                    for hook in ('post_llm_call', 'on_session_end') for name in sessions}, seen


HOSTED_PLUGIN = '''
from hermes_cli import plugin_session


def register(ctx):
    ctx.register_hook("post_llm_call", lambda **kw: [plugin_session.turn_origin(), ctx.turn_origin()])
'''


@pytest.mark.platforms('any')  # the host is a child process
def test_a_hosted_plugin_reads_it_through_ctx_and_nothing_at_module_level(tmp_path, monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli.plugins import PluginManager
    _home_with_plugins(tmp_path, monkeypatch, {'hostedorigin': HOSTED_PLUGIN}, isolation='host')
    manager = PluginManager()
    manager.discover_and_load()

    def turn(origin):
        event = _named(MessageEvent(text='x', source=_source('local'), internal=True), origin)
        tokens = set_session_vars(platform='local', chat_id='s', session_key='s', session_id='agent-s', turn_event=event)
        try:
            return manager.invoke_hook('post_llm_call', session_id='agent-s')
        finally:
            clear_session_vars(tokens)
    try:
        assert manager._plugins['hostedorigin'].error is None
        assert contextvars.Context().run(turn, 'delegation_result') == [[None, 'delegation_result']]
        assert contextvars.Context().run(turn, None) == [[None, None]]
        assert contextvars.Context().run(manager.invoke_hook, 'post_llm_call') == [[None, None]]
    finally:
        manager.unload()
        manager._plugin_host().shutdown()


def test_a_delegated_child_reads_nothing_of_its_parent_s_turn():
    from gateway.platforms.event import MessageEvent
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli.plugin_session import turn_origin
    from tools.delegate_tool_child_run import _ChildRun
    seen = []

    class Child:
        session_id, model, provider = 'child', 'some/model', 'openrouter'

        def run_conversation(self, **kwargs):
            seen.append(turn_origin())
            return {'completed': True, 'final_response': 'ok'}

    event = _named(MessageEvent(text='[process done]', source=_source('local'), internal=True), 'process_notification')
    tokens = set_session_vars(platform='local', chat_id='s', session_key='s', session_id='agent-s', turn_event=event)
    try:
        seen.append(turn_origin())
        _ChildRun(Child(), SimpleNamespace(_current_task_id=None), 0, 'Probe', None, None).await_child()
    finally:
        clear_session_vars(tokens)
    assert seen == ['process_notification', None]


@pytest.mark.parametrize('fork', ['plugin_fork:wisdom', 'side_question', 'background_review'])
def test_a_detached_fork_reads_nothing_of_its_parent_s_turn(fork):
    """A detached fork runs its own prompt in a copy of the turn's context: neither ``turn_origin()``
    nor ``ctx.turn_origin()`` names the parent's producer there, and both read it again outside."""
    from agent.turn_facade import bind_detached_fork_origin
    from gateway.platforms.event import MessageEvent
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli.plugin_session import turn_origin
    from hermes_cli.plugins import PluginManifest
    from hermes_cli.plugins_context import PluginContext
    ctx = PluginContext(PluginManifest(name='wisdom'), SimpleNamespace())
    event = _named(MessageEvent(text='[process done]', source=_source('local'), internal=True), 'process_notification')
    tokens = set_session_vars(platform='local', chat_id='s', session_key='s', session_id='agent-s', turn_event=event)
    try:
        seen = [turn_origin(), ctx.turn_origin()]
        with bind_detached_fork_origin(fork):
            seen += [turn_origin(), ctx.turn_origin()]
            # The fork's own thread, as ``plugin_session_fork`` runs it, sees the copied context.
            copied = contextvars.copy_context()
        seen += copied.run(lambda: [turn_origin(), ctx.turn_origin()])
        seen += [turn_origin(), ctx.turn_origin()]
    finally:
        clear_session_vars(tokens)
    assert seen == ['process_notification'] * 2 + [None] * 4 + ['process_notification'] * 2


@pytest.mark.asyncio
async def test_a_pre_upgrade_local_watch_match_retry_is_acknowledged_and_runs_once(tmp_path, monkeypatch):
    """A local watch match an older build admitted (no origin named) and retried after the upgrade
    with the identical event is acknowledged against the committed row and runs once, as the
    process notification it is."""
    from gateway.config import GatewayConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import build_session_context
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionActor
    from gateway.session_local import create_local_session
    from hermes_cli.plugin_session import turn_origin

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    seen = []

    async def answer(event):
        entry = runner.session_store.lookup_by_session_key(authority.sessions[ref.session_id].route)
        tokens = runner._set_session_env(build_session_context(event.source, runner.config, entry, event))
        try:
            seen.append((event.text.split(']')[0] + ']', turn_origin()))
        finally:
            runner._clear_session_env(tokens)
        return 'ok'

    monkeypatch.setattr(runner, '_handle_message', answer)
    try:
        viewer = SessionActor('test-owner', 'default', frozenset(
            {'session:create', 'session:submit', 'session:read'}), 'viewer')
        ref = create_local_session(authority, viewer, dict(request_id='gui', source='gui', model='fixture', toolsets=[]))
        await authority.attach(viewer, ref)  # a viewer is attached: the notice runs as a turn
        live = authority.sessions[ref.session_id]
        entry = runner.session_store.lookup_by_session_key(live.route)
        evt = {'type': 'watch_match', 'event_id': 'w1', 'session_id': 'proc_1', 'started_at': 1.0, 'session_key': live.route,
               'parent_session_id': entry.session_id, 'command': 'make test', 'output': 'ok', 'exit_code': 0}
        with monkeypatch.context() as held:
            held.setattr(type(authority), '_schedule', lambda self, r: None)  # hold the drain for the retry
            with monkeypatch.context() as old_build:
                old_build.setattr(MessageEvent, 'set_turn_origin', lambda self, origin: None)
                assert await runner._inject_watch_notification('[watch matched]', dict(evt)) is True
            assert await runner._inject_watch_notification('[watch matched]', dict(evt)) is True
        rows = [r for r in authority.persistence.admissions.list(session_id=ref.session_id, pending_only=False)
                if 'local_automation_v1' in r.payload]
        assert len(rows) == 1 and 'turn_origin' not in rows[0].payload['local_automation_v1']
        authority._schedule(ref)
        await async_wait_until(lambda: seen, timeout=10, what='the watch match')
        # The drain has run the FIFO dry: a second execution of the retry would have happened by now.
        await async_wait_until(lambda: live.task is not None and live.task.done(), timeout=10, what='the drain')
        assert authority.persistence.admissions.list(session_id=ref.session_id, pending_only=True) == []
    finally:
        runner.session_store.close_all_db_handles()
    assert seen == [('[watch matched]', 'process_notification')]
