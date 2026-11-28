"""S4: plugins read which client created the session (``session_client()``, prompt-section info), which
client submitted the running turn (``turn_client()``) and who is attached to its session
(``session_viewers()``)."""
import asyncio
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from tests.hermes_cli.plugin_host_support import _home_with_plugins

TERN = {'name': 'hermes-tern', 'version': '0.4.1'}

SECTION_PLUGIN = '''
def register(ctx):
    ctx.register_system_prompt_section(
        "probe.client", lambda info: "platform=" + info["platform"] + " client=" + info["client"]
        + " version=" + info["client_version"])
'''


@contextlib.contextmanager
def _turn_of(tmp_path, monkeypatch, **request):
    """The task-local state a local session's turn runs under (``policy_scope`` sets the same var)."""
    from gateway import session_policy
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    policy = session_policy.build_policy({'source': 'tui', 'cwd': str(tmp_path), **request}, {'model': {'default': 'm'}})
    token = session_policy._ACTIVE_POLICY.set(policy)
    try:
        yield policy
    finally:
        session_policy._ACTIVE_POLICY.reset(token)


def test_session_client_answers_for_the_running_session_only(tmp_path, monkeypatch):
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli.plugin_session import session_client
    assert session_client() is None  # outside any session
    with _turn_of(tmp_path, monkeypatch, client=dict(TERN)):
        first = session_client()
        assert first == TERN
        first['name'] = 'changed'  # a copy: the frozen request never changes
        assert session_client() == TERN
    with _turn_of(tmp_path, monkeypatch):
        assert session_client() is None  # Ink / Desktop / CLI / ACP send none
    tokens = set_session_vars(platform='telegram', chat_id='1', session_key='k')
    try:
        assert session_client() is None  # a messaging chat
    finally:
        clear_session_vars(tokens)


def test_a_stored_label_that_is_not_valid_reads_as_none():
    from gateway.session_client import policy_client
    for request in ({'client': {'name': 'Bad Name'}}, {'client': 'x'}, [], 'nope'):
        assert policy_client(SimpleNamespace(request_json=json.dumps(request))) is None
    assert policy_client(SimpleNamespace(request_json='{')) is None


def test_prompt_section_info_carries_the_client_as_strings(tmp_path, monkeypatch):
    from agent.system_prompt import _plugin_session_info
    agent = SimpleNamespace(session_id='s', model='m', provider='p', platform='tui')
    info = _plugin_session_info(agent)
    assert (info['client'], info['client_version']) == ('', '')
    with _turn_of(tmp_path, monkeypatch, client={'name': 'hermes-tern'}):
        info = _plugin_session_info(agent)
    assert (info['client'], info['client_version']) == ('hermes-tern', '')
    assert all(isinstance(value, str) for value in info.values())


@pytest.mark.platforms('any')  # the host is a child process
@pytest.mark.parametrize('isolation', ['host', 'in_process'])
def test_a_plugins_prompt_section_sees_the_client_in_either_isolation_mode(tmp_path, monkeypatch, isolation):
    """A hosted section's callable gets the session info as plain data (a read-only mapping crosses the
    host wire as a dict, not an opaque placeholder), so it reads ``client`` exactly as in-process."""
    from agent.system_prompt import _plugin_session_info
    from hermes_cli.plugins import PluginManager
    home = _home_with_plugins(tmp_path, monkeypatch, {'clientprobe': SECTION_PLUGIN}, isolation=isolation)
    manager = PluginManager()
    manager.discover_and_load()
    agent = SimpleNamespace(session_id='s', model='m', provider='p', platform='tui')
    try:
        assert manager._plugins['clientprobe'].error is None
        with _turn_of(home, monkeypatch, client=dict(TERN)):
            named = manager.render_system_prompt_sections(_plugin_session_info(agent))
        plain = manager.render_system_prompt_sections(_plugin_session_info(agent))
        assert [s.content for s in named] == ['platform=tui client=hermes-tern version=0.4.1']
        assert [s.content for s in plain] == ['platform=tui client= version=']
    finally:
        manager.unload()
        if isolation == 'host':
            manager._plugin_host().shutdown()


DESKTOP = {'name': 'hermes-desktop', 'version': '2.1.0', 'kind': 'desktop'}
WEBAPP = {'name': 'hermes-webapp', 'kind': 'webapp'}

TURN_PLUGIN = '''
import json
import threading

from hermes_cli import plugin_session

OUT = {out!r}
_LOCK = threading.Lock()


def register(ctx):
    def record(hook):
        def callback(session_id=None, **kw):
            line = {{"hook": hook, "session_id": session_id, "client": plugin_session.turn_client(),
                     "viewers": plugin_session.session_viewers(),
                     "ctx": [ctx.turn_client(), ctx.session_viewers()]}}
            with _LOCK, open(OUT, "a", encoding="utf-8") as out:
                out.write(json.dumps(line) + "\\n")
        return callback
    ctx.register_hook("post_llm_call", record("post_llm_call"))
    ctx.register_hook("on_session_end", record("on_session_end"))
'''


class _PairedModel(BaseHTTPRequestHandler):
    """Answers each turn only once BOTH turns are in flight and the test released them, so the two
    sessions' turns (and their end-of-turn hooks) run at the same time."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        text = next((m.get('content', '') for m in reversed(body.get('messages', [])) if m['role'] == 'user'), '')
        if isinstance(text, str) and text.startswith('TURN_'):
            self.server.pair.wait()
            self.server.release.wait(60)
        message = {'role': 'assistant', 'content': 'ACK'}
        if body.get('stream'):
            payload = ('data: ' + json.dumps({'id': 'x', 'choices': [{'index': 0, 'delta': message, 'finish_reason': 'stop'}]})
                       + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        else:
            payload = json.dumps({'id': 'x', 'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}],
                                  'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode()
            kind = 'application/json'
        try:
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _terminal(home, request_ids):
    with contextlib.closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
        statuses = dict(db.execute('SELECT request_id, status FROM session_admissions'))
    return all(statuses.get(rid) == 'terminal' for rid in request_ids)


def test_turn_hooks_read_their_own_turn_s_client_and_viewers_while_two_sessions_run(tmp_path, monkeypatch):
    """A real gateway process, two connections each running a turn at the same time on its own
    session: ``post_llm_call`` and ``on_session_end`` see their own turn's client and viewers
    (anything read from ``os.environ`` would show one session the other's)."""
    from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket
    from tests.waiting import wait_until

    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    out = tmp_path / 'hooks.jsonl'
    plugin = home / 'plugins' / 'turnprobe'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text(json.dumps({'name': 'turnprobe', 'version': '1.0'}))
    (plugin / '__init__.py').write_text(TURN_PLUGIN.format(out=str(out)))
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
        'plugins': {'enabled': ['turnprobe']},
    }))
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
        PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, PYTHONUNBUFFERED='1')
    reports = {'a': (DESKTOP, 'system', True), 'b': (WEBAPP, 'page', False)}

    async def drive(desc):
        async with websocket(home, desc) as ws_a, websocket(home, desc) as ws_b:
            sockets, sessions = {'a': ws_a, 'b': ws_b}, {}
            for name, ws in sockets.items():
                assert 'result' in await rpc(ws, 'client.capabilities', client=reports[name][0])
                created = await rpc(ws, 'session.create', request_id='s-' + name, source='gui', cwd=str(home), toolsets=[])
                sessions[name] = created['result']['session_id']
            for name, ws in sockets.items():
                reply = await rpc(ws, 'prompt.submit', session_id=sessions[name], submission_id='t-' + name,
                                  text='TURN_' + name.upper())
                assert reply.get('result', {}).get('status') == 'queued', reply
            # Both turns are waiting on the model now: the reports are the newest evidence.
            for name, ws in sockets.items():
                _label, basis, can_alert = reports[name]
                assert (await rpc(ws, 'client.activity', sessions=[sessions[name]], idle_seconds=0, basis=basis,
                                  can_alert=can_alert))['result'] == {}
            model.release.set()
            await asyncio.to_thread(wait_until, lambda: _terminal(home, ('t-a', 't-b')), timeout=60)
            return sessions

    try:
        with daemon(root, home, env, barrier=False) as (_proc, desc):
            sessions = asyncio.run(drive(desc))
    finally:
        model.release.set()
        model.shutdown()
        model.server_close()
        thread.join(timeout=5)

    lines = [json.loads(line) for line in out.read_text().splitlines()]
    by_session = {sid: name for name, sid in sessions.items()}
    assert {(line['hook'], by_session[line['session_id']]) for line in lines} == {
        (hook, name) for hook in ('post_llm_call', 'on_session_end') for name in ('a', 'b')}, lines
    for line in lines:
        label, basis, can_alert = reports[by_session[line['session_id']]]
        (viewer,) = line['viewers']
        assert line['client'] == label, line
        assert {key: viewer[key] for key in ('name', 'kind', 'basis', 'can_alert')} == {
            'name': label['name'], 'kind': label['kind'], 'basis': basis, 'can_alert': can_alert}, line
        assert 0 <= viewer['idle_seconds'] < 60, line
        ctx_client, ctx_viewers = line['ctx']  # read a moment later: only the idle time moved
        assert ctx_client == label and [v['name'] for v in ctx_viewers] == [label['name']], line


def test_outside_a_gateway_turn_and_in_a_messaging_turn_there_is_no_client_or_viewer():
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli.plugin_session import session_viewers, turn_client
    assert (turn_client(), session_viewers()) == (None, [])
    tokens = set_session_vars(platform='telegram', chat_id='1', session_key='agent:main:telegram:dm:1', session_id='s')
    try:
        assert (turn_client(), session_viewers()) == (None, [])  # no stamp; no gateway in this process
    finally:
        clear_session_vars(tokens)


HOSTED_TURN_PLUGIN = '''
from hermes_cli import plugin_session


def register(ctx):
    ctx.register_hook("post_llm_call", lambda **kw: {
        "module": [plugin_session.turn_client(), plugin_session.session_viewers()],
        "ctx": [ctx.turn_client(), ctx.session_viewers()]})
'''


@pytest.mark.asyncio
@pytest.mark.platforms('any')  # the host is a child process
async def test_a_hosted_plugin_reads_the_turn_through_ctx_and_nothing_at_module_level(tmp_path, monkeypatch):
    """``plugins.isolation: host``: the plugin's own process has no turn context and no gateway, so
    the module-level accessors answer ``None`` / ``[]`` there and do not fail. ``ctx.turn_client()``
    and ``ctx.session_viewers()`` call back into the gateway in the calling hook's context: they answer
    for that turn, and for no session when the caller has none bound."""
    import contextvars

    from gateway.session_context import clear_session_vars
    from gateway.session_surface import surface_turn_scope
    from hermes_cli.plugins import PluginManager
    _authority, client = await _served_session(tmp_path, monkeypatch)
    _home_with_plugins(tmp_path / 'plugins', monkeypatch, {'hostedturn': HOSTED_TURN_PLUGIN}, isolation='host')
    manager = PluginManager()
    manager.discover_and_load()

    def turn():
        tokens = _local_turn_vars()
        try:
            with surface_turn_scope({'client': DESKTOP}):
                return manager.invoke_hook('post_llm_call', session_id='agent-s')
        finally:
            clear_session_vars(tokens)
    try:
        assert manager._plugins['hostedturn'].error is None
        assert await asyncio.to_thread(turn) == [{'module': [None, []], 'ctx': [DESKTOP, [_DESKTOP_VIEWER]]}]
        unbound = await asyncio.to_thread(contextvars.Context().run, manager.invoke_hook, 'post_llm_call')
        assert unbound == [{'module': [None, []], 'ctx': [None, []]}]
    finally:
        manager.unload()
        manager._plugin_host().shutdown()
        await client.connection.close()


async def _served_session(tmp_path, monkeypatch):
    """A gateway authority this process serves (the turn runner's lookup) with its live session ``s``
    and one Desktop connection attached to it."""
    from tests.gateway.test_turn_client_stamp import _authority, _Client
    authority = await _authority(tmp_path / 'home', monkeypatch, None)
    monkeypatch.setattr('gateway.run._gateway_runner_ref', lambda: authority.runner)
    client = _Client(authority)
    assert 'result' in await client.call('client.capabilities', client=DESKTOP)
    assert 'result' in await client.call('session.resume', session_id='s')
    return authority, client


def _local_turn_vars():
    """The session vars a local session's turn binds (route and chat id are its canonical id)."""
    from gateway.session_context import set_session_vars
    return set_session_vars(platform='local', chat_id='s', session_key=_ROUTE, session_id='agent-s')


_ROUTE = 's'  # ``LiveSession(source, 's')``: the live session's route
_DESKTOP_VIEWER = {'name': 'hermes-desktop', 'kind': 'desktop', 'idle_seconds': None, 'basis': None, 'can_alert': None}


@pytest.mark.asyncio
async def test_a_delegated_child_reads_no_client_or_viewers_of_its_parent_s_turn(tmp_path, monkeypatch):
    """A child runs in a copy of its parent turn's context: that turn's admission is not the child's."""
    from gateway.session_context import clear_session_vars
    from gateway.session_surface import surface_turn_scope
    from hermes_cli.plugin_session import session_viewers, turn_client
    from tools.delegate_tool_child_run import _ChildRun
    _authority, client = await _served_session(tmp_path, monkeypatch)
    seen = []

    class Child:
        session_id, model, provider = 'child', 'some/model', 'openrouter'

        def run_conversation(self, **kwargs):
            seen.append((turn_client(), session_viewers()))
            return {'completed': True, 'final_response': 'ok'}

    def parent_turn():
        tokens = _local_turn_vars()
        try:
            with surface_turn_scope({'client': DESKTOP}):
                seen.append((turn_client(), session_viewers()))
                _ChildRun(Child(), SimpleNamespace(_current_task_id=None), 0, 'Probe', None, None).await_child()
        finally:
            clear_session_vars(tokens)
    try:
        await asyncio.to_thread(parent_turn)
        assert seen == [(DESKTOP, [_DESKTOP_VIEWER]), (None, [])], seen
    finally:
        await client.connection.close()


@pytest.mark.asyncio
async def test_outside_a_turn_nothing_is_read_from_the_process_environment(tmp_path, monkeypatch):
    """Only a route bound (a session open, a built-in slash command) is no session, even with the
    process's own session vars naming the session; a real turn that announced nothing still sees its
    viewers."""
    import contextvars

    from gateway.session_context import clear_session_vars, scoped_session_key
    from hermes_cli.plugin_session import session_viewers, turn_client
    _authority, client = await _served_session(tmp_path, monkeypatch)
    monkeypatch.setenv('HERMES_SESSION_PLATFORM', 'local')
    monkeypatch.setenv('HERMES_SESSION_CHAT_ID', 's')

    def route_only():
        with scoped_session_key(_ROUTE):
            return turn_client(), session_viewers()

    def unannounced_turn():
        tokens = _local_turn_vars()
        try:
            return turn_client(), session_viewers()
        finally:
            clear_session_vars(tokens)
    try:
        assert await asyncio.to_thread(contextvars.Context().run, route_only) == (None, [])
        assert await asyncio.to_thread(contextvars.Context().run, unannounced_turn) == (None, [_DESKTOP_VIEWER])
    finally:
        await client.connection.close()


@pytest.mark.asyncio
async def test_a_plugin_s_slash_command_reads_its_session_s_viewers_and_no_turn_client(tmp_path, monkeypatch):
    """The gateway runs a plugin's command handler with its session bound and under no admission
    (``_hm_run_plugin_handler``): the viewers are those of the session the command was typed in, and
    there is no turn client. A command typed in another session sees nothing of this one."""
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from hermes_cli.plugin_session import session_viewers, turn_client
    _authority, client = await _served_session(tmp_path, monkeypatch)
    runner = object.__new__(GatewayRunner)
    runner.config, runner._draining = GatewayConfig(), False
    seen = []

    def handler(raw_args):
        seen.append((turn_client(), session_viewers()))
        return 'ok'

    async def typed_in(chat_id, route):
        source = SessionSource(platform=Platform.LOCAL, chat_id=chat_id, chat_type='dm', user_id='owner')
        entry = SimpleNamespace(session_key=route, session_id='agent-' + chat_id, created_at=None, updated_at=None)
        return await runner._hm_run_plugin_handler(handler, MessageEvent(text='/who', source=source), source, entry, args='')
    try:
        assert await typed_in('s', _ROUTE) == 'ok'
        assert await typed_in('other', 'other') == 'ok'
        assert seen == [(None, [_DESKTOP_VIEWER]), (None, [])], seen
    finally:
        await client.connection.close()


def test_session_source_is_the_launch_policy_s_label_inside_a_local_session_s_turn(tmp_path, monkeypatch):
    from hermes_cli.plugin_session import session_source
    assert session_source() == ''  # no policy bound
    with _turn_of(tmp_path, monkeypatch, source='gui'):
        assert session_source() == 'gui'
    assert session_source() == ''


def test_session_source_returns_whatever_label_the_bound_policy_holds(tmp_path, monkeypatch):
    from dataclasses import replace
    from gateway import session_policy
    from hermes_cli.plugin_session import session_source
    with _turn_of(tmp_path, monkeypatch) as policy:
        # A label outside the six chat surfaces, as a board worker's session is created with.
        token = session_policy._ACTIVE_POLICY.set(replace(policy, source='kanban'))
        try:
            assert session_source() == 'kanban'
        finally:
            session_policy._ACTIVE_POLICY.reset(token)


def test_session_source_never_reads_the_process_environment(monkeypatch):
    from hermes_cli.plugin_session import session_source
    monkeypatch.setenv('HERMES_SESSION_SOURCE', 'gui')
    monkeypatch.setenv('HERMES_SESSION_PLATFORM', 'local')
    assert session_source() == ''


def test_in_delegated_child_is_true_only_inside_a_delegated_child_context():
    from agent.delegation_context import delegated_child_context
    from hermes_cli.plugin_session import in_delegated_child
    assert in_delegated_child() is False
    with delegated_child_context():
        assert in_delegated_child() is True
    assert in_delegated_child() is False


def test_a_child_s_subprocess_marker_counts_only_with_include_subprocesses(monkeypatch):
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER
    from hermes_cli.plugin_session import in_delegated_child
    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, '1')
    assert in_delegated_child() is False
    assert in_delegated_child(include_subprocesses=True) is True


def test_a_plugin_importing_the_session_accessors_gets_no_private_import_warning(tmp_path):
    from hermes_cli.plugin_validate import validate_plugin_dir
    from tests.hermes_cli.test_plugin_validate_private_imports import _plugin, _private_warnings
    plugin = _plugin(tmp_path / 'notify', {'gate.py': (
        'from hermes_cli.plugin_session import fork_origin, in_delegated_child, session_source\n')})
    assert _private_warnings(validate_plugin_dir(plugin)) == []


SESSION_FACTS_PLUGIN = '''
def register(ctx):
    ctx.register_hook("post_llm_call", lambda **kw: [
        ctx.session_source(), ctx.in_delegated_child(), ctx.in_delegated_child(include_subprocesses=True),
        ctx.fork_origin()])
'''


@pytest.mark.platforms('any')  # the host is a child process
@pytest.mark.parametrize('isolation', ['host', 'in_process'])
@pytest.mark.parametrize('origin', ['', 'background_review', 'side_question', 'plugin_fork:facts'],
                         ids=['curator', 'review', 'btw', 'plugin-fork'])
def test_the_ctx_session_accessors_answer_what_the_functions_answer(tmp_path, monkeypatch, isolation, origin):
    """Observe hook context and real tool-dispatch kwargs without calling an LLM."""
    from agent import background_review, curator
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER, delegated_child_context
    from agent.side_question import answer_side_question
    from hermes_cli import plugin_session
    from hermes_cli.plugins import PluginManager
    from model_tools import handle_function_call
    from run_agent import AIAgent
    from tools.registry import registry
    from tools.skill_provenance import reset_current_write_origin, set_current_write_origin

    dispatched = []
    monkeypatch.setattr(registry, 'dispatch', lambda *a, **kw: dispatched.append(kw) or '{}')
    home = _home_with_plugins(tmp_path, monkeypatch, {'facts': SESSION_FACTS_PLUGIN}, isolation=isolation)
    manager = PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr('agent.model_metadata.fetch_model_metadata', lambda *a, **kw: {})

    def facts():
        return [plugin_session.session_source(), plugin_session.in_delegated_child(),
                plugin_session.in_delegated_child(include_subprocesses=True), plugin_session.fork_origin()]

    def loop(agent, message, *args, **kwargs):
        # Vary write provenance independently: it must never control the fork label.
        token = set_current_write_origin('assistant_tool' if origin else 'background_review')
        try:
            assert facts() == ['gui', True, True, origin]
            assert manager.invoke_hook('post_llm_call') == [['gui', True, True, origin]]
            # The real /btw entry point denies dispatch even though the fork carries a label.
            if message.endswith('Side question: probe?'):
                before = len(dispatched)
                assert 'Side question (/btw) denied' in handle_function_call('skills_list', {})
                assert len(dispatched) == before
                return {'final_response': 'ok'}
            # Also check kwargs when dispatch is reachable, independent of /btw's whitelist.
            assert handle_function_call('skills_list', {}) == '{}'
            if origin.startswith('plugin_fork:'):
                assert dispatched[-1]['turn_origin'] == origin
            else:
                assert 'turn_origin' not in dispatched[-1]
            if message == 'raise':
                raise RuntimeError('loop error')
            return {'final_response': 'ok'}
        finally:
            reset_current_write_origin(token)

    monkeypatch.setattr('agent.conversation_loop.run_conversation', loop)
    with contextlib.ExitStack() as stack:
        if isolation == 'host':
            stack.callback(manager._plugin_host().shutdown)
        stack.callback(manager.unload)
        assert manager._plugins['facts'].error is None
        parent = AIAgent(api_key='test-key', base_url='https://openrouter.ai/api/v1',
                         model='test/model', quiet_mode=True, skip_context_files=True, skip_memory=True)
        stack.callback(parent.close)
        agent = parent
        if origin:
            agent, _, _ = background_review.build_cache_parity_fork(parent, {}, max_iterations=1, write_origin=origin)
            stack.callback(agent.close)
        assert manager.invoke_hook('post_llm_call') == [facts()] == [['', False, False, '']]
        with _turn_of(home, monkeypatch, source='gui'), delegated_child_context():
            assert agent.run_conversation('ok')['final_response'] == 'ok'
            assert manager.invoke_hook('post_llm_call') == [facts()] == [['gui', True, True, '']]
            with pytest.raises(RuntimeError, match='loop error'):
                agent.run_conversation('raise')
            if origin == 'side_question':
                assert answer_side_question('probe?', [], parent_agent=parent) == 'ok'
            elif origin == 'background_review':
                before = len(dispatched)
                background_review._run_review_in_thread(parent, [], 'review')
                assert len(dispatched) == before + 1
            elif not origin:
                runtime = {'api_key': 'test-key', 'base_url': 'https://openrouter.ai/api/v1'}
                monkeypatch.setattr(curator, '_resolve_review_provider',
                                    lambda: (runtime, 'test/model', 'openrouter', {}))
                assert curator._run_llm_review('curate')['final'] == 'ok'
            assert manager.invoke_hook('post_llm_call') == [facts()] == [['gui', True, True, '']]
        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, '1')  # a child's subprocess, no child context
        assert manager.invoke_hook('post_llm_call') == [facts()] == [['', False, True, '']]
