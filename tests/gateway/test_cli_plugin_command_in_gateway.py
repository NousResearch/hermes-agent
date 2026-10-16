"""A classic CLI plugin command runs in the gateway and its ``ctx.call_mcp`` reaches the gateway's MCP
registry (#134889 CLI-13, PLG-7).

A real daemon (``gateway.run``) loads a user plugin whose ``/probe-mcp-status`` calls ``probe-mcp``, a
fake server in the daemon's own MCP registry (tests/gateway/fixtures/plugin_mcp_daemon.py). The CLI's
dispatch reads the daemon's ``commands.catalog`` and sends ``slash.exec`` over its websocket, so the
command goes through ``gateway/session_commands.py`` -> ``_hm_run_plugin_handler``.
"""
import asyncio
from contextlib import asynccontextmanager, contextmanager
import json
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
import signal
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket
from tests.waiting import wait_until

_PLUGIN = '''import json
import os


def register(ctx):
    def status(arg):
        reply = ctx.call_mcp("probe-mcp", "echo", {"text": arg})
        return "PLUGIN_PID=%d MCP=%s" % (os.getpid(), json.dumps(reply, default=str))
    ctx.register_command("probe-mcp-status", status, description="Ask the fake MCP server")
    ctx.register_command("probe_under", lambda arg: "EXACT_UNDERSCORE", description="Exact name")
    ctx.register_command("probe-under", lambda arg: "HYPHEN_ALIAS", description="Telegram alias form")
    ctx.register_command("probe_only", lambda arg: "UNDERSCORE_ONLY", description="No hyphen twin")

    def boom(arg):
        raise RuntimeError("probe handler exploded")
    ctx.register_command("probe-boom", boom, description="Always fails")

    def where(arg):  # what the plugin docs tell a handler to read instead of os.getcwd()/os.environ
        from agent.runtime_cwd import resolve_agent_cwd
        from gateway.session_context import get_session_env
        return "WHERE=" + json.dumps({"cwd": str(resolve_agent_cwd()), "getcwd": os.getcwd(),
                                      "source": get_session_env("HERMES_SESSION_SOURCE"),
                                      "platform": get_session_env("HERMES_SESSION_PLATFORM"),
                                      "key": get_session_env("HERMES_SESSION_KEY")})
    ctx.register_command("probe-cwd", where, description="Where the handler runs")

    def sid(arg):  # the session a background job started from the handler records as its parent
        from gateway.session_context import get_session_env
        from tools.process_registry import process_registry
        started = json.loads(ctx.dispatch_tool("terminal", {"command": "true", "background": True}))
        job = process_registry.get(started.get("session_id") or "")
        return "SID=" + json.dumps({"id": get_session_env("HERMES_SESSION_ID"),
                                    "key": get_session_env("HERMES_SESSION_KEY"),
                                    "parent": getattr(job, "parent_session_id", None), "started": started})
    ctx.register_command("probe-sid", sid, description="The calling session")
    ctx.register_command("probe-args", lambda arg: "ARGS=" + json.dumps(arg), description="Echo the argument")
'''
# One command per gateway life: A serves /alpha, the restarted B only /beta.
_LIFE = '''import os


def register(ctx):
    ctx.register_command("NAME", lambda arg: "RAN_NAME PID=%d" % os.getpid(), description="NAME")
'''


@contextmanager
def _plugin_home(tmp_path, source):
    """``(root, home, env)``: a profile home with plugin ``probe-cmd`` and a loopback model."""
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    plugin = home / 'plugins' / 'probe-cmd'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text("name: probe-cmd\nversion: '1.0'\n")
    (plugin / '__init__.py').write_text(source)
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'plugin-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []},
        'plugins': {'enabled': ['probe-cmd'], 'entries': {'probe-cmd': {'mcp_allowlist': ['probe-mcp']}}},
    }))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1',
               PYTHONDONTWRITEBYTECODE='1')
    try:
        yield root, home, env
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)


def _no_local_plugins(monkeypatch):
    def no_local_plugin(*_a, **_k):
        pytest.fail('the CLI process loaded or ran plugin code')
    for name in ('get_plugin_command_handler', 'discover_plugins', 'get_plugin_commands'):
        monkeypatch.setattr(f'hermes_cli.plugins.{name}', no_local_plugin)


class _Wire:
    """The CLI connection's ``call`` over the daemon's real websocket, on one loop thread."""
    closed = final = False
    epoch = 1

    def __init__(self, home, desc):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.calls = []

        async def _open():
            return await websocket(home, desc)
        self.ws = self._run(_open())

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(60)

    def call(self, method, **params):
        from hermes_cli.gateway_client import GatewayRPCError
        self.calls.append(method)
        reply = self._run(rpc(self.ws, method, **params))
        if 'error' in reply:  # an owner refusal, as gateway_client raises it
            raise GatewayRPCError(str(reply['error'].get('message')), reply['error'].get('code'))
        return reply['result']

    def call_tagged(self, method, **params):
        return self.call(method, **params), self.epoch

    def call_until(self, method, _wake, **params):
        return self.call(method, **params)

    def close(self):
        self._run(self.ws.close())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


@pytest.mark.platforms("linux")
def test_cli_plugin_command_runs_in_the_gateway_and_reaches_its_mcp_registry(tmp_path, monkeypatch, capsys):
    import cli as cli_mod
    from cli import HermesCLI
    _no_local_plugins(monkeypatch)
    monkeypatch.setattr(cli_mod, '_ensure_skill_commands', dict)
    monkeypatch.setattr(cli_mod, 'get_skill_bundles', dict)
    work = tmp_path / 'work'
    work.mkdir()
    with _plugin_home(tmp_path, _PLUGIN) as (root, home, env):
        with daemon(root, home, env, barrier=True, fixture='plugin_mcp_daemon.py') as (proc, desc):
            wire = _Wire(home, desc)
            try:
                sid = wire.call('session.create', request_id='plugin-cli', source='cli', toolsets=[],
                                cwd=str(work))['session_id']
                monkeypatch.setattr('hermes_cli.gateway_side.connection_for', lambda _cli: wire)
                c = HermesCLI.__new__(HermesCLI)
                c.config, c.session_id = {}, sid
                capsys.readouterr()
                assert c._process_unregistered_slash('/probe-mcp-status Deep Check',
                                                     '/probe-mcp-status deep check') is True
                out = capsys.readouterr().out
                calls = list(wire.calls)
                typed_outs = {}
                for typed in ('/probe_under', '/probe_only', '/probe-boom', '/probe-cwd', '/probe-sid',
                              '/probe-args 2026–2027 — x'):
                    assert c._process_unregistered_slash(typed, typed) is True
                    typed_outs[typed] = capsys.readouterr().out
            finally:
                wire.close()
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
    assert calls[-2:] == ['commands.catalog', 'slash.exec'], calls
    assert 'EXACT_UNDERSCORE' in typed_outs['/probe_under'], typed_outs
    assert 'UNDERSCORE_ONLY' in typed_outs['/probe_only'], typed_outs
    # main's local path printed a failing handler's error; the gateway path must not swallow it
    assert 'Plugin command error: probe handler exploded' in typed_outs['/probe-boom'], typed_outs
    where = json.loads(typed_outs['/probe-cwd'].split('WHERE=', 1)[1].strip())
    # the session's folder and session vars are bound around the handler; the process cwd is the gateway's
    assert where['cwd'] == str(work) and where['getcwd'] != str(work), where
    assert where['platform'] == 'local' and where['key'], where
    # the calling session's identity, as a turn binds it: a background job records it as its parent
    ident = json.loads(typed_outs['/probe-sid'].split('SID=', 1)[1].strip())
    assert ident['id'] == sid and ident['key'] == where['key'], ident
    assert ident['parent'] == sid, ident
    # as typed: the messaging path's dash repair (MessageEvent.get_command_args) is not applied
    args = json.loads(typed_outs['/probe-args 2026–2027 — x'].split('ARGS=', 1)[1].strip())
    assert args == '2026–2027 — x', args
    assert proc.pid != os.getpid()
    assert f'PLUGIN_PID={proc.pid} ' in out, out  # the handler ran in the gateway process
    mcp = json.loads(json.loads(out.split('MCP=', 1)[1].strip())['result'])
    assert mcp == {'tool': 'echo', 'arguments': {'text': 'Deep Check'}, 'registry_pid': proc.pid}, out


def _dial_running(home):
    """The production dial pinned to an owner already running (``connect_gateway(endpoint)``), so a
    redial while none is up fails instead of starting one the test does not own."""
    from hermes_cli.gateway_client import GatewayClientError, connect_gateway
    from hermes_cli.gateway_runtime import discover_gateway_endpoint

    @asynccontextmanager
    async def dial():
        found = await asyncio.to_thread(discover_gateway_endpoint, home)
        if found.state != 'ready' or found.endpoint is None:
            raise GatewayClientError(f'no gateway is serving this home ({found.state})')
        async with connect_gateway(found.endpoint) as client:
            yield client
    return dial


@pytest.mark.platforms("linux")
def test_a_redial_to_a_restarted_gateway_and_slash_new_reread_the_plugin_commands(tmp_path, monkeypatch, capsys):
    """The classic CLI's own GatewayConnection and GatewaySessionStore across a real owner restart:
    gateway A serves /alpha, B (same home and socket) only /beta. A cached /alpha never runs: while no
    gateway answers it is a named refusal (no same-named skill stands in), on B it is no plugin
    command. /new through the gateway session store re-reads B's catalog for the new session."""
    import cli as cli_mod
    from cli import HermesCLI
    from hermes_cli.gateway_connection import GatewayConnection
    from hermes_cli.gateway_session_store import GatewaySessionStore
    _no_local_plugins(monkeypatch)
    skills, ran_skill = {'/alpha': {'name': 'alpha'}}, []
    monkeypatch.setattr(cli_mod, '_ensure_skill_commands', lambda: dict(skills))
    monkeypatch.setattr(cli_mod, 'get_skill_bundles', dict)
    monkeypatch.setattr(HermesCLI, '_run_skill_slash_command', lambda self, *a: ran_skill.append(a[0]))
    with _plugin_home(tmp_path, _LIFE.replace('NAME', 'alpha')) as (root, home, env):
        monkeypatch.setenv('HERMES_HOME', str(home))
        conn = GatewayConnection(connect=_dial_running(home))
        sent, real, real_until = [], conn.call_tagged, conn.call_until

        def spy(method, **params):
            sent.append((method, params.get('session_id')))
            return real(method, **params)

        def spy_until(method, wake, **params):  # slash.exec: no answer deadline
            sent.append((method, params.get('session_id')))
            return real_until(method, wake, **params)
        monkeypatch.setattr(conn, 'call_tagged', spy)
        monkeypatch.setattr(conn, 'call_until', spy_until)
        c = HermesCLI()
        c.agent, c._session_db, c._gateway_session_open = None, GatewaySessionStore(conn), False

        def typed(text):
            del sent[:]
            assert c._process_unregistered_slash(text, text) is True
            return capsys.readouterr().out, [m for m, _ in sent]
        try:
            with daemon(root, home, env, barrier=True, fixture='plugin_mcp_daemon.py') as (a, _desc):
                c._ensure_gateway_session()  # the CLI's startup session.create on A
                first = c.session_id
                capsys.readouterr()
                on_a = typed('/alpha')
                a.send_signal(signal.SIGINT)
                a.wait(timeout=20)
            wait_until(lambda: conn.closed, timeout=20, what='the CLI connection to see gateway A hang up')
            gone = typed('/alpha')
            (home / 'plugins' / 'probe-cmd' / '__init__.py').write_text(_LIFE.replace('NAME', 'beta'))
            skills.clear()
            with daemon(root, home, env, barrier=False) as (b, _desc):
                on_b = typed('/beta')
                epoch_b = conn.epoch
                stale = typed('/alpha')
                c.new_session(silent=True)
                capsys.readouterr()
                del sent[:]
                assert c._process_unregistered_slash('/beta', '/beta') is True
                after_new, new_calls = capsys.readouterr().out, list(sent)
                b.send_signal(signal.SIGINT)
                b.wait(timeout=20)
        finally:
            conn.close()
    assert on_a[1] == ['commands.catalog', 'slash.exec'] and f'RAN_alpha PID={a.pid}' in on_a[0], on_a
    # A hung up and nothing serves the home: the cached /alpha is not sent and the skill does not run
    assert 'slash.exec' not in gone[1] and "command list is unavailable" in gone[0], gone
    assert 'nothing was run' in gone[0] and ran_skill == [], (gone, ran_skill)
    assert epoch_b == 2 and on_b[1] == ['commands.catalog', 'slash.exec'], (epoch_b, on_b)
    assert f'RAN_beta PID={b.pid}' in on_b[0] and b.pid != a.pid, on_b
    # B's catalog has no /alpha: one refresh on the miss, never a slash.exec, no plugin output
    assert stale[1] == ['commands.catalog'] and 'RAN_alpha' not in stale[0], stale
    assert c.session_id != first and conn.epoch == epoch_b
    # /new reads the new session's own catalog (its skills come from that session's folder)
    assert new_calls == [('commands.catalog', c.session_id), ('slash.exec', c.session_id)], new_calls
    assert f'RAN_beta PID={b.pid}' in after_new, after_new


# /pidprobe stays; /pptwin shares its name with a skill. ``TWIN`` is dropped to stop registering it.
_RELOADED = '''import os


def register(ctx):
    ctx.register_command("pidprobe", lambda arg: "PIDPROBE PID=%d" % os.getpid(), description="Probe")
    TWIN
'''
_TWIN = 'ctx.register_command("pptwin", lambda arg: "PLUGIN_TWIN", description="Shadows a skill")'


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("unload", ["disable", "stop_registering"])
def test_a_command_a_plugin_reload_removed_is_dispatched_again(tmp_path, monkeypatch, capsys, unload):
    """The open CLI cached the session's catalog; then the running gateway reloads its plugins (what
    ``hermes plugins disable`` asks for) without the command, in the same session and gateway life.
    The cached name gets ``unsupported_command`` from ``slash.exec`` once: the CLI re-reads the catalog
    and dispatches the name again, so a same-named skill runs, and a name nothing else matches gets
    the "no longer has this plugin command" line, never the raw code, on that try or a later one."""
    import cli as cli_mod
    from cli import HermesCLI
    from gateway.control_socket import reload_gateway_plugins
    _no_local_plugins(monkeypatch)
    ran_skill = []
    monkeypatch.setattr(cli_mod, '_ensure_skill_commands', dict)  # the skill is the gateway's, below
    monkeypatch.setattr(cli_mod, 'get_skill_bundles', dict)
    monkeypatch.setattr(HermesCLI, '_run_skill_slash_command', lambda self, *a: ran_skill.append(a[0]))
    work = tmp_path / 'work'
    work.mkdir()
    with _plugin_home(tmp_path, _RELOADED.replace('TWIN', _TWIN)) as (root, home, env):
        (home / 'skills' / 'pptwin').mkdir(parents=True)
        (home / 'skills' / 'pptwin' / 'SKILL.md').write_text('---\nname: pptwin\ndescription: Twin skill\n---\nSay SKILL_TWIN.\n')
        with daemon(root, home, env, barrier=True, fixture='plugin_mcp_daemon.py') as (proc, desc):
            wire = _Wire(home, desc)
            try:
                sid = wire.call('session.create', request_id='plugin-reload', source='cli', toolsets=[],
                                cwd=str(work))['session_id']
                monkeypatch.setattr('hermes_cli.gateway_side.connection_for', lambda _cli: wire)
                c = HermesCLI.__new__(HermesCLI)
                c.config, c.session_id = {}, sid

                def typed(text):
                    del wire.calls[:]
                    assert c._process_unregistered_slash(text, text) is True
                    return capsys.readouterr().out, list(wire.calls)
                capsys.readouterr()
                before = typed('/pidprobe'), typed('/pptwin')
                if unload == 'disable':
                    config = json.loads((home / 'config.yaml').read_text())
                    config['plugins'].update(enabled=[], disabled=['probe-cmd'])
                    (home / 'config.yaml').write_text(json.dumps(config))
                else:
                    (home / 'plugins' / 'probe-cmd' / '__init__.py').write_text(_RELOADED.replace('TWIN', 'pass'))
                answer = reload_gateway_plugins(home)
                order = ('/pidprobe', '/pptwin', '/pidprobe') if unload == 'disable' else ('/pptwin', '/pidprobe', '/pptwin')
                after = [typed(text) for text in order]
            finally:
                wire.close()
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
    assert f'PIDPROBE PID={proc.pid}' in before[0][0] and 'PLUGIN_TWIN' in before[1][0], before
    assert (answer or {}).get('reloaded') is True, answer
    assert not any('unsupported_command' in out for out, _ in after), after
    if unload == 'disable':
        # the cached /pidprobe: refused once, one re-read, the existing line; nothing else ran
        assert after[0][1] == ['slash.exec', 'commands.catalog'], after[0]
        assert 'no longer has this plugin command' in after[0][0] and 'PIDPROBE' not in after[0][0], after[0]
        # the re-read is the session's catalog now: the skill runs without asking the gateway to exec
        assert 'slash.exec' not in after[1][1] and 'slash.exec' not in after[2][1], after
        assert ran_skill == ['/pptwin'] and 'PIDPROBE' not in after[2][0], (ran_skill, after)
    else:
        # the cached /pptwin: refused once, one re-read, then the same-named skill runs
        assert after[0][1] == ['slash.exec', 'commands.catalog'] and ran_skill == ['/pptwin', '/pptwin'], after
        assert f'PIDPROBE PID={proc.pid}' in after[1][0], after  # still the plugin's, in the gateway
        assert 'slash.exec' not in after[2][1] and 'PLUGIN_TWIN' not in after[2][0], after


_BROKEN = '''def register(ctx):
    ctx.register_command("pptwin", lambda arg: "PLUGIN_TWIN", description="Shadows a skill")
    raise RuntimeError("register exploded")
'''


@pytest.mark.platforms("linux")
def test_a_plugin_whose_register_fails_leaves_its_name_to_a_same_named_skill(tmp_path, monkeypatch, capsys):
    """Pins the review-bot question "failed plugins select other commands". The loader catches a failed
    ``register()``, drops what it registered and records ``loaded.error`` (plugins_loader); the
    plugin has no command, so the catalog lists none and reports no discovery error, and the typed
    name goes on to the same-named skill. Main's CLI did the same: its own load dropped the
    command, ``_get_plugin_cmd_handler_names()`` missed it, and the skill branch ran."""
    import cli as cli_mod
    from cli import HermesCLI
    _no_local_plugins(monkeypatch)
    ran_skill = []
    monkeypatch.setattr(cli_mod, '_ensure_skill_commands', lambda: {'/pptwin': {'name': 'pptwin'}})
    monkeypatch.setattr(cli_mod, 'get_skill_bundles', dict)
    monkeypatch.setattr(HermesCLI, '_run_skill_slash_command', lambda self, *a: ran_skill.append(a[0]))
    work = tmp_path / 'work'
    work.mkdir()
    with _plugin_home(tmp_path, _BROKEN) as (root, home, env):
        with daemon(root, home, env, barrier=True, fixture='plugin_mcp_daemon.py') as (proc, desc):
            wire = _Wire(home, desc)
            try:
                sid = wire.call('session.create', request_id='plugin-broken', source='cli', toolsets=[],
                                cwd=str(work))['session_id']
                catalog = wire.call('commands.catalog', session_id=sid)
                monkeypatch.setattr('hermes_cli.gateway_side.connection_for', lambda _cli: wire)
                c = HermesCLI.__new__(HermesCLI)
                c.config, c.session_id = {}, sid
                capsys.readouterr()
                del wire.calls[:]
                assert c._process_unregistered_slash('/pptwin', '/pptwin') is True
                out, calls = capsys.readouterr().out, list(wire.calls)
            finally:
                wire.close()
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
    plugin_rows = [row for cat in catalog['categories'] if cat['name'] == 'Plugin commands' for row in cat['pairs']]
    assert plugin_rows == [] and not catalog.get('plugin_discovery_error'), catalog
    assert calls == ['commands.catalog'] and ran_skill == ['/pptwin'], (calls, ran_skill)
    assert 'PLUGIN_TWIN' not in out, out


# /ppslow blocks on the FIFO GATE until the test writes a note; MARKS/started-<arg> and
# MARKS/done-<arg> show when the handler started and finished in the gateway.
_SLOW = '''import os


def register(ctx):
    def slow(arg):
        open(os.path.join(MARKS, "started-" + arg), "w").close()
        with open(GATE) as gate:
            note = gate.read()
        open(os.path.join(MARKS, "done-" + arg), "w").close()
        return "SLOW_DONE %s %s" % (arg, note)
    ctx.register_command("ppslow", slow, description="Waits for the test")
    ctx.register_command("ppquick", lambda arg: "QUICK_DONE", description="Answers at once")
'''


def _typed_in_thread(c, text):
    thread = threading.Thread(target=c._process_unregistered_slash, args=(text, text.lower()), daemon=True)
    thread.start()
    return thread


def _release(gate, note):
    with open(gate, 'w') as pipe:  # blocks until the handler opened its end
        pipe.write(note)


@pytest.mark.platforms("linux")
def test_a_plugin_command_outliving_the_rpc_limit_prints_its_reply_and_ctrl_c_stops_the_wait(
        tmp_path, monkeypatch, capsys):
    """Main's CLI ran the handler itself and waited for it. Through the gateway the CLI waits for the
    reply as long as the handler runs, not the 30 s RPC budget (injected here as 0, so a request on
    that budget gives up at once). Ctrl+C stops the wait with a line saying what happens next: the
    handler finishes in the gateway and its reply is dropped, while the next command already runs."""
    import cli as cli_mod
    from types import SimpleNamespace
    from prompt_toolkit.buffer import Buffer
    from cli import HermesCLI
    from hermes_cli import gateway_plugin_commands
    from hermes_cli.gateway_connection import GatewayConnection
    from hermes_cli.gateway_session_store import GatewaySessionStore
    _no_local_plugins(monkeypatch)
    monkeypatch.setattr(cli_mod, '_ensure_skill_commands', dict)
    monkeypatch.setattr(cli_mod, 'get_skill_bundles', dict)
    gate, marks = tmp_path / 'gate', tmp_path / 'marks'
    os.mkfifo(gate)
    marks.mkdir()
    source = _SLOW.replace('GATE', repr(str(gate))).replace('MARKS', repr(str(marks)))
    with _plugin_home(tmp_path, source) as (root, home, env):
        monkeypatch.setenv('HERMES_HOME', str(home))
        conn = GatewayConnection(connect=_dial_running(home))
        c = HermesCLI()
        c.agent, c._session_db, c._gateway_session_open = None, GatewaySessionStore(conn), False
        try:
            with daemon(root, home, env, barrier=True, fixture='plugin_mcp_daemon.py') as (proc, _desc):
                c._ensure_gateway_session()
                assert 'ppslow' in (gateway_plugin_commands.names(c) or {})  # cached: no catalog read below
                monkeypatch.setattr('hermes_cli.gateway_connection.DEFAULT_RPC_TIMEOUT', 0)
                capsys.readouterr()
                waiting = _typed_in_thread(c, '/ppslow long')
                wait_until(lambda: (marks / 'started-long').exists(), timeout=30, what='the handler to start')
                _release(gate, 'released')
                waiting.join(60)
                long_out = capsys.readouterr().out
                assert not waiting.is_alive()
                assert 'SLOW_DONE long released' in long_out, long_out
                assert 'did not answer' not in long_out, long_out

                stopped = _typed_in_thread(c, '/ppslow stop')
                wait_until(lambda: (marks / 'started-stop').exists(), timeout=30, what='the handler to start')
                app = SimpleNamespace(current_buffer=Buffer(), invalidate=lambda: None,
                                      exit=lambda: setattr(c, '_should_exit', True))
                c._tui_handle_ctrl_c(SimpleNamespace(app=app))  # the classic TUI's Ctrl+C binding
                stopped.join(60)
                stop_out = capsys.readouterr().out
                stop_alive, should_exit = stopped.is_alive(), c._should_exit
                # The stopped handler ran on a socket of its own: the next command does not wait for it.
                assert c._process_unregistered_slash('/ppquick', '/ppquick') is True
                after_out = capsys.readouterr().out
                _release(gate, 'late')
                wait_until(lambda: (marks / 'done-stop').exists(), timeout=30, what='the handler to finish')
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=20)
        finally:
            conn.close()
    assert not stop_alive and not should_exit, (stop_alive, should_exit)
    assert 'stopped waiting' in stop_out and 'keeps running in the gateway' in stop_out, stop_out
    assert 'SLOW_DONE' not in stop_out, stop_out
    assert 'QUICK_DONE' in after_out and 'SLOW_DONE' not in after_out, after_out  # the late reply is dropped
