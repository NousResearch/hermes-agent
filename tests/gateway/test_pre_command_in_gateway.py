"""``pre_command`` fires once, in the gateway, for every local surface, and the classic CLI's ``/plugins``
reads the gateway's loaded plugins (#134889 PLG-7).

A real daemon (``gateway.run``) loads a user plugin whose ``pre_command`` hook appends one JSON line per
call to a log. Commands reach the gateway the way each client sends them: ``slash.exec`` (classic CLI,
Ink TUI, Desktop) and ``command.dispatch`` (Ink TUI). The classic CLI runs in this test process, where
any plugin-manager read fails the test.
"""
from contextlib import contextmanager
import json
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
import signal
import subprocess
import sys
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon
from tests.gateway.test_cli_plugin_command_in_gateway import _Wire
from tests.waiting import wait_until

_PLUGIN = '''import json
import os
from pathlib import Path


def register(ctx):
    def seen(**kw):
        from hermes_constants import get_hermes_home
        with open(%r, "a") as log:
            log.write(json.dumps(dict(kw, pid=os.getpid(), home=str(get_hermes_home()))) + "\\n")
    ctx.register_hook("pre_command", seen)
    ctx.register_command("probe-hook-cmd", lambda arg: "ok", description="Probe command")
    ctx.register_skill("probe", Path(__file__).with_name("SKILL.md"), "Probe plugin skill")
    ctx.register_tool(name="probe_ping", toolset="probe_tools", schema={"name": "probe_ping",
        "description": "d", "parameters": {"type": "object", "properties": {}}}, handler=lambda args, **kw: "pong")
'''
_SKILL = "---\nname: probe\ndescription: Probe plugin skill\n---\nSay PROBE_SKILL_BODY.\n"


def _hook_lines(log):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def _wait_hooks(log, count):
    """The hook log once it holds ``count`` lines (a notice fires after its reply)."""
    def landed():
        lines = _hook_lines(log)
        return lines if len(lines) >= count else []
    return wait_until(landed, timeout=20, what=f'{count} pre_command lines in {log.name}')


def _hooks_through(log, command):
    """The hook log once ``command``'s line landed. One session's deliveries fire in arrival order, so
    every delivery queued before it has landed too: an observed command sent last is the barrier."""
    def landed():
        lines = _hook_lines(log)
        return lines if any(r['command'] == command for r in lines) else []
    return wait_until(landed, timeout=20, what=f'the pre_command line for /{command} in {log.name}')


def _probe_plugin(home, log):
    plugin = home / 'plugins' / 'probe-hook'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text("name: probe-hook\nversion: '1.0'\n")
    (plugin / '__init__.py').write_text(_PLUGIN % str(log))
    (plugin / 'SKILL.md').write_text(_SKILL)


@contextmanager
def _gateway(tmp_path, *, isolation=None, beta=False, project=False):
    """A real daemon whose profile loads the probe plugin: ``(proc, desc, home, work, env, log)``.
    ``isolation`` sets ``plugins.isolation``; ``beta`` also serves profile ``beta`` (multiplexed), whose
    own copy of the plugin logs to ``beta.jsonl``; ``project`` makes ``work`` a trusted repo with its own
    ``repo-skill`` (the daemon runs from the checkout, not from ``work``) and auto-loads the plugin skill."""
    root = Path(__file__).resolve().parents[2]
    home, user, work = tmp_path / 'state', tmp_path / 'user', tmp_path / 'work'
    home.mkdir(mode=0o700)
    user.mkdir()
    work.mkdir()
    log = tmp_path / 'pre_command.jsonl'
    _probe_plugin(home, log)
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    profile = {'model': {'provider': 'custom', 'default': 'plugin-model', 'base_url': url},
               'auxiliary': {'title_generation': {'enabled': False}},
               'plugins': {'enabled': ['probe-hook'], **({'isolation': isolation} if isolation else {})}}
    if project:
        (work / '.git').mkdir()
        (work / '.hermes' / 'skills' / 'repo-skill').mkdir(parents=True)
        (work / '.hermes' / 'skills' / 'repo-skill' / 'SKILL.md').write_text(
            '---\nname: repo-skill\ndescription: Repo-only skill\n---\nSay REPO_SKILL_BODY.\n')
        profile['skills'] = {'trusted_project_dirs': [str(work)], 'auto_load': ['probe-hook:probe']}
    (home / 'config.yaml').write_text(json.dumps({'gateway': {'multiplex_profiles': beta}, **profile}))
    if beta:
        (home / 'profiles' / 'beta').mkdir(parents=True)
        (home / 'profiles' / 'beta' / 'config.yaml').write_text(json.dumps(profile))
        _probe_plugin(home / 'profiles' / 'beta', tmp_path / 'beta.jsonl')
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            yield proc, desc, home, work, env, log, peer
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)


@pytest.mark.platforms("linux")
@pytest.mark.parametrize('mode, flags', [('oneshot', ['-t', 'terminal', '-s', 'probe-hook:probe']), ('interactive', [])])
def test_real_cli_startup_and_commands_run_no_plugin_code_in_the_cli(tmp_path, mode, flags):
    """Real startup (launch discovery, toolset validation, deferred startup, agent init) and real
    ``process_command`` dispatch in a fresh CLI process: no plugin entry point is reached there, not even
    for ``-s``/``skills.auto_load`` naming a plugin skill, ``/help``, a plugin skill, ``/reload-skills``,
    ``/tools`` (and its toolset validation) or completion, which read the gateway's lists, refreshed
    after a rescan here or by another client; each recognized command fires ``pre_command`` once, in the gateway, in typed order,
    with main's values. The CLI starts in a trusted repo the gateway does not run from: its project skill
    is listed and loaded from the session's folder."""
    from agent.i18n import t
    root = Path(__file__).resolve().parents[2]
    with _gateway(tmp_path, project=True) as (proc, _desc, home, work, env, log, _peer):
        probe = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/cli_plugin_free_startup.py'),
                                mode, *flags], cwd=work, env=env, capture_output=True, text=True, timeout=180, check=False)
        assert probe.returncode == 0, probe.stdout + probe.stderr
        result = json.loads(probe.stdout.strip().splitlines()[-1])
        fired = _wait_hooks(log, 9)
        config = (home / 'config.yaml').read_text()
    assert result['calls'] == [], result['calls']
    assert result['plugin_modules'] == [] and result['model_tools'] is False
    assert result['errors'] == [], result['errors']
    assert result['agent'] is True, result['output']
    first = fired[0]['session_key']
    assert [(r['pid'], r['surface'], r['platform'], r['command'], r['args_raw']) for r in fired] == [
        (proc.pid, 'cli', 'cli', 'plugins', ''), (proc.pid, 'cli', 'cli', 'history', ''),
        (proc.pid, 'cli', 'cli', 'busy', 'queue'), (proc.pid, 'cli', 'cli', 'help', ''),
        (proc.pid, 'cli', 'cli', 'help', 'skills'), (proc.pid, 'cli', 'cli', 'reload-skills', ''),
        (proc.pid, 'cli', 'cli', 'help', 'skills'), (proc.pid, 'cli', 'cli', 'tools', ''),
        (proc.pid, 'cli', 'cli', 'tools', 'disable probe_tools'),
    ], fired
    assert {r['session_key'] for r in fired} == {first}  # (``/tools disable`` then starts a new session)
    outputs = result['outputs']
    detail = (f"(1 {t('cli.plugins.count_noun_tools')}, 1 {t('cli.plugins.count_noun_hooks')}, "
              f"1 {t('cli.plugins.count_noun_commands')})")
    assert f'probe-hook v1.0 {detail}' in outputs['/plugins'], outputs['/plugins']
    assert '/probe-hook:probe' in outputs['/help skills'] and 'Probe plugin skill' in outputs['/help skills']
    assert '/repo-skill' in outputs['/help skills'], outputs['/help skills']
    assert t('cli.skills.loading_skill', name='probe-hook:probe') in outputs['/probe-hook:probe check it']
    plugin_skill, repo_skill, *later = result['queued']
    assert 'PROBE_SKILL_BODY' in plugin_skill and 'check it' in plugin_skill
    assert 'REPO_SKILL_BODY' in repo_skill and 'from the repo' in repo_skill, outputs['/repo-skill from the repo']
    assert 'fresh-skill' in outputs['/reload-skills'], outputs['/reload-skills']  # the gateway's rescan
    assert '/fresh-skill' in outputs['/help skills #2'], outputs['/help skills #2']  # not the list /help cached
    # Rescanned by another client: the first invocation already resolves it
    assert len(later) == 1 and 'OTHER-SKILL_BODY' in later[0] and 'go now' in later[0], outputs['/other-skill go now']
    disabled = outputs['/tools disable probe_tools']  # the gateway's plugin toolset is recognised and saved
    assert 'Disabled: probe_tools' in disabled and 'Unknown toolset' not in disabled, disabled
    assert 'probe_tools' in config
    assert result['completions']['/probe'] == ['probe-hook-cmd', 'probe-hook:probe']  # the gateway's command and skill
    assert 'probe_tools' in result['completions']['/tools enable ']
    assert 'probe_tools' not in result['completions']['/tools disable ']


@pytest.mark.platforms("linux")
def test_pre_command_fires_once_in_the_gateway_and_cli_plugins_reads_its_list(tmp_path, monkeypatch, capsys):
    import hermes_cli.plugins as plugins_mod
    from agent.i18n import t
    from cli import HermesCLI
    root = Path(__file__).resolve().parents[2]
    home, user, work = tmp_path / 'state', tmp_path / 'user', tmp_path / 'work'
    home.mkdir(mode=0o700)
    user.mkdir()
    work.mkdir()
    log = tmp_path / 'pre_command.jsonl'
    _probe_plugin(home, log)
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'plugin-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}},
        'plugins': {'enabled': ['probe-hook']},
    }))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')

    def no_local_plugin(*_a, **_k):
        pytest.fail('the CLI process loaded, read or fired plugin code')
    for name in ('get_plugin_manager', 'discover_plugins', 'fire_pre_command_hook', 'get_plugin_commands'):
        monkeypatch.setattr(plugins_mod, name, no_local_plugin)
    sent = {}
    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            wire = _Wire(home, desc)
            try:
                for source in ('cli', 'tui', 'gui'):
                    sent[source] = wire.call('session.create', request_id=f'pre-{source}', source=source,
                                             toolsets=[], cwd=str(work))['session_id']
                for source, sid in sent.items():
                    wire.call('slash.exec', session_id=sid, command='/help')
                wire.call('command.dispatch', session_id=sent['tui'], name='status', arg='')  # Ink's own read
                fired = _hook_lines(log)
                assert [(r['pid'], r['surface'], r['platform'], r['command'], r['alias_used'], r['session_key'])
                        for r in fired] == [
                    # Main's values on every local surface (Ink and Desktop fired it through the CLI
                    # slash worker): surface and platform "cli"; session_key tells the sessions apart.
                    (proc.pid, 'cli', 'cli', 'help', 'help', sent['cli']),
                    (proc.pid, 'cli', 'cli', 'help', 'help', sent['tui']),
                    (proc.pid, 'cli', 'cli', 'help', 'help', sent['gui']),
                    (proc.pid, 'cli', 'cli', 'status', 'status', sent['tui']),
                ], fired

                # The classic CLI: a command it forwards fires once, in the gateway; /plugins lists
                # the gateway's loaded plugins. Both read this profile's home as the CLI process does.
                monkeypatch.setenv('HERMES_HOME', str(home))
                monkeypatch.setattr('hermes_cli.gateway_side.connection_for', lambda _cli: wire)
                c = HermesCLI.__new__(HermesCLI)
                c.config, c.session_id = {}, sent['cli']
                assert c.process_command('/busy queue') is True
                assert [(r['pid'], r['command'], r['args_raw'], r['session_key'])
                        for r in _hook_lines(log)[len(fired):]] == [(proc.pid, 'busy', 'queue', sent['cli'])]
                capsys.readouterr()
                c._cmd_plugins('/plugins')
                listing = capsys.readouterr().out
            finally:
                wire.close()
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)

    assert proc.pid != os.getpid()
    assert not [m for m in sys.modules if m.startswith('hermes_plugins.')]
    assert 'plugins.list' in wire.calls
    detail = (f"(1 {t('cli.plugins.count_noun_tools')}, 1 {t('cli.plugins.count_noun_hooks')}, "
              f"1 {t('cli.plugins.count_noun_commands')})")
    assert f'probe-hook v1.0 {detail}' in listing, listing


@pytest.mark.platforms("linux")
def test_accepted_tui_cron_reads_fire_once_and_refused_writes_fire_nothing(tmp_path):
    """A TUI session's ``/cron`` reads (main's sidecar text) fire once through both RPCs; a refused
    ``/cron`` write, and ``/cron`` on a classic CLI session (refused there), fire nothing."""
    from hermes_cli.gateway_client import GatewayClientError
    with _gateway(tmp_path) as (_proc, desc, home, work, _env, log, _peer):
        wire = _Wire(home, desc)
        try:
            tui = wire.call('session.create', request_id='cron-tui', source='tui', toolsets=[], cwd=str(work))['session_id']
            cli = wire.call('session.create', request_id='cron-cli', source='cli', toolsets=[], cwd=str(work))['session_id']
            wire.call('slash.exec', session_id=tui, command='/cron')
            wire.call('command.dispatch', session_id=tui, name='cron', arg='list')
            wire.call('slash.exec', session_id=tui, command='/cron list --all')
            wire.call('slash.exec', session_id=tui, command='/status')
            for sid, command in ((tui, '/cron add 1h ping'), (cli, '/cron'), (cli, '/cron list')):
                with pytest.raises(GatewayClientError):
                    wire.call('slash.exec', session_id=sid, command=command)
            with pytest.raises(GatewayClientError):
                wire.call('command.dispatch', session_id=tui, name='cron', arg='remove job-1')
            fired = _hook_lines(log)
        finally:
            wire.close()
    assert [(r['command'], r['args_raw'], r['session_key']) for r in fired] == [
        ('cron', '', tui), ('cron', 'list', tui), ('cron', 'list --all', tui), ('status', '', tui)], fired


@pytest.mark.platforms("linux")
def test_commands_that_still_run_during_a_turn_fire_except_the_escape_hatches(tmp_path):
    """Main's CLI fired ``pre_command`` for every command it ran during a turn. Kept: ``/footer`` and
    ``/busy`` (``busy_policy='dispatch'``) and a CLI-local command fire; only messaging's escape
    hatches (``/stop``, ``/approve``, ``/deny``) stay outside plugin reach while the turn runs."""
    with _gateway(tmp_path) as (_proc, desc, home, work, _env, log, peer):
        wire = _Wire(home, desc)
        try:
            sid = wire.call('session.create', request_id='busy', source='cli', toolsets=[], cwd=str(work))['session_id']
            wire.call('prompt.submit', session_id=sid, input_id='busy-turn', text='BLOCK_STARTED')
            assert peer.blocked.wait(30)
            wire.call('slash.exec', session_id=sid, command='/footer status')
            wire.call('slash.exec', session_id=sid, command='/busy queue')
            assert wire.call('command.observe', session_id=sid, name='history', arg='') == {'queued': True}
            assert wire.call('command.observe', session_id=sid, name='stop', arg='') == {'queued': False}
            assert wire.call('command.observe', session_id=sid, name='approve', arg='') == {'queued': False}
            wire.call('slash.exec', session_id=sid, command='/deny')
            # The barrier: one session's deliveries land in arrival order, so once this last one has,
            # every earlier one has too (and an escape hatch's, had it fired, would be in the log).
            assert wire.call('command.observe', session_id=sid, name='version', arg='') == {'queued': True}
            fired = _hooks_through(log, 'version')
            peer.release.set()
        finally:
            wire.close()
    assert [(r['command'], r['args_raw']) for r in fired] == [('footer', 'status'), ('busy', 'queue'),
                                                             ('history', ''), ('version', '')], fired


class _ProfileWire(_Wire):
    """``_Wire`` on a ticket minted for ``profile`` (a served profile's home)."""

    def __init__(self, home, desc, profile):
        import asyncio
        from websockets.asyncio.client import connect
        from tests.gateway.test_normal_runtime_boot import control
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.calls = []
        grant = control(home, 'session-ticket', {'profile_id': str(profile), 'instance_id': desc['instance_id'],
                                                 'purpose': 'interactive'})

        async def _open():
            return await connect(desc['api_origin'].replace('http:', 'ws:') + '/api/ws',
                                 subprotocols=['hermes-gateway-v1', 'hermes-gateway-ticket.' + grant['ticket']])
        self.ws = self._run(_open())


@pytest.mark.platforms("linux")
def test_pre_command_runs_in_the_plugin_host_under_host_isolation(tmp_path):
    """``plugins.isolation: host``: the hook fires where the dispatcher runs the plugin, the profile's
    plugin host, never inside the gateway process; same values and order as in-process."""
    with _gateway(tmp_path, isolation='host') as (proc, desc, home, work, _env, log, _peer):
        wire = _Wire(home, desc)
        try:
            sid = wire.call('session.create', request_id='host', source='cli', toolsets=[], cwd=str(work))['session_id']
            wire.call('slash.exec', session_id=sid, command='/help')
            assert wire.call('command.observe', session_id=sid, name='history', arg='') == {'queued': True}
            fired = _hooks_through(log, 'history')
        finally:
            wire.close()
    assert [(r['surface'], r['platform'], r['command'], r['session_key']) for r in fired] == [
        ('cli', 'cli', 'help', sid), ('cli', 'cli', 'history', sid)], fired
    hosts = {r['pid'] for r in fired}
    assert len(hosts) == 1 and proc.pid not in hosts, (proc.pid, hosts)


@pytest.mark.platforms("linux")
def test_pre_command_fires_in_the_sessions_own_profile(tmp_path):
    """A multiplexing gateway: a ``beta`` session's commands reach ``beta``'s plugin, under ``beta``'s
    home, and never the launch profile's copy; the launch profile's session the reverse."""
    with _gateway(tmp_path, beta=True) as (_proc, desc, home, work, _env, log, _peer):
        beta_home = (home / 'profiles' / 'beta').resolve()
        root, beta = _Wire(home, desc), _ProfileWire(home, desc, beta_home)
        try:
            root_sid = root.call('session.create', request_id='root', source='cli', toolsets=[], cwd=str(work))['session_id']
            beta_sid = beta.call('session.create', request_id='beta', source='tui', toolsets=[], cwd=str(work))['session_id']
            beta.call('slash.exec', session_id=beta_sid, command='/help')
            assert beta.call('command.observe', session_id=beta_sid, name='history', arg='') == {'queued': True}
            root.call('slash.exec', session_id=root_sid, command='/status')
            beta_fired = _hooks_through(tmp_path / 'beta.jsonl', 'history')
            root_fired = _hooks_through(log, 'status')
        finally:
            root.close()
            beta.close()
    assert [(r['command'], r['session_key'], r['home']) for r in beta_fired] == [
        ('help', beta_sid, str(beta_home)), ('history', beta_sid, str(beta_home))], beta_fired
    assert [(r['command'], r['session_key'], r['home']) for r in root_fired] == [
        ('status', root_sid, str(home.resolve()))], root_fired


@pytest.mark.platforms("linux")
def test_a_command_ink_answers_itself_fires_once_in_the_gateway_when_observed(tmp_path):
    """Ink answers ``/usage`` and ``/history`` without ``slash.exec`` and reports them through
    ``command.observe``: each fires ``pre_command`` once, in the gateway, under the TUI session's
    profile, with main's values. A name only Ink knows (``/fortune``) is refused and fires nothing."""
    from hermes_cli.gateway_client import GatewayClientError
    with _gateway(tmp_path) as (proc, desc, home, work, _env, log, _peer):
        wire = _Wire(home, desc)
        try:
            tui = wire.call('session.create', request_id='ink', source='tui', toolsets=[], cwd=str(work))['session_id']
            assert wire.call('command.observe', session_id=tui, name='usage', arg='') == {'queued': True}
            with pytest.raises(GatewayClientError):
                wire.call('command.observe', session_id=tui, name='fortune', arg='')
            assert wire.call('command.observe', session_id=tui, name='history', arg='') == {'queued': True}
            fired = _hooks_through(log, 'history')
        finally:
            wire.close()
    assert [(r['pid'], r['surface'], r['platform'], r['command'], r['alias_used'], r['session_key'], r['home'])
            for r in fired] == [(proc.pid, 'cli', 'cli', command, command, tui, str(home.resolve()))
                                for command in ('usage', 'history')], fired


@pytest.mark.platforms("linux")
def test_commands_refused_during_a_turn_fire_nothing(tmp_path):
    """``/reasoning`` is a settings control the busy dispatcher rejects during a turn (default
    ``busy_policy``): through ``slash.exec`` and ``command.dispatch`` it is refused and fires nothing.
    A mid-run variant that refuses its arguments (``/goal <new text>``, ``/loop <new loop>``) fires
    nothing either; the same variant accepting a control verb (``/goal status``) fires once."""
    from agent.i18n import t
    with _gateway(tmp_path) as (_proc, desc, home, work, _env, log, peer):
        wire = _Wire(home, desc)
        try:
            sid = wire.call('session.create', request_id='refused', source='cli', toolsets=[], cwd=str(work))['session_id']
            wire.call('prompt.submit', session_id=sid, input_id='refused-turn', text='BLOCK_STARTED')
            assert peer.blocked.wait(30)
            replies = [wire.call('slash.exec', session_id=sid, command='/reasoning high'),
                       wire.call('command.dispatch', session_id=sid, name='reasoning', arg='high')]
            variants = [wire.call('slash.exec', session_id=sid, command='/goal ship the release'),
                        wire.call('command.dispatch', session_id=sid, name='loop', arg='every 5m check ci')]
            wire.call('slash.exec', session_id=sid, command='/goal status')
            assert wire.call('command.observe', session_id=sid, name='history', arg='') == {'queued': True}
            fired = _hooks_through(log, 'history')
            peer.release.set()
        finally:
            wire.close()
    assert [r['output'] for r in replies] == [t('gateway.busy.slash_rejected', command='reasoning')] * 2
    assert [r['output'] for r in variants] == [t('gateway.busy.reject_goal'), t('gateway.busy.reject_loop')]
    assert [(r['command'], r['args_raw']) for r in fired] == [('goal', 'status'), ('history', '')], fired


def test_commands_the_cli_refuses_on_the_gateway_fire_nothing(monkeypatch, capsys):
    """A named refusal (``/handoff``) and a registry command with no classic CLI handler run nothing in
    the CLI, so no ``command.observe`` goes out; a command the CLI runs is still reported."""
    from cli import HermesCLI
    from hermes_cli.commands import COMMAND_REGISTRY

    class Connection:
        epoch, final, closed = 1, False, False
        calls = []

        def call(self, method, **params):
            self.calls.append((method, params.get('name')))
            return {}

        def call_tagged(self, method, **params):
            return self.call(method, **params), self.epoch
    connection = Connection()
    monkeypatch.setattr('hermes_cli.gateway_side.connection_for', lambda _cli: connection)
    monkeypatch.setattr('hermes_cli.plugins.fire_pre_command_hook', lambda **_k: pytest.fail('fired locally'))
    unhandled = next(cmd.name for cmd in COMMAND_REGISTRY
                     if HermesCLI._slash_handler(cmd.name) is None and cmd.name != 'handoff')
    c = HermesCLI.__new__(HermesCLI)
    c.config, c.session_id = {}, 's1'
    c.process_command('/handoff telegram')
    c.process_command('/' + unhandled)
    assert [call for call in connection.calls if call[0] == 'command.observe'] == []
    c.process_command('/reload')
    assert ('command.observe', 'reload') in connection.calls


def _authority(state):
    from types import SimpleNamespace
    live = SimpleNamespace(source=object(), route='agent:main:local:dm:s1')
    return SimpleNamespace(sessions={'s1': live}, runner=object(), profile_id='/tmp/home',
                           _handle=lambda _ref: SimpleNamespace(execution_state=state))


@pytest.mark.parametrize('state, command, expected', [
    ('idle', 'status', 1), ('running', 'status', 1), ('idle', 'stop', 1),
    ('running', 'stop', 0), ('running', 'approve', 0), ('running', 'deny', 0)])
def test_only_the_escape_hatches_skip_pre_command_during_a_turn(monkeypatch, state, command, expected):
    import asyncio
    from contextlib import nullcontext
    from types import SimpleNamespace
    from gateway import session_commands
    calls = []
    monkeypatch.setattr('gateway.session_local.authorize_local_source', lambda _r, _s: True)
    monkeypatch.setattr('gateway.session_policy.policy_for_source', lambda _r, _s: SimpleNamespace(platform='tui'))
    monkeypatch.setattr('gateway.run._profile_runtime_scope', lambda _home: nullcontext())
    monkeypatch.setattr('hermes_cli.plugins.fire_pre_command_hook', lambda **kw: calls.append(kw))
    asyncio.run(session_commands._pre_command_before(_authority(state), SimpleNamespace(session_id='s1'),
                                                     command, command, ''))
    assert calls == [dict(surface='cli', command=command, alias_used=command, args_raw='',
                          session_key='s1', platform='cli')] * expected


def test_a_hung_pre_command_callback_is_bounded_and_each_session_keeps_its_order(monkeypatch):
    """``pre_command`` runs through the hook dispatcher's bounded path (``plugins.hook_callback_timeout``,
    abandoned worker, suppression), each delivery on a thread of its own, never the loop's shared
    executor. A callback that hangs for session A's first command: session B's observers run at once,
    A's next command waits only for the bound and skips the still-running callback (never overlapping
    or reordering it), later commands skip it, no extra worker piles up, and a session's backlog is capped."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import nullcontext
    from types import SimpleNamespace
    from gateway import session_commands
    from hermes_cli.plugins import PluginManager
    release, seen = threading.Event(), []

    def hung(session_key, command, **_kw):
        if (session_key, command) == ('A', 'first'):
            release.wait()

    def record(session_key, command, **_kw):
        seen.append((session_key, command, threading.current_thread().name.split('-')[1]))
    manager = PluginManager()
    manager._hooks['pre_command'] = [hung, record]
    monkeypatch.setattr('hermes_cli.plugins.get_plugin_manager', lambda: manager)
    monkeypatch.setattr('hermes_cli.plugins._resolve_hook_callback_timeout', lambda: 0.5)
    monkeypatch.setattr('gateway.session_local.authorize_local_source', lambda _r, _s: True)
    monkeypatch.setattr('gateway.run._profile_runtime_scope', lambda _home: nullcontext())
    live = SimpleNamespace(source=object(), route='agent:main:local:dm:s1')
    authority = SimpleNamespace(sessions=dict.fromkeys('ABC', live), runner=object(), profile_id='/tmp/home',
                                _handle=lambda _ref: SimpleNamespace(execution_state='idle'))

    class SharedExecutor(ThreadPoolExecutor):
        def submit(self, *_a, **_k):
            raise AssertionError('pre_command ran on the loop\'s shared default executor')

    def deliver(session, command):
        return session_commands._pre_command(authority, SimpleNamespace(session_id=session), command, command, '')

    async def run():
        asyncio.get_running_loop().set_default_executor(SharedExecutor(1))
        a1, b1 = deliver('A', 'first'), deliver('B', 'b1')
        await asyncio.wait({b1})  # B is not queued behind A's hung callback
        assert seen == [('B', 'b1', 'hook')] and not a1.done()
        a2 = deliver('A', 'second')
        await asyncio.wait({a1, a2})  # A1 returns at the bound; A2 after it, skipping the hung call
        assert seen[1:] == [('A', 'first', 'hook'), ('A', 'second', 'hook')]
        later = [deliver(s, 'first') for s in 'AB' * 10]  # the hung callback is suppressed now
        await asyncio.wait(later)
        assert len(seen) == 23
        backlog = [deliver('C', str(n)) for n in range(session_commands._PRE_COMMAND_PENDING + 5)]
        assert backlog.count(None) == 5  # past the cap a delivery is dropped (logged), not queued
        await asyncio.wait([task for task in backlog if task is not None])
    try:
        asyncio.run(run())
        assert list(manager._hook_abandoned.values()) == [{('pre_command', id(hung), 'A')}]  # one, ever
    finally:
        release.set()


def test_callbacks_waiting_out_their_bound_hold_no_worker_another_session_needs(monkeypatch):
    """Six sessions whose ``pre_command`` callback is still inside ``plugins.hook_callback_timeout`` hold
    no shared observer worker: another session's delivery runs at once, not when a slot frees."""
    import asyncio
    from contextlib import nullcontext
    from types import SimpleNamespace
    from gateway import session_commands
    from hermes_cli.plugins import PluginManager
    slow, inside, seen, lock = [f'S{n}' for n in range(6)], [], [], threading.Lock()
    release, all_in = threading.Event(), threading.Event()

    def callback(session_key, **_kw):
        if session_key not in slow:
            seen.append(session_key)
            return
        with lock:
            inside.append(session_key)
            if len(inside) == len(slow):
                all_in.set()
        release.wait()
    manager = PluginManager()
    manager._hooks['pre_command'] = [callback]
    monkeypatch.setattr('hermes_cli.plugins.get_plugin_manager', lambda: manager)
    monkeypatch.setattr('hermes_cli.plugins._resolve_hook_callback_timeout', lambda: 30)
    monkeypatch.setattr('gateway.session_local.authorize_local_source', lambda _r, _s: True)
    monkeypatch.setattr('gateway.run._profile_runtime_scope', lambda _home: nullcontext())
    live = SimpleNamespace(source=object(), route='agent:main:local:dm:s1')
    authority = SimpleNamespace(sessions=dict.fromkeys([*slow, 'fresh'], live), runner=object(),
                                profile_id='/tmp/home', _handle=lambda _ref: SimpleNamespace(execution_state='idle'))

    def deliver(session):
        return session_commands._pre_command(authority, SimpleNamespace(session_id=session), 'help', 'help', '')

    async def run():
        stuck = [deliver(s) for s in slow]
        try:
            fresh = deliver('fresh')
            done, _ = await asyncio.wait({fresh}, timeout=10)  # a deadline: GREEN returns at once
            assert done == {fresh} and seen == ['fresh'], (done, seen, inside)
            assert await asyncio.to_thread(all_in.wait, 10), inside  # every slow session entered its callback
        finally:
            release.set()
        await asyncio.wait(stuck)
    try:
        asyncio.run(run())
    finally:
        release.set()


def test_an_older_gateway_without_plugins_list_is_named_and_loads_nothing(monkeypatch, capsys):
    from hermes_cli import gateway_plugin_commands
    from hermes_cli.gateway_client import GatewayRPCError

    class Older:
        def call(self, method, **_params):
            raise GatewayRPCError('unknown_method', code=-32601)
    monkeypatch.setattr('hermes_cli.gateway_side.connection_for', lambda _cli: Older())
    monkeypatch.setattr('hermes_cli.plugins.get_plugin_manager', lambda: pytest.fail('loaded plugins locally'))
    assert gateway_plugin_commands.loaded(object()) == {}
    assert "predates this CLI" in capsys.readouterr().out
    monkeypatch.setattr('hermes_cli.gateway_side.connection_for', lambda _cli: None)
    assert gateway_plugin_commands.loaded(object()) is None  # off the gateway: main's in-process listing


def test_an_older_gateway_without_command_observe_is_named_once_and_nothing_fires_locally(monkeypatch, capsys):
    from hermes_cli import gateway_plugin_commands
    from hermes_cli.gateway_client import GatewayRPCError

    class Older:
        epoch = 1
        calls = []

        def call(self, method, **params):
            self.calls.append((method, params))
            raise GatewayRPCError('unknown_method', code=-32601)
    older = Older()
    monkeypatch.setattr('hermes_cli.gateway_side.connection_for', lambda _cli: older)
    monkeypatch.setattr('hermes_cli.plugins.fire_pre_command_hook', lambda **_k: pytest.fail('fired locally'))
    monkeypatch.setattr('hermes_cli.plugins.get_plugin_manager', lambda: pytest.fail('loaded plugins locally'))
    cli = type('CLI', (), {'session_id': 's1'})()
    assert gateway_plugin_commands.observe(cli, 'history', '') is True
    assert gateway_plugin_commands.observe(cli, 'version', '') is True
    assert older.calls == [('command.observe', {'session_id': 's1', 'name': 'history', 'arg': ''})]
    assert capsys.readouterr().out.count('predates this CLI') == 1
    older.epoch = 2  # a redial (maybe to a restarted, newer gateway) asks again, without a second notice
    assert gateway_plugin_commands.observe(cli, 'history', 'x') is True
    assert len(older.calls) == 2 and 'predates' not in capsys.readouterr().out
    monkeypatch.setattr('hermes_cli.gateway_side.connection_for', lambda _cli: None)
    assert gateway_plugin_commands.observe(cli, 'history', '') is False  # off the gateway: main's own hook
