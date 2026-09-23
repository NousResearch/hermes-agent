"""`hermes chat --safe-mode` is an owner-frozen policy executed out of process, not client env.

Contract recipe D-F: sentinels seeded in a disposable profile fire on an ordinary turn
(positive control); the canonical safe launch through the real CLI gets a loopback reply from
a managed worker that opened no plugin, profile YAML, rules or skill file and fired no sentinel;
a malformed profile does not block it; config-only keeps plugin discovery.
"""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import pty
import re
import select
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket


class Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if body.get('messages'):
            self.server.requests.append({'auth': self.headers.get('Authorization'), 'path': self.path, 'body': body})
        message = {'role': 'assistant', 'content': 'SAFE_REPLY_OK'}
        choice = {'index': 0, 'message': message, 'finish_reason': 'stop'}
        frame = {'id': 'local', 'model': body.get('model'), 'choices': [choice],
                 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
        kind = 'application/json'
        if body.get('stream'):
            choice['delta'] = choice.pop('message')
            payload = ('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        else:
            payload = json.dumps(frame).encode()
        try:
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _pty_cli(argv, cwd, env, timeout=90):
    """Real PTY launch: stdin/stdout/stderr are a terminal, like a user's shell."""
    master, slave = pty.openpty()
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    output, deadline = bytearray(), time.monotonic() + timeout
    with closing(os.fdopen(master, 'rb', buffering=0)) as reader:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([reader], [], [], .2)
            if ready:
                try:
                    chunk = reader.read(65536)
                except OSError:
                    break
                if not chunk:
                    break
                output.extend(chunk)
            elif proc.poll() is not None:
                break
    code = proc.wait(timeout=10)
    return code, output.decode('utf-8', 'replace')


@pytest.mark.linux_only
def test_canonical_safe_launch_is_frozen_policy_executed_in_isolated_worker(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user, work = tmp_path / 'state', tmp_path / 'user', tmp_path / 'work'
    home.mkdir(mode=0o700)
    user.mkdir()
    work.mkdir()
    events = home / 'events'
    events.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    # Sentinels: general plugin (import + pre_llm_call hook), shell hook, SOUL, project rules, skill.
    plugin = home / 'plugins' / 'sentinel'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text('name: sentinel\nversion: 1.0.0\nkind: standalone\n', encoding='utf-8')
    (plugin / '__init__.py').write_text(
        'import os\nfrom pathlib import Path\n'
        f'EVENTS = Path({str(events)!r})\n'
        '(EVENTS / f"plugin-import-{os.getpid()}").touch()\n'
        'def register(ctx):\n'
        '    ctx.register_hook("pre_llm_call", lambda **kw: (EVENTS / f"plugin-hook-{os.getpid()}").touch())\n',
        encoding='utf-8')
    shell_witness = tmp_path / 'shell_witness.py'
    shell_witness.write_text(f'import os; from pathlib import Path\nPath({str(events)!r}, "shell-" + str(os.getppid())).touch()\n', encoding='utf-8')
    (home / 'SOUL.md').write_text('SOUL_SENTINEL identity text', encoding='utf-8')
    (work / 'AGENTS.md').write_text('RULES_SENTINEL project rules', encoding='utf-8')
    skill = home / 'skills' / 'probe'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('---\nname: probe\ndescription: SKILL_SENTINEL probe skill\n---\nSKILL_SENTINEL body\n', encoding='utf-8')
    config = {'gateway': {'multiplex_profiles': False},
              'model': {'provider': 'custom', 'default': 'owner-model', 'base_url': url},
              'auxiliary': {'title_generation': {'enabled': False}},
              'plugins': {'enabled': ['sentinel']}, 'hooks_auto_accept': True,
              'hooks': {'pre_llm_call': [{'command': f'{sys.executable} {shell_witness}'}]},
              'memory': {'memory_enabled': False, 'user_profile_enabled': False},
              'platform_toolsets': {'cli': ['terminal']}}
    (home / 'config.yaml').write_text(json.dumps(config), encoding='utf-8')
    audit = tmp_path / 'opens.jsonl'
    site = tmp_path / 'audit-site'
    site.mkdir()
    (site / 'sitecustomize.py').write_text(
        "import json, os, sys\n"
        f"AUDIT = {str(audit)!r}\n"
        "def record(kind, **fields):\n"
        "    with open(AUDIT, 'a', encoding='utf-8') as f:\n"
        "        f.write(json.dumps({'kind': kind, 'pid': os.getpid(), **fields}) + '\\n')\n"
        "# sys.argv is not populated yet at site time; the kernel cmdline is.\n"
        "with open('/proc/self/cmdline', 'rb') as f:\n"
        "    cmdline = f.read().split(b'\\0')\n"
        "record('start', ppid=os.getppid(), argv=[a.decode('utf-8', 'replace') for a in cmdline if a])\n"
        "import threading\n"
        "_busy = threading.local()\n"
        "def witness(event, args):\n"
        "    if event != 'open' or AUDIT in str(args[0]) or getattr(_busy, 'on', False):\n"
        "        return\n"
        "    _busy.on = True\n"
        "    try:\n"
        "        path = str(args[0])\n"
        "        fields = {}\n"
        "        if '/plugins/' in path or path.endswith(('config.yaml', '.env', 'SOUL.md', 'AGENTS.md', 'SKILL.md')):\n"
        "            frame, stack = sys._getframe(1), []\n"
        "            while frame is not None and len(stack) < 14:\n"
        "                stack.append(f'{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}')\n"
        "                frame = frame.f_back\n"
        "            fields['stack'] = stack\n"
        "        record('open', path=path, mode=str(args[1]), **fields)\n"
        "    finally:\n"
        "        _busy.on = False\n"
        "sys.addaudithook(witness)\n", encoding='utf-8')
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), TERM='xterm',
               PYTHONPATH=os.pathsep.join([str(site), str(root)]), OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)

    def query(sql, args=()):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return db.execute(sql, args).fetchall()

    def records():
        return [json.loads(line) for line in audit.read_text(encoding='utf-8').splitlines()]

    def worker_pids(owner_pid):
        return sorted({r['pid'] for r in records() if r['kind'] == 'start' and r['ppid'] == owner_pid
                       and r['argv'][-2:] == ['-m', 'agent.managed_worker']})

    def system_text(request):
        return json.dumps([m for m in request['body']['messages'] if m['role'] in {'system', 'developer'}])

    async def ordinary_turn(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='ordinary', source='cli', cwd=str(work),
                                model='owner-model', provider='custom', base_url=url, api_key='loopback-only')
            assert 'result' in created, created
            sid = created['result']['session_id']
            submitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id='ordinary-input', text='ORDINARY_PROBE')
            assert 'result' in submitted, submitted
            async with asyncio.timeout(60):
                while query('SELECT status FROM session_admissions WHERE request_id=?', ('ordinary-input',)) != [('terminal',)]:
                    await asyncio.sleep(.05)
            return sid

    def safe_cli(request_tag, *flags):
        code, output = _pty_cli([sys.executable, '-m', 'hermes_cli.main', 'chat', *flags, '--provider', 'custom',
                                 '--base-url', url, '--model', 'safe-fixture', '--api-key', 'fixture',
                                 '-Q', '-q', 'SAFE_PROBE_' + request_tag], cwd=work, env=env)
        assert code == 0, output
        assert 'SAFE_REPLY_OK' in output, output
        sid = re.search(r'Session: (\S+)', output).group(1)
        return sid, output

    try:
        with daemon(root, home, env, barrier=False) as (owner, desc):
            # Positive control: the ordinary in-process turn fires every sentinel in the owner.
            asyncio.run(ordinary_turn(desc))
            ordinary = [r for r in peer.requests if r['body']['model'] == 'owner-model']
            assert len(ordinary) == 1, [r['body']['model'] for r in peer.requests]
            assert 'SOUL_SENTINEL' in system_text(ordinary[0]) and 'RULES_SENTINEL' in system_text(ordinary[0])
            skill_in_ordinary = 'SKILL_SENTINEL' in system_text(ordinary[0])
            fired = sorted(p.name for p in events.iterdir())
            assert f'plugin-import-{owner.pid}' in fired and f'plugin-hook-{owner.pid}' in fired, fired
            assert f'shell-{owner.pid}' in fired, fired
            owner_opens = [r['path'] for r in records() if r['kind'] == 'open' and r['pid'] == owner.pid]
            assert any('/plugins/' in p for p in owner_opens) and any(p.endswith('config.yaml') for p in owner_opens)

            # Canonical safe launch through the real CLI on a real PTY.
            safe_sid, output = safe_cli('one', '--safe-mode')
            assert query('SELECT status FROM session_admissions WHERE target_session_id=?', (safe_sid,)) == [('terminal',)]
            assert query('SELECT status FROM worker_executions WHERE session_id=?', (safe_sid,)) == [('terminal',)]
            policy = json.loads(query('SELECT value FROM state_meta WHERE key=?', ('gateway.local_policy.v1:' + safe_sid,))[0][0])['policy']
            assert policy['safe_mode'] is True and policy['ignore_user_config'] is True
            workers = worker_pids(owner.pid)
            assert len(workers) == 1 and workers[0] != owner.pid, workers
            worker = workers[0]
            worker_records = [r for r in records() if r['kind'] == 'open' and r['pid'] == worker]
            assert worker_records, 'audit hook not live in the worker'
            leaked = [(r['path'], r.get('stack')) for r in worker_records
                      if '/plugins/' in r['path'] or r['path'].endswith(('config.yaml', '.env', 'SOUL.md', 'AGENTS.md', 'SKILL.md'))]
            assert leaked == [], json.dumps(leaked, indent=1)
            assert not [name for name in os.listdir(events) if name.endswith(f'-{worker}')], sorted(os.listdir(events))
            def inferences(tag):
                # Title generation is a code default and also targets the explicit endpoint; the
                # probe turn itself must be inferred exactly once.
                return [r for r in peer.requests if r['body']['model'] == 'safe-fixture'
                        and r['body']['messages'][-1].get('content') == 'SAFE_PROBE_' + tag
                        and not str(r['body']['messages'][0].get('content', '')).startswith('You name chat sessions')]
            safe = inferences('one')
            assert len(safe) == 1 and safe[0]['auth'] == 'Bearer fixture' and safe[0]['path'] == '/v1/chat/completions', \
                [(r['path'], [str(m.get('content'))[:80] for m in r['body']['messages']]) for r in peer.requests]
            assert not any(s in system_text(safe[0]) for s in ('SOUL_SENTINEL', 'RULES_SENTINEL', 'SKILL_SENTINEL'))
            assert 'HERMES_SAFE_MODE' not in Path(f'/proc/{owner.pid}/environ').read_bytes().decode('utf-8', 'replace')

            # Malformed profile YAML: bypass launch still works and neither repairs nor rewrites it.
            (home / 'config.yaml').write_text('model: [unterminated\n', encoding='utf-8')
            before = (home / 'config.yaml').read_bytes()
            second_sid, _ = safe_cli('two', '--safe-mode')
            assert second_sid != safe_sid
            assert (home / 'config.yaml').read_bytes() == before
            second = inferences('two')
            assert len(second) == 1 and second[0]['body'].get('tools') == safe[0]['body'].get('tools')

            # Config-only keeps plugin discovery (the gate is safe_mode), yet reads no profile YAML.
            (home / 'config.yaml').write_text(json.dumps(config), encoding='utf-8')
            config_sid, _ = safe_cli('three', '--ignore-user-config')
            config_worker = [p for p in worker_pids(owner.pid) if p not in workers][-1]
            config_opens = [r['path'] for r in records() if r['kind'] == 'open' and r['pid'] == config_worker]
            assert any('/plugins/' in p for p in config_opens), 'config-only must still discover plugins'
            assert not [p for p in config_opens if p.endswith(('config.yaml', '.env'))]
            print(json.dumps({'owner_pid': owner.pid, 'safe_worker_pid': worker, 'config_only_worker_pid': config_worker,
                              'safe_session': safe_sid, 'skill_in_ordinary_prompt': skill_in_ordinary,
                              'sentinels_fired': fired, 'safe_worker_leaked_opens': leaked,
                              'model_requests': [r['body']['model'] for r in peer.requests]}))
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
