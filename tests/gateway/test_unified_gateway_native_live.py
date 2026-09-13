"""Native live proof of the unified gateway runtime on the CURRENT host OS.

An ordinary ``python -m gateway.run`` in a disposable HERMES_HOME with a loopback
model must, on Linux, macOS and native Windows alike: publish discovery + tickets;
attach WebSockets; admit/execute/receipt prompts; keep two clients FIFO on one
session; survive an owner hard-kill (``gateway.status.terminate_pid``) with the
committed queued admission executed exactly once; branch/delete with retryable
receipts; run the safe-mode CLI turn in an out-of-process managed worker that
opens no writable canonical SQLite; refuse a second owner while the first holds
``gateway.lock``; and release lock, control channel and API port on planned stop.
No POSIX-only primitive is used (no AF_UNIX at import, no SIGKILL, no pty).
"""
import asyncio
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

import pytest
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[2]
# Windows children need the system environment (Winsock, temp, PATHEXT) that the
# hermetic test process still carries; POSIX only ever uses the first three.
_CHILD_ENV_KEYS = ('PATH', 'LANG', 'TZ', 'SYSTEMROOT', 'SYSTEMDRIVE', 'WINDIR', 'COMSPEC', 'TEMP', 'TMP',
                   'PATHEXT', 'LOCALAPPDATA', 'APPDATA', 'PROGRAMDATA', 'NUMBER_OF_PROCESSORS',
                   'PROCESSOR_ARCHITECTURE')


class Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        text = ''
        if body.get('messages'):
            text = next((m.get('content', '') for m in reversed(body['messages']) if m['role'] == 'user'), '')
            first = str(body['messages'][0].get('content', ''))
            self.server.requests.append({'model': body.get('model'), 'text': text,
                                         'auth': self.headers.get('Authorization'),
                                         'title_generation': first.startswith('You name chat sessions')})
            if isinstance(text, str) and text.startswith('BLOCK_'):
                self.server.blocked.set()
                self.server.release.wait(60)
        message = {'role': 'assistant', 'content': 'NATIVE_ACK_' + str(text)}
        choice = {'index': 0, 'message': message, 'finish_reason': 'stop'}
        frame = {'id': 'native', 'model': body.get('model'), 'choices': [choice],
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


def control(home, verb, params=None):
    """Bootstrap control channel: private named pipe on Windows, AF_UNIX elsewhere."""
    request = json.dumps({'protocol': 1, 'id': 1, 'verb': verb, 'params': params or {}}).encode() + b'\n'
    if os.name == 'nt':
        from gateway.runtime_bootstrap_windows import query_runtime_control
        raw = query_runtime_control(home, request, 2)
    else:
        from gateway.control_socket import resolve_client_socket_path
        path = resolve_client_socket_path(home)
        if path is None:
            raise FileNotFoundError('control socket absent')
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.settimeout(2)
            peer.connect(str(path))
            peer.sendall(request)
            with peer.makefile('rb') as stream:
                raw = stream.readline()
    reply = json.loads(raw.split(b'\n', 1)[0])
    assert reply.get('ok'), reply
    return reply['result']


def control_absent(home):
    try:
        control(home, 'identify')
    except (OSError, ValueError, AssertionError):
        return True
    return False


async def rpc(ws, method, **params):
    await ws.send(json.dumps({'jsonrpc': '2.0', 'id': method, 'method': method, 'params': params}))
    async with asyncio.timeout(30):
        while True:
            reply = json.loads(await ws.recv())
            if reply.get('id') == method:
                return reply


async def ok(ws, method, **params):
    reply = await rpc(ws, method, **params)
    assert 'result' in reply, reply
    return reply['result']


def websocket(home, desc, purpose='interactive'):
    grant = control(home, 'session-ticket', {'profile_id': desc['served_profiles'][0]['profile_id'],
                                             'instance_id': desc['instance_id'], 'purpose': purpose})
    return connect(desc['api_origin'].replace('http:', 'ws:') + '/api/ws',
                   subprotocols=['hermes-gateway-v1', 'hermes-gateway-ticket.' + grant['ticket']],
                   open_timeout=20)


def process_start_time(pid):
    from gateway.status import get_process_start_time
    return get_process_start_time(pid)


def hard_kill(proc):
    """Owner death without cleanup: identity-guarded force kill, never a raw SIGKILL."""
    from gateway.status import terminate_pid
    terminate_pid(proc.pid, force=True, expected_start_time=process_start_time(proc.pid))
    proc.wait(timeout=30)


def planned_stop(home, proc, desc):
    """The ``hermes gateway stop`` contract: marker first, then the owner drains itself.

    The marker names the gateway's own pid from ``identify`` (what the PID file carries), not
    the Popen handle: a Windows uv venv ``python.exe`` is a trampoline whose pid differs."""
    pid = desc['pid']
    marker = {'target_pid': pid, 'target_start_time': process_start_time(pid),
              'stopper_pid': os.getpid(), 'written_at': datetime.now(timezone.utc).isoformat()}
    (home / '.gateway-planned-stop.json').write_text(json.dumps(marker), encoding='utf-8')
    try:
        return proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        detail = (home / 'logs' / 'gateway.log').read_text(encoding='utf-8', errors='replace')[-6000:]
        stacks = home / 'logs' / 'stacks.txt'
        if stacks.exists():
            artifacts = Path(os.environ.get('UGW_ARTIFACT_DIR', str(home.parent)))
            artifacts.mkdir(parents=True, exist_ok=True)
            (artifacts / f'stacks-stop-{proc.pid}.txt').write_text(stacks.read_text(encoding='utf-8', errors='replace'), encoding='utf-8')
        raise AssertionError(('planned stop did not exit within 90s', marker,
                              (home / '.gateway-planned-stop.json').exists(), detail))


@contextmanager
def daemon(home, env, log_name, *, fixture=None, expect_ready=True):
    command = [sys.executable, str(ROOT / 'tests/gateway/fixtures' / fixture)] if fixture else [sys.executable, '-m', 'gateway.run']
    with (home.parent / log_name).open('w+', encoding='utf-8', errors='replace') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            desc = {}
            deadline = time.monotonic() + 180
            while proc.poll() is None and time.monotonic() < deadline:
                try:
                    desc = control(home, 'identify')
                    if desc.get('state') == 'ready':
                        break
                except (OSError, ValueError, AssertionError):
                    pass
                time.sleep(.1)
            if expect_ready:
                log.flush(); log.seek(0)
                detail = {name: (home / 'logs' / name).read_text(encoding='utf-8', errors='replace')[-6000:]
                          for name in ('gateway.log',) if (home / 'logs' / name).exists()}
                stacks = home / 'logs' / 'stacks.txt'
                if stacks.exists():
                    # Full dump into the CI artifact directory; the assertion message would truncate it.
                    artifacts = Path(os.environ.get('UGW_ARTIFACT_DIR', str(home.parent)))
                    artifacts.mkdir(parents=True, exist_ok=True)
                    (artifacts / f'stacks-{proc.pid}.txt').write_text(stacks.read_text(encoding='utf-8', errors='replace'), encoding='utf-8')
                    detail['stacks.txt'] = stacks.read_text(encoding='utf-8', errors='replace')
                assert desc.get('state') == 'ready', (json.dumps(desc, default=str), proc.poll(), log.read()[-3000:], detail)
            yield proc, desc, log
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=20)


def query(home, sql, args=()):
    uri = (home / 'state.db').as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as db:
        return db.execute(sql, args).fetchall()


def admissions(home):
    return dict(query(home, 'SELECT request_id, status FROM session_admissions'))


def wait_status(home, request_id, status, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if admissions(home).get(request_id) == status:
            return
        time.sleep(.05)
    raise AssertionError((request_id, status, admissions(home)))


def api_port(desc):
    return urlsplit(desc['api_origin']).port


def port_open(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture
def peer():
    server = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    server.requests = []
    server.blocked, server.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def harness(tmp_path, peer):
    home, user, work = tmp_path / 'state', tmp_path / 'user', tmp_path / 'work'
    home.mkdir(mode=0o700); user.mkdir(); work.mkdir()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'native-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}},
        'platform_toolsets': {'cli': []}}), encoding='utf-8')
    audit = tmp_path / 'audit.jsonl'
    site = tmp_path / 'audit-site'
    site.mkdir()
    # strace is unavailable on Windows/macOS runners: every interpreter records its
    # argv (sys.orig_argv, populated before site) and every sqlite3.connect.
    (site / 'sitecustomize.py').write_text(
        "import json, os, sys\n"
        f"AUDIT = {str(audit)!r}\n"
        "def record(kind, **fields):\n"
        "    with open(AUDIT, 'a', encoding='utf-8') as f:\n"
        "        f.write(json.dumps({'kind': kind, 'pid': os.getpid(), **fields}) + '\\n')\n"
        "def ancestors():\n"
        "    try:\n"
        "        import psutil\n"
        "        return [(p.pid, p.name()) for p in psutil.Process().parents()[:4]]\n"
        "    except Exception as exc:\n"
        "        return [repr(exc)]\n"
        "record('start', ppid=os.getppid(), argv=list(sys.orig_argv), executable=sys.executable, ancestors=ancestors())\n"
        "def witness(event, args):\n"
        "    if event == 'sqlite3.connect':\n"
        "        record('sqlite', target=str(args[0]))\n"
        "sys.addaudithook(witness)\n"
        "# The managed worker reports failures to its owner as an opaque 'error' frame (no traceback\n"
        "# leaves the process by design); the frame is encoded inside the except block, so the live\n"
        "# exception is still visible here for the test's own diagnostics.\n"
        "import json as _json, traceback\n"
        "_dumps = _json.dumps\n"
        "def _dumps_witness(obj, *a, **kw):\n"
        "    if type(obj) is dict and obj.get('type') == 'error' and 'reason' in obj:\n"
        "        record('worker_error', reason=obj.get('reason'), traceback=traceback.format_exc())\n"
        "    return _dumps(obj, *a, **kw)\n"
        "_json.dumps = _dumps_witness\n", encoding='utf-8')
    env = {k: os.environ[k] for k in _CHILD_ENV_KEYS if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONUNBUFFERED='1',
               PYTHONPATH=os.pathsep.join([str(site), str(ROOT)]),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    return {'home': home, 'work': work, 'env': env, 'url': url, 'audit': audit, 'peer': peer}


def records(audit):
    if not audit.exists():
        return []
    return [json.loads(line) for line in audit.read_text(encoding='utf-8').splitlines() if line.strip()]


def test_native_gateway_runtime_live(harness):
    home, env, peer, url, audit = harness['home'], harness['env'], harness['peer'], harness['url'], harness['audit']
    receipt = {'os': sys.platform}

    async def discovery_attach_and_turn(desc):
        from websockets.exceptions import InvalidStatus
        ws_url = desc['api_origin'].replace('http:', 'ws:') + '/api/ws'
        with pytest.raises(InvalidStatus):
            async with connect(ws_url, open_timeout=20):
                pass
        assert 'session-authority-v1' in desc['capabilities'] and desc['authority_epoch'] > 0
        async with websocket(home, desc) as ws:
            assert ws.subprotocol == 'hermes-gateway-v1'
            created = await ok(ws, 'session.create', request_id='native-create', source='cli', cwd=str(home), toolsets=[])
            sid = created['session_id']
            initial = await ok(ws, 'session.resume', session_id=sid)
            admitted = await ok(ws, 'prompt.submit', session_id=sid, input_id='native-turn', text='NATIVE_TURN')
            assert admitted['status'] == 'queued', admitted
            async with asyncio.timeout(60):
                while True:
                    frame = json.loads(await ws.recv())
                    params = frame.get('params', {})
                    if frame.get('method') == 'event' and params.get('type') == 'message.complete':
                        assert 'NATIVE_ACK_NATIVE_TURN' in params['payload']['text'], frame
                        break
            assert admissions(home)['native-turn'] == 'terminal'
            replay = await ok(ws, 'session.events.since', session_id=sid, replay_epoch=initial['replay_epoch'],
                              last_sequence=initial['last_sequence'])
            assert [e['type'] for e in replay['events']].count('message.complete') == 1, replay
        return sid

    async def two_clients_fifo(desc, sid):
        async with websocket(home, desc) as first, websocket(home, desc) as second:
            await ok(second, 'session.resume', session_id=sid)
            await ok(first, 'session.resume', session_id=sid)
            head = await ok(first, 'prompt.submit', session_id=sid, input_id='fifo-head', text='BLOCK_FIFO_HEAD')
            assert head['status'] == 'queued', head
            assert await asyncio.to_thread(peer.blocked.wait, 30)
            tail = await ok(second, 'prompt.submit', session_id=sid, input_id='fifo-tail', text='FIFO_TAIL')
            assert tail['status'] == 'queued', tail
            assert admissions(home)['fifo-head'] == 'started' and admissions(home)['fifo-tail'] == 'queued', admissions(home)
            peer.release.set()
            completed = []
            async with asyncio.timeout(60):
                while len(completed) < 2:
                    frame = json.loads(await second.recv())
                    params = frame.get('params', {})
                    if frame.get('method') == 'event' and params.get('type') == 'message.complete':
                        completed.append(params['payload']['text'])
            assert completed == ['NATIVE_ACK_BLOCK_FIFO_HEAD', 'NATIVE_ACK_FIFO_TAIL'], completed
        texts = [r['text'] for r in peer.requests]
        assert texts.index('BLOCK_FIFO_HEAD') < texts.index('FIFO_TAIL'), texts
        peer.blocked.clear(); peer.release.clear()

    async def branch_and_delete(desc, sid):
        async with websocket(home, desc) as ws:
            before = await ok(ws, 'session.resume', session_id=sid)
            branch = dict(session_id=sid, request_id='native-branch', operation='branch', payload={},
                          expected_revision=before['revision'], expected_generation=before['execution_generation'])
            branched = await ok(ws, 'session.mutate', **branch)
            child = branched['branched_session_id']
            assert child != sid
            assert await ok(ws, 'session.mutate', **branch) == branched, 'branch retry must replay its receipt'
            stale = await rpc(ws, 'session.mutate', **{**branch, 'request_id': 'stale-branch'})
            assert stale['error']['message'] == 'revision_conflict', stale
            child_snapshot = await ok(ws, 'session.resume', session_id=child)
            assert len(child_snapshot['messages']) == len(before['messages']) > 0
            delete = dict(session_id=child, request_id='native-delete', operation='delete', payload={},
                          expected_revision=child_snapshot['revision'],
                          expected_generation=child_snapshot['execution_generation'])
            deleted = await ok(ws, 'session.mutate', **delete)
            assert child in deleted['deleted_ids'], deleted
            assert await ok(ws, 'session.mutate', **delete) == deleted, 'delete retry must replay its receipt'
            gone = await rpc(ws, 'session.resume', session_id=child)
            assert 'error' in gone, gone
            assert query(home, 'SELECT 1 FROM sessions WHERE id=?', (child,)) == []
            assert query(home, 'SELECT 1 FROM sessions WHERE id=?', (sid,)) == [(1,)]
        return child

    def lock_exclusion(owner, desc):
        from gateway.runtime_ownership import exclusive_maintenance, OwnershipConflict
        with pytest.raises(OwnershipConflict):
            with exclusive_maintenance([home]):
                pass
        with (home.parent / 'second.log').open('w+', encoding='utf-8', errors='replace') as log:
            second = subprocess.Popen([sys.executable, '-m', 'gateway.run'], cwd=ROOT, env=env,
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = second.wait(timeout=120)
            except subprocess.TimeoutExpired:
                second.kill(); second.wait(timeout=20)
                raise AssertionError('second gateway did not refuse while the first owns gateway.lock')
            log.flush(); log.seek(0)
            text = log.read()
        assert code != 0, text
        # The refusal is logged before the stderr handler exists (gap: silent on the console);
        # the durable errors.log carries the ownership refusal.
        errors = (home / 'logs' / 'errors.log').read_text(encoding='utf-8', errors='replace')
        assert 'already running' in text + errors or 'Cannot reserve gateway profiles' in text + errors, (text, errors[-2000:])
        assert owner.poll() is None
        assert control(home, 'identify')['instance_id'] == desc['instance_id']
        return code

    # Same ordinary gateway.run entry; the wrapper only dumps every thread's stack to
    # logs/stacks.txt after 60s so a readiness stall on a native runner names its frame.
    with daemon(home, {**env, 'UGW_STACK_DUMP_AFTER': '20'}, 'first.log', fixture='stack_dump_daemon.py') as (owner, desc, _):
        receipt['first_pid'] = owner.pid
        sid = asyncio.run(discovery_attach_and_turn(desc))
        asyncio.run(two_clients_fifo(desc, sid))
        receipt['branch_deleted'] = asyncio.run(branch_and_delete(desc, sid))
        receipt['second_owner_exit'] = lock_exclusion(owner, desc)
        port = api_port(desc)
        code = planned_stop(home, owner, desc)
        assert code == 0, (home.parent / 'first.log').read_text(encoding='utf-8', errors='replace')
    assert not (home / 'gateway.pid').exists()
    assert not (home / 'gateway.sock').exists()
    assert control_absent(home)
    assert not port_open(port)
    from gateway.runtime_ownership import exclusive_maintenance
    with exclusive_maintenance([home]):
        receipt['lock_released'] = True
    print(json.dumps(receipt))


def test_native_owner_hard_kill_recovers_queued_admission_once(harness):
    """Kill the owner (terminate_pid, never SIGKILL) with a committed queued admission held
    behind execution scheduling; every ordinary restart executes it exactly once."""
    home, env, peer = harness['home'], harness['env'], harness['peer']
    pids, epochs, sessions = [], [], {}

    async def before_crash(desc):
        async with websocket(home, desc) as ws:
            for name in ('blocked', 'safe'):
                sessions[name] = (await ok(ws, 'session.create', request_id=name, source='cli', cwd=str(home), toolsets=[]))['session_id']
            warm = await ok(ws, 'prompt.submit', session_id=sessions['safe'], input_id='warm', text='WARM')
            assert warm['status'] == 'queued'
            await asyncio.to_thread(wait_status, home, 'warm', 'terminal')
            await ok(ws, 'prompt.submit', session_id=sessions['blocked'], input_id='started', text='BLOCK_STARTED')
            assert await asyncio.to_thread(peer.blocked.wait, 30)
            await ok(ws, 'prompt.submit', session_id=sessions['blocked'], input_id='follower', text='NEVER_REPLAY')
            # The fixture daemon holds only scheduling for this text; the admission is committed.
            queued = await ok(ws, 'prompt.submit', session_id=sessions['safe'], input_id='queued', text='SAFE_QUEUE')
            assert queued['status'] == 'queued', queued
            await asyncio.to_thread(wait_status, home, 'queued', 'queued', 20)

    async def after_restart(desc):
        await asyncio.to_thread(wait_status, home, 'queued', 'terminal', 60)
        rows = admissions(home)
        assert rows['started'] == 'unknown' and rows['follower'] == 'queued', rows
        async with websocket(home, desc) as ws:
            resumed = await ok(ws, 'session.resume', session_id=sessions['safe'])
            history = json.dumps(resumed['messages'])
            assert 'NATIVE_ACK_WARM' in history and 'NATIVE_ACK_SAFE_QUEUE' in history, resumed
            paused = await ok(ws, 'session.resume', session_id=sessions['blocked'])
            assert any(p['status'] == 'unknown' for p in paused['pending']), paused

    with daemon(home, env, 'first.log', fixture='api_recovery_daemon.py') as (owner, desc, _):
        pids.append(owner.pid); epochs.append(desc['authority_epoch'])
        asyncio.run(before_crash(desc))
        hard_kill(owner)
        assert owner.poll() is not None
    peer.release.set()
    for name in ('restart-1.log', 'restart-2.log'):
        with daemon(home, {**env, 'UGW_STACK_DUMP_AFTER': '20'}, name, fixture='stack_dump_daemon.py') as (owner, desc, _):
            pids.append(owner.pid); epochs.append(desc['authority_epoch'])
            asyncio.run(after_restart(desc))
            assert planned_stop(home, owner, desc) == 0
    texts = [r['text'] for r in peer.requests]
    assert texts.count('SAFE_QUEUE') == 1 and texts.count('BLOCK_STARTED') == 1, texts
    assert 'NEVER_REPLAY' not in texts, texts
    assert epochs == sorted(set(epochs)) and len(set(pids)) == 3, (pids, epochs)
    print(json.dumps({'pids': pids, 'epochs': epochs, 'inference_texts': texts, 'rows': admissions(home)}))


def test_native_safe_mode_cli_executes_in_managed_worker(harness):
    """`hermes chat --safe-mode -q` over subprocess pipes: the turn runs in an out-of-process
    managed worker (pid != owner) that opens no writable canonical sqlite."""
    home, env, peer, url, audit = harness['home'], harness['env'], harness['peer'], harness['url'], harness['audit']

    def safe_mode_cli_worker(owner, desc, harness):
        # Popen.pid is the venv launcher on Windows; the daemon's own pid comes from identify.
        gateway_pid = desc['pid']
        command = [sys.executable, '-m', 'hermes_cli.main', 'chat', '--safe-mode', '--provider', 'custom',
                   '--base-url', url, '--model', 'safe-fixture', '--api-key', 'fixture', '-Q', '-q', 'SAFE_PROBE_NATIVE']
        proc = subprocess.Popen(command, cwd=harness['work'], env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = proc.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            proc.kill(); out, err = proc.communicate(timeout=10)
            raise AssertionError(('safe-mode CLI timed out', out, err))
        out, err = out.decode('utf-8', 'replace'), err.decode('utf-8', 'replace')
        starts = [r for r in records(audit) if r['kind'] == 'start' and 'agent.managed_worker' in ' '.join(r['argv'])]
        evidence = {'owner_pid': owner.pid, 'gateway_pid': gateway_pid, 'worker_starts': starts,
                    'worker_errors': [r for r in records(audit) if r['kind'] == 'worker_error'],
                    'worker_executions': query(home, 'SELECT execution_id,status FROM worker_executions'),
                    'gateway_log': (home / 'logs' / 'gateway.log').read_text(encoding='utf-8', errors='replace')[-4000:]}
        if proc.returncode != 0:
            # The assertion repr truncates the worker traceback; the artifact keeps it whole.
            artifacts = Path(os.environ.get('UGW_ARTIFACT_DIR', str(home.parent)))
            artifacts.mkdir(parents=True, exist_ok=True)
            (artifacts / 'safe-mode-evidence.json').write_text(json.dumps(evidence, indent=1), encoding='utf-8')
        assert proc.returncode == 0, (proc.returncode, out, err, json.dumps(evidence, indent=1))
        assert 'NATIVE_ACK_SAFE_PROBE_NATIVE' in out + err, (out, err)
        sid = re.search(r'Session: (\S+)', err).group(1)
        assert query(home, 'SELECT status FROM session_admissions WHERE target_session_id=?', (sid,)) == [('terminal',)]
        assert query(home, 'SELECT status FROM worker_executions WHERE session_id=?', (sid,)) == [('terminal',)]
        policy = json.loads(query(home, 'SELECT value FROM state_meta WHERE key=?', ('gateway.local_policy.v1:' + sid,))[0][0])
        assert policy['policy']['safe_mode'] is True and policy['policy']['ignore_user_config'] is True
        rows = records(audit)
        # The gateway is an ANCESTOR, not necessarily the parent: the uv venv python.exe trampoline
        # sits between the daemon's Popen and the real worker interpreter on Windows.
        workers = sorted({r['pid'] for r in rows if r['kind'] == 'start'
                          and gateway_pid in [r['ppid'], *(a[0] for a in r['ancestors'] if isinstance(a, list))]
                          and r['argv'][-2:] == ['-m', 'agent.managed_worker']})
        assert len(workers) == 1 and workers[0] != gateway_pid, [(r['pid'], r['ppid'], r['argv'][-3:]) for r in rows if r['kind'] == 'start']
        worker = workers[0]
        # Positive control: the witness is live in the gateway, so worker silence is real.
        assert [r for r in rows if r['kind'] == 'sqlite' and r['pid'] == gateway_pid and 'state.db' in r['target']]
        writable = [r['target'] for r in rows if r['kind'] == 'sqlite' and r['pid'] == worker
                    and 'state.db' in r['target'] and 'mode=ro' not in r['target']]
        assert writable == [], writable
        safe = [r for r in peer.requests if r['text'] == 'SAFE_PROBE_NATIVE' and not r['title_generation']]
        assert len(safe) == 1 and safe[0]['model'] == 'safe-fixture' and safe[0]['auth'] == 'Bearer fixture', peer.requests
        return {'worker_pid': worker, 'gateway_pid': gateway_pid, 'session': sid}

    with daemon(home, {**env, 'UGW_STACK_DUMP_AFTER': '20'}, 'worker.log', fixture='stack_dump_daemon.py') as (owner, desc, _):
        receipt = safe_mode_cli_worker(owner, desc, harness)
        assert planned_stop(home, owner, desc) == 0
    print(json.dumps(receipt))
