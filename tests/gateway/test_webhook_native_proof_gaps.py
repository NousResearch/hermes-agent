"""Real child execution, typed raw process and SIGKILL preparation boundaries.

The model is a loopback wire fixture; all origins, admissions, results and native
receipts are produced by the actual gateway. No ledger or receipt is seeded.
"""
import asyncio
from contextlib import closing, contextmanager
from http.server import ThreadingHTTPServer
import hashlib
import hmac
import json
from pathlib import Path
import socket
import sqlite3
import threading

import aiohttp
import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon
from tests.gateway.test_webhook_bot_chat_integration import WebhookModel, rows
from tests.waiting import async_wait_until


class ProofModel(WebhookModel):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if body.get('messages'):
            self.server.requests.append(body)
        messages = body.get('messages', [])
        text = next((m.get('content', '') for m in reversed(messages) if m['role'] == 'user'), '')
        tools = [m for m in messages if m['role'] == 'tool']
        finish = 'stop'
        message = {'role': 'assistant', 'content': 'RECEIVER_ACK'}
        if 'Webhook automation' not in str(text):
            child = next((tag for tag in ('CHILD_1_A', 'CHILD_1_B', 'CHILD_2_A', 'CHILD_2_B') if tag in str(text)), None)
            if child and not str(text).startswith('['):
                wave = int(child[6])
                self.server.started[wave - 1].add(child)
                assert self.server.release[wave - 1].wait(90), 'test did not release real child inference'
                message['content'] = child + '_ACTUAL_RESULT'
            elif text == 'RAW_SOURCE':
                if tools:
                    message['content'] = 'RAW_SOURCE_FINAL'
                else:
                    command = 'printf RAW_PROGRESS; sleep 2; printf RAW_FINAL'
                    if getattr(self.server, 'fail_raw', False):
                        command += '; exit 4'
                    message = tool_call('terminal', {'command': command,
                        'background': True, 'notify': True}, 'raw-process')
                    finish = 'tool_calls'
            elif text == 'GROUP_SOURCE' or str(text).startswith('[ASYNC DELEGATION'):
                wave = 1 if text == 'GROUP_SOURCE' else 2
                # A real grouped batch has two actual child conversations, not
                # two constructed automation admissions that pretend to be children.
                if wave == 2 and 'CHILD_2_A_ACTUAL_RESULT' in str(text):
                    message['content'] = 'GROUP_FINAL_2'
                elif (wave == 1 and tools) or (wave == 2 and tools and any('CHILD_2_A' in str(m) for m in messages if m['role'] == 'assistant')):
                    message['content'] = 'GROUP_SOURCE_FINAL' if wave == 1 else 'GROUP_FINAL_1'
                else:
                    message = tool_call('delegate_task', {'tasks': [
                        {'goal': f'CHILD_{wave}_A return the requested fixture result.', 'group': f'wave-{wave}'},
                        {'goal': f'CHILD_{wave}_B return the requested fixture result.', 'group': f'wave-{wave}'}],
                        'background': True}, f'children-{wave}')
                    finish = 'tool_calls'
        payload = json.dumps({'id': 'local', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode()
        kind = 'application/json'
        if body.get('stream'):
            if message.get('tool_calls'):
                message['tool_calls'][0]['index'] = 0
            payload = ('data: ' + json.dumps({'id': 'local', 'choices': [{'index': 0, 'delta': message,
                'finish_reason': finish}]}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def tool_call(name, arguments, identity):
    return {'role': 'assistant', 'content': None, 'tool_calls': [{'id': identity, 'type': 'function',
        'function': {'name': name, 'arguments': json.dumps(arguments)}}]}


def ledger(home):
    with closing(sqlite3.connect(home / 'state.db')) as db:
        return [json.loads(row[0]) for row in db.execute(
            "SELECT value FROM state_meta WHERE key LIKE 'webhook.final.intent.v1.%' OR key LIKE 'webhook.origin.v1.%'")]


@contextmanager
def runtime(tmp_path, monkeypatch, routes, *, raw=False):
    root = Path(__file__).resolve().parents[2]
    home, user, scratch = (tmp_path / name for name in ('state', 'user', 'runtime'))
    for directory in (home, user, scratch):
        directory.mkdir(mode=0o700, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOME', str(user))
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('XDG_RUNTIME_DIR', '/proc/self/cwd/runtime')
    peer = ThreadingHTTPServer(('127.0.0.1', 0), ProofModel)
    peer.requests, peer.started = [], [set(), set()]
    peer.release = [threading.Event(), threading.Event()]
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{peer.server_port}/v1'
    config = {'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': base},
        'gateway': {'standalone': True}, 'platforms': {'webhook': {'enabled': True, 'extra': {
            'host': '127.0.0.1', 'port': port, 'secret': 'owned-secret', 'routes': routes,
            'script_timeout_seconds': 90}}}, 'auxiliary': {'title_generation': {'enabled': False}},
        'terminal': {'cwd': str(home)}, 'display': {'background_process_notifications': 'all'},
        'streaming': {'enabled': False}, 'delegation': {'max_children': 2},
        'platform_toolsets': {'gui': [], 'webhook': ['terminal', 'delegation']}}
    (home / 'config.yaml').write_text(json.dumps(config))
    env = child_env() | {'HOME': str(user), 'USERPROFILE': str(user), 'HERMES_HOME': str(home),
        'PYTHONPATH': str(root), 'OPENAI_API_KEY': 'loopback-only', 'OPENAI_BASE_URL': base,
        'PYTHONUNBUFFERED': '1', 'XDG_RUNTIME_DIR': '/proc/self/cwd/runtime',
        'TEST_BARRIER': str(home / 'barrier'), 'TEST_HOOK_EFFECT': str(home / 'hook-effect'),
        'TEST_RAW_WATCHER': '1' if raw else '0'}
    try:
        yield root, home, env, port, peer
    finally:
        for event in peer.release:
            event.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)


async def post(port, route, identity, text):
    body = json.dumps({'text': text}).encode()
    headers = {'X-GitHub-Delivery': identity, 'X-Hub-Signature-256':
        'sha256=' + hmac.new(b'owned-secret', body, hashlib.sha256).hexdigest()}
    async with aiohttp.ClientSession() as client:
        async with client.post(f'http://127.0.0.1:{port}/webhooks/{route}', data=body, headers=headers) as response:
            return response.status, await response.json()


@pytest.mark.platforms('linux')
def test_real_grouped_children_transitive_finals_and_restart_recovery(tmp_path, monkeypatch, record_property):
    from tools.bot_live_delivery import read_delivery_result
    routes = {'group': {'prompt': '{text}', 'deliver': 'bot-chat'}}
    with runtime(tmp_path, monkeypatch, routes) as (root, home, env, port, peer):
        async def scenario(proc):
            assert (await post(port, 'group', 'group-request', 'GROUP_SOURCE'))[0] == 202
            await async_wait_until(lambda: any(r['request_id'] == 'group-request' and r['status'] == 'terminal' for r in rows(home)),
                timeout=60, what='real source finished without waiting for grouped children')
            original, = [r for r in rows(home) if r['request_id'] == 'group-request']
            native = json.loads(original['payload_json'])['native_text_v1']
            def ended():
                with closing(sqlite3.connect(home / 'state.db')) as db:
                    return db.execute('SELECT ended_at FROM sessions WHERE id=?', (original['target_session_id'],)).fetchone()[0]
            await async_wait_until(ended, timeout=20, what='one-shot source finalized')
            for wave in (1, 2):
                await async_wait_until(lambda: len(peer.started[wave - 1]) == 2, timeout=60, what=f'two real children wave {wave}')
                peer.release[wave - 1].set()
                await async_wait_until(lambda: len([r for r in rows(home) if json.loads(r['payload_json']).get('native_text_v1', {}).get('webhook_root')]) >= wave,
                    timeout=60, what=f'actual grouped completion admission {wave}')
            await async_wait_until(lambda: (home / 'barrier').exists(), timeout=60, what='second completion committed before native handoff')
            completions = [r for r in rows(home) if json.loads(r['payload_json']).get('native_text_v1', {}).get('webhook_root')]
            assert len(completions) == 2
            for completion in completions:
                value = json.loads(completion['payload_json'])['native_text_v1']
                assert value['webhook_root'] == original['admission_id']
                assert value['webhook_target'] == native['webhook_target']
                assert json.loads(value['automation']['identity'])[0] == 'async_delegation'
                assert completion['target_session_id'] == original['target_session_id'] and completion['outcome'] == 'completed'
            pending = (home / 'barrier').read_text()
            assert read_delivery_result(home, pending) is None
            proc.kill()
            await asyncio.to_thread(proc.wait, 10)
            return original, completions, pending
        with daemon(tmp_path, home, env, barrier=True, fixture=str(root / 'tests/gateway/fixtures/webhook_proof_daemon.py')) as (proc, _):
            try:
                original, completions, pending = asyncio.run(scenario(proc))
            except BaseException:
                print('GATEWAY', (home / 'first.log').read_text())
                print('REQUESTS', json.dumps(peer.requests))
                raise
        before = rows(home)
        async def recovered():
            await async_wait_until(lambda: (read_delivery_result(home, pending) or {}).get('status') == 'settled',
                timeout=60, what='committed grouped final recovered to actual native settlement')
            assert [r for r in rows(home) if json.loads(r['payload_json']).get('native_text_v1')] == [
                r for r in before if json.loads(r['payload_json']).get('native_text_v1')]
            finals = [v for v in ledger(home) if v['origin_kind'] in {'agent_final', 'completion_final'}]
            assert len(finals) == 3 and len({v['delivery_id'] for v in finals}) == 3
            receipts = [read_delivery_result(home, v['delivery_id']) for v in finals]
            assert all(r and r['status'] == 'settled' and r['reply'] == 'RECEIVER_ACK' for r in receipts)
            assert len({r['session_id'] for r in receipts}) == 1
            record_property('native_proof', json.dumps({'root': original, 'completions': completions, 'receipts': receipts}))
        with daemon(tmp_path, home, env, barrier=False):
            asyncio.run(recovered())
        for tag in ('CHILD_1_A', 'CHILD_1_B', 'CHILD_2_A', 'CHILD_2_B'):
            assert sum(tag in str(next((m.get('content', '') for m in reversed(r.get('messages', [])) if m['role'] == 'user'), ''))
                and 'ACTUAL_RESULT' not in str(r) for r in peer.requests) == 1


@pytest.mark.platforms('linux')
def test_raw_terminal_notice_reaches_native_door_without_completion_agent(tmp_path, monkeypatch, record_property):
    from tools.bot_live_delivery import read_delivery_result
    with runtime(tmp_path, monkeypatch, {'raw': {'prompt': '{text}', 'deliver': 'bot-chat'}}, raw=True) as (root, home, env, port, peer):
        async def scenario():
            assert (await post(port, 'raw', 'raw-request', 'RAW_SOURCE'))[0] == 202
            await async_wait_until(lambda: any(v['origin_kind'] == 'process_final' for v in ledger(home)), timeout=50, what='typed real raw process origin')
            value, = [v for v in ledger(home) if v['origin_kind'] == 'process_final']
            await async_wait_until(lambda: (read_delivery_result(home, value['delivery_id']) or {}).get('status') == 'settled',
                timeout=40, what='raw process native settlement')
            receipt = read_delivery_result(home, value['delivery_id'])
            assert 'RAW_FINAL' in receipt['message'] and receipt['reply'] == 'RECEIVER_ACK'
            assert value['process_id'] and value['process_started_at'] and value['source_outcome'] == 'completed'
            assert not any(json.loads(r['payload_json']).get('native_text_v1', {}).get('automation') for r in rows(home))
            assert len([v for v in ledger(home) if v['origin_kind'] == 'process_final']) == 1
            inputs = [next((m.get('content', '') for m in reversed(r.get('messages', [])) if m['role'] == 'user'), '') for r in peer.requests]
            assert len([text for text in inputs if 'Webhook automation' in str(text)]) == 2
            record_property('native_proof', json.dumps({'origin': value, 'receipt': receipt, 'inputs': inputs}))
        with daemon(tmp_path, home, env, barrier=True, fixture=str(root / 'tests/gateway/fixtures/webhook_proof_daemon.py')):
            try:
                asyncio.run(scenario())
            except BaseException:
                print((home / 'first.log').read_text())
                raise


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('kind', ['direct', 'hook'])
def test_sigkill_during_real_side_effect_preparation_does_not_reinvoke(tmp_path, monkeypatch, record_property, kind):
    scripts = tmp_path / 'state' / 'scripts'
    scripts.mkdir(parents=True)
    script = scripts / 'side-effect.py'
    script.write_text("from pathlib import Path\nimport os, signal\np = Path(__file__).with_name('script-effect')\nwith p.open('a') as f: f.write(str(os.getpid()) + '\\n')\nos.kill(os.getpid(), signal.SIGSTOP)\nprint('{\"text\": \"NEVER_COMMITTED_SCRIPT_REPLY\"}')\n")
    route = {'prompt': '{text}', 'deliver': 'bot-chat', **({'deliver_only': True, 'script': str(script)} if kind == 'direct' else {})}
    with runtime(tmp_path, monkeypatch, {kind: route}) as (root, home, env, port, peer):
        effect = scripts / 'script-effect' if kind == 'direct' else home / 'hook-effect'
        async def killed(proc):
            request = asyncio.create_task(post(port, kind, 'interrupted-request', 'HOOK_BLOCK' if kind == 'hook' else 'DIRECT_BLOCK'))
            await async_wait_until(effect.exists, timeout=30, what='actual side effect began after durable preparation')
            value, = ledger(home)
            assert value['state'] == 'preparing' and not rows(home)
            if kind == 'direct':
                import os
                import signal
                proc.send_signal(signal.SIGSTOP)
                os.kill(int(effect.read_text().strip()), signal.SIGKILL)
            proc.kill()
            await asyncio.to_thread(proc.wait, 10)
            try:
                await request
            except aiohttp.ClientError:
                pass
            return value
        with daemon(tmp_path, home, env, barrier=True, fixture=str(root / 'tests/gateway/fixtures/webhook_proof_daemon.py')) as (proc, _):
            saved = asyncio.run(killed(proc))
        async def retry():
            status, value = await post(port, kind, 'interrupted-request', 'HOOK_BLOCK' if kind == 'hook' else 'DIRECT_BLOCK')
            assert status == 202 and value['status'] == 'origin_unknown'
            assert value['origin_id'] == saved['origin_id'] and value['delivery_id'] == saved['delivery_id']
            stored, = ledger(home)
            assert stored['reason'] == 'preparation_interrupted' and stored['message'] == ''
            assert effect.read_text().count('\n') == 1 and not rows(home) and not peer.requests
            from tools.bot_live_delivery import read_delivery_result
            assert read_delivery_result(home, saved['delivery_id']) is None
            record_property('native_proof', json.dumps({'before': saved, 'after': stored, 'producer_sigkill': True}))
        # Keep the actual consumer installed: a faulty replay must really invoke it again.
        with daemon(tmp_path, home, env, barrier=True, fixture=str(root / 'tests/gateway/fixtures/webhook_proof_daemon.py')):
            asyncio.run(retry())


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('stage', ['pin', 'admitted'])
def test_native_owner_death_at_receipt_boundary_retains_exact_identity(tmp_path, monkeypatch, record_property, stage):
    from tools.bot_live_delivery import read_delivery_result
    route = {'direct': {'prompt': '{text}', 'deliver': 'bot-chat:receiver', 'deliver_only': True}}
    with runtime(tmp_path, monkeypatch, route) as (root, home, env, port, peer):
        target = home / 'profiles' / 'receiver'
        target.mkdir(parents=True)
        config = json.loads((home / 'config.yaml').read_text())
        config.pop('platforms')
        (target / 'config.yaml').write_text(json.dumps(config))
        target_env = env | {'HERMES_HOME': str(target), 'TEST_RECEIPT_STAGE': stage}
        fixture = str(root / 'tests/gateway/fixtures/webhook_proof_daemon.py')
        async def crash(proc):
            request = asyncio.create_task(post(port, 'direct', 'receipt-boundary', 'ACTUAL_DIRECT_FINAL'))
            await async_wait_until(lambda: (home / 'barrier').exists(), timeout=30, what='physical native receipt boundary')
            key = (home / 'barrier').read_text()
            pin = read_delivery_result(target, key)
            assert pin and pin['status'] == 'ambiguous' and not pin.get('admission_id')
            admitted = [r for r in rows(target) if r['request_id'] == 'bot:' + key]
            assert len(admitted) == (1 if stage == 'admitted' else 0)
            proc.kill()
            await asyncio.to_thread(proc.wait, 10)
            status, value = await request
            assert status == 202 and value['delivery_id'] == key
            # The live producer observes the actual pin before its owner is
            # restarted and may withdraw it. This is not seeded retry state.
            status, value = await post(port, 'direct', 'receipt-boundary', 'ACTUAL_DIRECT_FINAL')
            assert status == 202 and value['status'] == 'ambiguous'
            assert ledger(home)[0]['pre_admission'] is True
            return key, admitted
        async def recover():
            status, value = await post(port, 'direct', 'receipt-boundary', 'ACTUAL_DIRECT_FINAL')
            assert status == 202 and value['delivery_id'] == key
            if stage == 'pin':
                await async_wait_until(lambda: (read_delivery_result(target, key) or {}).get('status') == 'settled',
                    timeout=100, what='withdrawn pin same-ID delivery')
            else:
                # Native policy keeps the post-commit two-store pin explicitly
                # unknown, rather than pretending the ACK was published.
                assert value['status'] == 'ambiguous'
                assert read_delivery_result(target, key)['status'] == 'ambiguous'
            admitted = [r for r in rows(target) if r['request_id'] == 'bot:' + key]
            assert len(admitted) == 1
            if before:
                assert admitted[0]['admission_id'] == before[0]['admission_id']
            receipt = read_delivery_result(target, key)
            assert receipt and receipt['delivery_id'] == key
            assert len(peer.requests) <= 1, 'receipt uncertainty must not rerun recipient inference'
            assert not rows(home)
            record_property('native_proof', json.dumps({'stage': stage, 'before': before,
                'after': admitted, 'receipt': receipt, 'origin': value}))
        with daemon(tmp_path, home, env, barrier=False) as (source, _):
            with daemon(tmp_path, target, target_env, barrier=True, fixture=fixture) as (proc, _):
                key, before = asyncio.run(crash(proc))
            with daemon(tmp_path, target, target_env, barrier=True, fixture=fixture):
                try:
                    asyncio.run(recover())
                    assert source.poll() is None
                except BaseException:
                    print('RECEIPT_RECOVERY_LOG', (target / 'first.log').read_text())
                    print('LEDGER', ledger(home), 'RECEIPT', read_delivery_result(target, key), 'ROWS', rows(target))
                    raise


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('source_muted', [False, True])
def test_raw_diagnostic_policy_is_profile_bound_and_survives_restart(tmp_path, monkeypatch, record_property, source_muted):
    from tools.bot_live_delivery import read_delivery_result
    route = {'raw': {'prompt': '{text}', 'deliver': 'bot-chat:receiver'}}
    with runtime(tmp_path, monkeypatch, route, raw=True) as (root, home, env, port, peer):
        peer.fail_raw = True
        config = json.loads((home / 'config.yaml').read_text())
        config['display']['suppress_warning_notifications'] = source_muted
        (home / 'config.yaml').write_text(json.dumps(config))
        target = home / 'profiles' / 'receiver'
        target.mkdir(parents=True)
        target_config = dict(config)
        target_config.pop('platforms')
        target_config['display'] = dict(config['display'], suppress_warning_notifications=not source_muted)
        (target / 'config.yaml').write_text(json.dumps(target_config))
        fixture = str(root / 'tests/gateway/fixtures/webhook_proof_daemon.py')
        async def scenario():
            assert (await post(port, 'raw', 'diagnostic-policy', 'RAW_SOURCE'))[0] == 202
            await async_wait_until(lambda: any(v['origin_kind'] == 'process_final' for v in ledger(home)),
                timeout=50, what='real failed process policy disposition')
            value, = [v for v in ledger(home) if v['origin_kind'] == 'process_final']
            assert value['source_outcome'] == 'failed' and value['category'] == 'diagnostic'
            key = value['delivery_id']
            if source_muted:
                assert value['state'] == 'suppressed' and value['message'] == ''
                assert read_delivery_result(target, key) is None
            else:
                await async_wait_until(lambda: (read_delivery_result(target, key) or {}).get('status') == 'settled',
                    timeout=40, what='diagnostic reaches differently configured native receiver')
                receipt = read_delivery_result(target, key)
                assert receipt['notification_category'] == 'diagnostic'
                native, = [r for r in rows(target) if r['admission_id'] == receipt['admission_id']]
                descriptor = json.loads(native['payload_json'])['local_automation_v1']
                assert descriptor['notification_category'] == 'diagnostic' and native['outcome'] == 'completed'
            # The ordinary original final remains a result even when warnings are muted.
            root_intent, = [v for v in ledger(home) if v['origin_kind'] == 'agent_final']
            await async_wait_until(lambda: (read_delivery_result(target, root_intent['delivery_id']) or {}).get('status') == 'settled',
                timeout=40, what='ordinary source final is not warning-suppressed')
            record_property('native_proof', json.dumps({'source_muted': source_muted, 'recipient_muted': not source_muted,
                'process_origin': value, 'receipt': read_delivery_result(target, key)}))
            return key
        with daemon(tmp_path, target, env | {'HERMES_HOME': str(target)}, barrier=False):
            with daemon(tmp_path, home, env, barrier=True, fixture=fixture):
                try:
                    key = asyncio.run(scenario())
                except BaseException:
                    print('POLICY_LOG', (home / 'first.log').read_text())
                    raise
            prior = rows(target)
            requests = len(peer.requests)
            config['display']['suppress_warning_notifications'] = not source_muted
            (home / 'config.yaml').write_text(json.dumps(config))
            with daemon(tmp_path, home, env, barrier=False):
                value, = [v for v in ledger(home) if v['delivery_id'] == key]
                assert value['source_outcome'] == 'failed' and value['category'] == 'diagnostic'
                if source_muted:
                    assert value['state'] == 'suppressed' and read_delivery_result(target, key) is None
                assert rows(target) == prior and len(peer.requests) == requests
