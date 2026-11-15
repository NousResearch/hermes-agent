"""Typed /approve and /deny on a LOCAL session resolve the same waiting tool call as approval.respond."""
import asyncio
import json
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
import signal
import threading
import time

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket
from tests.gateway.test_api_pending_daemon import PromptPeer


async def _pending_approval(ws, sid):
    deadline = time.monotonic() + 30
    while True:
        snap = (await rpc(ws, 'session.resume', session_id=sid))['result']
        prompt = next((p for p in snap.get('prompts', []) if p['kind'] == 'approval'), None)
        if prompt:
            return prompt
        assert time.monotonic() < deadline, ('no pending approval', snap)
        await asyncio.sleep(.05)


async def _settled(ws, sid):
    deadline = time.monotonic() + 30
    while True:
        snap = (await rpc(ws, 'session.resume', session_id=sid))['result']
        if not snap['running'] and not snap['pending']:
            return snap
        assert time.monotonic() < deadline, ('turn never settled', snap)
        await asyncio.sleep(.1)


@pytest.mark.platforms("linux")
def test_typed_approve_and_deny_resolve_the_waiter_approval_respond_answers(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user, work = tmp_path / 'state', tmp_path / 'user', tmp_path / 'work'
    home.mkdir(mode=0o700)
    user.mkdir()
    work.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), PromptPeer)
    peer.requests = []
    targets = {name: work / name for name in ('approved', 'denied', 'responded')}
    for target in targets.values():
        target.mkdir()
        (target / 'owned.txt').write_text('owned')
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False}, 'approvals': {'mode': 'manual'},
        'model': {'provider': 'custom', 'default': 'approve-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}}))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    out = {}

    async def probe(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='approve', source='cli', toolsets=['terminal'],
                                cwd=str(work))
            sid = created['result']['session_id']
            for name, answer in (('approved', '/approve'), ('denied', '/deny not this one'),
                                 ('responded', 'approval.respond')):
                peer.target = targets[name]
                await rpc(ws, 'prompt.submit', session_id=sid, input_id=name, text='ASK_APPROVAL')
                prompt = await _pending_approval(ws, sid)
                assert targets[name].exists()
                if answer == 'approval.respond':
                    reply = await rpc(ws, 'approval.respond', session_id=sid, prompt_id=prompt['prompt_id'],
                                      execution_generation=prompt['execution_generation'], choice='once')
                    out[name] = reply['result']['status']
                    # The typed command finds nothing left: the same waiter was answered.
                    late = await rpc(ws, 'slash.exec', session_id=sid, command='approve')
                else:
                    reply = await rpc(ws, 'slash.exec', session_id=sid, command=answer.lstrip('/'))
                    out[name] = reply['result']['output']
                    # The prompt id approval.respond would answer is already resolved.
                    late = await rpc(ws, 'approval.respond', session_id=sid, prompt_id=prompt['prompt_id'],
                                     execution_generation=prompt['execution_generation'], choice='once')
                out[name + ' late'] = late.get('result') or late.get('error')
                await _settled(ws, sid)
    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            asyncio.run(probe(desc))
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
    assert 'approved' in out['approved'].lower() and not targets['approved'].exists(), out
    assert 'denied' in out['denied'].lower() and targets['denied'].exists(), out
    assert out['responded'] == 'resolved' and not targets['responded'].exists(), out
    for name in ('approved', 'denied'):
        late = out[name + ' late']
        assert late.get('status') == 'already_resolved' or 'message' in late, (name, late)
    assert 'no pending' in str(out['responded late']).lower() or 'output' in out['responded late'], out
    tools = [m for req in peer.requests for m in req.get('messages', []) if m['role'] == 'tool']
    assert any('not this one' in str(m['content']) for m in tools), tools
    assert os.path.exists(home / 'config.yaml')
