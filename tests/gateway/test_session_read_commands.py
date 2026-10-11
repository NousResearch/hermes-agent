"""Read/report slash commands run their real gateway handler for a LOCAL session (and only reads)."""
import asyncio
import json
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
import signal
import subprocess
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket


@pytest.mark.platforms("linux")
def test_local_session_reads_report_this_session_and_refuse_their_write_forms(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user, work = tmp_path / 'state', tmp_path / 'user', tmp_path / 'work'
    home.mkdir(mode=0o700)
    user.mkdir()
    work.mkdir()
    # The session's own checkout: /diff must read THIS cwd, not the gateway's directory.
    subprocess.run(['git', 'init', '-q', str(work)], check=True)
    (work / 'SESSION_CWD_MARKER.txt').write_text('untracked in the session checkout\n')
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    config = json.dumps({'gateway': {'multiplex_profiles': False},
                         'model': {'provider': 'custom', 'default': 'reads-model', 'base_url': url},
                         'auxiliary': {'title_generation': {'enabled': False}}})
    (home / 'config.yaml').write_text(config)
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    out = {}

    async def probe(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='reads', source='cli', toolsets=[], cwd=str(work))
            sid = created['result']['session_id']
            await rpc(ws, 'prompt.submit', session_id=sid, input_id='warm', text='WARM_READS')
            async with asyncio.timeout(45):
                while True:
                    snap = (await rpc(ws, 'session.resume', session_id=sid))['result']
                    if not snap['running'] and not snap['pending'] and 'RECOVERY_ACK_WARM_READS' in json.dumps(
                            snap['messages']):
                        break
                    await asyncio.sleep(.1)
            for command in ('usage', 'insights', 'profile', 'topup', 'diff', 'memory', 'kanban list',
                            'suggestions', 'bundles', 'usage reset', 'memory approval on', 'kanban create x',
                            'suggestions catalog', 'debug', 'agents', 'sessions', 'platform', 'fast'):
                reply = await rpc(ws, 'slash.exec', session_id=sid, command=command)
                out[command] = reply.get('result', {}).get('output') or reply.get('error', {}).get('message')
            dispatched = await rpc(ws, 'command.dispatch', session_id=sid, name='profile', arg='')
            out['dispatch profile'] = dispatched['result']['output']
    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            asyncio.run(probe(desc))
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
    # Session/account reports: this session's live agent and this profile's store.
    assert 'reads-model' in out['usage'] and 'API calls: 1' in out['usage'], out['usage']
    assert '**Sessions:** 1' in out['insights'] and '**Messages:** 2' in out['insights'], out['insights']
    assert str(home) in out['profile'] and out['dispatch profile'] == out['profile'], out['profile']
    assert 'Nous' in out['topup'], out['topup']
    # Workspace report: the session's frozen cwd.
    assert 'SESSION_CWD_MARKER.txt' in out['diff'], out['diff']
    # Review queues and boards, read-only views.
    assert 'No pending memory writes' in out['memory'], out['memory']
    assert out['kanban list'] == '(no matching tasks)', out['kanban list']
    assert 'No suggested automations' in out['suggestions'], out['suggestions']
    assert str(home / 'skill-bundles') in out['bundles'], out['bundles']
    # Write forms, messaging-model state, gateway-wide views and toggles stay refused.
    for command in ('usage reset', 'memory approval on', 'kanban create x', 'suggestions catalog', 'debug',
                    'agents', 'sessions', 'platform', 'fast'):
        assert out[command] == 'unsupported_command', (command, out[command])
    assert (home / 'config.yaml').read_text() == config
    assert not os.path.exists(home / 'cron' / 'jobs.json') or 'catalog' not in (home / 'cron' / 'jobs.json').read_text()
