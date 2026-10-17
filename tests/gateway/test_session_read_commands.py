"""Read/report slash commands run their real gateway handler for a LOCAL session (and only reads)."""
import asyncio
import json
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
import signal
import subprocess
import threading
import time

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket


def test_memory_review_verbs_are_local_session_controls():
    from gateway.session_commands import command_verdict

    assert command_verdict('memory', '') == 'read'
    assert command_verdict('memory', 'pending') == 'read'
    for args in ('approve entry-1', 'apply entry-1', 'reject all', 'deny entry-1', 'drop entry-1',
                 'approval on', 'mode off'):
        assert command_verdict('memory', args) == 'control'
    assert command_verdict('memory', 'search user') == 'refused'


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
            created = await rpc(ws, 'session.create', request_id='reads', source='cli', toolsets=['todo'], cwd=str(work))
            sid = created['result']['session_id']
            await rpc(ws, 'prompt.submit', session_id=sid, input_id='warm', text='WARM_READS')
            deadline = time.monotonic() + 45
            while True:
                snap = (await rpc(ws, 'session.resume', session_id=sid))['result']
                if not snap['running'] and not snap['pending'] and 'RECOVERY_ACK_WARM_READS' in json.dumps(
                        snap['messages']):
                    break
                assert time.monotonic() < deadline, ('warm turn never settled', snap)
                await asyncio.sleep(.1)
            for command in ('usage', 'insights', 'profile', 'topup', 'diff', 'memory', 'kanban list',
                            'suggestions', 'bundles', 'usage reset', 'memory approval on', 'kanban create x',
                            'suggestions catalog', 'agents', 'sessions', 'platform',
                            'toolsets', 'platforms', 'skills pending', 'skills reject nope', 'skills install x',
                            'memory reject nope'):
                try:
                    reply = await rpc(ws, 'slash.exec', session_id=sid, command=command)
                except TimeoutError:
                    # Name the command: an unnamed 20 s rpc timeout under load carried no evidence.
                    raise AssertionError(f'slash.exec {command!r} did not answer within the rpc bound') from None
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
    # Reports with no messaging handler: the classic CLI's text for this session's toolsets.
    assert '(*) todo ' in out['toolsets'] and '    web ' in out['toolsets'], out['toolsets']
    assert 'Telegram' in out['platforms'] and 'TELEGRAM_BOT_TOKEN' in out['platforms'], out['platforms']
    # The skill write-approval queue: the messaging handler's review verbs; the hub stays refused.
    assert 'No pending' in out['skills pending'] or 'pending' in out['skills pending'].lower(), out['skills pending']
    assert out['skills reject nope'] != 'unsupported_command', out['skills reject nope']
    assert out['skills install x'] == 'unsupported_command', out['skills install x']
    assert out['memory reject nope'] != 'unsupported_command', out['memory reject nope']
    # Write forms, messaging-model state and gateway-wide views stay refused (the per-session settings
    # run: tests/gateway/test_session_setting_commands.py).
    for command in ('usage reset', 'memory approval on', 'kanban create x', 'suggestions catalog',
                    'agents', 'sessions', 'platform'):
        assert out[command] == 'unsupported_command', (command, out[command])
    assert (home / 'config.yaml').read_text() == config
    assert not os.path.exists(home / 'cron' / 'jobs.json') or 'catalog' not in (home / 'cron' / 'jobs.json').read_text()
