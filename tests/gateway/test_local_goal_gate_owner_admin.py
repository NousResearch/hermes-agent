"""``/goal gate add`` on a local session: its owner counts as the admin (T7).

The messaging handler requires an explicitly configured gateway admin before a goal gate (a shell
command run with no later approval) is added. A local session has no messaging admin list; the
user who owns it is the one typing in their own CLI/TUI/Desktop, and ``slash.exec`` has already
proven that (``authorize_local_source`` plus the ``session:control`` capability). Messaging keeps
admin-only.
"""
import asyncio
import json
import signal
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket


@pytest.mark.platforms("linux")  # the daemon is stopped with SIGINT, which raises on Windows
def test_local_session_owner_adds_a_goal_gate(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user, work = tmp_path / 'state', tmp_path / 'user', tmp_path / 'work'
    home.mkdir(mode=0o700)
    user.mkdir()
    work.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'gate-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}},
    }))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    out = {}

    async def probe(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='gate', source='cli', toolsets=[], cwd=str(work))
            sid = created['result']['session_id']
            for command in ('goal ship the cache', 'goal gate add true', 'goal gate list'):
                reply = await rpc(ws, 'slash.exec', session_id=sid, command=command)
                out[command] = reply.get('result') or reply.get('error')

    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            asyncio.run(probe(desc))
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)

    added = out['goal gate add true']
    assert 'admin' not in added.get('output', '').lower(), ('local owner refused as non-admin', out)
    assert 'Gate added' in added['output'], out
    assert '$ true' in out['goal gate list']['output'], out


@pytest.mark.asyncio
async def test_messaging_goal_gate_add_stays_admin_only():
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    runner = object.__new__(GatewayRunner)

    class _Mgr:
        session_id = 's1'
        added = []

        def add_gate(self, command):
            self.added.append(command)
            raise AssertionError('gate added for a non-admin messaging sender')

    mgr = _Mgr()

    async def _mgr_for(_event):
        return mgr, None

    async def _inline(fn, *args):
        return fn(*args)

    runner._get_goal_manager_for_event = _mgr_for
    runner._run_in_executor_with_context = _inline
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='c', user_id='u', chat_type='dm')
    with patch('hermes_cli.goals.last_user_message_from_db', lambda _sid: None):
        out = await runner._handle_goal_command(MessageEvent(text='/goal gate add true', source=source))
    assert mgr.added == []
    assert 'admin' in out.lower()
