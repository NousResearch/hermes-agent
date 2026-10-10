"""A settled native reply survives an owner death before it reached the delivery ledger; an
uncommitted one is never delivered. Real Telegram polling ingress, SIGKILL, ordinary restart."""
from contextlib import closing, contextmanager
from http.server import ThreadingHTTPServer
import importlib.machinery
import json
from pathlib import Path
import queue
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon
from tests.gateway.test_native_telegram_startup_recovery import BotAPI, wait_for

pytestmark = [
    pytest.mark.skipif(importlib.machinery.PathFinder.find_spec("telegram") is None,
                       reason="python-telegram-bot not installed (on-demand extra)"),
    pytest.mark.platforms("linux"),
]


@contextmanager
def _owner(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    model = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    model.requests, model.blocked, model.release = [], threading.Event(), threading.Event()
    bot = ThreadingHTTPServer(('127.0.0.1', 0), BotAPI)
    bot.calls, bot.sent, bot.updates = [], [], queue.Queue()
    for peer in (model, bot):
        threading.Thread(target=peer.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{model.server_port}/v1'
    cfg = {'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': base},
           'gateway': {'multiplex_profiles': False},
           'platforms': {'telegram': {'enabled': True, 'extra': {
               'base_url': f'http://127.0.0.1:{bot.server_port}/bot', 'dm_policy': 'allowlist'}}},
           'auxiliary': {'title_generation': {'enabled': False}},
           'platform_toolsets': {'telegram': []}, 'terminal': {'cwd': str(home)}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, PYTHONUNBUFFERED='1',
               TELEGRAM_BOT_TOKEN='987654321:owned-loopback-fixture', TELEGRAM_ALLOWED_USERS='202')

    def status():
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return {json.loads(payload)['text']: state for state, payload in db.execute(
                'SELECT status,payload_json FROM session_admissions')}

    def replies(marker):
        # MarkdownV2 escapes ``_``: compare the unescaped text.
        return [body for body in bot.sent if marker in str(body.get('text', '')).replace('\\', '')]

    def diagnostic():
        return repr((status(), bot.sent)) + '\n' + '\n'.join(p.read_text() for p in (home / 'logs').glob('*.log'))

    def send(update_id, text):
        bot.updates.put({'update_id': update_id, 'message': {
            'message_id': update_id, 'date': int(time.time()),
            'chat': {'id': 202, 'type': 'private', 'first_name': 'Owned'},
            'from': {'id': 202, 'is_bot': False, 'first_name': 'Owned'}, 'text': text}})

    def crash_then_restart(marker):
        """Warm the route, send *marker*, let the barrier SIGKILL the owner, restart it plainly.
        Returns (status at the crash, deliveries of its answer, model calls for it)."""
        with daemon(root, home, env, barrier=True, fixture='native_reply_crash_daemon.py') as (proc, _):
            send(1, 'WARM_HISTORY')
            wait_for(lambda: replies('RECOVERY_ACK_WARM_HISTORY'), diagnostic)
            send(2, marker)
            assert proc.wait(timeout=60) == -9, diagnostic()
        assert not replies(marker)
        crashed = status()[marker]
        with daemon(root, home, env, barrier=False):
            wait_for(lambda: status().get(marker) != 'started', diagnostic)
            time.sleep(3)  # a (second) delivery would arrive inside this window
        prompts = [next(m['content'] for m in reversed(r['messages']) if m['role'] == 'user')
                   for r in model.requests]
        return crashed, len(replies('RECOVERY_ACK_' + marker)), sum(marker in p for p in prompts)

    try:
        yield SimpleNamespace(crash_then_restart=crash_then_restart, status=status, diagnostic=diagnostic)
    finally:
        for peer in (model, bot):
            peer.shutdown()
            peer.server_close()


def test_terminal_native_reply_is_delivered_once_after_owner_death_without_new_inference(tmp_path):
    with _owner(tmp_path) as probe:
        crashed, delivered, inferences = probe.crash_then_restart('KILL_BEFORE_LEDGER')
        assert crashed == 'terminal', probe.diagnostic()
        assert (delivered, inferences) == (1, 1), probe.diagnostic()


def test_reply_of_a_turn_lost_before_its_terminal_commit_is_never_delivered(tmp_path):
    with _owner(tmp_path) as probe:
        crashed, delivered, inferences = probe.crash_then_restart('KILL_BEFORE_COMMIT')
        assert crashed == 'started' and probe.status()['KILL_BEFORE_COMMIT'] == 'unknown', probe.diagnostic()
        assert (delivered, inferences) == (0, 1), probe.diagnostic()
