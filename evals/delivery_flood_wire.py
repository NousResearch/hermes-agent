"""Local-wire delivery ledger A/B, not a real Telegram bot or model run.

Run: python evals/delivery_flood_wire.py OUTPUT_DIR
Uses the real Telegram SDK, adapter final-send path, runner and SQLite stores.
Only the Telegram HTTP endpoint is replaced; waits and process ownership are real.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Wire:
    def __init__(self):
        self.calls = []
        self.deadlines = {}
        wire = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
                data = {k: v[0] for k, v in parse_qs(raw.decode()).items()}
                method = self.path.rsplit('/', 1)[-1].lower()
                status = 200
                result = True
                if method == 'getme':
                    result = {'id': 123456, 'is_bot': True, 'first_name': 'Fixture', 'username': 'fixture_bot'}
                elif method == 'sendmessage':
                    now = time.time()
                    chat = data['chat_id']
                    if chat in {'101', '201'} and chat not in wire.deadlines:
                        wire.deadlines[chat] = now + 61
                    if chat in wire.deadlines and now < wire.deadlines[chat]:
                        status = 429
                        result = {'ok': False, 'error_code': 429, 'description': 'Too Many Requests',
                                  'parameters': {'retry_after': math.ceil(wire.deadlines[chat] - now)}}
                    elif chat == '403':
                        status = 403
                        result = {'ok': False, 'error_code': 403, 'description': 'Forbidden: bot was blocked by the user'}
                    else:
                        result = {'message_id': len(wire.calls) + 1, 'date': int(now),
                                  'chat': {'id': int(chat), 'type': 'private'}, 'text': data.get('text', '')}
                    wire.calls.append({'at': now, 'method': method, 'status': status, **data})
                body = json.dumps(result if status != 200 else {'ok': True, 'result': result}).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/bot'

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


async def setup(url):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.run import GatewayRunner
    from gateway.session import SessionStore
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from telegram import Bot
    runner = object.__new__(GatewayRunner)
    runner.session_store = SessionStore(Path(os.environ['HERMES_HOME']) / 'sessions', GatewayConfig())
    runner._profile_adapters = {}
    runner._running = True
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token='123456:fixture', extra={}))
    adapter._bot = Bot('123456:fixture', base_url=url)
    await adapter._bot.initialize()
    runner.adapters = {Platform.TELEGRAM: adapter}
    adapter.gateway_runner = runner
    return runner, adapter


async def produce(runner, adapter, chat):
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=chat, thread_id='77')
    entry = await runner.async_session_store.get_or_create_session(source)
    await runner.async_session_store.mark_resume_pending(entry.session_key)
    event = MessageEvent(text='deliver a final answer', source=source, message_id='10')
    results = []
    await adapter._send_final_text(event, entry.session_key, 'LEDGER_' + chat, {'thread_id': '77'},
                                   False, 0, results.append)
    return entry.session_key


def rows():
    with sqlite3.connect(Path(os.environ['HERMES_HOME']) / 'state.db') as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute('SELECT * FROM delivery_obligations ORDER BY chat_id')]


async def cleanup(runner, adapter):
    runner._running = False
    tasks = list(getattr(runner, '_flood_redelivery_tasks', {}).values())
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await adapter._bot.shutdown()


async def scenario(wire, mode):
    from gateway import delivery_ledger as dl
    from gateway.config import Platform
    runner, adapter = await setup(wire.url)
    try:
        if mode == 'runtime':
            key = await produce(runner, adapter, '101')
            await produce(runner, adapter, '102')
            # A foreign bot's row cannot be claimed by the connected default bot.
            dl.record_obligation(obligation_id='foreign', session_key=key, platform='telegram',
                                 chat_id='999', thread_id='77', content='FOREIGN', adapter_profile='other')
            dl.mark_failed('foreign', 'flood_control:0.01')
            dl.record_obligation(obligation_id='permanent', session_key=key, platform='telegram',
                                 chat_id='403', thread_id='77', content='PERMANENT')
            dl.mark_failed('permanent', 'Forbidden: bot was blocked by the user')
            await runner._redeliver_failed_obligations_for_platform(Platform.TELEGRAM)
        else:
            # Real child owns the failed row and exits; no patched liveness predicate.
            proc = await asyncio.to_thread(subprocess.run,
                [sys.executable, str(Path(__file__).resolve()), '--produce', wire.url],
                env=os.environ.copy(), cwd=str(ROOT), stdin=subprocess.DEVNULL,
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20)
            if proc.returncode:
                raise RuntimeError(proc.stderr)
            key = json.loads(proc.stdout)['key']
            await runner._redeliver_pending_obligations()
        early = rows()
        early_calls = list(wire.calls)
        await asyncio.sleep(64)
        after = rows()
        # Explicit repeated sweeps cannot resend a confirmed delivery.
        await runner._redeliver_failed_obligations_for_platform(Platform.TELEGRAM)
        await runner._redeliver_failed_obligations_for_platform(Platform.TELEGRAM)
        final = rows()
        entry = await runner.async_session_store.get_or_create_session(
            __import__('gateway.session', fromlist=['SessionSource']).SessionSource(
                platform=Platform.TELEGRAM, chat_id='101' if mode == 'runtime' else '201', thread_id='77'))
        return {'mode': mode, 'early': early, 'early_wire': early_calls, 'after_timer': after,
                'final': final, 'wire': list(wire.calls), 'resume_pending': entry.resume_pending,
                'deadlines': dict(wire.deadlines)}
    finally:
        await cleanup(runner, adapter)


async def run_mode(output, mode):
    home = output / mode
    home.mkdir(parents=True, exist_ok=True)
    os.environ['HERMES_HOME'] = str(home)
    (home / 'config.yaml').write_text('gateway:\n  delivery_ledger: true\n')
    wire = Wire()
    try:
        result = await scenario(wire, mode)
        (output / f'{mode}.json').write_text(json.dumps(result, indent=2))
        return result
    finally:
        wire.close()


async def main():
    if sys.argv[1] == '--produce':
        runner, adapter = await setup(sys.argv[2])
        try:
            key = await produce(runner, adapter, '201')
            print(json.dumps({'key': key}))
        finally:
            await cleanup(runner, adapter)
        return
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for mode in ('runtime', 'boot'):
        result = await run_mode(output, mode)
        print(json.dumps({'mode': mode, 'states': [(r['chat_id'], r['state'], r['attempts']) for r in result['final']],
                          'wire_requests': len(result['wire']), 'resume_pending': result['resume_pending']}))


if __name__ == '__main__':
    asyncio.run(main())
