"""Local I/O probe for completion batching, not a hosted-model or native UI test.

Run with the repo's Python: evals/completion_backlog_probe.py REPO OUTPUT.json.
Real shell children feed ProcessRegistry; production CLI/poller/post-turn routing
feeds a loopback HTTP turn sink. The sink replaces chat/_run_prompt_submit, NOT
ownership, consumption, queue draining, batching, or formatting.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import queue
import shlex
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import Request, urlopen


def probe(surface, scenario, directory):
    from cli import HermesCLI
    from tools import process_registry as pr
    from tools.process_registry_notifications import format_process_notification
    from tui_gateway import server

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    registry = pr.ProcessRegistry()
    received, statuses = [], []

    class Sink(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"accepted":true}')

        def log_message(self, *_args):
            return

    wire = ThreadingHTTPServer(('127.0.0.1', 0), Sink)
    thread = threading.Thread(target=wire.serve_forever, daemon=True)
    thread.start()
    session = {'session_key': 'backlog-owner', 'history_lock': threading.RLock()}
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = session['session_key']
    cli._session_db = None
    cli._pending_input = queue.Queue()
    cli._pending_resume_sessions = []
    cli._typed_voice_stop = lambda _text: False
    cli.handle_bang_shell = lambda _text: False
    cli._print_user_message_preview = lambda _text: None
    cli._turn_summary_begin = lambda: None
    cli._app = SimpleNamespace(invalidate=lambda: None)
    cli._tui_after_turn = lambda: None

    def submit(text):
        request = Request(f'http://127.0.0.1:{wire.server_port}/turn',
                          data=json.dumps({'text': str(text)}).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=10) as response:
            assert response.status == 200
        session['running'] = False
        return True

    def tui_submit(_rid, _sid, _session, text, **_kwargs):
        return submit(text)

    cli.chat = lambda text, **_kwargs: submit(text)
    processes = []
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(pr, 'process_registry', registry))
            stack.enter_context(patch.object(pr, 'CHECKPOINT_PATH', directory / 'processes.json'))
            stack.enter_context(patch.object(server, '_sessions', {'ui-owner': session}))
            stack.enter_context(patch.object(server, '_get_db', lambda: None))
            stack.enter_context(patch.object(server, '_emit', lambda *args: statuses.append(args)))
            stack.enter_context(patch.object(server, '_run_prompt_submit', tui_submit))
            stack.enter_context(patch.object(server, '_drain_queued_prompt', lambda *_a: False))
            count = 1 if scenario == 'single' else 12
            gate = directory / 'release'
            for index in range(count):
                code = ('import pathlib,time,sys; p=pathlib.Path(sys.argv[1]); '
                        '\nwhile not p.exists(): time.sleep(.01)\n'
                        f'print("BACKLOG_{index}"); sys.exit({7 if index == count - 1 else 0})')
                process = registry.spawn_local(
                    f'{shlex.quote(sys.executable)} -c {shlex.quote(code)} {shlex.quote(str(gate))}',
                    cwd=str(directory), session_key='backlog-owner')
                process.notify_on_complete = True
                processes.append(process)
            gate.touch()
            deadline = time.monotonic() + 30
            while registry.completion_queue.qsize() < count and time.monotonic() < deadline:
                time.sleep(.01)
            assert registry.completion_queue.qsize() == count
            raw = list(registry.completion_queue.queue)
            delegation = None
            if scenario == 'mixed':
                from tools import async_delegation as ad
                delegation = {'type': 'async_delegation', 'delegation_id': f'deleg-{surface}',
                              'session_key': 'backlog-owner', 'goal': 'Fixture delegation',
                              'status': 'completed', 'summary': 'DELEGATION_RESULT',
                              'dispatched_at': time.time(), 'completed_at': time.time()}
                ad._persist_dispatch(delegation)
                ad._persist_completion(delegation, {'status': 'completed'})
                watch = {'type': 'watch_match', 'session_key': 'backlog-owner',
                         'session_id': processes[0].id, 'pattern': 'READY', 'output': 'WATCH_READY'}
                while not registry.completion_queue.empty():
                    registry.completion_queue.get_nowait()
                raw = raw[:4] + [watch] + raw[4:8] + [delegation] + raw[8:]
                for event in raw:
                    registry.completion_queue.put(event)
            expected = [format_process_notification(event) for event in raw]
            if scenario == 'foreign':
                session['session_key'] = cli.session_id = 'another-owner'
                server._sessions['actual-owner'] = {'session_key': 'backlog-owner'}
            if surface == 'cli':
                cli._drain_process_notifications('cli-idle')
                if scenario == 'consumed':
                    for process in processes:
                        assert registry.wait(process.id, timeout=1)["status"] == "exited"
                while not cli._pending_input.empty():
                    cli._tui_process_one_input(cli._pending_input.get_nowait())
            elif surface == 'post-turn':
                if scenario == 'consumed':
                    for process in processes:
                        assert registry.wait(process.id, timeout=1)["status"] == "exited"
                server._run_post_turn_followups('probe', 'ui-owner', session, {}, None)
            else:
                if scenario == 'consumed':
                    for process in processes:
                        assert registry.wait(process.id, timeout=1)["status"] == "exited"
                stop = threading.Event()
                # Skip unrelated scheduled jobs; exercise the production poller loop.
                stack.enter_context(patch.object(server, '_maybe_fire_tui_loop_tick', lambda *_a: None))
                stack.enter_context(patch.object(server, '_maybe_fire_tui_heartbeat_tick', lambda *_a: None))
                stack.enter_context(patch.object(server, '_notif_poll_kanban', lambda *_a: None))
                poller = threading.Thread(target=server._notification_poller_loop,
                                          args=(stop, 'ui-owner', session))
                poller.start()
                deadline = time.monotonic() + (1 if scenario == 'foreign' else 10)
                while time.monotonic() < deadline:
                    if scenario != 'foreign' and registry.completion_queue.empty():
                        break
                    time.sleep(.01)
                stop.set()
                poller.join(10)
                assert not poller.is_alive()
            texts = [item['text'] for item in received]
            delegation_delivered = None
            if delegation:
                with ad._transaction() as conn:
                    row = conn.execute("SELECT delivery_state, delivery_attempts FROM async_delegations WHERE delegation_id=?",
                                       (delegation['delegation_id'],)).fetchone()
                    delegation_delivered = tuple(row) == ('delivered', 1)
            return {'surface': surface, 'scenario': scenario, 'children': count,
                    'wire_turns': len(texts), 'texts': texts,
                    'delegation_delivered_once': delegation_delivered,
                    'payload_order': [next((i for i, turn in enumerate(texts) if payload in turn), -1) for payload in expected],
                    'all_payloads_preserved': all(any(text in turn for turn in texts) for text in expected),
                    'single_exact': texts == expected if count == 1 else None,
                    'queue_remaining': registry.completion_queue.qsize(),
                    'status_events': len(statuses)}
    finally:
        wire.shutdown()
        wire.server_close()
        thread.join()
        for process in processes:
            if not process.exited:
                registry.kill_process(process.id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('repo')
    parser.add_argument('output')
    args = parser.parse_args()
    sys.path.insert(0, args.repo)
    with tempfile.TemporaryDirectory(prefix='completion-probe-') as home:
        os.environ['HERMES_HOME'] = home
        os.environ['HOME'] = home
        results = [probe(surface, scenario, Path(home) / surface / scenario)
                   for surface in ('cli', 'poller', 'post-turn')
                   for scenario in ('backlog', 'single', 'consumed', 'foreign', 'mixed')]
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps([{k: v for k, v in row.items() if k != 'texts'} for row in results], indent=2))


if __name__ == '__main__':
    main()
