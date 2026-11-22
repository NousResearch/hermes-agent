"""Ink's /bg and /btw on the shared owner: a detached side agent off the session's FIFO.

On the per-TUI sidecar ``prompt.background`` / ``prompt.btw`` reply ``{task_id}`` at once and the
answer lands later on the parent session as ``background.complete`` / ``btw.complete``. The
canonical owner answered ``unknown_method``, so Ink refused both commands on the shared gateway.
"""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sqlite3
import threading

import pytest

from tests.gateway.test_session_busy_controls import admissions, owner, settled
from tests.gateway.fixtures.local_recovery_probe import rpc, websocket


async def call_and_await_event(ws, method, event_type, **params):
    """``method``'s reply plus the ``event_type`` frame naming its task, read off one socket (the
    shared ``rpc`` helper drops every frame that is not its reply)."""
    await ws.send(json.dumps({'jsonrpc': '2.0', 'id': method, 'method': method, 'params': params}))
    reply, events = None, []
    async with asyncio.timeout(90):
        while True:
            frame = json.loads(await ws.recv())
            if frame.get('id') == method:
                reply = frame
                if 'error' in reply:
                    return reply, None
            elif frame.get('method') == 'event' and frame['params'].get('type') == event_type:
                events.append(frame['params'])
            task = (reply or {}).get('result', {}).get('task_id')
            done = [e for e in events if e['payload'].get('task_id') == task]
            if task and done:
                return reply, done[0]


@pytest.mark.platforms("linux")
def test_background_prompt_runs_a_detached_agent_and_reports_on_the_session(tmp_path):
    with owner(tmp_path) as (home, peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                # A lazy session: no turn has run, so there is no resident agent to copy.
                sid = (await rpc(ws, 'session.create', request_id='bg', source='tui', toolsets=[]))['result']['session_id']
                reply, event = await call_and_await_event(ws, 'prompt.background', 'background.complete',
                                                          session_id=sid, text='BG_TASK')
                task = reply.get('result', {}).get('task_id', '')
                assert task.startswith('bg_'), reply
                assert event['session_id'] == sid
                assert event['payload'] == {'task_id': task, 'text': 'CONTROL_ACK'}, event
                assert any('BG_TASK' in json.dumps(r['messages']) for r in peer.requests)
                for params, reason in [({'session_id': sid, 'text': '  '}, 'invalid_params'),
                                       ({'session_id': sid, 'text': 'x', 'profile': 'foreign'}, 'profile_mismatch')]:
                    denied = await rpc(ws, 'prompt.background', **params)
                    assert denied.get('error', {}).get('message') == reason, denied
                # Off the FIFO and out of the parent transcript: no admission, no parent rows.
                assert not admissions(home)
                resumed = (await rpc(ws, 'session.resume', session_id=sid))['result']
                assert not resumed['messages'], resumed['messages']
        asyncio.run(probe())


class DelegatingModel(BaseHTTPRequestHandler):
    """The /bg prompt delegates in the background; the child answers CHILD_DONE; the parent's
    follow-up reports whether the tool result carried the child's answer or a detached dispatch."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        last = (body.get('messages') or [{}])[-1]
        message = {'role': 'assistant', 'content': 'CHILD_DONE'}
        if last.get('role') == 'tool':
            message['content'] = 'JOINED' if 'CHILD_DONE' in str(last.get('content')) else 'DETACHED'
        elif 'DELEGATE_BG' in str(last.get('content')):
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'bg-delegate', 'type': 'function', 'function': {
                    'name': 'delegate_task', 'arguments': json.dumps({'goal': 'CHILD_GOAL', 'background': True})}}]}
        finish = 'tool_calls' if message.get('tool_calls') else 'stop'
        payload = json.dumps({'id': 'bg', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
                              'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode()
        kind = 'application/json'
        if body.get('stream'):
            delta = dict(message, tool_calls=[dict(c, index=0) for c in message.get('tool_calls') or []] or None)
            payload = ('data: ' + json.dumps({'id': 'bg', 'choices': [{'index': 0, 'delta': delta,
                       'finish_reason': finish}]}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.mark.platforms("linux")
def test_background_prompt_joins_its_background_delegation(tmp_path):
    """#140569: a /bg task replies once and its bg_* session never gets another turn, so a
    background delegate_task inside it must join in the run, not leave an undrained ledger row."""
    model = ThreadingHTTPServer(('127.0.0.1', 0), DelegatingModel)
    threading.Thread(target=model.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{model.server_port}/v1'
    try:
        with owner(tmp_path, {'model': {'provider': 'custom', 'default': 'bg-fixture', 'base_url': url}}) as (
                home, _peer, desc):
            async def probe():
                async with websocket(home, desc) as ws:
                    sid = (await rpc(ws, 'session.create', request_id='bgd', source='tui',
                                     toolsets=['delegation']))['result']['session_id']
                    reply, event = await call_and_await_event(ws, 'prompt.background', 'background.complete',
                                                              session_id=sid, text='DELEGATE_BG')
                    assert event['payload']['text'] == 'JOINED', (reply, event)
            asyncio.run(probe())
            with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
                tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                detached = list(db.execute('SELECT * FROM async_delegations')) if 'async_delegations' in tables else []
            assert not detached, detached
    finally:
        model.shutdown()
        model.server_close()


@pytest.mark.platforms("linux")
def test_btw_answers_over_a_snapshot_without_touching_the_conversation(tmp_path):
    with owner(tmp_path) as (home, peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                # Lazy session (no resident agent): the stored transcript is the snapshot.
                lazy = (await rpc(ws, 'session.create', request_id='lazy', source='tui', toolsets=[]))['result']['session_id']
                reply, event = await call_and_await_event(ws, 'prompt.btw', 'btw.complete',
                                                          session_id=lazy, text='LAZY_SIDE')
                task = reply.get('result', {}).get('task_id', '')
                assert task.startswith('btw_'), reply
                assert event['payload'] == {'task_id': task, 'question': 'LAZY_SIDE', 'text': 'CONTROL_ACK'}, event
                # A session that ran a turn: the resident agent's conversation is the snapshot.
                sid = (await rpc(ws, 'session.create', request_id='btw', source='tui', toolsets=[]))['result']['session_id']
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='warm', text='WARM_TURN')
                await settled(home)
                before = (await rpc(ws, 'session.resume', session_id=sid))['result']['messages']
                peer.requests.clear()
                reply, event = await call_and_await_event(ws, 'prompt.btw', 'btw.complete',
                                                          session_id=sid, text='SIDE_QUESTION')
                task = reply['result']['task_id']
                assert event['session_id'] == sid
                assert event['payload'] == {'task_id': task, 'question': 'SIDE_QUESTION', 'text': 'CONTROL_ACK'}, event
                side = [r for r in peer.requests if 'SIDE_QUESTION' in json.dumps(r['messages'])]
                assert side and 'WARM_TURN' in json.dumps(side[0]['messages']), peer.requests
                denied = await rpc(ws, 'prompt.btw', session_id=sid, text='')
                assert denied.get('error', {}).get('message') == 'invalid_params', denied
                # History, FIFO and transcript stay untouched.
                assert [r[0] for r in admissions(home)] == ['warm']
                assert (await rpc(ws, 'session.resume', session_id=sid))['result']['messages'] == before
        asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_preview_restart_runs_a_hidden_terminal_agent_and_reports_on_the_session(tmp_path):
    """TUI-12: Desktop's preview-pane "restart dev server" (``preview.restart``) had no owner
    handler, so it always errored on the shared gateway."""
    with owner(tmp_path) as (home, peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='pv', source='tui', toolsets=[]))['result']['session_id']
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='warm', text='DEV_SERVER_TURN')
                await settled(home)
                before = (await rpc(ws, 'session.resume', session_id=sid))['result']['messages']
                reply, event = await call_and_await_event(
                    ws, 'preview.restart', 'preview.restart.complete', session_id=sid,
                    url='http://127.0.0.1:5173/', cwd=str(tmp_path), context='PREVIEW_CONSOLE')
                task = reply.get('result', {}).get('task_id', '')
                assert task.startswith('preview_'), reply
                assert event['session_id'] == sid
                assert event['payload'] == {'task_id': task, 'text': 'CONTROL_ACK'}, event
                sent = json.dumps(peer.requests[-1])
                assert 'http://127.0.0.1:5173/' in sent and 'PREVIEW_CONSOLE' in sent
                # The stored transcript is the restart agent's context (the trimmed history list).
                assert 'DEV_SERVER_TURN' in sent, sent
                tools = {t['function']['name'] for t in peer.requests[-1].get('tools') or []}
                assert 'terminal' in tools and 'delegate_task' not in tools, tools
                for params, reason in [({'session_id': sid, 'url': ' '}, 'invalid_params'),
                                       ({'session_id': sid, 'url': 'http://x', 'cwd': 3}, 'invalid_params'),
                                       ({'session_id': sid, 'url': 'http://x', 'profile': 'foreign'}, 'profile_mismatch')]:
                    denied = await rpc(ws, 'preview.restart', **params)
                    assert denied.get('error', {}).get('message') == reason, denied
                assert [r[0] for r in admissions(home)] == ['warm']
                assert (await rpc(ws, 'session.resume', session_id=sid))['result']['messages'] == before
        asyncio.run(probe())


def test_preview_restart_history_trims_a_plain_message_list():
    """TUI-12: the trimming helper takes a message snapshot, not a session-shaped dict, so the
    owner passes its stored transcript directly (the legacy sidecar snapshots under its lock)."""
    from tui_gateway.agent_callbacks import _preview_restart_history
    messages = [{'role': 'user', 'content': 'run vite'}, {'role': 'assistant', 'content': 'ok', 'reasoning': 'r'},
                {'role': 'tool', 'content': 'x' * 50, 'tool_call_id': 't'}]
    trimmed = _preview_restart_history(messages, max_tool_chars=10)
    assert [m['role'] for m in trimmed] == ['user', 'assistant', 'tool']
    assert 'reasoning' not in trimmed[1] and trimmed[2]['content'].startswith('x' * 10 + '\n... (truncated')
    assert _preview_restart_history([]) == [] and _preview_restart_history(None) == []
