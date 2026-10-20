"""Real owner, real delegation, authenticated viewers: a child's live progress with no file reads.

A loopback model drives one turn: the parent delegates an orchestrator child, which delegates a leaf
whose terminal command waits on a release file. While the leaf's tool runs, a second viewer of the
parent session reads the ``subagent.*`` frames and both children's ``subagent.transcript``; a viewer of
another session watches the same time. Every reply and frame lands in ``receipt.json``.
"""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import threading
import time
import traceback

ORCHESTRATOR = 'Subagent Spawning (Orchestrator Role)'
SUBAGENT = 'You are a focused subagent'


class Peer(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data": []}')

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = request.get('messages', [])
        system, answered = json.dumps(messages[:1]), any(m.get('role') == 'tool' for m in messages)
        if ORCHESTRATOR in system:
            reply = ('CHILD_FINAL', None) if answered else (None, ('delegate_task', {'tasks': [{'goal': 'LEAF_GOAL'}]}))
        elif SUBAGENT in system:
            reply = ('LEAF_FINAL', None) if answered else (None, ('terminal', {'command': self.server.command}))
        else:
            reply = ('PARENT_DONE', None) if answered else (None, ('delegate_task', {'tasks': [{'goal': 'CHILD_GOAL'}]}))
        content, call = reply
        message = {'role': 'assistant', 'content': content}
        if call:
            message['tool_calls'] = [{'index': 0, 'id': f'call-{call[0]}-{time.monotonic_ns()}', 'type': 'function',
                                      'function': {'name': call[0], 'arguments': json.dumps(call[1])}}]
        finish = 'tool_calls' if call else 'stop'
        self.send_response(200)
        if not request.get('stream'):
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'id': 'x', 'object': 'chat.completion', 'created': 1, 'model': 'm',
                'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}).encode())
            return
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        for delta, fin in [(message, None), ({}, finish)]:
            frame = {'id': 'x', 'object': 'chat.completion.chunk', 'created': 1, 'model': 'm',
                     'choices': [{'index': 0, 'delta': delta, 'finish_reason': fin}]}
            self.wfile.write(('data: ' + json.dumps(frame) + '\n\n').encode())
        self.wfile.write(b'data: [DONE]\n\n')
        self.wfile.flush()


class Viewer:
    """One authenticated socket: RPC replies by id, every event frame kept."""

    def __init__(self, ws):
        self.ws, self.events, self.waiters, self.rid = ws, [], {}, 0
        self.task = asyncio.create_task(self.read())

    async def read(self):
        async for raw in self.ws:
            frame = json.loads(raw)
            if frame.get('id') in self.waiters:
                self.waiters.pop(frame['id']).set_result(frame)
            elif frame.get('method') == 'event':
                self.events.append(frame['params'])

    async def rpc(self, method, **params):
        self.rid += 1
        future = asyncio.get_running_loop().create_future()
        self.waiters[self.rid] = future
        await self.ws.send(json.dumps({'jsonrpc': '2.0', 'id': self.rid, 'method': method, 'params': params}))
        frame = await asyncio.wait_for(future, 20)
        return frame.get('result', frame.get('error'))

    def subagent_frames(self):
        return [{'type': e['type'], 'session_id': e.get('session_id'), **{
            k: e['payload'].get(k) for k in ('subagent_id', 'parent_id', 'depth', 'tool_name', 'status',
                                              'child_session_id', 'goal')}}
                for e in self.events if e['type'].startswith('subagent.')]


async def until(what, check, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = await check()
        if found:
            return found
        await asyncio.sleep(0.1)
    raise AssertionError(f'timed out waiting for {what}')


async def probe(release):
    import websockets
    from gateway.run import GatewayRunner
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from gateway.session_authority import initialize_session_authority
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket

    runner = GatewayRunner()
    await initialize_session_authority(runner, profile_id='fixture', instance_id='subagent-live')
    api = await start_gateway_api(runner)
    web_server.app.state.auth_required = True
    origin = api.api_origin.replace('http:', 'ws:')

    async def viewer(actor):
        ticket = mint_ticket(user_id=actor, provider='fixture')
        return Viewer(await websockets.connect(f'{origin}/api/ws?ticket={ticket}', max_size=8 * 1024 * 1024))

    receipt, viewers = {}, []
    try:
        owner, peer, foreign = viewers[:] = [await viewer(a) for a in ('owner', 'peer_operator', 'foreign_viewer')]
        sid = (await owner.rpc('session.create', request_id='live-parent', source='gui',
                               toolsets=['terminal', 'delegation']))['session_id']
        other = (await foreign.rpc('session.create', request_id='live-other', source='gui', toolsets=[]))['session_id']
        await owner.rpc('session.resume', session_id=sid)
        await peer.rpc('session.resume', session_id=sid)
        await foreign.rpc('session.resume', session_id=other)
        await owner.rpc('prompt.submit', session_id=sid, input_id='delegate', text='delegate it')

        async def leaf_tool_running():
            rows = (await peer.rpc('subagent.list', session_id=sid)).get('subagents') or []
            leaf = next((r for r in rows if r.get('depth') == 1 and r.get('last_tool') == 'terminal'), None)
            if leaf is None:
                return None
            reply = await peer.rpc('subagent.transcript', session_id=sid, subagent_id=leaf['subagent_id'])
            return (rows, reply) if any(m.get('tool_calls') for m in reply['messages']) else None

        rows, receipt['leaf_running'] = await until('the leaf tool call in its stored transcript', leaf_tool_running)
        receipt['list'] = rows
        child = next(r for r in rows if r.get('depth') == 0)
        leaf = next(r for r in rows if r.get('depth') == 1)
        receipt['child_running'] = await peer.rpc('subagent.transcript', session_id=sid, subagent_id=child['subagent_id'])
        release.write_text('go')

        async def child_complete():
            return any(f['type'] == 'subagent.complete' and f['subagent_id'] == child['subagent_id']
                       for f in peer.subagent_frames())
        await until('the child subagent.complete frame', child_complete)
        receipt['leaf_after'] = await peer.rpc('subagent.transcript', session_id=sid, subagent_id=leaf['subagent_id'],
                                               after=receipt['leaf_running']['cursor'])
        receipt['leaf_done'] = await peer.rpc('subagent.transcript', session_id=sid, subagent_id=leaf['subagent_id'])
        receipt['child_done'] = await peer.rpc('subagent.transcript', session_id=sid, subagent_id=child['subagent_id'])
        receipt['foreign_read'] = await foreign.rpc('subagent.transcript', session_id=other,
                                                    subagent_id=leaf['subagent_id'])
        receipt['frames'] = peer.subagent_frames()
        receipt['foreign_frames'] = [e for e in foreign.events if e['type'].startswith('subagent.')
                                     or e.get('session_id') == sid]
        receipt['session_id'] = sid
    finally:
        for v in viewers:
            await v.ws.close()
        await stop_gateway_api(api)
    Path(os.environ['HERMES_HOME'], 'receipt.json').write_text(json.dumps(receipt))


def main():
    home = Path(os.environ['HERMES_HOME'])
    release = home / 'release'
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    peer.command = (f'for i in $(seq 1 600); do [ -f {shlex.quote(str(release))} ] && break; sleep 0.05; done; '
                    'echo LEAF_TOOL_DONE')
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, TERMINAL_ENV='local')
    (home / 'config.yaml').write_text(
        f'model:\n  default: local-live\n  provider: custom\n  base_url: {url}\n'
        'streaming:\n  enabled: false\ndelegation:\n  max_spawn_depth: 2\n'
        'auxiliary:\n  title_generation:\n    enabled: false\n')
    try:
        asyncio.run(probe(release))
    finally:
        peer.shutdown()


if __name__ == '__main__':
    status = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        status = 1
    os._exit(status)
