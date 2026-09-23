"""Disposable production-runtime peer; only provider and vendor wire are synthetic."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import threading
import traceback


class ModelPeer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"data": [{"id": "local-wire-stub", "object": "model"}]}).encode())

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        last = next((m.get("content", "") for m in reversed(request.get("messages", []))
                     if m.get("role") == "user"), "")
        marker = "WS_SHARED" if "WS_SHARED" in json.dumps(last) else "MESSAGING_WARM"
        reply = f"LOCAL_ACK_{marker}"
        if 'BLOCK_FIFO' in json.dumps(last):
            self.server.blocked.set()
            assert self.server.release.wait(15), 'fixture model gate timed out'
        self.server.requests.append(request) if 'messages' in request else self.server.metadata_requests.append(request)
        base = {"id": "chatcmpl-local", "model": "local-wire-stub", "created": 1}
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if request.get("stream") else "application/json")
        self.end_headers()
        if request.get("stream"):
            for delta, finish in [({"role": "assistant", "content": reply}, None), ({}, "stop")]:
                chunk = {**base, "object": "chat.completion.chunk", "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self.wfile.write(json.dumps({**base, "object": "chat.completion", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}).encode())


async def probe():
    import uvicorn
    import websockets
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    from tui_gateway import server
    from run_agent import AIAgent
    from shared_authority_recovery import prepare_crash, recover_probe, verify_envelope_guards

    class CaptureAdapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token="local-fixture"), Platform.TELEGRAM)
            self.sent = []
            self.deliveries = []

        async def connect(self, *, is_reconnect=False):
            return True

        async def disconnect(self):
            pass

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.deliveries.append({'chat_id': chat_id, 'content': content, 'reply_to': reply_to, 'metadata': metadata})
            self.sent.append(content)
            return SendResult(success=True, message_id=str(len(self.sent)))

        async def edit_message(self, chat_id, message_id, content, *, finalize=False):
            self.sent.append(content)
            return SendResult(success=True, message_id=message_id)

        async def send_typing(self, chat_id, metadata=None):
            pass

        async def get_chat_info(self, chat_id):
            return {"id": chat_id}

    runner = GatewayRunner()
    if importlib.util.find_spec("gateway.session_authority") is not None:
        from gateway.session_authority import initialize_session_authority
        await initialize_session_authority(runner, profile_id="default", instance_id="test-owner")
    adapter = CaptureAdapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    adapter.set_message_handler(runner._handle_message)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="fixture-chat", chat_type="dm", user_id="fixture-user")
    if os.environ.get('AUTHORITY_PROBE_MODE') == 'recover':
        await recover_probe(runner, adapter, source, peer)
        return
    event = MessageEvent(text="MESSAGING_WARM", source=source, message_id="warm-1")
    await adapter.handle_message(event)
    async with asyncio.timeout(35):
        while adapter._active_sessions:
            await asyncio.sleep(0.05)
    assert event._gateway_accepted, adapter.sent
    assert any("LOCAL_ACK_MESSAGING_WARM" in text for text in adapter.sent), adapter.sent
    entry = runner.session_store.get_or_create_session(source)
    warm_agent = runner._cached_agent_for(entry.session_key)
    assert isinstance(warm_agent, AIAgent), "Messaging warmup must retain a real live agent"

    api_handle = None
    if importlib.util.find_spec("gateway.run_api") is not None:
        from gateway.run_api import start_gateway_api
        api_handle = await start_gateway_api(runner)
        port = api_handle.socket.getsockname()[1]
    else:
        # Retain the historical transport only for the pre-authority A/B tree.
        app = web_server.app
        app.state.auth_required = True
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.setblocking(False)
        port = listener.getsockname()[1]
        http = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off", ws="websockets"))
        task = asyncio.create_task(http.serve(sockets=[listener]))
        async with asyncio.timeout(10):
            while not http.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
    frames = []
    receipt = {"messaging_completed": True, "stored_id": entry.session_id,
               "unauthenticated_rejected": False, "ws_completed": False, "same_agent": False}
    try:
        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}/api/ws"):
                raise AssertionError("Missing WS credential was accepted")
        except websockets.exceptions.InvalidStatus as exc:
            assert exc.response.status_code == 403, str(exc)
            receipt["unauthenticated_rejected"] = True
        if api_handle is not None:
            credential = f"token={web_server._SESSION_TOKEN}"
        else:
            credential = f"ticket={mint_ticket(user_id='fixture-user', provider='fixture')}"
        async with websockets.connect(f"ws://127.0.0.1:{port}/api/ws?{credential}") as ws:
            async def until(predicate):
                async with asyncio.timeout(35):
                    while True:
                        frame = json.loads(await ws.recv())
                        frames.append(frame)
                        if predicate(frame):
                            return frame

            await until(lambda f: f.get("params", {}).get("type") == "gateway.ready")
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session.resume",
                                      "params": {"session_id": entry.session_id}}))
            resumed = await until(lambda f: f.get("id") == 1)
            receipt["resume"] = resumed
            if "result" in resumed:
                sid = resumed["result"]["session_id"]
                await ws.send(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "prompt.submit",
                                          "params": {"session_id": sid, "text": "WS_SHARED", "input_id": "ws-1"}}))
                submitted = await until(lambda f: f.get("id") == 2)
                receipt["submit"] = submitted
                if "result" in submitted:
                    terminal = await until(lambda f: f.get("params", {}).get("type") in {"message.complete", "error"})
                    receipt["terminal"] = terminal
                    receipt["ws_completed"] = "LOCAL_ACK_WS_SHARED" in json.dumps(terminal)
                    authority = getattr(runner, 'session_authority', None)
                    if authority is not None:
                        from gateway.session_contract import SessionRef
                        receipt["same_agent"] = authority.agent(SessionRef('default', sid)) is warm_agent
                    else:
                        receipt["same_agent"] = server._sessions[sid].get("agent") is warm_agent
            receipt["messaging_outputs"] = adapter.sent
            if getattr(runner, 'session_authority', None) is not None:
                from hermes_state_runtime import list_session_admissions
                await adapter.handle_message(MessageEvent(text='BLOCK_FIFO', source=source, message_id='fifo-1'))
                assert await asyncio.to_thread(peer.blocked.wait, 5)
                queued_event = MessageEvent(text='FIFO_SECOND', source=source, message_id='fifo-2')
                queued_task = asyncio.create_task(adapter.handle_message(queued_event))
                try:
                    async with asyncio.timeout(3):
                        while not any(r['request_id'] == 'fifo-2' for r in list_session_admissions(
                                runner.session_authority.db, session_id=entry.session_id)):
                            await asyncio.sleep(0.01)
                    assert not adapter._pending_messages, 'second frontend queue owns accepted work'
                    receipt['durable_native_fifo'] = True
                    pending = list_session_admissions(runner.session_authority.db, session_id=entry.session_id)
                    queued_row = next(r for r in pending if r['request_id'] == 'fifo-2')
                    envelope = queued_row['payload'].get('native_text_v1')
                    assert envelope is not None, 'ACK has no durable native source envelope'
                    assert envelope['source']['user_id'] == 'fixture-user'
                    assert envelope['route'] == entry.session_key
                    published = await until(lambda f: f.get('params', {}).get('type') == 'session.info'
                        and any(r.get('input_id') == 'fifo-2' for r in f['params'].get('payload', {}).get('pending', [])))
                    assert any(r['text'] == 'FIFO_SECOND' for r in published['params']['payload']['pending'])
                    await ws.send(json.dumps({'jsonrpc': '2.0', 'id': 'queue-snapshot',
                        'method': 'session.resume', 'params': {'session_id': entry.session_id}}))
                    projected = (await until(lambda f: f.get('id') == 'queue-snapshot'))['result']['pending']
                    visible = next(r for r in projected if r['admission_id'] == queued_row['admission_id'])
                    assert visible['text'] == 'FIFO_SECOND' and visible['input_id'] == 'fifo-2', visible
                    assert not {'payload', 'native_text_v1', 'source', 'principal_id'} & visible.keys(), visible
                    receipt['queue_snapshot_text'] = visible['text']
                    queued_event.text = 'MUTATED_AFTER_ACK'
                    queued_event.source = SessionSource(platform=Platform.TELEGRAM, chat_id='wrong-chat', user_id='foreign')
                finally:
                    peer.release.set()
                    await queued_task
                    async with asyncio.timeout(10):
                        while adapter._active_sessions:
                            await asyncio.sleep(0.01)
                    task_owner = runner.session_authority.sessions[entry.session_id].task
                    if task_owner is not None:
                        await task_owner
                rows = list_session_admissions(runner.session_authority.db,
                                                session_id=entry.session_id, pending_only=False)
                assert [r['request_id'] for r in rows] == ['warm-1', 'ws-1', 'fifo-1', 'fifo-2'], rows
                assert all(r['status'] == 'terminal' and r['outcome'] == 'completed' for r in rows), rows
                assert runner._cached_agent_for(entry.session_key) is warm_agent
                last_users = [next((m.get('content', '') for m in reversed(r.get('messages', []))
                                   if m['role'] == 'user'), '') for r in peer.requests]
                assert 'FIFO_SECOND' in json.dumps(last_users[-1]), last_users
                assert 'MUTATED_AFTER_ACK' not in json.dumps(last_users), last_users
                assert any('LOCAL_ACK_WS_SHARED' in text for text in adapter.sent), adapter.sent
                before = len(rows)
                rejected = MessageEvent(text='UNAUTHORIZED', message_id='denied', source=SessionSource(
                    platform=Platform.TELEGRAM, chat_id='fixture-chat', chat_type='group', user_id='foreign'))
                await runner._handle_message(rejected)
                assert len(list_session_admissions(runner.session_authority.db,
                           session_id=entry.session_id, pending_only=False)) == before
                receipt['admissions'] = rows
                await verify_envelope_guards(runner, source)
                if os.environ.get('AUTHORITY_PROBE_MODE') == 'crash':
                    await prepare_crash(runner, adapter, source, peer)
    finally:
        Path(os.environ["HERMES_HOME"], "receipt.json").write_text(json.dumps(receipt, indent=2))
        Path(os.environ["HERMES_HOME"], "frames.json").write_text(json.dumps(frames, indent=2))
        if api_handle is not None:
            from gateway.run_api import stop_gateway_api
            await stop_gateway_api(api_handle)
        else:
            http.should_exit = True
            await asyncio.wait_for(task, 10)
            listener.close()
    print(json.dumps(receipt))


if __name__ == "__main__":
    peer = ThreadingHTTPServer(("127.0.0.1", 0), ModelPeer)
    peer.requests = []
    peer.metadata_requests = []
    peer.blocked = threading.Event()
    peer.release = threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{peer.server_port}/v1"
    os.environ.update(OPENAI_API_KEY="explicit-loopback-fixture", OPENAI_BASE_URL=base_url,
                      TELEGRAM_ALLOWED_USERS="fixture-user")
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(
        f"model:\n  default: local-wire-stub\n  provider: custom\n  base_url: {base_url}\n"
        f"auxiliary:\n  title_generation:\n    enabled: false\nterminal:\n  cwd: {os.environ['HERMES_HOME']}\n")
    status = 0
    try:
        asyncio.run(probe())
    except BaseException:
        traceback.print_exc()
        status = 1
    finally:
        peer.shutdown()
        peer.server_close()
        sys.stdout.flush()
        sys.stderr.flush()
    # Disposable child owns all runtime background threads (heartbeats, reapers,
    # provider pools); don't leave a lease-wait executor holding pytest hostage.
    os._exit(status)
