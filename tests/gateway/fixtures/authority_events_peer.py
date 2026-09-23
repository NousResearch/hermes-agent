"""Real authority execution with an OS-full observer and independent reader."""
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import threading
import traceback

from shared_authority_peer import ModelPeer


def main():
    peer = ThreadingHTTPServer(('127.0.0.1', 0), ModelPeer)
    peer.requests, peer.metadata_requests = [], []
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='explicit-loopback-fixture', OPENAI_BASE_URL=url,
                      TELEGRAM_ALLOWED_USERS='fixture-user')
    Path(os.environ['HERMES_HOME'], 'config.yaml').write_text(
        f'model:\n  default: local-wire-stub\n  provider: custom\n  base_url: {url}\n'
        'auxiliary:\n  title_generation:\n    enabled: false\n')
    from tui_gateway.transport import StdioTransport

    slow_read, slow_write = os.pipe()
    healthy_read, healthy_write = os.pipe()
    os.set_blocking(slow_write, False)
    filled = 0
    try:
        while True:
            filled += os.write(slow_write, b'x' * 4096)
    except BlockingIOError:
        pass
    os.set_blocking(slow_write, True)
    slow_stream = os.fdopen(slow_write, 'w', encoding='utf-8')
    healthy_stream = os.fdopen(healthy_write, 'w', encoding='utf-8')
    slow = StdioTransport(lambda: slow_stream, threading.Lock())
    healthy = StdioTransport(lambda: healthy_stream, threading.Lock())
    frames = queue.Queue()

    def read_healthy():
        with os.fdopen(healthy_read, 'r', encoding='utf-8') as stream:
            for line in stream:
                frames.put(json.loads(line))

    reader = threading.Thread(target=read_healthy, daemon=True)
    reader.start()
    ready, returned = threading.Event(), threading.Event()
    errors = []

    async def execute():
        from gateway.config import Platform
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource
        from gateway.session_authority import initialize_session_authority
        from gateway.session_controls import AuthorityConnection
        from gateway.session_contract import Submission
        from hermes_state_runtime import list_session_admissions
        from run_agent import AIAgent

        runner = GatewayRunner()
        authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='events-owner')
        ref = authority.register(SessionSource(platform=Platform.TELEGRAM, chat_id='event-chat',
                                               chat_type='dm', user_id='fixture-user'))
        a, b = [AuthorityConnection(authority, t, {'user_id': 'fixture-user'}) for t in (slow, healthy)]
        for connection in (a, b):
            await connection.resume(ref, {})
            await connection.resume(ref, {})
        assert len(authority.sessions[ref.session_id].subscribers) == 2
        ready.set()
        receipt = await authority.submit(b.actor, Submission('event-input', ref, {'text': 'WS_SHARED'}, 'queue'))
        await authority.sessions[ref.session_id].task
        assert isinstance(authority.agent(ref), AIAgent)
        row = next(r for r in list_session_admissions(authority.db, session_id=ref.session_id,
                                                     pending_only=False) if r['admission_id'] == receipt.admission_id)
        assert row['outcome'] == 'completed', row
        while True:
            frame = await asyncio.to_thread(frames.get, True, 5)
            if frame['params']['type'] == 'message.complete':
                frames.put(frame)
                break
        await a.close()
        assert len(authority.sessions[ref.session_id].subscribers) == 1
        await b.close()
        assert not authority.sessions[ref.session_id].subscribers
        assert not authority.events

    def producer():
        try:
            asyncio.run(execute())
        except BaseException:
            errors.append(traceback.format_exc())
        finally:
            returned.set()

    worker = threading.Thread(target=producer, daemon=True)
    worker.start()
    before_release = healthy_completed = False
    try:
        assert ready.wait(30), errors
        before_release = returned.wait(30)
        if before_release:
            frame = frames.get(timeout=5)
            healthy_completed = (frame['params']['type'] == 'message.complete'
                                 and 'LOCAL_ACK_WS_SHARED' in json.dumps(frame))
    finally:
        os.close(slow_read)
        worker.join(timeout=15)
        try:
            slow_stream.close()
        except BrokenPipeError:
            pass
        healthy_stream.close()
        reader.join(timeout=5)
        peer.shutdown()
        peer.server_close()
    assert not worker.is_alive(), 'producer did not terminate after cleanup'
    assert not errors, errors
    assert before_release, 'OS-full observer blocked real authority producer before reader release'
    assert healthy_completed, 'healthy observer did not complete before stalled reader release'
    # TUI imports may replace stdout with their protocol-safe logger.
    import sys
    sys.__stdout__.write(json.dumps({'os_buffer_full_bytes': filled,
                                    'producer_returned_before_release': before_release,
                                    'healthy_completed_before_release': healthy_completed,
                                    'model_calls': len(peer.requests)}) + '\n')
    sys.__stdout__.flush()


if __name__ == '__main__':
    status = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        status = 1
    os._exit(status)
