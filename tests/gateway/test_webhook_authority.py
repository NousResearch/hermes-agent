"""Signed TCP ingress must not acknowledge an uncommitted input."""
import asyncio
import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys


def test_signed_webhook_storage_failure_is_retryable(tmp_path):
    result = subprocess.run([sys.executable, __file__],
        env=dict(os.environ, HERMES_HOME=str(tmp_path)), capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    print((tmp_path / 'webhook-receipt.json').read_text())


async def probe(peer):
    import aiohttp
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.webhook import WebhookAdapter
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from hermes_state_runtime import list_session_admissions

    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='webhook-test')
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={
        'host': '127.0.0.1', 'port': 0, 'secret': 'owned-hook-secret',
        'routes': {'fixture': {'prompt': '{text}'}}}))
    adapter.gateway_runner = runner
    runner.adapters[Platform.WEBHOOK] = adapter
    runner._wire_adapter_handlers(adapter)
    assert await adapter.connect()
    port = next(iter(adapter._runner.sites))._server.sockets[0].getsockname()[1]
    url = f'http://127.0.0.1:{port}/webhooks/fixture'
    body = json.dumps({'text': 'BLOCK_FIFO'}).encode()
    headers = {'X-GitHub-Delivery': 'stable-provider-id', 'X-GitHub-Event': 'push',
        'X-Hub-Signature-256': 'sha256=' + hmac.new(b'owned-hook-secret', body, hashlib.sha256).hexdigest()}
    def rows():
        return [row for sid in authority.sessions for row in
                list_session_admissions(authority.db, session_id=sid, pending_only=False)]
    try:
        async with aiohttp.ClientSession() as client:
            denied = await client.post(url, data=body, headers={**headers, 'X-Hub-Signature-256': 'sha256=bad'})
            assert denied.status == 401 and not rows()
            authority.db._execute_write(lambda conn: conn.execute("CREATE TRIGGER fail_ingress BEFORE INSERT ON session_admissions BEGIN SELECT RAISE(ABORT, 'owned storage failure'); END"))
            failed = await client.post(url, data=body, headers=headers)
            assert failed.status == 503, (failed.status, await failed.text())
            assert not rows() and not peer.requests
            authority.db._execute_write(lambda conn: conn.execute('DROP TRIGGER fail_ingress'))
            accepted = await client.post(url, data=body, headers=headers)
            assert accepted.status == 202, (accepted.status, await accepted.text())
            assert len(rows()) == 1 and rows()[0]['request_id'] == 'stable-provider-id'
            assert await asyncio.to_thread(peer.blocked.wait, 10)
            duplicate = await client.post(url, data=body, headers=headers)
            assert duplicate.status in (200, 202), await duplicate.text()
            assert len(rows()) == 1
            peer.release.set()
            async with asyncio.timeout(25):
                while rows()[0]['status'] != 'terminal':
                    await asyncio.sleep(.01)
            # Lose the process-local optimization: durable identity still suppresses replay.
            adapter._seen_deliveries.clear()
            duplicate = await client.post(url, data=body, headers=headers)
            assert duplicate.status in (200, 202), await duplicate.text()
            assert len(rows()) == 1 and len(peer.requests) == 1
            async with asyncio.timeout(5):
                while adapter._background_tasks:
                    await asyncio.sleep(.01)
            assert authority.db.get_session(rows()[0]['target_session_id'])['ended_at'] is not None
            Path(os.environ['HERMES_HOME'], 'webhook-receipt.json').write_text(json.dumps({
                'failed_status': failed.status, 'retry_status': accepted.status,
                'admissions': len(rows()), 'inferences': len(peer.requests), 'signature_denied': denied.status}))
    finally:
        peer.release.set()
        await adapter.disconnect()


def main():
    from http.server import ThreadingHTTPServer
    import threading
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(1, str(Path(__file__).parent / 'fixtures'))
    from shared_authority_peer import ModelPeer
    peer = ThreadingHTTPServer(('127.0.0.1', 0), ModelPeer)
    peer.requests, peer.metadata_requests = [], []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    Path(os.environ['HERMES_HOME'], 'config.yaml').write_text(
        f'model:\n  default: local-wire-stub\n  provider: custom\n  base_url: {url}\n'
        'streaming:\n  enabled: false\nauxiliary:\n  title_generation:\n    enabled: false\n')
    try:
        asyncio.run(probe(peer))
    finally:
        peer.shutdown()
        peer.server_close()


if __name__ == '__main__':
    import traceback
    status = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
