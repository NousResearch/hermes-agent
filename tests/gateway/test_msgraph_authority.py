"""Graph boundary uses the canonical internal queue, not a background ACK."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys


def test_graph_boundary_commits_only_valid_notifications(tmp_path):
    result = subprocess.run([sys.executable, __file__], env=dict(os.environ, HERMES_HOME=str(tmp_path)),
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    print((tmp_path / 'graph-receipt.json').read_text())


async def probe(peer):
    import aiohttp
    import ipaddress
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.msgraph_webhook import MSGraphWebhookAdapter
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_envelope import restore_native
    from hermes_state_runtime import list_session_admissions
    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='graph-test')
    adapter = MSGraphWebhookAdapter(PlatformConfig(enabled=True, extra={
        'host': '127.0.0.1', 'port': 0, 'client_state': 'owned-state',
        'accepted_resources': ['users/owned/messages'], 'prompt': 'BLOCK_FIFO'}))
    runner.adapters[Platform.MSGRAPH_WEBHOOK] = adapter
    runner._wire_adapter_handlers(adapter)
    assert await adapter.connect()
    port = next(iter(adapter._runner.sites))._server.sockets[0].getsockname()[1]
    url = f'http://127.0.0.1:{port}{adapter._webhook_path}'
    notification = {'id': 'graph-1', 'subscriptionId': 'owned-sub',
                    'resource': 'users/owned/messages/one', 'clientState': 'owned-state'}
    def rows():
        return [r for sid in authority.sessions for r in list_session_admissions(
            authority.db, session_id=sid, pending_only=False)]
    try:
        async with aiohttp.ClientSession() as client:
            for change, status in [({'clientState': 'wrong'}, 403), ({'resource': 'users/foreign'}, 400)]:
                response = await client.post(url, json={'value': [{**notification, **change}]})
                assert response.status == status and not rows()
            adapter._allowed_source_networks = [ipaddress.ip_network('192.0.2.0/24')]
            denied = await client.post(url, json={'value': [notification]})
            assert denied.status == 403 and not rows()
            adapter._allowed_source_networks = []
            authority.db._execute_write(lambda conn: conn.execute("CREATE TRIGGER fail_graph BEFORE INSERT ON session_admissions BEGIN SELECT RAISE(ABORT, 'owned storage failure'); END"))
            response = await client.post(url, json={'value': [notification]})
            assert response.status == 503, response.status
            assert not rows() and not peer.requests
            authority.db._execute_write(lambda conn: conn.execute('DROP TRIGGER fail_graph'))
            response = await client.post(url, json={'value': [notification]})
            assert response.status == 202 and len(rows()) == 1
            assert restore_native(rows()[0]['payload'], runner).internal is True
            assert await asyncio.to_thread(peer.blocked.wait, 10)
            adapter._seen_receipts.clear()
            duplicate = await client.post(url, json={'value': [notification]})
            assert duplicate.status == 202 and len(rows()) == 1
            peer.release.set()
            async with asyncio.timeout(25):
                while rows()[0]['status'] != 'terminal':
                    await asyncio.sleep(.01)
            assert len(peer.requests) == 1
            # A plugin scheduler owns this route exclusively; do not also infer.
            calls = []
            async def scheduler(raw, event):
                calls.append(raw['id'])
            adapter.set_notification_scheduler(scheduler)
            response = await client.post(url, json={'value': [{**notification, 'id': 'plugin-only'}]})
            assert response.status == 202 and calls == ['plugin-only']
            assert len(rows()) == len(peer.requests) == 1
            Path(os.environ['HERMES_HOME'], 'graph-receipt.json').write_text(json.dumps({
                'admissions': len(rows()), 'inferences': len(peer.requests), 'boundary_controls': True,
                'internal': True, 'storage_retry': True, 'scheduler_exclusive': True}))
    finally:
        peer.release.set()
        await adapter.disconnect()


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(1, str(Path(__file__).parent))
    import test_webhook_authority as harness
    import traceback
    harness.probe = probe
    status = 0
    try:
        harness.main()
    except BaseException:
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
