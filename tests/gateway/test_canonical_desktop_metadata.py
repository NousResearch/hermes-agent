"""Canonical GUI metadata follows the authenticated local-session route."""
import asyncio
import json

import pytest

from tests.gateway.fixtures.local_recovery_probe import rpc, websocket
from tests.gateway.test_session_busy_controls import owner, settled


@pytest.mark.linux_only
def test_canonical_desktop_metadata_survives_create_info_and_reattach(tmp_path):
    """The native protocol, not the legacy contract number, identifies this route."""
    with owner(tmp_path) as (home, _peer, desc):
        async def probe():
            async with websocket(home, desc) as creator:
                created = await rpc(creator, 'session.create', request_id='desktop-metadata',
                                    source='gui', toolsets=[])
                assert 'result' in created, created
                sid = created['result']['session_id']
                created_info = created['result']['info']
                assert created_info['desktop_protocol'] == creator.subprotocol
                assert 'desktop_contract' not in created_info
                assert created_info['lazy'] is True

                lazy_info = await rpc(creator, 'session.info', session_id=sid)
                assert lazy_info['result']['desktop_protocol'] == creator.subprotocol
                assert lazy_info['result']['lazy'] is True

                # Read every frame: the RPC helper deliberately skips events,
                # but Desktop reconciles their metadata throughout each turn.
                await creator.send(json.dumps({'jsonrpc': '2.0', 'id': 'initialize',
                    'method': 'prompt.submit', 'params': {'session_id': sid,
                        'input_id': 'initialize', 'text': 'INITIALIZE'}}))
                submitted = saw_pending = finished = False
                async with asyncio.timeout(30):
                    while not submitted or not finished:
                        frame = json.loads(await creator.recv())
                        if frame.get('id') == 'initialize':
                            assert frame['result']['status'] == 'queued', frame
                            submitted = True
                        event = frame.get('params', {})
                        if event.get('type') == 'session.info' and event.get('session_id') == sid:
                            info = event['payload']
                            assert info['desktop_protocol'] == creator.subprotocol
                            assert 'desktop_contract' not in info
                            saw_pending |= bool(info['pending'])
                            finished = saw_pending and not info['pending'] and not info['running']
                await settled(home)

                initialized_info = await rpc(creator, 'session.info', session_id=sid)
                assert initialized_info['result']['desktop_protocol'] == creator.subprotocol
                assert initialized_info['result']['lazy'] is False

            async with websocket(home, desc) as reattached:
                resumed = await rpc(reattached, 'session.resume', session_id=sid)
                assert resumed['result']['info']['desktop_protocol'] == reattached.subprotocol
                assert resumed['result']['info']['lazy'] is False

        asyncio.run(probe())
