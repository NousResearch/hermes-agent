"""GW-2: an admission the authority is executing is never parked by the startup-restore gate.

A webhook POST that lands while startup auto-resume runs is committed to the durable FIFO. If the
gate then parked its execution, the row settled ``terminal`` with no turn, the webhook finalizer
ended the session in state.db, and the gate's later replay ran the input a second time outside its
admission (``routing key ... is ended in state.db but still live``, reply ``not_found``).
"""
import hashlib
import hmac
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter
from tests.gateway.fixtures.webhook_route_authority import mount_authority


@pytest.mark.asyncio
async def test_admitted_webhook_runs_through_startup_restore_gate(monkeypatch):
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import execute_admission
    from hermes_state_runtime import get_session_admission

    config = PlatformConfig(enabled=True, extra={
        'secret': 'owned-secret', 'routes': {'fixture': {'prompt': '{text}', 'deliver': 'log'}}})
    adapter = WebhookAdapter(config)
    app = web.Application()
    mount_authority(app, adapter)
    app.router.add_post('/webhooks/{route_name}', adapter._handle_webhook)
    async with TestClient(TestServer(app)) as client:
        runner = adapter._message_handler.__self__
        # Boot auto-resume is still running: the inbound gate is closed.
        runner._startup_restore_in_progress = True
        runner._startup_restore_queue = []
        body = json.dumps({'text': 'during restore'}).encode()
        headers = {'X-GitHub-Delivery': 'restore-one', 'X-Hub-Signature-256':
                   'sha256=' + hmac.new(b'owned-secret', body, hashlib.sha256).hexdigest()}
        response = await client.post('/webhooks/fixture', data=body, headers=headers)
        assert response.status == 202, await response.text()
        authority = runner.session_authority
        admission_id = authority.db._read_one('SELECT admission_id FROM session_admissions')['admission_id']
        row = get_session_admission(authority.db, admission_id=admission_id)

        turns = []

        async def turn_ran(event, source, _key):
            # First seam past the gate. This pins the gate decision only: claim, settlement,
            # webhook finalisation and delivery are not exercised here.
            turns.append(event.text)
            return 'answered'

        monkeypatch.setattr(runner, '_hm_pending_reply_intercepts', turn_ran)
        result = await execute_admission(authority, SessionRef(authority.profile_id, row['target_session_id']), row)

        assert runner._startup_restore_queue == [], 'the gate parked an admission it does not own'
        assert turns == ['during restore']
        assert result == 'answered'
