"""Durable receipt recovery through real HTTP handlers, without a listener."""
import json
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import make_mocked_request

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from gateway.platforms.api_server_runs import _close_run_state, _run_fingerprint


@pytest.mark.asyncio
@pytest.mark.parametrize('entry', ['poll', 'replay'])
@pytest.mark.parametrize('saved_status', ['completed', 'running'])
async def test_recovered_receipt_remains_bound_after_repeated_poll(tmp_path, monkeypatch, entry, saved_status):
    key = 'fixture-recovery-key'
    headers = {'Authorization': 'Bearer ' + key, 'Idempotency-Key': 'restart-receipt'}
    body = {'input': 'previously accepted input'}
    run_id = 'run_recovery'
    path = tmp_path / 'receipts.db'

    def adapter():
        value = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': key}))
        value._run_idempotency_store.close()
        value._run_idempotency_store = RunIdempotencyStore(str(path))
        return value

    def request(method='GET', authorization=None):
        request_headers = dict(headers)
        if authorization is not None:
            request_headers['Authorization'] = authorization
        value = make_mocked_request(method, '/v1/runs' if method == 'POST' else f'/v1/runs/{run_id}',
                                    headers=request_headers, match_info={'run_id': run_id})
        value.json = AsyncMock(return_value=body)
        return value

    first = adapter()
    scope = first._run_idempotency_scope(request())
    first._run_idempotency_store.reserve(scope, headers['Idempotency-Key'], _run_fingerprint(body, None),
        run_id, {'run_id': run_id, 'status': saved_status, 'output': 'retained output'},
        owner_pid=0, owner_started=0)
    _close_run_state(first)
    restarted = adapter()

    def forbidden(*_args, **_kwargs):
        pytest.fail('receipt recovery must not start new execution')

    monkeypatch.setattr(restarted, '_create_agent', forbidden)
    monkeypatch.setattr(restarted, '_resolve_route', forbidden)
    try:
        denied = await restarted._handle_get_run(request(authorization='Bearer wrong-profile-key'))
        assert denied.status == 401
        assert run_id not in restarted._run_receipt_stores
        if entry == 'replay':
            replay = await restarted._handle_runs(request('POST'))
            assert replay.status == 202
            assert json.loads(replay.body)['replayed'] is True
        statuses = []
        for _ in range(2):
            response = await restarted._handle_get_run(request())
            assert response.status == 200
            statuses.append(json.loads(response.body))
        assert statuses[0] == statuses[1]
        assert statuses[0]['status'] == ('completed' if saved_status == 'completed' else 'interrupted')
        assert restarted._run_receipt_stores[run_id] is restarted._run_idempotency_store
        restarted._set_run_status(run_id, statuses[0]['status'], output='persisted after recovery')
        record = restarted._run_idempotency_store.status_for_run(scope, run_id)
        assert record['status']['output'] == 'persisted after recovery'
        assert restarted._active_run_tasks == {}
    finally:
        _close_run_state(restarted)
