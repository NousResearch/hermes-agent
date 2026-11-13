"""In-band queued turns must not attribute dispatch to the opener's admission."""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from gateway.session_ingress import dispatch_admission_scope, _dispatch_admission
from tests.gateway.test_queued_final_ledger import _chain_runner_and_ctx
from tools import async_delegation as delegation


@pytest.mark.parametrize('has_event', [True, False])
def test_recursive_followup_dispatch_does_not_borrow_opener(tmp_path, monkeypatch, has_event):
    asyncio.run(_exercise(tmp_path, monkeypatch, has_event))


async def _exercise(tmp_path, monkeypatch, has_event):
    monkeypatch.setattr(delegation, '_db_path', lambda: tmp_path / 'state.db')
    GatewayRunner, runner, ctx, event = _chain_runner_and_ctx({})
    ctx.channel_prompt = None
    # These untrusted attributes must not be used as admission authority.
    event._origin_admission = 'forged'
    event._origin_owner = 'sid'
    async def recursive(**kwargs):
        record = {'delegation_id': 'child', 'session_key': ctx.session_key, 'dispatched_at': 1}
        delegation._persist_dispatch(record)
        return {'final_response': 'done', 'messages': []}
    runner._run_agent = AsyncMock(side_effect=recursive)
    with dispatch_admission_scope('sid', 'opener'):
        await GatewayRunner._run_agent_queued_followup(
            runner, ctx, None, 'followup', event if has_event else None, 'response',
            {'interrupted': True, 'messages': []}, None)
        assert _dispatch_admission.get() == ('sid', 'opener')
    with delegation._connect() as conn:
        payload = json.loads(conn.execute('SELECT task_json FROM async_delegations WHERE delegation_id=?', ('child',)).fetchone()[0])
    assert '_origin_admission' not in payload
    assert '_origin_owner' not in payload
