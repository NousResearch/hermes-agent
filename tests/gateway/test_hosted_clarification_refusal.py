"""Persisted hosted policies cannot acknowledge a question without a reply channel."""
from dataclasses import replace

import pytest


@pytest.mark.asyncio
async def test_hosted_clarify_is_declined_before_waiting(owner, tmp_path):
    from gateway.session_local import LocalSessionAdapter
    from gateway.session_policy import build_policy
    adapter = LocalSessionAdapter(owner)
    adapter.policies['room'] = replace(build_policy({'source': 'cli', 'cwd': str(tmp_path), 'model': 'm'}, {}),
                                       source='bot_room', platform='bot_room')
    result = await adapter.send_clarify(chat_id='room', question='Which?', choices=['one', 'two'],
                                      clarify_id='question', session_key='route')
    assert result.success is False and result.error_kind == 'forbidden'
