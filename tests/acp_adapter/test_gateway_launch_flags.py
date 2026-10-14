"""`hermes acp` launch flags freeze into the gateway session they create (#134889 ACP-13)."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from acp_adapter.gateway_server import GatewayACPAgent

_ENV = ('HERMES_YOLO_MODE', 'HERMES_SAFE_MODE', 'HERMES_IGNORE_USER_CONFIG', 'HERMES_IGNORE_RULES',
        'HERMES_ACCEPT_HOOKS')


def _agent(created):
    async def rpc(method, **params):
        if method == 'runtime.describe':
            return {'session_create': {'sources': ['acp']}, 'capabilities': ['acp-editor-policy-v1']}
        assert method == 'session.create', method
        created.append(params)
        return {'session_id': 's', 'info': {'cwd': params['cwd']}}

    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    agent._gateway.rpc.side_effect = rpc
    agent._session_fields = AsyncMock(return_value={})
    return agent


@pytest.mark.asyncio
async def test_launch_env_rides_session_create(monkeypatch, tmp_path):
    for name in _ENV:
        monkeypatch.setenv(name, '1')
    created = []
    await _agent(created).new_session(cwd=str(tmp_path))
    assert {k: created[0].get(k) for k in ('yolo', 'safe_mode', 'ignore_user_config', 'ignore_rules',
                                          'accept_hooks')} == dict.fromkeys(
        ('yolo', 'safe_mode', 'ignore_user_config', 'ignore_rules', 'accept_hooks'), True)


@pytest.mark.asyncio
async def test_cli_only_acp_flags_reach_the_session(monkeypatch, tmp_path):
    """--accept-hooks / --ignore-rules have no other route from the `hermes acp` parser to the owner."""
    for name in _ENV:
        monkeypatch.setenv(name, '')  # recorded, so cmd_acp's exports are undone after the test
    import acp_adapter.entry
    from hermes_cli.main_agent_cmds import cmd_acp
    monkeypatch.setattr(acp_adapter.entry, 'main', lambda argv: None)
    cmd_acp(SimpleNamespace(accept_hooks=True, ignore_rules=True))
    created = []
    await _agent(created).new_session(cwd=str(tmp_path))
    assert created[0].get('accept_hooks') is True and created[0].get('ignore_rules') is True
    assert 'yolo' not in created[0] and 'safe_mode' not in created[0]
