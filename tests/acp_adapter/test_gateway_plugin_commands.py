"""Plugin slash commands reach ACP editors: advertised from the gateway's catalog, run by its slash.exec."""
from types import SimpleNamespace

import pytest

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayRPCError


class Client:
    def __init__(self, catalog):
        self.catalog, self.calls = catalog, []

    async def rpc(self, method, **params):
        self.calls.append((method, params))
        if method == 'commands.catalog':
            if self.catalog is None:
                raise GatewayRPCError('unknown_method')
            return self.catalog
        if method == 'slash.exec':
            return {'output': 'RAN ' + params['command']}
        return {}


def _agent(monkeypatch, tmp_path, catalog):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    agent, client, updates = GatewayACPAgent(), Client(catalog), []

    async def session_update(session_id, update):
        updates.append(update)

    async def _client():
        return client

    async def _project_usage(*args):
        return None

    agent._conn = SimpleNamespace(session_update=session_update)
    agent._client, agent._project_usage = _client, _project_usage
    return agent, client, updates


@pytest.mark.asyncio
async def test_plugin_commands_are_advertised_and_forwarded(monkeypatch, tmp_path):
    agent, client, updates = _agent(monkeypatch, tmp_path, {'plugin_commands': {'away': 'Mark the captain away'}})
    await agent._after_attach('s1', 's1')
    advertised = {c.name: c.description for c in updates[-1].available_commands}
    assert advertised['away'] == 'Mark the captain away' and 'help' in advertised

    answered, text = await agent._run_slash('s1', 's1', '/away going to bed')
    assert (answered, text) == (True, None)
    assert ('slash.exec', {'session_id': 's1', 'command': '/away going to bed'}) in client.calls
    assert updates[-1].content.text == 'RAN /away going to bed\n'


@pytest.mark.asyncio
async def test_unknown_words_and_older_gateways_stay_prompt_text(monkeypatch, tmp_path):
    agent, client, updates = _agent(monkeypatch, tmp_path, None)
    await agent._after_attach('s1', 's1')
    assert 'away' not in {c.name for c in updates[0].available_commands}
    assert await agent._run_slash('s1', 's1', '/away going to bed') == (False, '/away going to bed')
    assert not [call for call in client.calls if call[0] == 'slash.exec']
