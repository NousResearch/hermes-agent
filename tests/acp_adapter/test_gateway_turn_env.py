"""The ACP viewer forwards only declared per-turn values, retaining them across a lost ack."""
import asyncio
from contextlib import asynccontextmanager
import logging

import acp
import pytest

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClientError


class Wire:
    def __init__(self, home, capabilities):
        self.home, self.capabilities = home, capabilities
        self.events, self.submits = asyncio.Queue(), []
        self.describes, self.lose_ack = 0, False

    async def rpc(self, method, **params):
        if method == 'runtime.describe':
            self.describes += 1
            return {'profile_id': str(self.home), 'capabilities': self.capabilities}
        if method == 'session.resume':
            return {'session_id': 's', 'pending': []}
        if method == 'config.set':
            return {}
        if method == 'prompt.submit':
            self.submits.append(params)
            if self.lose_ack:
                self.lose_ack = False
                raise GatewayClientError('Gateway disconnected; turn outcome is unknown.')
            return {'admission_id': params['input_id'], 'status': 'terminal', 'outcome': 'completed'}
        if method == 'prompt.receipt':
            return {'result': {'final_response': 'done'}}
        raise AssertionError(method)


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    for name in ('HERMES_SAFE_MODE', 'HERMES_IGNORE_USER_CONFIG', 'JOB_RUN_ID', 'JOB_API_KEY', 'JOB_MISSING'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr('hermes_cli.config_effective.load_user_config_effective', lambda path: {
        'terminal': {'client_env_turn_passthrough': ['JOB_RUN_ID', 'JOB_API_KEY', 'JOB_MISSING']}})
    agent = GatewayACPAgent()
    wire = Wire(agent._home, ['turn-env-v1'])

    @asynccontextmanager
    async def connect():
        yield wire

    monkeypatch.setattr('acp_adapter.gateway_server.connect_gateway', connect)
    agent._snapshots['s'] = {'pending': []}
    return agent, wire


async def send(agent, text='hello'):
    return await asyncio.wait_for(agent.prompt([acp.text_block(text)], 's'), 5)


@pytest.mark.asyncio
@pytest.mark.parametrize('capabilities, supported', [
    (['turn-env-v1'], True), ([], False), (None, False), ('turn-env-v1', False)])
async def test_capability_gates_exact_declared_viewer_values(viewer, monkeypatch, caplog, capabilities, supported):
    agent, wire = viewer
    wire.capabilities = capabilities
    monkeypatch.setenv('JOB_RUN_ID', 'private-run-one')
    monkeypatch.setenv('JOB_API_KEY', 'private-run-credential')
    monkeypatch.setenv('JOB_UNDECLARED', 'private-undeclared')
    try:
        with caplog.at_level(logging.DEBUG):
            await send(agent)
        params = wire.submits[0]
        if supported:
            assert params['turn_env'] == {'JOB_RUN_ID': 'private-run-one', 'JOB_API_KEY': 'private-run-credential'}
        else:
            assert 'turn_env' not in params
        assert 'private-' not in caplog.text
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_each_turn_reads_current_viewer_values_and_omits_unexported_names(viewer, monkeypatch):
    agent, wire = viewer
    try:
        for value in ('run-one', 'run-two', None):
            if value is None:
                monkeypatch.delenv('JOB_RUN_ID')
            else:
                monkeypatch.setenv('JOB_RUN_ID', value)
            await send(agent)
        assert [params.get('turn_env') for params in wire.submits] == [
            {'JOB_RUN_ID': 'run-one'}, {'JOB_RUN_ID': 'run-two'}, None]
        assert wire.describes == 1
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('original', ['original-private-run', None])
async def test_unacked_retry_keeps_original_values_across_reconnect(viewer, monkeypatch, caplog, original):
    agent, wire = viewer
    wire.lose_ack = True
    if original is not None:
        monkeypatch.setenv('JOB_RUN_ID', original)
    try:
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(GatewayClientError, match='outcome is unknown') as error:
                await send(agent)
            await agent._teardown()
            monkeypatch.setenv('JOB_RUN_ID', 'replacement-private-run')
            await send(agent)
        assert wire.submits[0] == wire.submits[1]
        assert wire.submits[0].get('turn_env') == ({'JOB_RUN_ID': original} if original else None)
        assert 'private-run' not in caplog.text + str(error.value)
        assert wire.describes == 2
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/queue work', '/steer work'])
async def test_command_that_becomes_a_prompt_carries_turn_values(viewer, monkeypatch, command):
    agent, wire = viewer
    monkeypatch.setenv('JOB_RUN_ID', 'command-run')
    try:
        await send(agent, command)
        assert wire.submits[0]['text'] == 'work'
        assert wire.submits[0]['turn_env'] == {'JOB_RUN_ID': 'command-run'}
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('flag', ['HERMES_SAFE_MODE', 'HERMES_IGNORE_USER_CONFIG'])
async def test_bypass_launch_reads_no_profile_and_sends_no_values(viewer, monkeypatch, flag):
    agent, wire = viewer
    monkeypatch.setenv(flag, '1')
    monkeypatch.setenv('JOB_RUN_ID', 'bypassed-run')

    def never(path):
        raise AssertionError('bypass launch must not read profile config')

    monkeypatch.setattr('hermes_cli.config_effective.load_user_config_effective', never)
    try:
        await send(agent)
        assert 'turn_env' not in wire.submits[0]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('supported_first', [False, True])
async def test_reconnect_uses_replacement_owners_capability(viewer, monkeypatch, supported_first):
    agent, wire = viewer
    monkeypatch.setenv('JOB_RUN_ID', 'reconnected-run')
    try:
        for supported in (supported_first, not supported_first):
            wire.capabilities = ['turn-env-v1'] if supported else []
            await send(agent)
            assert ('turn_env' in wire.submits[-1]) is supported
            await agent._teardown()
        assert wire.describes == 2
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_local_slash_does_not_collect_or_submit_turn_values(viewer, monkeypatch):
    agent, wire = viewer

    def never():
        raise AssertionError('local slash must not collect turn values')

    monkeypatch.setattr('hermes_cli.gateway_chat.shell_turn_env', never)
    try:
        await send(agent, '/queue')
        assert wire.submits == []
        assert wire.describes == 0
    finally:
        await agent.aclose()
