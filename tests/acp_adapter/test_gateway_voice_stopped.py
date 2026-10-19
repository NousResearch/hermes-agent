"""TUI-26: the owner's canonical prompt.submit may answer ``{voice_stopped: true}`` (a bare stop
phrase typed into a LOCAL session whose owner /voice mode is on: the voice chat ends and nothing is
admitted). The ACP adapter settles that prompt as an ended turn with the stop notice instead of
reading a missing ``admission_id``."""
import pytest
from acp.schema import TextContentBlock

from acp_adapter.gateway_server import GatewayACPAgent


class _VoiceStoppedClient:
    def __init__(self):
        self.calls = []

    async def rpc(self, method, **params):
        self.calls.append((method, params))
        if method == 'prompt.submit':
            return {'voice_stopped': True}
        return {}


class _Conn:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update):
        self.updates.append((session_id, update))


@pytest.mark.asyncio
async def test_acp_prompt_settles_a_voice_stopped_reply_as_an_ended_turn(tmp_path, monkeypatch):
    from agent.i18n import t
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    agent = GatewayACPAgent()
    agent._gateway = _VoiceStoppedClient()
    agent._conn = _Conn()
    agent._snapshots['s'] = {}
    response = await agent.prompt([TextContentBlock(type='text', text='stop')], 's')
    assert response.stop_reason == 'end_turn'
    assert [m for m, _ in agent._gateway.calls] == ['prompt.submit']
    assert 's' not in agent._admissions and not agent._inflight
    texts = [getattr(getattr(u, 'content', None), 'text', '') for _, u in agent._conn.updates]
    assert any(t('cli.voice.stop_phrase_typed') in text for text in texts), texts
