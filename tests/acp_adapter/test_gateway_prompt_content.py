"""ACP prompt blocks reach canonical admission with the same conversion main's server uses."""
import base64

import pytest
from acp.schema import ImageContentBlock, TextContentBlock

from acp_adapter.gateway_server import GatewayACPAgent

_ONE_PX_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)


class _Client:
    def __init__(self, agent):
        self.calls = []
        self.agent = agent

    async def rpc(self, method, **params):
        self.calls.append((method, params))
        self.agent._terminals['adm-1'] = {}
        return {'admission_id': 'adm-1'}


@pytest.mark.asyncio
async def test_image_prompt_is_staged_in_profile_cache_and_submitted_as_attachment(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from gateway.platforms.base import get_image_cache_dir
    agent = GatewayACPAgent()
    agent._gateway = _Client(agent)
    agent._snapshots['s'] = {}
    response = await agent.prompt([
        TextContentBlock(type='text', text='What is in this image?'),
        ImageContentBlock(type='image', data=base64.b64encode(_ONE_PX_PNG).decode(), mimeType='image/png'),
    ], 's')
    assert response.stop_reason == 'end_turn'
    ((method, params),) = agent._gateway.calls
    assert method == 'prompt.submit'
    assert params['text'] == 'What is in this image?'
    (attachment,) = params['attachments']
    assert attachment['mime'] == 'image/png'
    staged = tmp_path.joinpath(attachment['path'])
    assert staged.parent == get_image_cache_dir().resolve()
    assert staged.read_bytes() == _ONE_PX_PNG


@pytest.mark.asyncio
async def test_text_only_prompt_keeps_plain_submit(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    agent = GatewayACPAgent()
    agent._gateway = _Client(agent)
    agent._snapshots['s'] = {}
    await agent.prompt([TextContentBlock(type='text', text='hello')], 's')
    ((_, params),) = agent._gateway.calls
    assert params['text'] == 'hello' and 'attachments' not in params
