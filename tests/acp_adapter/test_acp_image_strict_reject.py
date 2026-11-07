"""An ACP image part admission cannot carry degrades to a text note; the user's text still runs (B2-12)."""
import base64

import pytest
from acp.schema import ImageContentBlock, ResourceContentBlock, TextContentBlock

from acp_adapter.gateway_server import GatewayACPAgent

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32
BMP = b"BM" + b"0" * 64


class _Client:
    def __init__(self, agent):
        self.calls, self.agent = [], agent

    async def rpc(self, method, **params):
        self.calls.append((method, params))
        self.agent._terminals['adm-1'] = {}
        return {'admission_id': 'adm-1'}


def _blocks(tmp_path, case):
    if case == 'jpg-named-png':
        (tmp_path / 'shot.jpg').write_bytes(PNG)
        return [ResourceContentBlock(type='resource_link', uri=(tmp_path / 'shot.jpg').as_uri(), name='shot.jpg')]
    if case == 'bmp':
        (tmp_path / 'pic.bmp').write_bytes(BMP)
        return [ResourceContentBlock(type='resource_link', uri=(tmp_path / 'pic.bmp').as_uri(), name='pic.bmp')]
    if case == 'oversize':
        return [ImageContentBlock(type='image', data=base64.b64encode(PNG).decode(), mimeType='image/png')]
    return [ImageContentBlock(type='image', data=base64.b64encode(b'\xff\xd8\xff' + b'0' * 32).decode(), mimeType='')]


@pytest.mark.asyncio
@pytest.mark.parametrize('case, mime', [('jpg-named-png', 'image/png'), ('jpeg-no-mime', 'image/jpeg'),
                                        ('bmp', None), ('oversize', None)])
async def test_mislabeled_or_unadmittable_image_never_rejects_the_prompt(tmp_path, monkeypatch, case, mime):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    if case == 'oversize':
        monkeypatch.setattr('gateway.platforms.base.get_inbound_media_max_bytes', lambda: 8)
    agent = GatewayACPAgent()
    agent._gateway = _Client(agent)
    agent._snapshots['s'] = {}
    response = await agent.prompt([TextContentBlock(type='text', text='what is this?'), *_blocks(tmp_path, case)], 's')
    assert response.stop_reason == 'end_turn'
    ((method, params),) = agent._gateway.calls
    assert method == 'prompt.submit' and params['text'].startswith('what is this?')
    if mime:
        assert [a['mime'] for a in params['attachments']] == [mime]
    else:
        assert 'attachments' not in params and '[Image omitted:' in params['text']
