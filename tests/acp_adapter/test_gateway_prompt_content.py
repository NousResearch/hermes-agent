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


@pytest.fixture
def image_host():
    """A real HTTP server on loopback serving one PNG (the editor's remote image URL)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = _ONE_PX_PNG if self.path == '/pic' else b'<html>not found</html>'
            self.send_response(200 if self.path == '/pic' else 404)
            self.send_header('Content-Type', 'image/png' if self.path == '/pic' else 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()


def _allow_loopback(monkeypatch, allowed):
    from tools import url_safety
    monkeypatch.setenv('HERMES_ALLOW_PRIVATE_URLS', 'true' if allowed else 'false')
    url_safety._reset_allow_private_cache()


@pytest.mark.asyncio
async def test_remote_image_url_reaches_admission_as_an_image_attachment(tmp_path, monkeypatch, image_host):
    """#134889 ACP-16: main's in-process server handed the model a remote image as an image part; the
    gateway path admits only staged bytes, so the adapter fetches it (SSRF-guarded) and stages it."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _allow_loopback(monkeypatch, True)
    agent = GatewayACPAgent()
    agent._gateway = _Client(agent)
    agent._snapshots['s'] = {}
    await agent.prompt([TextContentBlock(type='text', text='describe'),
                        ImageContentBlock(type='image', data='', uri=f'{image_host}/pic', mimeType='image/png')], 's')
    ((_, params),) = agent._gateway.calls
    assert params['text'] == 'describe'
    (attachment,) = params['attachments']
    assert attachment['mime'] == 'image/png'
    from pathlib import Path
    assert Path(attachment['path']).read_bytes() == _ONE_PX_PNG


@pytest.mark.asyncio
@pytest.mark.parametrize('path, allowed', [('/pic', False), ('/missing', True)], ids=['ssrf-blocked', 'http-404'])
async def test_a_remote_image_that_cannot_be_fetched_stays_a_text_note(tmp_path, monkeypatch, image_host, path, allowed):
    """A private address (SSRF guard) or a failed download never fails the prompt: the model still gets
    the URL as text, as before."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _allow_loopback(monkeypatch, allowed)
    agent = GatewayACPAgent()
    agent._gateway = _Client(agent)
    agent._snapshots['s'] = {}
    await agent.prompt([TextContentBlock(type='text', text='describe'),
                        ImageContentBlock(type='image', data='', uri=f'{image_host}{path}', mimeType='image/png')], 's')
    ((_, params),) = agent._gateway.calls
    assert 'attachments' not in params
    assert params['text'].startswith('describe\n[Image attached: ') and f'{image_host}{path}' in params['text']
