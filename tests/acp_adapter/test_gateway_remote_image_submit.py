"""A remote image URL is fetched once per logical prompt: an un-acked submit's retry reuses what the first
attempt staged (never a re-download that could change the content and so the input_id), and the editor's
Stop during the download keeps the prompt from being submitted at all (Greptile review on #140188)."""
import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from acp.schema import ImageContentBlock, TextContentBlock

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClientError
from tests.acp_adapter.test_gateway_prompt_content import _ONE_PX_PNG, _allow_loopback


@pytest.fixture
def host():
    """Loopback image host: the first GET of /pic serves a PNG, later ones 404 (a flaky or changed URL).
    ``gate`` (when set) holds every response until released (a slow download)."""
    state = {'hits': 0, 'gate': None, 'started': threading.Event()}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state['started'].set()
            if state['gate'] is not None:
                state['gate'].wait(10)
            state['hits'] += 1
            ok = state['hits'] == 1
            body = _ONE_PX_PNG if ok else b'gone'
            self.send_response(200 if ok else 404)
            self.send_header('Content-Type', 'image/png' if ok else 'text/plain')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state['url'] = f'http://127.0.0.1:{server.server_port}/pic'
    try:
        yield state
    finally:
        if state['gate'] is not None:
            state['gate'].set()
        server.shutdown()
        server.server_close()


class _LosesFirstAck:
    """The owner admits the first submit but its reply is lost (transport failure, not an RPC refusal)."""

    def __init__(self, agent):
        self.agent, self.calls = agent, []

    async def rpc(self, method, **params):
        self.calls.append((method, params))
        if len(self.calls) == 1:
            raise GatewayClientError('Gateway disconnected; turn outcome is unknown.')
        self.agent._terminals['adm-1'] = {}
        return {'admission_id': 'adm-1'}


def _prompt(url):
    return [TextContentBlock(type='text', text='describe'),
            ImageContentBlock(type='image', data='', uri=url, mimeType='image/png')]


@pytest.mark.asyncio
async def test_unacked_remote_image_retry_reuses_the_first_staging(tmp_path, monkeypatch, host):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _allow_loopback(monkeypatch, True)
    agent = GatewayACPAgent()
    agent._gateway = _LosesFirstAck(agent)
    agent._snapshots['s'] = {}
    with pytest.raises(GatewayClientError, match='outcome is unknown'):
        await agent.prompt(_prompt(host['url']), 's')
    await agent.prompt(_prompt(host['url']), 's')
    (_, first), (_, retry) = agent._gateway.calls
    # A second download would 404 and turn the image into URL text, a different prompt and a new input_id.
    assert host['hits'] == 1
    assert retry['input_id'] == first['input_id']
    assert retry['text'] == first['text'] == 'describe'
    assert retry['attachments'] == first['attachments']


@pytest.mark.asyncio
async def test_stop_during_a_remote_image_download_submits_nothing(tmp_path, monkeypatch, host):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    _allow_loopback(monkeypatch, True)
    host['gate'] = threading.Event()
    agent = GatewayACPAgent()
    agent._gateway = _LosesFirstAck(agent)
    agent._snapshots['s'] = {}
    task = asyncio.create_task(agent.prompt(_prompt(host['url']), 's'))
    assert await asyncio.to_thread(host['started'].wait, 10)
    await agent.cancel('s')
    host['gate'].set()
    response = await asyncio.wait_for(task, 10)
    assert response.stop_reason == 'cancelled'
    assert agent._gateway.calls == []
    assert 's' not in agent._pending_cancels and 's' not in agent._submitting
