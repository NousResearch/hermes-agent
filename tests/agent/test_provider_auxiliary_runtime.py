from providers import register_provider
from providers.base import ProviderProfile
from agent.auxiliary_client import _main_route_target

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread

import pytest


def test_virtual_profile_maps_auxiliary_target_without_leaking_transport_credentials():
    seen = []

    class VirtualProfile(ProviderProfile):
        def resolve_auxiliary_runtime(self, *, main_runtime, task=None):
            seen.append((dict(main_runtime), task))
            return {'provider': 'custom', 'model': 'acting-model'}

    register_provider(VirtualProfile(name='test-virtual-auxiliary'))
    runtime = {'provider': 'test-virtual-auxiliary', 'model': 'virtual-entry',
               'base_url': 'https://virtual.invalid', 'api_key': 'virtual-only-key',
               'api_mode': 'virtual-wire'}
    assert _main_route_target(runtime, 'compression') == ('custom', 'acting-model', '', '', '')
    assert seen == [(runtime, 'compression')]
    assert runtime['model'] == 'virtual-entry'


def test_inherited_noop_keeps_the_existing_main_runtime():
    register_provider(ProviderProfile(name='test-ordinary-auxiliary'))
    runtime = {'provider': 'test-ordinary-auxiliary', 'model': 'selected-model',
               'base_url': 'https://ordinary.invalid', 'api_key': 'own-key',
               'api_mode': 'chat_completions'}
    assert _main_route_target(runtime, 'compression') == (
        'test-ordinary-auxiliary', 'selected-model', 'https://ordinary.invalid', 'own-key', 'chat_completions')


@pytest.mark.parametrize('asynchronous', [False, True])
def test_virtual_vision_uses_own_credentials_and_preserves_explicit_override(tmp_path, monkeypatch, asynchronous):
    from agent.auxiliary_client import call_llm, async_call_llm

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('{}', encoding='utf-8')
    requests = []
    resolutions = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append((body, self.headers.get('Authorization')))
            payload = json.dumps({'id': 'fixture', 'object': 'chat.completion', 'created': 1,
                'model': body['model'], 'choices': [{'index': 0,
                    'message': {'role': 'assistant', 'content': 'IMAGE_OK'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f'http://127.0.0.1:{server.server_port}/v1'

    class VirtualVisionProfile(ProviderProfile):
        def resolve_auxiliary_runtime(self, *, main_runtime, task=None):
            resolutions.append(task)
            return {'provider': 'custom', 'model': 'vision-fixture', 'base_url': endpoint,
                    'api_key': 'target-only', 'api_mode': 'chat_completions'}

    register_provider(VirtualVisionProfile(name='test-virtual-vision'))
    runtime = {'provider': 'test-virtual-vision', 'model': 'virtual-model',
               'base_url': 'https://facade.invalid', 'api_key': 'must-not-leak'}
    image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,fixture'}}
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Describe this.'}, image]}]
    try:
        def invoke(**overrides):
            kwargs = dict(task='vision', messages=messages, main_runtime=runtime, max_tokens=12, timeout=5)
            kwargs.update(overrides)
            return asyncio.run(async_call_llm(**kwargs)) if asynchronous else call_llm(**kwargs)

        assert invoke().choices[0].message.content == 'IMAGE_OK'
        assert requests[-1][0]['model'] == 'vision-fixture'
        assert requests[-1][1] == 'Bearer target-only'
        assert requests[-1][0]['messages'] == messages
        assert resolutions == ['vision']
        assert invoke(provider='custom', model='manual-vision', base_url=endpoint,
                      api_key='manual-only').choices[0].message.content == 'IMAGE_OK'
        assert requests[-1][0]['model'] == 'manual-vision'
        assert requests[-1][1] == 'Bearer manual-only'
        assert resolutions == ['vision']
        assert len(requests) == 2
        assert runtime['api_key'] == 'must-not-leak'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
