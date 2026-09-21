"""The catalog path must honor allocated llama.cpp metadata without props."""
import json
import threading

import pytest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def test_allocated_context_from_catalog():
    from agent.model_metadata import get_model_context_length

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/v1/models':
                payload = {'data': [{'id': 'fixture-model', 'owned_by': 'llamacpp',
                    'context_length': 400000,
                    'meta': {'n_ctx': 200192, 'n_ctx_train': 262144}}]}
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
            else:
                body = b'not found'
                self.send_response(404)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = get_model_context_length('fixture-model',
            base_url=f'http://127.0.0.1:{server.server_port}/v1',
            provider='custom', custom_providers=[])
        assert result == 200192
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize('meta', [None, {}, [], {'n_ctx': 0}, {'n_ctx': -1}, {'n_ctx': True}, {'n_ctx': 'bad'}])
def test_unusable_runtime_metadata_preserves_advertised_window(meta):
    from agent.model_metadata import _context_length_from_model_payload
    assert _context_length_from_model_payload({'context_length': 12345, 'meta': meta}) == 12345


def test_empty_and_unrelated_nested_values_are_not_windows():
    from agent.model_metadata import _context_length_from_model_payload as read
    assert read({}) is None
    assert read({'other': {'n_ctx': 999}}) is None
    assert read({'max_tokens': 321}) == 321


def test_runtime_metadata_matches_local_sibling():
    from agent.model_metadata import _context_length_from_model_payload, _openai_models_list_context
    from unittest.mock import Mock
    payload = {'id': 'fixture-model', 'context_length': 400000,
               'meta': {'n_ctx': 200192, 'n_ctx_train': 262144}}
    client = Mock()
    client.get.return_value.status_code = 200
    client.get.return_value.json.return_value = {'data': [payload]}
    assert _context_length_from_model_payload(payload) == _openai_models_list_context(client, 'http://fixture', 'fixture-model') == 200192


def test_explicit_configuration_still_wins():
    from agent.model_metadata import get_model_context_length
    assert get_model_context_length('fixture-model', base_url='http://127.0.0.1:1/v1',
                                    config_context_length=8192, provider='custom') == 8192

