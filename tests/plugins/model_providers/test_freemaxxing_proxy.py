"""Real loopback dispatch contracts; no external inference or account mutation."""
from __future__ import annotations

import importlib
import json
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PLUGIN_DIR = ROOT / 'plugins' / 'model-providers' / 'freemaxxing'
# Test the packaged implementation directly. Registration/loader composition is
# covered separately through Hermes' actual provider loader, not a core patch.
NAME = '_freemaxxing_contract'
package = types.ModuleType(NAME)
package.__path__ = [str(PLUGIN_DIR)]
sys.modules.setdefault(NAME, package)
POLICY = importlib.import_module(NAME + '.policy')
POOL = importlib.import_module(NAME + '.pool')
SERVER = importlib.import_module(NAME + '.server')
PROTOCOL = importlib.import_module(NAME + '.protocol')
UPSTREAM = importlib.import_module(NAME + '.upstream')
TRANSPORT = importlib.import_module(NAME + '.transport')
TOKEN = 'test-only-freemaxxing-capability'


def completion(text='ok', **message):
    return {'id': 'completion-1', 'object': 'chat.completion', 'model': 'test-model',
            'choices': [{'index': 0, 'finish_reason': 'stop',
                         'message': {'role': 'assistant', 'content': text, **message}}]}


def event(delta=None, finish=None, **extra):
    return b'data: ' + json.dumps({'id': 'completion-1', 'choices': [
        {'index': 0, 'delta': delta or {}, 'finish_reason': finish}], **extra}).encode() + b'\n\n'


def valid_stream(text='ok'):
    return [event({'role': 'assistant'}), event({'content': text}),
            event(finish='stop'), b'data: [DONE]\n\n']


class Mock:
    def __init__(self, *, status=200, raw=None, stream=None, rows=None,
                 retry_after='1', delay=0, catalog_status=200, responder=None):
        self.status, self.raw, self.stream = status, raw, stream
        self.rows = rows if rows is not None else [{'id': 'test-model'}]
        self.retry_after, self.delay = retry_after, delay
        self.catalog_status, self.responder = catalog_status, responder
        self.calls, self.gets = [], 0
        self.port_ids = []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *_args):
                pass

            def do_GET(self):  # noqa: N802
                outer.gets += 1
                data = json.dumps({'data': outer.rows}).encode()
                self.send_response(outer.catalog_status)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(length))
                outer.calls.append((body, dict(self.headers)))
                outer.port_ids.append(self.client_address[1])
                status, raw, chunks = outer.status, outer.raw, outer.stream
                if outer.responder:
                    status, raw, chunks = outer.responder(body, dict(self.headers))
                if status == 0:
                    self.close_connection = True
                    return
                if outer.delay:
                    time.sleep(outer.delay)
                try:
                    self.send_response(status)
                    if status == 429:
                        self.send_header('Retry-After', outer.retry_after)
                    if 300 <= status < 400:
                        self.send_header('Location', 'http://127.0.0.1:1/stolen')
                    if chunks is not None:
                        self.send_header('Content-Type', 'text/event-stream')
                        self.send_header('Connection', 'close')
                        self.end_headers()
                        for chunk in chunks:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        self.close_connection = True
                    else:
                        if raw is None:
                            raw = json.dumps(completion()).encode()
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Content-Length', str(len(raw)))
                        self.end_headers()
                        self.wfile.write(raw)
                except OSError:
                    pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={'poll_interval': 0.02}, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server.server_address[1]}'

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(1)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv('NO_PROXY', '127.0.0.1,::1')
    owner = POOL.BackendPool(POLICY.Limits(connect=.3, read=.4, total=3, catalog=.3))
    mocks = []
    SERVER.stop_proxy()

    def add(*, name=None, kind='local', rows=None, key='key', tier=0, **kwargs):
        mock = Mock(rows=rows, **kwargs)
        mocks.append(mock)
        backend = POOL.Backend(name or f'backend-{len(mocks)}', mock.url,
                               api_key=key, kind=kind, tier=tier, free_tier_only=(kind == 'nous-portal'))
        backend.set_catalog(mock.rows, ttl=3600)
        owner.add(backend)
        return mock, backend

    server = SERVER.spawn_proxy(port=0, token=TOKEN, backend_pool=owner)
    yield types.SimpleNamespace(owner=owner, server=server, add=add)
    SERVER.stop_proxy(server)
    for mock in mocks:
        mock.close()


def request(env, body=None, *, token=TOKEN, path='/v1/chat/completions'):
    headers = {} if token is None else {'Authorization': f'Bearer {token}'}
    if body is None:
        body = {'model': 'freemaxxing', 'messages': [{'role': 'user', 'content': 'hi'}]}
    raw = json.dumps(body).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{env.server.server_address[1]}{path}',
                                 data=raw, headers={'Content-Type': 'application/json', **headers})
    return urllib.request.urlopen(req, timeout=5)


def body(**extra):
    return {'model': 'freemaxxing', 'messages': [{'role': 'user', 'content': 'hi'}], **extra}


def test_authenticated_public_dispatch(env):
    mock, _ = env.add()
    with request(env) as result:
        assert json.load(result)['choices'][0]['message']['content'] == 'ok'
    assert len(mock.calls) == 1


@pytest.mark.parametrize('token', [None, '', 'wrong', 'é'])
def test_unauthorized_cannot_call_upstream(env, token):
    mock, _ = env.add()
    with pytest.raises(urllib.error.HTTPError) as error:
        request(env, token=token)
    assert error.value.code == 401
    assert not mock.calls and not mock.gets


def test_minimal_health_and_authenticated_metadata_do_not_initialize(env):
    env.server.pool_ready = False
    env.server.pool_initializer = lambda: pytest.fail('metadata cannot initialize credentials')
    for path, token in [('/v1/healthz', None), ('/v1/models', TOKEN)]:
        headers = {'Authorization': f'Bearer {token}'} if token else {}
        req = urllib.request.Request(f'http://127.0.0.1:{env.server.server_address[1]}{path}', headers=headers)
        with urllib.request.urlopen(req) as response:
            data = json.load(response)
        assert ('health' not in data)
    assert not env.server.pool_ready


def test_runtime_guard_rechecked_after_initialization(env):
    mock, _ = env.add()
    env.server.runtime_guard = lambda: (_ for _ in ()).throw(RuntimeError('multiplex'))
    with pytest.raises(urllib.error.HTTPError) as error:
        request(env)
    assert error.value.code == 409 and not mock.calls


@pytest.mark.parametrize('status', [0, 429, 500, 503, 307])
def test_transport_failures_recover_on_another_provider(env, status):
    first, _ = env.add(status=status)
    second, _ = env.add()
    with request(env) as response:
        assert json.load(response)['id'] == 'completion-1'
    assert len(first.calls) == len(second.calls) == 1


@pytest.mark.parametrize('raw', [b'{bad', b'\xff', b'[]', b'{}', b'{"error": {"code": 500}}',
                                b'{"choices": []}', b'{"x": NaN}'])
def test_invalid_http_200_cannot_end_turn(env, raw):
    first, _ = env.add(raw=raw)
    second, _ = env.add()
    with request(env) as response:
        assert json.load(response)['id'] == 'completion-1'
    assert first.calls and second.calls


def test_model_failure_does_not_discard_other_models(env):
    rows = [{'id': 'bad'}, {'id': 'good'}]
    def responder(payload, _headers):
        return (404, b'{}', None) if payload['model'] == 'bad' else (200, None, None)
    mock, backend = env.add(rows=rows, responder=responder)
    with request(env) as response:
        json.load(response)
    assert [c[0]['model'] for c in mock.calls] == ['bad', 'good']
    assert backend.is_available()
    assert 'bad' not in backend.available_rows()


def test_account_429_does_not_try_other_models_in_same_quota(env):
    mock, backend = env.add(status=429, retry_after='86400', rows=[{'id': 'a'}, {'id': 'b'}])
    env.add()
    with request(env) as response:
        json.load(response)
    assert len(mock.calls) == 1
    assert backend.health()['retry_after'] > 86390
    backend.record_success(model='a', elapsed=1)
    assert not backend.is_available()


def test_caller_error_is_not_replayed(env):
    first, _ = env.add(status=400, raw=b'{"error":"bad messages"}')
    second, _ = env.add()
    with pytest.raises(urllib.error.HTTPError) as error:
        request(env)
    assert error.value.code == 400 and len(first.calls) == 1 and not second.calls


@pytest.mark.parametrize('text', ['unsupported tool_choice', 'context length exceeded', 'invalid model'])
def test_route_capability_errors_fail_over(env, text):
    env.add(status=400, raw=json.dumps({'error': text}).encode())
    second, _ = env.add()
    with request(env) as response:
        json.load(response)
    assert second.calls


def test_keyless_opencode_never_receives_real_key(env):
    mock, _ = env.add(kind='opencode-free', key='must-not-leak', rows=[{'id': 'mimo-v2.5-free'}])
    with request(env) as response:
        json.load(response)
    assert 'Authorization' not in mock.calls[0][1]
    assert mock.calls[0][1]['X-Title'] == 'Hermes Agent'


@pytest.mark.parametrize('model', ['paid/model', 'deepseek/deepseek-v4-flash-0731'])
def test_explicit_paid_model_cannot_bypass_admission(env, model):
    mock, _ = env.add(kind='openrouter', rows=[{'id': 'free/model:free'}, {'id': model}])
    with pytest.raises(urllib.error.HTTPError) as error:
        request(env, body(model=model))
    assert error.value.code == 503 and not mock.calls


def test_openrouter_zero_price_guard_replaces_caller_routing(env):
    mock, _ = env.add(kind='openrouter', rows=[{'id': 'model:free'}])
    with request(env, body(provider={'max_price': {'prompt': 20}, 'order': ['paid']})) as response:
        json.load(response)
    routing = mock.calls[0][0]['provider']
    assert set(routing['max_price'].values()) == {0}
    assert 'order' not in routing and routing['require_parameters']


def test_nous_requires_current_zero_price_not_old_model_name(env):
    rows = [{'id': 'deepseek/deepseek-v4-flash-0731'},
            {'id': 'free-model', 'pricing': {'prompt': '0', 'completion': '0'}}]
    mock, backend = env.add(kind='nous-portal', rows=rows)
    with request(env) as response:
        json.load(response)
    assert mock.calls[0][0]['model'] == 'free-model'
    backend.cached_models_until = 0
    assert not backend.available_rows()


@pytest.mark.parametrize('value', ['0.01', '-1', 'nan', 'Infinity', None, True, 'unknown'])
def test_ambiguous_or_nonzero_prices_are_excluded(value):
    backend = POOL.Backend('nous-portal', 'https://example.invalid', free_tier_only=True)
    backend.set_catalog([{'id': 'x', 'pricing': {'prompt': value, 'completion': '0'}}])
    assert not backend.available_rows()


def test_conflicting_duplicate_catalog_is_order_independent():
    backend = POOL.Backend('nous-portal', 'https://example.invalid', free_tier_only=True)
    free = {'id': 'x', 'pricing': {'prompt': 0, 'completion': 0}}
    paid = {'id': 'x', 'pricing': {'prompt': 1, 'completion': 1}}
    for rows in ([free, paid], [paid, free]):
        backend.set_catalog(rows)
        assert not backend.available_rows()


@pytest.mark.parametrize('field', ['models', 'plugins', 'transforms', 'web_search_options'])
def test_paid_extension_rejected_before_effect(env, field):
    mock, _ = env.add()
    with pytest.raises(urllib.error.HTTPError) as error:
        request(env, body(**{field: ['anything']}))
    assert error.value.code == 400 and not mock.calls


@pytest.mark.parametrize('chunks', [[], [b': keepalive\n\n'], [b'data: not-json\n\n'],
    [event({'content': 'uncommitted'})], [event({'role': 'assistant'}), b'data: [DONE]\n\n'],
    [event({'content': 'uncommitted'}), b'data: {"error":{"code":503}}\n\n'],
    [event({'tool_calls': [{'index': 0, 'id': 't', 'type': 'function',
                           'function': {'name': 'write', 'arguments': '{'}}]}),
     event(finish='tool_calls'), b'data: [DONE]\n\n']])
def test_any_upstream_stream_failure_recovers_before_commit(env, chunks):
    first, _ = env.add(stream=chunks)
    second, _ = env.add(stream=valid_stream('winner'))
    with request(env, body(stream=True)) as response:
        raw = response.read()
    assert b'winner' in raw and b'uncommitted' not in raw
    assert b'write' not in raw and raw.count(b'data: [DONE]') == 1
    assert len(first.calls) == len(second.calls) == 1


def test_fragmented_multiline_sse_with_tool_only_output(env):
    tool = {'index': 0, 'id': 't', 'type': 'function', 'function': {'name': 'read', 'arguments': '{'}}
    raw = (event({'tool_calls': [tool]}) +
           event({'tool_calls': [{'index': 0, 'function': {'arguments': '"path":"x"}'}}]}) +
           event(finish='tool_calls') + b'data: [DONE]\n\n')
    env.add(stream=[raw[i:i+7] for i in range(0, len(raw), 7)])
    with request(env, body(stream=True)) as response:
        assert response.read() == raw


def test_stream_stops_exactly_at_terminal_event(env):
    raw = b''.join(valid_stream())
    env.add(stream=[raw + b'data: {"error":"later-generation"}\n\n'])
    with request(env, body(stream=True)) as response:
        assert response.read() == raw


def test_sse_total_limit_applies_to_keepalives(env, monkeypatch):
    monkeypatch.setattr(PROTOCOL, '_MAX_RESPONSE_BODY_BYTES', 1024)
    env.add(stream=[b':' + b'a' * 2000 + b'\n\n'])
    env.add(stream=valid_stream())
    with request(env, body(stream=True)) as response:
        assert b'data: [DONE]' in response.read()


def test_complete_tool_call_is_preserved(env):
    tool = {'id': 'read-1', 'type': 'function', 'function': {'name': 'read', 'arguments': '{"path":"x"}'}}
    env.add(raw=json.dumps(completion(None, tool_calls=[tool])).encode())
    with request(env) as response:
        assert json.load(response)['choices'][0]['message']['tool_calls'] == [tool]


def test_tool_argument_truncation_does_not_escape(env):
    tool = {'id': 'write-1', 'type': 'function', 'function': {'name': 'write', 'arguments': '{'}}
    env.add(raw=json.dumps(completion(None, tool_calls=[tool])).encode())
    env.add()
    with request(env) as response:
        assert json.load(response)['choices'][0]['message']['content'] == 'ok'


def test_empty_catalog_is_cached_without_refetch_loop(env):
    mock, backend = env.add(rows=[])
    for _ in range(5):
        env.owner.refresh_catalogs()
    assert backend.cached_models == [] and mock.gets == 0


def test_catalog_failure_does_not_disable_known_safe_routes(env):
    mock, backend = env.add(kind='openrouter', rows=[{'id': 'model:free'}], catalog_status=503)
    backend.cached_models_until = 0
    env.owner.refresh_catalogs()
    for _ in range(100):
        if backend.catalog_retry_until:
            break
        time.sleep(.01)
    assert backend.is_available() and 'model:free' in backend.available_rows()
    assert mock.gets == 1


def test_catalog_refresh_singleflight(env):
    mock, backend = env.add()
    backend.cached_models_until = 0
    threads = [threading.Thread(target=env.owner.refresh_catalogs) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    env.owner.wait_for_catalog(POLICY.Budget(1))
    assert mock.gets == 1


def test_connection_reuse(env):
    mock, _ = env.add()
    for _ in range(3):
        with request(env) as response:
            response.read()
    assert len(set(mock.port_ids)) == 1


def test_session_affinity_is_bounded_and_never_forwarded(env):
    first, _ = env.add()
    with request(env, body(freemaxxing_session='session-a')) as response:
        response.read()
    assert 'freemaxxing_session' not in first.calls[0][0]
    for i in range(1500):
        env.owner.remember(f's-{i}', ('b', 'm'))
    assert len(env.owner._affinity) == 1024


def test_capacity_exhaustion_is_typed_not_queued(env):
    for _ in range(env.owner.limits.concurrency):
        env.server.inference_slots.acquire()
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            request(env)
        assert error.value.code == 503
        assert json.load(error.value)['error']['type'] == 'capacity'
    finally:
        for _ in range(env.owner.limits.concurrency):
            env.server.inference_slots.release()


def test_exhaustion_returns_retryable_error_never_done_marker(env):
    with pytest.raises(urllib.error.HTTPError) as error:
        request(env, body(stream=True))
    assert error.value.code == 503
    result = json.load(error.value)
    assert result['error']['retryable'] and not result['error']['router_state_mutated']
    assert 'choices' not in result


def test_same_listener_cannot_change_capability_or_pool(env):
    with pytest.raises(ValueError):
        SERVER.spawn_proxy(port=0, token=' ')
    with pytest.raises(RuntimeError):
        SERVER.spawn_proxy(port=0, token='other')
    with pytest.raises(RuntimeError):
        SERVER.spawn_proxy(port=0, token=TOKEN, backend_pool=POOL.BackendPool())


@pytest.mark.parametrize('raw,expected', [('nan',30),('inf',30),('-10',0),('9999',9999),('86400',86400)])
def test_retry_after_does_not_shorten_provider_reset(raw, expected):
    assert POLICY._parse_retry_after({'Retry-After': raw}) == expected


@pytest.mark.parametrize('url', ['https://example.com', 'http://localhost:1234',
                                 'http://127.0.0.1.evil.test', 'http://127.0.0.1@evil.test'])
def test_local_provider_cannot_be_remote(url):
    with pytest.raises(POLICY.FreePolicyError):
        POLICY.validate_url(url, local=True)


def test_untrusted_provider_is_not_implicitly_free():
    backend = POOL.Backend('unknown', 'https://example.com')
    assert not POLICY._accept_catalog_id(backend, 'free-model')


def test_native_responses_opencode_sku_is_not_misrouted():
    backend = POOL.Backend('opencode-free', 'https://opencode.ai/zen/v1')
    assert not POLICY._accept_catalog_id(backend, 'muse-spark-1.3-contributor-free')
    assert not POLICY._accept_catalog_id(backend, 'mimo-v2.5-free', {'api_mode': 'responses'})
    assert POLICY._accept_catalog_id(backend, 'new-free', {'api_mode': 'chat_completions'})


def test_nous_promotional_original_price_is_not_current_price():
    backend = POOL.Backend('nous-portal', 'https://example.invalid', free_tier_only=True)
    row = {'id': 'promotional', 'pricing': {'prompt': '0', 'completion': '0',
                                          'original': {'prompt': '1', 'completion': '2'}}}
    backend.set_catalog([row])
    assert backend.get_cached_models() == ['promotional']


def test_nous_catalog_price_without_account_boundary_is_not_authority():
    backend = POOL.Backend('nous-portal', 'https://example.invalid')
    backend.set_catalog([{'id': 'x', 'pricing': {'prompt': 0, 'completion': 0}}])
    assert not backend.available_rows()


def test_scoped_auto_selector_keeps_moa_advisors_on_distinct_providers(env):
    first, _ = env.add(name='first')
    second, _ = env.add(name='second')
    with request(env, body(model='second::freemaxxing')) as response:
        json.load(response)
    assert not first.calls and len(second.calls) == 1


def test_stale_catalog_cannot_publish_after_credential_rotation(env):
    _, backend = env.add()
    old = backend.credential_snapshot()
    backend.refresh = lambda: (backend.base_url, 'new-key')
    assert POOL._refresh_backend_credentials(backend, require_new=True, observed=old)
    backend.set_catalog([{'id': 'old-account-only'}], credentials=old)
    assert not backend.available_rows()


def test_auth_recovery_revalidates_catalog_for_new_credential(env):
    def responder(_body, headers):
        return (401, b'{}', None) if headers.get('Authorization') == 'Bearer old' else (200, None, None)
    mock, backend = env.add(key='old', responder=responder)
    calls = []
    def refresh():
        calls.append(1)
        return mock.url, 'new'
    backend.refresh = refresh
    with request(env) as response:
        json.load(response)
    assert len(calls) == 1 and mock.gets == 1
    assert [call[1]['Authorization'] for call in mock.calls] == ['Bearer old', 'Bearer new']


def test_refresh_waiters_reuse_the_actual_failed_credential_receipt(env):
    _, backend = env.add()
    old = backend.credential_snapshot()
    calls = []
    backend.refresh = lambda: (calls.append(1) or backend.base_url, 'new-key')
    threads = [threading.Thread(target=POOL._refresh_backend_credentials,
                               args=(backend,), kwargs={'require_new': True, 'observed': old})
               for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)
        assert not thread.is_alive()
    assert len(calls) == 1


@pytest.mark.parametrize('bad', [{}, '', 0, False])
def test_malformed_empty_tool_call_container_is_not_valid(bad):
    with pytest.raises(POLICY.TransientError):
        PROTOCOL.validate_completion(completion('text', tool_calls=bad))


@pytest.mark.parametrize('bad', [{}, '', 0, False])
def test_malformed_empty_streamed_tool_call_container_is_not_valid(bad):
    with pytest.raises(POLICY.TransientError):
        PROTOCOL.StreamValidator().event(json.dumps({'choices': [
            {'delta': {'tool_calls': bad}, 'finish_reason': None}]}).encode())


def test_server_executed_tools_cannot_spend_outside_free_model(env):
    mock, _ = env.add()
    with pytest.raises(urllib.error.HTTPError) as error:
        request(env, body(tools=[{'type': 'web_search'}]))
    assert error.value.code == 400 and not mock.calls


def test_unknown_opencode_endpoint_metadata_fails_closed():
    backend = POOL.Backend('opencode-free', 'https://opencode.ai/zen/v1')
    for endpoints in (None, 12, {}, 'chat/completions'):
        assert not POLICY._accept_catalog_id(backend, 'mimo-v2.5-free', {'supported_endpoints': endpoints})


def test_listener_guard_cannot_be_replaced(env):
    with pytest.raises(RuntimeError, match='runtime guard'):
        SERVER.spawn_proxy(port=0, token=TOKEN, runtime_guard=lambda: None)


def test_catalog_failure_does_not_invert_credential_and_state_locks(env, monkeypatch):
    _, backend = env.add()
    backend.cached_models = None
    backend.cached_models_until = 0
    entered = threading.Event()
    original = backend.set_catalog
    def set_catalog(*args, **kwargs):
        entered.set()
        return original(*args, **kwargs)
    def fail(*_args):
        raise POLICY.TransientError('catalog unavailable')
    monkeypatch.setattr(backend, 'set_catalog', set_catalog)
    monkeypatch.setattr(env.owner, '_fetch_models', fail)
    with backend.refresh_lock:
        env.owner.refresh_catalogs()
        assert entered.wait(1)
        acquired = backend._state_lock.acquire(timeout=.2)
        if acquired:
            backend._state_lock.release()
    assert acquired, 'catalog failure held state while waiting for credential authority'
