"""Log-shaped quota failures through real auxiliary routing and persisted pools."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import httpx
import pytest
import yaml

from agent import auxiliary_client as aux
from agent.credential_pool import CredentialPool, PooledCredential, STATUS_EXHAUSTED, load_pool
from agent.error_classifier import FailoverReason, classify_api_error
from agent.gemini_native_adapter import GeminiNativeClient, GeminiAPIError, gemini_http_error

GEMINI = 'https://generativelanguage.googleapis.com/v1beta'
NVIDIA = 'https://integrate.api.nvidia.com/v1'
LIMITED = 'gemini-test-limited'
HEALTHY = 'gemini-test-healthy'
NIM_LIMITED = 'nvidia/test-limited'
NIM_HEALTHY = 'nvidia/test-healthy'


def quota_body(kind='tokens', *, delay=48.100581664, scoped=True):
    metric = ('generate_content_free_tier_requests' if kind == 'requests'
              else 'generate_content_free_tier_input_token_count')
    limit = 0 if kind == 'zero' else 20 if kind == 'requests' else 250000
    message = ('You exceeded your current quota, please check your plan and billing details.\n'
               f'* Quota exceeded for metric: generativelanguage.googleapis.com/{metric}, '
               f'limit: {limit}, model: {LIMITED}\nPlease retry in {delay}s.')
    violations = [{'quotaMetric': metric, 'quotaDimensions': {'model': LIMITED}, 'quotaValue': str(limit)}]
    if kind == 'mixed':
        violations.append({'quotaMetric': 'project_requests', 'quotaDimensions': {'project': 'offline'}})
    details = [{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': f'{delay}s'}]
    if kind != 'text':
        details.insert(0, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': violations})
    if not scoped:
        # RetryInfo-only responses have no model-scope evidence and no prose delay.
        message = 'You exceeded your current quota, please check your plan and billing details.'
        details = details[-1:]
    return {'error': {'code': 429, 'status': 'RESOURCE_EXHAUSTED', 'message': message, 'details': details}}


def persist_keys(provider, count=4):
    entries = [PooledCredential(provider=provider, id=f'{provider}-{i}', label=f'offline-{i}',
               auth_type='api_key', priority=i, source='manual', access_token=f'offline-{provider}-{i}')
               for i in range(count)]
    pool = CredentialPool(provider, entries)
    pool._persist()
    return entries


@pytest.fixture
def routed_wire(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({
        'model': {'provider': 'gemini', 'default': HEALTHY},
        'auxiliary': {'transient_retries': 0, 'moa_reference': {
            'provider': 'gemini', 'model': LIMITED,
            'fallback_chain': [{'provider': 'gemini', 'model': HEALTHY}],
        }},
    }))
    persist_keys('gemini')
    persist_keys('nvidia')
    state = {'kind': 'tokens', 'requests': []}
    lock = threading.Lock()
    clients = []
    aux._reset_aux_unhealthy_cache()

    def upstream(request):
        body = json.loads(request.content)
        is_google = request.url.host == 'generativelanguage.googleapis.com'
        model = request.url.path.split('/models/', 1)[1].split(':', 1)[0] if is_google else body['model']
        key = request.headers.get('x-goog-api-key') or request.headers.get('authorization')
        with lock:
            state['requests'].append((model, key))
        if model == LIMITED:
            return httpx.Response(429, json=quota_body(state['kind']))
        if model == NIM_LIMITED:
            return httpx.Response(429, json={'message': 'ResourceExhausted: Worker local total request limit reached (32/32)'})
        if is_google:
            data = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'healthy route response'}]},
                                    'finishReason': 'STOP'}],
                    'usageMetadata': {'promptTokenCount': 84665, 'candidatesTokenCount': 3, 'totalTokenCount': 84668}}
            if ':streamGenerateContent' in request.url.path:
                return httpx.Response(200, text='data: ' + json.dumps(data) + '\n\n', headers={'content-type': 'text/event-stream'})
            return httpx.Response(200, json=data)
        data = {'id': 'offline', 'object': 'chat.completion', 'model': model,
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'healthy route response'}, 'finish_reason': 'stop'}]}
        if body.get('stream'):
            chunk = {'id': 'offline', 'object': 'chat.completion.chunk', 'model': model,
                     'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'healthy route response'}, 'finish_reason': 'stop'}]}
            return httpx.Response(200, text='data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n', headers={'content-type': 'text/event-stream'})
        return httpx.Response(200, json=data)

    transport = httpx.MockTransport(upstream)
    original_init = GeminiNativeClient.__init__

    def native_init(self, *args, **kwargs):
        client = httpx.Client(transport=transport)
        clients.append(client)
        kwargs['http_client'] = client
        original_init(self, *args, **kwargs)

    def http_client_kwargs(base_url, *, async_mode=False):
        client = httpx.AsyncClient(transport=transport) if async_mode else httpx.Client(transport=transport)
        clients.append(client)
        return {'http_client': client}

    monkeypatch.setattr(GeminiNativeClient, '__init__', native_init)
    monkeypatch.setattr(aux, '_openai_http_client_kwargs', http_client_kwargs)
    yield state, tmp_path
    aux._reset_aux_unhealthy_cache()
    aux._evict_cached_clients('gemini')
    aux._evict_cached_clients('nvidia')
    for client in clients:
        if isinstance(client, httpx.AsyncClient):
            asyncio.run(client.aclose())
        else:
            client.close()


@pytest.mark.parametrize('mode', ['sync', 'async', 'parallel'])
@pytest.mark.parametrize('kind', ['tokens', 'requests', 'zero', 'text', 'worker', 'cross-provider'])
def test_auxiliary_quota_keeps_other_routes_and_credentials_usable(routed_wire, mode, kind):
    state, home = routed_wire
    state['kind'] = kind if kind in {'tokens', 'requests', 'zero', 'text'} else 'tokens'
    provider, model = ('nvidia', NIM_LIMITED) if kind in {'worker', 'cross-provider'} else ('gemini', LIMITED)
    if kind in {'worker', 'cross-provider'}:
        config = yaml.safe_load((home / 'config.yaml').read_text())
        config['auxiliary']['moa_reference']['fallback_chain'] = (
            [{'provider': 'gemini', 'model': LIMITED}, {'provider': 'nvidia', 'model': NIM_HEALTHY}]
            if kind == 'cross-provider' else [{'provider': 'nvidia', 'model': NIM_HEALTHY}])
        (home / 'config.yaml').write_text(yaml.safe_dump(config))

    def invoke(p=provider, m=model):
        kwargs = dict(task='moa_reference', provider=p, model=m,
                      messages=[{'role': 'user', 'content': 'offline recovery check'}], max_tokens=64)
        return asyncio.run(aux.async_call_llm(**kwargs)) if mode == 'async' else aux.call_llm(**kwargs)

    if kind == 'cross-provider':
        # The existing auxiliary candidate policy raises non-auth fallback failures.
        # Both failed routes must remain available on the next call, not get payment bans.
        with pytest.raises(GeminiAPIError):
            invoke()
        assert not aux._is_provider_unhealthy('gemini')
        assert not aux._is_provider_unhealthy('nvidia')
        result = invoke('gemini', LIMITED)
        results, calls = [result], 2
    elif mode == 'parallel':
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: invoke(), range(4)))
        calls = 4
    else:
        results, calls = [invoke()], 1
    assert all(result.choices[0].message.content == 'healthy route response' for result in results)
    assert not aux._is_provider_unhealthy('gemini')
    assert not aux._is_provider_unhealthy('nvidia')
    # A model quota must not produce even an immediate duplicate call before fallback.
    failed_calls = sum(m == model for m, _ in state['requests'])
    assert failed_calls == (1 if kind == 'cross-provider' else calls)
    for p in ('gemini', 'nvidia'):
        reloaded = load_pool(p)
        assert reloaded.has_available()
        assert all(entry.last_status != STATUS_EXHAUSTED for entry in reloaded.entries())


@pytest.mark.parametrize('count', [1, 4])
@pytest.mark.parametrize('kind', ['google-retryinfo', 'google-mixed', 'nvidia-retryafter', 'billing', 'daily', 'auth'])
def test_auxiliary_pool_persists_retry_deadlines_without_losing_real_quota_walls(tmp_path, monkeypatch, kind, count):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    now = [time.time()]
    monkeypatch.setattr(time, 'time', lambda: now[0])
    provider = 'nvidia' if kind == 'nvidia-retryafter' else 'gemini'
    entries = persist_keys(provider, count)
    delay = 48.100581664
    if kind.startswith('google'):
        error = gemini_http_error(httpx.Response(429, json=quota_body('mixed' if kind == 'google-mixed' else 'tokens',
                                                                scoped=kind == 'google-mixed')))
    elif kind == 'nvidia-retryafter':
        from openai import RateLimitError
        response = httpx.Response(429, json={'status': 429, 'title': 'Too Many Requests'},
                                  headers={'retry-after': str(delay)}, request=httpx.Request('POST', NVIDIA))
        error = RateLimitError('Too Many Requests', response=response, body={'status': 429, 'title': 'Too Many Requests'})
    else:
        error = Exception({'billing': 'Payment required: insufficient credits', 'daily': 'Daily quota exceeded', 'auth': 'Invalid API key'}[kind])
        error.status_code = {'billing': 402, 'daily': 429, 'auth': 401}[kind]
    temporary = kind in {'google-retryinfo', 'google-mixed', 'nvidia-retryafter'}
    if temporary:
        assert not aux._is_payment_error(error)
        assert aux._is_rate_limit_error(error)
        assert classify_api_error(error, provider=provider).reason == FailoverReason.rate_limit
    else:
        assert aux._is_payment_error(error) is (kind != 'auth')
    first_reset = None
    for entry in entries:
        aux._recover_provider_pool(provider, error, failed_api_key=entry.runtime_api_key)
        persisted = next(row for row in load_pool(provider).entries() if row.id == entry.id)
        assert persisted.last_status == STATUS_EXHAUSTED
        if temporary:
            assert persisted.last_error_reset_at == pytest.approx(now[0] + delay, rel=0, abs=0.001)
            first_reset = first_reset or persisted.last_error_reset_at
        now[0] += 1
    assert not load_pool(provider).has_available()
    if temporary:
        now[0] = first_reset + 0.01
        recovered = load_pool(provider)
        assert recovered.has_available()
        assert recovered.select().id == entries[0].id
        assert sum(row.last_status == STATUS_EXHAUSTED for row in recovered.entries()) == count - 1
    elif kind in {'billing', 'daily'}:
        now[0] += 61
        assert not load_pool(provider).has_available()


@pytest.mark.parametrize('provider,model', [('gemini', HEALTHY), ('nvidia', NIM_HEALTHY)])
@pytest.mark.parametrize('reason', [FailoverReason.upstream_rate_limit, FailoverReason.rate_limit, FailoverReason.billing])
def test_fallback_model_failure_does_not_extend_primary_quota_deadline(provider, model, reason):
    from types import SimpleNamespace
    from agent.fallback_cooldown import _arm_rate_limit_cooldown
    deadline = time.monotonic() + 48.100581664
    agent = SimpleNamespace(
        provider=provider, model=model, _fallback_activated=True,
        _primary_runtime={'provider': 'gemini', 'model': LIMITED},
        _model_quota_retry_deadline=(provider, model, time.monotonic() + 300),
        _rate_limited_until=deadline, _rate_limit_backoff_count=2,
    )
    armed = _arm_rate_limit_cooldown(agent, reason)
    if provider == 'gemini' and reason != FailoverReason.upstream_rate_limit:
        # Ordinary account/credential-wide failures retain the existing policy.
        assert armed is not None
        assert agent._rate_limit_backoff_count == 3
    else:
        assert armed is None
        assert agent._rate_limited_until == deadline
        assert agent._rate_limit_backoff_count == 2
