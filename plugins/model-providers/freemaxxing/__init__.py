"""Optional Freemaxxing model-provider plugin; no core registration patches.

FREEMAXXING_ENABLED=1 is explicit activation. Discovery otherwise registers only
static metadata: no listener, credential lookup, catalog request, or worker.
"""
from __future__ import annotations

import os
import secrets
import threading

from providers import register_provider
from providers.base import ProviderProfile

from .policy import Limits, _OPENCODE_CHAT_MODELS, validate_url
from .proxy import Backend, pool, spawn_proxy, stop_proxy as _stop_proxy

_LOCAL_TOKEN = secrets.token_urlsafe(32)
_listener = None
_listener_lock = threading.RLock()
_NOUS_BASE_URL = 'https://inference-api.nousresearch.com/v1'


def _enabled():
    value = os.environ.get('FREEMAXXING_ENABLED', '').strip().lower()
    if value not in {'', '0', 'false', 'no', 'off', '1', 'true', 'yes', 'on'}:
        raise ValueError('FREEMAXXING_ENABLED must be a boolean')
    return value in {'1', 'true', 'yes', 'on'}


def _multiplex_active():
    try:
        from agent.secret_scope import is_multiplex_active
        return bool(is_multiplex_active())
    except Exception:
        # Unknown scope authority is not permission to read process credentials.
        return True


def _assert_single_profile_runtime():
    if _multiplex_active():
        raise RuntimeError('Freemaxxing does not support gateway.multiplex_profiles')


def _resolve_key(names):
    _assert_single_profile_runtime()
    from agent.secret_scope import current_secret_scope, get_secret
    if current_secret_scope() is not None:
        for name in names:
            value = get_secret(name)
            if value and str(value).strip():
                return str(value).strip()
        return ''  # A scoped miss cannot widen to dotenv or the process environment.
    from hermes_cli.config import get_env_value_prefer_dotenv
    for name in names:
        value = get_env_value_prefer_dotenv(name)
        if value and str(value).strip():
            return str(value).strip()
    return ''


def _resolve_nous_credentials():
    _assert_single_profile_runtime()
    from hermes_cli.auth import resolve_nous_runtime_credentials
    try:
        credentials = resolve_nous_runtime_credentials()
        key = str(credentials.get('api_key') or '').strip()
        base = str(credentials.get('base_url') or _NOUS_BASE_URL).rstrip('/')
        validate_url(base)
        if key:
            return base, key
    except Exception:
        pass
    return _NOUS_BASE_URL, _resolve_key(('NOUS_API_KEY',))


def _build_pool():
    _assert_single_profile_runtime()
    # Validate the complete configuration before retiring a working generation.
    allowed = {item.strip() for item in os.environ.get(
        'FREEMAXXING_FREE_TIER_PROVIDERS', '').split(',') if item.strip()}
    providers = {
        'groq': ('https://api.groq.com/openai/v1', ('GROQ_API_KEY',)),
        'gemini': ('https://generativelanguage.googleapis.com/v1beta/openai',
                   ('GEMINI_API_KEY', 'GOOGLE_API_KEY')),
        'mistral': ('https://api.mistral.ai/v1', ('MISTRAL_API_KEY',)),
    }
    if allowed - (providers.keys() | {'nous-portal'}):
        raise ValueError('unknown free-tier provider; no account was enrolled')
    local = os.environ.get('FREEMAXXING_LOCAL_BASE_URL', '').strip()
    if local:
        validate_url(local, local=True)
    backends = []
    # Nous prices can change and its API has no documented zero-price cap.
    # Enrollment therefore requires the operator's separate free-only account
    # assertion, plus fresh zero pricing for every admitted model.
    if 'nous-portal' in allowed:
        backends.append(Backend('nous-portal', _NOUS_BASE_URL,
                                refresh=_resolve_nous_credentials, tier=0,
                                free_tier_only=True))
    key = _resolve_key(('OPENROUTER_API_KEY',))
    if key:
        backend = Backend('openrouter', 'https://openrouter.ai/api/v1', api_key=key, tier=1)
        backend.set_cached_models(['openrouter/free'], ttl=0)
        backends.append(backend)
    backend = Backend('opencode-free', 'https://opencode.ai/zen/v1', tier=0,
                      default_model='mimo-v2.5-free')
    backend.set_cached_models(sorted(_OPENCODE_CHAT_MODELS), ttl=0)
    backends.append(backend)
    # A key alone never enrolls an allowance-based account in automatic routing.
    for name in sorted(allowed & providers.keys()):
        base, names = providers[name]
        key = _resolve_key(names)
        if key:
            backends.append(Backend(name, base, api_key=key, tier=2, free_tier_only=True))
    if local:
        backend = Backend('local', local, tier=3)
        models = [m.strip() for m in os.environ.get('FREEMAXXING_LOCAL_MODELS', '').split(',') if m.strip()]
        if models:
            backend.set_cached_models(models, ttl=0)
        backends.append(backend)
    pool.clear()
    for backend in backends:
        pool.add(backend)


def local_token():
    return _LOCAL_TOKEN


def _configured_port():
    port = int(os.environ.get('FREEMAXXING_PORT', '0'))
    if not 0 <= port <= 65535:
        raise ValueError('invalid FREEMAXXING_PORT')
    return port


def ensure_proxy():
    global _listener
    _assert_single_profile_runtime()
    if not _enabled():
        raise RuntimeError('Freemaxxing is disabled; set FREEMAXXING_ENABLED=1 explicitly')
    with _listener_lock:
        _listener = spawn_proxy(port=_configured_port(), token=_LOCAL_TOKEN,
                                pool_initializer=_build_pool,
                                runtime_guard=_assert_single_profile_runtime)
        return f'http://127.0.0.1:{_listener.server_address[1]}/v1'


def stop_proxy():
    global _listener
    with _listener_lock:
        _stop_proxy(_listener)
        _listener = None


class _LocalCapabilityEnvVars(tuple):
    def __new__(cls):
        return super().__new__(cls, ('FREEMAXXING_API_KEY',))

    def __iter__(self):
        if not _enabled() or _multiplex_active():
            return iter(())
        return super().__iter__()


class FreemaxxingProfile(ProviderProfile):
    def fetch_models(self, **_kwargs):
        return ['freemaxxing']

    def build_extra_body(self, *, session_id=None, **_context):
        return {'freemaxxing_session': str(session_id)} if session_id else {}


def _register():
    base = 'http://127.0.0.1:11435/v1'  # Inactive metadata only.
    if _enabled():
        pool.limits = Limits(
            total=float(os.environ.get('FREEMAXXING_REQUEST_TIMEOUT', '90')),
            attempts=int(os.environ.get('FREEMAXXING_MAX_ATTEMPTS', '12')),
            concurrency=int(os.environ.get('FREEMAXXING_CONCURRENCY', '8')),
        )
        base = ensure_proxy()
        os.environ['FREEMAXXING_API_KEY'] = _LOCAL_TOKEN
    register_provider(FreemaxxingProfile(
        name='freemaxxing', aliases=('fm', 'freemaxxing-auto'),
        display_name='Freemaxxing',
        description='Optional free-only provider pool with atomic completion recovery',
        env_vars=('FREEMAXXING_API_KEY',), base_url=base, auth_type='api_key',
        api_mode='chat_completions', supports_health_check=False,
        supports_vision=False, default_aux_model='freemaxxing',
        fallback_models=('freemaxxing',),
    ))
    # Existing plugin-side generic resolver bridge. No monkeypatch of provider
    # discovery, transport, session handling, or import-parent namespaces.
    from hermes_cli.auth import PROVIDER_REGISTRY, ProviderConfig
    runtime = ProviderConfig(id='freemaxxing', name='Freemaxxing', auth_type='api_key',
                             inference_base_url=base, api_key_env_vars=_LocalCapabilityEnvVars())
    for alias in ('freemaxxing', 'fm', 'freemaxxing-auto'):
        PROVIDER_REGISTRY[alias] = runtime


_register()
