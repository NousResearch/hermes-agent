"""Fail-open, metadata-only observation; never read ahead or inspect HTTP bodies."""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import logging
import threading
import time
import uuid

import httpx

logger = logging.getLogger(__name__)
_CONTEXT = contextvars.ContextVar('hermes_usage_observation', default={})
IDENTITY = ('session_id', 'task_id', 'turn_id', 'api_request_id', 'platform',
            'provider', 'model', 'api_mode', 'aux_task')
COUNTERS = ('prompt_tokens', 'completion_tokens', 'total_tokens', 'input_tokens',
            'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens',
            'prompt_cache_hit_tokens', 'cached_tokens', 'cache_write_tokens')
DETAILS = {'prompt_tokens_details': ('cached_tokens', 'audio_tokens', 'cache_write_tokens', 'cache_creation_input_tokens'),
           'input_tokens_details': ('cached_tokens', 'audio_tokens', 'cache_write_tokens', 'cache_creation_tokens'),
           'completion_tokens_details': ('reasoning_tokens', 'audio_tokens',
                                         'accepted_prediction_tokens', 'rejected_prediction_tokens'),
           'output_tokens_details': ('reasoning_tokens', 'audio_tokens'),
           'cache_creation': ('ephemeral_5m_input_tokens', 'ephemeral_1h_input_tokens')}
HTTP_EVENTS = ('http_attempt_start', 'http_attempt_headers', 'http_attempt_end')


def _enabled():
    try:
        from hermes_cli.lifecycle import has_hook
        return any(has_hook(name) for name in HTTP_EVENTS)
    except Exception:  # health: allow BLE001 -- optional observer discovery must fail open without logging private errors
        return False


def _fire(event, **payload):
    try:
        from hermes_cli.lifecycle import invoke_hook
        invoke_hook(event, **payload)
    except Exception:  # health: allow BLE001 -- subscriber errors cannot affect inference or leak arbitrary exception text
        logger.warning('usage observation failed: event=%s', event)


def _get(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _identifier(value):
    return value[:256] if isinstance(value, str) else None


def raw_usage(value):
    """Whitelist numeric counters, never model_dump an arbitrary response object."""
    if value is None:
        return None
    result = {}
    for key in COUNTERS:
        number = _get(value, key)
        if type(number) is int and number >= 0:
            result[key] = number
    for key, allowed in DETAILS.items():
        detail = _get(value, key)
        picked = {name: _get(detail, name) for name in allowed}
        picked = {name: n for name, n in picked.items() if type(n) is int and n >= 0}
        if picked:
            result[key] = picked
    return result or None


def response_fields(response):
    """Adapter-preserved native counters take precedence over translated usage."""
    try:
        preserved = _get(response, 'raw_provider_usage')
        return {'raw_provider_usage': raw_usage(preserved if preserved is not None else _get(response, 'usage')),
                'provider_response_id': _identifier(_get(response, 'provider_response_id') or _get(response, 'id')),
                'provider_request_id': _identifier(_get(response, '_request_id')),
                'raw_usage_stage': 'adapter_preserved' if preserved is not None else 'sdk_or_adapter_response'}
    except Exception:  # health: allow BLE001 -- SDK properties are untrusted; omit exception text to preserve metadata-only logs
        logger.warning('response usage observation failed')
        return {'raw_provider_usage': None, 'provider_response_id': None,
                'provider_request_id': None, 'raw_usage_stage': 'unavailable'}


@contextmanager
def request_scope(metadata):
    # A nested auxiliary request must not inherit the main request's identity.
    safe = {key: value[:256] for key in IDENTITY
            if isinstance(value := metadata.get(key), str)}
    token = _CONTEXT.set(safe)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


class _Attempt:
    def __init__(self, request):
        self.context = contextvars.copy_context()
        self.base = dict(_CONTEXT.get())
        endpoint = str(request.url.copy_with(path='', query=None, fragment=None))
        self.base.update(http_attempt_id=uuid.uuid4().hex,
                         endpoint_sha256=hashlib.sha256(endpoint.encode()).hexdigest(),
                         started_at=time.time(), status_code=None, provider_request_id=None)
        self.clock = time.monotonic()
        self.done = False
        self.lock = threading.Lock()
        self.context.copy().run(_fire, 'http_attempt_start', **self.base)

    def headers(self, response):
        self.base.update(status_code=response.status_code,
                         provider_request_id=_identifier(response.headers.get('x-request-id') or response.headers.get('request-id')))
        self.context.copy().run(_fire, 'http_attempt_headers', **self.base)

    def finish(self, outcome, error=None):
        with self.lock:
            if self.done:
                return
            self.done = True
        self.context.copy().run(_fire, 'http_attempt_end', **self.base, ended_at=time.time(),
              api_duration=max(0.0, time.monotonic() - self.clock), transport_outcome=outcome,
              error_type=type(error).__name__ if error is not None else None)


class _SyncBytes(httpx.SyncByteStream):
    def __init__(self, inner, attempt):
        self.inner, self.attempt = inner, attempt

    def __iter__(self):
        try:
            yield from self.inner
        except BaseException as exc:
            self.attempt.finish('body_closed' if isinstance(exc, GeneratorExit) else 'body_error', exc)
            raise
        else:
            self.attempt.finish('body_exhausted')

    def close(self):
        try:
            self.inner.close()
        finally:
            self.attempt.finish('body_closed')


class _AsyncBytes(httpx.AsyncByteStream):
    def __init__(self, inner, attempt):
        self.inner, self.attempt = inner, attempt

    async def __aiter__(self):
        try:
            async for chunk in self.inner:
                yield chunk
        except BaseException as exc:
            self.attempt.finish('body_closed' if isinstance(exc, GeneratorExit) else 'body_error', exc)
            raise
        else:
            self.attempt.finish('body_exhausted')

    async def aclose(self):
        try:
            await self.inner.aclose()
        finally:
            self.attempt.finish('body_closed')


class SyncTransport(httpx.BaseTransport):
    def __init__(self, inner):
        self.inner = inner

    @property
    def _pool(self):
        return getattr(self.inner, '_pool', None)

    def handle_request(self, request):
        if not _enabled():
            return self.inner.handle_request(request)
        attempt = _Attempt(request)
        try:
            response = self.inner.handle_request(request)
        except BaseException as exc:
            attempt.finish('transport_error', exc)
            raise
        attempt.headers(response)
        if response.is_stream_consumed:
            attempt.finish('body_exhausted')
        else:
            response.stream = _SyncBytes(response.stream, attempt)
        return response

    def close(self):
        self.inner.close()


class AsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner):
        self.inner = inner

    @property
    def _pool(self):
        return getattr(self.inner, '_pool', None)

    async def handle_async_request(self, request):
        if not _enabled():
            return await self.inner.handle_async_request(request)
        attempt = _Attempt(request)
        try:
            response = await self.inner.handle_async_request(request)
        except BaseException as exc:
            attempt.finish('transport_error', exc)
            raise
        attempt.headers(response)
        if response.is_stream_consumed:
            attempt.finish('body_exhausted')
        else:
            response.stream = _AsyncBytes(response.stream, attempt)
        return response

    async def aclose(self):
        await self.inner.aclose()


def observe_http_client(client):
    """shortcut: httpx 0.28 transport slots; revalidate when upgrading httpx.

    Wrap per-client handles, not shared pools; proxy/TLS/mount policy is unchanged.
    """
    if not _enabled():
        return client
    try:
        wrapper = AsyncTransport if isinstance(client, httpx.AsyncClient) else SyncTransport
        seen = {}
        def wrap(inner):
            if inner is None or isinstance(inner, (SyncTransport, AsyncTransport)):
                return inner
            if id(inner) not in seen:
                seen[id(inner)] = wrapper(inner)
            return seen[id(inner)]
        transport = wrap(client._transport)
        mounts = {pattern: wrap(inner) for pattern, inner in client._mounts.items()}
        client._transport, client._mounts = transport, mounts
    except (AttributeError, TypeError):
        logger.warning('HTTP transport observer unavailable; coverage is incomplete')
    return client


class _StreamState:
    def __init__(self, metadata, fire):
        self.context = contextvars.copy_context()
        self.metadata = dict(metadata)
        self.fire = fire
        self.fields = response_fields(None)
        self.normalized = None
        self.done = False
        self.clock = time.monotonic()
        self.lock = threading.Lock()

    def chunk(self, chunk):
        try:
            response = _get(chunk, 'response') or chunk
            fields = response_fields(response)
            for key, value in fields.items():
                if value is not None:
                    self.fields[key] = value
            usage = fields.get('raw_provider_usage')
            if usage is not None:
                from agent.usage_pricing import normalize_usage
                mode = 'codex_responses' if 'input_tokens' in usage and 'prompt_tokens' not in usage else 'chat_completions'
                cu = normalize_usage(usage, api_mode=mode)
                self.normalized = asdict(cu)
                self.normalized.pop('raw_usage', None)
                self.normalized.update(prompt_tokens=cu.prompt_tokens, total_tokens=cu.total_tokens)
        except Exception:  # health: allow BLE001 -- malformed SDK chunks must pass through; their exception text may contain content
            logger.warning('stream usage observation unavailable')

    def finish(self, outcome, error=None):
        with self.lock:
            if self.done:
                return
            self.done = True
        self.context.copy().run(self.fire, 'auxiliary_stream_end', **self.metadata, **self.fields, usage=self.normalized,
                  ended_at=time.time(), api_duration=max(0.0, time.time() - self.metadata.get('started_at', time.time())),
                  stream_consumption_duration=max(0.0, time.monotonic() - self.clock),
                  stream_outcome=outcome, error_type=type(error).__name__ if error is not None else None,
                  status_code=getattr(error, 'status_code', None))


class _SyncIterator:
    def __init__(self, source, state):
        self.source, self.state, self.iterator = source, state, None

    def __getattr__(self, key):
        return getattr(self.source, key)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if self.iterator is None:
                self.iterator = iter(self.source)
            chunk = next(self.iterator)
        except StopIteration:
            self.state.finish('consumed')
            raise
        except BaseException as exc:
            self.state.finish('consumer_error', exc)
            raise
        self.state.chunk(chunk)
        return chunk

    def close(self):
        try:
            close = getattr(self.source, 'close', None)
            if close is not None:
                close()
        finally:
            self.state.finish('closed_early')

    def __enter__(self):
        enter = getattr(self.source, '__enter__', None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, kind, value, traceback):
        try:
            exit_ = getattr(self.source, '__exit__', None)
            if exit_ is not None:
                return exit_(kind, value, traceback)
            close = getattr(self.source, 'close', None)
            if close is not None:
                close()
        finally:
            self.state.finish('consumer_error' if value is not None else 'closed_early', value)


class _AsyncIterator:
    def __init__(self, source, state):
        self.source, self.state, self.iterator = source, state, None

    def __getattr__(self, key):
        return getattr(self.source, key)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            if self.iterator is None:
                self.iterator = self.source.__aiter__()
            chunk = await self.iterator.__anext__()
        except StopAsyncIteration:
            self.state.finish('consumed')
            raise
        except BaseException as exc:
            self.state.finish('consumer_error', exc)
            raise
        self.state.chunk(chunk)
        return chunk

    async def aclose(self):
        try:
            close = getattr(self.source, 'aclose', None)
            if close is not None:
                await close()
        finally:
            self.state.finish('closed_early')

    async def __aenter__(self):
        enter = getattr(self.source, '__aenter__', None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, kind, value, traceback):
        try:
            exit_ = getattr(self.source, '__aexit__', None)
            if exit_ is not None:
                return await exit_(kind, value, traceback)
            close = getattr(self.source, 'aclose', None)
            if close is not None:
                await close()
        finally:
            self.state.finish('consumer_error' if value is not None else 'closed_early', value)


def observe_aux_stream(response, metadata, fire):
    if _get(response, 'choices') is not None or callable(getattr(response, 'model_dump', None)):
        return response
    if hasattr(response, '__aiter__'):
        return _AsyncIterator(response, _StreamState(metadata, fire))
    if hasattr(response, '__iter__') and not isinstance(response, (dict, list, str, bytes)):
        return _SyncIterator(response, _StreamState(metadata, fire))
    return response
