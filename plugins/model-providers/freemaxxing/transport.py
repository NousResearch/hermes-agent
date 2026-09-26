"""Reusable HTTP transport; no redirects, no credential-bearing ambient headers."""
from __future__ import annotations

import httpx

from .policy import (AuthError, Budget, ClientRequestError, Limits,
                     ModelNotFoundError, RateLimitError, TransientError,
                     _parse_retry_after, validate_url)


class Transport:
    def __init__(self, limits: Limits):
        self.limits = limits
        self.client = httpx.Client(
            follow_redirects=False,
            limits=httpx.Limits(max_connections=limits.concurrency,
                               max_keepalive_connections=limits.concurrency),
        )

    def close(self):
        self.client.close()

    def request(self, backend, method, path, budget: Budget, *, body=None, catalog=False, credentials=None):
        base, key = credentials or backend.credential_snapshot()
        validate_url(base, local=backend.kind == 'local')
        if backend.kind not in {'opencode-free', 'local'} and not key:
            raise AuthError('missing upstream credential')
        remaining = budget.remaining()
        read = min(self.limits.catalog if catalog else self.limits.read, remaining)
        headers = {'Accept': 'application/json', 'Accept-Encoding': 'identity',
                   'User-Agent': 'HermesAgent/Freemaxxing'}
        if backend.kind != 'opencode-free' and key:
            headers['Authorization'] = f'Bearer {key}'
        if backend.kind == 'opencode-free':
            headers.update({'HTTP-Referer': 'https://hermes-agent.nousresearch.com',
                            'X-Title': 'Hermes Agent'})
        timeout = httpx.Timeout(connect=min(self.limits.connect, remaining),
                                read=read, write=read, pool=min(1.0, remaining))
        try:
            request = self.client.build_request(method, base.rstrip('/') + path,
                                                headers=headers, json=body, timeout=timeout)
            response = self.client.send(request, stream=True)
        except (httpx.HTTPError, OSError) as exc:
            raise TransientError('upstream connection failed') from exc
        code = response.status_code
        if response.headers.get('Content-Encoding', 'identity').lower() not in {'', 'identity'}:
            response.close()
            raise TransientError('compressed upstream response refused at byte-boundary')
        if 200 <= code < 300:
            return response
        try:
            if code == 429:
                raise RateLimitError('upstream rate limit', _parse_retry_after(response.headers))
            if code in {401, 403}:
                raise AuthError('upstream credential rejected')
            if code == 404:
                raise ModelNotFoundError('model unavailable')
            if 300 <= code < 400:
                raise TransientError('upstream redirect refused')
            # Inspect only a bounded error body; do not publish provider text/secrets.
            raw = bytearray()
            for chunk in response.iter_raw():
                budget.remaining()
                raw.extend(chunk[:16_384 - len(raw)])
                if len(raw) >= 16_384:
                    break
            text = raw.decode('utf-8', errors='replace').lower()
            if code in {400, 413, 422} and any(word in text for word in (
                'context length', 'context_length', 'context window', 'maximum context',
                'model not found', 'invalid model', 'unknown model', 'not support',
                'unsupported', 'not available',
            )):
                raise ModelNotFoundError('route lacks requested model/capability/context')
            if 400 <= code < 500:
                raise ClientRequestError(f'upstream rejected request (HTTP {code})')
            raise TransientError(f'upstream unavailable (HTTP {code})')
        except (httpx.HTTPError, OSError) as exc:
            raise TransientError('upstream error response interrupted') from exc
        finally:
            response.close()


def read_bounded(response, budget: Budget, limit: int):
    raw = bytearray()
    try:
        for chunk in response.iter_raw():
            budget.remaining()
            if len(raw) + len(chunk) > limit:
                raise TransientError('upstream response exceeded the byte limit')
            raw.extend(chunk)
        return bytes(raw)
    except (httpx.HTTPError, OSError) as exc:
        raise TransientError('upstream response interrupted') from exc
    finally:
        response.close()
