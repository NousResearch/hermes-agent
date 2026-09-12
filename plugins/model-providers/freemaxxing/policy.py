"""Freemaxxing's request, resource, and spending boundaries.

A model name is not spending authority. Admission is checked again at dispatch;
unknown providers and contradictory catalog prices are never implicitly free.
"""
from __future__ import annotations

import email.utils
import math
import time
from dataclasses import dataclass
from datetime import timezone
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlsplit

_MAX_REQUEST_BODY_BYTES = 1_000_000
_MAX_RESPONSE_BODY_BYTES = 10_000_000
_MAX_CATALOG_BODY_BYTES = 2_000_000
_MAX_SSE_LINE_BYTES = 1_000_000
_MAX_SSE_EVENT_BYTES = 2_000_000
_CATALOG_TTL_SECONDS = 120.0
_ROUTER_MODELS = frozenset({'freemaxxing', 'fm', 'freemaxxing-auto'})
# Only chat-completions SKUs documented by OpenCode. A new SKU with explicit
# compatible endpoint metadata can also be admitted; a '-free' suffix alone
# must not cause a Responses/Messages model to be sent to the wrong API.
_OPENCODE_CHAT_MODELS = frozenset({
    'big-pickle', 'mimo-v2.5-free', 'ling-3.0-flash-fin-free',
    'nemotron-3-ultra-free', 'nemotron-3.5-lightning-free',
    'laguna-s-2.1-free',
})
_PRICE_PAIRS = (('prompt', 'completion'), ('input', 'output'),
                ('input_cost_per_token', 'output_cost_per_token'))


class TransientError(Exception):
    """Retryable upstream transport or completion-integrity failure."""


class ModelNotFoundError(Exception):
    """Route-local model/capability failure; do not poison the provider."""


class AuthError(Exception):
    """An account credential was rejected."""


class ClientRequestError(Exception):
    """Invalid caller request; do not replay it to other providers."""


class FreePolicyError(ClientRequestError):
    """The exact dispatch does not satisfy this plugin's free-only policy."""


class RateLimitError(Exception):
    def __init__(self, message: str, retry_after: float = 30.0):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(frozen=True)
class Limits:
    connect: float = 5.0
    read: float = 25.0
    total: float = 90.0
    catalog: float = 4.0
    attempts: int = 12
    concurrency: int = 8

    def __post_init__(self):
        for value in (self.connect, self.read, self.total, self.catalog):
            if not math.isfinite(value) or not 0 < value <= 600:
                raise ValueError('timeouts must be finite and between 0 and 600 seconds')
        if not 1 <= self.attempts <= 64 or not 1 <= self.concurrency <= 32:
            raise ValueError('invalid Freemaxxing attempt/concurrency bound')


class Budget:
    def __init__(self, seconds: float):
        self.deadline = time.monotonic() + seconds

    def remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TransientError('Freemaxxing recovery deadline exhausted')
        return remaining


def _is_router_model(model: str) -> bool:
    return str(model or '').strip().lower() in _ROUTER_MODELS


def _parse_retry_after(headers: Any) -> float:
    raw = headers.get('Retry-After') if headers is not None else None
    if raw is None:
        return 30.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        try:
            date = email.utils.parsedate_to_datetime(str(raw))
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            value = date.timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return 30.0
    # Do not shorten a daily/account quota reset into a five-minute retry storm.
    return max(0.0, value) if math.isfinite(value) else 30.0


def is_loopback(url: str) -> bool:
    try:
        parts = urlsplit(url)
        return (parts.scheme in {'http', 'https'} and
                parts.hostname in {'127.0.0.1', '::1'} and
                not parts.username and not parts.password and
                not parts.query and not parts.fragment and parts.port != 0)
    except ValueError:
        return False


def validate_url(url: str, *, local: bool = False) -> None:
    parts = urlsplit(url)
    if (not parts.hostname or parts.username or parts.password or parts.query or
            parts.fragment or (parts.scheme != 'https' and not is_loopback(url))):
        raise FreePolicyError('upstream URL must be HTTPS or an explicit numeric loopback')
    if local and not is_loopback(url):
        raise FreePolicyError('local free routes must use numeric loopback, not a remote host')


def _zero(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        number = Decimal(str(value))
        return number.is_finite() and number == 0
    except (InvalidOperation, TypeError, ValueError):
        return False


def free_price(row: dict) -> bool:
    pricing = row.get('pricing')
    if not isinstance(pricing, dict):
        return False
    # Nous exposes pre-discount prices as pricing.original (display metadata).
    pricing = {key: value for key, value in pricing.items() if key != 'original'}
    return (any(all(key in pricing for key in pair) for pair in _PRICE_PAIRS)
            and bool(pricing) and all(_zero(value) for value in pricing.values()))


def conflicting_price(row: dict) -> bool:
    pricing = row.get('pricing')
    if isinstance(pricing, dict):
        pricing = {key: value for key, value in pricing.items() if key != 'original'}
    return pricing is not None and (not isinstance(pricing, dict) or
                                   not pricing or not all(_zero(v) for v in pricing.values()))


def _accept_catalog_id(backend: Any, model_id: str, row: dict | None = None) -> bool:
    model = str(model_id or '').strip()
    if not model or model.startswith('~') or model.endswith(':batch'):
        return False
    row = row or {}
    if conflicting_price(row):
        return False
    kind = backend.kind
    if kind == 'openrouter':
        return model.endswith(':free') or model == 'openrouter/free'
    if kind == 'opencode-free':
        mode = row.get('api_mode')
        endpoints = row.get('supported_endpoints', [])
        if not isinstance(endpoints, list):
            return False
        chat = (mode == 'chat_completions' or 'chat/completions' in endpoints or
                (mode is None and not endpoints and model in _OPENCODE_CHAT_MODELS))
        return chat and (model.endswith('-free') or model == 'big-pickle')
    if kind == 'nous-portal':
        # No static paid/previously-promotional model escape hatch.
        return backend.free_tier_only and free_price(row)
    if kind == 'local':
        return is_loopback(backend.base_url)
    return kind in {'groq', 'gemini', 'mistral'} and backend.free_tier_only


def guard_body(backend: Any, body: dict, row: dict) -> dict:
    """Apply policy to aliases AND explicit models at the last pre-I/O boundary."""
    if not _accept_catalog_id(backend, str(body.get('model', '')), row):
        raise FreePolicyError('selected model has no current free-route authority')
    outgoing = dict(body)
    for tool in outgoing.get('tools') or []:
        if not isinstance(tool, dict) or tool.get('type') != 'function':
            raise ClientRequestError('only client-executed function tools are allowed')
    # These can install paid services or independent routing policies upstream.
    if any(key in outgoing for key in ('models', 'plugins', 'transforms', 'web_search_options')):
        raise ClientRequestError('upstream fallbacks and paid extensions are not allowed')
    if 'provider' in outgoing and backend.kind != 'openrouter':
        raise ClientRequestError('provider-routing overrides are not allowed')
    if backend.kind == 'openrouter':
        requested = outgoing.get('provider', {})
        if not isinstance(requested, dict):
            raise ClientRequestError('provider must be an object')
        # Never inherit an upstream caller's paid fallback/BYOK routing policy.
        outgoing['provider'] = {'max_price': {'prompt': 0, 'completion': 0,
                                            'image': 0, 'request': 0},
                                'require_parameters': True}
    return outgoing
