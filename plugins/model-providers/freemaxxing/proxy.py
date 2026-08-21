"""Freemaxxing router public module.

Split into :mod:`router` (backend authority) and :mod:`server` (authenticated
loopback wire) so policy, I/O, and lifecycle remain independently reviewable.
"""

from .router import (
    _MAX_CATALOG_BODY_BYTES,
    _MAX_REQUEST_BODY_BYTES,
    _MAX_RESPONSE_BODY_BYTES,
    _MAX_SSE_EVENT_BYTES,
    _MAX_SSE_LINE_BYTES,
    AuthError,
    Backend,
    BackendPool,
    ClientRequestError,
    ModelNotFoundError,
    RateLimitError,
    TransientError,
    _accept_catalog_id,
    _exhausted_message,
    _parse_retry_after,
    _resolve_auto_model,
    pool,
)
from .server import ChatCompletionsHandler, spawn_proxy, stop_proxy

__all__ = [
    "AuthError",
    "Backend",
    "BackendPool",
    "ChatCompletionsHandler",
    "ClientRequestError",
    "ModelNotFoundError",
    "RateLimitError",
    "TransientError",
    "_MAX_CATALOG_BODY_BYTES",
    "_MAX_REQUEST_BODY_BYTES",
    "_MAX_RESPONSE_BODY_BYTES",
    "_MAX_SSE_EVENT_BYTES",
    "_MAX_SSE_LINE_BYTES",
    "_accept_catalog_id",
    "_exhausted_message",
    "_parse_retry_after",
    "_resolve_auto_model",
    "pool",
    "spawn_proxy",
    "stop_proxy",
]
