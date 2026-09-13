"""One API credential, two shapes: a string, or a provider that mints one per request.

Most providers authenticate with a static string (an API key, an OAuth access token).
Two authenticate with a zero-argument callable instead, so a short-lived token can be
minted when it is needed:

* ``key_cmd`` custom providers — :class:`agent.command_token_source.CommandTokenSource`
* Azure Foundry with Entra ID — ``agent.azure_identity_adapter.build_token_provider``

Both shapes travel through the same ``api_key`` parameters, and the OpenAI SDK accepts
both. The trouble is everything around the SDK call:

* ``str(provider)`` is the object's repr. Sent as ``Authorization: Bearer <agent.command_
  token_source.CommandTokenSource object at 0x…>`` it looks like a credential bug in the
  gateway, not in Hermes.
* An SDK client built with a provider stores it in ``_api_key_provider`` and leaves
  ``client.api_key`` as ``""``. Code that rebuilds a client from ``client.api_key`` —
  an async twin, a routed copy, a fallback — silently drops the credential and 401s.
* ``AsyncOpenAI`` *awaits* its provider; a plain sync provider breaks it.

This module is the single place that knows those rules. Call sites decide what they
need, not how the shapes work:

=========================  =====================================================
Need                       Use
=========================  =====================================================
Is it a provider?          :func:`is_token_provider`
Is a credential present?   :func:`has_credential`
Store / pass it on         :func:`normalize_credential`
A string right now         :func:`materialize_credential` (probes, headers)
A string, never minting    :func:`static_credential` (lookups that must not mint)
Auth headers               :func:`bearer_headers`
Rebuild from a client      :func:`client_credential`
Hand to ``AsyncOpenAI``    :func:`async_credential`
=========================  =====================================================

Deliberately not here: masking a credential for display (banner, diagnostics, dashboard
each keep their own mask and use :func:`is_token_provider` to print a label instead), and
the strict Entra variant ``azure_identity_adapter.materialize_bearer_for_http``, which
raises instead of degrading because its caller must strip auth headers on failure.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, TypeGuard, Union, cast

logger = logging.getLogger(__name__)

TokenProvider = Callable[[], str]
AsyncTokenProvider = Callable[[], Awaitable[str]]
Credential = Union[str, TokenProvider, AsyncTokenProvider]


def is_token_provider(value: Any) -> TypeGuard[Union[TokenProvider, AsyncTokenProvider]]:
    """True for a callable credential. Strings are callable-free; this is the only test to use."""
    return callable(value) and not isinstance(value, str)


def normalize_credential(value: Any) -> Credential:
    """A stripped string, the provider itself, or ``""`` for anything else (``None``, objects).

    Never calls the provider: normalizing happens at configuration time, minting at request time.
    """
    if is_token_provider(value):
        return value
    if isinstance(value, str):
        return value.strip()
    return ""


def has_credential(value: Any) -> bool:
    """True for a non-empty string or any provider. Does not mint."""
    return is_token_provider(value) or (isinstance(value, str) and bool(value.strip()))


def materialize_credential(value: Any) -> str:
    """The credential as a string, minting from a sync provider if needed.

    Best effort by design — for probes and headers that must degrade, not crash: a provider
    that raises, returns a non-string, or is async yields ``""``. The provider's error is
    logged at debug level without its message, which may quote a token or a command line.
    """
    if is_token_provider(value):
        if inspect.iscoroutinefunction(value):
            return ""
        try:
            value = value()
        except Exception as exc:  # noqa: BLE001 — a probe must not crash on a token helper
            logger.debug("token provider failed while materializing a credential: %s",
                         type(exc).__name__)
            return ""
    return value.strip() if isinstance(value, str) else ""


def static_credential(value: Any) -> str:
    """The credential only if it is already a string; ``""`` for a provider. Never mints.

    For catalogue lookups and routing decisions that need a key only when one is at hand,
    where minting (a subprocess, an Entra round-trip) would cost more than the lookup.
    """
    return value.strip() if isinstance(value, str) else ""


def bearer_headers(value: Any, *, header: str = "Authorization", scheme: str = "Bearer") -> Dict[str, str]:
    """``{header: "<scheme> <token>"}`` from either shape, or ``{}`` when there is no token."""
    token = materialize_credential(value)
    if not token:
        return {}
    return {header: f"{scheme} {token}" if scheme else token}


def client_credential(client: Any) -> Credential:
    """The credential an SDK client was built with — its provider if it has one.

    ``client.api_key`` is ``""`` on a client built from a provider; the provider lives in
    ``_api_key_provider``. Read through the instance ``__dict__`` so a ``MagicMock`` client
    (whose every attribute is a callable mock) is not mistaken for one with a provider.
    """
    try:
        provider = vars(client).get("_api_key_provider")
    except TypeError:  # no __dict__ (slots, builtins)
        provider = None
    if is_token_provider(provider):
        return provider
    return normalize_credential(getattr(client, "api_key", ""))


def async_credential(value: Any) -> Union[str, AsyncTokenProvider]:
    """The credential in the shape ``AsyncOpenAI`` accepts.

    ``AsyncOpenAI`` awaits a provider. A sync provider is wrapped so its minting (possibly a
    subprocess) runs in a worker thread instead of blocking the event loop.
    """
    if not is_token_provider(value):
        return static_credential(value)
    if inspect.iscoroutinefunction(value):
        return value
    sync_provider = cast(TokenProvider, value)

    async def _minted() -> str:
        import asyncio  # lazy: this module sits on the CLI's cold-start import path

        token = await asyncio.to_thread(sync_provider)
        return token.strip() if isinstance(token, str) else ""

    return _minted
