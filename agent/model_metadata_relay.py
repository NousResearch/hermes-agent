"""Endpoint-scoped and relay-aware context metadata — facade sibling of ``model_metadata``.

Some windows belong to one provider *endpoint*, not to the model id: Kimi Coding serves ``k3`` at
1 Mi only on ``api.kimi.com/coding``, NVIDIA NIM serves ``deepseek-v4-pro`` at 262,144 while
DeepSeek's native endpoint is 1M. The table below is the single source for those bounds, and it
answers two ways (#142334):

* canonical host+path — the surface the window was confirmed on;
* ``(provider, model)`` when the host is an unknown relay — a gateway fronting that provider
  serves the same window, unless the host belongs to a different known provider.

Facade symbols (``_normalize_base_url``, ``_infer_provider_from_url``) are imported inside the
functions that need them: the facade imports this module at module level, so a module-level import
back would close the cycle.
"""

from __future__ import annotations

import contextlib
from typing import Optional
from urllib.parse import urlparse

# (host, canonical paths, model ids, context, owning provider) — see _endpoint_scoped_context_length.
_ENDPOINT_SCOPED_CONTEXT = (
    ("api.kimi.com", {"/coding", "/coding/v1"}, {"k3", "kimi-k3", "kimi-k3-cot"}, 1_048_576, "kimi-coding"),
    ("integrate.api.nvidia.com", {"/v1"}, {"deepseek-ai/deepseek-v4-pro"}, 262_144, "nvidia"),
)


def _relay_model_ids(model_key: str) -> set[str]:
    """Ids a relay may spell for the same model: the id itself plus its vendor-stripped tail
    (``kimi/k3`` → ``k3``). Gateways re-serve upstream ids vendor-prefixed, so the provider fallback
    has to see through the prefix. Deliberately NOT used for the canonical host+path match."""
    ids = {model_key}
    if "/" in model_key:
        ids.add(model_key.rsplit("/", 1)[1])
    return ids


def _endpoint_scoped_context_length(model: str, base_url: str, provider: str = "") -> Optional[int]:
    """Context confirmed for one provider endpoint only (see _ENDPOINT_SCOPED_CONTEXT): Kimi Coding
    serves K3 at 1 Mi only on the canonical ``api.kimi.com/coding`` host (legacy Moonshot keys do
    not); NVIDIA NIM serves deepseek-v4-pro at 262,144 while DeepSeek's native endpoint is 1M.

    The canonical host rules its own entries: reached over plaintext, on a non-443 port or at
    another path it is NOT the scoped surface, whatever *provider* says. Anywhere else the request
    is a relay/gateway hop unless the host belongs to a DIFFERENT known provider (that host owns
    its own metadata), and there the entry answers by ``(provider, model)`` — the provider string
    names whose models these are, so its declared window still holds behind any relay base URL
    (#142334).
    """
    # Facade-owned helpers: imported here, not at module level, so the facade's module-level
    # import of this sibling stays a one-way edge.
    from agent.model_metadata import _infer_provider_from_url, _normalize_base_url

    try:
        parsed = urlparse(_normalize_base_url(base_url))
        port = parsed.port
    except ValueError:
        return None
    # Canonical surface only: https://host[:443]/path with no credentials/query/fragment.
    canonical_surface = (
        parsed.scheme.lower() == "https" and port in (None, 443)
        and parsed.username is None and parsed.password is None
        and not parsed.query and not parsed.fragment
    )
    host, path, model_key = (parsed.hostname or "").lower(), parsed.path.rstrip("/"), model.strip().lower()
    provider_key = (provider or "").strip().lower()
    # A known provider host is never a relay: it would claim the provider's window for another
    # vendor's surface (e.g. NVIDIA's NIM bound leaking onto api.deepseek.com). No host at all
    # (blank or unparsable base_url) is no relay either — a broken route gets the safe window.
    relay_host = bool(host) and _infer_provider_from_url(base_url) is None
    for scoped_host, paths, models, ctx, owner in _ENDPOINT_SCOPED_CONTEXT:
        if host == scoped_host:
            return ctx if canonical_surface and path in paths and model_key in models else None
        if relay_host and provider_key and provider_key == owner and _relay_model_ids(model_key) & models:
            return ctx
    return None

