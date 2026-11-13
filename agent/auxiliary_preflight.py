"""Checks the auxiliary client runs before it builds a client: proxy env sanity and routing policy.

Moved out of ``agent/auxiliary_client.py`` (over its code-health size cap); that module re-exports
``_validate_proxy_env_urls`` and calls it through its own namespace, so existing patches keep working.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from utils import normalize_proxy_env_vars


def _validate_proxy_env_urls() -> None:
    """Fail fast on malformed proxy env URLs (a shell typo like ``:6153export`` otherwise surfaces as a cryptic httpx ``Invalid port``)."""
    from urllib.parse import urlparse
    normalize_proxy_env_vars()
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        value = str(os.environ.get(key) or "").strip()
        if not value:
            continue
        try:
            parsed = urlparse(value)
            if parsed.scheme:
                _ = parsed.port          # raises ValueError for e.g. '6153export'
        except ValueError as exc:
            raise RuntimeError(
                f"Malformed proxy environment variable {key}={value!r}. "
                "Fix or unset your proxy settings and try again."
            ) from exc


def check_aux_requested_route(provider: Optional[str], model: Optional[str], base_url: Optional[str]) -> None:
    """``routing_policy`` on the requested route, before any credential discovery (``RoutingPolicyError``)."""
    from hermes_cli.routing_policy import check_requested_route, check_route, current_routing_policy

    policy = current_routing_policy()
    check_requested_route(policy, requested_provider=str(provider or ""), model=str(model or ""))
    check_route(policy, provider=str(provider or ""), model=str(model or ""), base_url=str(base_url or ""))


def check_aux_resolved_route(provider: Optional[str], client: Any, model: Optional[str]) -> None:
    """``routing_policy`` on the route the resolver actually picked: ``auto`` discovery and aliases end on
    a concrete endpoint, and that endpoint/model must be allowed too. Every provider identity the client
    carries is checked, so ``auto`` landing on a denied provider is refused even when its host is not denied."""
    if client is None:
        return
    from agent.auxiliary_client import _fallback_provider_from_label
    from hermes_cli.routing_policy import check_route, current_routing_policy

    requested = str(provider or "").strip().lower()
    identities = [
        "" if requested in ("", "auto") else requested,
        # Set by auto discovery: the backend a discovery client was built for, and the route label it took.
        str(getattr(client, "_hermes_aux_backend_provider", "") or ""),
        _fallback_provider_from_label(str(getattr(client, "_hermes_aux_effective_provider", "") or "")),
    ]
    policy = current_routing_policy()
    base_url = str(getattr(client, "base_url", "") or "")
    for identity in dict.fromkeys(identities):
        check_route(policy, provider=identity, model=str(model or ""), base_url=base_url)
