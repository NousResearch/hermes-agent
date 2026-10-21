"""Model-discovery probe helpers; ``hermes_cli.models.probe_api_models`` is the entry point."""

from __future__ import annotations

import logging
from typing import Optional

from agent.anthropic_endpoints import _requires_bearer_auth
from providers import profile_for_endpoint_host

logger = logging.getLogger(__name__)


def probe_auth_headers(api_key: Optional[str], api_mode: Optional[str], base_url: str) -> dict[str, str]:
    """The credential headers the route's client sends, so discovery and inference agree. An Anthropic
    Messages route always carries ``anthropic-version`` and sends the key as ``Authorization: Bearer``
    where the adapter does (``_requires_bearer_auth``), else as ``x-api-key``; other routes use Bearer."""
    if not api_key:
        return {}
    if api_mode != "anthropic_messages":
        return {"Authorization": f"Bearer {api_key}"}
    if _requires_bearer_auth(base_url):
        return {"Authorization": f"Bearer {api_key}", "anthropic-version": "2023-06-01"}
    return {"x-api-key": api_key, "anthropic-version": "2023-06-01"}


def endpoint_profile_models(api_key: Optional[str], base_url: str, *, timeout: float) -> Optional[list[str]]:
    """The catalog from ``fetch_models`` of the provider plugin claiming *base_url*'s host
    (``ProviderProfile.endpoint_hosts``). None means use the generic ``/models`` probe: no claim, no
    key, or the plugin raised or returned no models, so an empty list is never authoritative."""
    profile = profile_for_endpoint_host(base_url)
    if profile is None or not api_key:
        return None
    try:
        models = profile.fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)
    except Exception:
        logger.warning("provider %s catalog failed for %s; using the generic probe",
                       profile.name, base_url, exc_info=True)
        return None
    return models or None
