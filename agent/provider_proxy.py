"""Per-provider/route HTTP proxy overrides (``provider_proxies`` config).

Some LLM aggregators rate-limit per source IP. ``provider_proxies`` maps a provider
id or a base_url hostname to an HTTP proxy URL, so the matching provider's API
traffic rides that proxy (e.g. a local rotating-exit Tor HTTP proxy) while every
other host stays direct:

    provider_proxies:
      opencode-zen: http://127.0.0.1:3128   # provider-id match
      opencode.ai: http://127.0.0.1:3128    # host match

Matching order: exact provider id first, then the base_url hostname. The host form
also covers keyless-healed traffic whose provider id changes at runtime (OpenCode
Zen → opencode-free) while the endpoint stays put. An explicit entry wins over the
``HTTPS_PROXY``/``ALL_PROXY`` environment and is deliberately NOT subject to
``NO_PROXY`` (the mapping is already the precise, per-route statement).

Auxiliary calls (compression, titles, vision) resolve by host only — they carry no
provider id at http-client build time; host keys cover them.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def provider_proxy_override(provider: Optional[str], base_url: Optional[str]) -> Optional[str]:
    """Proxy URL declared for this provider/route in ``provider_proxies`` config, else ``None``.

    Malformed URLs raise ``RuntimeError`` (mirroring ``_validate_proxy_env_urls``): a typo
    must fail loudly, not silently route direct while the operator believes Tor is on.
    """
    try:
        from hermes_cli.config import load_config_readonly

        mapping = (load_config_readonly() or {}).get("provider_proxies")
    except Exception:
        return None
    if not isinstance(mapping, dict) or not mapping:
        return None

    provider_key = str(provider or "").strip().lower()
    url = None
    if provider_key:
        url = mapping.get(provider_key)
    if url is None:
        try:
            from utils import base_url_hostname

            host = str(base_url_hostname(str(base_url or "")) or "").strip().lower()
        except Exception:
            host = ""
        if host:
            url = mapping.get(host)
    if url is None:
        return None

    url = str(url).strip()
    if not url:
        return None
    parsed = urlparse(url)
    if not parsed.scheme:
        raise RuntimeError(
            f"Malformed provider_proxies entry {url!r}: missing scheme (expected e.g. http://host:port)"
        )
    try:
        _ = parsed.port  # raises ValueError for e.g. '127.0.0.1:3128x'
    except ValueError as exc:
        raise RuntimeError(f"Malformed provider_proxies entry {url!r}: {exc}") from exc
    return url


def validate_provider_proxy_urls(config: Optional[dict[str, Any]] = None) -> None:
    """Fail fast on malformed ``provider_proxies`` values (startup / client-build preflight)."""
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            return
    mapping = (config or {}).get("provider_proxies")
    if not isinstance(mapping, dict):
        return
    for url in mapping.values():
        url = str(url or "").strip()
        if not url:
            continue
        parsed = urlparse(url)
        if not parsed.scheme:
            raise RuntimeError(
                f"Malformed provider_proxies entry {url!r}: missing scheme (expected e.g. http://host:port)"
            )
        try:
            _ = parsed.port  # raises ValueError for e.g. '127.0.0.1:3128x'
        except ValueError as exc:
            raise RuntimeError(f"Malformed provider_proxies entry {url!r}: {exc}") from exc
