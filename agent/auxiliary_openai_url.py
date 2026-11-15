"""The OpenAI-compatible URL a client built from a provider URL calls: dual-surface hosts that serve
both an Anthropic ``/anthropic`` route and an OpenAI sibling map to the sibling."""
from __future__ import annotations

import logging
from urllib.parse import urlparse

from utils import base_url_host_matches, base_url_hostname

logger = logging.getLogger("agent.auxiliary_client")


# Hosts exposing BOTH ``…/anthropic`` and a sibling OpenAI ``…/v1``. Matched on the URL *host*
# only: unconditional rewrites break Anthropic-only gateways.
_DUAL_SURFACE_ANTHROPIC_HOST_SUFFIXES = ("minimax.io", "minimax.chat", "minimaxi.com")
_DUAL_SURFACE_ANTHROPIC_HOST_PREFIXES = ("api.minimax.",)


def _is_dual_surface_anthropic_host(url: str) -> bool:
    """True when the URL's host is a known dual-surface (MiniMax-family) host."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return any(
        host == suffix or host.endswith("." + suffix) for suffix in _DUAL_SURFACE_ANTHROPIC_HOST_SUFFIXES
    ) or any(host.startswith(prefix) for prefix in _DUAL_SURFACE_ANTHROPIC_HOST_PREFIXES)


def _to_openai_base_url(base_url: str) -> str:
    """Normalize dual-surface Anthropic URLs to their OpenAI-compatible sibling.

    MiniMax-family: ``/anthropic`` → ``/v1``; ZAI Coding Plan → ``/coding/paas/v4`` (the general
    endpoint bills separately); Kimi Code ``/coding`` → ``/coding/v1`` (the OpenAI SDK path 404s
    without it). Anthropic-only gateways keep their path.
    """
    url = str(base_url or "").strip().rstrip("/")
    if base_url_hostname(url) == "api.actual.inc":
        from hermes_cli.auth import normalize_actual_base_url
        return normalize_actual_base_url(url)
    if url.endswith("/anthropic"):
        if base_url_host_matches(url, "open.bigmodel.cn") or base_url_host_matches(url, "api.z.ai"):
            rewritten = url[: -len("/anthropic")] + "/coding/paas/v4"
            logger.debug("Auxiliary client: rewrote ZAI base URL %s → %s", url, rewritten)
            return rewritten
        if _is_dual_surface_anthropic_host(url):
            rewritten = url[: -len("/anthropic")] + "/v1"
            logger.debug("Auxiliary client: rewrote dual-surface base URL %s → %s", url, rewritten)
            return rewritten
        logger.debug(
            "Auxiliary client: keeping Anthropic-only base URL %s (no dual-surface host match)", url)
        return url
    if base_url_host_matches(url, "api.kimi.com") and url.endswith("/coding"):
        rewritten = url + "/v1"
        logger.debug("Auxiliary client: rewrote Kimi base URL %s → %s", url, rewritten)
        return rewritten
    return url
