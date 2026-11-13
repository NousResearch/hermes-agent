"""Official hosted API hosts that always require a credential (#154794).

Custom resolution substitutes the ``no-key-required`` placeholder for an empty key so keyless local
servers (Ollama, llama.cpp, vLLM, a LAN box) work. Pointed at a provider's OFFICIAL hosted API
(api.openai.com, generativelanguage.googleapis.com, api.anthropic.com, openrouter.ai, ...) the same
placeholder can only be refused, so every run 401s/400s and cron keeps dispatching. The hosts are
derived from the provider registry's own base URLs (built-in rows plus registered provider
profiles) instead of a second hand-kept list: a host is "official and auth-required" when a
registered provider serves it with an API-key env var or an OAuth login, and it is not a
local/LAN address (``lmstudio``'s 127.0.0.1 default stays keyless).
"""

from __future__ import annotations

import logging
from typing import Optional

from utils import base_url_hostname

logger = logging.getLogger(__name__)

# auth_types whose provider authenticates every request (an API key or an OAuth login). aws_sdk
# (Bedrock) and vertex sign requests out of band; external_process never speaks HTTP to the host.
_AUTH_REQUIRED_TYPES = ("api_key", "oauth_device_code", "oauth_external", "oauth_minimax")


def _registry_hosts() -> dict[str, tuple[str, str, tuple]]:
    """``{hostname: (provider id, display name, key env vars)}`` for every registered auth-required
    provider with a fixed, non-local inference URL. Built-in auth rows first, then provider
    profiles (OpenRouter and other plugin-only providers live only there)."""
    from agent.model_metadata import is_local_endpoint
    hosts: dict[str, tuple[str, str, tuple]] = {}

    def _add(url: str, pid: str, name: str, auth_type: str, key_envs: tuple) -> None:
        host = base_url_hostname(url or "")
        if not host or host in hosts or is_local_endpoint(url):
            return
        key_envs = tuple(v for v in (key_envs or ()) if str(v).endswith(("_KEY", "_TOKEN")))
        if auth_type == "api_key" and not key_envs:
            return
        if auth_type in _AUTH_REQUIRED_TYPES:
            hosts[host] = (pid, name or pid, key_envs)

    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        for pid, cfg in PROVIDER_REGISTRY.items():
            _add(cfg.inference_base_url, pid, cfg.name, cfg.auth_type, tuple(cfg.api_key_env_vars))
    except Exception:  # never let a registry read error turn into a refusal
        logger.debug("provider registry unavailable for hosted-host lookup", exc_info=True)
    try:
        from providers import list_providers
        for profile in list_providers():
            _add(profile.base_url, profile.name, getattr(profile, "display_name", "") or profile.name,
                 profile.auth_type, tuple(profile.env_vars))
    except Exception:
        logger.debug("provider profiles unavailable for hosted-host lookup", exc_info=True)
    return hosts


def official_hosted_provider_for(base_url: str) -> Optional[tuple[str, str, tuple]]:
    """``(provider id, display name, key env vars)`` when *base_url*'s exact hostname is a registered provider's
    official, auth-required API host, else ``None``. Exact hostname only (parsed, never substring),
    so a lookalike (``api.openai.com.evil.test``) or a path spoof (``proxy.test/api.openai.com``)
    is not treated as official and a self-hosted gateway keeps its keyless placeholder."""
    host = base_url_hostname(base_url or "")
    if not host:
        return None
    return _registry_hosts().get(host)
