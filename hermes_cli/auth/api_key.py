"""API-key provider family (F-C2 step 5).

Third per-provider-family extraction from ``hermes_cli/auth/__init__.py``.
Unlike Nous (device-code OAuth) and Codex (OAuth via chatgpt.com),
everything else Hermes supports authenticates with an API key pulled
from environment variables: z.ai (GLM), Kimi (Moonshot), MiniMax,
GitHub Copilot (via the Copilot dedicated auth module), Lambda,
HuggingFace, and assorted other OpenAI-compatible endpoints.

This module holds the generic API-key resolver plus the two quirks
that don't fit anywhere else:

  * ``_resolve_kimi_base_url`` — Kimi accepts either
    ``api.moonshot.ai/v1`` (legacy platform keys, ``sk-``) or
    ``api.kimi.com/coding/v1`` (coding-plan keys, ``sk-kimi-``) and
    the caller has to pick the right one from the key prefix.

  * ``detect_zai_endpoint`` + ``ZAI_ENDPOINTS`` — z.ai has separate
    billing for general vs coding plans and separate global vs
    China endpoints. A key that works on one may return
    "Insufficient balance" on another, so setup probes each endpoint
    and records the first that accepts the key.

Public API (re-exported from ``hermes_cli.auth`` via ``from .api_key
import *`` at the bottom of ``__init__.py``):

    has_usable_secret
    resolve_api_key_provider_credentials
    get_api_key_provider_status
    detect_zai_endpoint
    KIMI_CODE_BASE_URL
    ZAI_ENDPOINTS

Internal helpers (also re-exported so tests patching
``hermes_cli.auth._resolve_api_key_provider_secret`` keep hitting
the canonical object):

    _resolve_api_key_provider_secret
    _resolve_kimi_base_url
    _PLACEHOLDER_SECRET_VALUES

Shared primitives (``ProviderConfig``, ``PROVIDER_REGISTRY``,
``AuthError``) stay in ``hermes_cli/auth/__init__.py`` and are
imported here. The Copilot token resolver lives in
``hermes_cli.copilot_auth`` and is imported lazily so Copilot's
transitive dependencies don't load every time an API-key provider
is resolved.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import httpx

from hermes_cli.auth.types import ApiKeyCredentials
from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    AuthError,
    ProviderConfig,
)

logger = logging.getLogger(__name__)

__all__ = [
    "KIMI_CODE_BASE_URL",
    "ZAI_ENDPOINTS",
    "_PLACEHOLDER_SECRET_VALUES",
    "_resolve_api_key_provider_secret",
    "_resolve_kimi_base_url",
    "detect_zai_endpoint",
    "get_api_key_provider_status",
    "has_usable_secret",
    "resolve_api_key_provider_credentials",
]


# =============================================================================
# Kimi dual-endpoint routing
# =============================================================================
#
# Kimi has two OpenAI-compatible endpoints: sk-kimi- prefixed keys route
# to api.kimi.com/coding/v1. Legacy keys from platform.moonshot.ai work
# on api.moonshot.ai/v1 (the default). Auto-detect when user hasn't set
# KIMI_BASE_URL explicitly.

KIMI_CODE_BASE_URL = "https://api.kimi.com/coding/v1"


def _resolve_kimi_base_url(api_key: str, default_url: str, env_override: str) -> str:
    """Return the correct Kimi base URL based on the API key prefix.

    If the user has explicitly set KIMI_BASE_URL, that always wins.
    Otherwise, sk-kimi- prefixed keys route to api.kimi.com/coding/v1.
    """
    if env_override:
        return env_override
    if api_key.startswith("sk-kimi-"):
        return KIMI_CODE_BASE_URL
    return default_url


# =============================================================================
# Generic API-key secret resolution
# =============================================================================

_PLACEHOLDER_SECRET_VALUES = {
    "*",
    "**",
    "***",
    "changeme",
    "your_api_key",
    "your-api-key",
    "placeholder",
    "example",
    "dummy",
    "null",
    "none",
}


def has_usable_secret(value: Any, *, min_length: int = 4) -> bool:
    """Return True when a configured secret looks usable, not empty/placeholder."""
    if not isinstance(value, str):
        return False
    cleaned = value.strip()
    if len(cleaned) < min_length:
        return False
    if cleaned.lower() in _PLACEHOLDER_SECRET_VALUES:
        return False
    return True


def _resolve_api_key_provider_secret(
    provider_id: str, pconfig: ProviderConfig
) -> Tuple[str, str]:
    """Resolve an API-key provider's token and indicate where it came from."""
    if provider_id == "copilot":
        # Use the dedicated copilot auth module for proper token validation
        try:
            from hermes_cli.copilot_auth import resolve_copilot_token
            token, source = resolve_copilot_token()
            if token:
                return token, source
        except ValueError as exc:
            logger.warning("Copilot token validation failed: %s", exc)
        except Exception:
            pass
        return "", ""

    for env_var in pconfig.api_key_env_vars:
        val = os.getenv(env_var, "").strip()
        if has_usable_secret(val):
            return val, env_var

    return "", ""


# =============================================================================
# z.ai endpoint detection
# =============================================================================

ZAI_ENDPOINTS = [
    # (id, base_url, default_model, label)
    ("global",        "https://api.z.ai/api/paas/v4",        "glm-5",   "Global"),
    ("cn",            "https://open.bigmodel.cn/api/paas/v4", "glm-5",   "China"),
    ("coding-global", "https://api.z.ai/api/coding/paas/v4",  "glm-4.7", "Global (Coding Plan)"),
    ("coding-cn",     "https://open.bigmodel.cn/api/coding/paas/v4", "glm-4.7", "China (Coding Plan)"),
]


def detect_zai_endpoint(api_key: str, timeout: float = 8.0) -> Optional[Dict[str, str]]:
    """Probe z.ai endpoints to find one that accepts this API key.

    Returns {"id": ..., "base_url": ..., "model": ..., "label": ...} for the
    first working endpoint, or None if all fail.
    """
    for ep_id, base_url, model, label in ZAI_ENDPOINTS:
        try:
            resp = httpx.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "stream": False,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "ping"}],
                },
                timeout=timeout,
            )
            if resp.status_code == 200:
                logger.debug("Z.AI endpoint probe: %s (%s) OK", ep_id, base_url)
                return {
                    "id": ep_id,
                    "base_url": base_url,
                    "model": model,
                    "label": label,
                }
            logger.debug("Z.AI endpoint probe: %s returned %s", ep_id, resp.status_code)
        except Exception as exc:
            logger.debug("Z.AI endpoint probe: %s failed: %s", ep_id, exc)
    return None


# =============================================================================
# Status + credential resolution
# =============================================================================

def get_api_key_provider_status(provider_id: str) -> Dict[str, Any]:
    """Status snapshot for API-key providers (z.ai, Kimi, MiniMax)."""
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "api_key":
        return {"configured": False}

    api_key = ""
    key_source = ""
    api_key, key_source = _resolve_api_key_provider_secret(provider_id, pconfig)

    env_url = ""
    if pconfig.base_url_env_var:
        env_url = os.getenv(pconfig.base_url_env_var, "").strip()

    if provider_id == "kimi-coding":
        base_url = _resolve_kimi_base_url(api_key, pconfig.inference_base_url, env_url)
    elif env_url:
        base_url = env_url
    else:
        base_url = pconfig.inference_base_url

    return {
        "configured": bool(api_key),
        "provider": provider_id,
        "name": pconfig.name,
        "key_source": key_source,
        "base_url": base_url,
        "logged_in": bool(api_key),  # compat with OAuth status shape
    }


def resolve_api_key_provider_credentials(provider_id: str) -> ApiKeyCredentials:
    """Resolve API key and base URL for an API-key provider.

    Returns dict with: provider, api_key, base_url, source.
    """
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "api_key":
        raise AuthError(
            f"Provider '{provider_id}' is not an API-key provider.",
            provider=provider_id,
            code="invalid_provider",
        )

    api_key = ""
    key_source = ""
    api_key, key_source = _resolve_api_key_provider_secret(provider_id, pconfig)

    env_url = ""
    if pconfig.base_url_env_var:
        env_url = os.getenv(pconfig.base_url_env_var, "").strip()

    if provider_id == "kimi-coding":
        base_url = _resolve_kimi_base_url(api_key, pconfig.inference_base_url, env_url)
    elif env_url:
        base_url = env_url.rstrip("/")
    else:
        base_url = pconfig.inference_base_url

    return {
        "provider": provider_id,
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "source": key_source or "default",
    }
