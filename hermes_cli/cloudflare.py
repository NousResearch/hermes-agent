"""Cloudflare Workers AI config/runtime helpers."""

from __future__ import annotations

import os
from typing import Any, Optional

from hermes_cli.config import get_compatible_custom_providers, load_config


_DEFAULT_BASE_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
_GATEWAY_BASE_TEMPLATE = "https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/workers-ai/v1"
_CLOUDFLARE_PROVIDER_ALIASES = frozenset(
    {
        "cloudflare",
        "cloudflare-workers-ai",
        "workers-ai",
        "workersai",
        "cf",
        "cf-ai",
        "custom:cloudflare",
        "custom:cloudflare-workers-ai",
        "custom:workers-ai",
        "custom:workersai",
    }
)


def is_cloudflare_provider_name(provider_id: str) -> bool:
    requested = str(provider_id or "").strip().lower()
    return requested in _CLOUDFLARE_PROVIDER_ALIASES


def _entry_looks_like_cloudflare(entry: dict[str, Any]) -> bool:
    base_url = str(entry.get("base_url") or entry.get("url") or entry.get("api") or "").strip()
    if "api.cloudflare.com/client/v4/accounts/" in base_url and "/ai/v1" in base_url:
        return True
    if "gateway.ai.cloudflare.com" in base_url:
        return True
    provider_key = str(entry.get("provider_key") or "").strip().lower()
    name = str(entry.get("name") or "").strip().lower()
    return provider_key == "cloudflare" or name == "cloudflare"


def _custom_provider_slug(display_name: str) -> str:
    return "custom:" + display_name.strip().lower().replace(" ", "-")


def _entry_matches_provider(entry: dict[str, Any], provider_id: str) -> bool:
    requested = str(provider_id or "").strip().lower()
    if not requested:
        return False
    display_name = str(entry.get("name") or "").strip()
    candidates = {
        requested,
        str(entry.get("provider_key") or "").strip().lower(),
        display_name.lower(),
        _custom_provider_slug(display_name).lower() if display_name else "",
    }
    if is_cloudflare_provider_name(requested) and _entry_looks_like_cloudflare(entry):
        return True
    return requested in candidates


def resolve_cloudflare_runtime_credentials(provider_id: str) -> dict[str, str]:
    """Resolve Cloudflare Workers AI runtime credentials from config/env.

    Priority:
      1. Custom provider config entry matching Cloudflare
      2. CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID env vars
      3. CLOUDFLARE_BASE_URL explicit override
    """
    requested = str(provider_id or "cloudflare").strip().lower() or "cloudflare"

    config = load_config()
    for entry in get_compatible_custom_providers(config):
        if not isinstance(entry, dict) or not _entry_matches_provider(entry, requested):
            continue
        api_key = str(entry.get("api_key") or "").strip()
        if not api_key:
            key_env = str(entry.get("key_env") or "").strip()
            if key_env:
                api_key = os.getenv(key_env, "").strip()
        base_url = str(entry.get("base_url") or entry.get("url") or entry.get("api") or "").strip().rstrip("/")
        if api_key and base_url:
            return {
                "provider": requested,
                "api_key": api_key,
                "base_url": base_url,
                "source": str(entry.get("name") or "custom_providers"),
            }

    api_key = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
    if not api_key:
        api_key = os.getenv("CLOUDFLARE_API_KEY", "").strip()

    base_url = os.getenv("CLOUDFLARE_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        if account_id:
            gateway_id = os.getenv("CLOUDFLARE_GATEWAY_ID", "").strip()
            if gateway_id:
                base_url = _GATEWAY_BASE_TEMPLATE.format(account_id=account_id, gateway_id=gateway_id)
            else:
                base_url = _DEFAULT_BASE_TEMPLATE.format(account_id=account_id)
    return {
        "provider": requested,
        "api_key": api_key,
        "base_url": base_url,
        "source": "environment" if api_key or base_url else "",
    }


def cloudflare_base_url_from_account_id(account_id: Optional[str]) -> str:
    account_id = str(account_id or "").strip()
    return _DEFAULT_BASE_TEMPLATE.format(account_id=account_id) if account_id else ""
