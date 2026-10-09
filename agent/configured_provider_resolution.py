"""Application access to canonical configured-provider facts."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from providers import match_configured_provider

logger = logging.getLogger(__name__)


def get_configured_provider_entry(
    requested_provider: str,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Load config and project one matched provider declaration without resolving secrets."""

    from hermes_cli.config import get_compatible_custom_providers, load_config, normalize_extra_headers
    if config is None:
        config = load_config()
    if not isinstance(config, dict):
        return None

    legacy = get_compatible_custom_providers(config)

    match = match_configured_provider(
        requested_provider,
        providers=config.get("providers"),
        custom_providers=legacy,
    )
    if match is None:
        return None

    raw = match.raw
    result: Dict[str, Any] = {
        "name": match.name,
        "base_url": match.base_url,
        "api_key": str(raw.get("api_key") or "").strip(),
    }
    if match.provider_key:
        result["provider_key"] = match.provider_key
    key_env = str(raw.get("key_env") or raw.get("api_key_env") or "").strip()
    if key_env:
        result["key_env"] = key_env
    key_cmd = str(raw.get("key_cmd") or "").strip()
    if key_cmd:
        result["key_cmd"] = key_cmd
    if match.model:
        result["model"] = match.model
    if match.api_mode:
        result["api_mode"] = match.api_mode
    if match.capabilities:
        result["capabilities"] = dict(match.capabilities)
    if match.extra_body:
        result["extra_body"] = dict(match.extra_body)
    extra_headers = normalize_extra_headers(raw.get("extra_headers"))
    if extra_headers:
        result["extra_headers"] = extra_headers
    return result
