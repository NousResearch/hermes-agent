"""Credential refresh helpers for auxiliary providers.

This module owns provider-specific refresh mechanics only. Cache eviction and
route normalization stay in ``agent.auxiliary_client`` so this helper has no
import cycle with the client cache.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

from agent.credential_pool import load_pool
from utils import env_float


def creds_have_api_key(creds: Optional[Union[Dict[str, Any], Tuple[Any, ...]]]) -> bool:
    """Return whether a provider credential payload contains a usable runtime key."""
    if creds is None:
        return False
    if isinstance(creds, dict):
        return bool(str(creds.get("api_key", "") or "").strip())
    if isinstance(creds, tuple) and creds:
        return bool(str(creds[0] or "").strip())
    return False


def refresh_copilot_credentials() -> bool:
    """Refresh GitHub Copilot credentials and clear the exchanged-JWT cache."""
    from hermes_cli.copilot_auth import (
        _jwt_cache,
        _token_fingerprint,
        exchange_copilot_token,
        resolve_copilot_token,
    )

    raw_token, _source = resolve_copilot_token()
    if not str(raw_token or "").strip():
        return False
    _jwt_cache.pop(_token_fingerprint(raw_token), None)
    exchange_copilot_token(raw_token)
    return True


def refresh_codex_credentials(force_refresh: bool = True) -> bool:
    """Refresh OpenAI Codex runtime credentials."""
    from hermes_cli.auth import resolve_codex_runtime_credentials

    return creds_have_api_key(resolve_codex_runtime_credentials(force_refresh=force_refresh))


def refresh_nous_credentials(
    timeout_seconds: float = env_float("HERMES_NOUS_TIMEOUT_SECONDS", 15),
    force_refresh: bool = True,
) -> bool:
    """Refresh Nous runtime credentials."""
    from hermes_cli.auth import resolve_nous_runtime_credentials

    return creds_have_api_key(
        resolve_nous_runtime_credentials(timeout_seconds=timeout_seconds, force_refresh=force_refresh)
    )


def refresh_anthropic_credentials(failed_api_key: str = "") -> bool:
    """Refresh the Anthropic credential matching ``failed_api_key`` when possible."""
    from agent.anthropic_credentials import _refresh_oauth_token, read_claude_code_credentials

    token = failed_api_key
    if not token:
        return False
    pool = load_pool("anthropic")
    if pool.entry_id_for_api_key(token):
        return pool.try_refresh_matching(api_key_hint=token) is not None
    creds = read_claude_code_credentials()
    # Never spend an ambient login's refresh rotation for another request's key.
    if isinstance(creds, dict) and creds.get("accessToken") == token and creds.get("refreshToken"):
        return bool(_refresh_oauth_token(creds))
    return False


def refresh_xai_oauth_credentials(force_refresh: bool = True) -> bool:
    """Refresh xAI OAuth credentials; prefer pool entries before singleton auth store."""
    pool = load_pool("xai-oauth")
    if pool and pool.has_credentials():
        pool.select()
        refreshed = pool.try_refresh_current()
        if refreshed is not None and str(getattr(refreshed, "runtime_api_key", "") or "").strip():
            return True
    from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

    return creds_have_api_key(resolve_xai_oauth_runtime_credentials(force_refresh=force_refresh))


def refresh_vertex_credentials() -> bool:
    """Refresh Vertex credentials by resolving a current bearer token and base URL."""
    from agent.vertex_adapter import get_vertex_config

    token, base_url = get_vertex_config()
    return bool(isinstance(token, str) and token.strip() and isinstance(base_url, str) and base_url.strip())
