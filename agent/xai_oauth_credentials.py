"""Identity and freshness checks for credentials already held by an xAI OAuth session."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from httpx import HTTPError

from hermes_cli.auth_constants import AuthError, _decode_jwt_claims
from hermes_cli.auth_xai import _xai_jwt_exp, _xai_url_problem

logger = logging.getLogger(__name__)


def _same_xai_oauth_account(old_token: Any, new_token: Any) -> bool:
    """Compare local credential subjects, without treating decoded JWTs as authenticated.

    Both tokens come from the active session or its profile's auth store. This is an account
    selection guard; xAI still validates the bearer on the request. Missing identity fails closed.
    """
    old_sub = _decode_jwt_claims(old_token).get("sub")
    new_sub = _decode_jwt_claims(new_token).get("sub")
    return bool(isinstance(old_sub, str) and old_sub.strip() and old_sub == new_sub)


def _xai_oauth_token_is_usable(token: Any) -> bool:
    """Whether local expiry metadata permits an adoption attempt; the server remains authoritative."""
    expires_at = _xai_jwt_exp(token)
    return expires_at is not None and math.isfinite(expires_at) and expires_at > time.time()


def _xai_oauth_route_is_trusted(base_url: Any) -> bool:
    """Use the OAuth resolver's existing HTTPS/x.ai origin boundary for live client refreshes."""
    if not isinstance(base_url, str) or not base_url.strip():
        return False
    try:
        problem, _host = _xai_url_problem(base_url)
    except ValueError:
        return False
    return problem is None


def _resolve_xai_oauth_recovery_credentials(old_key: str, *, force: bool) -> dict[str, Any] | None:
    from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

    try:
        stored = resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)
        stored_key = stored.get("api_key")
        if stored_key != old_key:
            if not _same_xai_oauth_account(old_key, stored_key):
                return None
            if _xai_oauth_token_is_usable(stored_key):
                # A sibling already rotated the grant; do not consume its single-use refresh token again.
                return stored
        refreshed = resolve_xai_oauth_runtime_credentials(force_refresh=force)
    except (AuthError, OSError, ValueError, HTTPError) as exc:
        logger.debug("xAI OAuth credential recovery failed (%s)", type(exc).__name__)
        return None
    # A login can replace the singleton between our read and the resolver's locked refresh.
    # Keep the original opaque-token refresh path, but never move a known JWT identity to another account.
    if _same_xai_oauth_account(old_key, old_key) and not (
        _same_xai_oauth_account(old_key, refreshed.get("api_key"))
        and _xai_oauth_token_is_usable(refreshed.get("api_key"))
    ):
        return None
    return refreshed


def _sync_xai_oauth_key_holders(agent, old_key: str, new_key: str) -> None:
    """Keep a fallback restore or same-turn auxiliary call from resurrecting the rejected bearer."""
    compressor = getattr(agent, "context_compressor", None)
    if (compressor is not None and getattr(compressor, "provider", None) == "xai-oauth"
            and getattr(compressor, "api_key", None) == old_key):
        compressor.api_key = new_key
    snapshot = getattr(agent, "_primary_runtime", None) or {}
    if snapshot.get("provider") == "xai-oauth":
        if snapshot.get("api_key") == old_key:
            snapshot["api_key"] = new_key
        kwargs = snapshot.get("client_kwargs") or {}
        if kwargs.get("api_key") == old_key:
            kwargs["api_key"] = new_key
    if snapshot.get("compressor_provider") == "xai-oauth" and snapshot.get("compressor_api_key") == old_key:
        snapshot["compressor_api_key"] = new_key
    from agent.auxiliary_client import _RUNTIME_MAIN_CONTEXT
    from agent.auxiliary_key_rotation import rotate_runtime_main_api_key
    runtime = _RUNTIME_MAIN_CONTEXT.get()
    if isinstance(runtime, dict) and runtime.get("provider") == "xai-oauth":
        rotate_runtime_main_api_key(old_key, new_key)


def refresh_xai_oauth_client_credentials(agent, *, force: bool = True) -> bool:
    """Catch a live client up to its account's stored grant, or refresh that grant once."""
    if not _xai_oauth_route_is_trusted(agent.base_url):
        return False
    old_key = str(agent.api_key or "").strip()
    creds = _resolve_xai_oauth_recovery_credentials(old_key, force=force)
    if creds is None:
        return False
    api_key, base_url = creds.get("api_key"), creds.get("base_url")
    if not isinstance(api_key, str) or not api_key.strip() or not _xai_oauth_route_is_trusted(base_url):
        return False
    api_key = api_key.strip()
    if api_key == old_key:
        return False
    old_api_key, old_base_url, old_kwargs = agent.api_key, agent.base_url, dict(agent._client_kwargs)
    if not agent._adopt_openai_credentials(api_key, base_url, reason="xai-oauth_credential_refresh"):
        # The common adoption method stages fields before constructing the client. A failed build
        # must leave the old client and its metadata paired, so the next recovery sees the failed bearer.
        agent.api_key, agent.base_url, agent._client_kwargs = old_api_key, old_base_url, old_kwargs
        return False
    _sync_xai_oauth_key_holders(agent, old_key, api_key)
    return True
