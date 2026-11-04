"""Adopt same-account xAI credentials before the session override's keyed fast path."""

from __future__ import annotations

import logging

from agent.xai_oauth_credentials import (
    _same_xai_oauth_account, _xai_oauth_route_is_trusted, _xai_oauth_token_is_usable,
)
from hermes_cli.route_identity import normalize_route_base_url

logger = logging.getLogger("gateway.run")


def refresh_xai_session_override(override: dict) -> None:
    """Refresh a managed xAI bearer without changing the session's account or endpoint.

    A /model snapshot otherwise bypasses provider resolution on every turn. Read the
    singleton first so an explicit credential from another account cannot refresh or
    adopt that grant. The live provider resolver then supplies a coherent bearer/pool.
    """
    old_key = override.get("api_key")
    if (
        override.get("provider") != "xai-oauth"
        or override.get("requested_provider") not in (None, "", "xai-oauth")
        or override.get("api_mode") != "codex_responses"
        or not _xai_oauth_route_is_trusted(override.get("base_url"))
        or not _same_xai_oauth_account(old_key, old_key)
    ):
        return
    from gateway.run import _resolve_runtime_agent_kwargs_for_provider
    from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

    try:
        singleton = resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)
        if not _same_xai_oauth_account(old_key, singleton.get("api_key")):
            return
        runtime = _resolve_runtime_agent_kwargs_for_provider(
            "xai-oauth", target_model=override.get("model") or None,
        )
    except (RuntimeError, OSError) as exc:
        # Keep the existing snapshot if resolution is temporarily unavailable; bounded
        # turn recovery still owns an authentication failure from the active client.
        logger.debug("xAI session override credential refresh unavailable (%s)", type(exc).__name__)
        return

    fresh_key = runtime.get("api_key")
    if (
        runtime.get("provider") != "xai-oauth"
        or runtime.get("requested_provider") not in (None, "", "xai-oauth")
        or runtime.get("api_mode") != "codex_responses"
        or normalize_route_base_url(runtime.get("base_url")) != normalize_route_base_url(override.get("base_url"))
        or not _same_xai_oauth_account(old_key, fresh_key)
        or not _xai_oauth_token_is_usable(fresh_key)
    ):
        return
    # Clearing an absent pool is intentional: never pair the new bearer with an old pool.
    override.update(api_key=fresh_key, credential_pool=runtime.get("credential_pool"))
