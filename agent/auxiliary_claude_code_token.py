"""Follow a Claude Code subscription token that `claude` refreshed under a running session.

Split from ``agent.auxiliary_client`` (facade size cap). Two consumers there:
``_refresh_anthropic_credentials`` (a 401 on a superseded token is recoverable: the
replacement on disk IS the recovery) and the main-runtime route (swap the stale session
key before the call, so the 401 never happens).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def is_superseded_claude_code_token(token: Any, creds: Any = None) -> bool:
    """True when ``token`` is a Claude Code OAuth token that is no longer the one on disk."""
    from agent.anthropic_credentials import _is_oauth_token, read_claude_code_credentials

    if not isinstance(token, str) or not token or not _is_oauth_token(token):
        return False
    if creds is None:
        creds = read_claude_code_credentials()
    live = creds.get("accessToken") if isinstance(creds, dict) else None
    return isinstance(live, str) and bool(live) and live != token


def refresh_anthropic_credentials(failed_api_key: str = "") -> bool:
    """Make the next attempt after an Anthropic 401 use a working credential; True when one exists.

    Moved from ``agent.auxiliary_client`` (facade size cap), which re-exports it as
    ``_refresh_anthropic_credentials`` for its refresher table.
    """
    from agent.anthropic_credentials import _refresh_oauth_token, read_claude_code_credentials
    from agent.auxiliary_client import load_pool

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
    # The failed token is a subscription token that `claude` has since replaced: the
    # replacement on disk IS the recovery. The retry rebuilds through the main route, where
    # supersede_stale_anthropic_oauth_key swaps it in.
    return is_superseded_claude_code_token(token, creds)


def supersede_stale_anthropic_oauth_key(api_key: Any, base_url: str) -> Any:
    """Swap a superseded Claude Code token in the session's runtime for the one on disk.

    Auxiliary tasks route onto the main session with the key captured when the agent was
    built. The main client mints its bearer per request, so it never 401s and nothing ever
    updates that captured key: hours after `claude` refreshes the subscription token, every
    auxiliary call (compression above all) presents the replaced one and gets 401 "OAuth
    access token has been revoked" while the token on disk works.

    Only a native-Anthropic OAuth token that the pool does not own is swapped. A pooled
    credential is pinned on purpose (a second account), an API key does not rotate, and a
    third-party anthropic_messages endpoint must never receive the ~/.claude token (#1739).
    """
    # Late facade import: tests and callers patch these names on agent.auxiliary_client.
    from agent.auxiliary_client import base_url_host_matches, load_pool

    if not isinstance(api_key, str) or not base_url_host_matches(base_url or "", "anthropic.com"):
        return api_key
    try:
        if load_pool("anthropic").entry_id_for_api_key(api_key):
            return api_key
        from agent.anthropic_credentials import read_claude_code_credentials

        creds = read_claude_code_credentials()
        if isinstance(creds, dict) and is_superseded_claude_code_token(api_key, creds):
            logger.info("Auxiliary: session's Claude Code token was refreshed on disk; using the current one")
            return creds["accessToken"]
    except Exception:  # health: allow BLE001 -- a pool/credential read failure keeps the session key, never ends the call
        logger.debug("Auxiliary: could not check the session's Claude Code token", exc_info=True)
    return api_key
