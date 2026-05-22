"""AIAgent credential refresh helpers — extracted from run_agent.py.

Handles OAuth credential refresh for Codex, Nous, Copilot, and Anthropic
providers. Each function takes ``agent`` (AIAgent instance) as its first
argument and preserves the test-patch contract via ``_ra()``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


def _ra():
    import run_agent
    return run_agent


def try_refresh_codex_client_credentials(agent, *, force: bool = True) -> bool:
    """Refresh Codex OAuth credentials if they're expiring."""
    if getattr(agent, "api_mode", None) != "codex_responses":
        return False
    try:
        from agent.codex_responses_adapter import _codex_credentials_expiring
        from hermes_cli.auth import (
            _auth_store_lock,
            _load_auth_store,
            _save_auth_store,
        )

        with _auth_store_lock:
            store = _load_auth_store()
            if not store:
                return False
            expiring = _codex_credentials_expiring(store)
            if not expiring:
                return False
            # Re-init the OAuth flow
            from hermes_cli.auth import _init_codex_oauth
            result = _init_codex_oauth(store)
            if result:
                agent._recreate_codex_client()
                return True
        return False
    except Exception as e:
        logger.debug("Codex credential refresh failed: %s", e)
        return False


def try_refresh_nous_client_credentials(agent, *, force: bool = True) -> bool:
    """Refresh Nous OAuth credentials if they're expiring."""
    provider = (getattr(agent, "provider", "") or "").strip().lower()
    if provider not in ("nous", "nous_portal"):
        return False
    try:
        from hermes_cli.auth import (
            _auth_store_lock,
            _load_auth_store,
            _save_auth_store,
            _is_token_expired,
        )

        with _auth_store_lock:
            store = _load_auth_store()
            if not store:
                return False
            # Check if nous token needs refresh
            tokens = store.get("tokens", {})
            nous_token = tokens.get("nous")
            if not nous_token:
                return False
            from hermes_cli.auth import _refresh_nous_portal_oauth
            result = _refresh_nous_portal_oauth(store)
            if result:
                agent._replace_primary_openai_client(reason="nous token refresh")
                return True
        return False
    except Exception as e:
        logger.debug("Nous credential refresh failed: %s", e)
        return False


def try_refresh_copilot_client_credentials(agent) -> bool:
    """Refresh GitHub Copilot OAuth token if expiring."""
    provider = (getattr(agent, "provider", "") or "").strip().lower()
    if provider != "copilot":
        return False
    try:
        from hermes_cli.copilot_auth import _ensure_copilot_token
        result = _ensure_copilot_token()
        if not result:
            return False
        # Token changed — rebuild the client so it picks up the new token
        agent._replace_primary_openai_client(reason="copilot token refresh")
        return True
    except Exception as e:
        logger.debug("Copilot credential refresh failed: %s", e)
        return False


def try_refresh_anthropic_client_credentials(agent) -> bool:
    """Refresh Anthropic OAuth credentials if available."""
    provider = (getattr(agent, "provider", "") or "").strip().lower()
    if provider not in ("anthropic",):
        return False
    try:
        from hermes_cli.auth import (
            _auth_store_lock,
            _load_auth_store,
        )

        with _auth_store_lock:
            store = _load_auth_store()
            if not store:
                return False
            from hermes_cli.auth import _REFRESH_TOKEN_MAP, _refresh_anthropic_oauth
            if "anthropic" in _REFRESH_TOKEN_MAP:
                result = _refresh_anthropic_oauth(store)
                if result:
                    agent._replace_primary_openai_client(reason="anthropic token refresh")
                    return True
        return False
    except Exception as e:
        logger.debug("Anthropic credential refresh failed: %s", e)
        return False


def swap_credential(agent, entry) -> None:
    """Swap to a different credential entry from the pool.

    Updates the agent's active credential (api_key, base_url) and rebuilds
    the OpenAI client so the next API call uses the new credential.
    """
    old_key = getattr(agent, "api_key", "")[:8]
    agent.api_key = entry.get("access_token", entry.get("api_key", ""))
    if entry.get("base_url"):
        agent.base_url = entry["base_url"]
    if entry.get("inference_base_url"):
        agent.base_url = entry["inference_base_url"]
    if entry.get("api_mode"):
        agent.api_mode = entry["api_mode"]

    new_key = agent.api_key[:8]
    logger.info(
        "Credential pool: swapped %s... -> %s... for provider=%s",
        old_key, new_key, agent.provider,
    )
    agent._replace_primary_openai_client(reason="credential pool swap")


def credential_pool_may_recover_rate_limit(agent) -> bool:
    """Check if the credential pool can auto-recover a rate-limit state.

    When a 429 exhausts one credential, the pool may have another entry
    with a different rate-limit budget.  Returns True if a swap happened.
    """
    pool = getattr(agent, "credential_pool", None)
    if pool is None:
        return False
    try:
        entry = pool.get_next()
        if entry:
            swap_credential(agent, entry)
            return True
    except Exception as e:
        logger.debug("Credential pool rate-limit recovery failed: %s", e)
    return False
