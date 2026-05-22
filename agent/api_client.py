"""AIAgent API client management — extracted from run_agent.py.

Handles OpenAI SDK client lifecycle: creation, replacement, credential
refresh, and connection health.  Each function takes ``agent`` as its
first argument.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _ra():
    import run_agent
    return run_agent


# ── Client lifecycle ──────────────────────────────────────────────


def close_openai_client(agent, client: Any, *, reason: str, shared: bool) -> None:
    """Close an OpenAI client, handling both shared and exclusive instances."""
    if client is None:
        return
    try:
        client.close()
    except Exception as e:
        logger.debug("Failed to close OpenAI client (%s): %s", reason, e)

    if not shared:
        agent._openai_clients.discard(id(client))


def replace_primary_openai_client(agent, *, reason: str) -> bool:
    """Close the current primary client and rebuild from _client_kwargs."""
    old_client = getattr(agent, "_openai_client", None)
    api_mode = getattr(agent, "api_mode", None) or ""
    try:
        from agent.agent_runtime_helpers import create_openai_client
        agent._openai_client = create_openai_client(
            agent,
            agent._client_kwargs.copy(),
            reason=reason,
            shared=False,
        )
        if old_client:
            close_openai_client(agent, old_client, reason=f"replace:{reason}", shared=False)
        return True
    except Exception as e:
        logger.error("Failed to replace primary OpenAI client: %s", e)
        agent._openai_client = old_client
        return False


def ensure_primary_openai_client(agent, *, reason: str) -> Any:
    """Return the primary client, creating it if missing."""
    client = getattr(agent, "_openai_client", None)
    if client is not None and not _is_openai_client_closed(client):
        return client
    from agent.agent_runtime_helpers import create_openai_client
    agent._openai_client = create_openai_client(
        agent,
        getattr(agent, "_client_kwargs", {}).copy(),
        reason=reason,
        shared=False,
    )
    return agent._openai_client


def create_request_openai_client(agent, *, reason: str, api_kwargs: Optional[dict] = None) -> Any:
    """Create a per-request OpenAI client (not shared across turns)."""
    from agent.agent_runtime_helpers import create_openai_client
    kwargs = getattr(agent, "_client_kwargs", {}).copy()
    if api_kwargs:
        base_url = api_kwargs.get("base_url") or kwargs.get("base_url", "")
        if base_url:
            kwargs["base_url"] = base_url
    return create_openai_client(agent, kwargs, reason=reason, shared=False)


def close_request_openai_client(agent, client: Any, *, reason: str) -> None:
    """Close a per-request OpenAI client."""
    close_openai_client(agent, client, reason=reason, shared=False)


def _is_openai_client_closed(client: Any) -> bool:
    """Check if an OpenAI client has been closed."""
    try:
        import openai
        if isinstance(client, openai.OpenAI):
            return getattr(client, "_is_closed", False)
    except ImportError:
        pass
    return False


# ── Connection health ─────────────────────────────────────────────


def cleanup_dead_connections(agent) -> bool:
    """Detect and clean up zombie TCP connections. Returns True if any were found."""
    try:
        import socket
        from urllib.parse import urlparse
        base_url = getattr(agent, "_base_url", None) or getattr(agent, "base_url", "")
        if not base_url:
            return False
        host = urlparse(base_url).hostname
        if not host:
            return False
        # Simple check: try to connect to the host on the expected port
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        try:
            s.connect((host, 443 if base_url.startswith("https") else 80))
            s.close()
            return False  # Connection works
        except (socket.timeout, ConnectionRefusedError, OSError):
            return True  # Connection is dead
        finally:
            try:
                s.close()
            except Exception:
                pass
    except Exception:
        return False


# ── Credential refresh ────────────────────────────────────────────


def try_refresh_codex_client_credentials(agent, *, force: bool = True) -> bool:
    """Refresh Codex/OAuth credentials and rebuild client if needed."""
    provider = getattr(agent, "provider", "") or ""
    api_mode = getattr(agent, "api_mode", "") or ""
    if api_mode != "codex_responses" or provider not in {"openai-codex", "xai-oauth"}:
        return False
    # Guard against silent account swap
    try:
        if provider == "openai-codex":
            from hermes_cli.auth import resolve_codex_runtime_credentials
            singleton_now = resolve_codex_runtime_credentials(refresh_if_expiring=False)
        else:
            from hermes_cli.auth import resolve_xai_oauth_runtime_credentials
            singleton_now = resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)
    except Exception as exc:
        logger.debug("%s singleton read failed: %s", provider, exc)
        return False

    singleton_key = str(singleton_now.get("api_key") or "").strip()
    active_key = str(getattr(agent, "api_key", "") or "").strip()
    if singleton_key and active_key and singleton_key != active_key:
        logger.debug("%s singleton differs from active; skipping force-refresh", provider)
        return False

    try:
        if provider == "openai-codex":
            from hermes_cli.auth import resolve_codex_runtime_credentials
            creds = resolve_codex_runtime_credentials(force_refresh=force)
        else:
            from hermes_cli.auth import resolve_xai_oauth_runtime_credentials
            creds = resolve_xai_oauth_runtime_credentials(force_refresh=force)
    except Exception as exc:
        logger.debug("%s credential refresh failed: %s", provider, exc)
        return False

    api_key = creds.get("api_key")
    base_url = creds.get("base_url")
    if not isinstance(api_key, str) or not api_key.strip():
        return False
    if not isinstance(base_url, str) or not base_url.strip():
        return False

    agent.api_key = api_key.strip()
    agent.base_url = base_url.strip().rstrip("/")
    agent._client_kwargs["api_key"] = agent.api_key
    agent._client_kwargs["base_url"] = agent.base_url

    return replace_primary_openai_client(agent, reason=f"{provider}_credential_refresh")


def try_refresh_nous_client_credentials(agent, *, force: bool = True) -> bool:
    """Refresh Nous credentials and rebuild the OpenAI client."""
    if getattr(agent, "api_mode", "") != "chat_completions" or getattr(agent, "provider", "") != "nous":
        return False
    try:
        from hermes_cli.auth import (
            NOUS_INFERENCE_AUTH_MODE_AUTO,
            NOUS_INFERENCE_AUTH_MODE_LEGACY,
            resolve_nous_runtime_credentials,
        )
        creds = resolve_nous_runtime_credentials(
            min_key_ttl_seconds=max(60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800"))),
            timeout_seconds=float(os.getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
            inference_auth_mode=(
                NOUS_INFERENCE_AUTH_MODE_LEGACY if force else NOUS_INFERENCE_AUTH_MODE_AUTO
            ),
        )
    except Exception as exc:
        logger.debug("Nous credential refresh failed: %s", exc)
        return False

    api_key = creds.get("api_key")
    base_url = creds.get("base_url")
    if not isinstance(api_key, str) or not api_key.strip():
        return False
    if not isinstance(base_url, str) or not base_url.strip():
        return False

    agent.api_key = api_key.strip()
    agent.base_url = base_url.strip().rstrip("/")
    agent._client_kwargs["api_key"] = agent.api_key
    agent._client_kwargs["base_url"] = agent.base_url
    agent._client_kwargs.pop("default_headers", None)

    return replace_primary_openai_client(agent, reason="nous_credential_refresh")


def try_refresh_copilot_client_credentials(agent) -> bool:
    """Refresh GitHub Copilot OAuth token if expiring."""
    if getattr(agent, "provider", "") != "copilot":
        return False
    try:
        from hermes_cli.copilot_auth import _ensure_copilot_token
        result = _ensure_copilot_token()
        if not result:
            return False
        return replace_primary_openai_client(agent, reason="copilot_token_refresh")
    except Exception as e:
        logger.debug("Copilot credential refresh failed: %s", e)
        return False


def try_refresh_anthropic_client_credentials(agent) -> bool:
    """Refresh Anthropic OAuth credentials if available."""
    if getattr(agent, "provider", "") not in ("anthropic",):
        return False
    try:
        from hermes_cli.auth import _auth_store_lock, _load_auth_store
        with _auth_store_lock():
            store = _load_auth_store()
            if not store:
                return False
            from hermes_cli.auth import _REFRESH_TOKEN_MAP, _refresh_anthropic_oauth
            if "anthropic" in _REFRESH_TOKEN_MAP:
                result = _refresh_anthropic_oauth(store)
                if result:
                    return replace_primary_openai_client(agent, reason="anthropic_token_refresh")
        return False
    except Exception as e:
        logger.debug("Anthropic credential refresh failed: %s", e)
        return False


def swap_credential(agent, entry) -> None:
    """Swap to a different credential entry from the pool."""
    old_key = getattr(agent, "api_key", "")[:8]
    agent.api_key = entry.get("access_token", entry.get("api_key", ""))
    if entry.get("base_url"):
        agent.base_url = entry["base_url"]
    if entry.get("inference_base_url"):
        agent.base_url = entry["inference_base_url"]
    if entry.get("api_mode"):
        agent.api_mode = entry["api_mode"]

    new_key = agent.api_key[:8]
    logger.info("Credential pool: swapped %s... -> %s... for provider=%s", old_key, new_key, agent.provider)
    replace_primary_openai_client(agent, reason="credential_pool_swap")


def credential_pool_may_recover_rate_limit(agent) -> bool:
    """Check if the credential pool can auto-recover a rate-limit state."""
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
