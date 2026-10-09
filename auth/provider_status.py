"""Read-only OAuth status snapshots; applications supply profile settings."""
from __future__ import annotations
import time
from datetime import datetime
from typing import Any, Dict, Optional
from auth import store as auth_storage, provider_state as auth_provider_state
from auth.constants import CODEX_RATE_LIMITED_CODE
from auth.pool_environment import PoolEnvironment
from auth.providers.nous_status import _pool_first_oauth_status
from auth.providers.codex import _codex_access_token_is_expiring, resolve_codex_runtime_credentials
from auth.providers.codex_quota import _codex_pool_rate_limit_status
from auth.providers.xai import _xai_access_token_is_expiring, resolve_xai_oauth_runtime_credentials
from auth.plugin_hooks import plugin_refresh_hook


def _codex_pool_rate_limited_status() -> Optional[Dict[str, Any]]:
    rate_limit = _codex_pool_rate_limit_status()
    if not rate_limit:
        return None
    return {
        "logged_in": True, "auth_store": str(auth_storage._auth_file_path()),
        "last_refresh": rate_limit.get("last_refresh"), "auth_mode": "chatgpt",
        "source": f"pool:{rate_limit.get('label') or 'unknown'}", "rate_limited": True,
        "error_code": CODEX_RATE_LIMITED_CODE,
        "error": (rate_limit.get("message")
                  or "Codex provider quota exhausted; retry after the usage limit resets."),
        "reset_at": rate_limit.get("reset_at")}


def get_codex_auth_status(*, environment: PoolEnvironment) -> Dict[str, Any]:
    """Status snapshot for Codex auth (pool first, then legacy provider state).

    Read-only by contract: status/doctor must never adopt, refresh or persist a credential (#68004)."""
    environment.require_current_scope()
    status = _pool_first_oauth_status(
        "openai-codex", is_expiring=_codex_access_token_is_expiring, auth_mode="chatgpt",
        resolve=lambda: resolve_codex_runtime_credentials(read_only=True, environment=environment),
        on_pool_miss=_codex_pool_rate_limited_status, environment=environment)
    return status

def get_xai_oauth_auth_status(*, environment: PoolEnvironment) -> Dict[str, Any]:
    environment.require_current_scope()
    # auth_mode is display/telemetry only; device-code is the only xAI OAuth flow, so report it
    # unconditionally (auth.json may still carry a legacy ``oauth_pkce`` label).
    return _pool_first_oauth_status(
        "xai-oauth", is_expiring=_xai_access_token_is_expiring, auth_mode="oauth_device_code",
        resolve=lambda: resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False), environment=environment)

def get_minimax_oauth_auth_status(*, environment: PoolEnvironment) -> Dict[str, Any]:
    """Return auth status dict for MiniMax OAuth provider."""
    environment.require_current_scope()
    state = auth_provider_state.get_provider_auth_state("minimax-oauth")
    if not state or not state.get("access_token"):
        return {"logged_in": False, "provider": "minimax-oauth"}
    try:
        token_valid = datetime.fromisoformat(state.get("expires_at", "")).timestamp() > time.time()
    except Exception:
        token_valid = True  # access_token is known non-empty here
    return {
        "logged_in": token_valid, "provider": "minimax-oauth",
        "region": state.get("region", "global"), "expires_at": state.get("expires_at")}

def _pool_entry_expired(entry: Any) -> bool:
    """A pooled OAuth row is expired when its ``expires_at_ms`` / ISO ``expires_at`` is in the past."""
    import time
    from auth.token_validation import _parse_iso_timestamp

    if entry.expires_at_ms is not None:
        return int(entry.expires_at_ms) <= int(time.time() * 1000)
    if entry.expires_at:
        epoch = _parse_iso_timestamp(entry.expires_at)
        return epoch is not None and epoch <= time.time()
    return False


def get_plugin_oauth_auth_status(provider_id: str, *, environment: PoolEnvironment) -> dict[str, Any]:
    """Status for an OAuth-shaped PLUGIN provider, read from the credential pool its ``auth_handler`` fills.

    ``configured`` = the profile is registered; ``logged_in`` = a pool row carries a live token;
    ``needs_refresh`` = every token is expired but refresh material (and a ``refresh_credential`` hook)
    exists. The presentation dispatcher gates this on plugin registration;
    bundled OAuth providers keep their bespoke status builders.
    """
    environment.require_current_scope()
    from auth.credential_pool import load_pool

    entries = [e for e in load_pool(provider_id, environment=environment).entries() if (e.access_token or e.agent_key or "").strip()]
    live = [e for e in entries if not _pool_entry_expired(e)]
    refreshable = [e for e in entries if e.refresh_token] if not live and plugin_refresh_hook(provider_id) else []
    return {
        "configured": True, "provider": provider_id, "logged_in": bool(live),
        "needs_refresh": bool(refreshable), "accounts": len(entries),
        "base_url": next((e.base_url for e in live + refreshable if e.base_url), "") or ""}
