"""Owner-aware singleton seeding and pool-to-singleton token synchronization."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential, _Seeder


def seed_tokens_singleton(seed: _Seeder, auth_store: dict[str, Any]) -> None:
    """Codex / xAI: surface the auth.json ``providers.<id>.tokens`` singleton as ``device_code``.

    Hermes owns its own Codex auth state and does NOT auto-import
    ~/.codex/auth.json: refresh tokens are single-use, so sharing them with
    Codex CLI / VS Code causes refresh_token_reused races. Adoption is an
    explicit one-time prompt via `hermes auth openai-codex`.
    """
    from agent.credential_pool import AUTH_TYPE_OAUTH, _store_owns_pool_provider, label_from_token
    from hermes_cli import auth as auth_mod
    from hermes_cli.auth_oauth_grants import owned_profile_reads_root_state

    if seed.provider == "xai-oauth":
        from hermes_cli.auth_xai import _load_xai_oauth_state_with_source
        state, source_path = _load_xai_oauth_state_with_source(auth_store, include_pool=False)
    else:
        state, source_path = auth_mod._load_provider_state_with_source(auth_store, seed.provider)
    if owned_profile_reads_root_state(auth_store, seed.provider, source_path):
        return
    tokens = state.get("tokens") if isinstance(state, dict) else None
    if not (isinstance(tokens, dict) and tokens.get("access_token")):
        return
    if seed.provider == "openai-codex":
        base_url = auth_mod.DEFAULT_CODEX_BASE_URL
        custom_label = str(state.get("label") or "").strip()
    else:
        base_url = auth_mod.DEFAULT_XAI_OAUTH_BASE_URL
        custom_label = ""
    # A profile singleton and root fallback rows have different owners even when
    # both say device_code. Keep the borrowed IDs out of source-based upsert/dedup.
    protected_ids = set()
    if (auth_mod._global_auth_file_path() is not None
            and source_path is not None and auth_mod._same_path(source_path, auth_mod._auth_file_path())
            and not _store_owns_pool_provider(auth_store, seed.provider)):
        protected_ids = {entry.id for entry in seed.entries}
    seed.upsert("device_code", {
        "auth_type": AUTH_TYPE_OAUTH,
        "access_token": tokens.get("access_token", ""),
        "refresh_token": tokens.get("refresh_token"),
        "base_url": base_url,
        "last_refresh": state.get("last_refresh"),
        "label": custom_label or label_from_token(tokens.get("access_token", ""), "device_code"),
    }, protected_ids=protected_ids)


def apply_entry_to_singleton_state(provider: str, entry: PooledCredential, state: dict[str, Any]) -> bool:
    """Copy *entry*'s tokens into the provider's auth.json ``state`` in place."""
    if provider == "nous":
        from agent.credential_pool import _NOUS_EXTRA_STATE_KEYS
        state["access_token"] = entry.access_token
        for key in ("refresh_token", "expires_at", "agent_key", "agent_key_expires_at"):
            if getattr(entry, key):
                state[key] = getattr(entry, key)
        for extra_key in _NOUS_EXTRA_STATE_KEYS:
            val = entry.extra.get(extra_key)
            if val is not None:
                state[extra_key] = val
        if entry.inference_base_url:
            state["inference_base_url"] = entry.inference_base_url
        return True
    tokens = state.get("tokens")
    if not isinstance(tokens, dict):
        return False
    tokens["access_token"] = entry.access_token
    if entry.refresh_token:
        tokens["refresh_token"] = entry.refresh_token
    if entry.last_refresh:
        state["last_refresh"] = entry.last_refresh
    state.pop("last_auth_error", None)
    return True
