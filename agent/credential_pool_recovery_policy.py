"""Small provider-specific decisions for credential-pool error recovery."""

from typing import Any

from agent.credential_pool import AUTH_TYPE_OAUTH


def rotate_anthropic_oauth_on_first_429(pool: Any, current_entry: Any) -> bool:
    """Prefer another distinct Anthropic OAuth login over retrying this one.

    Actual availability, per-model cooldowns and endpoint compatibility are
    still decided by the pool's normal mark_exhausted_and_rotate() path.
    """
    if getattr(pool, "provider", None) != "anthropic" or current_entry is None:
        return False
    if getattr(current_entry, "auth_type", None) != AUTH_TYPE_OAUTH:
        return False
    current_key = getattr(current_entry, "runtime_api_key", None)
    return any(
        entry.id != current_entry.id
        and entry.auth_type == AUTH_TYPE_OAUTH
        and bool(entry.runtime_api_key)
        and entry.runtime_api_key != current_key
        for entry in pool.entries()
    )
