"""Copy rotated pool credentials back into their owning singleton state."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential


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
