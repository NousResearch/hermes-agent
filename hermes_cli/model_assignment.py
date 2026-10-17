"""Persist submitted custom-endpoint credentials without inlining secrets."""

from __future__ import annotations

from hermes_cli.config import (
    custom_endpoint_key_env,
    get_env_value,
    save_env_value,
)


def persist_custom_endpoint_secret(provider: str, base_url: str, api_key: str) -> str:
    """Write a bare/named custom or local API key to ``.env``; return its ``key_env``.

    Empty string means this assignment has no custom-endpoint secret to stash
    (other providers, missing URL, or empty key). Raises if the ``.env`` write
    cannot be verified — callers must not persist a ``key_env`` pointer to a
    secret that is not actually on disk.
    """
    normalized_provider = provider.strip().lower()
    if normalized_provider not in {"custom", "local"} and not normalized_provider.startswith("custom:"):
        return ""
    if not base_url or not api_key.strip():
        return ""
    secret = api_key.strip()
    env_var = custom_endpoint_key_env(base_url)
    save_env_value(env_var, secret)
    if (get_env_value(env_var) or "").strip() != secret:
        raise RuntimeError(f"failed to persist {env_var} to .env")
    return env_var
