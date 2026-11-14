"""Small shared helpers for auxiliary credential-pool fallback paths."""
from __future__ import annotations

from typing import Any, Optional


def pool_runtime_base_url_for_key(entry: Any, runtime_key: Any, fallback: str = "") -> str:
    """Use a pool row endpoint only when it owns the runtime credential."""
    if entry is None:
        return str(fallback or "").strip().rstrip("/")
    entry_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    if str(entry_key or "").strip() != str(runtime_key or "").strip():
        return str(fallback or "").strip().rstrip("/")
    return pool_runtime_base_url(entry, fallback)


def pool_runtime_api_key(entry: Any) -> str:
    """Return the provider-normalized runtime key for a pool row."""
    key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    return str(key or "").strip()


def pool_runtime_base_url(entry: Any, fallback: str = "") -> str:
    """Return a pool row's provider-aware runtime endpoint."""
    if entry is None:
        return str(fallback or "").strip().rstrip("/")
    if getattr(entry, "provider", None) == "nous":
        from hermes_cli.auth import _nous_inference_env_override
        env_url = _nous_inference_env_override()
        if env_url:
            return env_url
    url = (
        getattr(entry, "runtime_base_url", None)
        or getattr(entry, "inference_base_url", None)
        or getattr(entry, "base_url", None)
        or fallback
    )
    return str(url or "").strip().rstrip("/")


def pool_credential_is_model_eligible(
    provider: str, runtime_key: Any, model: Optional[str], pool: Any = None,
) -> bool:
    """Return whether a known pool credential is not cooled for ``model``."""
    token = str(runtime_key or "").strip()
    if not token or not model or pool is None:
        if not token or not model:
            return True
        from agent.auxiliary_client import _load_pool_with_credentials
        pool = _load_pool_with_credentials(provider, " (model gate)")
        if pool is None:
            return True
    return not bool(pool.token_is_blocked(token, model=model))
