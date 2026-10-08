"""Named-provider route identity for the auxiliary facade (facade size cap).

Split from ``agent.auxiliary_client``. The preserve rule decides whether a first-class or
user-defined provider keeps its name alongside an explicit base_url, and ``_named_custom_api_key``
resolves the credential that named route uses. The late import in the preserve rule reads the
facade's ``_LOCAL_SERVER_ALIASES`` at call time, so the alias set stays owned there.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional


def _preserve_provider_with_base_url(prov: Optional[str]) -> bool:
    """True when a first-class provider keeps its identity alongside an explicit base_url."""
    from agent.auxiliary_client import _LOCAL_SERVER_ALIASES

    normalized = str(prov or "").strip().lower()
    if normalized in {"", "auto", "custom"} or normalized.startswith("custom:"):
        return False
    if normalized in _LOCAL_SERVER_ALIASES:
        return True  # the custom branch applies the /v1 tail only when it still sees the alias
    # #76602 — two independent lookups, each guarded by its own try/except so a partial
    # catalog-load failure in either path doesn't suppress the other. A user-defined
    # ``providers:`` entry keeps its name alongside an explicit base_url so the named-custom
    # branch resolves the entry's key/transport instead of the anonymous ``custom`` downgrade
    # (which sends ``no-key-required`` and 401s on auth-required endpoints).
    if _builtin_provider_present(normalized):
        return True
    if _named_custom_provider_present(normalized):
        return True
    return False


def _builtin_provider_present(name: str) -> bool:
    """Look up *name* in the built-in provider registry, returning False
    (not raising) when the catalog fails to load.

    Used by ``_preserve_provider_with_base_url`` so a built-in lookup
    exception cannot suppress the parallel user-defined provider lookup
    (#76602).
    """
    try:
        from hermes_cli.providers import get_provider

        return get_provider(name) is not None
    except Exception:
        # Keep the high-risk provider-backed routes safe even if provider
        # catalog loading is unavailable during early import/test paths.
        return name in {
            "anthropic",
            "copilot",
            "copilot-acp",
            "minimax-oauth",
            "nous",
            "openai-codex",
            "qwen-oauth",
            "xai-oauth",
        }


def _named_custom_provider_present(name: str) -> bool:
    """Look up *name* in the user-defined ``providers:`` section of
    config.yaml, returning False when the config is unavailable or
    fails to load.

    Used by ``_preserve_provider_with_base_url`` so a user-defined
    provider remains preserved even when the built-in registry raises
    (parallel lookup; each side fails independently — #76602).
    """
    try:
        from hermes_cli.runtime_provider import _get_named_custom_provider

        return _get_named_custom_provider(name) is not None
    except Exception:
        # Config not loaded yet (early import paths, tests) — fail closed:
        # never widen True just because the import / load failed.
        return False


def _named_custom_api_key(custom_entry: Dict[str, Any], provider: str, custom_base: str) -> Any:
    """Credential for a named custom provider: inline api_key → key_env → key_cmd → credential pool → placeholder.
    Aux resolves named custom providers here, not via _resolve_named_custom_runtime, so key_cmd must be
    honoured at the same precedence or every aux call 401s."""
    from agent.auxiliary_client import _scoped_key_env, load_pool

    custom_key: Any = (custom_entry.get("api_key") or "").strip()
    custom_key_env = (custom_entry.get("key_env") or custom_entry.get("api_key_env") or "").strip()
    if not custom_key and custom_key_env:
        custom_key = _scoped_key_env(custom_key_env)
    custom_key_cmd = str(custom_entry.get("key_cmd", "") or "").strip()
    if custom_key_cmd:
        from agent.command_token_source import build_command_token_provider
        custom_key = build_command_token_provider(custom_key_cmd, custom_entry.get("name") or provider) or custom_key
    if not custom_key:
        with contextlib.suppress(Exception):
            from agent.credential_pool import custom_provider_pool_key_candidates
            pool_name = custom_entry.get("provider_key") or custom_entry.get("name") or provider
            for pool_key in custom_provider_pool_key_candidates(custom_base, pool_name):
                try:
                    pool = load_pool(pool_key)
                except Exception:
                    continue
                if not pool.has_credentials():
                    continue
                pool_entry = pool.select()
                if pool_entry is None:
                    continue
                pool_api_key = getattr(pool_entry, "runtime_api_key", None) or getattr(pool_entry, "access_token", "") or ""
                if str(pool_api_key).strip():
                    custom_key = str(pool_api_key).strip()
                    break
    return custom_key or "no-key-required"
