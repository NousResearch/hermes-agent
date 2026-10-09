"""Provider-owned authentication hooks.

Provider existence and configuration come from :mod:`providers` and
:mod:`hermes_cli.provider_auth`. This module owns only optional auth behavior
shipped by provider profiles.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from hermes_cli.provider_auth import CORE_MANAGED_AUTH_PROVIDER_IDS, get_provider_config

PLUGIN_AUTH_ACTIONS = ("add", "status", "logout", "refresh")


def plugin_profile(provider: str) -> Optional[Any]:
    try:
        from providers import get_provider_profile
    except Exception:
        return None
    return get_provider_profile(provider)


def _profile_hook(provider: str, name: str) -> Optional[Callable[..., Any]]:
    hook = getattr(plugin_profile(provider), name, None)
    return hook if callable(hook) else None


def plugin_auth_handler(provider: str) -> Optional[Callable[[str, Any], Any]]:
    return _profile_hook(provider, "auth_handler")


def plugin_refresh_hook(provider: str) -> Optional[Callable[[Any], Any]]:
    return _profile_hook(provider, "refresh_credential")


def is_refreshable_oauth_provider(provider: str) -> bool:
    from agent.credential_pool import REFRESHABLE_OAUTH_PROVIDERS

    return provider in REFRESHABLE_OAUTH_PROVIDERS or plugin_refresh_hook(provider) is not None


def dispatch_plugin_auth(action: str, args: Any, provider: str) -> bool:
    """Offer an auth action to the provider profile before core handling."""

    handler = plugin_auth_handler(provider)
    if handler is None:
        return False
    try:
        return bool(handler(action, args))
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            f"{provider} auth handler failed for `{action}`: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def plugin_missing_auth_handler_error(provider: str, action: str) -> Optional[SystemExit]:
    """Reject non-core auth shapes that do not provide their required hook."""

    profile = plugin_profile(provider)
    if (
        profile is None
        or profile.auth_type == "api_key"
        or provider in CORE_MANAGED_AUTH_PROVIDER_IDS
        or plugin_auth_handler(provider) is not None
    ):
        return None
    return SystemExit(
        f"Provider '{provider}' declares auth_type '{profile.auth_type}' but its plugin ships no "
        f"auth_handler, so `hermes auth {action} {provider}` cannot be handled. Add "
        "`auth_handler=` to its ProviderProfile (see the model-provider plugin guide)."
    )


def _pool_entry_expired(entry: Any) -> bool:
    import time

    from hermes_cli.auth import _parse_iso_timestamp

    if entry.expires_at_ms is not None:
        return int(entry.expires_at_ms) <= int(time.time() * 1000)
    if entry.expires_at:
        epoch = _parse_iso_timestamp(entry.expires_at)
        return epoch is not None and epoch <= time.time()
    return False


def get_plugin_oauth_auth_status(provider_id: str) -> dict[str, Any]:
    """Status for OAuth-shaped providers whose authentication is profile-owned."""

    config = get_provider_config(provider_id)
    if (
        config is None
        or config.auth_type not in {"oauth_device_code", "oauth_external"}
        or config.id in CORE_MANAGED_AUTH_PROVIDER_IDS
    ):
        return {"logged_in": False}

    from agent.credential_pool import load_pool

    entries = [
        entry
        for entry in load_pool(config.id).entries()
        if (entry.access_token or entry.agent_key or "").strip()
    ]
    live = [entry for entry in entries if not _pool_entry_expired(entry)]
    refreshable = (
        [entry for entry in entries if entry.refresh_token]
        if not live and plugin_refresh_hook(config.id)
        else []
    )
    return {
        "configured": True,
        "provider": config.id,
        "logged_in": bool(live),
        "needs_refresh": bool(refreshable),
        "accounts": len(entries),
        "base_url": next(
            (entry.base_url for entry in live + refreshable if entry.base_url), ""
        ) or "",
        "hint": "" if live else f"Run `hermes auth add {config.id}` to sign in.",
    }
