"""Model-provider plugins as first-class citizens of ``hermes auth`` and the credential pool.

A ``plugins/model-providers/<name>/`` profile (``providers.base.ProviderProfile``) is mirrored into
``hermes_cli.auth.PROVIDER_REGISTRY`` with the ``auth_type`` it declares, so ``resolve_provider()``
accepts it whatever its auth shape. Non-api-key plugins own their login through two optional profile
callables that core consults BEFORE any built-in, name-keyed path:

- ``auth_handler(action, args) -> bool`` for ``hermes auth add|status|logout|refresh <name>``
  (``args`` is the parsed CLI namespace; truthy = the plugin owned the action, falsy = built-in path).
- ``refresh_credential(entry) -> Mapping | None`` for the credential pool: given the pooled
  ``PooledCredential`` it returns the rotated fields (``access_token``, ``refresh_token``,
  ``expires_at_ms`` …) or raises. A separate hook rather than ``auth_handler("refresh", …)`` because
  the pool has a credential row, not an argparse namespace, and needs tokens back, not a bool.

Every function here late-imports ``hermes_cli.auth`` names: this module is imported by the auth facade
right after ``PROVIDER_REGISTRY`` exists, and by ``agent.credential_pool``.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Providers handled outside the registry: copilot/kimi/zai have bespoke token refresh in auth.py;
# openrouter/custom are aggregator/user-supplied and runtime_provider relies on
# ``openrouter not in PROVIDER_REGISTRY``.
_REGISTRY_PLUGIN_SKIP = frozenset({"copilot", "kimi-coding", "kimi-coding-cn", "zai", "openrouter", "custom"})

PLUGIN_AUTH_ACTIONS = ("add", "status", "logout", "refresh")

# Names whose PROVIDER_REGISTRY row came from a plugin profile (not the built-in rows). Only these
# can be "OAuth-shaped with nobody to log them in": a bundled OAuth provider (nous, openai-codex …)
# declares the same auth_type but its login lives in core.
PLUGIN_MIRRORED_PROVIDERS: set[str] = set()


def _api_key_env_fields(pp: Any) -> tuple[tuple, str]:
    """Split a profile's ``env_vars`` into (api-key vars, base-URL var); the URL var may be ""."""
    is_url = lambda v: v.endswith("_BASE_URL") or v.endswith("_URL")  # noqa: E731
    return (tuple(v for v in pp.env_vars if not is_url(v)) or pp.env_vars,
            next((v for v in pp.env_vars if is_url(v)), None) or "")


def register_plugin_provider(pp: Any) -> None:
    """Mirror one profile into ``PROVIDER_REGISTRY`` under the ``auth_type`` it declares.

    Registering is what lets an out-of-tree provider pass ``resolve_provider()``'s known-provider
    gate ("Unknown provider"); OAuth-shaped plugins carry their own login via ``auth_handler``.
    """
    from hermes_cli.auth import PROVIDER_REGISTRY, ProviderConfig, _api_key_provider

    if pp.name in _REGISTRY_PLUGIN_SKIP:
        return
    if pp.auth_type == "api_key":
        if not pp.env_vars:
            return
        pconfig = _api_key_provider(pp.name, pp.display_name or pp.name, pp.base_url, *_api_key_env_fields(pp))
    else:
        pconfig = ProviderConfig(pp.name, pp.display_name or pp.name, pp.auth_type, inference_base_url=pp.base_url)
    PROVIDER_REGISTRY[pp.name] = pconfig
    PLUGIN_MIRRORED_PROVIDERS.add(pp.name)
    mirror_aliases(pconfig, pp)


def _user_owns_alias(pp: Any, alias: str) -> bool:
    """True when *alias* resolves to the ``$HERMES_HOME`` plugin *pp* in the ``providers`` layer."""
    try:
        from providers import get_provider_profile, provider_source
    except Exception:
        return False
    owner = get_provider_profile(alias)
    return owner is not None and owner.name == pp.name and provider_source(pp.name) == "user"


def mirror_aliases(pconfig: Any, pp: Any) -> None:
    """Point ``pp.aliases`` at *pconfig* so ``resolve_provider()`` resolves them too.

    A bundled plugin never steals an alias another row already holds; a ``$HERMES_HOME`` plugin
    whose alias the ``providers`` layer already resolves to it does — the same ownership rule as
    :func:`override_registry_row`, otherwise the alias kept resolving to the built-in row while
    ``providers.get_provider_profile`` followed the user's profile (#116668).
    """
    from hermes_cli.auth import PROVIDER_REGISTRY

    for alias in pp.aliases:
        if alias not in PROVIDER_REGISTRY or _user_owns_alias(pp, alias):
            PROVIDER_REGISTRY[alias] = pconfig


def override_registry_row(pconfig: Any, pp: Any) -> None:
    """A ``$HERMES_HOME`` plugin re-registering a name that already has a row wins for the fields
    it declares — ``display_name``, ``base_url`` and, on api-key rows, ``env_vars`` (#48450, #116668). ``register_provider()``
    is last-writer-wins for the profile; without this the runtime kept reading the built-in
    endpoint. In place, so alias rows sharing the object follow; idempotent, so re-sync is free.
    """
    if pp.display_name:
        pconfig.name = pp.display_name
    if pp.base_url:
        pconfig.inference_base_url = pp.base_url
    if pp.auth_type == "api_key" == pconfig.auth_type and pp.env_vars:
        pconfig.api_key_env_vars, url_var = _api_key_env_fields(pp)
        if url_var:
            pconfig.base_url_env_var = url_var


def sync_plugin_provider_registry() -> int:
    """Mirror provider-plugin profiles into ``PROVIDER_REGISTRY``; return how many were added.

    Idempotent (existing entries are never replaced — a user plugin re-registering a bundled name
    only rewrites the fields it declares, see :func:`override_registry_row`), so it is safe from
    resolution paths. It runs at
    auth import and again whenever a name is missing (:func:`registry_lookup`) or when ``providers``
    finishes discovery, because the import-time pass can observe a *partial* profile list: a plugin
    whose own imports pull ``hermes_cli.auth`` in mid-``_discover_providers()`` sees only what was
    registered so far, and every later plugin would otherwise fail with "Unknown provider" (#102123).
    """
    from hermes_cli.auth import BUILTIN_PROVIDER_IDS, PROVIDER_REGISTRY

    try:
        from providers import list_providers, provider_source
        profiles = list_providers()
    except Exception:
        return 0
    added = 0
    for pp in profiles:
        if pp.name in PROVIDER_REGISTRY:
            # Only rows core wrote (built-in or mirrored) — a row the plugin injected itself is its
            # own, more specific declaration and stays as written.
            core_row = pp.name in BUILTIN_PROVIDER_IDS or pp.name in PLUGIN_MIRRORED_PROVIDERS
            if core_row and pp.name not in _REGISTRY_PLUGIN_SKIP and provider_source(pp.name) == "user":
                override_registry_row(PROVIDER_REGISTRY[pp.name], pp)
                mirror_aliases(PROVIDER_REGISTRY[pp.name], pp)
            continue
        register_plugin_provider(pp)
        added += pp.name in PROVIDER_REGISTRY
    return added


def registry_lookup(provider_id: str) -> Optional[Any]:
    """``PROVIDER_REGISTRY.get`` that re-syncs plugin profiles on a miss."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if pconfig is None and sync_plugin_provider_registry():
        pconfig = PROVIDER_REGISTRY.get(provider_id)
    return pconfig


def plugin_profile(provider: str) -> Optional[Any]:
    """The registered ``ProviderProfile`` for *provider*, or None (also when the layer is unavailable)."""
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
    """The profile's ``refresh_credential`` hook, i.e. whether its pooled OAuth rows are refreshable."""
    return _profile_hook(provider, "refresh_credential")


def is_refreshable_oauth_provider(provider: str) -> bool:
    """Built-in refreshable set OR a plugin profile shipping ``refresh_credential``."""
    from agent.credential_pool import REFRESHABLE_OAUTH_PROVIDERS

    return provider in REFRESHABLE_OAUTH_PROVIDERS or plugin_refresh_hook(provider) is not None


def dispatch_plugin_auth(action: str, args: Any, provider: str) -> bool:
    """Offer ``hermes auth <action> <provider>`` to the provider's ``auth_handler``.

    True = the plugin owned the action (core prints nothing more). False = run the built-in path. A
    handler exception becomes a readable ``SystemExit`` naming provider and action.
    """
    handler = plugin_auth_handler(provider)
    if handler is None:
        return False
    try:
        return bool(handler(action, args))
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"{provider} auth handler failed for `{action}`: {type(exc).__name__}: {exc}") from exc


def plugin_missing_auth_handler_error(provider: str, action: str) -> Optional[SystemExit]:
    """Fail loud for a registered non-api-key plugin that ships no ``auth_handler``.

    Its login is not something core can perform (there is no token endpoint to call), and silently
    running the api-key prompt or reporting "Unknown provider" both hide the plugin bug.
    """
    if provider not in PLUGIN_MIRRORED_PROVIDERS:
        return None
    profile = plugin_profile(provider)
    if profile is None or profile.auth_type == "api_key" or plugin_auth_handler(provider) is not None:
        return None
    return SystemExit(
        f"Provider '{provider}' declares auth_type '{profile.auth_type}' but its plugin ships no "
        f"auth_handler, so `hermes auth {action} {provider}` cannot be handled. Add "
        "`auth_handler=` to its ProviderProfile (see the model-provider plugin guide).")


def _pool_entry_expired(entry: Any) -> bool:
    """A pooled OAuth row is expired when its ``expires_at_ms`` / ISO ``expires_at`` is in the past."""
    import time
    from hermes_cli.auth import _parse_iso_timestamp

    if entry.expires_at_ms is not None:
        return int(entry.expires_at_ms) <= int(time.time() * 1000)
    if entry.expires_at:
        epoch = _parse_iso_timestamp(entry.expires_at)
        return epoch is not None and epoch <= time.time()
    return False


def get_plugin_oauth_auth_status(provider_id: str) -> dict[str, Any]:
    """Status for an OAuth-shaped PLUGIN provider, read from the credential pool its ``auth_handler`` fills.

    ``configured`` = the profile is registered; ``logged_in`` = a pool row carries a live token;
    ``needs_refresh`` = every token is expired but refresh material (and a ``refresh_credential`` hook)
    exists. Bundled OAuth providers keep their bespoke builders (``_BESPOKE_STATUS_FUNCTIONS``) — this
    one is gated on the plugin-mirrored set so their status bytes never change.
    """
    if provider_id not in PLUGIN_MIRRORED_PROVIDERS:
        return {"logged_in": False}
    from agent.credential_pool import load_pool

    entries = [e for e in load_pool(provider_id).entries() if (e.access_token or e.agent_key or "").strip()]
    live = [e for e in entries if not _pool_entry_expired(e)]
    refreshable = [e for e in entries if e.refresh_token] if not live and plugin_refresh_hook(provider_id) else []
    return {
        "configured": True, "provider": provider_id, "logged_in": bool(live),
        "needs_refresh": bool(refreshable), "accounts": len(entries),
        "base_url": next((e.base_url for e in live + refreshable if e.base_url), "") or "",
        "hint": "" if live else f"Run `hermes auth add {provider_id}` to sign in."}
