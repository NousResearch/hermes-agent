"""Auxiliary client for OAuth-shaped model-provider plugins (``auth_type`` oauth_external / oauth_device_code).

Built-in OAuth routes (nous, openai-codex, xai-oauth) have their own branches in ``auxiliary_client``.
A plugin OAuth provider reaches this one: its credential is the pooled row ``hermes auth add <name>``
wrote, and its transport comes from ``ProviderProfile.create_client``. ``CredentialPool.select()``
rotates an expiring row before leasing it, so compression and titling never send a dead bearer.
"""

from __future__ import annotations

from typing import Any


def _configured_endpoint(provider: str) -> str:
    """The endpoint the main runtime would use: ``model.base_url`` when ``model.provider`` is this
    provider (a relay / proxy override), else the registered profile's own."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.runtime_provider import _config_base_url_for_provider, _get_model_config
    pconfig = PROVIDER_REGISTRY.get(provider)
    return _config_base_url_for_provider(_get_model_config(), provider) or (pconfig.inference_base_url if pconfig else "")


def is_plugin_oauth_pool(provider: str) -> bool:
    """The provider's pooled OAuth rows refresh through its profile's ``refresh_credential`` hook."""
    from hermes_cli.auth_plugin_providers import plugin_refresh_hook
    return plugin_refresh_hook(provider) is not None


def refresh_pooled_row(provider: str, failed_api_key: str) -> bool:
    """Force-refresh the pooled row that sent ``failed_api_key`` — the refresh the main agent's pool
    performs on a 401. False when no row holds that key: a key the pool never issued is not its to fix,
    and spending another row's single-use refresh token for it would lose that sign-in."""
    from agent import auxiliary_client as aux

    pool = aux._load_pool_with_credentials(provider, " (auth refresh)") if failed_api_key else None
    if pool is None or not pool.entry_id_for_api_key(failed_api_key):
        return False
    return pool.try_refresh_matching(api_key_hint=failed_api_key) is not None


def pool_superseded_key(provider: str, api_key: Any) -> bool:
    """True when a plugin OAuth pool no longer vouches for the session's ``api_key``: the row that held
    it has since been refreshed to a new token, or its stored expiry has passed. Pinning that key would
    send a bearer the pool already knows is dead; leasing from the pool refreshes it first."""
    if not isinstance(api_key, str) or not api_key or not is_plugin_oauth_pool(provider):
        return False
    from agent import auxiliary_client as aux
    from agent.credential_pool_plugin import plugin_row_is_expiring

    pool = aux._load_pool_with_credentials(provider, " (session key check)")
    if pool is None:
        return False
    entry_id = pool.entry_id_for_api_key(api_key)
    entry = next((e for e in pool.entries() if e.id == entry_id), None) if entry_id else None
    return entry is None or plugin_row_is_expiring(entry)


def resolve_plugin_oauth_client(req: Any) -> tuple[Any, Any]:
    """``(client, model)`` for ``req.provider``, or ``(None, None)`` when signed out or transport-less.

    An explicit key from the main runtime wins, so aux shares the session's bearer.
    """
    from agent import auxiliary_client as aux

    provider = req.provider
    api_key = aux._normalize_api_key(req.explicit_api_key)
    entry = None
    if not api_key:
        _exists, entry = aux._select_pool_entry(provider)
        api_key = str(getattr(entry, "runtime_api_key", "") or "") if entry is not None else ""
    base_url = (req.explicit_base_url or str(getattr(entry, "runtime_base_url", "") or "")
                or _configured_endpoint(provider)).strip().rstrip("/")
    client = aux._api_key_profile_supplied_client(provider, api_key=api_key, base_url=base_url) if api_key else None
    if client is None:
        aux._log_once_debug(aux._LOGGED_UNSUPPORTED_OAUTH_KEYS, provider,
                            "resolve_provider_client: OAuth provider %s has no signed-in credential or "
                            "plugin transport, try 'auto'", provider)
        return None, None
    model = aux._normalize_resolved_model(req.model or aux._get_aux_model_for_provider(provider), provider)
    return aux._route_client(req, client, model)
