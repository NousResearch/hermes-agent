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


def resolve_plugin_oauth_client(req: Any) -> tuple[Any, Any]:
    """``(client, model)`` for ``req.provider``, or ``(None, None)`` when signed out or transport-less.

    An explicit key from the main runtime wins, so aux shares the session's bearer.
    """
    from agent import auxiliary_client as aux
    from agent.auxiliary_model_scope import _select_pool_entry

    provider = req.provider
    api_key = aux._normalize_api_key(req.explicit_api_key)
    entry = None
    if not api_key:
        _exists, entry = _select_pool_entry(provider)
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
