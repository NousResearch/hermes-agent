"""Auxiliary client for OAuth-shaped model-provider plugins (``auth_type`` oauth_external / oauth_device_code).

Built-in OAuth routes (nous, openai-codex, xai-oauth) have their own branches in ``auxiliary_client``.
A plugin OAuth provider reaches this one: its credential is the pooled row ``hermes auth add <name>``
wrote, and a native ``ProviderProfile.create_client`` takes precedence over the standard transport. ``CredentialPool.select()``
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
    """``(client, model)`` for ``req.provider``, or ``(None, None)`` when signed out.

    An explicit key from the main runtime wins, so aux shares the session's bearer.
    """
    from agent import auxiliary_client as aux
    from agent.auxiliary_client_registry import _api_key_profile_supplied_client

    provider = req.provider
    api_key = aux._normalize_api_key(req.explicit_api_key)
    entry = None
    if not api_key:
        _exists, entry = aux._select_pool_entry(provider)
        api_key = str(getattr(entry, "runtime_api_key", "") or "") if entry is not None else ""
    base_url = (req.explicit_base_url or str(getattr(entry, "runtime_base_url", "") or "")
                or _configured_endpoint(provider)).strip().rstrip("/")
    client = _api_key_profile_supplied_client(provider, api_key=api_key, base_url=base_url) if api_key else None
    if client is None and api_key:
        return _resolve_standard_oauth_client(req)
    if client is None:
        aux._log_once_debug(aux._LOGGED_UNSUPPORTED_OAUTH_KEYS, provider,
                            "resolve_provider_client: OAuth provider %s has no signed-in credential or "
                            "plugin transport, try 'auto'", provider)
        return None, None
    model = aux._normalize_resolved_model(req.model or aux._get_aux_model_for_provider(provider), provider)
    return aux._route_client(req, client, model)


def _resolve_standard_oauth_client(req: Any) -> tuple[Any, Any]:
    """Standard OAuth uses the same credential, endpoint and transport as the main loop."""
    from agent import auxiliary_client as aux
    from hermes_cli.auth_constants import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from providers import get_provider_profile

    try:
        runtime = resolve_runtime_provider(
            requested=req.provider, target_model=req.model,
            explicit_api_key=req.explicit_api_key, explicit_base_url=req.explicit_base_url,
        )
    except AuthError:
        aux._log_once_debug(aux._LOGGED_UNSUPPORTED_OAUTH_KEYS, req.provider,
                            "Auxiliary OAuth provider %s has no usable credential", req.provider)
        return None, None
    base_url, api_key = runtime["base_url"], runtime["api_key"]
    model = aux._normalize_resolved_model(req.model or aux._get_aux_model_for_provider(req.provider), req.provider)
    if not model:
        return None, None
    headers = aux._endpoint_default_headers(base_url, req.provider, is_vision=req.is_vision)
    client = aux._create_openai_client(api_key=api_key, base_url=base_url,
                                     **({"default_headers": headers} if headers else {}))
    client._hermes_aux_effective_provider = req.provider
    profile = get_provider_profile(req.provider)
    api_mode = runtime["api_mode"] if profile and profile.fixed_api_mode else req.api_mode or runtime["api_mode"]
    routed = req._replace(api_mode=api_mode)
    return aux._route_client(routed, aux._wrap_transport(routed, client, model, base_url, api_key), model)
