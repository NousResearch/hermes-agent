"""Runtime route materialization for already selected provider identities.

Credential source selection belongs to auth.api_keys. Provider/endpoint policy
retains its existing application owner; this module does not select credentials.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict
from auth.api_keys import resolve_api_key_provider_secret
from auth.errors import AuthError
from auth.constants import LMSTUDIO_NOAUTH_PLACEHOLDER, ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
from hermes_cli.config_credentials import credential_pool_environment as _phase6_auth_environment
from hermes_cli.auth_zai_kimi import _resolve_kimi_base_url, _resolve_zai_base_url, _normalize_lmstudio_runtime_base_url

logger = logging.getLogger(__name__)

from hermes_cli.route_identity import is_actual_local_base_url, normalize_actual_base_url

def _default_api_key_base_url(api_key: str, default: str, env_url: str) -> str:
    return env_url.rstrip("/") if env_url else default


def _copilot_runtime_base_url(api_key: str, default: str, env_url: str) -> str:
    """Copilot's API base comes from the token-exchange response (endpoints.api, proxy-ep fallback),
    authoritative for Enterprise / proxied accounts; falls back to the registry default."""
    base_url = _default_api_key_base_url(api_key, default, env_url)
    try:
        from auth.providers.copilot import resolve_copilot_token, get_copilot_api_token
        raw_token, _ = resolve_copilot_token()
        if raw_token:
            resolved = (get_copilot_api_token(raw_token)[1] or "").strip()
            if resolved:
                base_url = resolved
    except Exception as exc:
        logger.debug("Copilot base URL resolution fell back to default: %s", exc)
    return base_url


_API_KEY_BASE_URL_RESOLVERS: Dict[str, Callable[[str, str, str], str]] = {
    "kimi-coding": _resolve_kimi_base_url,
    "kimi-coding-cn": _resolve_kimi_base_url,
    "zai": _resolve_zai_base_url,
    "copilot": _copilot_runtime_base_url,
    "lmstudio": lambda *a: _normalize_lmstudio_runtime_base_url(_default_api_key_base_url(*a)),
    "actual": lambda *a: normalize_actual_base_url(_default_api_key_base_url(*a))}


def resolve_api_key_provider_credentials(provider_id: str) -> Dict[str, Any]:
    """Resolve API key and base URL for an API-key provider."""
    from hermes_cli.auth import _registry_lookup

    pconfig = _registry_lookup(provider_id)
    if not pconfig or pconfig.auth_type != "api_key":
        raise AuthError(
            f"Provider '{provider_id}' is not an API-key provider.",
            provider=provider_id, code="invalid_provider")

    api_key, key_source = resolve_api_key_provider_secret(provider_id, pconfig, environment=_phase6_auth_environment())
    # No-auth LM Studio: a placeholder so runtime / auxiliary_client see the local server as
    # configured. doctor still reports unconfigured because the status path uses the raw secret.
    if not api_key and provider_id == "lmstudio":
        api_key = LMSTUDIO_NOAUTH_PLACEHOLDER
        key_source = key_source or "default"

    from hermes_cli.auth import _provider_env_base_url

    env_url = _provider_env_base_url(pconfig)
    resolve_url = _API_KEY_BASE_URL_RESOLVERS.get(provider_id, _default_api_key_base_url)
    base_url = resolve_url(api_key, pconfig.inference_base_url, env_url)
    # An API-key provider must never hand back an empty base URL (a set-but-empty
    # COPILOT_API_BASE_URL or similar env override otherwise wedges chat inference).
    if not isinstance(base_url, str) or not base_url.strip():
        base_url = pconfig.inference_base_url

    if not api_key and provider_id == "actual" and is_actual_local_base_url(base_url):
        api_key = ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
        key_source = key_source or "local-offline"
    return {
        "provider": provider_id, "api_key": api_key, "base_url": base_url.rstrip("/"),
        "source": key_source or "default"}


def resolve_external_process_provider_credentials(provider_id: str) -> Dict[str, Any]:
    """Resolve runtime details for local subprocess-backed providers."""
    from hermes_cli.auth import _registry_lookup

    pconfig = _registry_lookup(provider_id)
    if not pconfig or pconfig.auth_type != "external_process":
        raise AuthError(
            f"Provider '{provider_id}' is not an external-process provider.",
            provider=provider_id, code="invalid_provider")

    from hermes_cli.auth import _external_process_spec

    command, args, base_url, resolved_command, command_env_vars = _external_process_spec(pconfig)
    if not resolved_command and not base_url.startswith("acp+tcp://"):
        _hint = " or set " + "/".join(command_env_vars) if command_env_vars else ""
        raise AuthError(
            f"Could not find the '{provider_id}' CLI command "
            f"'{command or '(none configured)'}'. Install it{_hint}.",
            provider=provider_id,
            code="missing_external_process_cli")
    # api_key is a placeholder: the subprocess owns real auth. Keyed on the provider id so each
    # external-process provider gets a distinct value.
    return {
        "provider": provider_id, "api_key": pconfig.id or provider_id,
        "base_url": base_url.rstrip("/"), "command": resolved_command or command, "args": args,
        "source": "process"}
