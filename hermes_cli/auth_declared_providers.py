"""Credential and base-URL resolution for API-key providers.

A ``providers.<id>`` block is a real endpoint even when ``PROVIDER_REGISTRY`` has never heard of
it, so catalog probes and runtime resolution must be able to read its own key pointer and URL
(#145452). The sibling keeps that behaviour out of the ``auth`` facade.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def default_api_key_base_url(api_key: str, default: str, env_url: str) -> str:
    return env_url.rstrip("/") if env_url else default


def provider_env_base_url(pconfig: Any) -> str:
    if pconfig.id == "actual":
        from hermes_cli.config import read_raw_config
        from hermes_cli.providers import normalize_provider

        model = read_raw_config().get("model")
        if isinstance(model, dict) and normalize_provider(str(model.get("provider") or "")) == "actual":
            configured_url = str(model.get("base_url") or "").strip()
            if configured_url:
                return configured_url
    return os.getenv(pconfig.base_url_env_var, "").strip() if pconfig.base_url_env_var else ""


def resolve_declared_api_key_provider(provider_id: str) -> dict[str, Any]:
    """Credentials for an enabled ``providers.<id>`` block absent from ``PROVIDER_REGISTRY``.

    The entry's own pointers are honoured (inline ``api_key``, ``key_env``/``api_key_env``,
    ``key_cmd``) and ``${VAR}``/``${env:VAR}`` refs are expanded with the config loader's own
    expander, so an inline secret ref never reaches an ``Authorization`` header verbatim
    (#145452). ``key_cmd`` is minted here, when a caller asks for probe-ready credentials.

    Raises ``AuthError`` when no enabled entry exists, so an unknown provider still fails.
    """
    from hermes_cli.auth import AuthError
    from hermes_cli.config import _expand_env_vars, get_env_value_prefer_dotenv, load_config
    from hermes_cli.config_providers import find_provider_entry, is_provider_enabled

    try:
        config = load_config() or {}
    except (OSError, RuntimeError):
        logger.debug("Declared provider %r: config load failed", provider_id, exc_info=True)
        config = {}
    _stored_key, raw_entry = find_provider_entry(config.get("providers"), provider_id)
    if not isinstance(raw_entry, dict) or not is_provider_enabled(raw_entry):
        raise AuthError(
            f"Provider '{provider_id}' is not an API-key provider.",
            provider=provider_id, code="invalid_provider")
    entry = _expand_env_vars(raw_entry)
    base_url = str(entry.get("base_url") or entry.get("api") or entry.get("url") or "").strip().rstrip("/")
    if not base_url:
        raise AuthError(
            f"Provider '{provider_id}' is not an API-key provider.",
            provider=provider_id, code="invalid_provider")

    api_key: Any = ""
    source = "config"
    key_cmd = str(entry.get("key_cmd") or "").strip()
    if key_cmd:
        from agent.command_token_source import build_command_token_provider, materialize_probe_api_key
        token_source = build_command_token_provider(key_cmd, str(entry.get("name") or provider_id))
        api_key = materialize_probe_api_key(token_source)
        source = "key_cmd" if api_key else source
    if not api_key:
        api_key = str(entry.get("api_key") or "").strip()
        source = "api_key" if api_key else source
    key_env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
    if not api_key and key_env:
        api_key = (get_env_value_prefer_dotenv(key_env) or "").strip()
        source = key_env if api_key else source
    return {
        "provider": provider_id, "api_key": api_key, "base_url": base_url,
        "source": source}
