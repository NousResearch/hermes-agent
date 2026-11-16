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

# One token source per (profile home, key_cmd): its token is reused until expiry instead of
# re-running the helper on every catalog probe.
_KEY_CMD_SOURCES: dict[tuple[str, str], Any] = {}


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
    from hermes_cli.config import get_env_value_prefer_dotenv, load_config
    from hermes_cli.config_providers import find_provider_entry, is_provider_enabled

    try:
        config = load_config() or {}
    except (OSError, RuntimeError):
        logger.debug("Declared provider %r: config load failed", provider_id, exc_info=True)
        config = {}
    _stored_key, entry = find_provider_entry(config.get("providers"), provider_id)  # load_config() expanded ${VAR} refs
    if not isinstance(entry, dict) or not is_provider_enabled(entry):
        raise AuthError(
            f"Provider '{provider_id}' is not an API-key provider.",
            provider=provider_id, code="invalid_provider")
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
        from hermes_constants import hermes_home_key
        token_source = _KEY_CMD_SOURCES.setdefault(
            (hermes_home_key(), key_cmd), build_command_token_provider(key_cmd, str(entry.get("name") or provider_id)))
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


def relay_probe_api_key(provider_id: str, relay: str, api_key: str, cred_base_url: str) -> str:
    """*api_key* when it may ride a catalog probe of *relay*, else ``""`` (probe keyless).

    A registry provider's key is meant for the relay: inference sends it there too (#121387). A
    declared ``providers.<id>`` key belongs to that entry's own base_url, which is where inference
    resolves, so a ``model.base_url`` on another origin (stale or edited) must not receive it.
    """
    from hermes_cli.auth import is_known_auth_provider
    from utils import base_url_origin

    if not api_key or is_known_auth_provider(provider_id):
        return api_key
    origin = base_url_origin(relay)
    return api_key if origin[1] and origin == base_url_origin(cred_base_url) else ""


def custom_catalog_probe_api_key(model_cfg: dict[str, Any], base_url: str, *, minted: bool = True) -> Any:
    """Key for a bare ``custom`` catalog probe of *base_url*, else ``""`` (keyless).

    The runtime resolver's order (#145452): ``model.api_key`` / ``model.key_env`` when *base_url*
    is ``model.base_url``, then the key of
    a ``providers:`` entry owning the URL; only a URL no entry declares a credential for falls back
    to ``CUSTOM_API_KEY`` and the host-gated env keys (#91308), so an ambient OpenAI or OpenRouter
    key never reaches a third-party endpoint. ``minted=False`` hands back a ``key_cmd`` entry's token
    source unminted, for callers that mint only when they actually probe (the picker's cache reads).
    """
    from agent.command_token_source import materialize_probe_api_key
    from agent.secret_scope import get_secret_str
    from hermes_cli import runtime_provider as rp
    from hermes_cli.runtime_provider_custom import _model_cfg_key_env_for, _owning_entry_credential

    is_cfg_url = str(model_cfg.get("base_url") or "").strip().rstrip("/") == (base_url or "").strip().rstrip("/")
    own = (str(model_cfg.get("api_key") or "").strip() if is_cfg_url else "") or _model_cfg_key_env_for(model_cfg, base_url)
    if rp.has_usable_secret(own):
        return own
    declared, entry_key = _owning_entry_credential(rp, base_url)
    if declared:
        return materialize_probe_api_key(entry_key) if minted else entry_key
    candidates = [get_secret_str("CUSTOM_API_KEY", "").strip(), *rp._host_gated_env_key_candidates(base_url, ollama=True)]
    return next((c for c in candidates if rp.has_usable_secret(c)), "")
