"""Model alias resolution helpers.

Model aliases are local Hermes config templates under ``model_aliases``.
They are not provider-side model IDs, so callers must resolve them before
building an LLM request. Keep this module dependency-light so it can be used
by CLI startup, runtime provider resolution, and fallback-chain config.
"""

from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import urlparse


_VALID_API_MODES = {
    "chat_completions",
    "codex_responses",
    "anthropic_messages",
    "bedrock_converse",
    "codex_app_server",
}


def _clean_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def alias_api_mode(alias: dict[str, Any]) -> Optional[str]:
    """Return the normalized api_mode implied by a model alias."""
    raw = _clean_str(alias.get("api_mode") or alias.get("transport")).lower()
    if raw in _VALID_API_MODES:
        return raw

    api_format = _clean_str(alias.get("api_format")).lower()
    if api_format in {"anthropic", "anthropic_messages"}:
        return "anthropic_messages"
    if api_format in {"openai", "chat_completions", "openai_chat"}:
        return "chat_completions"
    if api_format in {"codex", "codex_responses", "responses"}:
        return "codex_responses"
    if api_format == "bedrock_converse":
        return "bedrock_converse"
    return None


def _endpoint_from_alias(alias: dict[str, Any]) -> str:
    endpoint_env = _clean_str(alias.get("endpoint_env"))
    if endpoint_env:
        env_value = os.getenv(endpoint_env, "").strip()
        if env_value:
            return env_value
    return _clean_str(alias.get("endpoint"))


def _endpoint_to_base_url(endpoint: str, api_mode: Optional[str]) -> str:
    """Convert a request endpoint to the base URL expected by transports.

    Alias configs historically stored full endpoints such as
    ``https://api.z.ai/api/anthropic/v1/messages``. The Anthropic transport
    expects the API root (``.../anthropic``), while OpenAI-compatible clients
    expect the root before ``/chat/completions``.
    """
    endpoint = (endpoint or "").strip().rstrip("/")
    if not endpoint:
        return ""

    parsed = urlparse(endpoint)
    path = parsed.path.rstrip("/")
    lowered = path.lower()

    suffixes: tuple[str, ...]
    if api_mode == "anthropic_messages":
        suffixes = ("/v1/messages", "/messages")
    elif api_mode in {"chat_completions", "codex_responses", None}:
        suffixes = ("/v1/chat/completions", "/chat/completions", "/responses")
    else:
        suffixes = ()

    for suffix in suffixes:
        if lowered.endswith(suffix):
            new_path = path[: -len(suffix)] or ""
            return parsed._replace(path=new_path, params="", query="", fragment="").geturl().rstrip("/")

    return endpoint


def get_model_alias_entry(config: dict[str, Any] | None, alias_name: Any) -> Optional[dict[str, Any]]:
    """Return a validated copy of ``model_aliases.<alias_name>`` if present."""
    key = _clean_str(alias_name).lower()
    if not key or not isinstance(config, dict):
        return None
    aliases = config.get("model_aliases")
    if not isinstance(aliases, dict):
        return None

    for name, entry in aliases.items():
        if _clean_str(name).lower() != key or not isinstance(entry, dict):
            continue
        provider = _clean_str(entry.get("provider"))
        model = _clean_str(entry.get("model"))
        if not provider or not model:
            return None
        result = dict(entry)
        result["name"] = _clean_str(name)
        result["provider"] = provider
        result["model"] = model
        api_mode = alias_api_mode(result)
        if api_mode:
            result["api_mode"] = api_mode
        base_url = _clean_str(result.get("base_url"))
        if not base_url:
            base_url = _endpoint_to_base_url(_endpoint_from_alias(result), api_mode)
        if base_url:
            result["base_url"] = base_url.rstrip("/")
        return result
    return None


def _normalize_model_config(model_cfg: Any) -> dict[str, Any]:
    if isinstance(model_cfg, dict):
        cfg = dict(model_cfg)
        if not cfg.get("default") and cfg.get("model"):
            cfg["default"] = cfg.get("model")
        return cfg
    if isinstance(model_cfg, str) and model_cfg.strip():
        return {"default": model_cfg.strip()}
    return {}


def resolve_model_alias(config: dict[str, Any] | None, model_name: Any) -> Optional[dict[str, Any]]:
    """Resolve ``model_name`` when it names a configured model alias."""
    return get_model_alias_entry(config, model_name)


def apply_model_alias_to_model_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return ``model`` config with ``model.default`` aliases expanded.

    If ``model.default`` (or legacy ``model.model``) names an entry under
    ``model_aliases``, the returned config contains the concrete provider/model
    route plus inherited base_url/api_mode/api_key when present. The original
    alias name is preserved as ``configured_alias_intent`` for diagnostics.
    """
    config = config or {}
    cfg = _normalize_model_config(config.get("model"))
    alias_name = cfg.get("default") or cfg.get("model")
    alias = get_model_alias_entry(config, alias_name)
    if alias is None:
        return cfg

    resolved = dict(cfg)
    resolved["configured_alias_intent"] = alias.get("name") or _clean_str(alias_name)
    resolved["default"] = alias["model"]
    resolved["provider"] = alias["provider"]

    for key in ("base_url", "api_mode", "api_key"):
        value = alias.get(key)
        if value and not resolved.get(key):
            resolved[key] = value

    return resolved
