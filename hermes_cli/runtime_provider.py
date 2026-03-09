"""Shared runtime provider resolution for CLI, gateway, cron, and helpers."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from hermes_cli.auth import (
    AuthError,
    PROVIDER_REGISTRY,
    format_auth_error,
    resolve_provider,
    resolve_nous_runtime_credentials,
    resolve_codex_runtime_credentials,
    resolve_api_key_provider_credentials,
)
from hermes_cli.config import load_config
from hermes_constants import OPENROUTER_BASE_URL


logger = logging.getLogger(__name__)


def _get_model_config() -> Dict[str, Any]:
    config = load_config()
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        return dict(model_cfg)
    if isinstance(model_cfg, str) and model_cfg.strip():
        return {"default": model_cfg.strip()}
    return {}


def _coerce_max_tokens(raw_value: Any, source: str) -> Optional[int]:
    if raw_value is None:
        return None

    if isinstance(raw_value, bool):
        logger.warning(f"Ignoring invalid {source} value for max_tokens: {raw_value!r}")
        return None

    if isinstance(raw_value, int):
        if raw_value > 0:
            return raw_value
        logger.warning(
            f"Ignoring non-positive {source} value for max_tokens: {raw_value!r}"
        )
        return None

    raw_text = str(raw_value).strip()
    if not raw_text:
        return None

    try:
        parsed = int(raw_text)
    except (TypeError, ValueError):
        logger.warning(f"Ignoring invalid {source} value for max_tokens: {raw_value!r}")
        return None

    if parsed <= 0:
        logger.warning(
            f"Ignoring non-positive {source} value for max_tokens: {raw_value!r}"
        )
        return None

    return parsed


def resolve_runtime_max_tokens() -> Optional[int]:
    env_value = _coerce_max_tokens(os.getenv("HERMES_MAX_TOKENS"), "HERMES_MAX_TOKENS")
    if env_value is not None:
        return env_value

    model_cfg = _get_model_config()
    return _coerce_max_tokens(model_cfg.get("max_tokens"), "config model.max_tokens")


def resolve_requested_provider(requested: Optional[str] = None) -> str:
    """Resolve provider request from explicit arg, env, then config."""
    if requested and requested.strip():
        return requested.strip().lower()

    env_provider = os.getenv("HERMES_INFERENCE_PROVIDER", "").strip().lower()
    if env_provider:
        return env_provider

    model_cfg = _get_model_config()
    cfg_provider = model_cfg.get("provider")
    if isinstance(cfg_provider, str) and cfg_provider.strip():
        return cfg_provider.strip().lower()

    return "auto"


def _resolve_openrouter_runtime(
    *,
    requested_provider: str,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    model_cfg = _get_model_config()
    cfg_base_url = (
        model_cfg.get("base_url") if isinstance(model_cfg.get("base_url"), str) else ""
    )
    cfg_provider = (
        model_cfg.get("provider") if isinstance(model_cfg.get("provider"), str) else ""
    )
    requested_norm = (requested_provider or "").strip().lower()
    cfg_provider = cfg_provider.strip().lower()

    env_openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    env_openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "").strip()

    use_config_base_url = False
    if requested_norm == "auto":
        if cfg_base_url.strip() and not explicit_base_url and not env_openai_base_url:
            if not cfg_provider or cfg_provider == "auto":
                use_config_base_url = True

    base_url = (
        (explicit_base_url or "").strip()
        or env_openai_base_url
        or (cfg_base_url.strip() if use_config_base_url else "")
        or env_openrouter_base_url
        or OPENROUTER_BASE_URL
    ).rstrip("/")

    # Choose API key based on whether the resolved base_url targets OpenRouter.
    # When hitting OpenRouter, prefer OPENROUTER_API_KEY (issue #289).
    # When hitting a custom endpoint (e.g. Z.ai, local LLM), prefer
    # OPENAI_API_KEY so the OpenRouter key doesn't leak to an unrelated
    # provider (issues #420, #560).
    _is_openrouter_url = "openrouter.ai" in base_url
    if _is_openrouter_url:
        api_key = (
            explicit_api_key
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
    else:
        api_key = (
            explicit_api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or ""
        )

    source = "explicit" if (explicit_api_key or explicit_base_url) else "env/config"

    return {
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "base_url": base_url,
        "api_key": api_key,
        "source": source,
    }


def resolve_runtime_provider(
    *,
    requested: Optional[str] = None,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve runtime provider credentials for agent execution."""
    requested_provider = resolve_requested_provider(requested)
    max_tokens = resolve_runtime_max_tokens()

    provider = resolve_provider(
        requested_provider,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
    )

    if provider == "nous":
        creds = resolve_nous_runtime_credentials(
            min_key_ttl_seconds=max(
                60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800"))
            ),
            timeout_seconds=float(os.getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
        )
        return {
            "provider": "nous",
            "api_mode": "chat_completions",
            "base_url": creds.get("base_url", "").rstrip("/"),
            "api_key": creds.get("api_key", ""),
            "max_tokens": max_tokens,
            "source": creds.get("source", "portal"),
            "expires_at": creds.get("expires_at"),
            "requested_provider": requested_provider,
        }

    if provider == "openai-codex":
        creds = resolve_codex_runtime_credentials()
        return {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": creds.get("base_url", "").rstrip("/"),
            "api_key": creds.get("api_key", ""),
            "max_tokens": max_tokens,
            "source": creds.get("source", "hermes-auth-store"),
            "last_refresh": creds.get("last_refresh"),
            "requested_provider": requested_provider,
        }

    # API-key providers (z.ai/GLM, Kimi, MiniMax, MiniMax-CN)
    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig and pconfig.auth_type == "api_key":
        creds = resolve_api_key_provider_credentials(provider)
        return {
            "provider": provider,
            "api_mode": "chat_completions",
            "base_url": creds.get("base_url", "").rstrip("/"),
            "api_key": creds.get("api_key", ""),
            "max_tokens": max_tokens,
            "source": creds.get("source", "env"),
            "requested_provider": requested_provider,
        }

    runtime = _resolve_openrouter_runtime(
        requested_provider=requested_provider,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
    )
    runtime["max_tokens"] = max_tokens
    runtime["requested_provider"] = requested_provider
    return runtime


def format_runtime_provider_error(error: Exception) -> str:
    if isinstance(error, AuthError):
        return format_auth_error(error)
    return str(error)
