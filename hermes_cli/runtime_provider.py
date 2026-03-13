"""Shared runtime provider resolution for CLI, gateway, cron, and helpers."""

from __future__ import annotations

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


def _get_model_config() -> Dict[str, Any]:
    config = load_config()
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        return dict(model_cfg)
    if isinstance(model_cfg, str) and model_cfg.strip():
        return {"default": model_cfg.strip()}
    return {}


def _get_model_profiles_config() -> Dict[str, Dict[str, Any]]:
    config = load_config()
    profiles = config.get("model_profiles")
    if isinstance(profiles, dict):
        normalized: Dict[str, Dict[str, Any]] = {}
        for name, value in profiles.items():
            if isinstance(name, str) and isinstance(value, dict):
                normalized[name.strip().lower()] = dict(value)
        return normalized
    return {}


def resolve_model_profile(profile: str) -> Dict[str, str]:
    """Resolve model/provider/base_url/api_key overrides for a named profile.

    Returns empty strings for missing fields so callers can cleanly apply
    fallback precedence.
    """
    key = (profile or "").strip().lower()
    if not key:
        return {
            "model": "",
            "provider": "",
            "base_url": "",
            "api_key": "",
            "api_key_env": "",
        }

    profiles = _get_model_profiles_config()
    raw = profiles.get(key, {})
    if not isinstance(raw, dict):
        raw = {}

    api_key_env = str(raw.get("api_key_env", "") or "").strip()
    api_key = str(raw.get("api_key", "") or "").strip()
    if (not api_key) and api_key_env:
        api_key = os.getenv(api_key_env, "").strip()

    return {
        "model": str(raw.get("model", "") or "").strip(),
        "provider": str(raw.get("provider", "") or "").strip().lower(),
        "base_url": str(raw.get("base_url", "") or "").strip(),
        "api_key": api_key,
        "api_key_env": api_key_env,
    }


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
    cfg_base_url = model_cfg.get("base_url") if isinstance(model_cfg.get("base_url"), str) else ""
    cfg_provider = model_cfg.get("provider") if isinstance(model_cfg.get("provider"), str) else ""
    requested_norm = (requested_provider or "").strip().lower()
    cfg_provider = cfg_provider.strip().lower()

    env_openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    env_openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "").strip()

    use_config_base_url = False
    if requested_norm == "auto":
        if cfg_base_url.strip() and not explicit_base_url and not env_openai_base_url:
            if not cfg_provider or cfg_provider == "auto":
                use_config_base_url = True

    # When the user explicitly requested the openrouter provider, skip
    # OPENAI_BASE_URL — it typically points to a custom / non-OpenRouter
    # endpoint and would prevent switching back to OpenRouter (#874).
    skip_openai_base = requested_norm == "openrouter"

    base_url = (
        (explicit_base_url or "").strip()
        or ("" if skip_openai_base else env_openai_base_url)
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

    provider = resolve_provider(
        requested_provider,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
    )

    if provider == "nous":
        creds = resolve_nous_runtime_credentials(
            min_key_ttl_seconds=max(60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800"))),
            timeout_seconds=float(os.getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
        )
        return {
            "provider": "nous",
            "api_mode": "chat_completions",
            "base_url": creds.get("base_url", "").rstrip("/"),
            "api_key": creds.get("api_key", ""),
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
            "source": creds.get("source", "hermes-auth-store"),
            "last_refresh": creds.get("last_refresh"),
            "requested_provider": requested_provider,
        }

    # Anthropic (native Messages API)
    if provider == "anthropic":
        from agent.anthropic_adapter import resolve_anthropic_token
        token = resolve_anthropic_token()
        if not token:
            raise AuthError(
                "No Anthropic credentials found. Set ANTHROPIC_API_KEY, "
                "run 'claude setup-token', or authenticate with 'claude /login'."
            )
        return {
            "provider": "anthropic",
            "api_mode": "anthropic_messages",
            "base_url": "https://api.anthropic.com",
            "api_key": token,
            "source": "env",
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
            "source": creds.get("source", "env"),
            "requested_provider": requested_provider,
        }

    runtime = _resolve_openrouter_runtime(
        requested_provider=requested_provider,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
    )
    runtime["requested_provider"] = requested_provider
    return runtime


def resolve_model_for_profile(profile: str, fallback_model: str) -> str:
    """Return profile model override when configured, else fallback."""
    profile_cfg = resolve_model_profile(profile)
    profile_model = profile_cfg.get("model", "")
    return profile_model or fallback_model


def format_runtime_provider_error(error: Exception) -> str:
    if isinstance(error, AuthError):
        return format_auth_error(error)
    return str(error)
