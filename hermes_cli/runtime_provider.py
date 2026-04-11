"""Shared runtime provider resolution for CLI, gateway, cron, and helpers."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from hermes_cli import auth as auth_mod
from agent.credential_pool import CredentialPool, PooledCredential, get_custom_provider_pool_key, load_pool
from hermes_cli.auth import (
    AuthError,
    DEFAULT_CODEX_BASE_URL,
    DEFAULT_QWEN_BASE_URL,
    PROVIDER_REGISTRY,
    _agent_key_is_usable,
    format_auth_error,
    resolve_provider,
    resolve_nous_runtime_credentials,
    resolve_codex_runtime_credentials,
    resolve_qwen_runtime_credentials,
    resolve_api_key_provider_credentials,
    resolve_external_process_provider_credentials,
    has_usable_secret,
)
from hermes_cli.config import load_config
from hermes_constants import OPENROUTER_BASE_URL


def _normalize_custom_provider_name(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


def _detect_api_mode_for_url(base_url: str) -> Optional[str]:
    """Auto-detect api_mode from the resolved base URL.

    Direct api.openai.com endpoints need the Responses API for GPT-5.x
    tool calls with reasoning (chat/completions returns 400).
    """
    normalized = (base_url or "").strip().lower().rstrip("/")
    if "api.openai.com" in normalized and "openrouter" not in normalized:
        return "codex_responses"
    return None


def _auto_detect_local_model(base_url: str) -> str:
    """Query a local server for its model name when only one model is loaded."""
    if not base_url:
        return ""
    try:
        import requests
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        resp = requests.get(url + "/models", timeout=5)
        if resp.ok:
            models = resp.json().get("data", [])
            if len(models) == 1:
                model_id = models[0].get("id", "")
                if model_id:
                    return model_id
    except Exception:
        pass
    return ""


def _get_model_config() -> Dict[str, Any]:
    config = load_config()
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        cfg = dict(model_cfg)
        # Accept "model" as alias for "default" (users intuitively write model.model)
        if not cfg.get("default") and cfg.get("model"):
            cfg["default"] = cfg["model"]
        default = (cfg.get("default") or "").strip()
        base_url = (cfg.get("base_url") or "").strip()
        is_local = "localhost" in base_url or "127.0.0.1" in base_url
        is_fallback = not default
        if is_local and is_fallback and base_url:
            detected = _auto_detect_local_model(base_url)
            if detected:
                cfg["default"] = detected
        return cfg
    if isinstance(model_cfg, str) and model_cfg.strip():
        return {"default": model_cfg.strip()}
    return {}


def _provider_supports_explicit_api_mode(provider: Optional[str], configured_provider: Optional[str] = None) -> bool:
    """Check whether a persisted api_mode should be honored for a given provider.

    Prevents stale api_mode from a previous provider leaking into a
    different one after a model/provider switch.  Only applies the
    persisted mode when the config's provider matches the runtime
    provider (or when no configured provider is recorded).
    """
    normalized_provider = (provider or "").strip().lower()
    normalized_configured = (configured_provider or "").strip().lower()
    if not normalized_configured:
        return True
    if normalized_provider == "custom":
        return normalized_configured == "custom" or normalized_configured.startswith("custom:")
    return normalized_configured == normalized_provider


def _copilot_runtime_api_mode(model_cfg: Dict[str, Any], api_key: str) -> str:
    configured_provider = str(model_cfg.get("provider") or "").strip().lower()
    configured_mode = _parse_api_mode(model_cfg.get("api_mode"))
    if configured_mode and _provider_supports_explicit_api_mode("copilot", configured_provider):
        return configured_mode

    model_name = str(model_cfg.get("default") or "").strip()
    if not model_name:
        return "chat_completions"

    try:
        from hermes_cli.models import copilot_model_api_mode

        return copilot_model_api_mode(model_name, api_key=api_key)
    except Exception:
        return "chat_completions"


_VALID_API_MODES = {"chat_completions", "codex_responses", "anthropic_messages"}


def _parse_api_mode(raw: Any) -> Optional[str]:
    """Validate an api_mode value from config. Returns None if invalid."""
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in _VALID_API_MODES:
            return normalized
    return None


def _resolve_runtime_from_pool_entry(
    *,
    provider: str,
    entry: PooledCredential,
    requested_provider: str,
    model_cfg: Optional[Dict[str, Any]] = None,
    pool: Optional[CredentialPool] = None,
) -> Dict[str, Any]:
    model_cfg = model_cfg or _get_model_config()
    base_url = (getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None) or "").rstrip("/")
    api_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    api_mode = "chat_completions"
    if provider == "openai-codex":
        api_mode = "codex_responses"
        base_url = base_url or DEFAULT_CODEX_BASE_URL
    elif provider == "qwen-oauth":
        api_mode = "chat_completions"
        base_url = base_url or DEFAULT_QWEN_BASE_URL
    elif provider == "anthropic":
        api_mode = "anthropic_messages"
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        cfg_base_url = ""
        if cfg_provider == "anthropic":
            cfg_base_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
        base_url = cfg_base_url or base_url or "https://api.anthropic.com"
    elif provider == "openrouter":
        base_url = base_url or OPENROUTER_BASE_URL
    elif provider == "nous":
        api_mode = "chat_completions"
    elif provider == "copilot":
        api_mode = _copilot_runtime_api_mode(model_cfg, getattr(entry, "runtime_api_key", ""))
    else:
        configured_provider = str(model_cfg.get("provider") or "").strip().lower()
        # Honour model.base_url from config.yaml when the configured provider
        # matches this provider — same pattern as the Anthropic branch above.
        # Only override when the pool entry has no explicit base_url (i.e. it
        # fell back to the hardcoded default).  Env var overrides win (#6039).
        pconfig = PROVIDER_REGISTRY.get(provider)
        pool_url_is_default = pconfig and base_url.rstrip("/") == pconfig.inference_base_url.rstrip("/")
        if configured_provider == provider and pool_url_is_default:
            cfg_base_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
            if cfg_base_url:
                base_url = cfg_base_url
        configured_mode = _parse_api_mode(model_cfg.get("api_mode"))
        if configured_mode and _provider_supports_explicit_api_mode(provider, configured_provider):
            api_mode = configured_mode
        elif provider in ("opencode-zen", "opencode-go"):
            from hermes_cli.models import opencode_model_api_mode
            api_mode = opencode_model_api_mode(provider, model_cfg.get("default", ""))
        elif base_url.rstrip("/").endswith("/anthropic"):
            api_mode = "anthropic_messages"

    # OpenCode base URLs end with /v1 for OpenAI-compatible models, but the
    # Anthropic SDK prepends its own /v1/messages to the base_url.  Strip the
    # trailing /v1 so the SDK constructs the correct path (e.g.
    # https://opencode.ai/zen/go/v1/messages instead of .../v1/v1/messages).
    if api_mode == "anthropic_messages" and provider in ("opencode-zen", "opencode-go"):
        base_url = re.sub(r"/v1/?$", "", base_url)

    return {
        "provider": provider,
        "api_mode": api_mode,
        "base_url": base_url,
        "api_key": api_key,
        "source": getattr(entry, "source", "pool"),
        "credential_pool": pool,
        "requested_provider": requested_provider,
    }


def resolve_requested_provider(requested: Optional[str] = None) -> str:
    """Resolve provider request from explicit arg, config, then env."""
    if requested and requested.strip():
        return requested.strip().lower()

    model_cfg = _get_model_config()
    cfg_provider = model_cfg.get("provider")
    if isinstance(cfg_provider, str) and cfg_provider.strip():
        return cfg_provider.strip().lower()

    # Prefer the persisted config selection over any stale shell/.env
    # provider override so chat uses the endpoint the user last saved.
    env_provider = os.getenv("HERMES_INFERENCE_PROVIDER", "").strip().lower()
    if env_provider:
        return env_provider

    return "auto"


def _try_resolve_from_custom_pool(
    base_url: str,
    provider_label: str,
    api_mode_override: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Check if a credential pool exists for a custom endpoint and return a runtime dict if so."""
    pool_key = get_custom_provider_pool_key(base_url)
    if not pool_key:
        return None
    try:
        pool = load_pool(pool_key)
        if not pool.has_credentials():
            return None
        entry = pool.select()
        if entry is None:
            return None
        pool_api_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
        if not pool_api_key:
            return None
        return {
            "provider": provider_label,
            "api_mode": api_mode_override or _detect_api_mode_for_url(base_url) or "chat_completions",
            "base_url": base_url,
            "api_key": pool_api_key,
            "source": f"pool:{pool_key}",
            "credential_pool": pool,
        }
    except Exception:
        return None


def _get_named_custom_provider(requested_provider: str) -> Optional[Dict[str, Any]]:
    requested_norm = _normalize_custom_provider_name(requested_provider or "")
    if not requested_norm or requested_norm == "custom":
        return None

    # Raw names should only map to custom providers when they are not already
    # valid built-in providers or aliases. Explicit menu keys like
    # ``custom:local`` always target the saved custom provider.
    if requested_norm == "auto":
        return None
    if not requested_norm.startswith("custom:"):
        try:
            auth_mod.resolve_provider(requested_norm)
        except AuthError:
            pass
        else:
            return None

    config = load_config()
    custom_providers = config.get("custom_providers")
    if not isinstance(custom_providers, list):
        if isinstance(custom_providers, dict):
            logger.warning(
                "custom_providers in config.yaml is a dict, not a list. "
                "Each entry must be prefixed with '-' in YAML. "
                "Run 'hermes doctor' for details."
            )
        return None

    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        base_url = entry.get("base_url")
        if not isinstance(name, str) or not isinstance(base_url, str):
            continue
        name_norm = _normalize_custom_provider_name(name)
        menu_key = f"custom:{name_norm}"
        if requested_norm not in {name_norm, menu_key}:
            continue
        result = {
            "name": name.strip(),
            "base_url": base_url.strip(),
            "api_key": str(entry.get("api_key", "") or "").strip(),
        }
        api_mode = _parse_api_mode(entry.get("api_mode"))
        if api_mode:
            result["api_mode"] = api_mode
        return result

    return None


def _resolve_named_custom_runtime(
    *,
    requested_provider: str,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    custom_provider = _get_named_custom_provider(requested_provider)
    if not custom_provider:
        return None

    base_url = (
        (explicit_base_url or "").strip()
        or custom_provider.get("base_url", "")
    ).rstrip("/")
    if not base_url:
        return None

    # Check if a credential pool exists for this custom endpoint
    pool_result = _try_resolve_from_custom_pool(base_url, "custom", custom_provider.get("api_mode"))
    if pool_result:
        return pool_result

    api_key_candidates = [
        (explicit_api_key or "").strip(),
        str(custom_provider.get("api_key", "") or "").strip(),
        os.getenv("OPENAI_API_KEY", "").strip(),
        os.getenv("OPENROUTER_API_KEY", "").strip(),
    ]
    api_key = next((candidate for candidate in api_key_candidates if has_usable_secret(candidate)), "")

    return {
        "provider": "custom",
        "api_mode": custom_provider.get("api_mode")
        or _detect_api_mode_for_url(base_url)
        or "chat_completions",
        "base_url": base_url,
        "api_key": api_key or "no-key-required",
        "source": f"custom_provider:{custom_provider.get('name', requested_provider)}",
    }


def _resolve_openrouter_runtime(
    *,
    requested_provider: str,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    model_cfg = _get_model_config()
    cfg_base_url = model_cfg.get("base_url") if isinstance(model_cfg.get("base_url"), str) else ""
    cfg_provider = model_cfg.get("provider") if isinstance(model_cfg.get("provider"), str) else ""
    cfg_api_key = ""
    for k in ("api_key", "api"):
        v = model_cfg.get(k)
        if isinstance(v, str) and v.strip():
            cfg_api_key = v.strip()
            break
    requested_norm = (requested_provider or "").strip().lower()
    cfg_provider = cfg_provider.strip().lower()

    env_openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "").strip()

    # Use config base_url when available and the provider context matches.
    # OPENAI_BASE_URL env var is no longer consulted — config.yaml is
    # the single source of truth for endpoint URLs.
    use_config_base_url = False
    if cfg_base_url.strip() and not explicit_base_url:
        if requested_norm == "auto":
            if not cfg_provider or cfg_provider == "auto":
                use_config_base_url = True
        elif requested_norm == "custom" and cfg_provider == "custom":
            use_config_base_url = True

    base_url = (
        (explicit_base_url or "").strip()
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
        api_key_candidates = [
            explicit_api_key,
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
        ]
    else:
        # Custom endpoint: use api_key from config when using config base_url (#1760).
        # When the endpoint is Ollama Cloud, check OLLAMA_API_KEY — it's
        # the canonical env var for ollama.com authentication.
        _is_ollama_url = "ollama.com" in base_url.lower()
        api_key_candidates = [
            explicit_api_key,
            (cfg_api_key if use_config_base_url else ""),
            (os.getenv("OLLAMA_API_KEY") if _is_ollama_url else ""),
            os.getenv("OPENAI_API_KEY"),
            os.getenv("OPENROUTER_API_KEY"),
        ]
    api_key = next(
        (str(candidate or "").strip() for candidate in api_key_candidates if has_usable_secret(candidate)),
        "",
    )

    source = "explicit" if (explicit_api_key or explicit_base_url) else "env/config"

    # When "custom" was explicitly requested, preserve that as the provider
    # name instead of silently relabeling to "openrouter" (#2562).
    # Also provide a placeholder API key for local servers that don't require
    # authentication — the OpenAI SDK requires a non-empty api_key string.
    effective_provider = "custom" if requested_norm == "custom" else "openrouter"

    # For custom endpoints, check if a credential pool exists
    if effective_provider == "custom" and base_url:
        pool_result = _try_resolve_from_custom_pool(
            base_url, effective_provider, _parse_api_mode(model_cfg.get("api_mode")),
        )
        if pool_result:
            return pool_result

    if effective_provider == "custom" and not api_key and not _is_openrouter_url:
        api_key = "no-key-required"

    return {
        "provider": effective_provider,
        "api_mode": _parse_api_mode(model_cfg.get("api_mode"))
        or _detect_api_mode_for_url(base_url)
        or "chat_completions",
        "base_url": base_url,
        "api_key": api_key,
        "source": source,
    }


def _resolve_explicit_runtime(
    *,
    provider: str,
    requested_provider: str,
    model_cfg: Dict[str, Any],
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    explicit_api_key = str(explicit_api_key or "").strip()
    explicit_base_url = str(explicit_base_url or "").strip().rstrip("/")
    if not explicit_api_key and not explicit_base_url:
        return None

    if provider == "anthropic":
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        cfg_base_url = ""
        if cfg_provider == "anthropic":
            cfg_base_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
        base_url = explicit_base_url or cfg_base_url or "https://api.anthropic.com"
        api_key = explicit_api_key
        if not api_key:
            from agent.anthropic_adapter import resolve_anthropic_token

            api_key = resolve_anthropic_token()
            if not api_key:
                raise AuthError(
                    "No Anthropic credentials found. Set ANTHROPIC_TOKEN or ANTHROPIC_API_KEY, "
                    "run 'claude setup-token', or authenticate with 'claude /login'."
                )
        return {
            "provider": "anthropic",
            "api_mode": "anthropic_messages",
            "base_url": base_url,
            "api_key": api_key,
            "source": "explicit",
            "requested_provider": requested_provider,
        }

    if provider == "openai-codex":
        base_url = explicit_base_url or DEFAULT_CODEX_BASE_URL
        api_key = explicit_api_key
        last_refresh = None
        if not api_key:
            creds = resolve_codex_runtime_credentials()
            api_key = creds.get("api_key", "")
            last_refresh = creds.get("last_refresh")
            if not explicit_base_url:
                base_url = creds.get("base_url", "").rstrip("/") or base_url
        return {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": base_url,
            "api_key": api_key,
            "source": "explicit",
            "last_refresh": last_refresh,
            "requested_provider": requested_provider,
        }

    if provider == "nous":
        state = auth_mod.get_provider_auth_state("nous") or {}
        base_url = (
            explicit_base_url
            or str(state.get("inference_base_url") or auth_mod.DEFAULT_NOUS_INFERENCE_URL).strip().rstrip("/")
        )
        # Only use agent_key for inference — access_token is an OAuth token for the
        # portal API (minting keys, refreshing tokens), not for the inference API.
        # Falling back to access_token sends an OAuth bearer token to the inference
        # endpoint, which returns 404 because it is not a valid inference credential.
        api_key = explicit_api_key or str(state.get("agent_key") or "").strip()
        expires_at = state.get("agent_key_expires_at") or state.get("expires_at")
        if not api_key:
            creds = resolve_nous_runtime_credentials(
                min_key_ttl_seconds=max(60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800"))),
                timeout_seconds=float(os.getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
            )
            api_key = creds.get("api_key", "")
            expires_at = creds.get("expires_at")
            if not explicit_base_url:
                base_url = creds.get("base_url", "").rstrip("/") or base_url
        return {
            "provider": "nous",
            "api_mode": "chat_completions",
            "base_url": base_url,
            "api_key": api_key,
            "source": "explicit",
            "expires_at": expires_at,
            "requested_provider": requested_provider,
        }

    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig and pconfig.auth_type == "api_key":
        env_url = ""
        if pconfig.base_url_env_var:
            env_url = os.getenv(pconfig.base_url_env_var, "").strip().rstrip("/")

        base_url = explicit_base_url
        if not base_url:
            if provider == "kimi-coding":
                creds = resolve_api_key_provider_credentials(provider)
                base_url = creds.get("base_url", "").rstrip("/")
            else:
                base_url = env_url or pconfig.inference_base_url

        api_key = explicit_api_key
        if not api_key:
            creds = resolve_api_key_provider_credentials(provider)
            api_key = creds.get("api_key", "")
            if not base_url:
                base_url = creds.get("base_url", "").rstrip("/")

        api_mode = "chat_completions"
        if provider == "copilot":
            api_mode = _copilot_runtime_api_mode(model_cfg, api_key)
        else:
            configured_mode = _parse_api_mode(model_cfg.get("api_mode"))
            if configured_mode:
                api_mode = configured_mode
            elif base_url.rstrip("/").endswith("/anthropic"):
                api_mode = "anthropic_messages"

        return {
            "provider": provider,
            "api_mode": api_mode,
            "base_url": base_url.rstrip("/"),
            "api_key": api_key,
            "source": "explicit",
            "requested_provider": requested_provider,
        }

    return None


def resolve_runtime_provider(
    *,
    requested: Optional[str] = None,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve runtime provider credentials for agent execution."""
    requested_provider = resolve_requested_provider(requested)

    custom_runtime = _resolve_named_custom_runtime(
        requested_provider=requested_provider,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
    )
    if custom_runtime:
        custom_runtime["requested_provider"] = requested_provider
        return custom_runtime

    provider = resolve_provider(
        requested_provider,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
    )
    model_cfg = _get_model_config()
    explicit_runtime = _resolve_explicit_runtime(
        provider=provider,
        requested_provider=requested_provider,
        model_cfg=model_cfg,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
    )
    if explicit_runtime:
        return explicit_runtime

    should_use_pool = provider != "openrouter"
    if provider == "openrouter":
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        cfg_base_url = str(model_cfg.get("base_url") or "").strip()
        env_openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        env_openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "").strip()
        has_custom_endpoint = bool(
            explicit_base_url
            or env_openai_base_url
            or env_openrouter_base_url
        )
        if cfg_base_url and cfg_provider in {"auto", "custom"}:
            has_custom_endpoint = True
        has_runtime_override = bool(explicit_api_key or explicit_base_url)
        should_use_pool = (
            requested_provider in {"openrouter", "auto"}
            and not has_custom_endpoint
            and not has_runtime_override
        )

    try:
        pool = load_pool(provider) if should_use_pool else None
    except Exception:
        pool = None
    if pool and pool.has_credentials():
        entry = pool.select()
        pool_api_key = ""
        if entry is not None:
            pool_api_key = (
                getattr(entry, "runtime_api_key", None)
                or getattr(entry, "access_token", "")
            )
        # For Nous, the pool entry's runtime_api_key is the agent_key — a
        # short-lived inference credential (~30 min TTL).  The pool doesn't
        # refresh it during selection (that would trigger network calls in
        # non-runtime contexts like `hermes auth list`).  If the key is
        # expired, clear pool_api_key so we fall through to
        # resolve_nous_runtime_credentials() which handles refresh + mint.
        if provider == "nous" and entry is not None and pool_api_key:
            min_ttl = max(60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800")))
            nous_state = {
                "agent_key": getattr(entry, "agent_key", None),
                "agent_key_expires_at": getattr(entry, "agent_key_expires_at", None),
            }
            if not _agent_key_is_usable(nous_state, min_ttl):
                logger.debug("Nous pool entry agent_key expired/missing, falling through to runtime resolution")
                pool_api_key = ""
        if entry is not None and pool_api_key:
            return _resolve_runtime_from_pool_entry(
                provider=provider,
                entry=entry,
                requested_provider=requested_provider,
                model_cfg=model_cfg,
                pool=pool,
            )

    if provider == "nous":
        try:
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
        except AuthError:
            if requested_provider != "auto":
                raise
            # Auto-detected Nous but credentials are stale/revoked —
            # fall through to env-var providers (e.g. OpenRouter).
            logger.info("Auto-detected Nous provider but credentials failed; "
                        "falling through to next provider.")

    if provider == "openai-codex":
        try:
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
        except AuthError:
            if requested_provider != "auto":
                raise
            # Auto-detected Codex but credentials are stale/revoked —
            # fall through to env-var providers (e.g. OpenRouter).
            logger.info("Auto-detected Codex provider but credentials failed; "
                        "falling through to next provider.")

    if provider == "qwen-oauth":
        try:
            creds = resolve_qwen_runtime_credentials()
            return {
                "provider": "qwen-oauth",
                "api_mode": "chat_completions",
                "base_url": creds.get("base_url", "").rstrip("/"),
                "api_key": creds.get("api_key", ""),
                "source": creds.get("source", "qwen-cli"),
                "expires_at_ms": creds.get("expires_at_ms"),
                "requested_provider": requested_provider,
            }
        except AuthError:
            if requested_provider != "auto":
                raise
            logger.info("Qwen OAuth credentials failed; "
                        "falling through to next provider.")

    if provider == "copilot-acp":
        creds = resolve_external_process_provider_credentials(provider)
        return {
            "provider": "copilot-acp",
            "api_mode": "chat_completions",
            "base_url": creds.get("base_url", "").rstrip("/"),
            "api_key": creds.get("api_key", ""),
            "command": creds.get("command", ""),
            "args": list(creds.get("args") or []),
            "source": creds.get("source", "process"),
            "requested_provider": requested_provider,
        }

    # Anthropic (native Messages API)
    if provider == "anthropic":
        from agent.anthropic_adapter import resolve_anthropic_token
        token = resolve_anthropic_token()
        if not token:
            raise AuthError(
                "No Anthropic credentials found. Set ANTHROPIC_TOKEN or ANTHROPIC_API_KEY, "
                "run 'claude setup-token', or authenticate with 'claude /login'."
            )
        # Allow base URL override from config.yaml model.base_url, but only
        # when the configured provider is anthropic — otherwise a non-Anthropic
        # base_url (e.g. Codex endpoint) would leak into Anthropic requests.
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        cfg_base_url = ""
        if cfg_provider == "anthropic":
            cfg_base_url = (model_cfg.get("base_url") or "").strip().rstrip("/")
        base_url = cfg_base_url or "https://api.anthropic.com"
        return {
            "provider": "anthropic",
            "api_mode": "anthropic_messages",
            "base_url": base_url,
            "api_key": token,
            "source": "env",
            "requested_provider": requested_provider,
        }

    # API-key providers (z.ai/GLM, Kimi, MiniMax, MiniMax-CN)
    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig and pconfig.auth_type == "api_key":
        creds = resolve_api_key_provider_credentials(provider)
        # Honour model.base_url from config.yaml when the configured provider
        # matches this provider — mirrors the Anthropic path above.  Without
        # this, users who set model.base_url to e.g. api.minimaxi.com/anthropic
        # (China endpoint) still get the hardcoded api.minimax.io default (#6039).
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        cfg_base_url = ""
        if cfg_provider == provider:
            cfg_base_url = (model_cfg.get("base_url") or "").strip().rstrip("/")
        base_url = cfg_base_url or creds.get("base_url", "").rstrip("/")
        api_mode = "chat_completions"
        if provider == "copilot":
            api_mode = _copilot_runtime_api_mode(model_cfg, creds.get("api_key", ""))
        else:
            configured_provider = str(model_cfg.get("provider") or "").strip().lower()
            # Only honor persisted api_mode when it belongs to the same provider family.
            configured_mode = _parse_api_mode(model_cfg.get("api_mode"))
            if configured_mode and _provider_supports_explicit_api_mode(provider, configured_provider):
                api_mode = configured_mode
            elif provider in ("opencode-zen", "opencode-go"):
                from hermes_cli.models import opencode_model_api_mode
                api_mode = opencode_model_api_mode(provider, model_cfg.get("default", ""))
            # Auto-detect Anthropic-compatible endpoints by URL convention
            # (e.g. https://api.minimax.io/anthropic, https://dashscope.../anthropic)
            elif base_url.rstrip("/").endswith("/anthropic"):
                api_mode = "anthropic_messages"
        # Strip trailing /v1 for OpenCode Anthropic models (see comment above).
        if api_mode == "anthropic_messages" and provider in ("opencode-zen", "opencode-go"):
            base_url = re.sub(r"/v1/?$", "", base_url)
        return {
            "provider": provider,
            "api_mode": api_mode,
            "base_url": base_url,
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


def format_runtime_provider_error(error: Exception) -> str:
    if isinstance(error, AuthError):
        return format_auth_error(error)
    return str(error)
