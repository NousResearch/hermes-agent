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
    resolve_gemini_oauth_runtime_credentials,
    resolve_api_key_provider_credentials,
    resolve_external_process_provider_credentials,
    has_usable_secret,
)
from hermes_cli.config import get_compatible_custom_providers, load_config
from hermes_constants import OPENROUTER_BASE_URL
from utils import base_url_host_matches, base_url_hostname


def _normalize_custom_provider_name(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


def _loopback_hostname(host: str) -> bool:
    h = (host or "").lower().rstrip(".")
    return h in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _config_base_url_trustworthy_for_bare_custom(cfg_base_url: str, cfg_provider: str) -> bool:
    """Decide whether ``model.base_url`` may back bare ``custom`` runtime resolution.

    GitHub #14676: the model picker can select Custom while ``model.provider`` still reflects a
    previous provider. Reject non-loopback URLs unless the YAML provider is already ``custom``,
    so a stale OpenRouter/Z.ai base_url cannot hijack local ``custom`` sessions.
    """
    cfg_provider_norm = (cfg_provider or "").strip().lower()
    bu = (cfg_base_url or "").strip()
    if not bu:
        return False
    if cfg_provider_norm == "custom":
        return True
    if base_url_host_matches(bu, "openrouter.ai"):
        return False
    return _loopback_hostname(base_url_hostname(bu))


def _detect_api_mode_for_url(base_url: str) -> Optional[str]:
    """Auto-detect api_mode from the resolved base URL.

    - Direct api.openai.com endpoints need the Responses API for GPT-5.x
      tool calls with reasoning (chat/completions returns 400).
    - Third-party Anthropic-compatible gateways (MiniMax, Zhipu GLM,
      LiteLLM proxies, etc.) conventionally expose the native Anthropic
      protocol under a ``/anthropic`` suffix — treat those as
      ``anthropic_messages`` transport instead of the default
      ``chat_completions``.
    - Kimi Code's ``api.kimi.com/coding`` endpoint also speaks the
      Anthropic Messages protocol (the /coding route accepts Claude
      Code's native request shape).
    """
    normalized = (base_url or "").strip().lower().rstrip("/")
    hostname = base_url_hostname(base_url)
    if hostname == "api.x.ai":
        return "codex_responses"
    if hostname == "api.openai.com":
        return "codex_responses"
    if normalized.endswith("/anthropic"):
        return "anthropic_messages"
    if hostname == "api.kimi.com" and "/coding" in normalized:
        return "anthropic_messages"
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


_VALID_API_MODES = {"chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse"}


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
    target_model: Optional[str] = None,
) -> Dict[str, Any]:
    model_cfg = model_cfg or _get_model_config()
    # When the caller is resolving for a specific target model (e.g. a /model
    # mid-session switch), prefer that over the persisted model.default. This
    # prevents api_mode being computed from a stale config default that no
    # longer matches the model actually being used — the bug that caused
    # opencode-zen /v1 to be stripped for chat_completions requests when
    # config.default was still a Claude model.
    effective_model = (target_model or model_cfg.get("default") or "")
    base_url = (getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None) or "").rstrip("/")
    api_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    api_mode = "chat_completions"
    if provider == "openai-codex":
        api_mode = "codex_responses"
        base_url = base_url or DEFAULT_CODEX_BASE_URL
    elif provider == "qwen-oauth":
        api_mode = "chat_completions"
        base_url = base_url or DEFAULT_QWEN_BASE_URL
    elif provider == "google-gemini-cli":
        api_mode = "chat_completions"
        base_url = base_url or "cloudcode-pa://google"
    elif provider == "minimax-oauth":
        # MiniMax OAuth tokens are valid only against the Anthropic Messages
        # compatible endpoint. Do not honor stale model.api_mode values from a
        # prior OpenAI-compatible provider, or the client will hit
        # /chat/completions under /anthropic and receive a bare nginx 404.
        api_mode = "anthropic_messages"
        pconfig = PROVIDER_REGISTRY.get(provider)
        base_url = base_url or (pconfig.inference_base_url if pconfig else "")
    elif provider == "anthropic":
        api_mode = "anthropic_messages"
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        cfg_base_url = ""
        if cfg_provider == "anthropic":
            cfg_base_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
        base_url = cfg_base_url or base_url or "https://api.anthropic.com"
    elif provider == "openrouter":
        base_url = base_url or OPENROUTER_BASE_URL
    elif provider == "xai":
        api_mode = "codex_responses"
    elif provider == "nous":
        api_mode = "chat_completions"
    elif provider == "copilot":
        api_mode = _copilot_runtime_api_mode(model_cfg, getattr(entry, "runtime_api_key", ""))
        base_url = base_url or PROVIDER_REGISTRY["copilot"].inference_base_url
    elif provider == "azure-foundry":
        # Azure Foundry: read api_mode and base_url from config
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        if cfg_provider == "azure-foundry":
            cfg_base_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
            if cfg_base_url:
                base_url = cfg_base_url
            configured_mode = _parse_api_mode(model_cfg.get("api_mode"))
            if configured_mode:
                api_mode = configured_mode
        # Model-family inference for GPT-5.x / codex / o1-o4: Azure rejects
        # /chat/completions on these with 400 "operation unsupported" — see
        # azure_foundry_model_api_mode() for rationale.  Skip when the user
        # explicitly picked anthropic_messages (Anthropic-style endpoint).
        if effective_model and api_mode != "anthropic_messages":
            try:
                from hermes_cli.models import azure_foundry_model_api_mode

                inferred = azure_foundry_model_api_mode(effective_model)
            except Exception:
                inferred = None
            if inferred:
                api_mode = inferred
        # For Anthropic-style endpoints, strip /v1 suffix
        if api_mode == "anthropic_messages":
            base_url = re.sub(r"/v1/?$", "", base_url)
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
        if provider in {"opencode-zen", "opencode-go"}:
            # Re-derive api_mode from the effective model rather than the
            # persisted api_mode: the opencode providers serve both
            # anthropic_messages and chat_completions models, so the previous
            # session's mode must not leak across /model switches.
            # Refs #16878.
            from hermes_cli.models import opencode_model_api_mode
            api_mode = opencode_model_api_mode(provider, effective_model)
        elif configured_mode and _provider_supports_explicit_api_mode(provider, configured_provider):
            api_mode = configured_mode
        else:
            # Auto-detect Anthropic-compatible endpoints (/anthropic suffix,
            # Kimi /coding, api.openai.com → codex_responses, api.x.ai →
            # codex_responses).
            detected = _detect_api_mode_for_url(base_url)
            if detected:
                api_mode = detected

    # OpenCode base URLs end with /v1 for OpenAI-compatible models, but the
    # Anthropic SDK prepends its own /v1/messages to the base_url.  Strip the
    # trailing /v1 so the SDK constructs the correct path (e.g.
    # https://opencode.ai/zen/go/v1/messages instead of .../v1/v1/messages).
    if api_mode == "anthropic_messages" and provider in {"opencode-zen", "opencode-go"}:
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
    provider_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Check if a credential pool exists for a custom endpoint and return a runtime dict if so."""
    pool_key = get_custom_provider_pool_key(base_url, provider_name=provider_name)
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
            canonical = auth_mod.resolve_provider(requested_norm)
        except AuthError:
            pass
        else:
            # A user-declared ``custom_providers`` entry whose name matches
            # only an *alias* (``kimi`` → built-in ``kimi-coding``) is the
            # user's intended target — alias rewriting would otherwise hijack
            # the request.  We only defer to the built-in when the raw name is
            # the canonical provider itself (``nous``, ``openrouter``, …) so
            # accidentally shadowing a canonical provider still resolves to
            # the built-in. See tests/hermes_cli/test_runtime_provider_resolution.py
            # ``test_named_custom_provider_does_not_shadow_builtin_provider``.
            if (canonical or "").strip().lower() == requested_norm:
                return None

    config = load_config()
    
    # First check providers: dict (new-style user-defined providers)
    providers = config.get("providers")
    if isinstance(providers, dict):
        for ep_name, entry in providers.items():
            if not isinstance(entry, dict):
                continue
            # Match exact name or normalized name
            name_norm = _normalize_custom_provider_name(ep_name)
            # Resolve the API key from the env var name stored in key_env
            key_env = str(entry.get("key_env", "") or "").strip()
            resolved_api_key = os.getenv(key_env, "").strip() if key_env else ""
            # Fall back to inline api_key when key_env is absent or unresolvable
            if not resolved_api_key:
                resolved_api_key = str(entry.get("api_key", "") or "").strip()

            if requested_norm in {ep_name, name_norm, f"custom:{name_norm}"}:
                # Found match by provider key
                base_url = entry.get("api") or entry.get("url") or entry.get("base_url") or ""
                if base_url:
                    result = {
                        "name": entry.get("name", ep_name),
                        "base_url": base_url.strip(),
                        "api_key": resolved_api_key,
                        "model": entry.get("default_model", ""),
                    }
                    # The v11→v12 migration writes the API mode under the new
                    # ``transport`` field, but hand-edited configs may still
                    # use the legacy ``api_mode`` spelling.  Accept both —
                    # the runtime normaliser ``_normalize_custom_provider_entry``
                    # already does, so without this lift every migrated config
                    # silently downgrades codex_responses / anthropic_messages
                    # providers to chat_completions in the resolved runtime.
                    api_mode = _parse_api_mode(entry.get("api_mode") or entry.get("transport"))
                    if api_mode:
                        result["api_mode"] = api_mode
                    return result
            # Also check the 'name' field if present
            display_name = entry.get("name", "")
            if display_name:
                display_norm = _normalize_custom_provider_name(display_name)
                if requested_norm in {display_name, display_norm, f"custom:{display_norm}"}:
                    # Found match by display name
                    base_url = entry.get("api") or entry.get("url") or entry.get("base_url") or ""
                    if base_url:
                        result = {
                            "name": display_name,
                            "base_url": base_url.strip(),
                            "api_key": resolved_api_key,
                            "model": entry.get("default_model", ""),
                        }
                        api_mode = _parse_api_mode(entry.get("api_mode") or entry.get("transport"))
                        if api_mode:
                            result["api_mode"] = api_mode
                        return result

    # Fall back to custom_providers: list (legacy format)
    custom_providers = config.get("custom_providers")
    if isinstance(custom_providers, dict):
        logger.warning(
            "custom_providers in config.yaml is a dict, not a list. "
            "Each entry must be prefixed with '-' in YAML. "
            "Run 'hermes doctor' for details."
        )
        return None

    custom_providers = get_compatible_custom_providers(config)
    if not custom_providers:
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
        provider_key = str(entry.get("provider_key", "") or "").strip()
        provider_key_norm = _normalize_custom_provider_name(provider_key) if provider_key else ""
        provider_menu_key = f"custom:{provider_key_norm}" if provider_key_norm else ""
        if requested_norm not in {name_norm, menu_key, provider_key_norm, provider_menu_key}:
            continue
        result = {
            "name": name.strip(),
            "base_url": base_url.strip(),
            "api_key": str(entry.get("api_key", "") or "").strip(),
        }
        key_env = str(entry.get("key_env", "") or "").strip()
        if key_env:
            result["key_env"] = key_env
        if provider_key:
            result["provider_key"] = provider_key
        api_mode = _parse_api_mode(entry.get("api_mode"))
        if api_mode:
            result["api_mode"] = api_mode
        model_name = str(entry.get("model", "") or "").strip()
        if model_name:
            result["model"] = model_name
        return result

    return None


def _resolve_named_custom_runtime(
    *,
    requested_provider: str,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    # Bare `provider="custom"` with an explicit base_url (e.g. propagated
    # from a `model_aliases:` direct-alias resolution) — build a runtime
    # directly so the alias's base_url actually takes effect.
    requested_norm = (requested_provider or "").strip().lower()
    if requested_norm == "custom" and explicit_base_url:
        base_url = explicit_base_url.strip().rstrip("/")
        # Check credential pool first — mirrors the named-custom-provider path
        # so bare `provider: custom` with a configured custom_providers entry
        # also gets its api_key from the pool instead of env var fallbacks.
        pool_result = _try_resolve_from_custom_pool(base_url, "custom", None)
        if pool_result:
            pool_result["source"] = "direct-alias"
            return pool_result
        api_key_candidates = [
            (explicit_api_key or "").strip(),
            os.getenv("OPENAI_API_KEY", "").strip(),
            os.getenv("OPENROUTER_API_KEY", "").strip(),
        ]
        api_key = next(
            (c for c in api_key_candidates if has_usable_secret(c)),
            "",
        ) or "no-key-required"
        return {
            "provider": "custom",
            "api_mode": _detect_api_mode_for_url(base_url) or "chat_completions",
            "base_url": base_url,
            "api_key": api_key,
            "source": "direct-alias",
            "requested_provider": requested_provider,
        }

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
    pool_result = _try_resolve_from_custom_pool(base_url, "custom", custom_provider.get("api_mode"), provider_name=custom_provider.get("name"))
    if pool_result:
        # Propagate the model name even when using pooled credentials —
        # the pool doesn't know about the custom_providers model field.
        model_name = custom_provider.get("model")
        if model_name:
            pool_result["model"] = model_name
        return pool_result

    api_key_candidates = [
        (explicit_api_key or "").strip(),
        str(custom_provider.get("api_key", "") or "").strip(),
        os.getenv(str(custom_provider.get("key_env", "") or "").strip(), "").strip(),
        os.getenv("OPENAI_API_KEY", "").strip(),
        os.getenv("OPENROUTER_API_KEY", "").strip(),
    ]
    api_key = next((candidate for candidate in api_key_candidates if has_usable_secret(candidate)), "")

    result = {
        "provider": "custom",
        "api_mode": custom_provider.get("api_mode")
        or _detect_api_mode_for_url(base_url)
        or "chat_completions",
        "base_url": base_url,
        "api_key": api_key or "no-key-required",
        "source": f"custom_provider:{custom_provider.get('name', requested_provider)}",
    }
    # Propagate the model name so callers can override self.model when the
    # provider name differs from the actual model string the API expects.
    if custom_provider.get("model"):
        result["model"] = custom_provider["model"]
    return result


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
    env_custom_base_url = os.getenv("CUSTOM_BASE_URL", "").strip()

    # Use config base_url when available and the provider context matches.
    # OPENAI_BASE_URL env var is no longer consulted — config.yaml is
    # the single source of truth for endpoint URLs.
    use_config_base_url = False
    if cfg_base_url.strip() and not explicit_base_url:
        if requested_norm == "auto":
            if not cfg_provider or cfg_provider == "auto":
                use_config_base_url = True
        elif requested_norm == "custom" and _config_base_url_trustworthy_for_bare_custom(
            cfg_base_url, cfg_provider
        ):
            use_config_base_url = True

    base_url = (
        (explicit_base_url or "").strip()
        or env_custom_base_url
        or (cfg_base_url.strip() if use_config_base_url else "")
        or env_openrouter_base_url
        or OPENROUTER_BASE_URL
    ).rstrip("/")

    # Choose API key based on whether the resolved base_url targets OpenRouter.
    # When hitting OpenRouter, prefer OPENROUTER_API_KEY (issue #289).
    # When hitting a custom endpoint (e.g. Z.ai, local LLM), prefer
    # OPENAI_API_KEY so the OpenRouter key doesn't leak to an unrelated
    # provider (issues #420, #560).
    _is_openrouter_url = base_url_host_matches(base_url, "openrouter.ai")
    if _is_openrouter_url:
        api_key_candidates = [
            explicit_api_key,
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
        ]
    else:
        # Custom endpoint: use api_key from config when using config base_url (#1760).
        # When the endpoint is Ollama Cloud, check OLLAMA_API_KEY — it's
        # the canonical env var for ollama.com authentication. Match on
        # HOST, not substring — a custom base_url whose path contains
        # "ollama.com" (e.g. http://127.0.0.1/ollama.com/v1) or whose
        # hostname is a look-alike (ollama.com.attacker.test) must not
        # receive the Ollama credential. See GHSA-76xc-57q6-vm5m.
        _is_ollama_url = base_url_host_matches(base_url, "ollama.com")
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
        # Pass requested_provider so pool lookup prefers name match over base_url,
        # fixing credential mix-ups when multiple custom providers share a base_url.
        pool_result = _try_resolve_from_custom_pool(
            base_url, effective_provider, _parse_api_mode(model_cfg.get("api_mode")),
            provider_name=requested_provider if requested_norm != "custom" else None,
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


def _resolve_azure_foundry_runtime(
    *,
    requested_provider: str,
    model_cfg: Dict[str, Any],
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
    target_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve an Azure Foundry runtime entry.

    Reads ``model.base_url`` + ``model.api_mode`` from config.yaml (or
    explicit overrides), pulls the API key from ``.env`` / env var, and
    strips a trailing ``/v1`` for Anthropic-style endpoints because the
    Anthropic SDK appends ``/v1/messages`` internally.

    Raises :class:`AuthError` when required values are missing.
    """
    explicit_api_key = str(explicit_api_key or "").strip()
    explicit_base_url_clean = str(explicit_base_url or "").strip().rstrip("/")

    cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
    cfg_base_url = ""
    cfg_api_mode = "chat_completions"
    if cfg_provider == "azure-foundry":
        cfg_base_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
        cfg_api_mode = _parse_api_mode(model_cfg.get("api_mode")) or "chat_completions"

    # Model-family inference: Azure Foundry deploys GPT-5.x / codex / o1-o4
    # reasoning models as Responses-API-only.  Calling /chat/completions
    # against them returns 400 "The requested operation is unsupported."
    # Upgrade api_mode when the model name matches, unless the user has
    # explicitly chosen anthropic_messages (Anthropic-style endpoint).
    effective_model = str(target_model or model_cfg.get("default") or "").strip()
    if effective_model and cfg_api_mode != "anthropic_messages":
        try:
            from hermes_cli.models import azure_foundry_model_api_mode

            inferred = azure_foundry_model_api_mode(effective_model)
        except Exception:
            inferred = None
        if inferred:
            cfg_api_mode = inferred

    env_base_url = os.getenv("AZURE_FOUNDRY_BASE_URL", "").strip().rstrip("/")
    base_url = explicit_base_url_clean or cfg_base_url or env_base_url
    if not base_url:
        raise AuthError(
            "Azure Foundry requires a base URL. Set it via 'hermes model' or "
            "the AZURE_FOUNDRY_BASE_URL environment variable."
        )

    api_key = explicit_api_key
    if not api_key:
        try:
            from hermes_cli.config import get_env_value
            api_key = get_env_value("AZURE_FOUNDRY_API_KEY") or ""
        except Exception:
            api_key = ""
    if not api_key:
        api_key = os.getenv("AZURE_FOUNDRY_API_KEY", "").strip()
    if not api_key:
        raise AuthError(
            "Azure Foundry requires an API key. Set AZURE_FOUNDRY_API_KEY in "
            "~/.hermes/.env or run 'hermes model' to configure."
        )

    # Anthropic SDK appends /v1/messages itself, so strip any trailing /v1
    # we inherited from the configured base_url to avoid double-/v1 paths.
    if cfg_api_mode == "anthropic_messages":
        base_url = re.sub(r"/v1/?$", "", base_url)

    source = "explicit" if (explicit_api_key or explicit_base_url) else "config"
    return {
        "provider": "azure-foundry",
        "api_mode": cfg_api_mode,
        "base_url": base_url,
        "api_key": api_key,
        "source": source,
        "requested_provider": requested_provider,
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

    # Azure Foundry: user-configured endpoint with selectable API mode
    if provider == "azure-foundry":
        return _resolve_azure_foundry_runtime(
            requested_provider=requested_provider,
            model_cfg=model_cfg,
            explicit_api_key=explicit_api_key,
            explicit_base_url=explicit_base_url,
        )

    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig and pconfig.auth_type == "api_key":
        env_url = ""
        if pconfig.base_url_env_var:
            env_url = os.getenv(pconfig.base_url_env_var, "").strip().rstrip("/")

        base_url = explicit_base_url
        if not base_url:
            if provider in {"kimi-coding", "kimi-coding-cn"}:
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
        elif provider == "xai":
            api_mode = "codex_responses"
        else:
            configured_mode = _parse_api_mode(model_cfg.get("api_mode"))
            if configured_mode:
                api_mode = configured_mode
            else:
                # Auto-detect from URL (Anthropic /anthropic suffix,
                # api.openai.com → Responses, Kimi /coding, etc.).
                detected = _detect_api_mode_for_url(base_url)
                if detected:
                    api_mode = detected

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
    target_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve runtime provider credentials for agent execution.

    target_model: Optional override for model_cfg.get("default") when
    computing provider-specific api_mode (e.g. OpenCode Zen/Go where different
    models route through different API surfaces). Callers performing an
    explicit mid-session model switch should pass the new model here so
    api_mode is derived from the model they are switching TO, not the stale
    persisted default. Other callers can leave it None to preserve existing
    behavior (api_mode derived from config).
    """
    requested_provider = resolve_requested_provider(requested)

    # Azure Anthropic short-circuit: when explicitly targeting an Azure endpoint
    # with provider="anthropic", bypass _resolve_named_custom_runtime (which would
    # return provider="custom" with chat_completions api_mode and no valid key).
    # Instead, use the Azure key directly with anthropic_messages api_mode.
    _eff_base = (explicit_base_url or "").strip()
    if requested_provider == "anthropic" and "azure.com" in _eff_base:
        _azure_key = (
            (explicit_api_key or "").strip()
            or os.getenv("AZURE_ANTHROPIC_KEY", "").strip()
            or os.getenv("ANTHROPIC_API_KEY", "").strip()
        )
        return {
            "provider": "anthropic",
            "api_mode": "anthropic_messages",
            "base_url": _eff_base.rstrip("/"),
            "api_key": _azure_key,
            "source": "azure-explicit",
            "requested_provider": requested_provider,
        }

    # Azure Foundry: user-configured endpoint with selectable API mode
    # (OpenAI-style chat_completions or Anthropic-style anthropic_messages).
    # Resolve before the custom-runtime / pool / generic paths so Azure
    # config is always picked up from model.base_url + model.api_mode,
    # regardless of whether the caller passed explicit_* args.
    if requested_provider == "azure-foundry":
        azure_runtime = _resolve_azure_foundry_runtime(
            requested_provider=requested_provider,
            model_cfg=_get_model_config(),
            explicit_api_key=explicit_api_key,
            explicit_base_url=explicit_base_url,
            target_model=target_model,
        )
        return azure_runtime

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
                target_model=target_model,
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

    if provider == "minimax-oauth":
        pconfig = PROVIDER_REGISTRY.get(provider)
        if pconfig and pconfig.auth_type == "oauth_minimax":
            from hermes_cli.auth import resolve_minimax_oauth_runtime_credentials
            creds = resolve_minimax_oauth_runtime_credentials()
            return {
                "provider": provider,
                "api_mode": "anthropic_messages",
                "base_url": creds["base_url"],
                "api_key": creds["api_key"],
                "source": creds.get("source", "oauth"),
                "requested_provider": requested_provider,
            }

    if provider == "google-gemini-cli":
        try:
            creds = resolve_gemini_oauth_runtime_credentials()
            return {
                "provider": "google-gemini-cli",
                "api_mode": "chat_completions",
                "base_url": creds.get("base_url", ""),
                "api_key": creds.get("api_key", ""),
                "source": creds.get("source", "google-oauth"),
                "expires_at_ms": creds.get("expires_at_ms"),
                "email": creds.get("email", ""),
                "project_id": creds.get("project_id", ""),
                "requested_provider": requested_provider,
            }
        except AuthError:
            if requested_provider != "auto":
                raise
            logger.info("Google Gemini OAuth credentials failed; "
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
        # Allow base URL override from config.yaml model.base_url, but only
        # when the configured provider is anthropic — otherwise a non-Anthropic
        # base_url (e.g. Codex endpoint) would leak into Anthropic requests.
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        cfg_base_url = ""
        if cfg_provider == "anthropic":
            cfg_base_url = (model_cfg.get("base_url") or "").strip().rstrip("/")
        base_url = cfg_base_url or "https://api.anthropic.com"

        # For Azure AI Foundry endpoints, use ANTHROPIC_API_KEY directly —
        # Claude Code OAuth tokens (sk-ant-oat01) are not accepted by Azure.
        # Azure keys don't start with "sk-ant-" so resolve_anthropic_token()
        # would find the Claude Code OAuth token first (priority 3) and return
        # that instead, causing 401s. Detect Azure endpoints and use the env
        # key directly to bypass the OAuth priority chain.
        _is_azure_endpoint = "azure.com" in base_url.lower() or (
            cfg_base_url and "azure.com" in cfg_base_url.lower()
        )
        if _is_azure_endpoint:
            # Honor user-specified env var hints on the model config before
            # falling back to the built-in AZURE_ANTHROPIC_KEY / ANTHROPIC_API_KEY
            # chain.  Accept both `key_env` (Hermes canonical — matches the
            # custom_providers field name) and `api_key_env` (documented in the
            # Azure Foundry guide and read by most Hermes-compatible importers).
            # Matches the config.yaml examples in website/docs/guides/azure-foundry.md.
            token = ""
            for hint_key in ("key_env", "api_key_env"):
                env_var = str(model_cfg.get(hint_key) or "").strip()
                if env_var:
                    token = os.getenv(env_var, "").strip()
                    if token:
                        break
            # Next: an inline api_key on the model config (useful in multi-profile
            # setups that want to avoid env-var juggling).
            if not token:
                token = str(model_cfg.get("api_key") or "").strip()
            # Finally fall back to the historical fixed names.
            if not token:
                token = (
                    os.getenv("AZURE_ANTHROPIC_KEY", "").strip()
                    or os.getenv("ANTHROPIC_API_KEY", "").strip()
                )
            if not token:
                raise AuthError(
                    "No Azure Anthropic API key found. Set AZURE_ANTHROPIC_KEY or "
                    "ANTHROPIC_API_KEY, or point key_env/api_key_env in your "
                    "config.yaml model section at a custom env var."
                )
        else:
            from agent.anthropic_adapter import resolve_anthropic_token
            token = resolve_anthropic_token()
            if not token:
                raise AuthError(
                    "No Anthropic credentials found. Set ANTHROPIC_TOKEN or ANTHROPIC_API_KEY, "
                    "run 'claude setup-token', or authenticate with 'claude /login'."
                )
        return {
            "provider": "anthropic",
            "api_mode": "anthropic_messages",
            "base_url": base_url,
            "api_key": token,
            "source": "env",
            "requested_provider": requested_provider,
        }

    # AWS Bedrock (native Converse API via boto3)
    if provider == "bedrock":
        from agent.bedrock_adapter import (
            has_aws_credentials,
            resolve_aws_auth_env_var,
            resolve_bedrock_region,
            is_anthropic_bedrock_model,
        )
        # When the user explicitly selected bedrock (not auto-detected),
        # trust boto3's credential chain — it handles IMDS, ECS task roles,
        # Lambda execution roles, SSO, and other implicit sources that our
        # env-var check can't detect.
        is_explicit = requested_provider in {"bedrock", "aws", "aws-bedrock", "amazon-bedrock", "amazon"}
        if not is_explicit and not has_aws_credentials():
            raise AuthError(
                "No AWS credentials found for Bedrock. Configure one of:\n"
                "  - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY\n"
                "  - AWS_PROFILE (for SSO / named profiles)\n"
                "  - IAM instance role (EC2, ECS, Lambda)\n"
                "Or run 'aws configure' to set up credentials.",
                code="no_aws_credentials",
            )
        # Read bedrock-specific config from config.yaml
        _bedrock_cfg = load_config().get("bedrock", {})
        # Region priority: config.yaml bedrock.region → env var → us-east-1
        region = (_bedrock_cfg.get("region") or "").strip() or resolve_bedrock_region()
        auth_source = resolve_aws_auth_env_var() or "aws-sdk-default-chain"
        # Build guardrail config if configured
        _gr = _bedrock_cfg.get("guardrail", {})
        guardrail_config = None
        if _gr.get("guardrail_identifier") and _gr.get("guardrail_version"):
            guardrail_config = {
                "guardrailIdentifier": _gr["guardrail_identifier"],
                "guardrailVersion": _gr["guardrail_version"],
            }
            if _gr.get("stream_processing_mode"):
                guardrail_config["streamProcessingMode"] = _gr["stream_processing_mode"]
            if _gr.get("trace"):
                guardrail_config["trace"] = _gr["trace"]
        # Dual-path routing: Claude models use AnthropicBedrock SDK for full
        # feature parity (prompt caching, thinking budgets, adaptive thinking).
        # Non-Claude models use the Converse API for multi-model support.
        _current_model = str(model_cfg.get("default") or "").strip()
        if is_anthropic_bedrock_model(_current_model):
            # Claude on Bedrock → AnthropicBedrock SDK → anthropic_messages path
            runtime = {
                "provider": "bedrock",
                "api_mode": "anthropic_messages",
                "base_url": f"https://bedrock-runtime.{region}.amazonaws.com",
                "api_key": "aws-sdk",
                "source": auth_source,
                "region": region,
                "bedrock_anthropic": True,  # Signal to use AnthropicBedrock client
                "requested_provider": requested_provider,
            }
        else:
            # Non-Claude (Nova, DeepSeek, Llama, etc.) → Converse API
            runtime = {
                "provider": "bedrock",
                "api_mode": "bedrock_converse",
                "base_url": f"https://bedrock-runtime.{region}.amazonaws.com",
                "api_key": "aws-sdk",
                "source": auth_source,
                "region": region,
                "requested_provider": requested_provider,
            }
        if guardrail_config:
            runtime["guardrail_config"] = guardrail_config
        return runtime

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
        elif provider == "xai":
            api_mode = "codex_responses"
        else:
            configured_provider = str(model_cfg.get("provider") or "").strip().lower()
            # Only honor persisted api_mode when it belongs to the same provider family.
            configured_mode = _parse_api_mode(model_cfg.get("api_mode"))
            if provider in {"opencode-zen", "opencode-go"}:
                # opencode-zen/go must always re-derive api_mode from the
                # target model (not the stale persisted api_mode), because
                # the same provider serves both anthropic_messages
                # (e.g. minimax-m2.7) and chat_completions (e.g.
                # deepseek-v4-flash) and switching models via /model would
                # otherwise carry the previous mode forward, stripping /v1
                # from base_url for chat_completions models and 404'ing.
                # Refs #16878.
                from hermes_cli.models import opencode_model_api_mode
                _effective = target_model or model_cfg.get("default", "")
                api_mode = opencode_model_api_mode(provider, _effective)
            elif configured_mode and _provider_supports_explicit_api_mode(provider, configured_provider):
                api_mode = configured_mode
            else:
                # Auto-detect Anthropic-compatible endpoints by URL convention
                # (e.g. https://api.minimax.io/anthropic, https://dashscope.../anthropic)
                # plus api.openai.com → codex_responses and api.x.ai → codex_responses.
                detected = _detect_api_mode_for_url(base_url)
                if detected:
                    api_mode = detected
        # Strip trailing /v1 for OpenCode Anthropic models (see comment above).
        if api_mode == "anthropic_messages" and provider in {"opencode-zen", "opencode-go"}:
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
