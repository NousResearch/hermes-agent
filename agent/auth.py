"""
Unified credential resolver for ALL Hermes surfaces.

Single entry point for credential resolution — both auxiliary_client.py
(CLI path) and runtime_provider.py (Gateway/Desktop path) call this module.

This follows the pattern established by PR #30911 (Codex unification),
extended to all 19+ providers.

Precedence (highest to lowest):
  1. Explicit overrides (from function args — e.g. /model switch)
  2. Env var override (GLM_BASE_URL, DEEPSEEK_BASE_URL, etc.)
  3. Config.yaml model.base_url (when model.provider matches)
  4. Provider-specific resolver (probe, prefix detection, OAuth)
  5. Pool entry base_url (from auth.json)
  6. PROVIDER_REGISTRY inference_base_url (hardcoded default)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from agent.credential_pool import PooledCredential

logger = logging.getLogger(__name__)

# ── SSRF guard (shared with web_server.py) ─────────────────────────────────

_BLOCKED_HOSTS = frozenset({
    "169.254.169.254",      # AWS/Azure/GCP metadata
    "metadata.google.internal",
    "metadata",
    "fd00:ec2::254",        # AWS IMDSv6
})


def _validate_base_url_safe(url: str) -> str:
    """Sanitize a user-supplied base_url to prevent SSRF.

    Blocks:
    - Non-http(s) schemes (file://, gopher://, dict://, etc.)
    - Cloud metadata endpoints (169.254.169.254, metadata.google.internal)
    - Internal link-local addresses (169.254.x.x)
    - Null byte injection

    Unlike the web_server version, this never raises — it returns the
    original URL if validation fails, and the caller decides what to do
    (env vars and config.yaml are trusted sources, so we only warn).
    """
    import urllib.parse
    import logging
    if not url or not url.strip():
        return url
    url = url.strip()
    # Null byte injection
    if "\x00" in url or "\0" in url or "\u0000" in url:
        logging.getLogger(__name__).warning("Null byte in base_url, ignoring")
        return ""
    parsed = urllib.parse.urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https", ""):
        logging.getLogger(__name__).warning("Non-http scheme in base_url: %s", scheme)
        return ""
    hostname = (parsed.hostname or "").lower()
    if hostname in _BLOCKED_HOSTS:
        logging.getLogger(__name__).warning("Blocked metadata host in base_url: %s", hostname)
        return ""
    if hostname.startswith("169.254."):
        logging.getLogger(__name__).warning("Blocked link-local address: %s", hostname)
        return ""
    return url


# ── Lazy imports to avoid circular dependencies ─────────────────────────────

def _auth():
    import hermes_cli.auth as _m
    return _m

def _rp():
    import hermes_cli.runtime_provider as _m
    return _m

def _models():
    import hermes_cli.models as _m
    return _m

def _constants():
    import hermes_constants as _m
    return _m

# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class ResolvedCredential:
    """The output of the unified resolver — everything a surface needs."""
    provider: str
    api_key: str
    base_url: str
    api_mode: str           # "chat_completions" | "anthropic_messages" | "codex_responses" | "gemini_native"
    source: str             # "manual" | "env" | "config" | "registry" | "oauth" | "explicit" | "pool"
    entry: Optional[PooledCredential] = None
    expires_at: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_env(key: str) -> str:
    """Read an env var, preferring ~/.hermes/.env over os.environ."""
    raw = os.environ.get(key, "").strip()
    if raw:
        return raw
    try:
        from hermes_cli.env_loader import load_env
        env_file = load_env()
        return env_file.get(key, "").strip()
    except Exception:
        return ""


def _get_registry_url(provider: str) -> str:
    """Return the PROVIDER_REGISTRY inference_base_url for a provider."""
    pconfig = _auth().PROVIDER_REGISTRY.get(provider)
    return pconfig.inference_base_url if pconfig else ""


def _get_base_url_env_var(provider: str) -> str:
    """Return the env var name for base URL override (e.g. GLM_BASE_URL)."""
    pconfig = _auth().PROVIDER_REGISTRY.get(provider)
    if pconfig and pconfig.base_url_env_var:
        return pconfig.base_url_env_var
    return ""


# ── Main resolver ───────────────────────────────────────────────────────────

def resolve_provider_credentials(
    *,
    provider: str,
    entry: Optional[PooledCredential] = None,
    model_cfg: Optional[Dict[str, Any]] = None,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
    target_model: Optional[str] = None,
) -> ResolvedCredential:
    """Unified credential resolver for ALL surfaces (CLI, Gateway, Desktop).

    SINGLE ENTRY POINT — both auxiliary_client.py and runtime_provider.py
    call this function. No provider-specific logic lives in those files.

    This function consolidates ~55 branches of provider-specific logic that
    were previously scattered across 4 files (auth.py, runtime_provider.py,
    auxiliary_client.py, credential_pool.py).
    """
    auth = _auth()
    rp = _rp()
    model_cfg = model_cfg or {}

    pconfig = auth.PROVIDER_REGISTRY.get(provider)
    registry_url = pconfig.inference_base_url if pconfig else ""

    # ── Step 1: Extract base_url and api_key from pool entry ──────────
    entry_url = ""
    api_key = ""
    if entry:
        entry_url = (
            getattr(entry, "runtime_base_url", None)
            or getattr(entry, "base_url", None)
            or ""
        ).rstrip("/")
        api_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")

    if explicit_api_key:
        api_key = explicit_api_key
    if explicit_base_url:
        entry_url = explicit_base_url.rstrip("/")

    # ── Step 2: Env var override ──────────────────────────────────────
    env_url = ""
    env_var_name = _get_base_url_env_var(provider)
    if env_var_name:
        env_url = _get_env(env_var_name).strip().rstrip("/")

    # ── Step 3: Config.yaml override ──────────────────────────────────
    cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
    cfg_url = ""
    if cfg_provider == provider:
        cfg_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
    configured_mode = rp._parse_api_mode(model_cfg.get("api_mode"))
    effective_model = target_model or model_cfg.get("default", "")

    # ── Step 4: Provider-specific resolution ──────────────────────────
    resolved_url = entry_url or registry_url
    api_mode = "chat_completions"
    source = "pool" if entry else "registry"
    extras: Dict[str, Any] = {}

    # ═════════════════════════════════════════════════════════════════
    # CATEGORY A: OAuth providers (wrap existing resolve_* functions)
    # ═════════════════════════════════════════════════════════════════

    if provider == "openai-codex":
        api_mode = "codex_responses"
        if not api_key:
            creds = auth.resolve_codex_runtime_credentials()
            api_key = creds.get("api_key", "")
            resolved_url = creds.get("base_url", "").rstrip("/")
            source = creds.get("source", "hermes-auth-store")
            extras["last_refresh"] = creds.get("last_refresh")
        resolved_url = resolved_url or auth.DEFAULT_CODEX_BASE_URL

    elif provider == "xai-oauth":
        api_mode = "codex_responses"
        if not api_key:
            creds = auth.resolve_xai_oauth_runtime_credentials()
            api_key = creds.get("api_key", "")
            resolved_url = creds.get("base_url", "").rstrip("/")
            source = creds.get("source", "hermes-auth-store")
            extras["last_refresh"] = creds.get("last_refresh")
        resolved_url = resolved_url or auth.DEFAULT_XAI_OAUTH_BASE_URL

    elif provider == "qwen-oauth":
        api_mode = "chat_completions"
        if not api_key:
            creds = auth.resolve_qwen_runtime_credentials()
            api_key = creds.get("api_key", "")
            resolved_url = creds.get("base_url", "").rstrip("/")
            source = creds.get("source", "qwen-cli")
            extras["expires_at_ms"] = creds.get("expires_at_ms")
        resolved_url = resolved_url or auth.DEFAULT_QWEN_BASE_URL

    elif provider == "minimax-oauth":
        # MiniMax OAuth tokens are valid ONLY against the Anthropic Messages
        # compatible endpoint. Do not honor stale model.api_mode values.
        api_mode = "anthropic_messages"
        if not api_key:
            creds = auth.resolve_minimax_oauth_runtime_credentials()
            api_key = creds.get("api_key", "")
            resolved_url = creds.get("base_url", "").rstrip("/")
            source = creds.get("source", "oauth")
        resolved_url = resolved_url or (pconfig.inference_base_url if pconfig else "")

    elif provider == "nous":
        api_mode = "chat_completions"
        # Check for Nous inference base URL override
        nous_override = auth.DEFAULT_NOUS_INFERENCE_URL
        try:
            state = auth.get_provider_auth_state("nous") or {}
            nous_override = str(state.get("inference_base_url") or nous_override).strip().rstrip("/")
        except Exception:
            pass
        if not api_key:
            creds = auth.resolve_nous_runtime_credentials()
            api_key = creds.get("api_key", "")
            resolved_url = creds.get("base_url", "").rstrip("/")
            source = creds.get("source", "portal")
            extras["expires_at"] = creds.get("expires_at")
        resolved_url = resolved_url or nous_override

    # ═════════════════════════════════════════════════════════════════
    # CATEGORY B: API-key providers with dedicated resolver
    # ═════════════════════════════════════════════════════════════════

    elif provider == "zai":
        # Z.AI: probe coding/anthropic/standard endpoints (6 total)
        resolved_url = auth._resolve_zai_base_url(
            api_key=api_key,
            default_url=entry_url or registry_url,
            env_override=env_url,
        )
        detected = rp._detect_api_mode_for_url(resolved_url)
        if detected:
            api_mode = detected

    elif provider in ("kimi-coding", "kimi-coding-cn"):
        # Kimi: detect coding-plan keys by prefix (sk-kimi-)
        resolved_url = auth._resolve_kimi_base_url(
            api_key=api_key,
            default_url=entry_url or registry_url,
            env_override=env_url,
        )
        detected = rp._detect_api_mode_for_url(resolved_url)
        if detected:
            api_mode = detected

    # ═════════════════════════════════════════════════════════════════
    # CATEGORY C: API-key providers with inline logic (extracted here)
    # ═════════════════════════════════════════════════════════════════

    elif provider == "anthropic":
        api_mode = "anthropic_messages"
        # Honor config override with safety check
        if cfg_url and rp._anthropic_base_url_override_ok(cfg_url):
            resolved_url = cfg_url
        else:
            resolved_url = entry_url or "https://api.anthropic.com"

    elif provider == "openrouter":
        resolved_url = resolved_url or _constants().OPENROUTER_BASE_URL
        detected = rp._detect_api_mode_for_url(resolved_url)
        if detected:
            api_mode = detected

    elif provider == "copilot":
        api_mode = rp._copilot_runtime_api_mode(model_cfg, api_key)
        resolved_url = resolved_url or (pconfig.inference_base_url if pconfig else "")

    elif provider == "copilot-acp":
        # External process provider (Codex ACP)
        creds = auth.resolve_external_process_provider_credentials(provider)
        api_key = creds.get("api_key", "")
        resolved_url = creds.get("base_url", "").rstrip("/")
        api_mode = "chat_completions"
        source = creds.get("source", "process")
        extras["command"] = creds.get("command", "")
        extras["args"] = list(creds.get("args") or [])

    elif provider == "xai":
        # xAI API key (not OAuth) — uses codex_responses
        api_mode = "codex_responses"
        resolved_url = resolved_url or "https://api.x.ai/v1"

    elif provider == "azure-foundry":
        # Azure Foundry: read api_mode and base_url from config
        if cfg_provider == "azure-foundry" and cfg_url:
            resolved_url = cfg_url
        if configured_mode:
            api_mode = configured_mode
        # Model-family inference for GPT-5.x / codex / o1-o4
        if effective_model and api_mode != "anthropic_messages":
            try:
                inferred = _models().azure_foundry_model_api_mode(effective_model)
            except Exception:
                inferred = None
            if inferred:
                api_mode = inferred

    elif provider == "lmstudio":
        resolved_url = resolved_url or "http://localhost:1234/v1"
        resolved_url = auth._normalize_lmstudio_runtime_base_url(resolved_url)

    elif provider == "gemini":
        # Gemini: check if native or OpenAI-compatible
        try:
            from agent.gemini_native_adapter import is_native_gemini_base_url
            if is_native_gemini_base_url(resolved_url):
                api_mode = "gemini_native"
        except ImportError:
            pass

    elif provider == "bedrock":
        # AWS Bedrock — uses Anthropic Messages API through AWS
        api_mode = "anthropic_messages"

    elif provider in ("minimax", "minimax-cn"):
        # MiniMax API-key: enforce region-correct endpoint
        if provider == "minimax-cn" and "api.minimax.io" in resolved_url:
            resolved_url = "https://api.minimaxi.com/anthropic"
        elif provider == "minimax" and "api.minimaxi.com" in resolved_url:
            resolved_url = "https://api.minimax.io/anthropic"
        detected = rp._detect_api_mode_for_url(resolved_url)
        if detected:
            api_mode = detected

    elif provider in ("opencode-zen", "opencode-go"):
        # OpenCode: re-derive api_mode from the effective model
        api_mode = _models().opencode_model_api_mode(provider, effective_model)
        resolved_url = _models().normalize_opencode_base_url(provider, api_mode, resolved_url)

    else:
        # ═════════════════════════════════════════════════════════════
        # GENERIC API-key provider (deepseek, stepfun, arcee, gmi,
        # alibaba, nvidia, huggingface, xiaomi, tencent-tokenhub, etc.)
        # ═════════════════════════════════════════════════════════════

        # If no api_key from pool, try resolve_api_key_provider_credentials
        # (handles env vars, .env file, and singleton sources)
        if not api_key:
            try:
                creds = auth.resolve_api_key_provider_credentials(provider)
                api_key = creds.get("api_key", "")
                if not resolved_url and creds.get("base_url"):
                    resolved_url = creds["base_url"].rstrip("/")
                source = creds.get("source", "env")
            except Exception:
                pass

        detected = rp._detect_api_mode_for_url(resolved_url)
        if detected:
            api_mode = detected
        if configured_mode and rp._provider_supports_explicit_api_mode(provider, cfg_provider):
            api_mode = configured_mode

        # Honor config.yaml model.base_url when provider matches and pool
        # entry has no explicit base_url (or it equals the registry default).
        if cfg_url and cfg_provider == provider:
            pool_url_is_default = (
                not entry_url
                or (pconfig and entry_url.rstrip("/") == pconfig.inference_base_url.rstrip("/"))
            )
            if pool_url_is_default:
                resolved_url = cfg_url

    # ── Step 5: Apply precedence ──────────────────────────────────────
    # explicit > env > resolved (which already folded config conditionally) > registry
    # NOTE: cfg_url is NOT in this chain unconditionally — it was applied to
    # resolved_url in Step 4 only when pool_url_is_default is True, preserving
    # per-credential endpoints (mirrors runtime_provider.py:475-485 guard).
    final_url = (explicit_base_url.rstrip("/") if explicit_base_url else "") or env_url or resolved_url or registry_url

    # ── Step 6: Empty base_url fallback ───────────────────────────────
    # This is the fix for the base_url="" bug that caused the cascade
    if not final_url:
        final_url = registry_url

    # ── Step 7: SSRF guard — validate the final resolved URL ──────────
    # This runs on EVERY path (pool, config, env, explicit) so blocked
    # hosts never reach the HTTP client.
    final_url = _validate_base_url_safe(final_url)

    # ── Step 8: Strip /v1 for Anthropic endpoints ─────────────────────
    if api_mode == "anthropic_messages":
        final_url = re.sub(r"/v1/?$", "", final_url)

    # ── Step 8: Determine source label ────────────────────────────────
    if explicit_api_key or explicit_base_url:
        source = "explicit"
    elif env_url and env_url == final_url:
        source = "env"
    elif cfg_url and cfg_url == final_url:
        source = "config"
    elif entry:
        source = getattr(entry, "source", "pool")
    elif not source or source == "registry":
        source = "registry"

    # ── Step 9: Optional codex app-server runtime ─────────────────────
    api_mode = rp._maybe_apply_codex_app_server_runtime(
        provider=provider, api_mode=api_mode, model_cfg=model_cfg
    )

    logger.debug(
        "resolve_provider_credentials: provider=%s base_url=%s api_mode=%s source=%s",
        provider, final_url, api_mode, source,
    )

    return ResolvedCredential(
        provider=provider,
        api_key=api_key,
        base_url=final_url,
        api_mode=api_mode,
        source=source,
        entry=entry,
        extras=extras,
    )
