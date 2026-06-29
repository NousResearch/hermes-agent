"""Shared runtime provider resolution for CLI, gateway, cron, and helpers."""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlparse
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from hermes_cli import auth as auth_mod
from agent.credential_pool import CredentialPool, PooledCredential, get_custom_provider_pool_key, load_pool
from agent.secret_scope import get_secret as _get_secret
from hermes_cli.auth import (
    AuthError,
    DEFAULT_CODEX_BASE_URL,
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_XAI_OAUTH_BASE_URL,
    PROVIDER_REGISTRY,
    _agent_key_is_usable,
    _nous_inference_env_override,
    format_auth_error,
    resolve_provider,
    resolve_nous_runtime_credentials,
    resolve_codex_runtime_credentials,
    resolve_xai_oauth_runtime_credentials,
    resolve_qwen_runtime_credentials,
    resolve_api_key_provider_credentials,
    resolve_external_process_provider_credentials,
    has_usable_secret,
)
from hermes_cli.config import get_compatible_custom_providers, load_config
from hermes_constants import OPENROUTER_BASE_URL
from utils import base_url_host_matches, base_url_hostname, env_int


def _getenv(name: str, default: str = "") -> str:
    """Profile-scoped replacement for ``os.getenv`` on credential/provider reads.

    Routes through the secret scope (Workstream A): identical to ``os.getenv``
    when multiplexing is off, scope-aware (and fail-closed on an unscoped read)
    when on. Genuinely-global vars are handled inside ``get_secret`` and still
    read ``os.environ``. Keeps the ``(name, default) -> str`` contract every
    call site here already relies on.
    """
    val = _get_secret(name, default)
    return val if val is not None else default


def _normalize_custom_provider_name(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


def _loopback_hostname(host: str) -> bool:
    h = (host or "").lower().rstrip(".")
    return h in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _config_base_url_trustworthy_for_bare_custom(cfg_base_url: str, cfg_provider: str) -> bool:
    """Decide whether ``model.base_url`` may back bare ``custom`` runtime resolution.

    GitHub #14676: the model picker can select Custom while ``model.provider`` still reflects a
    previous provider. Reject non-loopback URLs unless the YAML provider is already ``custom``
    (or one of the local-server aliases that resolve to ``custom`` — ollama, vllm, llamacpp, …),
    so a stale OpenRouter/Z.ai base_url cannot hijack local ``custom`` sessions.
    """
    cfg_provider_norm = (cfg_provider or "").strip().lower()
    bu = (cfg_base_url or "").strip()
    if not bu:
        return False
    if cfg_provider_norm == "custom":
        return True
    # GitHub #27132: provider aliases that resolve to "custom" at runtime
    # (ollama, vllm, llamacpp, …) should be trusted the same way "custom"
    # is, otherwise a legit LAN/WireGuard ollama endpoint silently falls
    # through to OpenRouter.
    try:
        from hermes_cli.auth import resolve_provider as _resolve_provider

        if _resolve_provider(cfg_provider_norm) == "custom":
            return True
    except Exception:
        pass
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
    path = urlparse(normalized).path.rstrip("/")
    if path.endswith("/anthropic") or path.endswith("/anthropic/v1"):
        return "anthropic_messages"
    if hostname == "api.kimi.com" and "/coding" in normalized:
        return "anthropic_messages"
    return None


def _resolve_plain_custom_api_mode(model_cfg: Dict[str, Any], base_url: str) -> str:
    """Resolve api_mode for legacy/plain ``provider: custom`` endpoints.

    Custom endpoints should stay conservative by default. Only direct OpenAI/xAI
    URLs imply Responses API automatically; named custom providers can opt in via
    their own ``api_mode`` field. This also prevents a stale persisted
    ``model.api_mode: codex_responses`` from forcing generic relays onto the
    Responses path after upgrades or /reset.
    """
    configured_mode = _parse_api_mode(model_cfg.get("api_mode"))
    detected_mode = _detect_api_mode_for_url(base_url)

    if configured_mode == "codex_responses" and detected_mode != "codex_responses":
        logger.info(
            "Ignoring persisted custom api_mode=codex_responses for non-OpenAI endpoint %s",
            base_url or "(unknown)",
        )
        configured_mode = None

    return configured_mode or detected_mode or "chat_completions"


def _host_derived_api_key(base_url: str) -> str:
    """Look up `<VENDOR>_API_KEY` in the env, derived from the base URL host.

    Examples:
        https://api.deepseek.com/v1   → DEEPSEEK_API_KEY
        https://api.groq.com/openai/v1 → GROQ_API_KEY
        https://api.mistral.ai/v1     → MISTRAL_API_KEY
        https://generativelanguage.googleapis.com/v1beta/openai/ → GOOGLEAPIS_API_KEY

    Returns the env value (stripped) or "". Never returns env vars whose names
    are already explicitly checked elsewhere — those are handled by their own
    host-gated paths (OPENAI/OPENROUTER/OLLAMA).

    The vendor label is the *registrable* portion of the hostname: strip
    ``api.`` / ``www.`` prefixes, then take the second-to-last label
    (``api.deepseek.com`` → ``deepseek``). Falls back to "" for hostnames
    that don't yield a usable vendor label (IPs, loopback, single-label
    hosts).
    """
    hostname = base_url_hostname(base_url)
    if not hostname:
        return ""
    # Reject IPv4 / IPv6 / loopback — no meaningful vendor label.
    if any(ch.isdigit() for ch in hostname.split(".")[-1]):
        # Last label starts with a digit → likely IP. (TLDs are never numeric.)
        return ""
    if hostname in ("localhost",) or ":" in hostname:
        return ""
    labels = [lbl for lbl in hostname.split(".") if lbl]
    # Strip common API/CDN prefixes.
    while labels and labels[0] in ("api", "www"):
        labels.pop(0)
    if len(labels) < 2:
        return ""
    # Take the *registrable* label (second-to-last). For typical provider
    # hosts this is what users intuitively call "the vendor":
    #   deepseek.com               → labels[-2] = "deepseek"  ✓
    #   api.groq.com → groq.com    → labels[-2] = "groq"      ✓
    #   api.mistral.ai             → labels[-2] = "mistral"   ✓
    # Crucially, lookalike hosts pick the ATTACKER's label, not the spoofed
    # vendor:
    #   api.deepseek.com.attacker.test → labels[-2] = "attacker"
    # so DEEPSEEK_API_KEY stays put and the chain falls through to
    # no-key-required. This mirrors how `base_url_host_matches` resists the
    # same lookalike attack for explicit hosts.
    vendor = labels[-2]
    # Sanitize to env var charset: A-Z, 0-9, underscore.
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in vendor).upper()
    if not sanitized or not sanitized[0].isalpha():
        return ""
    # Don't re-derive env vars already handled by explicit host-gated paths.
    if sanitized in ("OPENAI", "OPENROUTER", "OLLAMA"):
        return ""
    env_name = f"{sanitized}_API_KEY"
    return (_getenv(env_name, "") or "").strip()


def _anthropic_base_url_override_ok(base_url: str) -> bool:
    """Decide whether a configured ``model.base_url`` may back native Anthropic.

    Native ``provider: anthropic`` resolution honors ``model.base_url`` so users
    can point at Anthropic-compatible endpoints (official Anthropic/Claude hosts,
    Azure Foundry, MiniMax/Zhipu/LiteLLM-style ``/anthropic`` proxies, Kimi's
    ``/coding`` route). But a config can carry a *stale* non-Anthropic URL — e.g.
    ``provider: anthropic`` left with ``base_url: https://openrouter.ai/api/v1``
    after a provider switch — which would route Anthropic OAuth/setup-token
    traffic to an OpenAI-compatible aggregator and 404. Ignore those.

    Returns True only when the URL plausibly speaks the Anthropic Messages
    protocol; otherwise the caller falls back to ``https://api.anthropic.com``.
    """
    candidate = (base_url or "").strip()
    if not candidate:
        return False

    hostname = (base_url_hostname(candidate) or "").lower()
    if not hostname:
        return False

    # Official Anthropic / Claude hosts.
    if hostname == "api.anthropic.com" or hostname.endswith(".anthropic.com") or hostname.endswith(".claude.com"):
        return True
    # Azure Foundry Anthropic endpoints (handled specially downstream).
    if hostname.endswith(".azure.com"):
        return True
    # Anthropic-compatible proxies conventionally expose the native Messages
    # protocol under a ``/anthropic`` suffix, and Kimi under ``/coding`` — same
    # signal _detect_api_mode_for_url() uses to pick anthropic_messages.
    if _detect_api_mode_for_url(candidate) == "anthropic_messages":
        return True
    # Bare api.kimi.com without the /coding path is not an Anthropic endpoint.
    return False


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
    except Exception as exc:
        # Log instead of silently swallowing — aids debugging when
        # local model auto-detection fails unexpectedly.
        logger.debug("Auto-detect model from %s failed: %s", base_url, exc)
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


_VALID_API_MODES = {
    "chat_completions",
    "codex_responses",
    "anthropic_messages",
    "bedrock_converse",
    # Optional opt-in: hand the entire turn to a `codex app-server` subprocess
    # so terminal/file-ops/patching/sandboxing run inside Codex's own runtime
    # instead of Hermes' tool dispatch. Gated behind config key
    # `model.openai_runtime == "codex_app_server"` AND provider in
    # {"openai", "openai-codex"}. Default is unchanged.
    "codex_app_server",
}


def _parse_api_mode(raw: Any) -> Optional[str]:
    """Validate an api_mode value from config. Returns None if invalid."""
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in _VALID_API_MODES:
            return normalized
    return None


def _nous_inference_base_url_override() -> str:
    """Return the trusted Nous runtime base URL override, if configured.

    Delegates to ``auth._nous_inference_env_override`` so every
    ``NOUS_INFERENCE_BASE_URL`` read shares one normalization path
    (trailing-slash stripping, blank → empty). The env source is trusted
    and intentionally bypasses the network host allowlist there.
    """
    return _nous_inference_env_override() or ""


def _maybe_apply_codex_app_server_runtime(
    *,
    provider: str,
    api_mode: str,
    model_cfg: Optional[Dict[str, Any]],
) -> str:
    """Optional opt-in: rewrite api_mode → "codex_app_server" for OpenAI/Codex
    providers when the user has explicitly enabled that runtime via
    `model.openai_runtime: codex_app_server` in config.yaml.

    Default behavior is preserved: when the key is unset, "auto", or empty,
    this function is a no-op. Only providers in {"openai", "openai-codex"}
    are eligible — other providers (anthropic, openrouter, etc.) cannot be
    rerouted through codex.

    Returns the (possibly-rewritten) api_mode."""
    if not model_cfg:
        return api_mode
    if provider not in {"openai", "openai-codex"}:
        return api_mode
    runtime = str(model_cfg.get("openai_runtime") or "").strip().lower()
    if runtime == "codex_app_server":
        return "codex_app_server"
    return api_mode


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
    elif provider == "xai-oauth":
        api_mode = "codex_responses"
        base_url = base_url or DEFAULT_XAI_OAUTH_BASE_URL
    elif provider == "qwen-oauth":
        api_mode = "chat_completions"
        base_url = base_url or DEFAULT_QWEN_BASE_URL
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
            if not _anthropic_base_url_override_ok(cfg_base_url):
                cfg_base_url = ""
        base_url = cfg_base_url or base_url or "https://api.anthropic.com"
    elif provider == "openrouter":
        base_url = base_url or OPENROUTER_BASE_URL
    elif provider == "xai":
        api_mode = "codex_responses"
    elif provider == "nous":
        api_mode = "chat_completions"
        base_url = _nous_inference_base_url_override() or base_url
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

    # Optional opt-in: route OpenAI/Codex turns through `codex app-server`.
    # Inert when `model.openai_runtime` is unset or "auto".
    api_mode = _maybe_apply_codex_app_server_runtime(
        provider=provider, api_mode=api_mode, model_cfg=model_cfg
    )

    if provider == "lmstudio":
        base_url = auth_mod._normalize_lmstudio_runtime_base_url(base_url)

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
    env_provider = _getenv("HERMES_INFERENCE_PROVIDER", "").strip().lower()
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


def _lift_max_output_tokens(entry: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Propagate a per-provider output cap onto the resolved runtime dict.

    Accepts ``max_output_tokens`` or ``max_tokens`` on a ``custom_providers``
    entry so a provider block can pin its own output limit. Gateway and CLI
    map this onto ``AIAgent.max_tokens`` only when the top-level
    ``model.max_tokens`` isn't set, so the documented global key still wins.
    """
    for _k in ("max_output_tokens", "max_tokens"):
        _v = entry.get(_k)
        if isinstance(_v, int) and _v > 0:
            result["max_output_tokens"] = _v
            return


def _get_named_custom_provider(requested_provider: str) -> Optional[Dict[str, Any]]:
    requested_norm = _normalize_custom_provider_name(requested_provider or "")
    if not requested_norm:
        return None

    # Bare "custom" is normally an incomplete spec — the canonical form is
    # "custom:<name>" — and is otherwise owned by the model.base_url "bare
    # custom" trust path. BUT a user may literally name a ``providers:`` (or
    # legacy ``custom_providers:``) entry "custom" (e.g. ``providers.custom``
    # pointing at cliproxy). We used to return None here *before* scanning
    # config, so such an entry was never matched and resolution fell through to
    # the global default (Codex) — the cause of cron jobs with
    # ``provider: "custom"`` failing with ``auth_unavailable: providers=codex``.
    # Fall through to the config scan instead; if no entry is literally named
    # "custom" it still returns None at the end, preserving the trust path.

    # Raw names should only map to custom providers when they are not already
    # valid built-in providers or aliases. Explicit menu keys like
    # ``custom:local`` always target the saved custom provider. Bare "custom"
    # is exempt from the shadow check — it is not a built-in to defer to.
    if requested_norm == "auto":
        return None
    if requested_norm != "custom" and not requested_norm.startswith("custom:"):
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
            resolved_api_key = _getenv(key_env, "").strip() if key_env else ""
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
                    extra_body = entry.get("extra_body")
                    if isinstance(extra_body, dict):
                        result["extra_body"] = dict(extra_body)
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
                    _lift_max_output_tokens(entry, result)
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
                        extra_body = entry.get("extra_body")
                        if isinstance(extra_body, dict):
                            result["extra_body"] = dict(extra_body)
                        api_mode = _parse_api_mode(entry.get("api_mode") or entry.get("transport"))
                        if api_mode:
                            result["api_mode"] = api_mode
                        _lift_max_output_tokens(entry, result)
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
        extra_body = entry.get("extra_body")
        if isinstance(extra_body, dict):
            result["extra_body"] = dict(extra_body)
        api_mode = _parse_api_mode(entry.get("api_mode"))
        if api_mode:
            result["api_mode"] = api_mode
        model_name = str(entry.get("model", "") or "").strip()
        if model_name:
            result["model"] = model_name
        _lift_max_output_tokens(entry, result)
        return result

    return None


def has_named_custom_provider(requested_provider: str) -> bool:
    """Return True when config defines a custom provider matching the request.

    Thin public wrapper around :func:`_get_named_custom_provider` so other
    modules (e.g. the cronjob tool) can decide whether a provider name will
    actually resolve to a configured ``providers:`` / ``custom_providers:``
    entry — without reaching into a private helper or duplicating the scan.
    """
    try:
        return _get_named_custom_provider(requested_provider) is not None
    except Exception:
        return False


def find_custom_provider_identity(base_url: str) -> Optional[str]:
    """Map an endpoint URL back to its canonical ``custom:<name>`` menu key.

    Returns the ``custom:<normalized-name>`` slug of the first ``providers:``
    / ``custom_providers:`` entry whose base_url matches, or ``None`` when no
    entry owns the URL.

    Session persistence stores the agent's *resolved* provider, and for every
    named custom endpoint that is the literal string ``"custom"`` — the entry
    name is lost, and the api_key is deliberately never persisted. The
    endpoint URL is the one durable fact that survives the round-trip, so
    this reverse lookup lets persist/rebuild paths recover the entry identity
    (and with it key_env/api_key/api_mode resolution via
    :func:`_get_named_custom_provider`) instead of failing with
    ``auth_unavailable`` or silently rebuilding with placeholder credentials.
    """
    target = _normalize_base_url_for_match(base_url)
    if not target:
        return None
    try:
        config = load_config()
    except Exception:
        return None

    providers = config.get("providers")
    if isinstance(providers, dict):
        for ep_name, entry in providers.items():
            if not isinstance(entry, dict):
                continue
            entry_url = (
                entry.get("api") or entry.get("url") or entry.get("base_url") or ""
            )
            if _normalize_base_url_for_match(entry_url) == target:
                return f"custom:{_normalize_custom_provider_name(str(ep_name))}"

    try:
        custom_providers = get_compatible_custom_providers(config)
    except Exception:
        custom_providers = None
    for entry in custom_providers or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        if _normalize_base_url_for_match(entry.get("base_url")) == target:
            return f"custom:{_normalize_custom_provider_name(name)}"

    return None


def canonical_custom_identity(
    *,
    base_url: Optional[str] = None,
    config_provider: Optional[str] = None,
) -> Optional[str]:
    """Recover a routable ``custom:<name>`` identity for a bare custom provider.

    The bare string ``"custom"`` is the *resolved billing class* shared by
    every named ``providers:`` / ``custom_providers:`` entry — it is NOT a
    routable provider identity (``resolve_runtime_provider("custom")`` falls
    through to the OpenRouter default URL with no api_key, which surfaces to
    the user as "No LLM provider configured").

    Any code path that persists or restores a session's provider override
    must run the resolved provider through this helper so a bare ``"custom"``
    is upgraded back to its durable ``custom:<name>`` menu key. Two recovery
    sources, in priority order:

    1. ``base_url`` — reverse-lookup the entry that owns the endpoint URL
       (the one fact that always survives the persistence round-trip when a
       URL was recorded).
    2. ``config_provider`` — the active ``config.model.provider`` (or its
       ``provider``/``HERMES_INFERENCE_PROVIDER`` equivalent). When the agent
       was built without a base_url on the override (the recurring
       Desktop/TUI regression vector), the configured provider is the only
       durable identity left, so fall back to it when it names a real entry.

    Returns ``custom:<name>`` when a routable identity is recovered, else
    ``None`` (caller keeps whatever it had — bare ``"custom"`` only as a last
    resort, e.g. a genuine ad-hoc endpoint with no config entry).
    """
    # 1. Reverse-lookup by endpoint URL.
    if base_url:
        identity = find_custom_provider_identity(base_url)
        if identity:
            return identity

    # 2. Fall back to the configured provider when it names a real entry.
    candidate = str(config_provider or "").strip()
    if not candidate:
        try:
            candidate = str(_get_model_config().get("provider") or "").strip()
        except Exception:
            candidate = ""
    if not candidate:
        candidate = os.environ.get("HERMES_INFERENCE_PROVIDER", "").strip()

    candidate_norm = _normalize_custom_provider_name(candidate)
    # A bare/non-routable candidate cannot heal a bare custom override.
    if not candidate_norm or candidate_norm in {"custom", "auto", "openrouter"}:
        return None
    # Only return it when it actually resolves to a configured custom entry,
    # so we never invent a `custom:<x>` that resolution can't honor.
    try:
        if _get_named_custom_provider(candidate) is not None:
            if candidate_norm.startswith("custom:"):
                return candidate_norm
            return f"custom:{candidate_norm}"
    except Exception:
        pass
    return None


def _normalize_base_url_for_match(value) -> str:
    return str(value or "").strip().rstrip("/").lower()


def _custom_provider_request_overrides(custom_provider: Dict[str, Any]) -> Dict[str, Any]:
    extra_body = custom_provider.get("extra_body")
    if not isinstance(extra_body, dict) or not extra_body:
        return {}
    return {"extra_body": dict(extra_body)}


def _resolve_named_custom_runtime(
    *,
    requested_provider: str,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    # Bare `provider="custom"` with an explicit base_url (e.g. propagated
    # from a `model_aliases:` direct-alias resolution) — build a runtime
    # directly so the alias's base_url actually takes effect.
    #
    # GitHub #27132: provider aliases that resolve to "custom" at runtime
    # (ollama, vllm, llamacpp, …) are treated identically here, so a YAML
    # `provider: ollama` with a LAN/WireGuard `base_url` doesn't silently
    # fall through to OpenRouter.
    requested_norm = (requested_provider or "").strip().lower()
    if requested_norm and requested_norm != "custom":
        try:
            from hermes_cli.auth import resolve_provider as _resolve_provider

            if _resolve_provider(requested_norm) == "custom":
                requested_norm = "custom"
        except Exception:
            pass
    if requested_norm == "custom" and explicit_base_url:
        base_url = explicit_base_url.strip().rstrip("/")
        # Check credential pool first — mirrors the named-custom-provider path
        # so bare `provider: custom` with a configured custom_providers entry
        # also gets its api_key from the pool instead of env var fallbacks.
        pool_result = _try_resolve_from_custom_pool(base_url, "custom", None)
        if pool_result:
            pool_result["source"] = "direct-alias"
            return pool_result
        _da_is_openai_url   = base_url_host_matches(base_url, "openai.com") or base_url_host_matches(base_url, "openai.azure.com")
        _da_is_openrouter   = base_url_host_matches(base_url, "openrouter.ai")
        api_key_candidates = [
            (explicit_api_key or "").strip(),
            # Gate env key fallbacks on authoritative hosts (#28660)
            (_getenv("OPENAI_API_KEY", "").strip()     if _da_is_openai_url else ""),
            (_getenv("OPENROUTER_API_KEY", "").strip() if _da_is_openrouter  else ""),
            # Bonus (#28660): derive `<VENDOR>_API_KEY` from the host so users
            # who set DEEPSEEK_API_KEY / GROQ_API_KEY / MISTRAL_API_KEY get the
            # intuitive match without configuring `custom_providers` first.
            _host_derived_api_key(base_url),
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
        if isinstance(custom_provider.get("max_output_tokens"), int):
            pool_result["max_output_tokens"] = custom_provider["max_output_tokens"]
        request_overrides = _custom_provider_request_overrides(custom_provider)
        if request_overrides:
            pool_result["request_overrides"] = {
                **dict(pool_result.get("request_overrides") or {}),
                **request_overrides,
            }
        return pool_result

    _cp_is_openai_url   = base_url_host_matches(base_url, "openai.com") or base_url_host_matches(base_url, "openai.azure.com")
    _cp_is_openrouter   = base_url_host_matches(base_url, "openrouter.ai")
    api_key_candidates = [
        (explicit_api_key or "").strip(),
        str(custom_provider.get("api_key", "") or "").strip(),
        _getenv(str(custom_provider.get("key_env", "") or "").strip(), "").strip(),
        # Gate provider env keys on their authoritative hosts — sending
        # OPENAI_API_KEY to a local-llm endpoint leaks credentials (#28660).
        (_getenv("OPENAI_API_KEY", "").strip()     if _cp_is_openai_url  else ""),
        (_getenv("OPENROUTER_API_KEY", "").strip() if _cp_is_openrouter  else ""),
        # Bonus (#28660): derive `<VENDOR>_API_KEY` from the host as a final
        # fallback when key_env wasn't set explicitly.
        _host_derived_api_key(base_url),
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
    if isinstance(custom_provider.get("max_output_tokens"), int):
        result["max_output_tokens"] = custom_provider["max_output_tokens"]
    request_overrides = _custom_provider_request_overrides(custom_provider)
    if request_overrides:
        result["request_overrides"] = request_overrides
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
    # GitHub #27132: provider aliases that resolve to "custom" (ollama,
    # vllm, llamacpp, …) follow the same base_url trust + routing rules
    # as a bare `provider: custom`. Normalising here keeps every check
    # below — `requested_norm == "custom"`, the trust check, the pool
    # gate up the stack — alias-aware without duplicating the alias map.
    if requested_norm and requested_norm != "custom":
        try:
            from hermes_cli.auth import resolve_provider as _resolve_provider

            if _resolve_provider(requested_norm) == "custom":
                requested_norm = "custom"
        except Exception:
            pass

    env_openrouter_base_url = _getenv("OPENROUTER_BASE_URL", "").strip()
    env_custom_base_url = _getenv("CUSTOM_BASE_URL", "").strip()

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
    # Also treat explicitly-configured OpenRouter mirrors/proxies as OpenRouter
    # for key selection — if the user set OPENROUTER_BASE_URL or requested
    # provider=openrouter explicitly, OPENROUTER_API_KEY should still be used.
    _is_openrouter_context = _is_openrouter_url or (
        requested_norm == "openrouter"
        and (env_openrouter_base_url or base_url == env_openrouter_base_url)
        and base_url == (env_openrouter_base_url or "").rstrip("/")
    )
    if _is_openrouter_context:
        api_key_candidates = [
            explicit_api_key,
            _getenv("OPENROUTER_API_KEY"),
            _getenv("OPENAI_API_KEY"),
        ]
    else:
        # Custom endpoint: use api_key from config when using config base_url (#1760).
        # When the endpoint is Ollama Cloud, check OLLAMA_API_KEY — it's
        # the canonical env var for ollama.com authentication. Match on
        # HOST, not substring — a custom base_url whose path contains
        # "ollama.com" (e.g. http://127.0.0.1/ollama.com/v1) or whose
        # hostname is a look-alike (ollama.com.attacker.test) must not
        # receive the Ollama credential. See GHSA-76xc-57q6-vm5m.
        _is_ollama_url    = base_url_host_matches(base_url, "ollama.com")
        _is_openai_url    = base_url_host_matches(base_url, "openai.com")
        _is_openai_azure  = base_url_host_matches(base_url, "openai.azure.com")
        # Gate each provider key on its own host — sending OPENAI_API_KEY or
        # OPENROUTER_API_KEY to an unrelated custom endpoint (DeepSeek, Groq,
        # Mistral, …) leaks credentials and causes 401s (issue #28660).
        # Mirrors the OLLAMA_API_KEY host-gate added in GHSA-76xc-57q6-vm5m.
        api_key_candidates = [
            explicit_api_key,
            (cfg_api_key if use_config_base_url else ""),
            (_getenv("OLLAMA_API_KEY")     if _is_ollama_url                       else ""),
            (_getenv("OPENAI_API_KEY")     if (_is_openai_url or _is_openai_azure) else ""),
            (_getenv("OPENROUTER_API_KEY") if _is_openrouter_url                   else ""),
            # Bonus (#28660): derive `<VENDOR>_API_KEY` from the host so users
            # who set DEEPSEEK_API_KEY / GROQ_API_KEY / MISTRAL_API_KEY get the
            # intuitive match. Helper returns "" for IPs/loopback and for env
            # vars already handled by the explicit host-gated paths above.
            _host_derived_api_key(base_url),
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
        "api_mode": _resolve_plain_custom_api_mode(model_cfg, base_url)
        if effective_provider == "custom"
        else _parse_api_mode(model_cfg.get("api_mode"))
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

    When ``model.auth_mode == "entra_id"`` (and the model is OpenAI-style),
    the returned ``api_key`` is a zero-arg callable produced by
    :func:`agent.azure_identity_adapter.build_token_provider` rather than
    a string. Downstream code that constructs an OpenAI SDK client passes
    this through unchanged (the SDK accepts ``Callable[[], str]`` for
    ``api_key`` and calls it before every request). Code paths that need
    a string (logging, manual HTTP probes, header injection) must use the
    helpers in ``agent.azure_identity_adapter``.

    Raises :class:`AuthError` when required values are missing.
    """
    explicit_api_key = str(explicit_api_key or "").strip()
    explicit_base_url_clean = str(explicit_base_url or "").strip().rstrip("/")

    cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
    cfg_base_url = ""
    cfg_api_mode = "chat_completions"
    cfg_auth_mode = "api_key"
    cfg_entra: Dict[str, Any] = {}
    if cfg_provider == "azure-foundry":
        cfg_base_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
        cfg_api_mode = _parse_api_mode(model_cfg.get("api_mode")) or "chat_completions"
        cfg_auth_mode = str(model_cfg.get("auth_mode") or "api_key").strip().lower() or "api_key"
        _entra = model_cfg.get("entra")
        if isinstance(_entra, dict):
            cfg_entra = _entra

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

    env_base_url = _getenv("AZURE_FOUNDRY_BASE_URL", "").strip().rstrip("/")
    base_url = explicit_base_url_clean or cfg_base_url or env_base_url
    if not base_url:
        raise AuthError(
            "Azure Foundry requires a base URL. Set it via 'hermes model' or "
            "the AZURE_FOUNDRY_BASE_URL environment variable."
        )

    # Anthropic SDK appends /v1/messages itself, so strip any trailing /v1
    # we inherited from the configured base_url to avoid double-/v1 paths.
    if cfg_api_mode == "anthropic_messages":
        base_url = re.sub(r"/v1/?$", "", base_url)

    # ── Entra ID (Microsoft Foundry recommended path) ──────────────────
    #
    # OpenAI-style endpoints use the OpenAI SDK's native callable
    # ``api_key=`` contract — the SDK mints a fresh JWT per request
    # automatically.
    #
    # Anthropic-style endpoints (Claude on Foundry) take the callable
    # too: :func:`agent.anthropic_adapter.build_anthropic_client`
    # detects the callable and constructs an ``httpx.Client`` with a
    # request event hook that injects a fresh ``Authorization: Bearer``
    # header per request (the Anthropic SDK does not accept callables
    # natively). From the runtime resolver's perspective both modes
    # are identical — return the callable api_key and let the
    # downstream SDK wrapper handle the contract difference.
    if cfg_auth_mode == "entra_id":
        if explicit_api_key:
            # User passed --api-key on the CLI while config says entra_id —
            # honour the explicit string (escape hatch for one-off testing).
            api_key: Any = explicit_api_key
            source = "explicit"
            auth_mode = "api_key"
        else:
            try:
                from agent.azure_identity_adapter import (
                    EntraIdentityConfig,
                    SCOPE_AI_AZURE_DEFAULT,
                    build_token_provider,
                )
            except Exception as exc:
                raise AuthError(
                    "Azure Foundry Entra ID auth requires the 'azure-identity' "
                    "package. Install it with: pip install azure-identity "
                    f"(import failed: {exc})"
                ) from exc

            scope = (
                str(cfg_entra.get("scope") or "").strip()
                or SCOPE_AI_AZURE_DEFAULT
            )
            try:
                entra_config = EntraIdentityConfig(
                    scope=scope,
                )
                token_provider = build_token_provider(config=entra_config)
            except ImportError as exc:
                raise AuthError(str(exc)) from exc
            api_key = token_provider
            source = "entra_id"
            auth_mode = "entra_id"

        clean_entra = {}
        if auth_mode == "entra_id":
            configured_scope = str(cfg_entra.get("scope") or "").strip()
            if configured_scope:
                clean_entra["scope"] = configured_scope

        return {
            "provider": "azure-foundry",
            "api_mode": cfg_api_mode,
            "base_url": base_url,
            "api_key": api_key,
            "auth_mode": auth_mode,
            "entra": clean_entra,
            "source": source,
            "requested_provider": requested_provider,
        }

    # ── Static API key (legacy / default) ──────────────────────────────
    api_key = explicit_api_key
    if not api_key:
        try:
            from hermes_cli.config import get_env_value
            api_key = get_env_value("AZURE_FOUNDRY_API_KEY") or ""
        except Exception:
            api_key = ""
    if not api_key:
        api_key = _getenv("AZURE_FOUNDRY_API_KEY", "").strip()
    if not api_key:
        raise AuthError(
            "Azure Foundry requires an API key. Set AZURE_FOUNDRY_API_KEY in "
            "~/.hermes/.env or run 'hermes model' to configure. To use "
            "keyless Microsoft Entra ID auth instead, set "
            "model.auth_mode: entra_id in config.yaml (or pick "
            "'Microsoft Entra ID' in 'hermes model')."
        )

    source = "explicit" if (explicit_api_key or explicit_base_url) else "config"
    return {
        "provider": "azure-foundry",
        "api_mode": cfg_api_mode,
        "base_url": base_url,
        "api_key": api_key,
        "auth_mode": "api_key",
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
            if not _anthropic_base_url_override_ok(cfg_base_url):
                cfg_base_url = ""
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
            or _nous_inference_base_url_override()
            or str(state.get("inference_base_url") or auth_mod.DEFAULT_NOUS_INFERENCE_URL).strip().rstrip("/")
        )
        # Only use the agent_key compatibility field for inference when it
        # contains a NAS invoke JWT; raw OAuth access_token fallback is handled
        # by resolve_nous_runtime_credentials().
        api_key = explicit_api_key or (
            str(state.get("agent_key") or "").strip()
            if _agent_key_is_usable(
                state,
                max(60, env_int("HERMES_NOUS_MIN_KEY_TTL_SECONDS", 1800)),
            )
            else ""
        )
        expires_at = state.get("agent_key_expires_at") or state.get("expires_at")
        if not api_key:
            creds = resolve_nous_runtime_credentials(
                timeout_seconds=float(_getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
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
            env_url = _getenv(pconfig.base_url_env_var, "").strip().rstrip("/")

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

    if requested_provider == "moa":
        return {
            "provider": "moa",
            "api_mode": "chat_completions",
            "base_url": "moa://local",
            "api_key": "moa-virtual-provider",
            "source": "moa-virtual-provider",
            "requested_provider": requested_provider,
        }

    # Azure Anthropic short-circuit: when explicitly targeting an Azure endpoint
    # with provider="anthropic", bypass _resolve_named_custom_runtime (which would
    # return provider="custom" with chat_completions api_mode and no valid key).
    # Instead, use the Azure key directly with anthropic_messages api_mode.
    _eff_base = (explicit_base_url or "").strip()
    if requested_provider == "anthropic" and "azure.com" in _eff_base:
        _azure_key = (
            (explicit_api_key or "").strip()
            or _getenv("AZURE_ANTHROPIC_KEY", "").strip()
            or _getenv("ANTHROPIC_API_KEY", "").strip()
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

    # If provider is "auto" (or unset) but config.yaml has an explicit base_url
    # pointing at a custom/local endpoint (e.g. Ollama at localhost:11434),
    # route through the OpenAI-compatible resolver instead of letting
    # resolve_provider() pick up an ANTHROPIC_API_KEY or OPENAI_API_KEY from
    # the environment and send the request to a cloud API. Fixes #3846.
    if not explicit_base_url and not explicit_api_key:
        model_cfg = _get_model_config()
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        cfg_base_url = str(model_cfg.get("base_url") or "").strip()
        if cfg_base_url and cfg_provider in ("auto", ""):
            # Check that base_url isn't one of the well-known cloud API roots
            # (OpenRouter, Anthropic, OpenAI). If it's something else (Ollama,
            # LM Studio, vLLM, …) we honour it directly. The full detection
            # logic lives in _resolve_openrouter_runtime; we just skip the
            # resolve_provider() call so env-var credentials don't shadow it.
            # Match on HOST, not substring, so a look-alike base_url
            # (e.g. http://api.anthropic.com.attacker.test/v1, or one whose
            # path merely contains "openai.com") cannot evade the bypass and
            # leak a cloud credential. Mirrors the host-gating used for
            # API-key selection in _resolve_openrouter_runtime.
            _known_cloud_hosts = (
                "openrouter.ai",
                "anthropic.com",
                "openai.com",
            )
            if not any(
                base_url_host_matches(cfg_base_url, host)
                for host in _known_cloud_hosts
            ):
                runtime = _resolve_openrouter_runtime(
                    requested_provider=requested_provider,
                    explicit_api_key=explicit_api_key,
                    explicit_base_url=explicit_base_url,
                )
                runtime["requested_provider"] = requested_provider
                return runtime

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
        env_openai_base_url = _getenv("OPENAI_BASE_URL", "").strip()
        env_openrouter_base_url = _getenv("OPENROUTER_BASE_URL", "").strip()
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
        # For Nous, the pool entry's runtime_api_key is the agent_key
        # compatibility field. It must be an invoke JWT. The pool doesn't
        # refresh it during selection (that would trigger network calls in
        # non-runtime contexts like `hermes auth list`). If the key is
        # expired/missing, refresh the selected pool entry before falling back
        # to singleton auth resolution.
        if provider == "nous" and entry is not None:
            min_ttl = max(60, env_int("HERMES_NOUS_MIN_KEY_TTL_SECONDS", 1800))
            nous_state = {
                "agent_key": getattr(entry, "agent_key", None),
                "agent_key_expires_at": getattr(entry, "agent_key_expires_at", None),
                "scope": getattr(entry, "scope", None),
            }
            if not _agent_key_is_usable(nous_state, min_ttl):
                logger.debug("Nous pool entry agent_key expired/missing, refreshing selected pool entry")
                try:
                    refreshed = pool.try_refresh_current()
                except Exception as exc:
                    logger.debug("Nous pool entry refresh failed: %s", exc)
                    refreshed = None
                if refreshed is not None:
                    entry = refreshed
                    pool_api_key = (
                        getattr(entry, "runtime_api_key", None)
                        or getattr(entry, "access_token", "")
                    )
                    nous_state = {
                        "agent_key": getattr(entry, "agent_key", None),
                        "agent_key_expires_at": getattr(entry, "agent_key_expires_at", None),
                        "scope": getattr(entry, "scope", None),
                    }
                if not pool_api_key or not _agent_key_is_usable(nous_state, min_ttl):
                    logger.debug("Nous pool entry agent_key still unavailable, falling through to runtime resolution")
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
                timeout_seconds=float(_getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
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

    if provider == "xai-oauth":
        try:
            creds = resolve_xai_oauth_runtime_credentials()
            return {
                "provider": "xai-oauth",
                "api_mode": "codex_responses",
                "base_url": (creds.get("base_url") or "").rstrip("/") or DEFAULT_XAI_OAUTH_BASE_URL,
                "api_key": creds.get("api_key", ""),
                "source": creds.get("source", "hermes-auth-store"),
                "last_refresh": creds.get("last_refresh"),
                "requested_provider": requested_provider,
            }
        except AuthError:
            if requested_provider != "auto":
                raise
            logger.info("Auto-detected xAI OAuth provider but credentials failed; "
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
            if not _anthropic_base_url_override_ok(cfg_base_url):
                cfg_base_url = ""
        base_url = cfg_base_url or "https://api.anthropic.com"

        # For Microsoft Foundry endpoints, use ANTHROPIC_API_KEY directly —
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
                    token = _getenv(env_var, "").strip()
                    if token:
                        break
            # Next: an inline api_key on the model config (useful in multi-profile
            # setups that want to avoid env-var juggling).
            if not token:
                token = str(model_cfg.get("api_key") or "").strip()
            # Finally fall back to the historical fixed names.
            if not token:
                token = (
                    _getenv("AZURE_ANTHROPIC_KEY", "").strip()
                    or _getenv("ANTHROPIC_API_KEY", "").strip()
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
        _current_model = str(target_model or model_cfg.get("default") or "").strip()
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
        if provider == "lmstudio":
            base_url = auth_mod._normalize_lmstudio_runtime_base_url(base_url)
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
