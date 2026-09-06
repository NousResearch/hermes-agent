"""Shared runtime provider resolution for CLI, gateway, cron, and helpers: the resolution ORDER
(:func:`resolve_runtime_provider`), api_mode / base_url helpers and the pool / OAuth / explicit paths.
Custom-provider lookup lives in :mod:`hermes_cli.runtime_provider_custom`; Azure Foundry,
OpenRouter/bare-custom, Bedrock and external-process builders in
:mod:`hermes_cli.runtime_provider_backends` — both re-exported here so
``hermes_cli.runtime_provider.<name>`` imports and test patches keep working."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

from hermes_cli import auth as auth_mod
from agent.credential_pool import (  # custom_provider_pool_key_candidates is read via origin by runtime_provider_custom
    CredentialPool, PooledCredential, credential_pool_matches_provider, custom_provider_pool_key_candidates,  # noqa: F401
    load_pool,
)
from agent.secret_scope import get_secret as _get_secret
from hermes_cli.auth import (  # resolve_external_process_provider_credentials is read via origin by runtime_provider_backends
    ACTUAL_LOCAL_NOAUTH_PLACEHOLDER, AuthError, DEFAULT_CODEX_BASE_URL, DEFAULT_QWEN_BASE_URL, DEFAULT_XAI_OAUTH_BASE_URL,
    PROVIDER_REGISTRY, _agent_key_is_usable, _nous_inference_env_override, format_auth_error, resolve_provider,
    resolve_nous_runtime_credentials, resolve_codex_runtime_credentials, resolve_xai_oauth_runtime_credentials,
    resolve_qwen_runtime_credentials, resolve_api_key_provider_credentials,
    resolve_external_process_provider_credentials,  # noqa: F401
    has_usable_secret, is_actual_local_base_url, normalize_actual_base_url,
)
from hermes_cli import config as _config_mod
from hermes_cli import models as _models  # attribute access keeps ``hermes_cli.models.<name>`` patches effective
from hermes_constants import OPENROUTER_BASE_URL
from hermes_cli.providers import determine_api_mode, is_official_openai_host, nous_api_mode
from utils import base_url_host_matches, base_url_hostname, env_int


# Late-bound delegates, deliberately NOT module-level from-imports: this module is often imported
# lazily, so its first import can happen while a test has ``hermes_cli.config.load_config`` patched
# — a from-import would bind the MagicMock permanently and poison every later caller.
def load_config():
    return _config_mod.load_config()


def get_compatible_custom_providers(config=None):
    return _config_mod.get_compatible_custom_providers(config)


def normalize_extra_headers(value):
    return _config_mod.normalize_extra_headers(value)


def _getenv(name: str, default: str = "") -> str:
    """Profile-scoped ``os.getenv`` for credential/provider reads: identical to ``os.getenv`` when
    multiplexing is off; scope-aware (fail-closed on an unscoped read) when on."""
    val = _get_secret(name, default)
    return val if val is not None else default


def _loopback_hostname(host: str) -> bool:
    return (host or "").lower().rstrip(".") in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _resolves_to_custom(name: str) -> bool:
    """True when a provider alias (ollama, vllm, llamacpp, …) resolves to ``custom``."""
    try:
        return auth_mod.resolve_provider(name) == "custom"
    except Exception:
        return False


def _config_base_url_trustworthy_for_bare_custom(cfg_base_url: str, cfg_provider: str) -> bool:
    """Whether ``model.base_url`` may back bare ``custom`` runtime resolution. The picker can select
    Custom while ``model.provider`` still names a previous provider, so non-loopback URLs are rejected
    unless the YAML provider is already ``custom`` or a local-server alias (ollama/vllm/llamacpp —
    else a legit LAN ollama endpoint falls through to OpenRouter): a stale OpenRouter/Z.ai base_url
    cannot hijack local sessions.

    See #14676.
    """
    cfg_provider_norm = (cfg_provider or "").strip().lower()
    bu = (cfg_base_url or "").strip()
    return bool(bu) and (cfg_provider_norm == "custom" or _resolves_to_custom(cfg_provider_norm)
                         or (not base_url_host_matches(bu, "openrouter.ai") and _loopback_hostname(base_url_hostname(bu))))


# ── api_mode detection ─────────────────────────────────────────────────────────────────────

# Hosts that only speak one wire protocol. Mirrors host_mandated_api_mode in hermes_cli/providers.py
# so the runtime resolver stays in lockstep: api.meta.ai — prompt caching only on Responses;
# api.router.com — /v1/chat/completions is a minimal shim; api.anthropic.com — native Messages.
_HOST_MANDATED_API_MODES = {
    "api.x.ai": "codex_responses", "api.meta.ai": "codex_responses", "api.actual.inc": "codex_responses",
    "api.router.com": "codex_responses", "api.anthropic.com": "anthropic_messages",
}

# codex_app_server is opt-in: hand the whole turn to a `codex app-server` subprocess (Codex's own
# tool runtime), gated on `model.openai_runtime == "codex_app_server"` AND provider in {openai, openai-codex}.
_VALID_API_MODES = {"chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse", "codex_app_server", "agent_runtime"}


def _detect_api_mode_for_url(base_url: str) -> Optional[str]:
    """Auto-detect api_mode from the resolved base URL, or None. Exact-hostname matches reject
    lookalike subdomains (api.anthropic.com.attacker.test) and path-segment spoofing
    (proxy.test/api.anthropic.com/v1). Official OpenAI hosts (incl. us./eu. data-residency hosts)
    need Responses for GPT-5.x tool calls with reasoning.

    - Direct api.anthropic.com endpoints must use the native Messages API (``/v1/messages``). Anthropic also
    exposes an OpenAI-compat ``/chat/completions`` shim on the same host, but Pro/Max OAuth subscriptions
    are only billed against the native Messages route; hitting the shim accounts against a separate "extra
    usage" pool that is empty by default and surfaces as HTTP 400 "You're out of extra usage."  See issue
    #32243. - Third-party Anthropic-compatible gateways (MiniMax, Zhipu GLM, LiteLLM proxies, etc.)
    conventionally expose the native Anthropic protocol under a ``/anthropic`` suffix — treat those as
    ``anthropic_messages`` transport instead of the default ``chat_completions``. - Kimi Code's
    ``api.kimi.com/coding`` endpoint also speaks the Anthropic Messages protocol (the /coding route accepts
    Claude Code's native request shape).
    """
    normalized = (base_url or "").strip().lower().rstrip("/")
    hostname = base_url_hostname(base_url)
    mandated = _HOST_MANDATED_API_MODES.get(hostname) or ("codex_responses" if is_official_openai_host(base_url) else None)
    if mandated:
        return mandated
    path = urlparse(normalized).path.rstrip("/")
    if path.endswith(("/anthropic", "/anthropic/v1")) or (hostname == "api.kimi.com" and "/coding" in normalized):
        # Direct native Anthropic host: realign with providers.determine_api_mode, which already maps this
        # host to anthropic_messages. The exact-hostname match rejects lookalike subdomains
        # (api.anthropic.com.attacker.test) and path-segment spoofing (proxy.test/api.anthropic.com/v1).
        # (#32243)
        return "anthropic_messages"
    return None


def _parse_api_mode(raw: Any) -> Optional[str]:
    """Validate an api_mode from config (None if invalid). Legacy/alias spellings (``openai``,
    ``anthropic``, ``responses``, …) are canonicalized first so old configs keep their transport
    instead of silently falling through to hostname-based detection."""
    normalized = _config_mod._canonical_api_mode(raw).lower() if isinstance(raw, str) else ""
    return normalized if normalized in _VALID_API_MODES else None


def _fallback_api_mode(provider: str, base_url: str, model: str = "") -> str:
    """api_mode when no explicit/persisted mode applies: URL detection (host-mandated wire shapes)
    first, then the transport the provider overlay declares via ``providers.determine_api_mode``
    (``openai-api`` pointed at us.api.openai.com 400'd on every tool call without it), then
    ``chat_completions``."""
    return _detect_api_mode_for_url(base_url) or determine_api_mode(provider, base_url, model) or "chat_completions"


def _resolve_plain_custom_api_mode(model_cfg: Dict[str, Any], base_url: str) -> str:
    """api_mode for legacy/plain ``provider: custom`` endpoints — conservative by default: only
    direct OpenAI/xAI/Meta URLs imply Responses; named custom providers opt in via ``api_mode``."""
    configured_mode = _parse_api_mode(model_cfg.get("api_mode"))
    detected_mode = _detect_api_mode_for_url(base_url)
    if configured_mode == "codex_responses" and detected_mode != "codex_responses":
        logger.info("Ignoring persisted custom api_mode=codex_responses for non-OpenAI endpoint %s", base_url or "(unknown)")
        configured_mode = None
    return configured_mode or detected_mode or "chat_completions"


def _provider_supports_explicit_api_mode(provider: Optional[str], configured_provider: Optional[str] = None) -> bool:
    """Whether a persisted api_mode may be honored for ``provider`` — only when the config's
    provider matches (or none is recorded), so a stale mode never leaks across a switch."""
    p, c = (provider or "").strip().lower(), (configured_provider or "").strip().lower()
    return not c or (c == "custom" or c.startswith("custom:") if p == "custom" else c == p)


def _configured_api_mode(provider: str, model_cfg: Dict[str, Any]) -> Optional[str]:
    """Persisted ``model.api_mode`` when valid and recorded for this provider, else None."""
    configured_mode = _parse_api_mode(model_cfg.get("api_mode"))
    return configured_mode if configured_mode and _provider_supports_explicit_api_mode(provider, _cfg_provider(model_cfg)) else None


def _effective_model(model_cfg: Dict[str, Any], target_model: Optional[str]) -> str:
    """The caller's target model (e.g. /model switch) beats the persisted default, else api_mode
    is computed from a stale default."""
    return target_model or model_cfg.get("default") or ""


def _copilot_runtime_api_mode(model_cfg: Dict[str, Any], api_key: str, *, target_model: Optional[str] = None) -> str:
    configured_mode = _configured_api_mode("copilot", model_cfg)
    if configured_mode:
        return configured_mode
    # Use the model being resolved, not the persisted default: a Claude MoA slot inheriting
    # codex_responses from a GPT-5 default fails with "model ... does not support Responses API".
    model_name = str(_effective_model(model_cfg, target_model)).strip()
    try:
        return _models.copilot_model_api_mode(model_name, api_key=api_key) if model_name else "chat_completions"
    except Exception:
        return "chat_completions"


def _azure_inferred_api_mode(effective_model: str, api_mode: str) -> str:
    """Upgrade api_mode for GPT-5.x / codex / o1-o4 deployments on Azure Foundry (Azure 400s
    /chat/completions on these). Skipped when the user explicitly picked anthropic_messages."""
    if not effective_model or api_mode == "anthropic_messages":
        return api_mode
    try:
        return _models.azure_foundry_model_api_mode(effective_model) or api_mode
    except Exception:
        return api_mode


def _configured_or_fallback_api_mode(provider: str, model_cfg: Dict[str, Any], base_url: str, effective_model: Any, *,
                                     opencode_by_model: bool) -> str:
    """Persisted ``model.api_mode`` when it belongs to this provider, else URL/transport fallback.
    OpenCode Zen/Go serve both anthropic_messages and chat_completions models, so (when
    ``opencode_by_model``) their mode is always re-derived from the effective model."""
    if opencode_by_model and _models.opencode_provider_family(provider) is not None:
        return _models.opencode_model_api_mode(provider, effective_model)
    return _configured_api_mode(provider, model_cfg) or _fallback_api_mode(provider, base_url, effective_model)


def _api_key_provider_api_mode(provider: str, model_cfg: Dict[str, Any], api_key: str, base_url: str, effective_model: Any, *,
                               opencode_by_model: bool) -> str:
    """api_mode for a registry ``api_key`` provider (explicit and env/config paths)."""
    if provider == "copilot":
        return _copilot_runtime_api_mode(model_cfg, api_key, target_model=effective_model)
    if provider in ("xai", "actual"):
        # Ramp Router: Responses-native host — /v1/chat/completions is only a minimal compatibility shim,
        # while reasoning and caching support live on /v1/responses (docs.router.com/api/endpoint). Mirrors
        # the host_mandated_api_mode clause in hermes_cli/providers.py so the runtime resolver stays in
        # lockstep. Exact hostname per #32243.
        return "codex_responses"
    return _configured_or_fallback_api_mode(provider, model_cfg, base_url, effective_model, opencode_by_model=opencode_by_model)


def _maybe_apply_codex_app_server_runtime(*, provider: str, api_mode: str, model_cfg: Optional[Dict[str, Any]]) -> str:
    """Opt-in rewrite to "codex_app_server" via ``model.openai_runtime``; only ``openai`` /
    ``openai-codex`` are eligible. No-op when unset, "auto", or empty."""
    if model_cfg and provider in {"openai", "openai-codex"} and str(model_cfg.get("openai_runtime") or "").strip().lower() == "codex_app_server":
        return "codex_app_server"
    return api_mode


# ── base_url / credential helpers ──────────────────────────────────────────────────────────

_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
_NO_ANTHROPIC_CREDENTIALS_MSG = ("No Anthropic credentials found. Set ANTHROPIC_TOKEN or ANTHROPIC_API_KEY, "
                                 "run 'claude setup-token', or authenticate with 'claude /login'.")


def _runtime(provider: str, api_mode: str, base_url: Any, api_key: Any, **extra: Any) -> Dict[str, Any]:
    """Build a resolved-runtime dict; ``extra`` carries source/requested_provider/provider-specific keys."""
    return {"provider": provider, "api_mode": api_mode, "base_url": base_url, "api_key": api_key, **extra}


def _cfg_provider(model_cfg: Dict[str, Any]) -> str:
    return str(model_cfg.get("provider") or "").strip().lower()


def _config_base_url_for_provider(model_cfg: Dict[str, Any], provider: str) -> str:
    """``model.base_url`` (stripped, no trailing slash) only when ``model.provider`` is
    ``provider`` — a stale base_url must not leak into another provider."""
    return str(model_cfg.get("base_url") or "").strip().rstrip("/") if _cfg_provider(model_cfg) == provider else ""


def _anthropic_base_url_override_ok(base_url: str) -> bool:
    """Whether a configured ``model.base_url`` plausibly speaks the Anthropic Messages protocol:
    official Anthropic/Claude hosts, Azure Foundry, or ``/anthropic`` / Kimi ``/coding`` proxies
    (the same signal :func:`_detect_api_mode_for_url` uses). Otherwise the caller falls back to
    ``https://api.anthropic.com`` so a stale non-Anthropic URL cannot hijack native Anthropic."""
    candidate = (base_url or "").strip()
    hostname = (base_url_hostname(candidate) or "").lower() if candidate else ""
    return bool(hostname) and (hostname == "api.anthropic.com" or hostname.endswith((".anthropic.com", ".claude.com", ".azure.com"))
                               or _detect_api_mode_for_url(candidate) == "anthropic_messages")


def _anthropic_cfg_base_url(model_cfg: Dict[str, Any]) -> str:
    """Config base_url for native Anthropic, or "" when absent/untrustworthy."""
    cfg_base_url = _config_base_url_for_provider(model_cfg, "anthropic")
    return cfg_base_url if _anthropic_base_url_override_ok(cfg_base_url) else ""


def _anthropic_token_or_raise() -> str:
    from agent.anthropic_credentials import resolve_anthropic_token
    token = resolve_anthropic_token()
    if not token:
        raise AuthError(_NO_ANTHROPIC_CREDENTIALS_MSG)
    return token


def _host_derived_api_key(base_url: str) -> str:
    """``<VENDOR>_API_KEY`` from the env, vendor = registrable hostname label (``api.deepseek.com``
    → ``deepseek``). Lookalike hosts pick the ATTACKER's label (api.deepseek.com.attacker.test →
    "attacker") so DEEPSEEK_API_KEY stays put. "" for IPs/loopback/single-label hosts and for
    OPENAI/OPENROUTER/OLLAMA, which have their own host-gated paths."""
    hostname = base_url_hostname(base_url)
    if not hostname or any(ch.isdigit() for ch in hostname.split(".")[-1]) or hostname == "localhost" or ":" in hostname:
        return ""
    labels = [lbl for lbl in hostname.split(".") if lbl]
    while labels and labels[0] in ("api", "www"):
        labels.pop(0)
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in labels[-2]).upper() if len(labels) >= 2 else ""
    if not sanitized or not sanitized[0].isalpha() or sanitized in ("OPENAI", "OPENROUTER", "OLLAMA"):
        return ""
    return (_getenv(f"{sanitized}_API_KEY", "") or "").strip()


def _host_gated_env_key_candidates(base_url: str, *, ollama: bool) -> list:
    """Env API keys gated on their authoritative hosts, then the host-derived ``<VENDOR>_API_KEY``.
    Sending OPENAI/OPENROUTER/OLLAMA keys to an unrelated endpoint leaks credentials
    (GHSA-76xc-57q6-vm5m); match on HOST, not substring. ``_host_derived_api_key`` skips OLLAMA, so
    callers that want it opt in via ``ollama``."""
    is_openai = base_url_host_matches(base_url, "openai.com") or base_url_host_matches(base_url, "openai.azure.com")
    candidates = [_getenv("OLLAMA_API_KEY", "").strip() if base_url_host_matches(base_url, "ollama.com") else ""] if ollama else []
    return candidates + [_getenv("OPENAI_API_KEY", "").strip() if is_openai else "",
                         _getenv("OPENROUTER_API_KEY", "").strip() if base_url_host_matches(base_url, "openrouter.ai") else "",
                         _host_derived_api_key(base_url)]


def _pool_entry_api_key(entry: Any) -> str:
    return getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")


def _pool_entry_base_url(entry: Any) -> str:
    return getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None) or ""


def _nous_entry_key_usable(entry: Any, min_ttl: int) -> bool:
    return _agent_key_is_usable({k: getattr(entry, k, None) for k in ("agent_key", "agent_key_expires_at", "scope")}, min_ttl)


def _nous_min_key_ttl() -> int:
    return max(60, env_int("HERMES_NOUS_MIN_KEY_TTL_SECONDS", 1800))


def _resolve_nous_creds() -> Dict[str, Any]:
    return resolve_nous_runtime_credentials(timeout_seconds=float(_getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")))


def _finalize_base_url(provider: str, api_mode: str, base_url: str) -> str:
    """Shared tail for pool-entry and api-key paths: OpenCode /v1 rule (OpenCode URLs end with /v1
    for OpenAI-compatible models but the Anthropic SDK prepends its own /v1/messages — strip for
    anthropic_messages, re-append otherwise), then LM Studio normalization."""
    if _models.opencode_provider_family(provider) is not None:
        base_url = _models.normalize_opencode_base_url(provider, api_mode, base_url)
    if provider == "lmstudio":
        base_url = auth_mod._normalize_lmstudio_runtime_base_url(base_url)
    return base_url


# ── model config ───────────────────────────────────────────────────────────────────────────


def _auto_detect_local_model(base_url: str) -> str:
    """Query a local server for its model name when only one model is loaded."""
    if not base_url:
        return ""
    try:
        import requests
        url = base_url.rstrip("/")
        resp = requests.get((url if url.endswith("/v1") else url + "/v1") + "/models", timeout=(2, 3))
        if resp.ok:
            models = resp.json().get("data", [])
            if len(models) == 1 and models[0].get("id", ""):
                return models[0]["id"]
    except Exception as exc:
        logger.debug("Auto-detect model from %s failed: %s", base_url, exc)
    return ""


def _get_model_config() -> Dict[str, Any]:
    """``model`` config section with ``model`` accepted as an alias for ``default``, a dict
    ``default`` split into model/provider, and a local single-model server auto-detected."""
    config = load_config()
    model_cfg = config.get("model")
    if isinstance(model_cfg, str) and model_cfg.strip():
        return {"default": model_cfg.strip()}
    if not isinstance(model_cfg, dict):
        return {}
    cfg = dict(model_cfg)
    if not cfg.get("default") and cfg.get("model"):
        cfg["default"] = cfg["model"]
    _default = cfg.get("default")
    if isinstance(_default, dict):
        cfg_model, cfg_provider = _config_mod.split_model_config_default(_default)
        cfg_provider = cfg_provider or str(model_cfg.get("provider") or "")
        cfg["default"] = cfg_model
        if cfg_provider and not cfg.get("provider"):
            cfg["provider"] = cfg_provider
        _default = cfg_model
    base_url = (cfg.get("base_url") or "").strip()
    if not str(_default or "").strip() and base_url and base_url_hostname(base_url) in ("localhost", "127.0.0.1"):
        detected = _auto_detect_local_model(base_url)
        if detected:
            cfg["default"] = detected
    return cfg


def resolve_requested_provider(requested: Optional[str] = None) -> str:
    """Provider request from explicit arg, then config, then ``HERMES_INFERENCE_PROVIDER``, else
    "auto". Config beats the env so chat uses the endpoint the user last saved, not a stale
    shell/.env override."""
    if requested and requested.strip():
        return requested.strip().lower()
    cfg_provider = _get_model_config().get("provider")
    if isinstance(cfg_provider, str) and cfg_provider.strip():
        return cfg_provider.strip().lower()
    return _getenv("HERMES_INFERENCE_PROVIDER", "").strip().lower() or "auto"


# ── extracted collaborators (re-exported; see module docstring) ────────────────────────────

from hermes_cli.runtime_provider_custom import (  # noqa: E402,F401
    _apply_custom_provider_extras, _custom_provider_request_overrides, _filter_capabilities, _find_custom_identity,
    _get_named_custom_provider, _lift_common_custom_fields, _lift_extra_headers, _lift_max_output_tokens,
    _lift_model_capabilities, _normalize_base_url_for_match, _normalize_custom_provider_name, _resolve_named_custom_runtime,
    _try_resolve_from_custom_pool, canonical_custom_identity, find_custom_provider_identity,
    find_custom_provider_identity_by_model, has_named_custom_provider, is_routable_provider,
)
from hermes_cli.runtime_provider_backends import (  # noqa: E402,F401
    _is_external_process_provider, _resolve_azure_foundry_runtime, _resolve_bedrock_runtime,
    _resolve_external_process_runtime, _resolve_openrouter_runtime,
)


# ── credential-pool entries ────────────────────────────────────────────────────────────────

# Pool-entry providers whose api_mode is fixed: provider -> (api_mode, default base_url when the
# pool entry carries none). Callables are evaluated lazily (registry lookups). MiniMax OAuth tokens
# are valid only against the Anthropic Messages endpoint, so a stale model.api_mode from a prior
# OpenAI-compatible provider is never honoured for it (it would 404 on /chat/completions).
_POOL_ENTRY_SIMPLE_MODES: Dict[str, tuple] = {
    "openai-codex": ("codex_responses", DEFAULT_CODEX_BASE_URL), "xai-oauth": ("codex_responses", DEFAULT_XAI_OAUTH_BASE_URL),
    "qwen-oauth": ("chat_completions", DEFAULT_QWEN_BASE_URL), "openrouter": ("chat_completions", OPENROUTER_BASE_URL),
    "minimax-oauth": ("anthropic_messages", lambda: getattr(PROVIDER_REGISTRY.get("minimax-oauth"), "inference_base_url", "")),
    "xai": ("codex_responses", ""),
}


def _pool_entry_mode_and_url(provider, entry, model_cfg, effective_model, base_url) -> tuple:
    """(api_mode, base_url) for a pool entry of ``provider``."""
    if provider in _POOL_ENTRY_SIMPLE_MODES:
        api_mode, default_url = _POOL_ENTRY_SIMPLE_MODES[provider]
        return api_mode, base_url or (default_url() if callable(default_url) else default_url)
    if provider == "anthropic":
        return "anthropic_messages", _anthropic_cfg_base_url(model_cfg) or base_url or _ANTHROPIC_DEFAULT_BASE_URL
    if provider == "nous":
        return nous_api_mode(effective_model), (_nous_inference_env_override() or "") or base_url
    if provider == "copilot":
        api_mode = _copilot_runtime_api_mode(model_cfg, getattr(entry, "runtime_api_key", ""), target_model=effective_model)
        return api_mode, base_url or PROVIDER_REGISTRY["copilot"].inference_base_url
    if provider == "azure-foundry":
        api_mode = "chat_completions"
        if _cfg_provider(model_cfg) == "azure-foundry":
            base_url = _config_base_url_for_provider(model_cfg, "azure-foundry") or base_url
            api_mode = _parse_api_mode(model_cfg.get("api_mode")) or api_mode
        api_mode = _azure_inferred_api_mode(effective_model, api_mode)
        return api_mode, (re.sub(r"/v1/?$", "", base_url) if api_mode == "anthropic_messages" else base_url)
    # Honour model.base_url only when the pool entry carries no explicit base_url (i.e. it fell
    # back to the registry default). Env var overrides win.
    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig and base_url.rstrip("/") == pconfig.inference_base_url.rstrip("/"):
        base_url = _config_base_url_for_provider(model_cfg, provider) or base_url
    return _configured_or_fallback_api_mode(provider, model_cfg, base_url, effective_model, opencode_by_model=True), base_url


def _resolve_runtime_from_pool_entry(*, provider: str, entry: PooledCredential, requested_provider: str,
                                     model_cfg: Optional[Dict[str, Any]] = None, pool: Optional[CredentialPool] = None,
                                     target_model: Optional[str] = None) -> Dict[str, Any]:
    model_cfg = model_cfg or _get_model_config()
    api_mode, base_url = _pool_entry_mode_and_url(provider, entry, model_cfg, _effective_model(model_cfg, target_model),
                                                  _pool_entry_base_url(entry).rstrip("/"))
    base_url = _finalize_base_url(provider, api_mode, base_url)
    api_mode = _maybe_apply_codex_app_server_runtime(provider=provider, api_mode=api_mode, model_cfg=model_cfg)
    return _runtime(provider, api_mode, base_url, _pool_entry_api_key(entry), source=getattr(entry, "source", "pool"),
                    credential_pool=pool, requested_provider=requested_provider)


def _openrouter_should_use_pool(requested_provider, model_cfg, explicit_api_key, explicit_base_url) -> bool:
    """OpenRouter pool only for a plain openrouter/auto request with no custom endpoint or override."""
    cfg_base_url = str(model_cfg.get("base_url") or "").strip()
    env_base_urls = _getenv("OPENAI_BASE_URL", "").strip() or _getenv("OPENROUTER_BASE_URL", "").strip()
    has_custom_endpoint = bool(explicit_base_url or env_base_urls or (cfg_base_url and _cfg_provider(model_cfg) in {"auto", "custom"}))
    return requested_provider in {"openrouter", "auto"} and not has_custom_endpoint and not bool(explicit_api_key or explicit_base_url)


def _refresh_nous_pool_entry(pool: CredentialPool, entry: Any, pool_api_key: str):
    """Nous pool entries carry the agent_key (an invoke JWT) which the pool does not refresh on
    selection (avoids network calls in `hermes auth list`); refresh here before falling back to
    singleton auth resolution. Returns (entry, pool_api_key) — key "" when still unusable."""
    min_ttl = _nous_min_key_ttl()
    if _nous_entry_key_usable(entry, min_ttl):
        return entry, pool_api_key
    logger.debug("Nous pool entry agent_key expired/missing, refreshing selected pool entry")
    try:
        refreshed = pool.try_refresh_current()
    except Exception as exc:
        logger.debug("Nous pool entry refresh failed: %s", exc)
        refreshed = None
    if refreshed is not None:
        entry, pool_api_key = refreshed, _pool_entry_api_key(refreshed)
    if not pool_api_key or not _nous_entry_key_usable(entry, min_ttl):
        logger.debug("Nous pool entry agent_key still unavailable, falling through to runtime resolution")
        pool_api_key = ""
    return entry, pool_api_key


def _resolve_from_pool(provider: str, requested_provider: str, model_cfg: Dict[str, Any], explicit_api_key, explicit_base_url,
                       target_model) -> Optional[Dict[str, Any]]:
    """Runtime from the provider's credential pool, or None to continue down the ladder."""
    should_use_pool = provider != "openrouter" or _openrouter_should_use_pool(requested_provider, model_cfg, explicit_api_key,
                                                                             explicit_base_url)
    try:
        pool = load_pool(provider) if should_use_pool else None
    except Exception:
        pool = None
    if not (pool and pool.has_credentials()):
        return None
    entry = pool.select()
    if entry is None:
        return None
    pool_api_key = _pool_entry_api_key(entry)
    if provider == "nous":
        entry, pool_api_key = _refresh_nous_pool_entry(pool, entry, pool_api_key)
    if pool_api_key and credential_pool_matches_provider(pool, provider, base_url=_pool_entry_base_url(entry)):
        return _resolve_runtime_from_pool_entry(provider=provider, entry=entry, requested_provider=requested_provider,
                                                model_cfg=model_cfg, pool=pool, target_model=target_model)
    return None


# ── explicit (--api-key / --base-url) path ─────────────────────────────────────────────────


def _explicit_anthropic(requested_provider, model_cfg, api_key, base_url, target_model):
    base_url = base_url or _anthropic_cfg_base_url(model_cfg) or _ANTHROPIC_DEFAULT_BASE_URL
    api_key = api_key or _anthropic_token_or_raise()
    return _runtime("anthropic", "anthropic_messages", base_url, api_key, source="explicit", requested_provider=requested_provider)


def _creds_fallback(api_key, explicit_base_url, base_url, expiry, expiry_key, resolve):
    """When no explicit key was given, take api_key / expiry / base_url from stored credentials
    (an explicit --base-url still wins over the stored one)."""
    if api_key:
        return api_key, base_url, expiry
    creds = resolve()
    return creds.get("api_key", ""), explicit_base_url or creds.get("base_url", "").rstrip("/") or base_url, creds.get(expiry_key)


def _explicit_codex(requested_provider, model_cfg, api_key, explicit_base_url, target_model):
    api_key, base_url, last_refresh = _creds_fallback(api_key, explicit_base_url, explicit_base_url or DEFAULT_CODEX_BASE_URL,
                                                      None, "last_refresh", resolve_codex_runtime_credentials)
    return _runtime("openai-codex", "codex_responses", base_url, api_key, source="explicit", last_refresh=last_refresh,
                    requested_provider=requested_provider)


def _explicit_nous(requested_provider, model_cfg, api_key, explicit_base_url, target_model):
    state = auth_mod.get_provider_auth_state("nous") or {}
    base_url = (explicit_base_url or _nous_inference_env_override()
                or str(state.get("inference_base_url") or auth_mod.DEFAULT_NOUS_INFERENCE_URL).strip().rstrip("/"))
    # The agent_key compatibility field is used for inference only when it holds a NAS invoke JWT;
    # raw OAuth access_token fallback is handled by resolve_nous_runtime_credentials().
    api_key = api_key or (str(state.get("agent_key") or "").strip() if _agent_key_is_usable(state, _nous_min_key_ttl()) else "")
    api_key, base_url, expires_at = _creds_fallback(api_key, explicit_base_url, base_url,
                                                    state.get("agent_key_expires_at") or state.get("expires_at"), "expires_at",
                                                    _resolve_nous_creds)
    return _runtime("nous", nous_api_mode(_effective_model(model_cfg, target_model)), base_url, api_key, source="explicit",
                    expires_at=expires_at, requested_provider=requested_provider)


def _actual_local_key(provider: str, api_key: str, base_url: str) -> str:
    """Actual Computer's loopback daemon speaks a no-auth local API — substitute the placeholder key."""
    return ACTUAL_LOCAL_NOAUTH_PLACEHOLDER if provider == "actual" and not api_key and is_actual_local_base_url(base_url) else api_key


def _actual_url(provider: str, base_url: str) -> str:
    return normalize_actual_base_url(base_url) if provider == "actual" else base_url


def _explicit_api_key_provider(provider, pconfig, requested_provider, model_cfg, api_key, base_url, target_model):
    if not base_url:
        if provider in {"kimi-coding", "kimi-coding-cn"}:
            base_url = resolve_api_key_provider_credentials(provider).get("base_url", "").rstrip("/")
        else:
            env_url = _getenv(pconfig.base_url_env_var, "").strip().rstrip("/") if pconfig.base_url_env_var else ""
            base_url = env_url or pconfig.inference_base_url
    base_url = _actual_url(provider, base_url)
    if not api_key:
        creds = resolve_api_key_provider_credentials(provider)
        api_key = creds.get("api_key", "")
        if not base_url:
            base_url = _actual_url(provider, creds.get("base_url", "").rstrip("/"))
    api_mode = _api_key_provider_api_mode(provider, model_cfg, api_key, base_url, target_model or model_cfg.get("default", ""),
                                          opencode_by_model=False)
    api_key = _actual_local_key(provider, api_key, base_url)
    return _runtime(provider, api_mode, base_url.rstrip("/"), api_key, source="explicit", requested_provider=requested_provider)


# Providers with a dedicated explicit-credential builder; everything else goes through the
# registry ``api_key`` path (or None when the provider takes no explicit creds).
_EXPLICIT_RESOLVERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "anthropic": _explicit_anthropic, "openai-codex": _explicit_codex, "nous": _explicit_nous,
    "azure-foundry": lambda rq, mc, key, url, tm: _resolve_azure_foundry_runtime(requested_provider=rq, model_cfg=mc,
                                                                                 explicit_api_key=key, explicit_base_url=url),
}


def _resolve_explicit_runtime(*, provider: str, requested_provider: str, model_cfg: Dict[str, Any],
                              explicit_api_key: Optional[str] = None, explicit_base_url: Optional[str] = None,
                              target_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
    explicit_api_key = str(explicit_api_key or "").strip()
    explicit_base_url = str(explicit_base_url or "").strip().rstrip("/")
    if not explicit_api_key and not explicit_base_url:
        return None
    resolver = _EXPLICIT_RESOLVERS.get(provider)
    if resolver is not None:
        return resolver(requested_provider, model_cfg, explicit_api_key, explicit_base_url, target_model)
    pconfig = PROVIDER_REGISTRY.get(provider)
    if not (pconfig and pconfig.auth_type == "api_key"):
        return None
    return _explicit_api_key_provider(provider, pconfig, requested_provider, model_cfg, explicit_api_key, explicit_base_url, target_model)


# ── OAuth / auth-store providers ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _OAuthRuntimeSpec:
    """Env/auth-store OAuth providers resolved by a single credential call."""

    resolve: Callable[[], Dict[str, Any]]
    api_mode: Any  # str, or callable(model) -> str
    default_source: str
    expiry_key: str
    failure_msg: str
    default_base_url: str = ""


# ``resolve`` entries are late-bound lambdas so tests can monkeypatch the module-level
# ``resolve_*_runtime_credentials`` names.
_OAUTH_RUNTIME_PROVIDERS: Dict[str, _OAuthRuntimeSpec] = {
    "nous": _OAuthRuntimeSpec(_resolve_nous_creds, nous_api_mode, "portal", "expires_at",
                              "Auto-detected Nous provider but credentials failed"),
    "openai-codex": _OAuthRuntimeSpec(lambda: resolve_codex_runtime_credentials(), "codex_responses", "hermes-auth-store",
                                      "last_refresh", "Auto-detected Codex provider but credentials failed"),
    "xai-oauth": _OAuthRuntimeSpec(lambda: resolve_xai_oauth_runtime_credentials(), "codex_responses", "hermes-auth-store",
                                   "last_refresh", "Auto-detected xAI OAuth provider but credentials failed", DEFAULT_XAI_OAUTH_BASE_URL),
    "qwen-oauth": _OAuthRuntimeSpec(lambda: resolve_qwen_runtime_credentials(), "chat_completions", "qwen-cli",
                                    "expires_at_ms", "Qwen OAuth credentials failed"),
}


def _resolve_oauth_runtime(provider, requested_provider, model_cfg, target_model) -> Optional[Dict[str, Any]]:
    """Runtime from an ``_OAUTH_RUNTIME_PROVIDERS`` spec. On AuthError: re-raise for an explicit
    request; for "auto" (auto-detected but credentials stale/revoked) log and return None so the
    ladder falls through to env-var providers (e.g. OpenRouter)."""
    spec = _OAUTH_RUNTIME_PROVIDERS[provider]
    try:
        creds = spec.resolve()
    except AuthError:
        if requested_provider != "auto":
            raise
        logger.info("%s; falling through to next provider.", spec.failure_msg)
        return None
    api_mode = spec.api_mode(_effective_model(model_cfg, target_model)) if callable(spec.api_mode) else spec.api_mode
    return _runtime(provider, api_mode, (creds.get("base_url") or "").rstrip("/") or spec.default_base_url,
                    creds.get("api_key", ""), source=creds.get("source", spec.default_source),
                    **{spec.expiry_key: creds.get(spec.expiry_key)}, requested_provider=requested_provider)


def _minimax_oauth_runtime(provider, requested_provider) -> Optional[Dict[str, Any]]:
    pconfig = PROVIDER_REGISTRY.get(provider)
    if not (pconfig and pconfig.auth_type == "oauth_minimax"):
        return None
    creds = auth_mod.resolve_minimax_oauth_runtime_credentials()
    return _runtime(provider, "anthropic_messages", creds["base_url"], creds["api_key"], source=creds.get("source", "oauth"),
                    requested_provider=requested_provider)


# ── env/config paths for anthropic and registry api_key providers ──────────────────────────


def _azure_anthropic_env_key(model_cfg: Dict[str, Any]) -> str:
    """Azure Anthropic key: `key_env` / `api_key_env` hints on the model config, then an inline
    api_key (multi-profile setups), then the historical fixed names."""
    for hint_key in ("key_env", "api_key_env"):
        env_var = str(model_cfg.get(hint_key) or "").strip()
        if env_var and (token := _getenv(env_var, "").strip()):
            return token
    return (str(model_cfg.get("api_key") or "").strip() or _getenv("AZURE_ANTHROPIC_KEY", "").strip()
            or _getenv("ANTHROPIC_API_KEY", "").strip())


def _anthropic_env_runtime(requested_provider: str, model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Native Anthropic (Messages API) from env/auth store; ``model.base_url`` honoured only when
    the configured provider is anthropic (else a Codex endpoint would leak into Anthropic requests)."""
    base_url = _anthropic_cfg_base_url(model_cfg) or _ANTHROPIC_DEFAULT_BASE_URL
    # Microsoft Foundry endpoints reject Claude Code OAuth tokens, which resolve_anthropic_token()
    # would return first — use the env key directly.
    if base_url_host_matches(base_url, "azure.com"):
        token = _azure_anthropic_env_key(model_cfg)
        if not token:
            raise AuthError("No Azure Anthropic API key found. Set AZURE_ANTHROPIC_KEY or ANTHROPIC_API_KEY, or point "
                            "key_env/api_key_env in your config.yaml model section at a custom env var.")
    else:
        token = _anthropic_token_or_raise()
    return _runtime("anthropic", "anthropic_messages", base_url, token, source="env", requested_provider=requested_provider)


def _api_key_provider_runtime(provider, pconfig, requested_provider, model_cfg, target_model) -> Dict[str, Any]:
    """Registry ``api_key`` providers (z.ai/GLM, Kimi, MiniMax, copilot, …) from env/config."""
    creds = resolve_api_key_provider_credentials(provider)
    # Actual Computer: a loopback model_cfg base_url selects the daemon's no-auth local API; inject
    # the placeholder BEFORE the usable-secret gate (mirrors the env-driven path).
    if provider == "actual" and not has_usable_secret(creds.get("api_key")):
        cfg_url = _config_base_url_for_provider(model_cfg, provider)
        if is_actual_local_base_url(normalize_actual_base_url(cfg_url or creds.get("base_url", "").rstrip("/"))):
            creds = {**creds, "api_key": ACTUAL_LOCAL_NOAUTH_PLACEHOLDER, "source": creds.get("source") or "local-offline"}
    # An explicitly selected API-key provider is authoritative: an empty key would defer failure
    # to the first request and make a later fallback look like a silent provider switch.
    if not has_usable_secret(creds.get("api_key")):
        hint = f" Set {', '.join(pconfig.api_key_env_vars)}." if pconfig.api_key_env_vars else ""
        raise AuthError(f"No usable credentials found for provider '{provider}'.{hint}", provider=provider, code="missing_api_key")
    # Honour model.base_url when the configured provider matches (e.g. api.minimaxi.com China endpoint).
    base_url = _actual_url(provider, _config_base_url_for_provider(model_cfg, provider) or creds.get("base_url", "").rstrip("/"))
    api_mode = _api_key_provider_api_mode(provider, model_cfg, creds.get("api_key", ""), base_url,
                                          target_model or model_cfg.get("default", ""), opencode_by_model=True)
    base_url = _finalize_base_url(provider, api_mode, base_url)
    api_key = _actual_local_key(provider, creds.get("api_key", ""), base_url)
    return _runtime(provider, api_mode, base_url, api_key, source=creds.get("source", "env"), requested_provider=requested_provider)


# ── the resolution ladder ──────────────────────────────────────────────────────────────────

_VERTEX_NAMES = ("vertex", "google-vertex", "vertex-ai", "gcp-vertex", "vertexai")
_LOCAL_BYPASS_CLOUD_HOSTS = ("openrouter.ai", "anthropic.com", "openai.com")


def _resolve_agent_runtime_profile(
    *,
    requested_provider: str,
    model_cfg: Dict[str, Any],
    explicit_base_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve a provider profile that delegates the whole turn to a runtime.

    Provider profiles are the existing discovery surface for independently
    packaged providers. A profile declaring ``agent_runtime`` has no API
    endpoint or credential contract for Hermes to resolve; the selected
    runtime is responsible for its own authentication. Keep this check before
    the ordinary provider resolver so a missing profile credential cannot make
    a whole-turn runtime fall through to another provider.
    """
    requested_norm = str(requested_provider or "").strip().lower()
    if not requested_norm or requested_norm in {"auto", "custom"}:
        return None

    try:
        from providers import get_provider_profile

        profile = get_provider_profile(requested_norm)
    except Exception:
        return None
    if profile is None:
        return None
    if _parse_api_mode(getattr(profile, "api_mode", None)) != "agent_runtime":
        return None

    provider = str(getattr(profile, "name", "") or requested_norm).strip().lower()
    configured_provider = str(model_cfg.get("provider") or "").strip().lower()
    configured_base_url = ""
    if configured_provider in {requested_norm, provider}:
        configured_base_url = str(model_cfg.get("base_url") or "").strip()
    base_url = (
        str(explicit_base_url or "").strip()
        or configured_base_url
        or str(getattr(profile, "base_url", "") or "").strip()
    ).rstrip("/")
    if not base_url:
        # The CLI keeps a non-empty endpoint in its runtime snapshot even
        # when the selected runtime has no network endpoint. This structural
        # sentinel is never opened as a provider connection.
        base_url = f"runtime://{provider}"
    return {
        "provider": provider,
        "api_mode": "agent_runtime",
        "base_url": base_url,
        # No SDK client consumes this field for agent_runtime. Keep the
        # structural value empty rather than copying any ambient credential.
        "api_key": "",
        "source": "provider-profile",
        "requested_provider": requested_provider,
    }


def _provider_config_is_enabled(provider: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """Apply the shared explicit providers.<name>.enabled gate."""
    if config is None:
        config = _config_mod.load_config()
    providers = config.get("providers") if isinstance(config, dict) else None
    block = providers.get(provider) if isinstance(providers, dict) else None
    return not isinstance(block, dict) or _config_mod.is_provider_enabled(block)


def _raise_if_provider_disabled(requested_provider: str) -> None:
    """Honour ``providers.<name>.enabled: false`` for built-ins too (the custom lookup gate only
    covers custom blocks); a typed error lets the fallback chain advance."""
    if not _provider_config_is_enabled(requested_provider):
        raise ValueError(f"provider {requested_provider!r} is disabled in config "
                         f"(providers.{requested_provider}.enabled: false)")


def _resolve_vertex_runtime(requested_provider: str) -> Dict[str, Any]:
    """Vertex AI (OAuth2). The credential *path* (GOOGLE_APPLICATION_CREDENTIALS) must never be
    treated as a static API key; a short-lived token is minted per call, and mid-session expiry is
    recovered on 401 by run_agent._try_refresh_vertex_client_credentials()."""
    from agent.vertex_adapter import get_vertex_config
    token, base_url = get_vertex_config()
    if not token or not base_url:
        raise AuthError("Vertex AI credentials could not be resolved. Vertex uses OAuth2 (not a static API key): provide a "
                        "service-account JSON via GOOGLE_APPLICATION_CREDENTIALS (or VERTEX_CREDENTIALS_PATH) in ~/.hermes/.env, "
                        "or run 'gcloud auth application-default login' for ADC. Set the GCP project/region under vertex: in "
                        "config.yaml if they aren't embedded in the credentials. Run `hermes setup` to install Vertex support.")
    return _runtime("vertex", "chat_completions", base_url.rstrip("/"), token, source="vertex-oauth", requested_provider=requested_provider)


def _resolve_requested_shortcuts(requested_provider, explicit_api_key, explicit_base_url, target_model) -> Optional[Dict[str, Any]]:
    """Providers decided on the REQUESTED name alone, before custom / pool / generic paths."""
    if requested_provider == "moa":
        return _runtime("moa", "chat_completions", "moa://local", "moa-virtual-provider", source="moa-virtual-provider",
                        requested_provider=requested_provider)
    # Azure Anthropic short-circuit: an explicit Azure endpoint with provider="anthropic" must
    # bypass _resolve_named_custom_runtime (which would yield custom/chat_completions/no key).
    eff_base = (explicit_base_url or "").strip()
    if requested_provider == "anthropic" and base_url_host_matches(eff_base, "azure.com"):
        return _runtime("anthropic", "anthropic_messages", eff_base.rstrip("/"),
                        (explicit_api_key or "").strip() or _azure_anthropic_env_key({}), source="azure-explicit",
                        requested_provider=requested_provider)
    # Azure Foundry resolves before the custom-runtime / pool / generic paths so its config is
    # always picked up from model.base_url + model.api_mode, with or without explicit_* args.
    if requested_provider == "azure-foundry":
        return _resolve_azure_foundry_runtime(requested_provider=requested_provider, model_cfg=_get_model_config(),
                                              explicit_api_key=explicit_api_key, explicit_base_url=explicit_base_url,
                                              target_model=target_model)
    if requested_provider in _VERTEX_NAMES:
        return _resolve_vertex_runtime(requested_provider)
    return None


def _local_endpoint_bypass(requested_provider: str, explicit_api_key, explicit_base_url) -> Optional[Dict[str, Any]]:
    """provider "auto"/unset with a config base_url at a custom/local endpoint routes through the
    OpenAI-compatible resolver, so resolve_provider() cannot pick up an env ANTHROPIC/OPENAI key
    and send the request to a cloud API. Only non-cloud roots take the bypass; match on HOST, not
    substring, so a look-alike (api.anthropic.com.attacker.test) cannot leak a cloud credential."""
    model_cfg = _get_model_config()
    cfg_base_url = str(model_cfg.get("base_url") or "").strip()
    if (not cfg_base_url or _cfg_provider(model_cfg) not in ("auto", "")
            or any(base_url_host_matches(cfg_base_url, host) for host in _LOCAL_BYPASS_CLOUD_HOSTS)):
        return None
    return _openrouter_fallback(requested_provider, explicit_api_key, explicit_base_url)


def _tag(runtime: Optional[Dict[str, Any]], requested_provider: str) -> Optional[Dict[str, Any]]:
    """Stamp ``requested_provider`` on a runtime built by a collaborator that does not set it."""
    if runtime:
        runtime["requested_provider"] = requested_provider
    return runtime


def _openrouter_fallback(requested_provider, explicit_api_key, explicit_base_url) -> Dict[str, Any]:
    return _tag(_resolve_openrouter_runtime(requested_provider=requested_provider, explicit_api_key=explicit_api_key,
                                            explicit_base_url=explicit_base_url), requested_provider)


def _opencode_free_runtime(provider, requested_provider, model_cfg, target_model) -> Optional[Dict[str, Any]]:
    """OpenCode Zen free tier (*-free slugs) is served ANONYMOUSLY on the Zen relay only: unknown
    bearers 401 and the Go relay rejects free models, so free slugs route through the keyless Zen
    runtime BEFORE the pool / explicit / api_key paths."""
    if _models.opencode_provider_family(provider) is None:
        return None
    model = str(target_model or model_cfg.get("default") or model_cfg.get("model") or "").strip()
    return _tag(_models.opencode_zen_free_runtime(provider, model), requested_provider)


def resolve_runtime_provider(*, requested: Optional[str] = None, explicit_api_key: Optional[str] = None,
                             explicit_base_url: Optional[str] = None, target_model: Optional[str] = None) -> Dict[str, Any]:
    """Resolve runtime provider credentials for agent execution. Ladder (order is behavior — each
    rung returns or raises, else falls to the next):
      1. disabled-provider guard (``providers.<name>.enabled: false``)
      2. requested-name shortcuts: moa, anthropic@azure, azure-foundry, vertex
      3. named custom provider / llamacpp alias / bare-custom direct alias
      4. local-endpoint bypass (no explicit creds, config base_url at a non-cloud host)
      5. ``auth.resolve_provider`` → OpenCode free tier → explicit --api-key/--base-url path
      6. credential pool (OpenRouter pool only without custom endpoint/override)
      7. OAuth specs (nous/codex/xai/qwen; "auto" swallows AuthError and logs) → minimax-oauth
         → external-process → anthropic env → bedrock → registry api_key providers
      8. OpenRouter / bare-custom fallback
    target_model overrides model_cfg["default"] when computing provider-specific api_mode (e.g.
    OpenCode Zen/Go where different models route through different API surfaces)."""
    requested_provider = resolve_requested_provider(requested)
    _raise_if_provider_disabled(requested_provider)
    profile_runtime = _resolve_agent_runtime_profile(
        requested_provider=requested_provider, model_cfg=_get_model_config(),
        explicit_base_url=explicit_base_url,
    )
    if profile_runtime is not None:
        return profile_runtime
    return next(r for r in _ladder_rungs(requested_provider, explicit_api_key, explicit_base_url, target_model) if r)


def _ladder_rungs(requested_provider, explicit_api_key, explicit_base_url, target_model):
    """Ladder rungs 2-8, yielded lazily so each is evaluated only when the previous one returned
    nothing; the last rung (OpenRouter / bare-custom fallback) always yields a runtime."""
    yield _resolve_requested_shortcuts(requested_provider, explicit_api_key, explicit_base_url, target_model)
    yield _tag(_resolve_named_custom_runtime(requested_provider=requested_provider, explicit_api_key=explicit_api_key,
                                             explicit_base_url=explicit_base_url, target_model=target_model), requested_provider)
    # If provider is "auto" (or unset) but config.yaml has an explicit base_url pointing at a custom/local
    # endpoint (e.g. Ollama at localhost:11434), route through the OpenAI-compatible resolver instead of
    # letting resolve_provider() pick up an ANTHROPIC_API_KEY or OPENAI_API_KEY from the environment and
    # send the request to a cloud API. Fixes #3846.
    if not explicit_base_url and not explicit_api_key:
        yield _local_endpoint_bypass(requested_provider, explicit_api_key, explicit_base_url)
    provider = resolve_provider(requested_provider, explicit_api_key=explicit_api_key, explicit_base_url=explicit_base_url)
    model_cfg = _get_model_config()
    yield _opencode_free_runtime(provider, requested_provider, model_cfg, target_model)
    yield _resolve_explicit_runtime(provider=provider, requested_provider=requested_provider, model_cfg=model_cfg,
                                    explicit_api_key=explicit_api_key, explicit_base_url=explicit_base_url,
                                    target_model=target_model)
    yield _resolve_from_pool(provider, requested_provider, model_cfg, explicit_api_key, explicit_base_url, target_model)
    if provider in _OAUTH_RUNTIME_PROVIDERS:
        yield _resolve_oauth_runtime(provider, requested_provider, model_cfg, target_model)
    if provider == "minimax-oauth":
        yield _minimax_oauth_runtime(provider, requested_provider)
    if _is_external_process_provider(provider):
        yield _resolve_external_process_runtime(provider, requested_provider)
    if provider == "anthropic":
        yield _anthropic_env_runtime(requested_provider, model_cfg)
    if provider == "bedrock":
        yield _resolve_bedrock_runtime(requested_provider, model_cfg, target_model)
    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig and pconfig.auth_type == "api_key":
        yield _api_key_provider_runtime(provider, pconfig, requested_provider, model_cfg, target_model)
    yield _openrouter_fallback(requested_provider, explicit_api_key, explicit_base_url)


def format_runtime_provider_error(error: Exception) -> str:
    return format_auth_error(error) if isinstance(error, AuthError) else str(error)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'custom_provider_aliases': ('hermes_cli.providers', 'custom_provider_aliases'),
    'custom_provider_slug': ('hermes_cli.providers', 'custom_provider_slug'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
