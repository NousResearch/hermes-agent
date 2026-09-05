"""Multi-provider authentication system for Hermes Agent.

- ``ProviderConfig`` / ``PROVIDER_REGISTRY`` describe every known inference provider.
- The auth store (``~/.hermes/auth.json``) holds per-provider state, the credential pool and
  suppression markers; ``_auth_store_lock`` / ``_load_auth_store`` / ``_save_auth_store`` are the
  only I/O primitives (cross-process flock, atomic 0o600 writes).
- ``resolve_provider()`` picks the active provider via the documented priority chain.
- ``OAUTH_PROVIDER_FLOWS`` maps each OAuth provider to its resolver/status builder; the flows live in
  ``auth_nous``/``auth_codex``/``auth_xai``/``auth_qwen``/``auth_minimax``/``auth_spotify`` and are
  re-imported here so ``hermes_cli.auth.<name>`` stays the public/patchable surface."""

from __future__ import annotations

import json
import logging
import os
import shutil
import shlex
import stat
import threading
import time
import uuid
import webbrowser  # noqa: F401  (tests patch auth_mod.webbrowser.open; same module object)

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import partial
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from hermes_cli.config import (
    get_hermes_home, get_config_path, read_raw_config, require_readable_config_before_write)
from hermes_constants import OPENROUTER_BASE_URL, secure_parent_dir
from agent.credential_persistence import sanitize_borrowed_credential_payload
from utils import atomic_replace, atomic_yaml_write, env_float, is_truthy_value  # noqa: F401  (env_float: agent.credential_pool reads auth_mod.env_float)
from hermes_cli.auth_zai_kimi import (  # noqa: F401  re-exported
    KIMI_CODE_BASE_URL, ZAI_ENDPOINTS, _normalize_lmstudio_runtime_base_url, _resolve_kimi_base_url,
    _resolve_zai_base_url, detect_zai_endpoint)
from hermes_cli.auth_model_picker import (  # noqa: F401  re-exported
    _prompt_model_selection, _save_model_choice)
from hermes_cli.auth_device_flow import (  # noqa: F401  re-exported
    _can_open_graphical_browser, _default_verify, _is_remote_session,
    _nous_device_auth_timeout_message, _offer_existing_oauth_credentials,
    _poll_device_token_generic, _poll_for_token, _print_device_code_instructions,
    _print_login_success, _print_loopback_ssh_hint, _prompt_yes_no, _request_device_code,
    _resolve_verify, _ssh_user_at_host)
from hermes_cli.auth_oauth_grants import (  # noqa: F401  re-exported
    SINGLE_USE_REFRESH_POOL_PROVIDERS, _oauth_heal_clean_marks, _oauth_heal_notices,
    consume_oauth_heal_notices, heal_forked_single_use_oauth_grants,
    strip_cloned_single_use_oauth_grants)
from hermes_cli.auth_nous import (  # noqa: F401  re-exported
    NOUS_SESSION_TERMINAL, NOUS_SESSION_UNKNOWN, NOUS_SESSION_VALID, _ALLOWED_NOUS_INFERENCE_HOSTS,
    _agent_key_is_usable, _apply_nous_refreshed_tokens, _assert_nous_inference_jwt_usable,
    _compute_nous_auth_status, _format_nous_entitlement_auth_error, _healed_nous_inference_url,
    _login_nous, _merge_shared_nous_oauth_state, _migrate_stale_nous_portal_url,
    _nous_device_code_login, _nous_inference_env_override, _nous_invoke_jwt_is_usable,
    _nous_invoke_jwt_status, _nous_portal_env_override, _nous_shared_store_lock,
    _nous_shared_store_path, _pool_first_oauth_status, _quarantine_nous_oauth_state,
    _quarantine_nous_pool_entries, _read_shared_nous_state, _refresh_access_token,
    _refresh_nous_or_quarantine, _select_nous_invoke_jwt, _sync_nous_pool_from_auth_store,
    _token_fingerprint, _try_import_shared_nous_state, _validate_nous_inference_url_from_network,
    _write_shared_nous_state, fetch_nous_models, get_nous_auth_status_local,
    get_nous_session_validity, persist_nous_credentials, refresh_nous_oauth_from_state,
    resolve_nous_runtime_credentials, step_up_nous_billing_scope)
from hermes_cli.auth_minimax import (  # noqa: F401  re-exported
    _MINIMAX_OAUTH_ERROR_BODY_LIMIT, _login_minimax_oauth, _minimax_oauth_login, _minimax_pkce_pair,
    _minimax_poll_token, _minimax_post_form, _minimax_request_user_code,
    _minimax_resolve_token_expiry_unix, _minimax_response_error_text, _minimax_save_auth_state,
    _refresh_minimax_oauth_state, build_minimax_oauth_token_provider,
    resolve_minimax_oauth_runtime_credentials)
from hermes_cli.auth_xai import (  # noqa: F401  re-exported
    _login_xai_oauth, _read_xai_oauth_tokens, _refresh_xai_oauth_tokens, _save_xai_oauth_tokens,
    _write_through_xai_oauth_to_global_root, _xai_access_token_is_expiring,
    _xai_oauth_device_code_login, _xai_oauth_discovery, _xai_oauth_poll_device_token,
    _xai_oauth_request_device_code, _xai_proactive_refresh_skew_seconds,
    _xai_validate_inference_base_url, refresh_xai_oauth_pure, resolve_xai_oauth_runtime_credentials)
from hermes_cli.auth_codex import (  # noqa: F401  re-exported
    _codex_access_token_is_expiring, _codex_device_code_login, _codex_http_client,
    _codex_pool_rate_limit_status, _codex_quota_probe_cache, _codex_usage_probe_url,
    _import_codex_cli_tokens, _is_codex_rate_limit_shaped, _login_openai_codex,
    _probe_codex_quota_restored, _read_codex_tokens, _refresh_codex_auth_tokens, _save_codex_tokens,
    clear_codex_pool_quota_cooldowns, refresh_codex_oauth_pure, resolve_codex_runtime_credentials)
from hermes_cli.auth_spotify import (  # noqa: F401  re-exported
    _refresh_spotify_oauth_state, get_spotify_auth_status, login_spotify_command,
    resolve_spotify_runtime_credentials)
from hermes_cli.auth_qwen import (  # noqa: F401  re-exported
    _qwen_access_token_is_expiring, _qwen_cli_auth_path, _read_qwen_cli_tokens,
    _refresh_qwen_cli_tokens, _save_qwen_cli_tokens, get_qwen_auth_status,
    resolve_qwen_runtime_credentials)
from hermes_cli.auth_claude_sdk import get_claude_agent_sdk_auth_status  # noqa: F401  re-exported
from hermes_cli.auth_constants import (  # noqa: F401  re-exported
    _decode_jwt_claims, AUTH_STORE_VERSION, AUTH_LOCK_TIMEOUT_SECONDS, DEFAULT_NOUS_PORTAL_URL,
    DEFAULT_NOUS_INFERENCE_URL, DEFAULT_NOUS_CLIENT_ID, NOUS_BILLING_MANAGE_SCOPE,
    DEFAULT_NOUS_SCOPE, NOUS_DEVICE_CODE_SOURCE, NOUS_AUTH_PATH_INVOKE_JWT,
    ACCESS_TOKEN_REFRESH_SKEW_SECONDS, NOUS_INVOKE_JWT_MIN_TTL_SECONDS, DEFAULT_CODEX_BASE_URL,
    DEFAULT_XAI_OAUTH_BASE_URL, MINIMAX_OAUTH_CLIENT_ID, MINIMAX_OAUTH_SCOPE,
    MINIMAX_OAUTH_GLOBAL_BASE, MINIMAX_OAUTH_CN_BASE, MINIMAX_OAUTH_GLOBAL_INFERENCE,
    MINIMAX_OAUTH_CN_INFERENCE, MINIMAX_OAUTH_REFRESH_SKEW_SECONDS, DEFAULT_QWEN_BASE_URL,
    DEFAULT_GITHUB_MODELS_BASE_URL, DEFAULT_COPILOT_ACP_BASE_URL, DEFAULT_OLLAMA_CLOUD_BASE_URL,
    DEFAULT_ACTUAL_BASE_URL, DEFAULT_ACTUAL_LOCAL_BASE_URL, STEPFUN_STEP_PLAN_INTL_BASE_URL,
    STEPFUN_STEP_PLAN_CN_BASE_URL, CODEX_OAUTH_CLIENT_ID, CODEX_OAUTH_TOKEN_URL,
    CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS, XAI_OAUTH_CLIENT_ID, XAI_OAUTH_SCOPE,
    XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS, QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL, DEFAULT_SPOTIFY_API_BASE_URL, SPOTIFY_DOCS_URL,
    DEFAULT_SPOTIFY_SCOPE, SERVICE_PROVIDER_NAMES, LMSTUDIO_NOAUTH_PLACEHOLDER,
    ACTUAL_LOCAL_NOAUTH_PLACEHOLDER, CODEX_RATE_LIMITED_CODE, AuthError, _nous_err, httpx)

logger = logging.getLogger(__name__)

try:
    import fcntl
except Exception:
    fcntl = None
try:
    import msvcrt
except Exception:
    msvcrt = None

def is_actual_local_base_url(base_url: str) -> bool:
    """Return True for Actual's loopback local API endpoint."""
    try:
        host = (urlparse(base_url or "").hostname or "").lower().rstrip(".")
    except Exception:
        return False
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def normalize_actual_base_url(base_url: str) -> str:
    """Return Actual's OpenAI-compatible base URL (hosted api.actual.inc or the loopback local server;
    both expose a /v1 surface for the Responses transport)."""
    url = str(base_url or "").strip().rstrip("/")
    if not url:
        return DEFAULT_ACTUAL_BASE_URL
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        path = parsed.path.rstrip("/")
    except Exception:
        return url
    if path in {"", "/"} and (host == "api.actual.inc" or is_actual_local_base_url(url)):
        return url + "/v1"
    return url


# ── Provider Registry ───────────────────────────────────────────────────────────────────────────────

@dataclass
class ProviderConfig:
    """Describes a known inference provider."""
    id: str
    name: str
    auth_type: str  # "oauth_device_code", "oauth_external", "oauth_minimax", "api_key", ...
    portal_base_url: str = ""
    inference_base_url: str = ""
    client_id: str = ""
    scope: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    api_key_env_vars: tuple = ()  # API-key providers: env vars to check, in priority order
    base_url_env_var: str = ""  # optional env var overriding the base URL


def _api_key_provider(
    id: str, name: str, inference_base_url: str, api_key_env_vars: tuple,
    base_url_env_var: str = "", auth_type: str = "api_key") -> ProviderConfig:
    """Compact constructor for the common env-var-keyed provider shape."""
    return ProviderConfig(
        id=id, name=name, auth_type=auth_type, inference_base_url=inference_base_url,
        api_key_env_vars=api_key_env_vars, base_url_env_var=base_url_env_var)


# Registry rows in priority order (resolve_provider() scans api_key rows in this order). A tuple
# row is ``_api_key_provider(id, name, inference_base_url, api_key_env_vars[, base_url_env_var
# [, auth_type]])``; OAuth / bespoke rows are full ``ProviderConfig`` objects.
_REGISTRY_ROWS: Tuple[Any, ...] = (
    ProviderConfig(
        "nous", "Nous Portal", "oauth_device_code", portal_base_url=DEFAULT_NOUS_PORTAL_URL,
        inference_base_url=DEFAULT_NOUS_INFERENCE_URL, client_id=DEFAULT_NOUS_CLIENT_ID,
        scope=DEFAULT_NOUS_SCOPE),
    ProviderConfig("openai-codex", "OpenAI Codex", "oauth_external", inference_base_url=DEFAULT_CODEX_BASE_URL),
    ("openai-api", "OpenAI API", "https://api.openai.com/v1", ("OPENAI_API_KEY",), "OPENAI_BASE_URL"),
    ProviderConfig(
        "xai-oauth", "xAI Grok OAuth (SuperGrok / Premium+)", "oauth_external",
        inference_base_url=DEFAULT_XAI_OAUTH_BASE_URL),
    ProviderConfig("qwen-oauth", "Qwen OAuth", "oauth_external", inference_base_url=DEFAULT_QWEN_BASE_URL),
    ("lmstudio", "LM Studio", "http://127.0.0.1:1234/v1", ("LM_API_KEY",), "LM_BASE_URL"),
    ("copilot", "GitHub Copilot", DEFAULT_GITHUB_MODELS_BASE_URL,
     ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"), "COPILOT_API_BASE_URL"),
    ProviderConfig(
        "copilot-acp", "GitHub Copilot ACP", "external_process",
        inference_base_url=DEFAULT_COPILOT_ACP_BASE_URL, base_url_env_var="COPILOT_ACP_BASE_URL"),
    ("gemini", "Google AI Studio", "https://generativelanguage.googleapis.com/v1beta",
     ("GOOGLE_API_KEY", "GEMINI_API_KEY"), "GEMINI_BASE_URL"),
    ("zai", "Z.AI / GLM", "https://api.z.ai/api/paas/v4",
     ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"), "GLM_BASE_URL"),
    # Legacy platform.moonshot.ai keys use this endpoint (OpenAI-compat); sk-kimi- (Kimi Code)
    # keys are auto-redirected to api.kimi.com/coding by _resolve_kimi_base_url().
    ("kimi-coding", "Kimi / Moonshot", "https://api.moonshot.ai/v1",
     ("KIMI_API_KEY", "KIMI_CODING_API_KEY"), "KIMI_BASE_URL"),
    ("kimi-coding-cn", "Kimi / Moonshot (China)", "https://api.moonshot.cn/v1", ("KIMI_CN_API_KEY",)),
    ("stepfun", "StepFun Step Plan", STEPFUN_STEP_PLAN_INTL_BASE_URL, ("STEPFUN_API_KEY",), "STEPFUN_BASE_URL"),
    ("arcee", "Arcee AI", "https://api.arcee.ai/api/v1", ("ARCEEAI_API_KEY",), "ARCEE_BASE_URL"),
    ("gmi", "GMI Cloud", "https://api.gmi-serving.com/v1", ("GMI_API_KEY",), "GMI_BASE_URL"),
    ("actual", "Actual Computer", DEFAULT_ACTUAL_BASE_URL, ("ACTUAL_API_KEY",), "ACTUAL_BASE_URL"),
    ("minimax", "MiniMax", "https://api.minimax.io/anthropic", ("MINIMAX_API_KEY",), "MINIMAX_BASE_URL"),
    ProviderConfig(
        "minimax-oauth", "MiniMax (OAuth \u00b7 minimax.io)", "oauth_minimax",
        portal_base_url=MINIMAX_OAUTH_GLOBAL_BASE, inference_base_url=MINIMAX_OAUTH_GLOBAL_INFERENCE,
        client_id=MINIMAX_OAUTH_CLIENT_ID, scope=MINIMAX_OAUTH_SCOPE,
        extra={"region": "global", "cn_portal_base_url": MINIMAX_OAUTH_CN_BASE,
               "cn_inference_base_url": MINIMAX_OAUTH_CN_INFERENCE}),
    # CLAUDE_CODE_OAUTH_TOKEN is NOT an API key despite auth_type="api_key": `claude setup-token`
    # yields an `sk-ant-oat01…` OAuth token (401s as x-api-key, 429s as bare Bearer). It stays in
    # this tuple because the tuple doubles as the credential-DISCOVERY list
    # (agent/credential_pool.py builds its env scan from it); the adapter routes it down the OAuth
    # path by prefix. Only ANTHROPIC_API_KEY and ANTHROPIC_TOKEN are usable as literal API keys.
    ("anthropic", "Anthropic", "https://api.anthropic.com",
     ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"), "ANTHROPIC_BASE_URL"),
    ("alibaba", "Qwen Cloud", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
     ("DASHSCOPE_API_KEY",), "DASHSCOPE_BASE_URL"),
    ("alibaba-coding-plan", "Alibaba Cloud (Coding Plan)", "https://coding-intl.dashscope.aliyuncs.com/v1",
     ("ALIBABA_CODING_PLAN_API_KEY", "DASHSCOPE_API_KEY"), "ALIBABA_CODING_PLAN_BASE_URL"),
    ("minimax-cn", "MiniMax (China)", "https://api.minimaxi.com/anthropic", ("MINIMAX_CN_API_KEY",),
     "MINIMAX_CN_BASE_URL"),
    ("deepseek", "DeepSeek", "https://api.deepseek.com/v1", ("DEEPSEEK_API_KEY",), "DEEPSEEK_BASE_URL"),
    ("xai", "xAI", "https://api.x.ai/v1", ("XAI_API_KEY",), "XAI_BASE_URL"),
    ("nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1", ("NVIDIA_API_KEY",), "NVIDIA_BASE_URL"),
    ("ai-gateway", "Vercel AI Gateway", "https://ai-gateway.vercel.sh/v1", ("AI_GATEWAY_API_KEY",),
     "AI_GATEWAY_BASE_URL"),
    ("opencode-zen", "OpenCode Zen", "https://opencode.ai/zen/v1", ("OPENCODE_ZEN_API_KEY",),
     "OPENCODE_ZEN_BASE_URL"),
    # OpenCode Go mixes API surfaces by model (GLM/Kimi: OpenAI chat under /v1; MiniMax and
    # Qwen 3.7: Anthropic Messages under /v1/messages). Keep the base at /v1; api_mode is per-model.
    ("opencode-go", "OpenCode Go", "https://opencode.ai/zen/go/v1", ("OPENCODE_GO_API_KEY",),
     "OPENCODE_GO_BASE_URL"),
    # Deliberately NO api_key_env_vars: the free tier is served anonymously (any unrecognized bearer
    # is a 401), so there is no secret to configure. Select via `hermes model` / `/model free`.
    ("opencode-free", "OpenCode Free", "https://opencode.ai/zen/v1", ()),
    ("kilocode", "Kilo Code", "https://api.kilo.ai/api/gateway", ("KILOCODE_API_KEY",), "KILOCODE_BASE_URL"),
    ("huggingface", "Hugging Face", "https://router.huggingface.co/v1", ("HF_TOKEN",), "HF_BASE_URL"),
    ("xiaomi", "Xiaomi MiMo", "https://api.xiaomimimo.com/v1", ("XIAOMI_API_KEY",), "XIAOMI_BASE_URL"),
    ("tencent-tokenhub", "Tencent TokenHub", "https://tokenhub.tencentmaas.com/v1", ("TOKENHUB_API_KEY",),
     "TOKENHUB_BASE_URL"),
    ("tencent-tokenplan", "Tencent TokenPlan", "https://api.lkeap.cloud.tencent.com/plan/anthropic",
     ("TOKENPLAN_API_KEY",), "TOKENPLAN_BASE_URL"),
    ("ollama-cloud", "Ollama Cloud", DEFAULT_OLLAMA_CLOUD_BASE_URL, ("OLLAMA_API_KEY",), "OLLAMA_BASE_URL"),
    ("bedrock", "AWS Bedrock", "https://bedrock-runtime.us-east-1.amazonaws.com", (), "BEDROCK_BASE_URL",
     "aws_sdk"),
    # No static inference_base_url: Vertex's endpoint is computed per request from project_id +
    # region (agent/vertex_adapter.py build_vertex_base_url), not a fixed host.
    ("vertex", "Google Vertex AI", "", (), "", "vertex"),
    ("azure-foundry", "Azure Foundry", "", ("AZURE_FOUNDRY_API_KEY",), "AZURE_FOUNDRY_BASE_URL"))
PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {
    p.id: p for p in (r if isinstance(r, ProviderConfig) else _api_key_provider(*r) for r in _REGISTRY_ROWS)
}

# Providers handled outside the registry: copilot/kimi/zai have bespoke token refresh here;
# openrouter/custom are aggregator/user-supplied and runtime_provider relies on
# ``openrouter not in PROVIDER_REGISTRY``.
_REGISTRY_PLUGIN_SKIP = frozenset({"copilot", "kimi-coding", "kimi-coding-cn", "zai", "openrouter", "custom"})


def _register_plugin_provider(pp: Any) -> None:
    """Auto-register one providers/ profile (plugins/model-providers/<name>/) not declared above.

    External-process (ACP) providers have no API-key env vars; registering them is what lets an
    out-of-tree provider pass ``resolve_provider()``'s known-provider gate ("Unknown provider")."""
    if pp.auth_type == "external_process":
        pconfig = ProviderConfig(
            pp.name, pp.display_name or pp.name, "external_process", inference_base_url=pp.base_url)
    elif pp.auth_type == "api_key" and pp.env_vars and pp.name not in _REGISTRY_PLUGIN_SKIP:
        is_url = lambda v: v.endswith("_BASE_URL") or v.endswith("_URL")  # noqa: E731
        pconfig = _api_key_provider(
            pp.name, pp.display_name or pp.name, pp.base_url,
            tuple(v for v in pp.env_vars if not is_url(v)) or pp.env_vars,
            next((v for v in pp.env_vars if is_url(v)), None) or "")
    else:
        return
    PROVIDER_REGISTRY[pp.name] = pconfig
    for alias in pp.aliases:  # so resolve_provider() resolves them too
        PROVIDER_REGISTRY.setdefault(alias, pconfig)


try:
    from providers import list_providers as _list_providers_for_registry
    for _pp in _list_providers_for_registry():
        if _pp.name not in PROVIDER_REGISTRY:
            _register_plugin_provider(_pp)
except Exception:
    pass


def get_anthropic_key() -> str:
    """First usable Anthropic credential (``.env`` preferred over a stale shell export), or ``""``.

    Order mirrors ``PROVIDER_REGISTRY["anthropic"].api_key_env_vars``.

    Checks both the ``.env`` file and the process environment, preferring ``~/.hermes/.env`` so a deliberate
    key rotation isn't shadowed by a stale shell export (matches the api-key resolution path — see #20591).
    """
    from hermes_cli.config import get_env_value_prefer_dotenv
    env_vars = PROVIDER_REGISTRY["anthropic"].api_key_env_vars
    return next((v for v in (get_env_value_prefer_dotenv(var) or "" for var in env_vars) if v), "")


# ── Secret validation ───────────────────────────────────────────────────────────────────────────────

_PLACEHOLDER_SECRET_VALUES = {
    "*", "**", "***", "changeme", "your_api_key", "your_api_key_here", "your-api-key",
    "placeholder", "example", "dummy", "null", "none"}


def has_usable_secret(value: Any, *, min_length: int = 4) -> bool:
    """Return True when a configured secret looks usable, not empty/placeholder."""
    if not isinstance(value, str):
        return False
    cleaned = value.strip()
    return len(cleaned) >= min_length and cleaned.lower() not in _PLACEHOLDER_SECRET_VALUES


# Known API-key prefixes per provider. Only listed providers get prefix validation; everyone else
# is fail-open. Keeps an obviously malformed key in .env (truncated paste, wrong provider's key)
# from silently shadowing a valid credential-pool entry and producing opaque 401s.
# See #93593.
KNOWN_PROVIDER_KEY_PREFIXES: Dict[str, tuple] = {
    "openrouter": ("sk-or-",),  # all OpenRouter keys are sk-or-... (currently sk-or-v1-)
}


def _usable_declared_secret(provider_id: str, value: Any, source: str) -> Optional[str]:
    """*value* stripped when it is a usable, prefix-valid secret; None (after warning on a provable
    prefix mismatch, so it never shadows a later credential source) otherwise. Providers without a
    declared prefix are fail-open."""
    val = str(value or "").strip()
    if not has_usable_secret(val):
        return None
    prefixes = KNOWN_PROVIDER_KEY_PREFIXES.get(provider_id)
    if prefixes and not any(val.startswith(p) for p in prefixes):
        logger.warning(
            "Ignoring %s for provider %r: value does not match the expected key "
            "prefix (%s). Falling back to the next credential source. Fix or "
            "remove the malformed key to silence this warning.",
            source, provider_id, " or ".join(prefixes))
        return None
    return val


def _resolve_api_key_provider_secret(provider_id: str, pconfig: ProviderConfig) -> tuple[str, str]:
    """Resolve an API-key provider's token and indicate where it came from."""
    if provider_id == "copilot":
        # The dedicated copilot auth module does proper token validation/exchange.
        try:
            from hermes_cli.copilot_auth import resolve_copilot_token, get_copilot_api_token
            token, source = resolve_copilot_token()
            if token:
                api_token, _base_url = get_copilot_api_token(token)
                return api_token, source
        except ValueError as exc:
            logger.warning("Copilot token validation failed: %s", exc)
        except Exception:
            pass
        return "", ""

    # Prefer ~/.hermes/.env over os.environ so a deliberate key rotation in .env isn't shadowed by
    # a stale shell export inherited from a parent process (Codex CLI, test runners, etc.).
    from hermes_cli.config import get_env_value_prefer_dotenv
    for env_var in pconfig.api_key_env_vars:
        val = _usable_declared_secret(provider_id, get_env_value_prefer_dotenv(env_var), env_var)
        if val:
            # A provably malformed key (declared prefix mismatch) must not shadow a valid credential-pool
            # entry (#93593). Warn and keep looking instead of returning it.
            return val, env_var

    # Fallback: credential pool (e.g. zai key stored via auth.json). Prefer the pool's own
    # selection (peek) but try the rest too so one malformed entry doesn't block a valid one.
    pool_source = f"credential_pool:{provider_id}"
    try:
        from agent.credential_pool import load_pool
        pool = load_pool(provider_id)
        if pool and pool.has_credentials():
            entry = pool.peek()
            candidates = [entry] if entry is not None else []
            try:
                for extra in pool.entries():
                    if extra is not None and all(extra is not c for c in candidates):
                        candidates.append(extra)
            except Exception:
                pass
            for entry in candidates:
                key = getattr(entry, "access_token", "") or getattr(entry, "runtime_api_key", "")
                val = _usable_declared_secret(provider_id, key, pool_source)
                if val:
                    return val, pool_source
    except Exception:
        pass
    return "", ""


# ── Error formatting (AuthError itself lives in auth_constants) ─────────────────────────────────────

def is_rate_limited_auth_error(error: Exception) -> bool:
    """True when an :class:`AuthError` is upstream rate-limiting / quota: transient, and
    re-authenticating cannot fix it, so callers should say "retry later", not ``hermes auth``."""
    return (isinstance(error, AuthError) and not error.relogin_required
            and error.code == CODEX_RATE_LIMITED_CODE)


# Entitlement failures: Nous gets a Portal-aware message; other providers a fixed generic one (or
# the raw error when no generic text exists for the code).
_GENERIC_ENTITLEMENT_MESSAGES = {
    "subscription_required": "No active paid subscription found. Please purchase/activate a subscription, then retry.",
    "insufficient_credits": "Subscription credits are exhausted. Top up/renew credits, then retry."}
_ENTITLEMENT_ERROR_CODES = frozenset(_GENERIC_ENTITLEMENT_MESSAGES) | {
    "subscription_expired", "no_usable_credits", "account_missing", "member_spend_cap_exceeded"}


def format_auth_error(error: Exception) -> str:
    """Map auth failures to concise user-facing guidance."""
    if not isinstance(error, AuthError) or is_rate_limited_auth_error(error):
        # Rate-limit / quota errors are not credential problems: never append "re-authenticate".
        return str(error)
    if error.relogin_required:
        return f"{error} Run `hermes model` to re-authenticate."
    if error.code in _ENTITLEMENT_ERROR_CODES:
        if error.provider == "nous":
            return _format_nous_entitlement_auth_error(error)
        generic = _GENERIC_ENTITLEMENT_MESSAGES.get(error.code)
        if generic:
            return generic
    if error.code == "temporarily_unavailable":
        return f"{error} Please retry in a few seconds."
    return str(error)


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


# ── Auth Store — persistence layer for ~/.hermes/auth.json ──────────────────────────────────────────

def _auth_file_path() -> Path:
    path = get_hermes_home() / "auth.json"
    # Seat belt: under pytest, refuse to touch the real user's auth store (tests that forgot to
    # monkeypatch HERMES_HOME or escaped the hermetic conftest). In production: one dict lookup.
    if (os.environ.get("PYTEST_CURRENT_TEST")
            and _same_path(path, Path.home() / ".hermes" / "auth.json")):
        raise RuntimeError(
            f"Refusing to touch real user auth store during test run: {path}. "
            "Set HERMES_HOME to a tmp_path in your test fixture, or run "
            "via scripts/run_tests.sh for hermetic CI-parity env.")
    return path


def _global_auth_file_path() -> Optional[Path]:
    """Global-root auth.json in profile mode; None when profile and global root are the same dir.

    Read-only fallback path, so no pytest seat belt here (it lives on ``_auth_file_path()``)."""
    try:
        from hermes_constants import get_default_hermes_root
        global_root = get_default_hermes_root()
    except Exception:
        return None
    return None if _same_path(get_hermes_home(), global_root) else global_root / "auth.json"


def _load_global_auth_store() -> Dict[str, Any]:
    """Load the global-root auth store (read-only fallback, mtime-memoised); ``{}`` when absent or
    unreadable — a malformed global store must never break profile reads."""
    global _global_auth_store_cache
    global_path = _global_auth_file_path()
    if global_path is None or not global_path.exists():
        _global_auth_store_cache = None
        return {}
    try:
        cache_key: Optional[Tuple[str, int]] = (
            str(global_path.resolve(strict=False)), global_path.stat().st_mtime_ns)
    except Exception:
        cache_key = None
    cached = _global_auth_store_cache
    if cache_key is not None and cached is not None and cached[:2] == cache_key:
        return cached[2]
    if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("HOME"):
        real_root = Path(os.environ["HOME"]) / ".hermes" / "auth.json"
        try:
            if global_path.resolve(strict=False) == real_root.resolve(strict=False):
                _global_auth_store_cache = None
                return {}
        except Exception:
            pass
    try:
        store = _load_auth_store(global_path)
    except Exception:
        _global_auth_store_cache = None
        return {}
    if cache_key is not None:
        _global_auth_store_cache = (*cache_key, store)
    return store


_auth_target_lock_holders: Dict[str, threading.local] = {}
_auth_target_lock_holders_guard = threading.Lock()


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except Exception:
        return left == right


def _is_same_auth_store(left: Path, right: Path) -> bool:
    """True when two auth paths name ONE store rather than two copies.
    ``_same_path`` resolves symlinks and ``..``; ``samefile`` adds hardlinks and bind-mounts
    (same inode under two resolved names). Used by the forked-grant heal: a shared store has
    no "other side" to consolidate.

    See #101356.
    """
    if _same_path(left, right):
        return True
    try:
        return left.samefile(right)
    except OSError:
        return False


def _resolved_key(path: Path) -> str:
    """Canonical string for *path* (resolved when possible) used as a cache / lock-holder key."""
    try:
        return str(path.resolve(strict=False))
    except Exception:
        return str(path)


def _auth_lock_holder_for(target_path: Path) -> threading.local:
    """Return a reentrancy tracker keyed to one canonical auth-store path."""
    with _auth_target_lock_holders_guard:
        return _auth_target_lock_holders.setdefault(_resolved_key(target_path), threading.local())


def _kernel_lock(lock_file: Any, acquire: bool) -> None:
    """Non-blocking exclusive flock (fcntl) or 1-byte msvcrt lock at offset 0; ``acquire=False`` releases."""
    if fcntl:
        fcntl.flock(lock_file.fileno(), (fcntl.LOCK_EX | fcntl.LOCK_NB) if acquire else fcntl.LOCK_UN)
    else:
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK, 1)


@contextmanager
def _file_lock(
    lock_path: Path, holder: threading.local, timeout_seconds: float, timeout_message: str):
    """Cross-process advisory flock helper, reentrant per-thread via ``holder.depth``.

    Falls back to a depth-only guard when neither ``fcntl`` nor ``msvcrt`` is available. Callers
    supply their own ``threading.local`` so independent locks (profile store vs global root vs the
    shared Nous store) track reentrancy separately."""
    if getattr(holder, "depth", 0) > 0:
        holder.depth += 1
        try:
            yield
        finally:
            holder.depth -= 1
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock_file = None
        if fcntl is not None or msvcrt is not None:
            # msvcrt.locking needs a non-empty file with the pointer at 0. This convenience write can
            # race another holder's byte-range lock and raise PermissionError (reproduced with 20
            # concurrent processes on Windows); losing the race just means the file already has
            # content, so swallow it.
            if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
                try:
                    lock_path.write_text(" ", encoding="utf-8")
                except (OSError, PermissionError):
                    pass
            lock_file = stack.enter_context(lock_path.open("r+" if msvcrt else "a+", encoding="utf-8"))
            deadline = time.monotonic() + max(1.0, timeout_seconds)
            while True:
                try:
                    _kernel_lock(lock_file, True)
                    break
                except (BlockingIOError, OSError, PermissionError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(timeout_message)
                    time.sleep(0.05)

        holder.depth = 1
        try:
            yield
        finally:
            holder.depth = 0
            if lock_file is not None:
                try:
                    _kernel_lock(lock_file, False)
                except (OSError, IOError):
                    pass


@contextmanager
def _auth_store_lock(
    timeout_seconds: float = AUTH_LOCK_TIMEOUT_SECONDS, *, target_path: Optional[Path] = None):
    """Cross-process advisory lock for one auth.json read/write transaction.

    ``target_path`` is required for profile-to-global write-throughs: each path has its own
    reentrancy tracker and kernel lock. Lock ordering invariant: ``_auth_store_lock`` FIRST (outer),
    ``_nous_shared_store_lock`` SECOND (inner), else deadlock against a concurrent shared import."""
    auth_path = target_path if target_path is not None else _auth_file_path()
    with _file_lock(
        auth_path.with_suffix(".lock"), _auth_lock_holder_for(auth_path), timeout_seconds,
        "Timed out waiting for auth store lock"):
        yield


def _empty_auth_store() -> Dict[str, Any]:
    return {"version": AUTH_STORE_VERSION, "providers": {}}


def _load_auth_store(auth_file: Optional[Path] = None) -> Dict[str, Any]:
    auth_file = auth_file or _auth_file_path()
    if not auth_file.exists():
        return _empty_auth_store()
    try:
        raw = json.loads(auth_file.read_text(encoding="utf-8-sig"))
    except OSError:
        # Exists but unreadable (EMFILE, EACCES, EIO, stalled mount): contents are not bad, and this
        # module read-modify-writes everywhere, so an empty store here is one _save_auth_store()
        # away from erasing every credential. Fail loudly.
        logger.warning(
            "auth: could not read %s, leaving the store on disk untouched "
            "rather than degrading to an empty one",
            auth_file, exc_info=True)
        raise
    except Exception as exc:
        # Genuine corruption: unparseable JSON or non-UTF-8 bytes. Preserve a copy, but never
        # advertise a backup that was not written.
        corrupt_path = auth_file.with_suffix(".json.corrupt")
        try:
            shutil.copy2(auth_file, corrupt_path)
            preserved = True
        except Exception:
            preserved = False
            logger.debug("auth: could not preserve a copy of the corrupt store at %s", corrupt_path,
                         exc_info=True)
        logger.warning(
            "auth: failed to parse %s (%s), starting with empty store. %s %s",
            auth_file, exc,
            "Corrupt file preserved at" if preserved else "A copy could NOT be preserved at",
            corrupt_path)
        return _empty_auth_store()

    if isinstance(raw, dict) and (
        isinstance(raw.get("providers"), dict) or isinstance(raw.get("credential_pool"), dict)):
        raw.setdefault("providers", {})
        if isinstance(raw.get("providers"), dict):
            _migrate_stale_nous_portal_url(raw["providers"])
        return raw

    if isinstance(raw, dict) and isinstance(raw.get("systems"), dict):  # legacy "systems" format
        systems = raw["systems"]
        providers = {"nous": systems["nous_portal"]} if "nous_portal" in systems else {}
        return {**_empty_auth_store(), "providers": providers,
                "active_provider": "nous" if providers else None}
    return _empty_auth_store()


def _write_private_file_atomic(
    target: Path, payload: str, *, replace: Optional[Callable[[Any, Any], Any]] = None,
    fsync_dir: bool = False) -> None:
    """Write *payload* to *target* via a 0o600 temp file + atomic rename.

    ``os.open(O_EXCL, 0o600)`` closes the TOCTOU window where ``write_text()`` + post-write
    ``chmod`` briefly exposed tokens at process umask. The per-process random temp suffix avoids
    collisions between concurrent writers and stale leftovers from a crashed prior write."""
    target.parent.mkdir(parents=True, exist_ok=True)
    secure_parent_dir(target)  # refuses to chmod /, top-level dirs, or the install tree
    tmp_path = target.with_name(f"{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        (replace or atomic_replace)(tmp_path, target)
        if fsync_dir:
            try:
                dir_fd = os.open(str(target.parent), os.O_RDONLY)
            except OSError:
                pass
            else:
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _save_auth_store(auth_store: Dict[str, Any], target_path: Optional[Path] = None) -> Path:
    """Atomically persist *auth_store* (0o600, parent tightened to 0o700) to the active store, or to
    an explicit *target_path* (e.g. the global-root write-through for rotating xAI OAuth grants)."""
    auth_file = target_path if target_path is not None else _auth_file_path()
    # Tighten parent dir to 0o700 so siblings can't traverse to creds. No-op on Windows (POSIX mode bits not
    # enforced); ignore failures. secure_parent_dir refuses to chmod /, top-level dirs, or the hermes-agent
    # install tree (#25821, #93050).
    auth_store["version"] = AUTH_STORE_VERSION
    auth_store["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_private_file_atomic(auth_file, json.dumps(auth_store, indent=2) + "\n", fsync_dir=True)
    try:
        auth_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return auth_file


def _store_section(auth_store: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return ``auth_store[key]`` as a dict, replacing a missing/non-dict value in place."""
    section = auth_store.get(key)
    if not isinstance(section, dict):
        section = auth_store[key] = {}
    return section


def _provider_state_in(store: Dict[str, Any], provider_id: str) -> Optional[Dict[str, Any]]:
    """Shallow copy of ``store["providers"][provider_id]`` when it is a dict, else None."""
    providers = store.get("providers") if store else None
    state = providers.get(provider_id) if isinstance(providers, dict) else None
    return dict(state) if isinstance(state, dict) else None


def _load_provider_state_with_source(
    auth_store: Dict[str, Any], provider_id: str,
) -> tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """Provider state plus the auth.json path it came from (profile first, then the global root).

    Refresh paths that rotate single-use OAuth refresh tokens must write the updated chain back to
    the same store they read."""
    state = _provider_state_in(auth_store, provider_id)
    if state is not None:
        return state, _auth_file_path()
    global_state = _provider_state_in(_load_global_auth_store(), provider_id)
    return (global_state, _global_auth_file_path()) if global_state is not None else (None, None)


def _load_provider_state(auth_store: Dict[str, Any], provider_id: str) -> Optional[Dict[str, Any]]:
    """Provider state; in profile mode falls back to the global-root ``auth.json`` per provider (same
    shadowing as ``read_credential_pool``), so profile workers see globally-authed providers."""
    return _load_provider_state_with_source(auth_store, provider_id)[0]


@contextmanager
def _provider_state_transaction(provider_id: str):
    """Lock the active auth store and any global fallback source, in that order.

    Re-reading the source after its lock is acquired prevents stale refreshes and whole-file lost
    updates without inverting the documented auth -> shared lock order."""
    with _auth_store_lock():
        auth_store = _load_auth_store()
        state, source_path = _load_provider_state_with_source(auth_store, provider_id)
        if source_path is None or _same_path(source_path, _auth_file_path()):
            yield auth_store, state, source_path
            return
        with _auth_store_lock(target_path=source_path):
            yield auth_store, _provider_state_in(_load_auth_store(source_path), provider_id), source_path


def _store_provider_state(
    auth_store: Dict[str, Any], provider_id: str, state: Dict[str, Any], *, set_active: bool = True,
) -> None:
    _store_section(auth_store, "providers")[provider_id] = state
    if set_active:
        auth_store["active_provider"] = provider_id


def _save_provider_state(auth_store: Dict[str, Any], provider_id: str, state: Dict[str, Any]) -> None:
    """Write *state* under ``providers`` and make *provider_id* the active provider."""
    _store_provider_state(auth_store, provider_id, state, set_active=True)


def _save_active_provider_state(provider_id: str, state: Dict[str, Any]) -> Path:
    """Lock, load, write *state* as the active provider, save. Returns the auth store path."""
    with _auth_store_lock():
        auth_store = _load_auth_store()
        _save_provider_state(auth_store, provider_id, state)
        return _save_auth_store(auth_store)


def _persist_provider_state_to_store(
    provider_id: str, state: Dict[str, Any], target_path: Path, *, set_active: bool = False,
) -> Path:
    """Merge one provider into a specific auth store under that store's lock."""
    with _auth_store_lock(target_path=target_path):
        auth_store = _load_auth_store(target_path)
        _store_provider_state(auth_store, provider_id, dict(state), set_active=set_active)
        return _save_auth_store(auth_store, target_path=target_path)


def _save_provider_state_to_source(
    auth_store: Dict[str, Any], provider_id: str, state: Dict[str, Any], source_path: Optional[Path],
) -> None:
    """Persist provider state back to the auth store it was read from."""
    if source_path is None or _same_path(source_path, _auth_file_path()):
        _save_provider_state(auth_store, provider_id, state)
        _save_auth_store(auth_store)
    else:
        _persist_provider_state_to_store(provider_id, state, source_path, set_active=True)


def mark_provider_active_if_unset(provider_id: str) -> None:
    """Set ``active_provider`` only when none is set yet: the first ``hermes auth add`` credential must
    make its provider active (else setup reports "No inference provider configured"); later adds
    leave the user's choice untouched."""
    with _auth_store_lock():
        auth_store = _load_auth_store()
        if not (auth_store.get("active_provider") or "").strip():
            auth_store["active_provider"] = provider_id
            _save_auth_store(auth_store)


def is_known_auth_provider(provider_id: str) -> bool:
    normalized = (provider_id or "").strip().lower()
    return normalized in PROVIDER_REGISTRY or normalized in SERVICE_PROVIDER_NAMES


def get_auth_provider_display_name(provider_id: str) -> str:
    normalized = (provider_id or "").strip().lower()
    if normalized in PROVIDER_REGISTRY:
        return PROVIDER_REGISTRY[normalized].name
    return SERVICE_PROVIDER_NAMES.get(normalized, provider_id)


def is_runtime_provider_routable(provider_id: str) -> bool:
    """Whether runtime resolution recognizes a provider identity (a capability check, not a credential
    check): ``resolve_provider`` normalization plus the special runtime identities outside the registry."""
    normalized = (provider_id or "").strip().lower()
    if not normalized:
        return False
    if normalized in {"auto", "openrouter", "custom", "moa"} or normalized.startswith("custom:"):
        return True
    try:
        resolve_provider(normalized)
    except AuthError:
        return False
    return True


def read_credential_pool(provider_id: Optional[str] = None) -> Dict[str, Any]:
    """Return the persisted credential pool, or one provider slice.

    In profile mode the global-root ``auth.json`` is a read-only fallback applied per provider ONLY
    when the profile has zero entries for it (``hermes auth add`` in the profile shadows global)."""
    pool = _load_auth_store().get("credential_pool")
    pool = pool if isinstance(pool, dict) else {}
    global_pool = _load_global_auth_store().get("credential_pool")
    global_pool = global_pool if isinstance(global_pool, dict) else {}

    if provider_id is None:
        merged = dict(pool)
        for gp_key, gp_entries in global_pool.items():
            existing = merged.get(gp_key)
            if not (isinstance(gp_entries, list) and gp_entries):
                continue
            if not (isinstance(existing, list) and existing):  # profile wins when it has ANY entries
                merged[gp_key] = list(gp_entries)
        return merged

    provider_entries = pool.get(provider_id)
    if isinstance(provider_entries, list) and provider_entries:
        return list(provider_entries)
    global_entries = global_pool.get(provider_id)
    return list(global_entries) if isinstance(global_entries, list) else []


_POOL_STATUS_FIELDS = (
    "last_status", "last_status_at", "last_error_code", "last_error_reason", "last_error_message",
    "last_error_reset_at")


def _merge_disk_cooldown_state(
    entry: Dict[str, Any], disk_entry: Optional[Dict[str, Any]], provider_id: str,
) -> Dict[str, Any]:
    """Keep a newer on-disk cooldown/quarantine over a stale in-memory one.

    ``write_credential_pool`` persists an in-memory snapshot that may predate another process
    marking the same credential exhausted/dead; without this merge the later rewrite resurrects a
    rate-limited key as healthy and both processes resume hammering it."""
    if not isinstance(disk_entry, dict):
        return entry
    try:
        from agent.credential_pool import (
            PooledCredential, STATUS_DEAD, STATUS_EXHAUSTED, _exhausted_until, _parse_absolute_timestamp,
        )

        disk_status = disk_entry.get("last_status")
        if disk_status not in (STATUS_DEAD, STATUS_EXHAUSTED):
            return entry
        # A token change means the caller re-authed this entry and intentionally cleared its status:
        # never resurrect the old cooldown onto fresh credentials.
        mem_access = entry.get("access_token") or ""
        disk_access = disk_entry.get("access_token") or ""
        if mem_access and disk_access and mem_access != disk_access:
            return entry
        disk_ts = _parse_absolute_timestamp(disk_entry.get("last_status_at")) or 0.0
        mem_ts = _parse_absolute_timestamp(entry.get("last_status_at")) or 0.0
        if disk_ts <= mem_ts:
            return entry
        if disk_status == STATUS_EXHAUSTED:
            until = _exhausted_until(PooledCredential.from_dict(provider_id, disk_entry))
            if until is None or until <= time.time():
                return entry
        return {**entry, **{f: disk_entry.get(f) for f in _POOL_STATUS_FIELDS}}
    except Exception:  # pragma: no cover - best-effort merge
        return entry


def _entry_ids(entries: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    return {e.get("id"): e for e in entries if isinstance(e, dict) and e.get("id")}


def write_credential_pool(
    provider_id: str, entries: List[Dict[str, Any]], *,
    removed_ids: Optional[Iterable[str]] = None,
    status_cleared_ids: Optional[Iterable[str]] = None,
) -> Path:
    """Persist one provider's credential pool under auth.json.

    Final disk-boundary sanitizer for borrowed credentials (callers may pass raw dicts). Entries on
    disk but missing from *entries* (added concurrently) are merged back unless in *removed_ids*,
    so a rotation/exhaustion rewrite never drops a concurrent credential. Entries in
    *status_cleared_ids* were cleared deliberately (``hermes auth reset``) and skip the
    recency merge, which would otherwise read their cleared ``last_status_at`` (None ->
    epoch 0) as a stale snapshot and copy a still-binding cooldown back."""
    removed = {rid for rid in (removed_ids or ()) if rid}
    with _auth_store_lock():
        auth_store = _load_auth_store()
        pool = _store_section(auth_store, "credential_pool")
        sanitized = [
            sanitize_borrowed_credential_payload(e, provider_id) if isinstance(e, dict) else e
            for e in entries]
        existing_list = pool.get(provider_id)
        existing_list = existing_list if isinstance(existing_list, list) else []
        existing_by_id = _entry_ids(existing_list)
        new_ids = set(_entry_ids(sanitized))
        status_cleared = {cid for cid in (status_cleared_ids or ()) if cid}
        merged: List[Dict[str, Any]] = [
            _merge_disk_cooldown_state(
                e, None if e.get("id") in status_cleared else existing_by_id.get(e.get("id")), provider_id,
            )
            if isinstance(e, dict) else e
            for e in sanitized]
        for disk_entry in existing_list:
            disk_id = disk_entry.get("id") if isinstance(disk_entry, dict) else None
            if disk_id and disk_id not in new_ids and disk_id not in removed:
                merged.append(sanitize_borrowed_credential_payload(disk_entry, provider_id))
        pool[provider_id] = merged
        return _save_auth_store(auth_store)


def _suppressed_source_list(suppressed: Dict[str, Any], provider_id: str) -> Optional[List[str]]:
    """Canonical (list-form) suppressed sources for *provider_id*; a legacy mapping (keys = source
    names) is migrated to the list form in place."""
    raw_sources = suppressed.get(provider_id)
    if isinstance(raw_sources, list):
        return raw_sources
    if isinstance(raw_sources, dict):
        suppressed[provider_id] = [str(name) for name in raw_sources]
        return suppressed[provider_id]
    return None


def suppress_credential_source(provider_id: str, source: str) -> None:
    """Mark a credential source as suppressed so it won't be re-seeded."""
    with _auth_store_lock():
        auth_store = _load_auth_store()
        suppressed = _store_section(auth_store, "suppressed_sources")
        provider_list = _suppressed_source_list(suppressed, provider_id)
        if provider_list is None:
            provider_list = suppressed[provider_id] = []
        if source not in provider_list:
            provider_list.append(source)
        _save_auth_store(auth_store)


def is_source_suppressed(provider_id: str, source: str) -> bool:
    """Check if a credential source has been suppressed by the user."""
    try:
        return source in _load_auth_store().get("suppressed_sources", {}).get(provider_id, [])
    except Exception:
        return False


def unsuppress_credential_source(provider_id: str, source: str) -> bool:
    """Clear a suppression marker so the source will be re-seeded on the next load."""
    with _auth_store_lock():
        auth_store = _load_auth_store()
        suppressed = auth_store.get("suppressed_sources")
        if not isinstance(suppressed, dict):
            return False
        provider_list = _suppressed_source_list(suppressed, provider_id)
        if provider_list is None or source not in provider_list:
            return False
        provider_list.remove(source)
        if not provider_list:
            suppressed.pop(provider_id, None)
        if not suppressed:
            auth_store.pop("suppressed_sources", None)
        _save_auth_store(auth_store)
        return True


def get_provider_auth_state(provider_id: str) -> Optional[Dict[str, Any]]:
    """Persisted auth state for a provider (profile first, global-root fallback), or None."""
    return _load_provider_state(_load_auth_store(), provider_id)


def nous_token_has_billing_scope() -> bool:
    """Return True if the currently-held Nous token carries ``billing:manage``.

    Reads the persisted ``scope`` string saved at login (``_save_provider_state``
    stores ``token_data.get("scope") or scope``). A space-delimited match. Used by
    the lazy step-up: if False, the first billing call will 403 ``insufficient_scope``
    anyway, but checking up front lets a surface skip a doomed round-trip.
    """
    try:
        state = get_provider_auth_state("nous") or {}
    except Exception:
        return False
    scope = state.get("scope")
    if not isinstance(scope, str):
        return False
    return NOUS_BILLING_MANAGE_SCOPE in scope.split()


def get_active_provider() -> Optional[str]:
    """Return the currently active provider ID from auth store."""
    return _load_auth_store().get("active_provider")


def _active_provider_is(normalized: str) -> bool:
    active = (_load_auth_store().get("active_provider") or "").strip().lower()
    return bool(active) and active == normalized


def _slot_selects(slot: Any, normalized: str) -> bool:
    return isinstance(slot, dict) and (slot.get("provider") or "").strip().lower() == normalized


def _config_selects_provider(normalized: str) -> bool:
    """config.yaml ``model.provider``, or a MoA advisor/aggregator slot naming the provider.

    MoA presets are explicit model selections too: ``provider: anthropic`` in a MoA slot opts into
    Anthropic credentials for that slot even when the main model is another provider; otherwise
    Claude Code OAuth entries get pruned by ``load_pool("anthropic")`` and MoA advisors fail with
    "no ANTHROPIC_API_KEY" while the picker says Anthropic is logged in."""
    from hermes_cli.config import load_config
    cfg = load_config()
    if _slot_selects(cfg.get("model"), normalized):
        return True

    def _moa_block_matches(block: Any) -> bool:
        return isinstance(block, dict) and (
            any(_slot_selects(s, normalized) for s in block.get("reference_models") or [])
            or _slot_selects(block.get("aggregator"), normalized))

    moa_cfg = cfg.get("moa")
    if not isinstance(moa_cfg, dict):
        return False
    presets = moa_cfg.get("presets")
    presets = presets.values() if isinstance(presets, dict) else ()
    return _moa_block_matches(moa_cfg) or any(_moa_block_matches(p) for p in presets)


def _explicit_pool_entry_present(normalized: str) -> bool:
    """Pool rows from EXPLICIT Hermes flows (manual add / device-code / PKCE) or live env keys;
    ambient borrowed sources (gh_cli / claude_code / qwen-cli) are deliberately excluded."""
    return any(_pool_entry_is_explicit(entry) for entry in read_credential_pool(normalized))


# Set by Claude Code itself, not by the user explicitly configuring anthropic in Hermes.
_IMPLICIT_ENV_VARS = frozenset({"CLAUDE_CODE_OAUTH_TOKEN"})
_EXPLICIT_POOL_SOURCES = frozenset({"device_code", "loopback_pkce", "hermes_pkce", "manual"})
_VERTEX_PROVIDER_IDS = ("vertex", "google-vertex", "vertex-ai", "gcp-vertex", "vertexai")


def _env_secret(name: str) -> bool:
    return has_usable_secret(os.getenv(name, ""))


def _explicit_env_credentials_present(normalized: str) -> bool:
    """True when the user has pasted an explicit credential env var for *normalized*.

    Falls back to the models.dev ``ProviderDef`` (same shape) for non-registry providers such as
    openrouter. AWS SDK providers are checked via explicit env vars only — NOT boto3's chain, so
    ambient EC2 IMDS / SSO profiles never auto-surface."""
    pconfig = PROVIDER_REGISTRY.get(normalized)
    if pconfig is None:
        from hermes_cli.providers import get_provider
        pconfig = get_provider(normalized)
        if not pconfig:
            return False
    if pconfig.auth_type == "api_key":
        return any(_env_secret(v) for v in pconfig.api_key_env_vars if v not in _IMPLICIT_ENV_VARS)
    if pconfig.auth_type == "aws_sdk":
        return _env_secret("AWS_BEARER_TOKEN_BEDROCK") or (
            _env_secret("AWS_ACCESS_KEY_ID") and _env_secret("AWS_SECRET_ACCESS_KEY"))
    return False


def _pool_entry_is_explicit(entry: Any) -> bool:
    """True for pool rows the user created via an explicit Hermes flow (or a still-live env key)."""
    if not isinstance(entry, dict):
        return False
    source = str(entry.get("source") or "").strip().lower()
    if source.startswith("env:"):
        # A stale env-seeded entry survives in auth.json after the user deletes the env var: only
        # count it when the referenced var still resolves to a usable secret NOW.
        # See #55790.
        env_var = entry.get("source", "").split(":", 1)[1].strip()
        return bool(env_var and _env_secret(env_var))
    return bool(source) and (source in _EXPLICIT_POOL_SOURCES or source.startswith("manual:"))


def _keyless_provider_has_explicit_config(normalized: str) -> bool:
    """Vertex / Bedrock count as explicit when Hermes-scoped routing config is present.

    Uses has_explicit_vertex_config(), NOT has_vertex_credentials(): the latter also counts an
    ambient GOOGLE_APPLICATION_CREDENTIALS path (commonly set for unrelated GCP work). Only
    Hermes-scoped signals (VERTEX_PROJECT_ID / vertex.project_id / VERTEX_CREDENTIALS_PATH) count
    here."""
    if normalized in _VERTEX_PROVIDER_IDS:
        from agent.vertex_adapter import has_explicit_vertex_config
        return bool(has_explicit_vertex_config())
    if normalized == "bedrock":
        from hermes_cli.config import load_config
        bedrock_cfg = load_config().get("bedrock")
        return isinstance(bedrock_cfg, dict) and bool(str(bedrock_cfg.get("region") or "").strip())
    return False


# Ordered explicit-configuration checks: ``(check, best_effort)``. Best-effort checks treat an
# exception as "no"; the env-var check is NOT best-effort — a failure there must surface rather
# than let a later, weaker signal decide.
_EXPLICIT_CONFIG_CHECKS: Tuple[Tuple[Callable[[str], bool], bool], ...] = (
    (_active_provider_is, True), (_config_selects_provider, True),
    (_explicit_env_credentials_present, False), (_explicit_pool_entry_present, True),
    (_keyless_provider_has_explicit_config, True))


def is_provider_explicitly_configured(provider_id: str) -> bool:
    """True only if the user explicitly configured this provider: auth.json ``active_provider``,
    config.yaml ``model.provider`` / MoA slots, a pasted provider env var, a pool entry from a
    Hermes-initiated flow, or Hermes-scoped routing config for keyless cloud-SDK providers. Ambient
    borrowed credentials (gh CLI, qwen-cli, ~/.claude/.credentials.json) never count."""
    normalized = (provider_id or "").strip().lower()
    for check, best_effort in _EXPLICIT_CONFIG_CHECKS:
        try:
            if check(normalized):
                return True
        except Exception as exc:
            if not best_effort:
                raise
            logger.debug("explicit-config check %s failed for %s: %s", check.__name__, provider_id, exc)
    return False


def clear_provider_auth(provider_id: Optional[str] = None) -> bool:
    """Clear auth state for a provider (the active one when *provider_id* is None). Used by
    ``hermes logout``. Returns True if something was cleared."""
    with _auth_store_lock():
        auth_store = _load_auth_store()
        target = provider_id or auth_store.get("active_provider")
        if not target:
            return False
        cleared = False
        for section in ("providers", "credential_pool"):
            entries = _store_section(auth_store, section)
            if target in entries:
                del entries[target]
                cleared = True
        if auth_store.get("active_provider") == target:
            auth_store["active_provider"] = None
            cleared = True
        if cleared:
            _save_auth_store(auth_store)
        return cleared


def deactivate_provider() -> None:
    """Clear active_provider without deleting credentials: used when the user switches to a non-OAuth
    provider (OpenRouter, custom) so auto-resolution doesn't keep picking the OAuth provider."""
    with _auth_store_lock():
        auth_store = _load_auth_store()
        auth_store["active_provider"] = None
        _save_auth_store(auth_store)


# ── Provider Resolution — picks which provider to use ───────────────────────────────────────────────


def _get_config_hint_for_unknown_provider(provider_name: str) -> str:
    """Return a helpful hint string when provider resolution fails."""
    try:
        from hermes_cli.config import validate_config_structure
        issues = validate_config_structure()
        if not issues:
            return ""
        lines = ["Config issue detected — run 'hermes doctor' for full diagnostics:"]
        for ci in issues:
            lines.append(f"  [{'ERROR' if ci.severity == 'error' else 'WARNING'}] {ci.message}")
            if ci.hint and ci.hint.splitlines()[0]:
                lines.append(f"    → {ci.hint.splitlines()[0]}")
        return "\n".join(lines)
    except Exception:
        return ""


def _refuse_env_adoption_if_config_corrupt() -> None:
    """Refuse env-key/pool auto-adoption of openrouter while config.yaml is corrupt.

    A corrupt config loads as ``DEFAULT_CONFIG`` (no ``model.provider``), so the env sniff would
    silently adopt the PAID openrouter provider over whatever the broken config really names.
    Fires ONLY on the auto path and clears itself once the file parses again."""
    try:
        from hermes_cli.config import get_active_config_parse_failure
        err = get_active_config_parse_failure()
        if not err:
            return
        path = get_config_path()
    except Exception as e:
        logger.debug("Could not probe config parse-failure state: %s", e)
        return
    raise AuthError(
        f"config.yaml at {path} is corrupt ({err}) — refusing to auto-select "
        f"an inference provider from environment keys. Fix the YAML (a backup "
        f"was saved next to it) or run hermes setup.",
        code="corrupt_config")


# Provider aliases accepted by resolve_provider(). Plugin-declared aliases
# (plugins/model-providers/<name>/) are layered on at call time; this hardcoded
# table remains authoritative for existing names.
_PROVIDER_ALIASES: Dict[str, str] = {
    "glm": "zai", "z-ai": "zai", "z.ai": "zai", "zhipu": "zai",
    "google": "gemini", "google-gemini": "gemini", "google-ai-studio": "gemini",
    "x-ai": "xai", "x.ai": "xai", "grok": "xai",
    "xai-oauth": "xai-oauth", "x-ai-oauth": "xai-oauth",
    "grok-oauth": "xai-oauth", "xai-grok-oauth": "xai-oauth",
    "kimi": "kimi-coding", "kimi-for-coding": "kimi-coding", "moonshot": "kimi-coding",
    "kimi-cn": "kimi-coding-cn", "moonshot-cn": "kimi-coding-cn",
    "step": "stepfun", "stepfun-coding-plan": "stepfun",
    "arcee-ai": "arcee", "arceeai": "arcee",
    "gmi-cloud": "gmi", "gmicloud": "gmi",
    "actual-computer": "actual", "actualcomputer": "actual", "aci": "actual",
    "minimax-china": "minimax-cn", "minimax_cn": "minimax-cn",
    "minimax-portal": "minimax-oauth", "minimax-global": "minimax-oauth", "minimax_oauth": "minimax-oauth",
    "alibaba_coding": "alibaba-coding-plan", "alibaba-coding": "alibaba-coding-plan",
    "alibaba_coding_plan": "alibaba-coding-plan",
    "claude": "anthropic", "claude-code": "anthropic",
    "github": "copilot", "github-copilot": "copilot",
    "github-models": "copilot", "github-model": "copilot",
    "github-copilot-acp": "copilot-acp", "copilot-acp-agent": "copilot-acp",
    "aigateway": "ai-gateway", "vercel": "ai-gateway", "vercel-ai-gateway": "ai-gateway",
    "opencode": "opencode-zen", "zen": "opencode-zen",
    "free": "opencode-free", "opencode_free": "opencode-free",
    "qwen-portal": "qwen-oauth", "qwen-cli": "qwen-oauth", "qwen-oauth": "qwen-oauth",
    "hf": "huggingface", "hugging-face": "huggingface", "huggingface-hub": "huggingface",
    "mimo": "xiaomi", "xiaomi-mimo": "xiaomi",
    "tencent": "tencent-tokenhub", "tokenhub": "tencent-tokenhub",
    "tencent-cloud": "tencent-tokenhub", "tencentmaas": "tencent-tokenhub",
    "tokenplan": "tencent-tokenplan", "tencent-lkeap": "tencent-tokenplan",
    "aws": "bedrock", "aws-bedrock": "bedrock", "amazon-bedrock": "bedrock", "amazon": "bedrock",
    "go": "opencode-go", "opencode-go-sub": "opencode-go",
    "kilo": "kilocode", "kilo-code": "kilocode", "kilo-gateway": "kilocode",
    "lmstudio": "lmstudio", "lm-studio": "lmstudio", "lm_studio": "lmstudio",
    # Local server aliases — route through the generic custom provider
    "ollama": "custom", "ollama_cloud": "ollama-cloud",
    "vllm": "custom", "llamacpp": "custom",
    "llama.cpp": "custom", "llama-cpp": "custom"}


def _plugin_aliases() -> Dict[str, str]:
    """``_PROVIDER_ALIASES`` extended with aliases declared in plugins/model-providers/<name>/."""
    aliases = dict(_PROVIDER_ALIASES)
    try:
        from providers import list_providers as _lp
        for _pp in _lp():
            for _alias in _pp.aliases:
                aliases.setdefault(_alias, _pp.name)
    except Exception:
        pass
    return aliases


def _scoped_key_env_reader() -> Callable[[str], str]:
    """Scope-aware key reader for provider auto-detection.

    Under multiplex a secondary profile's keys live only in its secret scope, not os.environ. Catch
    ONLY ImportError: any other auxiliary_client failure must propagate rather than silently
    falling back to os.getenv (a traceless fail-open)."""
    try:
        # Scope-aware key reads: under multiplex a secondary profile's API keys live only in its secret
        # scope, not os.environ — a bare getenv here would find nothing and auto-resolution would report "No
        # LLM provider configured" for every secondary profile (same class as #86905).
        from agent.auxiliary_client import _scoped_key_env
        return _scoped_key_env
    except ImportError:
        logger.warning(
            "agent.auxiliary_client unavailable (%s); provider auto-detection "
            "will read keys from the process environment only — under "
            "multiplex, secondary profiles may report 'No LLM provider'.",
            "import failed")
        return lambda name: os.getenv(name) or ""


def _openrouter_auto_detected(scoped_key_env: Callable[[str], str]) -> bool:
    """True when an OpenRouter credential exists via env key or the credential pool (a key added via
    `hermes auth add openrouter` has no env var; without the pool check it is invisible to
    auto-detection and requests go out with no Authorization header)."""
    if any(has_usable_secret(scoped_key_env(v)) for v in ("OPENAI_API_KEY", "OPENROUTER_API_KEY")):
        return True
    try:
        # Auto-detect an OpenRouter credential added via `hermes auth add openrouter` (manual pool entry, no
        # env var). Without this, a key that only lives in the credential pool is invisible to
        # auto-detection — the user sees `hermes auth list` showing the credential while requests go out
        # with no Authorization header ("HTTP 401: Missing Authentication header"). The env-var check above
        # only covers keys exported as OPENROUTER_API_KEY / OPENAI_API_KEY. See issue #42130.
        from agent.credential_pool import load_pool as _load_pool
        return bool(_load_pool("openrouter").has_credentials())
    except Exception as e:
        logger.debug("Could not check OpenRouter credential pool: %s", e)
        return False


def _logged_in_oauth_active_provider() -> Optional[str]:
    """auth.json ``active_provider`` when it is a registry provider that reports logged in."""
    try:
        _maybe = _load_auth_store().get("active_provider")
        if _maybe and _maybe in PROVIDER_REGISTRY and get_auth_status(_maybe).get("logged_in"):
            return _maybe
    except Exception as e:
        logger.debug("Could not pre-read active auth provider: %s", e)
    return None


def _config_model_provider() -> Tuple[Any, Optional[str]]:
    """``(model_cfg, provider)`` from config.yaml when ``model.provider`` names a registry provider.

    The normal chat/gateway path resolves config.provider upstream in resolve_requested_provider();
    this is the safety net for the lone direct caller (main.py resolve_provider("auto"))."""
    try:
        from hermes_cli.config import load_config
        model_cfg = (load_config() or {}).get("model")
        provider = model_cfg.get("provider") if isinstance(model_cfg, dict) else None
        provider = provider.strip().lower() if isinstance(provider, str) else ""
        return model_cfg, (provider if provider in PROVIDER_REGISTRY else None)
    except Exception as e:
        logger.debug("Could not read config.yaml model.provider for auto-resolution: %s", e)
        return None, None


# API-key providers never auto-selected from env: GitHub tokens are commonly present for repo/tool
# access and must not hijack inference; LM Studio is a local server whose availability isn't
# implied by LM_API_KEY (may be offline; no-auth setup uses a placeholder). Both need an explicit
# choice.
_NO_AUTO_DETECT_PROVIDERS = frozenset({"copilot", "lmstudio"})


def _env_key_auto_detected(
    scoped_key_env: Callable[[str], str], oauth_active: Optional[str]) -> Optional[str]:
    """First registry api_key provider (registry order) with a usable env key, warning when it
    preempts a logged-in OAuth provider so a stale key in ~/.hermes/.env never switches silently."""
    for pid, pconfig in PROVIDER_REGISTRY.items():
        if pconfig.auth_type != "api_key" or pid in _NO_AUTO_DETECT_PROVIDERS:
            continue
        for env_var in pconfig.api_key_env_vars:
            if has_usable_secret(scoped_key_env(env_var)):
                if oauth_active and oauth_active != pid:
                    logger.warning(
                        # An exported API key now wins over a logged-in OAuth provider (the #29285 fix).
                        # Surface that so a user who deliberately uses OAuth but has a stale key in
                        # ~/.hermes/.env isn't silently switched without knowing why.
                        "Provider resolved to %r via %s, preempting your "
                        "logged-in OAuth provider %r. If you meant to use the "
                        "OAuth login, unset %s or set `model.provider` "
                        "explicitly.",
                        pid, env_var, oauth_active, env_var)
                return pid
    return None


def resolve_provider(
    requested: Optional[str] = None,
    *,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None) -> str:
    """Determine which inference provider to use.

    "auto" priority (explicit intent beats a stale OAuth login): 1. CLI api_key/base_url ->
    "openrouter"; 2. config.yaml ``model.provider``; 3. OPENAI_API_KEY / OPENROUTER_API_KEY ->
    "openrouter"; 4. OpenRouter pool; 5. provider env keys; 6. auth.json ``active_provider``;
    7. AWS Bedrock chain; 8. AuthError(no_provider_configured).

    1. 3. 4. 5. Provider-specific API keys (GLM, Kimi, MiniMax, ...) -> that provider 7. 8. Error (no
    provider configured) See #29285.
    """
    normalized = (requested or "auto").strip().lower()
    normalized = _plugin_aliases().get(normalized, normalized)

    if normalized in ("openrouter", "custom") or normalized in PROVIDER_REGISTRY:
        return normalized
    if normalized != "auto":
        hint = _get_config_hint_for_unknown_provider(normalized)
        tail = (f"\n\n{hint}" if hint else " Check 'hermes model' for available providers, "
                "or run 'hermes doctor' to diagnose config issues.")
        raise AuthError(f"Unknown provider '{normalized}'." + tail, code="invalid_provider")

    if explicit_api_key or explicit_base_url:  # one-off CLI creds always mean openrouter/custom
        return "openrouter"

    _model_cfg, cfg_provider = _config_model_provider()
    if cfg_provider:
        return cfg_provider

    _scoped_key_env = _scoped_key_env_reader()
    if _openrouter_auto_detected(_scoped_key_env):
        _refuse_env_adoption_if_config_corrupt()
        return "openrouter"

    # Determined up front so the env-key tier can warn when an exported key preempts it; the actual
    # OAuth fallback still happens after the env-key tier.
    _oauth_active = _logged_in_oauth_active_provider()
    env_pid = _env_key_auto_detected(_scoped_key_env, _oauth_active)
    if env_pid:
        return env_pid

    # Logged-in OAuth provider is a LAST-RESORT fallback (it used to sit above the env/config
    # checks, so a stale login silently overrode explicit intent).
    # Logged-in OAuth provider (auth.json `active_provider`) — a LAST-RESORT fallback, chosen only when the
    # user expressed no other preference above. Demoted here so explicit intent always wins. See #29285.
    if _oauth_active:
        if isinstance(_model_cfg, dict) and _model_cfg and not _model_cfg.get("provider"):
            logger.warning(
                "Provider resolved to logged-in OAuth provider %r because "
                "config.yaml `model` has no `provider` key. If you meant a "
                "different provider, set `model.provider` explicitly.",
                _oauth_active)
        return _oauth_active

    # AWS Bedrock via the boto3 credential chain (IAM roles, SSO, env vars); after API-key providers
    # so explicit keys always win.
    try:
        from agent.bedrock_adapter import has_aws_credentials
        if has_aws_credentials():
            return "bedrock"
    except ImportError:
        pass  # boto3 not installed
    raise AuthError(
        "No inference provider configured. Run 'hermes model' to choose a "
        "provider and model, or set an API key (OPENROUTER_API_KEY, "
        "OPENAI_API_KEY, etc.) in ~/.hermes/.env.",
        code="no_provider_configured")


# ── Timestamp / TTL helpers ─────────────────────────────────────────────────────────────────────────

def _utc_now_z() -> str:
    """Current UTC time as an ISO-8601 string with a ``Z`` suffix (last_refresh format)."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_timestamp(value: Any) -> Optional[float]:
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _is_expiring(expires_at_iso: Any, skew_seconds: int) -> bool:
    expires_epoch = _parse_iso_timestamp(expires_at_iso)
    return expires_epoch is None or expires_epoch <= (time.time() + skew_seconds)


def _tls_state_from_verify(verify: Any) -> Dict[str, Any]:
    """Persistable ``tls`` block derived from an httpx ``verify`` value."""
    return {"insecure": verify is False, "ca_bundle": verify if isinstance(verify, str) else None}


def _last_auth_error_marker(
    provider: str, error: "AuthError", *, reason: str, default_code: Optional[str] = None,
) -> Dict[str, Any]:
    """The ``last_auth_error`` record persisted when dead OAuth material is quarantined."""
    return {
        "provider": provider, "message": str(error), "reason": reason, "relogin_required": True,
        "code": error.code if default_code is None else (error.code or default_code),
        "at": datetime.now(timezone.utc).isoformat()}


_FLAT_OAUTH_TOKEN_KEYS = ("access_token", "refresh_token", "expires_at", "expires_in", "obtained_at")


def _quarantine_flat_oauth_state(state: Dict[str, Any], provider: str, exc: "AuthError") -> None:
    """Strip dead tokens from a flat OAuth state after a terminal runtime refresh failure so
    subsequent calls fail fast without a network retry (mirrors the Nous / xAI / Codex pattern)."""
    for _k in _FLAT_OAUTH_TOKEN_KEYS:
        state.pop(_k, None)
    state["last_auth_error"] = _last_auth_error_marker(
        provider, exc, reason="runtime_refresh_failure", default_code="refresh_failed")


def _coerce_ttl_seconds(expires_in: Any) -> int:
    try:
        return max(0, int(expires_in))
    except Exception:
        return 0


def _optional_base_url(value: Any) -> Optional[str]:
    cleaned = value.strip().rstrip("/") if isinstance(value, str) else ""
    return cleaned or None


# Valid Nous Portal hosts; a stored portal_base_url outside this set is a misconfiguration and falls
# back to the default. localhost / 127.0.0.1 are for local development and testing.
_NOUS_PORTAL_ALLOWED_HOSTS: FrozenSet[str] = frozenset({
    "portal.nousresearch.com", "localhost", "127.0.0.1"})

# Per-process memo for resolve_nous_access_token: startup runs one check_fn per managed tool and
# each would trigger its own ~15s blocking refresh of an expired token; a short-TTL memo collapses
# the burst into one round-trip. Callers needing freshness use force_fresh/refresh_nous_oauth_pure.
_RESOLVE_TOKEN_CACHE_LOCK = threading.Lock()
_RESOLVE_TOKEN_CACHE: "tuple[float, str] | None" = None
_RESOLVE_TOKEN_CACHE_TTL_S = 5.0


def _nous_portal_base_url(state: Dict[str, Any]) -> str:
    """HERMES_PORTAL_BASE_URL / NOUS_PORTAL_BASE_URL is the trusted operator override and wins
    OUTRIGHT, bypassing the host allowlist (which exists to reject an untrusted network-provided
    value, not one the operator configured). Otherwise the stored/default value, allowlist-gated."""
    env_portal_override = _nous_portal_env_override()
    if env_portal_override:
        return env_portal_override.rstrip("/")
    portal_base_url = _optional_base_url(state.get("portal_base_url")) or DEFAULT_NOUS_PORTAL_URL
    portal_base_url = portal_base_url.rstrip("/")
    host = urlparse(portal_base_url).hostname
    if host and host not in _NOUS_PORTAL_ALLOWED_HOSTS:
        logger.warning(
            "auth: ignoring invalid portal_base_url %r (host %r not in allowlist), using default",
            portal_base_url, host)
        return DEFAULT_NOUS_PORTAL_URL
    return portal_base_url


def resolve_nous_access_token(
    *,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    refresh_skew_seconds: int = ACCESS_TOKEN_REFRESH_SKEW_SECONDS) -> str:
    """Resolve a refresh-aware Nous Portal access token for managed tool gateways."""
    global _RESOLVE_TOKEN_CACHE
    # Only a default-TLS resolution is memoised; error paths never populate the memo.
    memoable = not insecure and ca_bundle is None
    if memoable:
        with _RESOLVE_TOKEN_CACHE_LOCK:
            cached = _RESOLVE_TOKEN_CACHE
        if cached is not None and (time.monotonic() - cached[0]) < _RESOLVE_TOKEN_CACHE_TTL_S:
            return cached[1]

    def _memo(token: str) -> str:
        global _RESOLVE_TOKEN_CACHE
        if memoable:
            with _RESOLVE_TOKEN_CACHE_LOCK:
                _RESOLVE_TOKEN_CACHE = (time.monotonic(), token)
        return token

    with _provider_state_transaction("nous") as (auth_store, state, state_source_path):
        if not state:
            raise _nous_err("Hermes is not logged into Nous Portal.", relogin=True)
        portal_base_url = _nous_portal_base_url(state)
        client_id = str(state.get("client_id") or DEFAULT_NOUS_CLIENT_ID)
        verify = _resolve_verify(insecure=insecure, ca_bundle=ca_bundle, auth_state=state)
        persist = lambda: _save_provider_state_to_source(  # noqa: E731
            auth_store, "nous", state, state_source_path)

        lock_timeout = max(timeout_seconds + 5.0, AUTH_LOCK_TIMEOUT_SECONDS)
        with _nous_shared_store_lock(timeout_seconds=lock_timeout):
            merged_shared = _merge_shared_nous_oauth_state(state)
            access_token = state.get("access_token")
            refresh_token = state.get("refresh_token")
            if not isinstance(access_token, str) or not access_token:
                raise _nous_err("No access token found for Nous Portal login.", relogin=True)

            if not _is_expiring(state.get("expires_at"), refresh_skew_seconds):
                if merged_shared:
                    persist()
                # Memoise the valid-token fast path too: each check_fn otherwise pays two
                # cross-process file locks to get here. The token has >= refresh_skew_seconds (>=
                # 120s) of life, so a 5s memo can never serve an expired token.
                return _memo(access_token)

            if not isinstance(refresh_token, str) or not refresh_token:
                raise _nous_err("Session expired and no refresh token is available.", relogin=True)

            with httpx.Client(timeout=httpx.Timeout(timeout_seconds or 15.0),
                              headers={"Accept": "application/json"}, verify=verify) as client:
                refreshed = _refresh_nous_or_quarantine(
                    client=client, auth_store=auth_store, state=state, portal_base_url=portal_base_url,
                    client_id=client_id, refresh_token=refresh_token,
                    reason="managed_access_token_refresh_failure", persist=persist)

            _apply_nous_refreshed_tokens(state, refreshed, refresh_token)
            state["portal_base_url"] = portal_base_url
            state["client_id"] = client_id
            state["tls"] = _tls_state_from_verify(verify)
            persist()
            _write_shared_nous_state(state)
            return _memo(state["access_token"])


# ── Status helpers ──────────────────────────────────────────────────────────────────────────────────

# Process-level memo for get_nous_auth_status(): it validates via a synchronous refresh POST
# (~350ms) and read-only UI surfaces call it many times per render (~31x per menu paint), burning
# single-use refresh tokens. Keyed on auth.json path + mtime so profile switches don't share a memo
# and login/logout/add/remove invalidate naturally.
_NOUS_AUTH_STATUS_CACHE_TTL = 15.0  # seconds
_nous_auth_status_cache: Optional[Tuple[float, str, Optional[float], Dict[str, Any]]] = None

# mtime-keyed memo for _load_global_auth_store(): (path, mtime_ns, store); same invalidation rule.
_global_auth_store_cache: Optional[Tuple[str, int, Dict[str, Any]]] = None


def _auth_file_cache_key() -> Tuple[str, Optional[float]]:
    auth_file = _auth_file_path()
    try:
        return _resolved_key(auth_file), auth_file.stat().st_mtime
    except Exception:  # missing file included: key without an mtime
        return _resolved_key(auth_file), None


def invalidate_nous_auth_status_cache() -> None:
    """Clear the get_nous_auth_status() memo (for code paths that mutate Nous auth state without
    touching auth.json, e.g. tests; login/logout invalidate via the mtime check automatically)."""
    global _nous_auth_status_cache
    _nous_auth_status_cache = None


def get_nous_auth_status() -> Dict[str, Any]:
    """Status snapshot for Nous auth, memoised ~15s keyed on the auth.json mtime.

    Prefers the auth-store provider state (the live source of truth for refresh) and validates it by
    resolving runtime credentials so revoked refresh sessions do not show up as a healthy login."""
    global _nous_auth_status_cache
    now = time.monotonic()
    auth_file_key, mtime = _auth_file_cache_key()
    cached = _nous_auth_status_cache
    if (cached is not None and cached[1:3] == (auth_file_key, mtime)
            and (now - cached[0]) < _NOUS_AUTH_STATUS_CACHE_TTL):
        return dict(cached[3])
    status = _compute_nous_auth_status()
    _nous_auth_status_cache = (now, auth_file_key, mtime, dict(status))
    return status


@dataclass(frozen=True)
class OAuthProviderFlow:
    """Per-provider OAuth plumbing, keyed by provider id in ``OAUTH_PROVIDER_FLOWS``.

    Callables are named (strings) and looked up in this module at call time so
    ``monkeypatch.setattr("hermes_cli.auth.resolve_codex_runtime_credentials", ...)`` applies."""
    provider_id: str
    resolve_fn: str
    status_fn: str
    terminal_refresh_codes: FrozenSet[str] = frozenset()  # retrying the same refresh token cannot succeed
    # ``hermes logout`` with no active provider falls back to config.yaml ``model.provider`` only
    # for providers whose credentials live in auth.json.
    logout_from_config: bool = False

    def resolve(self, **kwargs: Any) -> Dict[str, Any]:
        return globals()[self.resolve_fn](**kwargs)

    def status(self) -> Dict[str, Any]:
        return globals()[self.status_fn]()

    def is_terminal_refresh_error(self, exc: Exception) -> bool:
        return (
            isinstance(exc, AuthError) and exc.provider == self.provider_id
            and exc.code in self.terminal_refresh_codes and bool(exc.relogin_required))


_OAUTH_GRANT_DEAD_CODES = frozenset({"invalid_grant", "invalid_token", "refresh_token_reused"})

OAUTH_PROVIDER_FLOWS: Dict[str, OAuthProviderFlow] = {
    "nous": OAuthProviderFlow(
        "nous", "resolve_nous_runtime_credentials", "get_nous_auth_status",
        terminal_refresh_codes=_OAUTH_GRANT_DEAD_CODES, logout_from_config=True),
    "openai-codex": OAuthProviderFlow(
        "openai-codex", "resolve_codex_runtime_credentials", "get_codex_auth_status",
        terminal_refresh_codes=_OAUTH_GRANT_DEAD_CODES | {"codex_refresh_failed", "codex_auth_missing_refresh_token"},
        logout_from_config=True),
    "xai-oauth": OAuthProviderFlow(
        "xai-oauth", "resolve_xai_oauth_runtime_credentials", "get_xai_oauth_auth_status",
        terminal_refresh_codes=frozenset({"xai_refresh_failed", "xai_auth_missing_refresh_token"}),
        logout_from_config=True),
    "qwen-oauth": OAuthProviderFlow(
        "qwen-oauth", "resolve_qwen_runtime_credentials", "get_qwen_auth_status"),
    "minimax-oauth": OAuthProviderFlow(
        "minimax-oauth", "resolve_minimax_oauth_runtime_credentials", "get_minimax_oauth_auth_status"),
}


def _is_terminal_refresh_error(exc: Exception, provider: str) -> bool:
    """True when retrying the same *provider* refresh token cannot succeed."""
    return OAUTH_PROVIDER_FLOWS[provider].is_terminal_refresh_error(exc)


_is_terminal_nous_refresh_error = partial(_is_terminal_refresh_error, provider="nous")
_is_terminal_xai_oauth_refresh_error = partial(_is_terminal_refresh_error, provider="xai-oauth")
_is_terminal_codex_oauth_refresh_error = partial(
    _is_terminal_refresh_error, provider="openai-codex")


def _codex_pool_rate_limited_status() -> Optional[Dict[str, Any]]:
    rate_limit = _codex_pool_rate_limit_status()
    if not rate_limit:
        return None
    return {
        "logged_in": True, "auth_store": str(_auth_file_path()),
        "last_refresh": rate_limit.get("last_refresh"), "auth_mode": "chatgpt",
        "source": f"pool:{rate_limit.get('label') or 'unknown'}", "rate_limited": True,
        "error_code": CODEX_RATE_LIMITED_CODE,
        "error": (rate_limit.get("message")
                  or "Codex provider quota exhausted; retry after the usage limit resets."),
        "reset_at": rate_limit.get("reset_at")}


def get_codex_auth_status() -> Dict[str, Any]:
    """Status snapshot for Codex auth (pool first, then legacy provider state)."""
    return _pool_first_oauth_status(
        "openai-codex", is_expiring=_codex_access_token_is_expiring, auth_mode="chatgpt",
        resolve=resolve_codex_runtime_credentials, on_pool_miss=_codex_pool_rate_limited_status)


def get_xai_oauth_auth_status() -> Dict[str, Any]:
    # auth_mode is display/telemetry only; device-code is the only xAI OAuth flow, so report it
    # unconditionally (auth.json may still carry a legacy ``oauth_pkce`` label).
    return _pool_first_oauth_status(
        "xai-oauth", is_expiring=_xai_access_token_is_expiring, auth_mode="oauth_device_code",
        resolve=resolve_xai_oauth_runtime_credentials)


def _provider_env_base_url(pconfig: ProviderConfig) -> str:
    return os.getenv(pconfig.base_url_env_var, "").strip() if pconfig.base_url_env_var else ""


def _provider_is_keyless(provider_id: str) -> bool:
    """HermesOverlay keyless flag — the same source the provider catalog and GUI contract tests use."""
    try:
        from hermes_cli.providers import HERMES_OVERLAYS
        return bool(getattr(HERMES_OVERLAYS.get(provider_id), "keyless", False))
    except Exception:
        return False


def get_api_key_provider_status(provider_id: str) -> Dict[str, Any]:
    """Status snapshot for API-key providers (z.ai, Kimi, MiniMax)."""
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "api_key":
        return {"configured": False}
    status = {
        "configured": True, "provider": provider_id, "name": pconfig.name, "key_source": "keyless",
        "base_url": pconfig.inference_base_url, "logged_in": True}
    if _provider_is_keyless(provider_id):
        # Keyless providers (opencode-free) are served anonymously: every install counts as
        # configured.
        return status

    api_key, key_source = _resolve_api_key_provider_secret(provider_id, pconfig)
    env_url = _provider_env_base_url(pconfig)
    if provider_id in {"kimi-coding", "kimi-coding-cn"}:
        base_url = _resolve_kimi_base_url(api_key, pconfig.inference_base_url, env_url)
    else:
        base_url = env_url or pconfig.inference_base_url
    actual_local_noauth = False
    if provider_id == "actual":
        base_url = normalize_actual_base_url(base_url)
        actual_local_noauth = not api_key and is_actual_local_base_url(base_url)
    configured = bool(api_key) or actual_local_noauth
    status.update(  # logged_in mirrors configured for compat with the OAuth status shape
        configured=configured, base_url=base_url, logged_in=configured,
        key_source=key_source or ("local-offline" if actual_local_noauth else ""))
    return status


def _external_process_auth_evidence(provider_id: str) -> tuple[bool, Optional[str]]:
    """Best-effort POSITIVE evidence ``(verified, source)`` that an external-process CLI is authed.

    False means "not verifiable from here", NOT "signed out" (the Copilot CLI may use an OS keychain
    Hermes can't read). Deliberately subprocess-free: spawning ``gh auth token`` from status
    endpoints/pickers re-creates the cold-start stall copilot_auth.py avoids."""
    if provider_id != "copilot-acp":
        return False, None
    # 1. Supported env tokens — the same vars the Copilot CLI itself honors.
    try:
        from hermes_cli.copilot_auth import COPILOT_ENV_VARS, validate_copilot_token
        for env_var in COPILOT_ENV_VARS:
            val = os.getenv(env_var, "").strip()
            if val and validate_copilot_token(val)[0]:
                return True, f"env: {env_var}"
    except Exception as exc:
        logger.debug("copilot-acp env token evidence check failed: %s", exc)
    # 2. The Copilot CLI's own plaintext token store (written by `copilot login` when no OS keychain
    #    is available). The file is JSONC — strip //-comment lines before parsing.
    try:
        cli_config = os.path.expanduser("~/.copilot/config.json")
        if os.path.isfile(cli_config):
            with open(cli_config, "r", encoding="utf-8", errors="ignore") as fh:
                raw = "\n".join(
                    line for line in fh.read().splitlines() if not line.lstrip().startswith("//"))
            tokens = (json.loads(raw) if raw.strip() else {}).get("copilotTokens")
            if isinstance(tokens, dict) and any(
                isinstance(v, str) and v.strip() for v in tokens.values()):
                return True, "~/.copilot/config.json"
    except Exception as exc:
        logger.debug("copilot-acp CLI config evidence check failed: %s", exc)
    # 3. Known on-disk GitHub Copilot credential stores (the same files models.py fingerprints).
    for cred_path in ("~/.config/github-copilot/hosts.json", "~/.config/github-copilot/apps.json"):
        try:
            expanded = os.path.expanduser(cred_path)
            if os.path.isfile(expanded) and os.path.getsize(expanded) > 2:
                return True, cred_path
        except OSError:
            continue
    return False, None


def _external_process_spec(
    pconfig: ProviderConfig) -> tuple[str, List[str], str, Optional[str], tuple[str, ...]]:
    """``(command, args, base_url, resolved_command, command_env_vars)`` for an ACP provider.

    Launch details come from the provider's own profile (copilot-acp: HERMES_COPILOT_ACP_COMMAND /
    COPILOT_CLI_PATH / HERMES_COPILOT_ACP_ARGS), so out-of-tree providers describe their binary."""
    base_url = _provider_env_base_url(pconfig) or pconfig.inference_base_url
    try:
        from providers import get_provider_profile as _get_provider_profile
        profile = _get_provider_profile(pconfig.id)
    except Exception:
        profile = None
    command_env_vars = tuple(getattr(profile, "process_command_env_vars", ()) or ())
    args_env_var = str(getattr(profile, "process_args_env_var", "") or "")
    command = (next((v for v in (os.getenv(var, "").strip() for var in command_env_vars) if v), "")
               or str(getattr(profile, "process_command", "") or ""))
    raw_args = os.getenv(args_env_var, "").strip() if args_env_var else ""
    args = shlex.split(raw_args) if raw_args else list(getattr(profile, "process_args", ()) or [])
    return command, args, base_url, shutil.which(command) if command else None, command_env_vars


def get_external_process_provider_status(provider_id: str) -> Dict[str, Any]:
    """Status snapshot for providers that run a local subprocess.

    ``configured``/``logged_in`` are structural (executable resolves or TCP endpoint set): the
    subprocess owns real auth. ``auth_verified``/``auth_source`` carry positive evidence only."""
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "external_process":
        return {"configured": False}
    command, args, base_url, resolved_command, _ = _external_process_spec(pconfig)
    available = bool(resolved_command or base_url.startswith("acp+tcp://"))
    auth_verified, auth_source = _external_process_auth_evidence(provider_id)
    return {
        "configured": available, "provider": provider_id, "name": pconfig.name, "command": command,
        "args": args, "resolved_command": resolved_command, "base_url": base_url,
        "logged_in": available, "auth_verified": auth_verified, "auth_source": auth_source}


def _get_aws_sdk_auth_status(target: str) -> Dict[str, Any]:
    """AWS SDK providers (Bedrock) — check via boto3 credential chain."""
    try:
        from agent.bedrock_adapter import has_aws_credentials
        return {"logged_in": has_aws_credentials(), "provider": target}
    except ImportError:
        return {"logged_in": False, "provider": target, "error": "boto3 not installed"}


def get_auth_status(provider_id: Optional[str] = None) -> Dict[str, Any]:
    """Generic auth status dispatcher: bespoke builders (``OAUTH_PROVIDER_FLOWS`` plus Spotify /
    Azure Foundry) first, then the registry ``auth_type`` so a whole provider class (e.g. every
    external-process ACP backend) gets a real status. Builders are looked up by NAME at call time so
    tests that patch ``hermes_cli.auth.get_*_auth_status`` still apply."""
    target = (provider_id or get_active_provider() or "").strip().lower()
    if not target:
        return {"logged_in": False}
    status_fn_name = _BESPOKE_STATUS_FUNCTIONS.get(target)
    if status_fn_name:
        return globals()[status_fn_name]()
    pconfig = PROVIDER_REGISTRY.get(target)
    if pconfig and pconfig.auth_type in _STATUS_BY_AUTH_TYPE:
        return globals()[_STATUS_BY_AUTH_TYPE[pconfig.auth_type]](target)
    return {"logged_in": False}


# Bespoke status builders (name -> looked up in this module at call time) win over the
# auth_type-keyed fallbacks below.
_BESPOKE_STATUS_FUNCTIONS: Dict[str, str] = {
    **{pid: flow.status_fn for pid, flow in OAUTH_PROVIDER_FLOWS.items()},
    "spotify": "get_spotify_auth_status",
    "azure-foundry": "_get_azure_foundry_auth_status",
    # oauth_external plugin profile: _register_plugin_provider skips it, so the auth_type fallback
    # below never sees claude-agent-sdk; its SDK-owned subscription login is probed structurally.
    "claude-agent-sdk": "get_claude_agent_sdk_auth_status"}
_STATUS_BY_AUTH_TYPE: Dict[str, str] = {
    "external_process": "get_external_process_provider_status",
    "api_key": "get_api_key_provider_status",
    "aws_sdk": "_get_aws_sdk_auth_status"}


def _get_azure_foundry_auth_status() -> Dict[str, Any]:
    """Structural auth status for Azure Foundry.

    ``entra_id``: ``azure-identity`` importable — never invokes the Entra credential chain (keeps
    CLI startup flat; ``hermes doctor`` runs the live probe). ``api_key`` (default): usable
    ``AZURE_FOUNDRY_API_KEY``."""
    info: Dict[str, Any] = {"provider": "azure-foundry"}
    try:
        from hermes_cli.config import load_config, get_env_value_prefer_dotenv
        cfg = load_config()
    except Exception:
        cfg = {}
    model_cfg = cfg.get("model") if isinstance(cfg, dict) else None
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    auth_mode = str(model_cfg.get("auth_mode") or "api_key").strip().lower() or "api_key"
    info["auth_mode"] = auth_mode
    info["base_url"] = str(model_cfg.get("base_url") or "").strip()

    if auth_mode == "entra_id":
        try:
            from agent.azure_identity_adapter import (
                EntraIdentityConfig, SCOPE_AI_AZURE_DEFAULT, has_azure_identity_installed)
            installed = has_azure_identity_installed()
            entra_cfg = model_cfg["entra"] if isinstance(model_cfg.get("entra"), dict) else {}
            identity_config = EntraIdentityConfig.from_dict(entra_cfg, default_scope=SCOPE_AI_AZURE_DEFAULT)
            info.update(
                azure_identity_installed=installed, scope=identity_config.scope, credential_probe="not_run",
                credential_verified=False, logged_in=bool(installed),
                hint=(
                    "azure-identity is installed; live credential validation "
                    "is skipped here. Run `hermes doctor` to verify token acquisition."
                ) if installed else (
                    "azure-identity not installed. Install with: "
                    "pip install azure-identity  (or rely on Hermes' "
                    "lazy-install at first use)."))
        except Exception as exc:
            info["logged_in"] = False
            info["error"] = f"azure-identity check failed: {exc}"
        return info

    try:
        api_key = get_env_value_prefer_dotenv("AZURE_FOUNDRY_API_KEY") or ""
    except Exception:
        api_key = os.getenv("AZURE_FOUNDRY_API_KEY", "")
    info["logged_in"] = has_usable_secret(api_key)
    return info


def _default_api_key_base_url(api_key: str, default: str, env_url: str) -> str:
    return env_url.rstrip("/") if env_url else default


def _copilot_runtime_base_url(api_key: str, default: str, env_url: str) -> str:
    """Copilot's API base comes from the token-exchange response (endpoints.api, proxy-ep fallback),
    authoritative for Enterprise / proxied accounts; falls back to the registry default."""
    base_url = _default_api_key_base_url(api_key, default, env_url)
    try:
        from hermes_cli.copilot_auth import resolve_copilot_token, get_copilot_api_token
        raw_token, _ = resolve_copilot_token()
        if raw_token:
            resolved = (get_copilot_api_token(raw_token)[1] or "").strip()
            if resolved:
                base_url = resolved
    except Exception as exc:
        logger.debug("Copilot base URL resolution fell back to default: %s", exc)
    return base_url


# Providers whose runtime base URL is not simply env-override-or-registry-default:
# ``(api_key, registry_default, env_override) -> base_url``.
_API_KEY_BASE_URL_RESOLVERS: Dict[str, Callable[[str, str, str], str]] = {
    "kimi-coding": _resolve_kimi_base_url,
    "kimi-coding-cn": _resolve_kimi_base_url,
    "zai": _resolve_zai_base_url,
    "copilot": _copilot_runtime_base_url,
    "lmstudio": lambda *a: _normalize_lmstudio_runtime_base_url(_default_api_key_base_url(*a)),
    "actual": lambda *a: normalize_actual_base_url(_default_api_key_base_url(*a))}


def resolve_api_key_provider_credentials(provider_id: str) -> Dict[str, Any]:
    """Resolve API key and base URL for an API-key provider."""
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "api_key":
        raise AuthError(
            f"Provider '{provider_id}' is not an API-key provider.",
            provider=provider_id, code="invalid_provider")

    api_key, key_source = _resolve_api_key_provider_secret(provider_id, pconfig)
    # No-auth LM Studio: a placeholder so runtime / auxiliary_client see the local server as
    # configured. doctor still reports unconfigured because the status path uses the raw secret.
    if not api_key and provider_id == "lmstudio":
        api_key = LMSTUDIO_NOAUTH_PLACEHOLDER
        key_source = key_source or "default"

    env_url = _provider_env_base_url(pconfig)
    resolve_url = _API_KEY_BASE_URL_RESOLVERS.get(provider_id, _default_api_key_base_url)
    base_url = resolve_url(api_key, pconfig.inference_base_url, env_url)
    # An API-key provider must never hand back an empty base URL (a set-but-empty
    # COPILOT_API_BASE_URL or similar env override otherwise wedges chat inference).
    if not _nonempty_str(base_url):
        base_url = pconfig.inference_base_url

    if not api_key and provider_id == "actual" and is_actual_local_base_url(base_url):
        api_key = ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
        key_source = key_source or "local-offline"
    return {
        "provider": provider_id, "api_key": api_key, "base_url": base_url.rstrip("/"),
        "source": key_source or "default"}


def resolve_external_process_provider_credentials(provider_id: str) -> Dict[str, Any]:
    """Resolve runtime details for local subprocess-backed providers."""
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "external_process":
        raise AuthError(
            f"Provider '{provider_id}' is not an external-process provider.",
            provider=provider_id, code="invalid_provider")

    command, args, base_url, resolved_command, command_env_vars = _external_process_spec(pconfig)
    if not resolved_command and not base_url.startswith("acp+tcp://"):
        _hint = " or set " + "/".join(command_env_vars) if command_env_vars else ""
        raise AuthError(
            f"Could not find the '{provider_id}' CLI command "
            f"'{command or '(none configured)'}'. Install it{_hint}.",
            provider=provider_id,
            code="missing_external_process_cli")
    # api_key is a placeholder: the subprocess owns real auth. Keyed on the provider id so each
    # external-process provider gets a distinct value.
    return {
        "provider": provider_id, "api_key": pconfig.id or provider_id,
        "base_url": base_url.rstrip("/"), "command": resolved_command or command, "args": args,
        "source": "process"}


# ── CLI Commands — login / logout ───────────────────────────────────────────────────────────────────

def _update_config_for_provider(
    provider_id: str, inference_base_url: str, default_model: Optional[str] = None) -> Path:
    """Update config.yaml and auth.json to reflect the active provider.

    *default_model*, when given, is written as ``model.default`` in the same step so the gateway
    (which re-reads config per message) can't pick up the new provider before model selection
    finishes and send an OpenRouter-style ``vendor/model`` name to a direct API."""
    with _auth_store_lock():  # so auto-resolution picks this provider
        auth_store = _load_auth_store()
        auth_store["active_provider"] = provider_id
        _save_auth_store(auth_store)

    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    require_readable_config_before_write(config_path)
    config = read_raw_config()
    current_model = config.get("model")
    if isinstance(current_model, dict):
        model_cfg = dict(current_model)
    else:
        model_cfg = {"default": current_model.strip()} if _nonempty_str(current_model) else {}
    model_cfg["provider"] = provider_id
    if inference_base_url and inference_base_url.strip():
        model_cfg["base_url"] = inference_base_url.rstrip("/")
    else:
        model_cfg.pop("base_url", None)  # clear stale base_url when switching providers

    # Built-in providers resolve credentials from env/auth state, not inline model.api_key left
    # over from a previous custom provider.
    from hermes_cli.config import clear_model_endpoint_credentials
    clear_model_endpoint_credentials(model_cfg)

    # An OpenRouter-formatted default like "anthropic/claude-opus-4.6" fails on direct-API
    # providers.
    if default_model:
        cur_default = model_cfg.get("default", "")
        if not cur_default or "/" in cur_default:
            model_cfg["default"] = default_model
    config["model"] = model_cfg
    atomic_yaml_write(config_path, config, sort_keys=False)
    return config_path


def _get_config_provider() -> Optional[str]:
    """Return model.provider from config.yaml, normalized, if present."""
    try:
        config = read_raw_config()
    except Exception:
        return None
    model = config.get("model") if config else None
    provider = model.get("provider") if isinstance(model, dict) else None
    return (provider.strip().lower() or None) if isinstance(provider, str) else None


def _should_reset_config_provider_on_logout(provider_id: Optional[str]) -> bool:
    """True when logout should reset model.provider (a registry provider config.yaml selects)."""
    normalized = (provider_id or "").strip().lower()
    return normalized in PROVIDER_REGISTRY and _get_config_provider() == normalized


def _logout_default_provider_from_config() -> Optional[str]:
    """Fallback logout target when auth.json has no active provider but config.yaml still selects an
    OAuth provider (e.g. openai-codex) — otherwise logout said "No provider is currently logged in"
    and never reset model.provider."""
    provider = _get_config_provider()
    flow = OAUTH_PROVIDER_FLOWS.get(provider or "")
    return provider if flow and flow.logout_from_config else None


def _reset_config_provider() -> Path:
    """Reset config.yaml provider back to auto after logout."""
    config_path = get_config_path()
    if not config_path.exists():
        return config_path
    require_readable_config_before_write(config_path)
    config = read_raw_config()
    if not config:
        return config_path
    model = config.get("model")
    if isinstance(model, dict):
        model["provider"] = "auto"
        if "base_url" in model:
            model["base_url"] = OPENROUTER_BASE_URL
    atomic_yaml_write(config_path, config, sort_keys=False)
    return config_path


def login_command(args) -> None:
    """Deprecated: use 'hermes model' or 'hermes setup' instead."""
    print("The 'hermes login' command has been removed.\nUse 'hermes auth' to manage credentials,\n"
          "'hermes model' to select a provider, or 'hermes setup' for full setup.")
    raise SystemExit(0)


def get_minimax_oauth_auth_status() -> Dict[str, Any]:
    """Return auth status dict for MiniMax OAuth provider."""
    state = get_provider_auth_state("minimax-oauth")
    if not state or not state.get("access_token"):
        return {"logged_in": False, "provider": "minimax-oauth"}
    try:
        token_valid = datetime.fromisoformat(state.get("expires_at", "")).timestamp() > time.time()
    except Exception:
        token_valid = True  # access_token is known non-empty here
    return {
        "logged_in": token_valid, "provider": "minimax-oauth",
        "region": state.get("region", "global"), "expires_at": state.get("expires_at")}


def logout_command(args) -> None:
    """Clear auth state for a provider."""
    provider_id = getattr(args, "provider", None)
    if provider_id and not is_known_auth_provider(provider_id):
        print(f"Unknown provider: {provider_id}")
        raise SystemExit(1)
    target = provider_id or get_active_provider() or _logout_default_provider_from_config()
    if not target:
        print("No provider is currently logged in.")
        return
    should_reset_config = _should_reset_config_provider_on_logout(target)
    provider_name = get_auth_provider_display_name(target)
    if not (clear_provider_auth(target) or should_reset_config):
        print(f"No auth state found for {provider_name}.")
        return
    if should_reset_config:
        _reset_config_provider()
    print(f"Logged out of {provider_name}.")
    if not should_reset_config:
        print("Model provider configuration was unchanged.")
    elif os.getenv("OPENROUTER_API_KEY"):
        print("Hermes will use OpenRouter for inference.")
    else:
        print("Run `hermes model` or configure an API key to use Hermes.")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from http.server import BaseHTTPRequestHandler  # noqa: F401,E402
from http.server import HTTPServer  # noqa: F401,E402
from typing import TYPE_CHECKING  # noqa: F401,E402
import base64  # noqa: F401,E402
import hashlib  # noqa: F401,E402
from urllib.parse import parse_qs  # noqa: F401,E402
import ssl  # noqa: F401,E402
import subprocess  # noqa: F401,E402
import sys  # noqa: F401,E402
from urllib.parse import urlencode  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'CODEX_OAUTH_USER_AGENT': ('hermes_cli.auth_constants', 'CODEX_OAUTH_USER_AGENT'),
    'CODEX_QUOTA_PROBE_MIN_INTERVAL_SECONDS': ('hermes_cli.auth_codex', 'CODEX_QUOTA_PROBE_MIN_INTERVAL_SECONDS'),
    'DEFAULT_SPOTIFY_REDIRECT_URI': ('hermes_cli.auth_constants', 'DEFAULT_SPOTIFY_REDIRECT_URI'),
    'DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS': ('hermes_cli.auth_constants', 'DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS'),
    'MINIMAX_OAUTH_GRANT_TYPE': ('hermes_cli.auth_constants', 'MINIMAX_OAUTH_GRANT_TYPE'),
    'NOUS_INFERENCE_INVOKE_SCOPE': ('hermes_cli.auth_constants', 'NOUS_INFERENCE_INVOKE_SCOPE'),
    'NOUS_SHARED_STORE_FILENAME': ('hermes_cli.auth_nous', 'NOUS_SHARED_STORE_FILENAME'),
    'OAUTH_OVER_SSH_DOCS_URL': ('hermes_cli.auth_constants', 'OAUTH_OVER_SSH_DOCS_URL'),
    'QWEN_OAUTH_CLIENT_ID': ('hermes_cli.auth_constants', 'QWEN_OAUTH_CLIENT_ID'),
    'QWEN_OAUTH_TOKEN_URL': ('hermes_cli.auth_constants', 'QWEN_OAUTH_TOKEN_URL'),
    'SINGLE_USE_OAUTH_SINGLETON_FILES': ('hermes_cli.auth_oauth_grants', 'SINGLE_USE_OAUTH_SINGLETON_FILES'),
    'SPOTIFY_ACCESS_TOKEN_REFRESH_SKEW_SECONDS': ('hermes_cli.auth_constants', 'SPOTIFY_ACCESS_TOKEN_REFRESH_SKEW_SECONDS'),
    'SPOTIFY_DASHBOARD_URL': ('hermes_cli.auth_constants', 'SPOTIFY_DASHBOARD_URL'),
    'XAI_OAUTH_DEVICE_CODE_URL': ('hermes_cli.auth_constants', 'XAI_OAUTH_DEVICE_CODE_URL'),
    'XAI_OAUTH_DISCOVERY_URL': ('hermes_cli.auth_constants', 'XAI_OAUTH_DISCOVERY_URL'),
    'XAI_OAUTH_ISSUER': ('hermes_cli.auth_constants', 'XAI_OAUTH_ISSUER'),
    'refresh_nous_oauth_pure': ('hermes_cli.auth_nous', 'refresh_nous_oauth_pure'),
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
