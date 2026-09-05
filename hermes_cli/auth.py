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
    provider_id: str, entries: List[Dict[str, Any]], *, removed_ids: Optional[Iterable[str]] = None,
) -> Path:
    """Persist one provider's credential pool under auth.json.

    Final disk-boundary sanitizer for borrowed credentials (callers may pass raw dicts). Entries on
    disk but missing from *entries* (added concurrently) are merged back unless in *removed_ids*,
    so a rotation/exhaustion rewrite never drops a concurrent credential."""
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
        merged: List[Dict[str, Any]] = [
            _merge_disk_cooldown_state(e, existing_by_id.get(e.get("id")), provider_id)
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
    "azure-foundry": "_get_azure_foundry_auth_status"}
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

def _spotify_wait_for_callback(
    redirect_uri: str,
    *,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    host, port, path = _spotify_validate_redirect_uri(redirect_uri)
    handler_cls, result = _make_spotify_callback_handler(path)

    class _ReuseHTTPServer(HTTPServer):
        allow_reuse_address = True

    try:
        server = _ReuseHTTPServer((host, port), handler_cls)
    except OSError as exc:
        raise AuthError(
            f"Could not bind Spotify callback server on {host}:{port}: {exc}",
            provider="spotify",
            code="spotify_callback_bind_failed",
        ) from exc

    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    deadline = time.monotonic() + max(5.0, timeout_seconds)
    try:
        while time.monotonic() < deadline:
            if result["code"] or result["error"]:
                return result
            time.sleep(0.1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
    raise AuthError(
        "Spotify authorization timed out waiting for the local callback.",
        provider="spotify",
        code="spotify_callback_timeout",
    )


def _spotify_token_payload_to_state(
    token_payload: Dict[str, Any],
    *,
    client_id: str,
    redirect_uri: str,
    requested_scope: str,
    accounts_base_url: str,
    api_base_url: str,
    previous_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    expires_in = _coerce_ttl_seconds(token_payload.get("expires_in", 0))
    expires_at = datetime.fromtimestamp(now.timestamp() + expires_in, tz=timezone.utc)
    state = dict(previous_state or {})
    state.update({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "accounts_base_url": accounts_base_url,
        "api_base_url": api_base_url,
        "scope": requested_scope,
        "granted_scope": str(token_payload.get("scope") or requested_scope).strip(),
        "token_type": str(token_payload.get("token_type", "Bearer") or "Bearer").strip() or "Bearer",
        "access_token": str(token_payload.get("access_token", "") or "").strip(),
        "refresh_token": str(
            token_payload.get("refresh_token")
            or state.get("refresh_token")
            or ""
        ).strip(),
        "obtained_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "expires_in": expires_in,
        "auth_type": "oauth_pkce",
    })
    return state


def _spotify_exchange_code_for_tokens(
    *,
    client_id: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
    accounts_base_url: str,
    timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    try:
        response = httpx.post(
            f"{accounts_base_url}/api/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "client_id": client_id,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
            timeout=timeout_seconds,
        )
    except Exception as exc:
        raise AuthError(
            f"Spotify token exchange failed: {exc}",
            provider="spotify",
            code="spotify_token_exchange_failed",
        ) from exc

    if response.status_code >= 400:
        detail = response.text.strip()
        raise AuthError(
            "Spotify token exchange failed."
            + (f" Response: {detail}" if detail else ""),
            provider="spotify",
            code="spotify_token_exchange_failed",
        )
    payload = response.json()
    if not isinstance(payload, dict) or not str(payload.get("access_token", "") or "").strip():
        raise AuthError(
            "Spotify token response did not include an access_token.",
            provider="spotify",
            code="spotify_token_exchange_invalid",
        )
    return payload


def _refresh_spotify_oauth_state(
    state: Dict[str, Any],
    *,
    timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    refresh_token = str(state.get("refresh_token", "") or "").strip()
    if not refresh_token:
        raise AuthError(
            "Spotify refresh token missing. Run `hermes auth spotify` again.",
            provider="spotify",
            code="spotify_refresh_token_missing",
            relogin_required=True,
        )

    client_id = _spotify_client_id(state=state)
    accounts_base_url = _spotify_accounts_base_url(state)
    try:
        response = httpx.post(
            f"{accounts_base_url}/api/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
            },
            timeout=timeout_seconds,
        )
    except Exception as exc:
        raise AuthError(
            f"Spotify token refresh failed: {exc}",
            provider="spotify",
            code="spotify_refresh_failed",
        ) from exc

    if response.status_code >= 400:
        detail = response.text.strip()
        raise AuthError(
            "Spotify token refresh failed. Run `hermes auth spotify` again."
            + (f" Response: {detail}" if detail else ""),
            provider="spotify",
            code="spotify_refresh_failed",
            relogin_required=True,
        )

    payload = response.json()
    if not isinstance(payload, dict) or not str(payload.get("access_token", "") or "").strip():
        raise AuthError(
            "Spotify refresh response did not include an access_token.",
            provider="spotify",
            code="spotify_refresh_invalid",
            relogin_required=True,
        )

    return _spotify_token_payload_to_state(
        payload,
        client_id=client_id,
        redirect_uri=_spotify_redirect_uri(state=state),
        requested_scope=str(state.get("scope") or DEFAULT_SPOTIFY_SCOPE),
        accounts_base_url=accounts_base_url,
        api_base_url=_spotify_api_base_url(state),
        previous_state=state,
    )


def resolve_spotify_runtime_credentials(
    *,
    force_refresh: bool = False,
    refresh_if_expiring: bool = True,
    refresh_skew_seconds: int = SPOTIFY_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> Dict[str, Any]:
    with _auth_store_lock():
        auth_store = _load_auth_store()
        state = _load_provider_state(auth_store, "spotify")
        if not state:
            raise AuthError(
                "Spotify is not authenticated. Run `hermes auth spotify` first.",
                provider="spotify",
                code="spotify_auth_missing",
                relogin_required=True,
            )

        should_refresh = bool(force_refresh)
        if not should_refresh and refresh_if_expiring:
            should_refresh = _is_expiring(state.get("expires_at"), refresh_skew_seconds)
        if should_refresh:
            try:
                state = _refresh_spotify_oauth_state(state)
                _store_provider_state(auth_store, "spotify", state, set_active=False)
                _save_auth_store(auth_store)
            except AuthError as exc:
                if exc.relogin_required and state.get("refresh_token"):
                    # Terminal refresh failure — clear dead tokens from auth.json
                    # so subsequent calls fail fast without a network retry.
                    # Mirrors the Nous / xAI-OAuth / Codex-OAuth / MiniMax pattern.
                    for _k in ("access_token", "refresh_token", "expires_at", "expires_in", "obtained_at"):
                        state.pop(_k, None)
                    state["last_auth_error"] = {
                        "provider": "spotify",
                        "code": exc.code or "refresh_failed",
                        "message": str(exc),
                        "reason": "runtime_refresh_failure",
                        "relogin_required": True,
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                    try:
                        _store_provider_state(auth_store, "spotify", state, set_active=False)
                        _save_auth_store(auth_store)
                    except Exception as _save_exc:
                        logger.debug("Spotify OAuth: failed to persist quarantined state: %s", _save_exc)
                raise

    access_token = str(state.get("access_token", "") or "").strip()
    if not access_token:
        raise AuthError(
            "Spotify access token missing. Run `hermes auth spotify` again.",
            provider="spotify",
            code="spotify_access_token_missing",
            relogin_required=True,
        )

    return {
        "provider": "spotify",
        "access_token": access_token,
        "api_key": access_token,
        "token_type": str(state.get("token_type", "Bearer") or "Bearer"),
        "base_url": _spotify_api_base_url(state),
        "scope": str(state.get("granted_scope") or state.get("scope") or "").strip(),
        "client_id": _spotify_client_id(state=state),
        "redirect_uri": _spotify_redirect_uri(state=state),
        "expires_at": state.get("expires_at"),
        "refresh_token": str(state.get("refresh_token", "") or "").strip(),
    }


def get_spotify_auth_status() -> Dict[str, Any]:
    state = get_provider_auth_state("spotify")
    if not state:
        return {"logged_in": False}

    expires_at = state.get("expires_at")
    refresh_token = str(state.get("refresh_token", "") or "").strip()
    return {
        "logged_in": bool(refresh_token or not _is_expiring(expires_at, 0)),
        "auth_type": state.get("auth_type", "oauth_pkce"),
        "client_id": state.get("client_id"),
        "redirect_uri": state.get("redirect_uri"),
        "scope": state.get("granted_scope") or state.get("scope"),
        "expires_at": expires_at,
        "api_base_url": state.get("api_base_url"),
        "has_refresh_token": bool(refresh_token),
    }


def _spotify_interactive_setup(redirect_uri_hint: str) -> str:
    """Walk the user through creating a Spotify developer app, persist the
    resulting client_id to ~/.hermes/.env, and return it.

    Raises SystemExit if the user aborts or submits an empty value.
    """
    from hermes_cli.config import save_env_value

    print()
    print("=" * 70)
    print("Spotify first-time setup")
    print("=" * 70)
    print()
    print("Spotify requires every user to register their own lightweight")
    print("developer app. This takes about two minutes and only has to be")
    print("done once per machine.")
    print()
    print(f"Full guide: {SPOTIFY_DOCS_URL}")
    print()
    print("Steps:")
    print(f"  1. Opening {SPOTIFY_DASHBOARD_URL} in your browser...")
    print("  2. Click 'Create app' and fill in:")
    print("       App name:     anything (e.g. hermes-agent)")
    print("       Description:  anything")
    print(f"       Redirect URI: {redirect_uri_hint}")
    print("       API/SDK:      Web API")
    print("  3. Agree to the terms, click Save.")
    print("  4. Open the app's Settings page and copy the Client ID.")
    print("  5. Paste it below.")
    print()

    if not _is_remote_session():
        try:
            webbrowser.open(SPOTIFY_DASHBOARD_URL)
        except Exception:
            pass

    from hermes_cli.cli_output import line_input

    try:
        raw = line_input("Spotify Client ID: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit("Spotify setup cancelled.")

    if not raw:
        print()
        print(f"No Client ID entered. See {SPOTIFY_DOCS_URL} for the full guide.")
        raise SystemExit("Spotify setup cancelled: empty Client ID.")

    # Persist so subsequent `hermes auth spotify` runs skip the wizard.
    save_env_value("HERMES_SPOTIFY_CLIENT_ID", raw)
    # Only persist the redirect URI if it's non-default, to avoid pinning
    # users to a value the default might later change to.
    if redirect_uri_hint and redirect_uri_hint != DEFAULT_SPOTIFY_REDIRECT_URI:
        save_env_value("HERMES_SPOTIFY_REDIRECT_URI", redirect_uri_hint)

    print()
    print("Saved HERMES_SPOTIFY_CLIENT_ID to ~/.hermes/.env")
    print()
    return raw


def login_spotify_command(args) -> None:
    existing_state = get_provider_auth_state("spotify") or {}

    # Interactive wizard: if no client_id is configured anywhere, walk the
    # user through creating the Spotify developer app instead of crashing
    # with "HERMES_SPOTIFY_CLIENT_ID is required".
    explicit_client_id = getattr(args, "client_id", None)
    try:
        client_id = _spotify_client_id(explicit_client_id, existing_state)
    except AuthError as exc:
        if getattr(exc, "code", "") != "spotify_client_id_missing":
            raise
        client_id = _spotify_interactive_setup(
            redirect_uri_hint=getattr(args, "redirect_uri", None) or DEFAULT_SPOTIFY_REDIRECT_URI,
        )

    redirect_uri = _spotify_redirect_uri(getattr(args, "redirect_uri", None), existing_state)
    scope = _spotify_scope_string(getattr(args, "scope", None) or existing_state.get("scope"))
    accounts_base_url = _spotify_accounts_base_url(existing_state)
    api_base_url = _spotify_api_base_url(existing_state)
    open_browser = not getattr(args, "no_browser", False)

    code_verifier = _spotify_code_verifier()
    code_challenge = _spotify_code_challenge(code_verifier)
    state_nonce = uuid.uuid4().hex
    authorize_url = _spotify_build_authorize_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        state=state_nonce,
        code_challenge=code_challenge,
        accounts_base_url=accounts_base_url,
    )

    print("Starting Spotify PKCE login...")
    print(f"Client ID: {client_id}")
    print(f"Redirect URI: {redirect_uri}")
    print("Make sure this redirect URI is allow-listed in your Spotify app settings.")
    print()
    print("Open this URL to authorize Hermes:")
    print(authorize_url)
    print()
    print(f"Full setup guide: {SPOTIFY_DOCS_URL}")
    print()

    _print_loopback_ssh_hint(redirect_uri, docs_url=SPOTIFY_DOCS_URL)

    if open_browser and not _is_remote_session() and _can_open_graphical_browser():
        try:
            opened = webbrowser.open(authorize_url)
        except Exception:
            opened = False
        if opened:
            print("Browser opened for Spotify authorization.")
        else:
            print("Could not open the browser automatically; use the URL above.")

    callback = _spotify_wait_for_callback(
        redirect_uri,
        timeout_seconds=float(getattr(args, "timeout", None) or 180.0),
    )
    if callback.get("error"):
        detail = callback.get("error_description") or callback["error"]
        raise SystemExit(f"Spotify authorization failed: {detail}")
    if callback.get("state") != state_nonce:
        raise SystemExit("Spotify authorization failed: state mismatch.")

    token_payload = _spotify_exchange_code_for_tokens(
        client_id=client_id,
        code=str(callback.get("code") or ""),
        redirect_uri=redirect_uri,
        code_verifier=code_verifier,
        accounts_base_url=accounts_base_url,
        timeout_seconds=float(getattr(args, "timeout", None) or 20.0),
    )
    spotify_state = _spotify_token_payload_to_state(
        token_payload,
        client_id=client_id,
        redirect_uri=redirect_uri,
        requested_scope=scope,
        accounts_base_url=accounts_base_url,
        api_base_url=api_base_url,
    )

    with _auth_store_lock():
        auth_store = _load_auth_store()
        _store_provider_state(auth_store, "spotify", spotify_state, set_active=False)
        saved_to = _save_auth_store(auth_store)

    print("Spotify login successful!")
    print(f"  Auth state: {saved_to}")
    print("  Provider state saved under providers.spotify")
    print(f"  Docs: {SPOTIFY_DOCS_URL}")

# =============================================================================
# OpenRouter OAuth PKCE
# =============================================================================

OPENROUTER_AUTH_URL = "https://openrouter.ai/auth"
OPENROUTER_TOKEN_EXCHANGE_URL = "https://openrouter.ai/api/v1/auth/keys"
OPENROUTER_CALLBACK_TIMEOUT_SECONDS = 300.0


def _make_openrouter_callback_handler(
    expected_path: str,
) -> tuple[type[BaseHTTPRequestHandler], dict[str, Any]]:
    """Create a callback handler class for the OpenRouter OAuth redirect.

    Returns (handler_class, result_dict) where result_dict is populated
    with ``code`` or ``error`` when the callback is received.
    """
    result: dict[str, Any] = {"code": None, "error": None}

    class _OpenRouterCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            nonlocal result
            parsed_path = urlparse(self.path)
            if parsed_path.path != expected_path:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found.")
                return

            params = parse_qs(parsed_path.query)
            result["code"] = params.get("code", [None])[0]
            result["error"] = params.get("error", [None])[0]

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if result["error"]:
                body = "<html><body><h1>OpenRouter authorization failed.</h1><p>You can close this tab.</p></body></html>"
            else:
                body = "<html><body><h1>OpenRouter authorization received.</h1><p>You can close this tab.</p></body></html>"
            self.wfile.write(body.encode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return _OpenRouterCallbackHandler, result


def _openrouter_wait_for_callback(
    server: HTTPServer,
    result: dict[str, Any],
    *,
    timeout_seconds: float = OPENROUTER_CALLBACK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Wait for the OpenRouter OAuth callback on a pre-bound server.

    Starts the server thread, waits for the redirect, then shuts down.
    Returns the result dict (``code`` or ``error`` key).
    """
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + max(5.0, timeout_seconds)
    try:
        while time.monotonic() < deadline:
            if result["code"] or result["error"]:
                return result
            time.sleep(0.1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
    raise AuthError(
        "OpenRouter authorization timed out waiting for the local callback.",
        provider="openrouter",
        code="openrouter_callback_timeout",
    )


def _openrouter_exchange_code(
    *,
    code: str,
    code_verifier: str,
    code_challenge_method: str = "S256",
    timeout_seconds: float = 20.0,
) -> str:
    """Exchange an authorization code for an OpenRouter API key.

    Returns the API key string.
    """
    payload = {
        "code": code,
        "code_verifier": code_verifier,
        "code_challenge_method": code_challenge_method,
    }
    try:
        response = httpx.post(
            OPENROUTER_TOKEN_EXCHANGE_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        raise AuthError(
            f"OpenRouter token exchange failed: {exc}",
            provider="openrouter",
            code="openrouter_token_exchange_failed",
        ) from exc

    if response.status_code == 400:
        raise AuthError(
            "OpenRouter token exchange failed: invalid code_challenge_method. "
            "Make sure you're using the same method in both steps.",
            provider="openrouter",
            code="openrouter_token_exchange_invalid",
        )
    if response.status_code == 403:
        detail = response.text.strip()
        raise AuthError(
            "OpenRouter authorization failed: the code or code_verifier is invalid, "
            "or the authorization code has expired (codes expire after 10 minutes)."
            + (f" Response: {detail}" if detail else ""),
            provider="openrouter",
            code="openrouter_token_exchange_denied",
        )
    if response.status_code >= 400:
        detail = response.text.strip()
        raise AuthError(
            "OpenRouter token exchange failed."
            + (f" Response: {detail}" if detail else ""),
            provider="openrouter",
            code="openrouter_token_exchange_failed",
        )

    data = response.json()
    if not isinstance(data, dict):
        raise AuthError(
            "OpenRouter token exchange response was not a JSON object.",
            provider="openrouter",
            code="openrouter_token_exchange_invalid",
        )
    key = str(data.get("key") or "").strip()
    if not key:
        raise AuthError(
            "OpenRouter token exchange response did not include a 'key' field.",
            provider="openrouter",
            code="openrouter_token_exchange_invalid",
        )
    return key


def _openrouter_pkce_login(
    *,
    open_browser: bool = True,
    timeout_seconds: float = OPENROUTER_CALLBACK_TIMEOUT_SECONDS,
) -> str:
    """Run the full OpenRouter OAuth PKCE login flow.

    Returns the API key string.
    """
    # 1. Generate PKCE code verifier and challenge.
    code_verifier = _oauth_pkce_code_verifier()
    code_challenge = _oauth_pkce_code_challenge(code_verifier)

    # 2. Build the callback handler and bind the server to a free port.
    handler_cls, result = _make_openrouter_callback_handler("/callback")

    class _ReuseHTTPServer(HTTPServer):
        allow_reuse_address = True

    try:
        server = _ReuseHTTPServer(("127.0.0.1", 0), handler_cls)
    except OSError as exc:
        raise AuthError(
            f"Could not bind OpenRouter callback server: {exc}",
            provider="openrouter",
            code="openrouter_callback_bind_failed",
        ) from exc

    actual_port = server.server_address[1]
    redirect_uri = f"http://127.0.0.1:{actual_port}/callback"
    auth_url = (
        f"{OPENROUTER_AUTH_URL}"
        f"?callback_url={redirect_uri}"
        f"&code_challenge={code_challenge}"
        "&code_challenge_method=S256"
    )

    # 3. Open browser and wait for the redirect.
    if open_browser:
        print(f"Opening browser to:\n  {auth_url}")
        webbrowser.open(auth_url)
    else:
        print(f"Open this URL in your browser:\n  {auth_url}")

    print("\nWaiting for authorization...")
    result = _openrouter_wait_for_callback(
        server, result, timeout_seconds=timeout_seconds
    )

    error = result.get("error")
    if error:
        raise AuthError(
            f"OpenRouter authorization failed: {error}",
            provider="openrouter",
            code="openrouter_auth_denied",
        )
    code = str(result.get("code") or "").strip()
    if not code:
        raise AuthError(
            "OpenRouter authorization did not return a code.",
            provider="openrouter",
            code="openrouter_auth_no_code",
        )

    # 4. Exchange the code for an API key (separate, shorter timeout).
    print("Exchanging authorization code for API key...")
    api_key = _openrouter_exchange_code(
        code=code,
        code_verifier=code_verifier,
        code_challenge_method="S256",
        timeout_seconds=20.0,
    )
    print("API key obtained successfully!\n")
    return api_key


# =============================================================================
# SSH / remote session detection
# =============================================================================

def _is_remote_session() -> bool:
    """Detect environments where loopback OAuth can't reach the local browser.

    Historically only SSH was checked, but #26923 surfaced that
    **browser-only remote consoles** (GCP Cloud Shell, GitHub
    Codespaces, AWS EC2 Instance Connect, Gitpod, Replit, etc.) hit
    the exact same problem — the user has a browser on their laptop
    but the loopback listener is bound on the remote VM that the
    laptop's browser can't reach.  These environments typically don't
    set ``SSH_CLIENT`` / ``SSH_TTY``, so the SSH-only check left
    them with no guidance and no fallback.
    """
    if os.getenv("SSH_CLIENT") or os.getenv("SSH_TTY"):
        return True
    # Browser-only remote IDEs / cloud shells.  Keep this list narrow
    # (well-known, documented env vars set by the host platform) so
    # we don't falsely trip on a developer's local shell.
    for var in (
        "CLOUD_SHELL",         # GCP Cloud Shell
        "CODESPACES",          # GitHub Codespaces
        "CODESPACE_NAME",      # GitHub Codespaces (alt)
        "GITPOD_WORKSPACE_ID", # Gitpod
        "REPL_ID",             # Replit
        "STACKBLITZ",          # StackBlitz
    ):
        if os.getenv(var):
            return True
    return False


# Console/text-mode browsers that ``webbrowser`` will happily launch INSIDE
# the terminal.  Opening one of these is worse than not opening anything —
# it hijacks the user's TTY with an unusable text browser (the xAI OAuth
# "Account Management" page rendered in w3m, reported May 2026) instead of
# letting them copy the URL to a real browser.  When the resolved browser is
# one of these we refuse to auto-open and fall back to the print-the-URL
# path, same as a remote session.
_CONSOLE_BROWSER_NAMES: FrozenSet[str] = frozenset(
    {
        "w3m",
        "lynx",
        "links",
        "links2",
        "elinks",
        "www-browser",
        "browsh",  # TUI browser — still hijacks the terminal
    }
)


def _can_open_graphical_browser() -> bool:
    """Return True only when a *graphical* browser is likely to open.

    ``webbrowser.open()`` resolves to whatever the platform offers, and on a
    headless / CLI-only Linux box with no GUI browser installed that is often
    a text-mode browser (w3m/lynx/links) which launches inside the terminal
    and takes over the user's session.  This guard distinguishes "a real
    windowed browser will pop up" from "a console browser will hijack the
    TTY", so callers can fall back to printing the URL instead.

    Heuristics:
      * Respect ``$BROWSER`` — if it names a known console browser, refuse.
      * On Linux, require a display server (``$DISPLAY`` / ``$WAYLAND_DISPLAY``)
        unless ``$BROWSER`` points at something graphical; no display server
        almost always means no GUI browser.
      * Ask ``webbrowser.get()`` what it resolved to and refuse when the
        underlying command is a known console browser.
      * macOS and Windows always have a usable default GUI browser.
    """
    import webbrowser as _webbrowser

    def _names_console_browser(value: str) -> bool:
        token = value.strip().split()[0] if value.strip() else ""
        base = os.path.basename(token).lower()
        return base in _CONSOLE_BROWSER_NAMES

    browser_env = os.environ.get("BROWSER", "")
    if browser_env and _names_console_browser(browser_env):
        return False

    if sys.platform.startswith("linux"):
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        # An explicit graphical $BROWSER can work without $DISPLAY in odd
        # setups, but a console $BROWSER already returned False above, so the
        # only way to reach here with a $BROWSER set is a graphical one.
        if not has_display and not browser_env:
            return False

    try:
        controller = _webbrowser.get()
    except Exception:
        # No browser resolvable at all → definitely don't auto-open.
        return False

    candidate = (
        getattr(controller, "name", "")
        or getattr(controller, "basename", "")
        or ""
    )
    if candidate and _names_console_browser(candidate):
        return False

    return True


def _ssh_user_at_host() -> str:
    """Return best-effort 'user@hostname' for the SSH tunnel hint command.

    Falls back to placeholder tokens when the values cannot be determined so
    the hint is always syntactically valid even if not copy-pasteable.
    """
    try:
        import socket as _socket
        hostname = _socket.gethostname() or "<this-host>"
    except OSError:
        hostname = "<this-host>"
    user = os.getenv("USER") or os.getenv("LOGNAME") or "<user>"
    return f"{user}@{hostname}"


def _print_loopback_ssh_hint(redirect_uri: str, *, docs_url: str | None = None) -> None:
    """Print an SSH tunnel hint when running a loopback-redirect OAuth flow on a
    remote host. The auth server (Spotify, MCP servers, ...) will redirect the
    user's browser to ``127.0.0.1:<port>/callback``. If the browser is on a
    different machine than the loopback listener (the usual SSH case), the
    redirect can't reach the listener without a local port forward.

    The hint is best-effort: silent if we don't think we're remote, or if we
    can't parse a host/port out of the redirect URI.

    Pass ``docs_url`` for a provider-specific guide; the generic OAuth-over-SSH
    guide is always shown after it.
    """
    if not _is_remote_session():
        return
    try:
        parsed = urlparse(redirect_uri)
    except Exception:
        return
    host = parsed.hostname or ""
    port = parsed.port
    if host not in {"127.0.0.1", "::1", "localhost"} or not port:
        return
    divider = "-" * 60
    print()
    print(divider)
    print("Remote session detected — SSH tunnel required")
    print(divider)
    print(f"Hermes is waiting for the OAuth callback on {redirect_uri}")
    print("but your browser is on a different machine. Run this command")
    print("in a NEW terminal on your local machine BEFORE opening the URL:")
    print()
    print(f"  ssh -N -L {port}:127.0.0.1:{port} {_ssh_user_at_host()}")
    print()
    print("Then open the authorize URL above in your local browser.")
    if docs_url:
        print(f"Provider docs:      {docs_url}")
    print(f"SSH/jump-box guide: {OAUTH_OVER_SSH_DOCS_URL}")
    print(divider)
    print()


# =============================================================================
# OpenAI Codex auth — tokens stored in ~/.hermes/auth.json (not ~/.codex/)
#
# Hermes maintains its own Codex OAuth session separate from the Codex CLI
# and VS Code extension. This prevents refresh token rotation conflicts
# where one app's refresh invalidates the other's session.
# =============================================================================

def _read_codex_tokens(*, _lock: bool = True) -> Dict[str, Any]:
    """Read Codex OAuth tokens from Hermes auth store (~/.hermes/auth.json).
    
    Returns dict with 'tokens' (access_token, refresh_token) and 'last_refresh'.
    Raises AuthError if no Codex tokens are stored.
    """
    if _lock:
        with _auth_store_lock():
            auth_store = _load_auth_store()
    else:
        auth_store = _load_auth_store()
    state = _load_provider_state(auth_store, "openai-codex")
    if not state:
        raise AuthError(
            "No Codex credentials stored. Run `hermes auth` to authenticate.",
            provider="openai-codex",
            code="codex_auth_missing",
            relogin_required=True,
        )
    tokens = state.get("tokens")
    if not isinstance(tokens, dict):
        raise AuthError(
            "Codex auth state is missing tokens. Run `hermes auth` to re-authenticate.",
            provider="openai-codex",
            code="codex_auth_invalid_shape",
            relogin_required=True,
        )
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise AuthError(
            "Codex auth is missing access_token. Run `hermes auth` to re-authenticate.",
            provider="openai-codex",
            code="codex_auth_missing_access_token",
            relogin_required=True,
        )
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise AuthError(
            "Codex auth is missing refresh_token. Run `hermes auth` to re-authenticate.",
            provider="openai-codex",
            code="codex_auth_missing_refresh_token",
            relogin_required=True,
        )
    return {
        "tokens": tokens,
        "last_refresh": state.get("last_refresh"),
    }


def _sync_codex_pool_entries(
    auth_store: Dict[str, Any],
    tokens: Dict[str, str],
    last_refresh: Optional[str],
    previous_singleton_tokens: Optional[Dict[str, str]] = None,
) -> None:
    """Mirror a fresh Codex re-auth into the credential_pool OAuth entries.

    The runtime selects credentials from ``credential_pool.openai-codex``, not
    from ``providers.openai-codex.tokens``.  A re-auth invalidates the prior
    OAuth pair server-side, but pool entries keep holding the now-consumed
    refresh token plus any stale error markers — so the next request spends a
    dead token and gets a 401 ``token_invalidated``.

    What gets refreshed:

    * ``device_code`` — the singleton-seeded entry written by the device-code
      OAuth flow when the user logged in via ``hermes setup`` / the model
      picker.  Always synced with the fresh tokens.
    * ``manual:device_code`` — entries created by ``hermes auth add openai-codex``
      that use the same device-code OAuth mechanism.  ONLY synced if the
      entry's existing access_token matches the *previous* singleton
      access_token (i.e. the entry is a legacy singleton-alias from the
      #33000 workaround era).  Manual entries whose tokens never matched the
      singleton represent INDEPENDENT accounts added via
      ``hermes auth add openai-codex`` and must not be overwritten by a
      re-auth that targeted a different account (regression for #39236).

      The original #33538 fix refreshed every ``manual:device_code`` entry
      unconditionally.  That worked when ``manual:device_code`` only meant
      "legacy alias of the singleton", but the same source string is now
      also produced by independent-account additions, and the broad sync
      silently clobbered distinct accounts with the latest-authenticated
      token pair.  The access_token-match check distinguishes the two cases
      without changing the source-string contract.

    What does NOT get refreshed:

    * ``manual:api_key`` and any other non-device-code manual sources — those
      are independent credentials (an explicit API key, a different ChatGPT
      account, etc.) and must not be overwritten by a single re-auth.
    * ``manual:device_code`` entries whose access_token does NOT match the
      previous singleton — see above; these are independent accounts.

    Error markers (``last_status``, ``last_error_*``) are cleared ONLY on
    entries that actually had their tokens rewritten by this re-auth.
    Independent entries keep their own error state (their 401/429 markers
    belong to that account's own auth flow, not this re-auth).
    """
    access_token = tokens.get("access_token")
    if not access_token:
        return
    refresh_token = tokens.get("refresh_token")
    pool = auth_store.get("credential_pool")
    if not isinstance(pool, dict):
        return
    entries = pool.get("openai-codex")
    if not isinstance(entries, list):
        return
    # Previous singleton access_token (before this re-auth overwrote it) —
    # used to distinguish legacy singleton-aliases from independent accounts.
    # When None or empty, no manual entry can be treated as an alias (which
    # is the right default for first-ever-save or a freshly initialized
    # auth.json).
    prev_at = None
    if isinstance(previous_singleton_tokens, dict):
        prev_at = previous_singleton_tokens.get("access_token") or None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source = entry.get("source")
        if source == "device_code":
            # Singleton-seeded mirror — always refresh.
            refresh_this_entry = True
        elif source == "manual:device_code":
            # Refresh only if this entry's existing access_token matches the
            # previous singleton access_token (i.e. it is a true alias of the
            # singleton from the #33000 workaround era).  An entry with its
            # own distinct token material is an independent account and must
            # be left alone (#39236).
            refresh_this_entry = bool(
                prev_at and entry.get("access_token") == prev_at
            )
        else:
            # ``manual:api_key`` and any future non-device-code sources.
            refresh_this_entry = False
        if not refresh_this_entry:
            continue
        entry["access_token"] = access_token
        if refresh_token:
            entry["refresh_token"] = refresh_token
        if last_refresh:
            entry["last_refresh"] = last_refresh
        entry["last_status"] = None
        entry["last_status_at"] = None
        entry["last_error_code"] = None
        entry["last_error_reason"] = None
        entry["last_error_message"] = None
        entry["last_error_reset_at"] = None


def _save_codex_tokens(tokens: Dict[str, str], last_refresh: str = None, label: str = None) -> None:
    """Save Codex OAuth tokens to Hermes auth store (~/.hermes/auth.json)."""
    if last_refresh is None:
        last_refresh = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with _auth_store_lock():
        auth_store = _load_auth_store()
        state = _load_provider_state(auth_store, "openai-codex") or {}
        # Capture the previous singleton tokens BEFORE overwriting them.  The
        # pool-sync step uses this to distinguish legacy singleton-aliases
        # (which should be refreshed) from independent accounts that
        # ``hermes auth add openai-codex`` created (which must not be
        # overwritten — see #39236).
        previous_singleton_tokens = state.get("tokens") if isinstance(state.get("tokens"), dict) else None
        state["tokens"] = tokens
        state["last_refresh"] = last_refresh
        state["auth_mode"] = "chatgpt"
        if label and str(label).strip():
            state["label"] = str(label).strip()
        _save_provider_state(auth_store, "openai-codex", state)
        _sync_codex_pool_entries(
            auth_store,
            tokens,
            last_refresh,
            previous_singleton_tokens=previous_singleton_tokens,
        )
        _save_auth_store(auth_store)


def _recover_codex_tokens_from_cli(reason: str) -> Optional[Dict[str, str]]:
    """Adopt a valid Codex CLI token pair into Hermes auth, if available."""
    imported = _import_codex_cli_tokens()
    # Require BOTH tokens before adopting: persisting a payload without a
    # usable refresh_token would only break the next refresh cycle.
    if not (
        imported
        and str(imported.get("access_token", "") or "").strip()
        and str(imported.get("refresh_token", "") or "").strip()
    ):
        return None
    logger.info("Codex auth recovered from Codex CLI auth.json (%s).", reason)
    _save_codex_tokens(imported)
    return dict(imported)


def _codex_http_client(**kwargs: Any) -> "httpx.Client":
    """Build an ``httpx.Client`` for Codex OAuth/probe endpoints with racing.

    Same broken-IPv6 failure mode as the chat transport (#13834): a host that
    advertises AAAA records but blackholes IPv6 makes each serial connect
    attempt eat the full connect timeout before IPv4 is tried, so token
    refresh / device login / usage probes time out where the official Codex
    CLI (which races families per RFC 8305) works. Install the same
    Happy-Eyeballs sync backend #94388 added for the chat transport.

    Best-effort: if the racing backend can't be installed (unexpected
    httpx/httpcore internals, mocked client in tests), the client still works
    with the default serial connect behavior. Proxy-backed transports are
    intentionally left on the default backend (the TCP connect goes to the
    proxy, not to auth.openai.com/chatgpt.com).
    """
    client = httpx.Client(**kwargs)
    try:
        from agent.process_bootstrap import enable_happy_eyeballs_on_client

        enable_happy_eyeballs_on_client(client)
    except Exception:
        pass
    return client


def refresh_codex_oauth_pure(
    access_token: str,
    refresh_token: str,
    *,
    timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    """Refresh Codex OAuth tokens without mutating Hermes auth state."""
    del access_token  # Access token is only used by callers to decide whether to refresh.
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise AuthError(
            "Codex auth is missing refresh_token. Run `hermes auth` to re-authenticate.",
            provider="openai-codex",
            code="codex_auth_missing_refresh_token",
            relogin_required=True,
        )

    timeout = httpx.Timeout(max(5.0, float(timeout_seconds)))
    with _codex_http_client(
        timeout=timeout,
        headers={
            "Accept": "application/json",
            "User-Agent": CODEX_OAUTH_USER_AGENT,
        },
    ) as client:
        response = client.post(
            CODEX_OAUTH_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_OAUTH_CLIENT_ID,
            },
        )

    if response.status_code == 429:
        # Upstream rate-limit / usage-quota exhaustion on the token endpoint.
        # The stored refresh token is still valid here — re-authenticating
        # cannot lift a quota cap. Classify distinctly from auth failures so
        # callers surface a "retry later" notice instead of a misleading
        # "run hermes auth" prompt (see issue #32790).
        retry_after = _parse_retry_after_seconds(getattr(response, "headers", None))
        if retry_after is not None:
            message = (
                f"Codex provider quota exhausted (429); retry after {retry_after}s. "
                "Credentials are still valid."
            )
        else:
            message = (
                "Codex provider quota exhausted (429). Credentials are still valid; "
                "retry after the usage limit resets."
            )
        raise AuthError(
            message,
            provider="openai-codex",
            code=CODEX_RATE_LIMITED_CODE,
            relogin_required=False,
        )

    if response.status_code != 200:
        code = "codex_refresh_failed"
        message = f"Codex token refresh failed with status {response.status_code}."
        relogin_required = False
        try:
            err = response.json()
            if isinstance(err, dict):
                err_obj = err.get("error")
                # OpenAI shape: {"error": {"code": "...", "message": "...", "type": "..."}}
                if isinstance(err_obj, dict):
                    nested_code = err_obj.get("code") or err_obj.get("type")
                    if isinstance(nested_code, str) and nested_code.strip():
                        code = nested_code.strip()
                    nested_msg = err_obj.get("message")
                    if isinstance(nested_msg, str) and nested_msg.strip():
                        message = f"Codex token refresh failed: {nested_msg.strip()}"
                # OAuth spec shape: {"error": "code_str", "error_description": "..."}
                elif isinstance(err_obj, str) and err_obj.strip():
                    code = err_obj.strip()
                    err_desc = err.get("error_description") or err.get("message")
                    if isinstance(err_desc, str) and err_desc.strip():
                        message = f"Codex token refresh failed: {err_desc.strip()}"
        except Exception:
            pass
        if code in {"invalid_grant", "invalid_token", "invalid_request"}:
            relogin_required = True
        if code == "refresh_token_reused":
            message = (
                "Codex refresh token was already consumed by another client "
                "(e.g. Codex CLI or VS Code extension). "
                "Run `codex` in your terminal to generate fresh tokens, "
                "then run `hermes auth` to re-authenticate."
            )
            relogin_required = True
        # A 401/403 from the token endpoint always means the refresh token
        # is invalid/expired — force relogin even if the body error code
        # wasn't one of the known strings above.
        if response.status_code in {401, 403} and not relogin_required:
            relogin_required = True
        raise AuthError(
            message,
            provider="openai-codex",
            code=code,
            relogin_required=relogin_required,
        )

    try:
        refresh_payload = response.json()
    except Exception as exc:
        raise AuthError(
            "Codex token refresh returned invalid JSON.",
            provider="openai-codex",
            code="codex_refresh_invalid_json",
            relogin_required=True,
        ) from exc

    refreshed_access = refresh_payload.get("access_token")
    if not isinstance(refreshed_access, str) or not refreshed_access.strip():
        raise AuthError(
            "Codex token refresh response was missing access_token.",
            provider="openai-codex",
            code="codex_refresh_missing_access_token",
            relogin_required=True,
        )

    updated = {
        "access_token": refreshed_access.strip(),
        "refresh_token": refresh_token.strip(),
        "last_refresh": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    next_refresh = refresh_payload.get("refresh_token")
    if isinstance(next_refresh, str) and next_refresh.strip():
        updated["refresh_token"] = next_refresh.strip()
    return updated


def _refresh_codex_auth_tokens(
    tokens: Dict[str, str],
    timeout_seconds: float,
) -> Dict[str, str]:
    """Refresh Codex access token using the refresh token.
    
    Saves the new tokens to Hermes auth store automatically.
    """
    try:
        refreshed = refresh_codex_oauth_pure(
            str(tokens.get("access_token", "") or ""),
            str(tokens.get("refresh_token", "") or ""),
            timeout_seconds=timeout_seconds,
        )
    except AuthError as exc:
        # Self-heal cross-store refresh_token rotation. Hermes keeps its OWN
        # Codex OAuth token (per profile + top-level), separate from the Codex
        # CLI's ~/.codex/auth.json. OAuth refresh_tokens are single-use, so when
        # the Codex CLI (or another Hermes process) rotates the shared token,
        # this frozen copy's refresh_token goes stale and the refresh fails with
        # a relogin-required error (invalid_grant / refresh_token_reused / 401).
        # Before surfacing that as a hard 401 to the turn, adopt the canonical
        # fresh token from ~/.codex/auth.json (the Codex CLI keeps it current) so
        # idle profiles / desktop sessions recover automatically instead of
        # 401'ing until a manual re-auth. Transient failures (e.g. 429 quota)
        # keep relogin_required=False — the stored token is still valid there, so
        # we never self-heal those and re-raise unchanged.
        if not getattr(exc, "relogin_required", False):
            raise
        imported = _recover_codex_tokens_from_cli(
            f"refresh_token rejected: {getattr(exc, 'code', None) or 'auth_error'}"
        )
        if not imported:
            raise
        return imported

    updated_tokens = dict(tokens)
    updated_tokens["access_token"] = refreshed["access_token"]
    updated_tokens["refresh_token"] = refreshed["refresh_token"]

    _save_codex_tokens(updated_tokens)
    return updated_tokens


def _import_codex_cli_tokens() -> Optional[Dict[str, str]]:
    """Try to read tokens from ~/.codex/auth.json (Codex CLI shared file).
    
    Returns tokens dict if valid and not expired, None otherwise.
    Does NOT write to the shared file.
    """
    codex_home = os.getenv("CODEX_HOME", "").strip()
    if not codex_home:
        codex_home = str(Path.home() / ".codex")
    auth_path = Path(codex_home).expanduser() / "auth.json"
    if not auth_path.is_file():
        return None
    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8-sig"))
        tokens = payload.get("tokens")
        if not isinstance(tokens, dict):
            return None
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        if not access_token or not refresh_token:
            return None
        # Reject expired tokens — importing stale tokens from ~/.codex/
        # that can't be refreshed leaves the user stuck with "Login successful!"
        # but no working credentials.
        if _codex_access_token_is_expiring(access_token, 0):
            logger.debug(
                "Codex CLI tokens at %s are expired — skipping import.", auth_path,
            )
            return None
        return dict(tokens)
    except Exception:
        return None


def resolve_codex_runtime_credentials(
    *,
    force_refresh: bool = False,
    refresh_if_expiring: bool = True,
    refresh_skew_seconds: int = CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> Dict[str, Any]:
    """Resolve runtime credentials from Hermes's own Codex token store.

    Falls back to the credential pool when the singleton (``providers.openai-codex.tokens``)
    has no usable access_token but the pool (``credential_pool.openai-codex``) does. This
    closes the divergence between the chat path (singleton-only via this function) and
    the auxiliary path (pool-first via ``_read_codex_access_token``). Without this
    fallback, a user whose tokens live only in the pool — for example after a manual
    pool seed, a partial re-auth, or pool-only restoration from a backup — gets a bare
    HTTP 401 ``Missing Authentication header`` from the wire instead of a usable
    credential. See issue #32992.
    """
    read_error: Optional[AuthError] = None
    try:
        data = _read_codex_tokens()
    except AuthError as exc:
        read_error = exc
        if getattr(exc, "relogin_required", False) and getattr(exc, "code", None) in {
            "codex_auth_missing_access_token",
            "codex_auth_missing_refresh_token",
            "codex_auth_invalid_shape",
        }:
            imported = _recover_codex_tokens_from_cli(str(getattr(exc, "code", None) or "auth_error"))
            if imported:
                data = {"tokens": imported, "last_refresh": imported.get("last_refresh")}
            else:
                data = None
        else:
            data = None

    if data is None:
        pool_token = _pool_codex_access_token()
        if pool_token:
            base_url = (
                os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/")
                or DEFAULT_CODEX_BASE_URL
            )
            return {
                "provider": "openai-codex",
                "base_url": base_url,
                "api_key": pool_token,
                "source": "credential_pool",
                "last_refresh": None,
                "auth_mode": "chatgpt",
            }
        pool_rate_limit = _codex_pool_rate_limit_status()
        if pool_rate_limit:
            # Before surfacing the persisted cooldown, ask the Codex usage
            # endpoint whether the quota actually reset early (banked reset
            # redeemed, plan upgraded, window reset upstream).  The persisted
            # ``last_error_reset_at`` can be days in the future while the
            # account is already usable again — see issue #43747.
            stale_token = str(pool_rate_limit.get("access_token") or "").strip()
            if stale_token and _probe_codex_quota_restored(
                stale_token,
                base_url=pool_rate_limit.get("base_url"),
            ):
                logger.info(
                    "Codex quota restored upstream — clearing stale pool cooldown(s)."
                )
                clear_codex_pool_quota_cooldowns()
                pool_token = _pool_codex_access_token()
                if pool_token:
                    base_url = (
                        os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/")
                        or DEFAULT_CODEX_BASE_URL
                    )
                    return {
                        "provider": "openai-codex",
                        "base_url": base_url,
                        "api_key": pool_token,
                        "source": "credential_pool",
                        "last_refresh": None,
                        "auth_mode": "chatgpt",
                    }
            reset_at = pool_rate_limit.get("reset_at")
            if isinstance(reset_at, (int, float)) and reset_at > time.time():
                remaining = int(reset_at - time.time())
                message = (
                    f"Codex provider quota exhausted (429); retry after {remaining}s. "
                    "Credentials are still valid."
                )
            else:
                message = (
                    "Codex provider quota exhausted (429). Credentials are still valid; "
                    "retry after the usage limit resets."
                )
            raise AuthError(
                message,
                provider="openai-codex",
                code=CODEX_RATE_LIMITED_CODE,
                relogin_required=False,
            )
        if read_error is not None:
            raise read_error
        raise AuthError(
            "No Codex credentials stored. Run `hermes auth` to authenticate.",
            provider="openai-codex",
            code="codex_auth_missing",
            relogin_required=True,
        )

    tokens = dict(data["tokens"])
    access_token = str(tokens.get("access_token", "") or "").strip()
    refresh_timeout_seconds = env_float("HERMES_CODEX_REFRESH_TIMEOUT_SECONDS", 20)

    should_refresh = bool(force_refresh)
    if (not should_refresh) and refresh_if_expiring:
        should_refresh = _codex_access_token_is_expiring(access_token, refresh_skew_seconds)
    if should_refresh:
        # Re-read under lock to avoid racing with other Hermes processes
        with _auth_store_lock(timeout_seconds=max(float(AUTH_LOCK_TIMEOUT_SECONDS), refresh_timeout_seconds + 5.0)):
            data = _read_codex_tokens(_lock=False)
            tokens = dict(data["tokens"])
            access_token = str(tokens.get("access_token", "") or "").strip()

            should_refresh = bool(force_refresh)
            if (not should_refresh) and refresh_if_expiring:
                should_refresh = _codex_access_token_is_expiring(access_token, refresh_skew_seconds)

            if should_refresh:
                tokens = _refresh_codex_auth_tokens(tokens, refresh_timeout_seconds)
                access_token = str(tokens.get("access_token", "") or "").strip()

    base_url = (
        os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/")
        or DEFAULT_CODEX_BASE_URL
    )

    return {
        "provider": "openai-codex",
        "base_url": base_url,
        "api_key": access_token,
        "source": "hermes-auth-store",
        "last_refresh": data.get("last_refresh"),
        "auth_mode": "chatgpt",
    }


def _is_codex_rate_limit_shaped(
    code: Any,
    reason: Any,
    message: Any,
) -> bool:
    """True when persisted pool-entry error metadata describes a 429/quota stop."""
    reason_l = str(reason or "").lower()
    message_l = str(message or "").lower()
    return (
        code == 429
        or "rate_limit" in reason_l
        or "usage_limit" in reason_l
        or "quota" in reason_l
        or "rate limit" in message_l
        or "usage limit" in message_l
        or "quota" in message_l
    )


# Throttle for the live Codex quota probe below.  The probe runs on the hot
# credential-selection path while the pool is exhausted, so without a floor a
# busy gateway would hammer the usage endpoint on every model/auxiliary call.
CODEX_QUOTA_PROBE_MIN_INTERVAL_SECONDS = 300  # 5 minutes
_codex_quota_probe_cache: Dict[str, Tuple[float, Optional[bool]]] = {}
_codex_quota_probe_lock = threading.Lock()


def _codex_usage_probe_url(base_url: Optional[str]) -> str:
    """Resolve the Codex usage endpoint for a probe.

    Mirrors the Codex CLI's PathStyle split (codex-rs backend-client, same
    logic as ``agent.account_usage._codex_backend_urls``): base URLs
    containing ``/backend-api`` use the ChatGPT ``/wham/usage`` path;
    everything else uses ``/api/codex/usage``.  Kept local so this low-level
    auth module doesn't import the auxiliary account-usage module.
    """
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        normalized = (
            os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/")
            or DEFAULT_CODEX_BASE_URL
        )
    if normalized.endswith("/codex"):
        normalized = normalized[: -len("/codex")]
    prefix = normalized + ("/wham" if "/backend-api" in normalized else "/api/codex")
    return prefix + "/usage"


def _probe_codex_quota_restored(
    access_token: Any,
    *,
    base_url: Optional[str] = None,
    min_interval_seconds: float = CODEX_QUOTA_PROBE_MIN_INTERVAL_SECONDS,
) -> Optional[bool]:
    """Ask the Codex usage endpoint whether this account's quota is usable again.

    Hermes persists a Codex 429's ``reset_at`` locally and freezes the
    credential until it elapses — but the upstream window can reopen EARLY
    (the user redeems a banked rate-limit reset via the Codex CLI/ChatGPT UI,
    upgrades their plan, or OpenAI resets the window).  This probe detects
    that: it GETs the same ``/usage`` endpoint the Codex CLI uses and checks
    the reported windows.

    Returns:
      * ``True``  — every reported rate-limit window is below 100% used;
        the account can serve requests again and stale local cooldowns
        should be lifted.
      * ``False`` — a window is still fully used (or the probe itself 429'd);
        keep the cooldown.
      * ``None``  — indeterminate (no token, network error, unexpected
        payload/status); keep the cooldown.

    Probes are throttled per access token (module-local cache) so the hot
    selection path can fire this freely.
    """
    token = str(access_token or "").strip()
    if not token:
        return None
    # Real Codex access tokens are JWTs. Refusing to probe non-JWT tokens
    # avoids pointless network calls for corrupt/placeholder entries (and
    # keeps hermetic test fixtures with dummy tokens offline).
    if not _decode_jwt_claims(token):
        return None
    cache_key = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    now = time.monotonic()
    with _codex_quota_probe_lock:
        cached = _codex_quota_probe_cache.get(cache_key)
        if cached is not None and (now - cached[0]) < min_interval_seconds:
            return cached[1]
        # Reserve the slot immediately so concurrent selectors don't stampede
        # the endpoint while this probe is in flight.
        _codex_quota_probe_cache[cache_key] = (now, None)

    result: Optional[bool] = None
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "codex-cli",
        }
        # Best-effort ChatGPT-Account-Id from the JWT (the backend requires it
        # for some account shapes; harmless to omit for others).
        claims = _decode_jwt_claims(token)
        account_id = (
            claims.get("https://api.openai.com/auth", {}).get("chatgpt_account_id")
            if isinstance(claims.get("https://api.openai.com/auth"), dict)
            else None
        )
        if isinstance(account_id, str) and account_id.strip():
            headers["ChatGPT-Account-Id"] = account_id.strip()
        with _codex_http_client(timeout=10.0) as client:
            response = client.get(_codex_usage_probe_url(base_url), headers=headers)
        if response.status_code == 200:
            payload = response.json() or {}
            rate_limit = payload.get("rate_limit") or {}
            worst_used: Optional[float] = None
            for key in ("primary_window", "secondary_window"):
                used = (rate_limit.get(key) or {}).get("used_percent")
                if isinstance(used, (int, float)):
                    worst_used = max(worst_used or 0.0, float(used))
            if worst_used is not None:
                result = worst_used < 100.0
        elif response.status_code == 429:
            result = False
    except Exception:
        logger.debug("Codex quota probe failed", exc_info=True)
        result = None

    with _codex_quota_probe_lock:
        _codex_quota_probe_cache[cache_key] = (now, result)
    return result


def clear_codex_pool_quota_cooldowns(access_token: Optional[str] = None) -> int:
    """Clear rate-limit cooldowns on persisted openai-codex pool entries.

    Called after the upstream quota is KNOWN to be restored (a successful
    ``/usage reset`` redemption, or a positive live probe) so auth.json stops
    freezing credentials behind a stale ``last_error_reset_at``.  Only lifts
    ``exhausted`` entries whose error metadata is 429/quota-shaped — DEAD
    (terminal auth) entries and non-rate-limit failures are untouched.

    When *access_token* is given, only the matching entry is cleared;
    otherwise every rate-limited entry clears (a redeemed banked reset
    restores the whole account, and any entry that is genuinely still
    exhausted just re-freezes with fresh metadata on its next 429).

    Returns the number of entries cleared.
    """
    cleared = 0
    try:
        with _auth_store_lock():
            auth_store = _load_auth_store()
            pool = auth_store.get("credential_pool")
            entries = pool.get("openai-codex") if isinstance(pool, dict) else None
            if not isinstance(entries, list):
                return 0
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if entry.get("last_status") != "exhausted":
                    continue
                if access_token and str(entry.get("access_token") or "") != access_token:
                    continue
                if not _is_codex_rate_limit_shaped(
                    entry.get("last_error_code"),
                    entry.get("last_error_reason"),
                    entry.get("last_error_message"),
                ):
                    continue
                entry["last_status"] = None
                entry["last_status_at"] = None
                entry["last_error_code"] = None
                entry["last_error_reason"] = None
                entry["last_error_message"] = None
                entry["last_error_reset_at"] = None
                cleared += 1
            if cleared:
                _save_auth_store(auth_store)
    except Exception:
        logger.debug("Failed to clear Codex pool quota cooldowns", exc_info=True)
    return cleared


def _codex_pool_rate_limit_status() -> Optional[Dict[str, Any]]:
    """Return metadata for a pool-only Codex credential in quota cooldown."""
    def _parse_reset_at(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            numeric = float(value)
            if numeric <= 0:
                return None
            return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None
            try:
                numeric = float(raw)
            except ValueError:
                numeric = None
            if numeric is not None:
                return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
        return None

    try:
        with _auth_store_lock():
            auth_store = _load_auth_store()
        pool = auth_store.get("credential_pool")
        if not isinstance(pool, dict):
            return None
        entries = pool.get("openai-codex")
        if not isinstance(entries, list):
            return None
        now = time.time()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            token = entry.get("access_token")
            if not isinstance(token, str) or not token.strip():
                continue
            if entry.get("last_status") != "exhausted":
                continue
            code = entry.get("last_error_code")
            reason = str(entry.get("last_error_reason") or "").lower()
            message = str(entry.get("last_error_message") or "").lower()
            is_rate_limited = (
                code == 429
                or "rate_limit" in reason
                or "usage_limit" in reason
                or "quota" in reason
                or "rate limit" in message
                or "usage limit" in message
                or "quota" in message
            )
            if not is_rate_limited:
                continue
            reset_at = _parse_reset_at(entry.get("last_error_reset_at"))
            if reset_at is not None and reset_at <= now:
                continue
            return {
                "label": entry.get("label"),
                "last_refresh": entry.get("last_refresh"),
                "reset_at": reset_at,
                "reason": entry.get("last_error_reason"),
                "message": entry.get("last_error_message"),
                "access_token": token.strip(),
                "base_url": entry.get("base_url"),
            }
    except Exception:
        logger.debug("Codex pool rate-limit lookup failed", exc_info=True)
    return None


def _pool_codex_access_token() -> str:
    """Return the most-recent usable access_token from the openai-codex pool.

    Used as a fallback by ``resolve_codex_runtime_credentials`` when the
    singleton has no creds.  Reads ``credential_pool.openai-codex`` entries
    directly from auth.json and picks the first non-empty access_token,
    preferring entries that are not currently in an exhaustion cooldown.
    Returns ``""`` when no usable entry is found (caller handles by raising
    the original AuthError).
    """
    try:
        with _auth_store_lock():
            auth_store = _load_auth_store()
        pool = auth_store.get("credential_pool")
        if not isinstance(pool, dict):
            return ""
        entries = pool.get("openai-codex")
        if not isinstance(entries, list):
            return ""

        def _entry_usable(entry: Dict[str, Any]) -> bool:
            if not isinstance(entry, dict):
                return False
            token = entry.get("access_token")
            if not isinstance(token, str) or not token.strip():
                return False
            # Skip entries currently in an exhaustion cooldown window.
            reset_at = entry.get("last_error_reset_at")
            if isinstance(reset_at, (int, float)) and reset_at > time.time():
                return False
            return True

        for entry in entries:
            if _entry_usable(entry):
                return str(entry.get("access_token", "")).strip()
    except Exception:
        logger.debug("Codex pool fallback lookup failed", exc_info=True)
    return ""


# =============================================================================
# xAI Grok OAuth — tokens stored in ~/.hermes/auth.json
# =============================================================================

def _xai_oauth_state_from_store(auth_store: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return usable xAI OAuth state from provider state or credential pool."""
    state = _load_provider_state(auth_store, "xai-oauth")
    tokens = state.get("tokens") if isinstance(state, dict) else None
    if isinstance(tokens, dict):
        access_token = str(tokens.get("access_token", "") or "").strip()
        refresh_token = str(tokens.get("refresh_token", "") or "").strip()
        if access_token and refresh_token:
            return state

    credential_pool = auth_store.get("credential_pool")
    entries = (
        credential_pool.get("xai-oauth")
        if isinstance(credential_pool, dict)
        else None
    )
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            access_token = str(entry.get("access_token", "") or "").strip()
            refresh_token = str(entry.get("refresh_token", "") or "").strip()
            if not access_token or not refresh_token:
                continue
            merged = dict(state or {})
            merged["tokens"] = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": str(entry.get("token_type") or "Bearer"),
            }
            if entry.get("last_refresh"):
                merged["last_refresh"] = entry.get("last_refresh")
            merged.setdefault("auth_mode", "oauth_pkce")
            return merged

    return state if isinstance(state, dict) else None


def _xai_oauth_state_has_usable_tokens(state: Optional[Dict[str, Any]]) -> bool:
    tokens = state.get("tokens") if isinstance(state, dict) else None
    return (
        isinstance(tokens, dict)
        and bool(str(tokens.get("access_token", "") or "").strip())
        and bool(str(tokens.get("refresh_token", "") or "").strip())
    )


def _read_xai_oauth_tokens(*, _lock: bool = True) -> Dict[str, Any]:
    if _lock:
        with _auth_store_lock():
            auth_store = _load_auth_store()
    else:
        auth_store = _load_auth_store()
    state = _xai_oauth_state_from_store(auth_store)
    if not _xai_oauth_state_has_usable_tokens(state):
        global_state = _xai_oauth_state_from_store(_load_global_auth_store())
        if _xai_oauth_state_has_usable_tokens(global_state):
            state = global_state
    if not state:
        raise AuthError(
            "No xAI OAuth credentials stored. Select xAI Grok OAuth (SuperGrok / Premium+) in `hermes model`.",
            provider="xai-oauth",
            code="xai_auth_missing",
            relogin_required=True,
        )
    tokens = state.get("tokens")
    if not isinstance(tokens, dict):
        raise AuthError(
            "xAI OAuth state is missing tokens. Re-authenticate with `hermes model`.",
            provider="xai-oauth",
            code="xai_auth_invalid_shape",
            relogin_required=True,
        )
    access_token = str(tokens.get("access_token", "") or "").strip()
    refresh_token = str(tokens.get("refresh_token", "") or "").strip()
    if not access_token:
        raise AuthError(
            "xAI OAuth state is missing access_token. Re-authenticate with `hermes model`.",
            provider="xai-oauth",
            code="xai_auth_missing_access_token",
            relogin_required=True,
        )
    if not refresh_token:
        raise AuthError(
            "xAI OAuth state is missing refresh_token. Re-authenticate with `hermes model`.",
            provider="xai-oauth",
            code="xai_auth_missing_refresh_token",
            relogin_required=True,
        )
    return {
        "tokens": tokens,
        "last_refresh": state.get("last_refresh"),
        "discovery": state.get("discovery") or {},
        "redirect_uri": state.get("redirect_uri"),
    }


def _profile_has_own_xai_oauth_state(auth_store: Dict[str, Any]) -> bool:
    """True when this store has its OWN ``providers.xai-oauth`` block.

    Distinguishes a profile that genuinely shadows the root xAI grant from
    one that only *reads* root via ``_load_provider_state``'s fallback. Only
    the latter needs the refresh write-through below.
    """
    providers = auth_store.get("providers")
    return isinstance(providers, dict) and isinstance(providers.get("xai-oauth"), dict)


def _write_through_xai_oauth_to_global_root(state: Dict[str, Any]) -> None:
    """Persist a rotated xAI OAuth ``state`` into the global-root auth.json.

    Best-effort write-through for the multi-profile rotation hazard (#43589):
    xAI rotates the refresh_token on every refresh, so when a profile session
    refreshes a grant it resolved from the root fallback, the rotated chain
    must land back in root. Otherwise root keeps a now-revoked refresh token
    and every other profile reading the stale root grant dies with
    ``invalid_grant`` once its access token expires.

    Only updates ``providers.xai-oauth`` in the root store; never touches the
    profile store (the caller already saved that). Swallows all errors — a
    failed write-through degrades to the pre-existing behavior (root stale),
    it must never break the profile's own successful save.
    """
    global_path = _global_auth_file_path()
    if global_path is None:
        # Classic mode (profile == root); the profile save already hit root.
        return
    # Seat belt: under pytest, refuse to write the real user's
    # ~/.hermes/auth.json even when HERMES_HOME points at a profile path
    # (mirrors the read-side guard in _load_global_auth_store). Uses the
    # unmodified HOME env, not Path.home() which fixtures may monkeypatch.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        real_home_env = os.environ.get("HOME", "")
        if real_home_env:
            real_root = Path(real_home_env) / ".hermes" / "auth.json"
            try:
                if global_path.resolve(strict=False) == real_root.resolve(strict=False):
                    return
            except Exception:
                return
    try:
        _persist_provider_state_to_store(
            "xai-oauth",
            state,
            global_path,
            set_active=False,
        )
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("xAI OAuth: write-through to global root failed: %s", exc)


def _save_xai_oauth_tokens(
    tokens: Dict[str, Any],
    *,
    discovery: Optional[Dict[str, Any]] = None,
    redirect_uri: str = "",
    last_refresh: Optional[str] = None,
    auth_mode: str = "oauth_device_code",
    set_active: bool = True,
) -> None:
    """Persist xAI OAuth tokens into the auth store.

    When *set_active* is True (default), also promote ``xai-oauth`` to
    ``active_provider`` — appropriate for intentional model/auth login.
    Pass ``set_active=False`` for side-tool credential bootstrap (TTS/setup,
    tools config, dashboard token save, token refresh) so inference routing
    is unchanged.
    """
    if last_refresh is None:
        last_refresh = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with _auth_store_lock():
        auth_store = _load_auth_store()
        # A profile that lacks its own xai-oauth block is reading the root
        # grant through _load_provider_state's fallback. When such a profile
        # refreshes the (rotating) grant, we must write the rotated chain back
        # to root too, or root is left holding a revoked refresh token (#43589).
        # #74339: the old key-presence check (_profile_has_own_xai_oauth_state)
        # decided write-through based on whether the profile had a
        # providers.xai-oauth key BEFORE the save — but _store_provider_state
        # unconditionally creates that key below. Use
        # _load_provider_state_with_source to learn where the grant was
        # resolved from and write back only to that source.
        state, source_path = _load_provider_state_with_source(
            auth_store, "xai-oauth"
        )
        if state is None:
            state = {}
        state["tokens"] = tokens
        state["last_refresh"] = last_refresh
        state["auth_mode"] = auth_mode
        if discovery:
            state["discovery"] = discovery
        if redirect_uri:
            state["redirect_uri"] = redirect_uri
        global_root = _global_auth_file_path()
        is_from_root = bool(
            source_path is not None
            and global_root is not None
            and _same_path(source_path, global_root)
        )
        if is_from_root:
            # Grant was resolved from root — write back to root only.
            # Do NOT call _store_provider_state on the profile auth_store
            # (it would create a shadowing providers.xai-oauth key that
            # disables write-through on the next refresh — #74339).
            _write_through_xai_oauth_to_global_root(state)
        else:
            # Profile genuinely owns this — write to profile store.
            _store_provider_state(
                auth_store, "xai-oauth", state, set_active=set_active
            )
            _save_auth_store(auth_store)


def _xai_access_token_is_expiring(access_token: str, skew_seconds: int = 0) -> bool:
    if not isinstance(access_token, str) or "." not in access_token:
        return False
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return False
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8"))
        exp = payload.get("exp")
        if not isinstance(exp, (int, float)):
            return False
        return float(exp) <= (time.time() + max(0, int(skew_seconds)))
    except Exception:
        return False


def _xai_proactive_refresh_skew_seconds(access_token: str) -> int:
    """How far before JWT ``exp`` to proactively refresh xAI OAuth tokens.

    SuperGrok sessions can still ship multi-hour access tokens, where the
    gateway-oriented :data:`XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS` window
    makes sense. Device-code logins often return ~15-minute JWTs; applying
    the full hour-long skew to those forces a refresh on *every* credential
    resolution (chat turn, Imagine tool call, ``hermes auth status``, …),
    which burns single-use refresh tokens and races concurrent callers into
    ``invalid_grant`` quarantine.
    """
    max_skew = XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS
    if not isinstance(access_token, str) or "." not in access_token:
        return max_skew
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return max_skew
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8"))
        exp = payload.get("exp")
        if not isinstance(exp, (int, float)):
            return max_skew
        remaining = float(exp) - time.time()
        if remaining <= 0:
            return max_skew
        if remaining <= 45 * 60:
            return min(120, max_skew)
        return max_skew
    except Exception:
        return max_skew


def _xai_validate_oauth_endpoint(url: str, *, field: str) -> str:
    """Refuse any OIDC discovery endpoint that isn't HTTPS on the xAI origin.

    The OIDC discovery response is a long-lived, low-frequency request whose
    output is cached in ``~/.hermes/auth.json``. A single MITM during initial
    login could substitute a malicious ``token_endpoint``; that URL would
    then receive the refresh_token on every subsequent refresh — a permanent
    credential leak from a one-time MITM. Validating scheme + host pins the
    cached endpoint to the xAI auth origin (or a future ``*.x.ai`` subdomain
    if xAI migrates) so the cache poisoning loses its persistence guarantee.

    RFC 8414 §2 requires the issuer to be ``https://`` and SHOULD-keeps the
    token_endpoint on the same origin; we enforce both. ``x.ai`` is the
    bare apex, so we accept either exact host match or any ``.x.ai`` suffix.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise AuthError(
            f"xAI OIDC discovery returned a non-HTTPS {field}: {url!r}.",
            provider="xai-oauth",
            code="xai_discovery_invalid",
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise AuthError(
            f"xAI OIDC discovery {field} is missing a hostname: {url!r}.",
            provider="xai-oauth",
            code="xai_discovery_invalid",
        )
    if host != "x.ai" and not host.endswith(".x.ai"):
        raise AuthError(
            f"xAI OIDC discovery {field} host {host!r} is not on the xAI origin "
            f"(expected x.ai or a *.x.ai subdomain). Refusing to use a cached "
            f"endpoint that may have been substituted by a MITM during initial "
            f"discovery; re-authenticate with `hermes model` to re-fetch.",
            provider="xai-oauth",
            code="xai_discovery_invalid",
        )
    return url


def _xai_validate_inference_base_url(value: str, *, fallback: str) -> str:
    """Refuse a non-xAI base_url for the OAuth-authenticated inference path.

    The xAI Grok OAuth bearer is a high-value, long-lived credential tied to
    the user's SuperGrok subscription. ``XAI_BASE_URL`` / ``HERMES_XAI_BASE_URL``
    let users repoint the inference endpoint (handy for staging or a local
    proxy), but the env override is also a credential-leak vector: a tampered
    ``.env`` or hostile shell init that sets
    ``XAI_BASE_URL=https://attacker.example/v1`` would ship the OAuth access
    token to a third party on every request, silently.

    Pin the inference origin to ``api.x.ai`` (or any ``*.x.ai`` subdomain xAI
    may add). On rejection, fall back to the default and log a warning rather
    than raise — a bad env var should not deadlock authentication, but it
    should also never leak the bearer.

    ``value`` is the already-stripped, trailing-slash-trimmed candidate from
    env. Empty input returns ``fallback`` unchanged.
    """
    candidate = (value or "").strip().rstrip("/")
    if not candidate:
        return fallback
    try:
        parsed = urlparse(candidate)
    except Exception:
        logger.warning(
            "Ignoring malformed xAI base_url override %r; using %s instead.",
            candidate, fallback,
        )
        return fallback
    if parsed.scheme != "https":
        logger.warning(
            "Refusing non-HTTPS xAI base_url override %r (xai-oauth bearer would "
            "be sent in cleartext); falling back to %s.",
            candidate, fallback,
        )
        return fallback
    host = (parsed.hostname or "").lower()
    if not host:
        logger.warning(
            "Ignoring xAI base_url override %r with no hostname; using %s instead.",
            candidate, fallback,
        )
        return fallback
    if host != "x.ai" and not host.endswith(".x.ai"):
        logger.warning(
            "Refusing xAI base_url override %r — host %r is not on the xAI origin "
            "(expected x.ai or a *.x.ai subdomain). The xai-oauth bearer is only "
            "valid against xAI's inference API; sending it elsewhere would leak "
            "the credential. Falling back to %s.",
            candidate, host, fallback,
        )
        return fallback
    return candidate


def _xai_oauth_discovery(timeout_seconds: float = 15.0) -> Dict[str, str]:
    try:
        response = httpx.get(
            XAI_OAUTH_DISCOVERY_URL,
            headers={"Accept": "application/json"},
            timeout=timeout_seconds,
        )
    except Exception as exc:
        raise AuthError(
            f"xAI OIDC discovery failed: {exc}",
            provider="xai-oauth",
            code="xai_discovery_failed",
        ) from exc
    if response.status_code != 200:
        raise AuthError(
            f"xAI OIDC discovery returned status {response.status_code}.",
            provider="xai-oauth",
            code="xai_discovery_failed",
        )
    try:
        payload = response.json()
    except Exception as exc:
        raise AuthError(
            f"xAI OIDC discovery returned invalid JSON: {exc}",
            provider="xai-oauth",
            code="xai_discovery_invalid_json",
        ) from exc
    if not isinstance(payload, dict):
        raise AuthError(
            "xAI OIDC discovery response was not a JSON object.",
            provider="xai-oauth",
            code="xai_discovery_incomplete",
        )
    authorization_endpoint = str(payload.get("authorization_endpoint", "") or "").strip()
    token_endpoint = str(payload.get("token_endpoint", "") or "").strip()
    if not authorization_endpoint or not token_endpoint:
        raise AuthError(
            "xAI OIDC discovery response was missing required endpoints.",
            provider="xai-oauth",
            code="xai_discovery_incomplete",
        )
    _xai_validate_oauth_endpoint(authorization_endpoint, field="authorization_endpoint")
    _xai_validate_oauth_endpoint(token_endpoint, field="token_endpoint")
    return {
        "authorization_endpoint": authorization_endpoint,
        "token_endpoint": token_endpoint,
    }


def refresh_xai_oauth_pure(
    access_token: str,
    refresh_token: str,
    *,
    token_endpoint: str = "",
    timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    del access_token
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise AuthError(
            "xAI OAuth is missing refresh_token. Re-authenticate with `hermes model`.",
            provider="xai-oauth",
            code="xai_auth_missing_refresh_token",
            relogin_required=True,
        )
    endpoint = token_endpoint.strip() or _xai_oauth_discovery(timeout_seconds)["token_endpoint"]
    # Re-validate cached endpoints on the refresh hot path: an auth.json
    # written by an older Hermes (or hand-edited) may carry a non-xAI
    # token_endpoint that would receive every future refresh_token in
    # plaintext if we trusted it blindly. Cheap suffix check; fast-fail
    # with a clear error so the user can re-run `hermes model` to refetch.
    _xai_validate_oauth_endpoint(endpoint, field="token_endpoint")
    timeout = httpx.Timeout(max(5.0, float(timeout_seconds)))
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        response = client.post(
            endpoint,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "client_id": XAI_OAUTH_CLIENT_ID,
                "refresh_token": refresh_token,
            },
        )
    if response.status_code != 200:
        detail = response.text.strip()
        # ``403`` from xAI's token endpoint is almost always a tier /
        # entitlement gate (the OAuth grant exists but the account isn't
        # on the allowlist for API access).  Re-running ``hermes model``
        # won't fix that — surface a separate error code so
        # ``format_auth_error`` doesn't append a misleading
        # re-authenticate hint, and point users at the ``XAI_API_KEY``
        # fallback.  See #26847.
        if response.status_code == 403:
            raise AuthError(
                "xAI token refresh failed with HTTP 403."
                + (f" Response: {detail}" if detail else "")
                + " This OAuth account is not authorized for xAI API"
                  " access — xAI may be restricting API/OAuth use to"
                  " specific SuperGrok tiers despite the in-app"
                  " subscription being active. Re-logging in won't"
                  " change that; set ``XAI_API_KEY`` and switch to"
                  " ``provider: xai`` (API-key path) if available, or"
                  " upgrade your subscription at https://x.ai/grok.",
                provider="xai-oauth",
                code="xai_oauth_tier_denied",
                relogin_required=False,
            )
        raise AuthError(
            "xAI token refresh failed."
            + (f" Response: {detail}" if detail else ""),
            provider="xai-oauth",
            code="xai_refresh_failed",
            relogin_required=(response.status_code in {400, 401}),
        )
    try:
        payload = response.json()
    except Exception as exc:
        raise AuthError(
            f"xAI token refresh returned invalid JSON: {exc}",
            provider="xai-oauth",
            code="xai_refresh_invalid_json",
        ) from exc
    if not isinstance(payload, dict):
        raise AuthError(
            "xAI token refresh response was not a JSON object.",
            provider="xai-oauth",
            code="xai_refresh_invalid_response",
            relogin_required=True,
        )
    refreshed_access = str(payload.get("access_token", "") or "").strip()
    if not refreshed_access:
        raise AuthError(
            "xAI token refresh response was missing access_token.",
            provider="xai-oauth",
            code="xai_refresh_missing_access_token",
            relogin_required=True,
        )
    updated = {
        "access_token": refreshed_access,
        "refresh_token": str(payload.get("refresh_token") or refresh_token).strip(),
        "id_token": str(payload.get("id_token") or "").strip(),
        "expires_in": payload.get("expires_in"),
        "token_type": str(payload.get("token_type") or "Bearer").strip() or "Bearer",
        "last_refresh": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    return updated


def _refresh_xai_oauth_tokens(
    tokens: Dict[str, Any],
    *,
    token_endpoint: str,
    redirect_uri: str = "",
    timeout_seconds: float,
) -> Dict[str, Any]:
    # Re-persist whatever auth_mode is already stored (legacy pre-device-code
    # logins may still carry ``oauth_pkce``): the refresh hot path must not
    # relabel how the grant was originally obtained.
    try:
        state = _load_provider_state(_load_auth_store(), "xai-oauth") or {}
        auth_mode = str(state.get("auth_mode") or "oauth_device_code")
    except Exception:
        auth_mode = "oauth_device_code"
    refreshed = refresh_xai_oauth_pure(
        str(tokens.get("access_token", "") or ""),
        str(tokens.get("refresh_token", "") or ""),
        token_endpoint=token_endpoint,
        timeout_seconds=timeout_seconds,
    )
    updated_tokens = dict(tokens)
    updated_tokens["access_token"] = refreshed["access_token"]
    updated_tokens["refresh_token"] = refreshed["refresh_token"]
    if refreshed.get("id_token"):
        updated_tokens["id_token"] = refreshed["id_token"]
    if refreshed.get("expires_in") is not None:
        updated_tokens["expires_in"] = refreshed["expires_in"]
    if refreshed.get("token_type"):
        updated_tokens["token_type"] = refreshed["token_type"]
    _save_xai_oauth_tokens(
        updated_tokens,
        discovery={"token_endpoint": token_endpoint},
        redirect_uri=redirect_uri,
        last_refresh=refreshed["last_refresh"],
        auth_mode=auth_mode,
        # Refresh must not flip active_provider — TTS/side tools can refresh
        # xAI tokens while chat still routes through another provider.
        set_active=False,
    )
    return updated_tokens


def resolve_xai_oauth_runtime_credentials(
    *,
    force_refresh: bool = False,
    refresh_if_expiring: bool = True,
    refresh_skew_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    data = _read_xai_oauth_tokens()
    tokens = dict(data["tokens"])
    access_token = str(tokens.get("access_token", "") or "").strip()
    refresh_timeout_seconds = env_float("HERMES_XAI_REFRESH_TIMEOUT_SECONDS", 20)
    discovery = dict(data.get("discovery") or {})
    token_endpoint = str(discovery.get("token_endpoint", "") or "").strip()
    redirect_uri = str(data.get("redirect_uri", "") or "").strip()

    effective_skew = (
        int(refresh_skew_seconds)
        if refresh_skew_seconds is not None
        else _xai_proactive_refresh_skew_seconds(access_token)
    )
    should_refresh = bool(force_refresh)
    if (not should_refresh) and refresh_if_expiring:
        should_refresh = _xai_access_token_is_expiring(access_token, effective_skew)
    if should_refresh:
        with _auth_store_lock(timeout_seconds=max(float(AUTH_LOCK_TIMEOUT_SECONDS), refresh_timeout_seconds + 5.0)):
            data = _read_xai_oauth_tokens(_lock=False)
            tokens = dict(data["tokens"])
            access_token = str(tokens.get("access_token", "") or "").strip()
            discovery = dict(data.get("discovery") or {})
            token_endpoint = str(discovery.get("token_endpoint", "") or "").strip()
            redirect_uri = str(data.get("redirect_uri", "") or "").strip()
            effective_skew = (
                int(refresh_skew_seconds)
                if refresh_skew_seconds is not None
                else _xai_proactive_refresh_skew_seconds(access_token)
            )
            should_refresh = bool(force_refresh)
            if (not should_refresh) and refresh_if_expiring:
                should_refresh = _xai_access_token_is_expiring(access_token, effective_skew)
            if should_refresh:
                if not token_endpoint:
                    token_endpoint = _xai_oauth_discovery(refresh_timeout_seconds)["token_endpoint"]
                try:
                    tokens = _refresh_xai_oauth_tokens(
                        tokens,
                        token_endpoint=token_endpoint,
                        redirect_uri=redirect_uri,
                        timeout_seconds=refresh_timeout_seconds,
                    )
                    access_token = str(tokens.get("access_token", "") or "").strip()
                except AuthError as exc:
                    if _is_terminal_xai_oauth_refresh_error(exc):
                        # Terminal failure (HTTP 400/401/403 — invalid_grant, token revoked).
                        # Clear dead tokens from auth.json so subsequent sessions fail fast
                        # without a network retry. Mirrors credential_pool.py quarantine.
                        try:
                            _q_store = _load_auth_store()
                            _q_state = _load_provider_state(_q_store, "xai-oauth") or {}
                            _q_tokens = dict(_q_state.get("tokens") or {})
                            _q_tokens.pop("access_token", None)
                            _q_tokens.pop("refresh_token", None)
                            _q_state["tokens"] = _q_tokens
                            _q_state["last_auth_error"] = {
                                "provider": "xai-oauth",
                                "code": exc.code or "xai_refresh_failed",
                                "message": str(exc),
                                "reason": "runtime_refresh_failure",
                                "relogin_required": True,
                                "at": datetime.now(timezone.utc).isoformat(),
                            }
                            _store_provider_state(_q_store, "xai-oauth", _q_state, set_active=False)
                            _save_auth_store(_q_store)
                        except Exception as _save_exc:
                            logger.debug(
                                "xAI OAuth: failed to persist quarantined state: %s", _save_exc,
                            )
                    raise

    base_url = _xai_validate_inference_base_url(
        os.getenv("HERMES_XAI_BASE_URL", "").strip().rstrip("/")
        or os.getenv("XAI_BASE_URL", "").strip().rstrip("/"),
        fallback=DEFAULT_XAI_OAUTH_BASE_URL,
    )
    return {
        "provider": "xai-oauth",
        "base_url": base_url,
        "api_key": access_token,
        "source": "hermes-auth-store",
        "last_refresh": data.get("last_refresh"),
        # Display/telemetry only. Device-code is the only supported xAI OAuth
        # flow, so report it unconditionally — auth.json may still carry a
        # legacy ``oauth_pkce`` label, which the refresh path preserves as-is.
        "auth_mode": "oauth_device_code",
    }


# =============================================================================
# TLS verification helper
# =============================================================================

def _default_verify() -> bool | ssl.SSLContext:
    """Platform-aware default SSL verify for httpx clients.

    On macOS with Homebrew Python, the system OpenSSL cannot locate the
    system trust store and valid public certs fail verification. When
    certifi is importable we pin its bundle explicitly; elsewhere we
    defer to httpx's built-in default (certifi via its own dependency).
    Mirrors the weixin fix in 3a0ec1d93.
    """
    if sys.platform == "darwin":
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
    return True


def _resolve_verify(
    *,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    auth_state: Optional[Dict[str, Any]] = None,
) -> bool | ssl.SSLContext:
    tls_state = auth_state.get("tls") if isinstance(auth_state, dict) else {}
    tls_state = tls_state if isinstance(tls_state, dict) else {}

    effective_insecure = (
        is_truthy_value(insecure, default=False) if insecure is not None
        else is_truthy_value(tls_state.get("insecure", False), default=False)
    )
    effective_ca = (
        ca_bundle
        or tls_state.get("ca_bundle")
        or os.getenv("HERMES_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE")
        or os.getenv("REQUESTS_CA_BUNDLE")
    )

    if effective_insecure:
        return False
    if effective_ca:
        ca_path = str(effective_ca)
        if not os.path.isfile(ca_path):
            logger.warning(
                "CA bundle path does not exist: %s — falling back to default certificates",
                ca_path,
            )
            return _default_verify()
        return ssl.create_default_context(cafile=ca_path)
    return _default_verify()


# =============================================================================
# OAuth Device Code Flow — generic, parameterized by provider
# =============================================================================

def _request_device_code(
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    scope: Optional[str],
) -> Dict[str, Any]:
    """POST to the device code endpoint. Returns device_code, user_code, etc."""
    response = client.post(
        f"{portal_base_url}/api/oauth/device/code",
        data={
            "client_id": client_id,
            **({"scope": scope} if scope else {}),
        },
    )
    response.raise_for_status()
    data = response.json()

    required_fields = [
        "device_code", "user_code", "verification_uri",
        "verification_uri_complete", "expires_in", "interval",
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"Device code response missing fields: {', '.join(missing)}")
    return data


def _nous_device_auth_timeout_message(portal_base_url: str) -> str:
    """Actionable timeout text for Nous device-code login failures.

    A bare "Timed out waiting for device authorization" gives the user
    nothing to act on. The most common cause is Portal sign-in failing in
    the opened browser tab (including the server-side CAPTCHA loop from
    #20605), so point at the Portal login page and the retry command.
    """
    portal = (portal_base_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
    return (
        "Timed out waiting for device authorization.\n"
        "  Portal sign-in is required before the device code can be approved.\n"
        "  If the browser showed a CAPTCHA / 'You did not pass CAPTCHA' error,\n"
        "  finish signing in at the Portal in a normal browser tab, then retry:\n"
        "    hermes portal\n"
        f"  Portal login: {portal}/login"
    )


def _poll_for_token(
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    device_code: str,
    expires_in: int,
    poll_interval: int,
) -> Dict[str, Any]:
    """Poll the token endpoint until the user approves or the code expires."""
    deadline = time.monotonic() + max(1, expires_in)
    current_interval = max(1, min(poll_interval, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS))

    while time.monotonic() < deadline:
        response = client.post(
            f"{portal_base_url}/api/oauth/token",
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": client_id,
                "device_code": device_code,
            },
        )

        if response.status_code == 200:
            payload = response.json()
            if "access_token" not in payload:
                raise ValueError("Token response did not include access_token")
            return payload

        try:
            error_payload = response.json()
        except Exception:
            response.raise_for_status()
            raise RuntimeError("Token endpoint returned a non-JSON error response")

        error_code = error_payload.get("error", "")
        if error_code == "authorization_pending":
            time.sleep(current_interval)
            continue
        if error_code == "slow_down":
            current_interval = min(current_interval + 1, 30)
            time.sleep(current_interval)
            continue

        description = error_payload.get("error_description") or "Unknown authentication error"
        raise RuntimeError(f"{error_code}: {description}")

    # Enriched at the SOURCE so every caller inherits the guidance:
    # the CLI login (_nous_device_code_login) and the dashboard/desktop
    # poller (web_server._nous_poller, which surfaces str(e) to the UI).
    raise TimeoutError(_nous_device_auth_timeout_message(portal_base_url))


# =============================================================================
# Nous Portal — token refresh and model discovery
# =============================================================================

# -----------------------------------------------------------------------------
# Shared Nous token store — lets OAuth credentials persist across profiles
# so a new `hermes --profile <name> auth add nous --type oauth` can one-tap
# import instead of running the full device-code flow every time.
#
# File lives at ${HERMES_SHARED_AUTH_DIR}/nous_auth.json, defaulting to
# ``<hermes-root>/shared/nous_auth.json`` where ``<hermes-root>`` is what
# ``get_default_hermes_root()`` returns — ``~/.hermes`` on Linux/macOS,
# ``%LOCALAPPDATA%\hermes`` on native Windows, or the Docker/custom root.
# It is OUTSIDE any named profile's HERMES_HOME so named profiles (which
# typically live under ``<hermes-root>/profiles/<name>/``) all see the
# same file.
#
# Written on successful login and on every runtime refresh so the stored
# refresh_token stays current even if one profile refreshes and rotates it.
# If ever the stored refresh_token does go stale server-side, import fails
# gracefully and the user falls back to the normal device-code flow.
# -----------------------------------------------------------------------------

NOUS_SHARED_STORE_FILENAME = "nous_auth.json"
_nous_shared_lock_holder = threading.local()


def _nous_shared_auth_dir() -> Path:
    """Resolve the directory that holds the shared Nous token store.

    Honors ``HERMES_SHARED_AUTH_DIR`` so tests can redirect it to a tmp
    path without touching the real user's home. Defaults to
    ``<hermes-root>/shared/``, where ``<hermes-root>`` is what
    :func:`hermes_constants.get_default_hermes_root` returns — so
    Linux/macOS classic installs land at ``~/.hermes/shared/``, native
    Windows installs at ``%LOCALAPPDATA%\\hermes\\shared\\``, and
    Docker / custom ``HERMES_HOME`` deployments at
    ``<HERMES_HOME>/shared/``. Sits outside any named profile so all
    profiles under the same root share the store.
    """
    override = os.getenv("HERMES_SHARED_AUTH_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root() / "shared"


def _nous_shared_store_path() -> Path:
    path = _nous_shared_auth_dir() / NOUS_SHARED_STORE_FILENAME
    # Seat belt: if pytest is running and this resolves to a path under the
    # real user's Hermes root, refuse rather than silently corrupt cross-profile
    # state. Tests must set HERMES_SHARED_AUTH_DIR to a tmp_path (conftest
    # does not do this automatically — mirror the _auth_file_path() guard
    # so forgetting to set it fails loudly instead of writing to the real
    # shared store).
    if os.environ.get("PYTEST_CURRENT_TEST"):
        from hermes_constants import get_default_hermes_root
        real_home_shared = (
            get_default_hermes_root() / "shared" / NOUS_SHARED_STORE_FILENAME
        ).resolve(strict=False)
        try:
            resolved = path.resolve(strict=False)
        except Exception:
            resolved = path
        if resolved == real_home_shared:
            raise RuntimeError(
                f"Refusing to touch real user shared Nous auth store during test run: "
                f"{path}. Set HERMES_SHARED_AUTH_DIR to a tmp_path in your test fixture."
            )
    return path


@contextmanager
def _nous_shared_store_lock(timeout_seconds: float = AUTH_LOCK_TIMEOUT_SECONDS):
    """Cross-profile lock for the shared Nous OAuth store.

    Lock ordering invariant: if both this and ``_auth_store_lock`` need
    to be held, acquire ``_auth_store_lock`` FIRST. All runtime refresh
    paths follow this order. The one exception is
    ``_try_import_shared_nous_state``, which holds this lock alone for
    the entire refresh cycle so concurrent imports on sibling profiles
    can't race on the single-use shared refresh token; that helper must
    NOT be called with ``_auth_store_lock`` already held.
    """
    try:
        lock_path = _nous_shared_store_path().with_suffix(".lock")
    except RuntimeError:
        # No HERMES_HOME yet (pre-setup): fall through without locking.
        yield
        return

    with _file_lock(
        lock_path,
        _nous_shared_lock_holder,
        timeout_seconds,
        "Timed out waiting for shared Nous auth lock",
    ):
        yield


def _merge_shared_nous_oauth_state(state: Dict[str, Any]) -> bool:
    """Copy fresher shared OAuth tokens into a profile-local Nous state."""
    shared = _read_shared_nous_state()
    if not shared:
        return False

    shared_refresh = shared.get("refresh_token")
    if not isinstance(shared_refresh, str) or not shared_refresh.strip():
        return False

    local_refresh = state.get("refresh_token")
    shared_access_exp = _parse_iso_timestamp(shared.get("expires_at")) or 0.0
    local_access_exp = _parse_iso_timestamp(state.get("expires_at")) or 0.0
    refresh_changed = shared_refresh.strip() != str(local_refresh or "").strip()
    fresher_access = shared_access_exp > local_access_exp
    if not refresh_changed and not fresher_access:
        return False

    for key in (
        "access_token",
        "refresh_token",
        "token_type",
        "scope",
        "client_id",
        "portal_base_url",
        "inference_base_url",
        "obtained_at",
        "expires_at",
    ):
        value = shared.get(key)
        if value not in {None, ""}:
            state[key] = value
    return True


def _write_shared_nous_state(state: Dict[str, Any]) -> None:
    """Persist a minimal copy of the Nous OAuth state to the shared store.

    Best-effort: any failure is swallowed after logging. The shared store
    is a convenience layer; the per-profile auth.json remains the source
    of truth.

    We deliberately omit the runtime ``agent_key`` compatibility field;
    the OAuth tokens are the cross-profile source of truth.
    """
    refresh_token = state.get("refresh_token")
    access_token = state.get("access_token")
    if not (isinstance(refresh_token, str) and refresh_token.strip()):
        # No refresh_token = nothing worth sharing across profiles
        return
    if not (isinstance(access_token, str) and access_token.strip()):
        return

    shared = {
        "_schema": 1,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": state.get("token_type") or "Bearer",
        "scope": state.get("scope") or DEFAULT_NOUS_SCOPE,
        "client_id": state.get("client_id") or DEFAULT_NOUS_CLIENT_ID,
        "portal_base_url": state.get("portal_base_url") or DEFAULT_NOUS_PORTAL_URL,
        "inference_base_url": state.get("inference_base_url") or DEFAULT_NOUS_INFERENCE_URL,
        "obtained_at": state.get("obtained_at"),
        "expires_at": state.get("expires_at"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with _nous_shared_store_lock():
            path = _nous_shared_store_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            # secure_parent_dir refuses to chmod /, top-level dirs, or the
            # hermes-agent install tree (#25821, #93050).
            secure_parent_dir(path)
            tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
            # Create with 0o600 atomically via os.open(O_EXCL) — closes the TOCTOU
            # window where write_text() + post-write chmod briefly exposed Nous
            # refresh_token at process umask. See #19673, #21148.
            fd = os.open(
                str(tmp),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                stat.S_IRUSR | stat.S_IWUSR,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(shared, indent=2, sort_keys=True))
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
            finally:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError:
                    pass
        _oauth_trace(
            "nous_shared_store_written",
            path=str(path),
            refresh_token_fp=_token_fingerprint(refresh_token),
        )
    except Exception as exc:
        logger.debug("Failed to write shared Nous auth store: %s", exc)


def _read_shared_nous_state() -> Optional[Dict[str, Any]]:
    """Return the shared Nous OAuth state if present and well-formed.

    Returns ``None`` when the file is missing, unreadable, malformed, or
    lacks required fields. Callers should treat ``None`` as "no shared
    credentials available — fall through to device-code".
    """
    try:
        path = _nous_shared_store_path()
    except RuntimeError:
        # Test seat belt tripped — treat as missing
        return None
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        logger.debug("Shared Nous auth store at %s is unreadable: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    refresh_token = payload.get("refresh_token")
    access_token = payload.get("access_token")
    if not (isinstance(refresh_token, str) and refresh_token.strip()):
        return None
    if not (isinstance(access_token, str) and access_token.strip()):
        return None
    return payload


def _clear_shared_nous_state(reason: str) -> None:
    """Remove the shared Nous OAuth store after a terminal token failure."""
    try:
        with _nous_shared_store_lock():
            path = _nous_shared_store_path()
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        _oauth_trace("nous_shared_store_cleared", reason=reason)
    except Exception as exc:
        logger.debug("Failed to clear shared Nous auth store: %s", exc)


def _is_terminal_nous_refresh_error(exc: Exception) -> bool:
    """True when retrying the same Nous refresh token cannot succeed."""
    return (
        isinstance(exc, AuthError)
        and exc.provider == "nous"
        and exc.code in {"invalid_grant", "invalid_token", "refresh_token_reused"}
        and bool(exc.relogin_required)
    )


def _is_terminal_xai_oauth_refresh_error(exc: Exception) -> bool:
    """True when retrying the same xAI OAuth refresh token cannot succeed.

    ``xai_refresh_failed`` covers HTTP 400/401/403 from the token endpoint
    (invalid_grant, token revoked, refresh_token_reused).
    ``xai_auth_missing_refresh_token`` means the pool entry has no refresh
    token at all — retrying will never work.
    Both carry ``relogin_required=True``; transient failures (429, 5xx) do not.
    """
    return (
        isinstance(exc, AuthError)
        and exc.provider == "xai-oauth"
        and exc.code in {"xai_refresh_failed", "xai_auth_missing_refresh_token"}
        and bool(exc.relogin_required)
    )


def _is_terminal_codex_oauth_refresh_error(exc: Exception) -> bool:
    """True when retrying the same Codex OAuth refresh token cannot succeed.

    ``codex_refresh_failed`` covers HTTP 400/401/403 from the token endpoint
    (invalid_grant, token revoked, refresh_token_reused).
    ``codex_auth_missing_refresh_token`` means the pool entry has no refresh
    token at all — retrying will never work.
    Both carry ``relogin_required=True``; transient failures (429, 5xx) do not.
    """
    return (
        isinstance(exc, AuthError)
        and exc.provider == "openai-codex"
        and exc.code in {
            "codex_refresh_failed",
            "codex_auth_missing_refresh_token",
            "invalid_grant",
            "invalid_token",
            "refresh_token_reused",
        }
        and bool(exc.relogin_required)
    )


def _quarantine_nous_oauth_state(
    state: Dict[str, Any],
    error: AuthError,
    *,
    reason: str,
) -> None:
    """Keep routing metadata but remove dead OAuth material so it is not replayed."""
    # Forensic logging BEFORE we clear the token material. A hosted agent
    # can take a terminal invalid_grant and get quarantined here silently: the
    # only downstream signal is a "No access token found" WARNING once the pool
    # is already empty, which is too late to root-cause. A managed log drain may
    # be WARNING-only, so this MUST be logger.warning (INFO never reaches it).
    #
    # Redaction safety: emit ONLY the 12-char SHA-256 hex prefix of the refresh
    # token (correlates to NAS's refreshTokenHash without leaking the secret) plus
    # sizes/booleans. NEVER pass a raw token/agent_key into the log call — Hermes
    # has a known bug class where credential-shaped literals get corrupted in logs.
    forensic: Dict[str, Any] = {
        "reason": reason,
        "error_code": error.code,
        # No session_id field exists on Nous state; provenance is client_id +
        # agent_key_id (both non-secret routing identifiers).
        "client_id": state.get("client_id"),
        "agent_key_id": state.get("agent_key_id"),
        "refresh_token_fp": _token_fingerprint(state.get("refresh_token")),
    }

    # On-disk integrity of the auth store at the moment of quarantine.
    try:
        auth_path = _auth_file_path()
        forensic["auth_json_path"] = str(auth_path)
        try:
            st = os.stat(auth_path)
            forensic["auth_json_size"] = st.st_size
            forensic["auth_json_mtime"] = st.st_mtime
            forensic["auth_json_exists"] = True
        except FileNotFoundError:
            forensic["auth_json_exists"] = False
    except Exception as exc:  # pragma: no cover - never let logging break quarantine
        forensic["auth_json_stat_error"] = repr(exc)

    # Was the token already past its own expiry when it was rejected?
    already_expired: Optional[bool] = None
    expires_at_raw = state.get("expires_at")
    if isinstance(expires_at_raw, str) and expires_at_raw:
        try:
            parsed = datetime.fromisoformat(expires_at_raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            already_expired = parsed < datetime.now(timezone.utc)
        except ValueError:
            already_expired = None
    forensic["token_already_expired"] = already_expired

    logger.warning(
        "Nous OAuth state quarantined (terminal auth death): %s",
        json.dumps(forensic, sort_keys=True, ensure_ascii=False),
    )

    for key in (
        "access_token",
        "refresh_token",
        "expires_at",
        "expires_in",
        "obtained_at",
        "agent_key",
        "agent_key_id",
        "agent_key_expires_at",
        "agent_key_expires_in",
        "agent_key_reused",
        "agent_key_obtained_at",
    ):
        state.pop(key, None)
    state["last_auth_error"] = {
        "provider": "nous",
        "code": error.code,
        "message": str(error),
        "reason": reason,
        "relogin_required": True,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    _clear_shared_nous_state(reason)
    invalidate_nous_auth_status_cache()


def _quarantine_nous_pool_entries(
    auth_store: Dict[str, Any],
    error: AuthError,
    *,
    reason: str,
) -> bool:
    """Remove singleton-seeded Nous pool entries that contain dead OAuth state."""
    pool = auth_store.get("credential_pool")
    if not isinstance(pool, dict):
        return False
    entries = pool.get("nous")
    if not isinstance(entries, list):
        return False

    retained = []
    removed = False
    singleton_sources = {NOUS_DEVICE_CODE_SOURCE, f"manual:{NOUS_DEVICE_CODE_SOURCE}"}
    for entry in entries:
        if isinstance(entry, dict) and entry.get("source") in singleton_sources:
            removed = True
            continue
        retained.append(entry)

    if removed:
        pool["nous"] = retained
        _oauth_trace(
            "nous_pool_device_code_quarantined",
            reason=reason,
            error_code=error.code,
        )
    return removed


def _try_import_shared_nous_state(
    *,
    timeout_seconds: float = 15.0,
) -> Optional[Dict[str, Any]]:
    """Attempt to rehydrate Nous OAuth state from the shared store.

    Reads the shared file (if present), runs a forced refresh using the
    stored refresh_token to produce a fresh inference JWT scoped to this
    profile, and returns the full auth_state dict ready
    for ``persist_nous_credentials()``.

    Returns ``None`` when no shared state is available or the rehydrate
    fails for any reason (expired refresh_token, portal unreachable,
    etc.) — caller should then fall through to the normal device-code
    flow.
    """
    try:
        with _nous_shared_store_lock(timeout_seconds=max(timeout_seconds + 5.0, AUTH_LOCK_TIMEOUT_SECONDS)):
            shared = _read_shared_nous_state()
            if not shared:
                return None

            # Build a full state dict so refresh_nous_oauth_from_state has every
            # field it needs. force_refresh=True gets us a fresh access_token
            # for this profile.
            state: Dict[str, Any] = {
                "access_token": shared.get("access_token"),
                "refresh_token": shared.get("refresh_token"),
                "client_id": shared.get("client_id") or DEFAULT_NOUS_CLIENT_ID,
                "portal_base_url": shared.get("portal_base_url") or DEFAULT_NOUS_PORTAL_URL,
                "inference_base_url": shared.get("inference_base_url") or DEFAULT_NOUS_INFERENCE_URL,
                "token_type": shared.get("token_type") or "Bearer",
                "scope": shared.get("scope") or DEFAULT_NOUS_SCOPE,
                "obtained_at": shared.get("obtained_at"),
                "expires_at": shared.get("expires_at"),
                "agent_key": None,
                "agent_key_expires_at": None,
                "tls": {"insecure": False, "ca_bundle": None},
            }

            def _persist_shared_refresh(updated_state: Dict[str, Any], _reason: str) -> None:
                _write_shared_nous_state(updated_state)

            refreshed = refresh_nous_oauth_from_state(
                state,
                timeout_seconds=timeout_seconds,
                force_refresh=True,
                on_state_update=_persist_shared_refresh,
            )
            _write_shared_nous_state(refreshed)
    except AuthError as exc:
        _oauth_trace(
            "nous_shared_import_failed",
            error_type=type(exc).__name__,
            error_code=getattr(exc, "code", None),
        )
        if _is_terminal_nous_refresh_error(exc):
            _clear_shared_nous_state("shared_import_terminal_refresh_failure")
        logger.debug("Shared Nous import failed: %s", exc)
        return None
    except Exception as exc:
        _oauth_trace(
            "nous_shared_import_failed",
            error_type=type(exc).__name__,
        )
        logger.debug("Shared Nous import failed: %s", exc)
        return None

    return refreshed


def _refresh_access_token(
    *,
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    refresh_token: str,
) -> Dict[str, Any]:
    response = client.post(
        f"{portal_base_url}/api/oauth/token",
        headers={"x-nous-refresh-token": refresh_token},
        data={
            "grant_type": "refresh_token",
            "client_id": client_id,
        },
    )

    if response.status_code == 200:
        payload = response.json()
        if "access_token" not in payload:
            raise AuthError("Refresh response missing access_token",
                            provider="nous", code="invalid_token", relogin_required=True)
        return payload

    try:
        error_payload = response.json()
    except Exception as exc:
        raise AuthError("Refresh token exchange failed",
                        provider="nous", relogin_required=True) from exc

    code = str(error_payload.get("error", "invalid_grant"))
    description = str(error_payload.get("error_description") or "Refresh token exchange failed")
    relogin = code in {"invalid_grant", "invalid_token", "refresh_token_reused"}

    # Detect the OAuth 2.1 "refresh token reuse" signal from the Nous portal
    # server and surface an actionable message.  This fires when an external
    # process (health-check script, monitoring tool, custom self-heal hook)
    # called POST /api/oauth/token with Hermes's refresh_token without
    # persisting the rotated token back to auth.json — the server then
    # retires the original RT, Hermes's next refresh uses it, and the whole
    # session chain gets revoked as a token-theft signal (#15099).
    lowered = description.lower()
    if code == "refresh_token_reused" or "reuse" in lowered or "reuse detected" in lowered:
        description = (
            "Nous Portal detected refresh-token reuse and revoked this session.\n"
            "This usually means an external process (monitoring script, "
            "custom self-heal hook, or another Hermes install sharing "
            "~/.hermes/auth.json) called POST /api/oauth/token with Hermes's "
            "refresh token without persisting the rotated token back.\n"
            "Nous refresh tokens are single-use — only Hermes may call the "
            "refresh endpoint. For health checks, use `hermes auth status` "
            "instead.\n"
            "Re-authenticate with: hermes auth add nous"
        )
        relogin = True

    raise AuthError(description, provider="nous", code=code, relogin_required=relogin)


def fetch_nous_models(
    *,
    inference_base_url: str,
    api_key: str,
    timeout_seconds: float = 15.0,
    verify: bool | str = True,
) -> List[str]:
    """Fetch available model IDs from the Nous inference API."""
    timeout = httpx.Timeout(timeout_seconds)
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
        response = client.get(
            f"{inference_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )

    if response.status_code != 200:
        description = f"/models request failed with status {response.status_code}"
        try:
            err = response.json()
            description = str(err.get("error_description") or err.get("error") or description)
        except Exception as e:
            logger.debug("Could not parse error response JSON: %s", e)
        raise AuthError(description, provider="nous", code="models_fetch_failed")

    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list):
        return []

    model_ids: List[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            mid = model_id.strip()
            # Skip Hermes models — they're not reliable for agentic tool-calling
            if "hermes" in mid.lower():
                continue
            model_ids.append(mid)

    # Sort: prefer opus > pro > haiku/flash > sonnet (sonnet is cheap/fast,
    # users who want the best model should see opus first).
    def _model_priority(mid: str) -> tuple:
        low = mid.lower()
        if "opus" in low:
            return (0, mid)
        if "pro" in low and "sonnet" not in low:
            return (1, mid)
        if "sonnet" in low:
            return (3, mid)
        return (2, mid)

    model_ids.sort(key=_model_priority)
    return list(dict.fromkeys(model_ids))


def _agent_key_is_usable(state: Dict[str, Any], min_ttl_seconds: int) -> bool:
    key = state.get("agent_key")
    if not isinstance(key, str) or not key.strip():
        return False
    return _nous_invoke_jwt_is_usable(
        key,
        scope=state.get("scope"),
        expires_at=state.get("agent_key_expires_at"),
        min_ttl_seconds=max(0, int(min_ttl_seconds)),
    )


# Per-process memo for resolve_nous_access_token. Startup runs
# check_tool_availability once per managed-tool check_fn (browser, image_gen,
# etc.), and each one independently triggers a ~15s blocking token-refresh
# network call when the stored token is expired. On a slow/constrained host that
# serial burst stretches startup to many minutes. A short-TTL memo collapses the
# burst into a single network round-trip; callers that need freshness use
# separate flows (force_fresh / refresh_nous_oauth_pure) and are unaffected.
_RESOLVE_TOKEN_CACHE_LOCK = threading.Lock()
_RESOLVE_TOKEN_CACHE: "tuple[float, str] | None" = None
_RESOLVE_TOKEN_CACHE_TTL_S = 5.0


def resolve_nous_access_token(
    *,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    refresh_skew_seconds: int = ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> str:
    """Resolve a refresh-aware Nous Portal access token for managed tool gateways."""
    global _RESOLVE_TOKEN_CACHE
    # Memo: collapse the startup burst of managed-tool check_fns into one
    # network refresh. Only cache a successful, non-forced resolution for a
    # short window; force_fresh / error paths bypass and don't populate it.
    if not insecure and ca_bundle is None:
        with _RESOLVE_TOKEN_CACHE_LOCK:
            if _RESOLVE_TOKEN_CACHE is not None:
                cached_at, cached_token = _RESOLVE_TOKEN_CACHE
                if (time.monotonic() - cached_at) < _RESOLVE_TOKEN_CACHE_TTL_S:
                    return cached_token
    with _provider_state_transaction("nous") as (
        auth_store,
        state,
        state_source_path,
    ):

        if not state:
            raise AuthError(
                "Hermes is not logged into Nous Portal.",
                provider="nous",
                relogin_required=True,
            )

        # HERMES_PORTAL_BASE_URL / NOUS_PORTAL_BASE_URL is the trusted
        # operator/deployment override (mirrors NOUS_INFERENCE_BASE_URL) and
        # must win OUTRIGHT — including over a stored value — and bypass the
        # host allowlist entirely, since the allowlist exists to reject an
        # untrusted network-provided value, not one the operator configured.
        # Only fall through to the stored/default value + allowlist gate when
        # no override is set.
        env_portal_override = _nous_portal_env_override()
        if env_portal_override:
            portal_base_url = env_portal_override.rstrip("/")
        else:
            portal_base_url = (
                _optional_base_url(state.get("portal_base_url"))
                or DEFAULT_NOUS_PORTAL_URL
            ).rstrip("/")

            parsed_portal_url = urlparse(portal_base_url)
            if parsed_portal_url.hostname and parsed_portal_url.hostname not in _NOUS_PORTAL_ALLOWED_HOSTS:
                logger.warning(
                    "auth: ignoring invalid portal_base_url %r (host %r not in allowlist), using default",
                    portal_base_url, parsed_portal_url.hostname,
                )
                portal_base_url = DEFAULT_NOUS_PORTAL_URL

        client_id = str(state.get("client_id") or DEFAULT_NOUS_CLIENT_ID)
        verify = _resolve_verify(insecure=insecure, ca_bundle=ca_bundle, auth_state=state)

        with _nous_shared_store_lock(timeout_seconds=max(timeout_seconds + 5.0, AUTH_LOCK_TIMEOUT_SECONDS)):
            merged_shared = _merge_shared_nous_oauth_state(state)
            access_token = state.get("access_token")
            refresh_token = state.get("refresh_token")
            if not isinstance(access_token, str) or not access_token:
                raise AuthError(
                    "No access token found for Nous Portal login.",
                    provider="nous",
                    relogin_required=True,
                )

            if not _is_expiring(state.get("expires_at"), refresh_skew_seconds):
                if merged_shared:
                    _save_provider_state_to_source(auth_store, "nous", state, state_source_path)
                # Populate the memo on the valid-token fast path too: the
                # startup burst usually finds a *valid* token, but each
                # check_fn call still pays two cross-process file locks and
                # state reads to reach this return. The token has at least
                # refresh_skew_seconds (>= 120s) of life here, so a 5s memo
                # can never serve an expired token.
                if not insecure and ca_bundle is None:
                    with _RESOLVE_TOKEN_CACHE_LOCK:
                        _RESOLVE_TOKEN_CACHE = (time.monotonic(), access_token)
                return access_token

            if not isinstance(refresh_token, str) or not refresh_token:
                raise AuthError(
                    "Session expired and no refresh token is available.",
                    provider="nous",
                    relogin_required=True,
                )

            timeout = httpx.Timeout(timeout_seconds if timeout_seconds else 15.0)
            with httpx.Client(
                timeout=timeout,
                headers={"Accept": "application/json"},
                verify=verify,
            ) as client:
                try:
                    refreshed = _refresh_access_token(
                        client=client,
                        portal_base_url=portal_base_url,
                        client_id=client_id,
                        refresh_token=refresh_token,
                    )
                except AuthError as exc:
                    if _is_terminal_nous_refresh_error(exc):
                        _quarantine_nous_oauth_state(
                            state,
                            exc,
                            reason="managed_access_token_refresh_failure",
                        )
                        _quarantine_nous_pool_entries(
                            auth_store,
                            exc,
                            reason="managed_access_token_refresh_failure",
                        )
                        _save_provider_state_to_source(auth_store, "nous", state, state_source_path)
                    raise

            now = datetime.now(timezone.utc)
            access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
            state["access_token"] = refreshed["access_token"]
            state["refresh_token"] = refreshed.get("refresh_token") or refresh_token
            state["token_type"] = refreshed.get("token_type") or state.get("token_type") or "Bearer"
            state["scope"] = refreshed.get("scope") or state.get("scope")
            state["obtained_at"] = now.isoformat()
            state["expires_in"] = access_ttl
            state["expires_at"] = datetime.fromtimestamp(
                now.timestamp() + access_ttl,
                tz=timezone.utc,
            ).isoformat()
            state["portal_base_url"] = portal_base_url
            state["client_id"] = client_id
            state["tls"] = {
                "insecure": verify is False,
                "ca_bundle": verify if isinstance(verify, str) else None,
            }
            _save_provider_state_to_source(auth_store, "nous", state, state_source_path)
            _write_shared_nous_state(state)
            resolved = state["access_token"]
            if not insecure and ca_bundle is None:
                with _RESOLVE_TOKEN_CACHE_LOCK:
                    _RESOLVE_TOKEN_CACHE = (time.monotonic(), resolved)
            return resolved


def refresh_nous_oauth_pure(
    access_token: str,
    refresh_token: str,
    client_id: str,
    portal_base_url: str,
    inference_base_url: str,
    *,
    token_type: str = "Bearer",
    scope: str = DEFAULT_NOUS_SCOPE,
    obtained_at: Optional[str] = None,
    expires_at: Optional[str] = None,
    agent_key: Optional[str] = None,
    agent_key_expires_at: Optional[str] = None,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    force_refresh: bool = False,
    on_state_update: Optional[Callable[[Dict[str, Any], str], None]] = None,
) -> Dict[str, Any]:
    """Refresh Nous OAuth state without mutating auth.json directly.

    ``on_state_update`` is called after a successful access-token refresh.
    Callers that own persistent state can use it to save the newly rotated
    refresh token before later validation can fail.
    """
    state: Dict[str, Any] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "client_id": client_id or DEFAULT_NOUS_CLIENT_ID,
        "portal_base_url": (portal_base_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/"),
        "inference_base_url": (inference_base_url or DEFAULT_NOUS_INFERENCE_URL).rstrip("/"),
        "token_type": token_type or "Bearer",
        "scope": scope or DEFAULT_NOUS_SCOPE,
        "obtained_at": obtained_at,
        "expires_at": expires_at,
        "agent_key": agent_key,
        "agent_key_expires_at": agent_key_expires_at,
        "tls": {
            "insecure": bool(insecure),
            "ca_bundle": ca_bundle,
        },
    }
    verify = _resolve_verify(insecure=insecure, ca_bundle=ca_bundle, auth_state=state)
    timeout = httpx.Timeout(timeout_seconds if timeout_seconds else 15.0)

    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
        current_invoke_jwt_status = _nous_invoke_jwt_status(
            state.get("access_token"),
            scope=state.get("scope"),
            expires_at=state.get("expires_at"),
        )
        if force_refresh or current_invoke_jwt_status is not None:
            refresh_token_value = state.get("refresh_token")
            if not isinstance(refresh_token_value, str) or not refresh_token_value:
                if current_invoke_jwt_status is not None:
                    raise AuthError(
                        "Nous Portal access token is not a usable inference JWT "
                        f"({current_invoke_jwt_status}) and no refresh token is available. "
                        "Re-authenticate with: hermes auth add nous",
                        provider="nous",
                        code=current_invoke_jwt_status,
                        relogin_required=True,
                    )
                raise AuthError(
                    "No refresh token is available for Nous Portal.",
                    provider="nous",
                    relogin_required=True,
                )
            refreshed = _refresh_access_token(
                client=client,
                portal_base_url=state["portal_base_url"],
                client_id=state["client_id"],
                refresh_token=refresh_token_value,
            )
            now = datetime.now(timezone.utc)
            access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
            state["access_token"] = refreshed["access_token"]
            state["refresh_token"] = refreshed.get("refresh_token") or refresh_token_value
            state["token_type"] = refreshed.get("token_type") or state.get("token_type") or "Bearer"
            state["scope"] = refreshed.get("scope") or state.get("scope")
            # Heal a poisoned stored value: when the Portal-returned URL is
            # rejected by the allowlist (returns None), reset to the production
            # default instead of leaving a previously-persisted bad host (e.g. a
            # stale staging URL) in place. Without this reset, an auth.json that
            # was poisoned before the allowlist existed keeps re-validating to
            # None on every refresh and silently re-uses the dead endpoint —
            # the "falling back to default" warning never actually takes effect.
            refreshed_url = _validate_nous_inference_url_from_network(refreshed.get("inference_base_url"))
            state["inference_base_url"] = refreshed_url or DEFAULT_NOUS_INFERENCE_URL
            state["obtained_at"] = now.isoformat()
            state["expires_in"] = access_ttl
            state["expires_at"] = datetime.fromtimestamp(
                now.timestamp() + access_ttl, tz=timezone.utc
            ).isoformat()
            if on_state_update is not None:
                on_state_update(dict(state), "post_refresh_access_token")

        _assert_nous_inference_jwt_usable(state)
        _select_nous_invoke_jwt(state)

    return state


def refresh_nous_oauth_from_state(
    state: Dict[str, Any],
    *,
    timeout_seconds: float = 15.0,
    force_refresh: bool = False,
    on_state_update: Optional[Callable[[Dict[str, Any], str], None]] = None,
) -> Dict[str, Any]:
    """Refresh Nous OAuth from a state dict. Thin wrapper around refresh_nous_oauth_pure."""
    tls = state.get("tls") or {}
    return refresh_nous_oauth_pure(
        state.get("access_token", ""),
        state.get("refresh_token", ""),
        state.get("client_id", "hermes-cli"),
        state.get("portal_base_url", DEFAULT_NOUS_PORTAL_URL),
        state.get("inference_base_url", DEFAULT_NOUS_INFERENCE_URL),
        token_type=state.get("token_type", "Bearer"),
        scope=state.get("scope", DEFAULT_NOUS_SCOPE),
        obtained_at=state.get("obtained_at"),
        expires_at=state.get("expires_at"),
        agent_key=state.get("agent_key"),
        agent_key_expires_at=state.get("agent_key_expires_at"),
        timeout_seconds=timeout_seconds,
        insecure=tls.get("insecure"),
        ca_bundle=tls.get("ca_bundle"),
        force_refresh=force_refresh,
        on_state_update=on_state_update,
    )


def persist_nous_credentials(
    creds: Dict[str, Any],
    *,
    label: Optional[str] = None,
):
    """Persist Nous OAuth credentials as the singleton provider state
    and ensure the credential pool is in sync.

    Nous credentials are read at runtime from two independent locations:

    - ``providers.nous``: singleton state read by
      ``resolve_nous_runtime_credentials()`` during 401 recovery and by
      ``_seed_from_singletons()`` during pool load.
    - ``credential_pool.nous``: used by the runtime ``pool.select()`` path.

    Historically ``hermes auth add nous`` wrote a ``manual:device_code`` pool
    entry only, skipping ``providers.nous``. When the runtime credential
    expired, the recovery path read the empty singleton state and raised
    ``AuthError`` silently (``logger.debug`` at INFO level).

    This helper writes ``providers.nous`` then calls ``load_pool("nous")`` so
    ``_seed_from_singletons`` materialises the canonical ``device_code`` pool
    entry from the singleton.  Re-running login upserts the same entry in
    place; the pool never accumulates duplicate device_code rows.

    ``label`` is an optional user-chosen display name (from
    ``hermes auth add nous --label <name>``).  It gets embedded in the
    singleton state so that ``_seed_from_singletons`` uses it as the pool
    entry's label on every subsequent ``load_pool("nous")`` instead of the
    auto-derived token fingerprint.  When ``None``, the auto-derived label
    via ``label_from_token`` is used (unchanged default behaviour).

    Returns the upserted :class:`PooledCredential` entry (or ``None`` if
    seeding somehow produced no match — shouldn't happen).
    """
    from agent.credential_pool import load_pool

    state = dict(creds)
    if label and str(label).strip():
        state["label"] = str(label).strip()

    with _auth_store_lock():
        auth_store = _load_auth_store()
        _save_provider_state(auth_store, "nous", state)
        _save_auth_store(auth_store)

    # Mirror to the shared store so a new profile can one-tap import
    # these credentials via `hermes auth add nous --type oauth`. Best-
    # effort: any I/O failure is logged and swallowed (the per-profile
    # auth.json is still the source of truth).
    _write_shared_nous_state(state)

    pool = load_pool("nous")
    return next(
        (e for e in pool.entries() if e.source == NOUS_DEVICE_CODE_SOURCE),
        None,
    )


def _sync_nous_pool_from_auth_store() -> None:
    """Best-effort pool reseed after providers.nous changes; never fail login."""
    try:
        from agent.credential_pool import load_pool

        load_pool("nous")
    except Exception as exc:
        logger.debug("Failed to sync Nous credential pool from auth store: %s", exc)


def resolve_nous_runtime_credentials(
    *,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    force_refresh: bool = False,
    stale_access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve Nous inference credentials for runtime use.

    Ensures access_token is a valid inference-scoped JWT, refreshing it when
    needed. Concurrent processes coordinate through the auth store file lock.

    ``stale_access_token`` is the bearer that just failed upstream (401). When
    set together with ``force_refresh``, the refresh POST is skipped if the
    store — re-read under the lock — already holds a *different*, usable
    token: another process won the rotation, so this caller adopts it
    instead of rotating the shared grant again. Without this, N concurrent
    processes hitting the same hourly expiry issue N refreshes, and each
    rotation invalidates the token a sibling just adopted (Sep 2026: 120
    subagents, 81 refreshes, ~540 401s in eight minutes).
    """
    sequence_id = uuid.uuid4().hex[:12]

    with _provider_state_transaction("nous") as (
        auth_store,
        state,
        state_source_path,
    ):

        if not state:
            raise AuthError("Hermes is not logged into Nous Portal.",
                            provider="nous", relogin_required=True)

        def _already_rotated_by_peer(token: Any) -> bool:
            return bool(
                force_refresh
                and stale_access_token
                and isinstance(token, str)
                and token
                and token != stale_access_token
                and _nous_invoke_jwt_status(
                    token,
                    scope=state.get("scope"),
                    expires_at=state.get("expires_at"),
                ) is None
            )

        persisted_state = dict(state)
        state_persisted = False

        def _resolve_effective_routing_metadata() -> tuple[str, str, str, str]:
            """Resolve every routing value that shared OAuth state can replace."""
            portal_url = (
                _optional_base_url(state.get("portal_base_url"))
                or os.getenv("HERMES_PORTAL_BASE_URL")
                or os.getenv("NOUS_PORTAL_BASE_URL")
                or DEFAULT_NOUS_PORTAL_URL
            ).rstrip("/")

            # A persisted/stale portal_base_url is where the refresh token gets
            # POSTed on refresh — reject any host outside the allowlist so a
            # poisoned value can't exfiltrate the bearer, healing to the default.
            # Trusted operator env overrides bypass this network-value gate.
            env_portal_override = _nous_portal_env_override()
            if env_portal_override:
                portal_url = env_portal_override.rstrip("/")
            else:
                parsed_portal_url = urlparse(portal_url)
                portal_host = parsed_portal_url.hostname
                loopback_http = (
                    parsed_portal_url.scheme == "http"
                    and portal_host in {"localhost", "127.0.0.1"}
                )
                trusted_scheme = (
                    parsed_portal_url.scheme == "https" or loopback_http
                )
                if (
                    not portal_host
                    or portal_host not in _NOUS_PORTAL_ALLOWED_HOSTS
                    or not trusted_scheme
                ):
                    logger.warning(
                        "auth: ignoring invalid portal_base_url %r "
                        "(host %r or scheme not allowed), using default",
                        portal_url,
                        portal_host,
                    )
                    portal_url = DEFAULT_NOUS_PORTAL_URL

            # Re-validate persisted network-provenance on every shared merge.
            # The env override is runtime-only and must never be persisted.
            stored_inference_url = (
                _validate_nous_inference_url_from_network(
                    _optional_base_url(state.get("inference_base_url"))
                )
                or DEFAULT_NOUS_INFERENCE_URL
            )
            effective_inference_url = (
                _nous_inference_env_override() or stored_inference_url
            )
            effective_client_id = str(
                state.get("client_id") or DEFAULT_NOUS_CLIENT_ID
            )
            return (
                portal_url,
                stored_inference_url,
                effective_inference_url,
                effective_client_id,
            )

        (
            portal_base_url,
            stored_inference_base_url,
            inference_base_url,
            client_id,
        ) = _resolve_effective_routing_metadata()

        def _persist_state(reason: str) -> None:
            nonlocal persisted_state, state_persisted
            # Skip writes where only derived TTL countdowns changed; this keeps
            # the mtime-keyed Nous auth-status cache warm during read paths.
            if (
                _nous_effective_provider_state(state)
                == _nous_effective_provider_state(persisted_state)
            ):
                _oauth_trace(
                    "nous_state_persist_skipped",
                    sequence_id=sequence_id,
                    reason=reason,
                )
                return
            try:
                _save_provider_state_to_source(auth_store, "nous", state, state_source_path)
            except Exception as exc:
                _oauth_trace(
                    "nous_state_persist_failed",
                    sequence_id=sequence_id,
                    reason=reason,
                    error_type=type(exc).__name__,
                )
                raise
            _oauth_trace(
                "nous_state_persisted",
                sequence_id=sequence_id,
                reason=reason,
                refresh_token_fp=_token_fingerprint(state.get("refresh_token")),
                access_token_fp=_token_fingerprint(state.get("access_token")),
            )
            persisted_state = dict(state)
            state_persisted = True
            # Mirror post-refresh state to the shared store so sibling
            # profiles don't hold stale refresh_tokens after rotation.
            # Best-effort — any failure is logged and swallowed inside
            # _write_shared_nous_state.
            _write_shared_nous_state(state)

        verify = _resolve_verify(insecure=insecure, ca_bundle=ca_bundle, auth_state=state)
        timeout = httpx.Timeout(timeout_seconds if timeout_seconds else 15.0)
        _oauth_trace(
            "nous_runtime_credentials_start",
            sequence_id=sequence_id,
            refresh_token_fp=_token_fingerprint(state.get("refresh_token")),
        )

        with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
            access_token = state.get("access_token")
            refresh_token = state.get("refresh_token")

            if not isinstance(access_token, str) or not access_token:
                with _nous_shared_store_lock(
                    timeout_seconds=max(timeout_seconds + 5.0, AUTH_LOCK_TIMEOUT_SECONDS)
                ):
                    if _merge_shared_nous_oauth_state(state):
                        access_token = state.get("access_token")
                        refresh_token = state.get("refresh_token")
                        (
                            portal_base_url,
                            stored_inference_base_url,
                            inference_base_url,
                            client_id,
                        ) = _resolve_effective_routing_metadata()
                        _persist_state("runtime_shared_merge_missing_access_token")

            if not isinstance(access_token, str) or not access_token:
                raise AuthError("No access token found for Nous Portal login.",
                                provider="nous", relogin_required=True)

            invoke_jwt_status = _nous_invoke_jwt_status(
                access_token,
                scope=state.get("scope"),
                expires_at=state.get("expires_at"),
            )
            # Under the store lock: if the bearer that failed upstream is no
            # longer the one on disk and the on-disk one is usable, a peer
            # already rotated — adopt, never re-POST the shared grant.
            if _already_rotated_by_peer(access_token):
                _oauth_trace(
                    "refresh_skipped_peer_rotated",
                    sequence_id=sequence_id,
                    access_token_fp=_token_fingerprint(access_token),
                )
                force_refresh = False
            if force_refresh or invoke_jwt_status is not None:
                with _nous_shared_store_lock(timeout_seconds=max(timeout_seconds + 5.0, AUTH_LOCK_TIMEOUT_SECONDS)):
                    if _merge_shared_nous_oauth_state(state):
                        access_token = state.get("access_token")
                        refresh_token = state.get("refresh_token")
                        (
                            portal_base_url,
                            stored_inference_base_url,
                            inference_base_url,
                            client_id,
                        ) = _resolve_effective_routing_metadata()
                        invoke_jwt_status = _nous_invoke_jwt_status(
                            access_token,
                            scope=state.get("scope"),
                            expires_at=state.get("expires_at"),
                        )
                        _persist_state("post_shared_merge_access_unusable")
                        if _already_rotated_by_peer(access_token):
                            _oauth_trace(
                                "refresh_skipped_peer_rotated",
                                sequence_id=sequence_id,
                                access_token_fp=_token_fingerprint(access_token),
                            )
                            force_refresh = False

                    if force_refresh or invoke_jwt_status is not None:
                        if not isinstance(refresh_token, str) or not refresh_token:
                            reason = invoke_jwt_status or "force_refresh"
                            raise AuthError(
                                "Nous Portal access token is not a usable inference JWT "
                                f"({reason}) and no refresh token is available. "
                                "Re-authenticate with: hermes auth add nous",
                                provider="nous",
                                code=reason,
                                relogin_required=True,
                            )

                        refresh_reason = "force_refresh" if force_refresh else (invoke_jwt_status or "access_unusable")
                        _oauth_trace(
                            "refresh_start",
                            sequence_id=sequence_id,
                            reason=refresh_reason,
                            refresh_token_fp=_token_fingerprint(refresh_token),
                        )
                        try:
                            refreshed = _refresh_access_token(
                                client=client, portal_base_url=portal_base_url,
                                client_id=client_id, refresh_token=refresh_token,
                            )
                        except AuthError as exc:
                            if _is_terminal_nous_refresh_error(exc):
                                _quarantine_nous_oauth_state(
                                    state,
                                    exc,
                                    reason="runtime_access_refresh_failure",
                                )
                                _quarantine_nous_pool_entries(
                                    auth_store,
                                    exc,
                                    reason="runtime_access_refresh_failure",
                                )
                                _persist_state("terminal_runtime_access_refresh_failure")
                            raise
                        now = datetime.now(timezone.utc)
                        access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
                        previous_refresh_token = refresh_token
                        state["access_token"] = refreshed["access_token"]
                        state["refresh_token"] = refreshed.get("refresh_token") or refresh_token
                        state["token_type"] = refreshed.get("token_type") or state.get("token_type") or "Bearer"
                        state["scope"] = refreshed.get("scope") or state.get("scope")
                        # Heal a poisoned stored value (see refresh_nous_oauth_pure):
                        # reject → reset to production default, don't keep a stale
                        # staging host that re-validates to None every refresh.
                        # This (validated, network-provenance) value is what gets
                        # persisted to auth.json below. The NOUS_INFERENCE_BASE_URL
                        # env override is layered on for the client/return value
                        # only (see below) — it is never persisted.
                        refreshed_url = _validate_nous_inference_url_from_network(refreshed.get("inference_base_url"))
                        stored_inference_base_url = refreshed_url or DEFAULT_NOUS_INFERENCE_URL
                        inference_base_url = (
                            _nous_inference_env_override() or stored_inference_base_url
                        )
                        # Persist network-derived routing with rotated tokens so
                        # a later JWT validation failure cannot leave the profile
                        # and shared stores on stale metadata. Never persist the
                        # operator-only env overlay.
                        state["inference_base_url"] = stored_inference_base_url
                        state["obtained_at"] = now.isoformat()
                        state["expires_in"] = access_ttl
                        state["expires_at"] = datetime.fromtimestamp(
                            now.timestamp() + access_ttl, tz=timezone.utc
                        ).isoformat()
                        access_token = state["access_token"]
                        refresh_token = state["refresh_token"]
                        _oauth_trace(
                            "refresh_success",
                            sequence_id=sequence_id,
                            reason=refresh_reason,
                            previous_refresh_token_fp=_token_fingerprint(previous_refresh_token),
                            new_refresh_token_fp=_token_fingerprint(refresh_token),
                        )
                        # Persist immediately so validation failures cannot drop rotated refresh tokens.
                        _persist_state("post_refresh_access_token")

            _assert_nous_inference_jwt_usable(
                state,
                access_token=access_token,
            )
            _select_nous_invoke_jwt(
                state,
                access_token=access_token,
                sequence_id=sequence_id,
            )

            # Persist routing and TLS metadata for non-interactive refresh.
            # Persist the validated, network-provenance URL — NEVER the env
            # override (which is a runtime-only overlay; persisting it would
            # leak a dev/staging host into auth.json and survive unsetting it).
            state["portal_base_url"] = portal_base_url
            state["inference_base_url"] = stored_inference_base_url
            state["client_id"] = client_id
            state["tls"] = {
                "insecure": verify is False,
                "ca_bundle": verify if isinstance(verify, str) else None,
            }

        _persist_state("resolve_nous_runtime_credentials_final")

    if state_persisted:
        _sync_nous_pool_from_auth_store()

    api_key = state.get("agent_key")
    if not isinstance(api_key, str) or not api_key:
        raise AuthError("Failed to resolve a Nous inference API key",
                        provider="nous", code="server_error")

    expires_at = state.get("agent_key_expires_at")
    expires_epoch = _parse_iso_timestamp(expires_at)
    expires_in = (
        max(0, int(expires_epoch - time.time()))
        if expires_epoch is not None
        else _coerce_ttl_seconds(state.get("agent_key_expires_in"))
    )

    return {
        "provider": "nous",
        "base_url": inference_base_url,
        "api_key": api_key,
        "key_id": state.get("agent_key_id"),
        "expires_at": expires_at,
        "expires_in": expires_in,
        "source": NOUS_AUTH_PATH_INVOKE_JWT,
        # Preserve the public semantic source label while exposing the concrete
        # store separately for diagnostics. Refresh persistence uses
        # state_source_path internally and must not overload this field.
        "auth_path": NOUS_AUTH_PATH_INVOKE_JWT,
        "state_path": str(state_source_path or _auth_file_path()),
    }


# =============================================================================
# Status helpers
# =============================================================================

def _empty_nous_auth_status() -> Dict[str, Any]:
    return {
        "logged_in": False,
        "portal_base_url": None,
        "inference_base_url": None,
        "access_expires_at": None,
        "agent_key_expires_at": None,
        "has_refresh_token": False,
        "inference_credential_present": False,
        "credential_source": None,
    }


def _snapshot_nous_pool_status() -> Dict[str, Any]:
    """Best-effort status from the credential pool.

    This is a fallback only. The auth-store provider state is the runtime source
    of truth because it is what ``resolve_nous_runtime_credentials()`` refreshes.
    """
    try:
        from agent.credential_pool import load_pool

        pool = load_pool("nous")
        if not pool or not pool.has_credentials():
            return _empty_nous_auth_status()

        entries = list(pool.entries())
        if not entries:
            return _empty_nous_auth_status()

        def _entry_sort_key(entry: Any) -> tuple[float, float, int]:
            agent_exp = _parse_iso_timestamp(getattr(entry, "agent_key_expires_at", None)) or 0.0
            access_exp = _parse_iso_timestamp(getattr(entry, "expires_at", None)) or 0.0
            priority = int(getattr(entry, "priority", 0) or 0)
            return (agent_exp, access_exp, -priority)

        entry = max(entries, key=_entry_sort_key)
        runtime_key = getattr(entry, "runtime_api_key", None)
        if not runtime_key:
            return _empty_nous_auth_status()
        access_token = getattr(entry, "access_token", None)
        auth_type = str(getattr(entry, "auth_type", "") or "").strip().lower()
        refresh_token = getattr(entry, "refresh_token", None)
        is_portal_oauth = bool(access_token) and (
            auth_type.startswith("oauth") or bool(refresh_token)
        )
        label = getattr(entry, "label", "unknown")
        portal_status_url = None
        if is_portal_oauth:
            portal_status_url = (
                getattr(entry, "portal_base_url", None)
                or DEFAULT_NOUS_PORTAL_URL
            )

        return {
            "logged_in": is_portal_oauth,
            "portal_base_url": portal_status_url,
            "inference_base_url": getattr(entry, "inference_base_url", None)
            or getattr(entry, "runtime_base_url", None)
            or getattr(entry, "base_url", None),
            "access_token": access_token if is_portal_oauth else None,
            "access_expires_at": getattr(entry, "expires_at", None),
            "agent_key_expires_at": getattr(entry, "agent_key_expires_at", None),
            "has_refresh_token": bool(refresh_token),
            "inference_credential_present": True,
            "credential_source": f"pool:{label}",
            "source": f"pool:{label}",
        }
    except Exception:
        return _empty_nous_auth_status()


# ── Process-level memo for get_nous_auth_status() ──
# get_nous_auth_status() validates state by calling resolve_nous_runtime_credentials(),
# which does a synchronous OAuth refresh POST to portal.nousresearch.com. That can take
# ~350ms even on the failure path, and read-only UI surfaces (`hermes tools`, status panels,
# subscription-feature checks) call it many times per render — `hermes tools` → "All Platforms"
# was firing the refresh ~31× during one menu paint, racking up >13s of HTTP and burning
# single-use refresh tokens. Cache the snapshot for a few seconds, keyed on the auth.json
# path + mtime so that profile switches do not share a process memo and
# `hermes auth login/logout/add/remove` invalidate naturally on the next call.
_NOUS_AUTH_STATUS_CACHE_TTL = 15.0  # seconds
_nous_auth_status_cache: Optional[Tuple[float, str, Optional[float], Dict[str, Any]]] = None

# mtime-keyed memo for _load_global_auth_store(): (path, mtime_ns, store).
# Same invalidation contract as _nous_auth_status_cache — the global auth
# file changes only when a global-scope auth write touches it.
_global_auth_store_cache: Optional[Tuple[str, int, Dict[str, Any]]] = None


def _auth_file_cache_key() -> Tuple[str, Optional[float]]:
    auth_file = _auth_file_path()
    try:
        auth_file_key = str(auth_file.resolve(strict=False))
    except Exception:
        auth_file_key = str(auth_file)
    try:
        return auth_file_key, auth_file.stat().st_mtime
    except FileNotFoundError:
        return auth_file_key, None
    except Exception:
        return auth_file_key, None


def invalidate_nous_auth_status_cache() -> None:
    """Clear the get_nous_auth_status() process-level memo.

    Call this from any code path that mutates Nous auth state without going
    through resolve_nous_runtime_credentials() (e.g. tests). Login/logout
    flows touch auth.json, so the mtime check below invalidates them
    automatically — explicit invalidation is the belt-and-braces option.
    """
    global _nous_auth_status_cache
    _nous_auth_status_cache = None


def get_nous_auth_status() -> Dict[str, Any]:
    """Status snapshot for Nous auth.

    Prefer the auth-store provider state, because that is the live source of
    truth for refresh operations. When provider state exists, validate it
    by resolving runtime credentials so revoked refresh sessions do not show up
    as a healthy login. If provider state is absent, fall back to the credential
    pool for the just-logged-in / not-yet-promoted case.

    The returned snapshot is memoised for ~15s keyed on the auth.json mtime,
    so menu/status surfaces that ask repeatedly don't trigger one refresh POST
    per call. Login/logout flows write to auth.json and therefore invalidate
    the cache automatically; tests can also call
    ``invalidate_nous_auth_status_cache()`` explicitly.
    """
    global _nous_auth_status_cache
    now = time.monotonic()
    auth_file_key, mtime = _auth_file_cache_key()
    cached = _nous_auth_status_cache
    if cached is not None:
        cached_at, cached_auth_file_key, cached_mtime, cached_status = cached
        if (
            cached_auth_file_key == auth_file_key
            and cached_mtime == mtime
            and (now - cached_at) < _NOUS_AUTH_STATUS_CACHE_TTL
        ):
            return dict(cached_status)

    status = _compute_nous_auth_status()
    _nous_auth_status_cache = (now, auth_file_key, mtime, dict(status))
    return status


def _compute_nous_auth_status() -> Dict[str, Any]:
    """Uncached implementation of get_nous_auth_status(). See that function."""
    state = get_provider_auth_state("nous")
    if state:
        base_status = {
            "logged_in": bool(state.get("access_token")),
            "portal_base_url": state.get("portal_base_url"),
            "inference_base_url": state.get("inference_base_url"),
            "access_expires_at": state.get("expires_at"),
            "agent_key_expires_at": state.get("agent_key_expires_at"),
            "has_refresh_token": bool(state.get("refresh_token")),
            "access_token": state.get("access_token"),
            "inference_credential_present": bool(
                state.get("access_token") or state.get("agent_key")
            ),
            "credential_source": "auth_store",
            "source": "auth_store",
        }
        try:
            creds = resolve_nous_runtime_credentials()
            refreshed_state = get_provider_auth_state("nous") or state
            base_status.update(
                {
                    "logged_in": True,
                    "portal_base_url": refreshed_state.get("portal_base_url") or base_status.get("portal_base_url"),
                    "inference_base_url": creds.get("base_url")
                    or refreshed_state.get("inference_base_url")
                    or base_status.get("inference_base_url"),
                    "access_expires_at": refreshed_state.get("expires_at") or base_status.get("access_expires_at"),
                    "agent_key_expires_at": creds.get("expires_at")
                    or refreshed_state.get("agent_key_expires_at")
                    or base_status.get("agent_key_expires_at"),
                    "has_refresh_token": bool(refreshed_state.get("refresh_token")),
                    "inference_credential_present": True,
                    "credential_source": "auth_store",
                    "source": f"runtime:{creds.get('source', 'portal')}",
                    "key_id": creds.get("key_id"),
                }
            )
            return base_status
        except AuthError as exc:
            base_status.update({
                "logged_in": False,
                "error": str(exc),
                "relogin_required": bool(getattr(exc, "relogin_required", False)),
                "error_code": getattr(exc, "code", None),
            })
            return base_status

    return _snapshot_nous_pool_status()


def get_nous_auth_status_local() -> Dict[str, Any]:
    """Refresh-free Nous auth snapshot for read-only display surfaces.

    Unlike :func:`get_nous_auth_status`, this NEVER calls
    ``resolve_nous_runtime_credentials()`` and therefore never performs an
    OAuth refresh POST or consumes a single-use refresh token. It reports the
    persisted auth-store state, classifying the access token with a local
    invoke-JWT decode only.

    Use this from status panels, doctor checks, and polled dashboard
    endpoints. Explicit auth actions (login flows, portal operations that
    need a live credential) should keep using ``get_nous_auth_status()``.

    ``logged_in`` here means "a persisted login exists that the runtime can
    use or refresh": a currently-usable invoke JWT, or a refresh token that
    has not been terminally quarantined. It does not prove the refresh token
    is still accepted server-side — only a live resolve can do that.
    """
    try:
        state = get_provider_auth_state("nous")
    except Exception:
        state = None

    if not state:
        return _snapshot_nous_pool_status()

    access_token = state.get("access_token")
    jwt_reason = _nous_invoke_jwt_status(
        access_token,
        scope=state.get("scope"),
        expires_at=state.get("expires_at"),
    )
    last_err = state.get("last_auth_error")
    terminal = bool(
        isinstance(last_err, dict)
        and last_err.get("relogin_required")
        and not (access_token or state.get("refresh_token"))
    )
    logged_in = (jwt_reason is None) or (
        bool(state.get("refresh_token")) and not terminal
    )

    status: Dict[str, Any] = {
        "logged_in": logged_in,
        "portal_base_url": state.get("portal_base_url"),
        "inference_base_url": state.get("inference_base_url"),
        "access_token": access_token,
        "access_expires_at": state.get("expires_at"),
        "agent_key_expires_at": state.get("agent_key_expires_at"),
        "has_refresh_token": bool(state.get("refresh_token")),
        "inference_credential_present": bool(
            access_token or state.get("agent_key")
        ),
        "credential_source": "auth_store",
        "source": "auth_store_local",
    }
    if terminal and isinstance(last_err, dict):
        status["relogin_required"] = True
        status["error_code"] = last_err.get("code")
        status["error"] = last_err.get("message") or "re-login required"
    return status


# Enum values reported on the dashboard /api/status as ``nous_session_valid``.
# NAS's health sweep re-mints the bootstrap session ONLY on "terminal"; "valid"
# and "unknown" are no-ops. Keep this set small and stable — NAS parses it with
# a permissive schema, so new members are non-breaking but should stay rare.
NOUS_SESSION_VALID = "valid"
NOUS_SESSION_TERMINAL = "terminal"
NOUS_SESSION_UNKNOWN = "unknown"


def get_nous_session_validity() -> str:
    """Classify the Nous bootstrap session for the dashboard /api/status probe.

    Returns one of:
      - ``"valid"``    — a usable Nous credential is present (login healthy).
      - ``"terminal"`` — the Nous session has taken a terminal auth failure
        (invalid_grant / quarantined / relogin required). This is the sole
        signal NAS acts on to re-mint a hosted-agent bootstrap session.
      - ``"unknown"``  — indeterminate (no Nous provider state, or a transient/
        non-terminal error). Never triggers a re-mint.

    Determinable with NO working token — it reads local auth-store state only,
    which is exactly the condition a dead hosted box is in. This function is
    called by the frequently-polled public ``/api/status`` endpoint, so it must
    never resolve credentials or perform an OAuth refresh.

    ANTI-FLAP CONTRACT: only a *terminal* failure maps to "terminal". A normal
    mid-rotation blip, a transient network error, or a merely-expiring token
    must NOT report "terminal" (that would trigger a spurious NAS re-mint on a
    healthy box). We key "terminal" on the auth layer's own terminal signal
    (`relogin_required`) plus a persisted quarantine marker, never on a bare
    "not logged in".
    """
    # A persisted quarantine marker is the strongest, most stable terminal
    # signal: the refresh path writes `last_auth_error.relogin_required=True`
    # into the Nous provider state when it clears dead tokens (the exact path
    # that produced the incident's "No access token found"). Read it directly
    # so we report "terminal" even after the in-memory AuthError is long gone.
    try:
        state = get_provider_auth_state("nous")
    except Exception:
        return NOUS_SESSION_UNKNOWN

    if not state:
        return NOUS_SESSION_UNKNOWN

    last_err = state.get("last_auth_error")
    if isinstance(last_err, dict) and last_err.get("relogin_required"):
        # Only terminal while there is no usable credential left. If a later
        # successful login repopulated tokens, the stale marker must not
        # keep reporting terminal.
        if not (state.get("access_token") or state.get("refresh_token")):
            return NOUS_SESSION_TERMINAL

    if _nous_invoke_jwt_status(
        state.get("access_token"),
        scope=state.get("scope"),
        expires_at=state.get("expires_at"),
    ) is None:
        return NOUS_SESSION_VALID

    # Missing, malformed, expired, or merely expiring credentials are not proof
    # of a terminal session. Runtime inference/keepalive paths own refreshes;
    # the health endpoint remains side-effect free and reports indeterminate.
    return NOUS_SESSION_UNKNOWN


def get_codex_auth_status() -> Dict[str, Any]:
    """Status snapshot for Codex auth.
    
    Checks the credential pool first (where `hermes auth` stores credentials),
    then falls back to the legacy provider state.
    """
    # Check credential pool first — this is where `hermes auth` and
    # `hermes model` store device_code tokens.
    try:
        from agent.credential_pool import load_pool
        pool = load_pool("openai-codex")
        if pool and pool.has_credentials():
            entry = pool.select()
            if entry is not None:
                api_key = (
                    getattr(entry, "runtime_api_key", None)
                    or getattr(entry, "access_token", "")
                )
                if api_key and not _codex_access_token_is_expiring(api_key, 0):
                    return {
                        "logged_in": True,
                        "auth_store": str(_auth_file_path()),
                        "last_refresh": getattr(entry, "last_refresh", None),
                        "auth_mode": "chatgpt",
                        "source": f"pool:{getattr(entry, 'label', 'unknown')}",
                        "api_key": api_key,
                    }
            rate_limit = _codex_pool_rate_limit_status()
            if rate_limit:
                return {
                    "logged_in": True,
                    "auth_store": str(_auth_file_path()),
                    "last_refresh": rate_limit.get("last_refresh"),
                    "auth_mode": "chatgpt",
                    "source": f"pool:{rate_limit.get('label') or 'unknown'}",
                    "rate_limited": True,
                    "error_code": CODEX_RATE_LIMITED_CODE,
                    "error": (
                        rate_limit.get("message")
                        or "Codex provider quota exhausted; retry after the usage limit resets."
                    ),
                    "reset_at": rate_limit.get("reset_at"),
                }
    except Exception:
        pass

    # Fall back to legacy provider state
    try:
        creds = resolve_codex_runtime_credentials()
        return {
            "logged_in": True,
            "auth_store": str(_auth_file_path()),
            "last_refresh": creds.get("last_refresh"),
            "auth_mode": creds.get("auth_mode"),
            "source": creds.get("source"),
            "api_key": creds.get("api_key"),
        }
    except AuthError as exc:
        return {
            "logged_in": False,
            "auth_store": str(_auth_file_path()),
            "error": str(exc),
        }


def get_xai_oauth_auth_status() -> Dict[str, Any]:
    try:
        from agent.credential_pool import load_pool

        pool = load_pool("xai-oauth")
        if pool and pool.has_credentials():
            entry = pool.select()
            if entry is not None:
                api_key = (
                    getattr(entry, "runtime_api_key", None)
                    or getattr(entry, "access_token", "")
                )
                if api_key and not _xai_access_token_is_expiring(api_key, 0):
                    return {
                        "logged_in": True,
                        "auth_store": str(_auth_file_path()),
                        "last_refresh": getattr(entry, "last_refresh", None),
                        # Display/telemetry only. Device-code is the only xAI
                        # OAuth flow, so report it unconditionally (auth.json
                        # may still carry a legacy ``oauth_pkce`` label).
                        "auth_mode": "oauth_device_code",
                        "source": f"pool:{getattr(entry, 'label', 'unknown')}",
                        "api_key": api_key,
                    }
    except Exception:
        pass

    try:
        creds = resolve_xai_oauth_runtime_credentials()
        return {
            "logged_in": True,
            "auth_store": str(_auth_file_path()),
            "last_refresh": creds.get("last_refresh"),
            "auth_mode": creds.get("auth_mode"),
            "source": creds.get("source"),
            "api_key": creds.get("api_key"),
        }
    except AuthError as exc:
        return {
            "logged_in": False,
            "auth_store": str(_auth_file_path()),
            "error": str(exc),
        }


def get_api_key_provider_status(provider_id: str) -> Dict[str, Any]:
    """Status snapshot for API-key providers (z.ai, Kimi, MiniMax)."""
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "api_key":
        return {"configured": False}

    # Keyless providers (opencode-free) are served anonymously: no credential
    # exists, so every install counts as configured/logged in. Derived from
    # the HermesOverlay keyless flag — the same source the provider catalog
    # and GUI contract tests use.
    try:
        from hermes_cli.providers import HERMES_OVERLAYS
        _overlay = HERMES_OVERLAYS.get(provider_id)
    except Exception:
        _overlay = None
    if _overlay is not None and getattr(_overlay, "keyless", False):
        return {
            "configured": True,
            "provider": provider_id,
            "name": pconfig.name,
            "key_source": "keyless",
            "base_url": pconfig.inference_base_url,
            "logged_in": True,
        }

    api_key = ""
    key_source = ""
    api_key, key_source = _resolve_api_key_provider_secret(provider_id, pconfig)

    env_url = ""
    if pconfig.base_url_env_var:
        env_url = os.getenv(pconfig.base_url_env_var, "").strip()

    if provider_id in {"kimi-coding", "kimi-coding-cn"}:
        base_url = _resolve_kimi_base_url(api_key, pconfig.inference_base_url, env_url)
    elif env_url:
        base_url = env_url
    else:
        base_url = pconfig.inference_base_url

    if provider_id == "actual":
        base_url = normalize_actual_base_url(base_url)

    actual_local_noauth = (
        provider_id == "actual"
        and not api_key
        and is_actual_local_base_url(base_url)
    )

    return {
        "configured": bool(api_key) or actual_local_noauth,
        "provider": provider_id,
        "name": pconfig.name,
        "key_source": key_source or ("local-offline" if actual_local_noauth else ""),
        "base_url": base_url,
        "logged_in": bool(api_key) or actual_local_noauth,  # compat with OAuth status shape
    }


def _external_process_auth_evidence(provider_id: str) -> tuple[bool, Optional[str]]:
    """Best-effort POSITIVE evidence that an external-process provider's CLI
    is authenticated.

    Returns ``(verified, source)``. ``verified`` is only ever True on hard
    evidence (a supported env token, or a known on-disk credential store).
    False means "not verifiable from here", NOT "signed out" — the Copilot
    CLI may hold its session in an OS keychain Hermes can't read. Callers
    must therefore treat False as unknown, never as proof of absence.

    Deliberately subprocess-free: this runs from status endpoints and pickers,
    and spawning ``gh auth token`` there re-creates the cold-start stall
    (#60800) that copilot_auth.py works to avoid.
    """
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
    # 2. The Copilot CLI's own plaintext token store (~/.copilot/config.json,
    #    written by `copilot login` when no OS keychain is available). The file
    #    is JSONC — strip //-comment lines before parsing.
    try:
        cli_config = os.path.expanduser("~/.copilot/config.json")
        if os.path.isfile(cli_config):
            with open(cli_config, "r", encoding="utf-8", errors="ignore") as fh:
                raw = "\n".join(
                    line for line in fh.read().splitlines()
                    if not line.lstrip().startswith("//")
                )
            data = json.loads(raw) if raw.strip() else {}
            tokens = data.get("copilotTokens")
            if isinstance(tokens, dict) and any(
                isinstance(v, str) and v.strip() for v in tokens.values()
            ):
                return True, "~/.copilot/config.json"
    except Exception as exc:
        logger.debug("copilot-acp CLI config evidence check failed: %s", exc)
    # 3. Known on-disk GitHub Copilot credential stores (the same locations
    #    models.py already fingerprints as external credential files).
    for cred_path in (
        "~/.config/github-copilot/hosts.json",
        "~/.config/github-copilot/apps.json",
    ):
        try:
            expanded = os.path.expanduser(cred_path)
            if os.path.isfile(expanded) and os.path.getsize(expanded) > 2:
                return True, cred_path
        except OSError:
            continue
    return False, None


def get_external_process_provider_status(provider_id: str) -> Dict[str, Any]:
    """Status snapshot for providers that run a local subprocess.

    ``configured``/``logged_in`` stay structural (the executable resolves or a
    TCP endpoint is set) because the spawned subprocess owns its real auth.
    ``auth_verified``/``auth_source`` carry positive credential evidence when
    Hermes can actually see some — absence of evidence is not absence of auth.
    """
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "external_process":
        return {"configured": False}

    command = (
        os.getenv("HERMES_COPILOT_ACP_COMMAND", "").strip()
        or os.getenv("COPILOT_CLI_PATH", "").strip()
        or "copilot"
    )
    raw_args = os.getenv("HERMES_COPILOT_ACP_ARGS", "").strip()
    args = shlex.split(raw_args) if raw_args else ["--acp", "--stdio"]
    base_url = os.getenv(pconfig.base_url_env_var, "").strip() if pconfig.base_url_env_var else ""
    if not base_url:
        base_url = pconfig.inference_base_url

    resolved_command = shutil.which(command) if command else None
    auth_verified, auth_source = _external_process_auth_evidence(provider_id)
    return {
        "configured": bool(resolved_command or base_url.startswith("acp+tcp://")),
        "provider": provider_id,
        "name": pconfig.name,
        "command": command,
        "args": args,
        "resolved_command": resolved_command,
        "base_url": base_url,
        "logged_in": bool(resolved_command or base_url.startswith("acp+tcp://")),
        "auth_verified": auth_verified,
        "auth_source": auth_source,
    }


def get_auth_status(provider_id: Optional[str] = None) -> Dict[str, Any]:
    """Generic auth status dispatcher."""
    target = (provider_id or get_active_provider() or "").strip().lower()
    if not target:
        return {"logged_in": False}
    if target == "spotify":
        return get_spotify_auth_status()
    if target == "nous":
        return get_nous_auth_status()
    if target == "openai-codex":
        return get_codex_auth_status()
    if target == "xai-oauth":
        return get_xai_oauth_auth_status()
    if target == "qwen-oauth":
        return get_qwen_auth_status()
    if target == "minimax-oauth":
        return get_minimax_oauth_auth_status()
    if target == "azure-foundry":
        return _get_azure_foundry_auth_status()
    pconfig = PROVIDER_REGISTRY.get(target)
    # External-process providers (copilot-acp today; kiro/devin/junie-style ACP
    # backends tomorrow) — dispatch on auth_type, not a hardcoded slug, so every
    # provider of this class gets a real status instead of the
    # ``{"logged_in": False}`` fallthrough.
    if pconfig and pconfig.auth_type == "external_process":
        return get_external_process_provider_status(target)
    # API-key providers
    if pconfig and pconfig.auth_type == "api_key":
        return get_api_key_provider_status(target)
    # AWS SDK providers (Bedrock) — check via boto3 credential chain
    if pconfig and pconfig.auth_type == "aws_sdk":
        try:
            from agent.bedrock_adapter import has_aws_credentials
            return {"logged_in": has_aws_credentials(), "provider": target}
        except ImportError:
            return {"logged_in": False, "provider": target, "error": "boto3 not installed"}
    return {"logged_in": False}


def _get_azure_foundry_auth_status() -> Dict[str, Any]:
    """Return structural auth status for Azure Foundry.

    ``logged_in`` is structural, matching other non-OAuth provider status
    checks:

      * ``auth_mode == "entra_id"`` AND ``azure-identity`` is importable
        (we do NOT mint a token here; ``hermes doctor`` runs the live
        probe and reports whether the credential chain can acquire one).
      * ``auth_mode == "api_key"`` (default) AND ``AZURE_FOUNDRY_API_KEY``
        is set with a usable value.

    Never invokes the Entra credential chain — keeps CLI startup latency
    flat regardless of token-service / az login state.
    """
    info: Dict[str, Any] = {"provider": "azure-foundry"}
    try:
        from hermes_cli.config import load_config, get_env_value_prefer_dotenv
        cfg = load_config()
    except Exception:
        cfg = {}

    model_cfg = cfg.get("model") if isinstance(cfg, dict) else None
    auth_mode = "api_key"
    base_url = ""
    if isinstance(model_cfg, dict):
        auth_mode = str(model_cfg.get("auth_mode") or "api_key").strip().lower() or "api_key"
        base_url = str(model_cfg.get("base_url") or "").strip()
    info["auth_mode"] = auth_mode
    info["base_url"] = base_url

    if auth_mode == "entra_id":
        try:
            from agent.azure_identity_adapter import (
                EntraIdentityConfig,
                SCOPE_AI_AZURE_DEFAULT,
                has_azure_identity_installed,
            )
            installed = has_azure_identity_installed()
            entra_cfg = {}
            if isinstance(model_cfg, dict) and isinstance(model_cfg.get("entra"), dict):
                entra_cfg = model_cfg["entra"]
            identity_config = EntraIdentityConfig.from_dict(
                entra_cfg,
                default_scope=SCOPE_AI_AZURE_DEFAULT,
            )
            info["azure_identity_installed"] = installed
            info["scope"] = identity_config.scope
            info["credential_probe"] = "not_run"
            info["credential_verified"] = False
            info["logged_in"] = bool(installed)
            if not installed:
                info["hint"] = (
                    "azure-identity not installed. Install with: "
                    "pip install azure-identity  (or rely on Hermes' "
                    "lazy-install at first use)."
                )
            else:
                info["hint"] = (
                    "azure-identity is installed; live credential validation "
                    "is skipped here. Run `hermes doctor` to verify token acquisition."
                )
            return info
        except Exception as exc:
            info["logged_in"] = False
            info["error"] = f"azure-identity check failed: {exc}"
            return info

    # api_key mode (default)
    try:
        api_key = get_env_value_prefer_dotenv("AZURE_FOUNDRY_API_KEY") or ""
    except Exception:
        api_key = os.getenv("AZURE_FOUNDRY_API_KEY", "")
    info["logged_in"] = has_usable_secret(api_key)
    return info


def resolve_api_key_provider_credentials(provider_id: str) -> Dict[str, Any]:
    """Resolve API key and base URL for an API-key provider.

    Returns dict with: provider, api_key, base_url, source.
    """
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "api_key":
        raise AuthError(
            f"Provider '{provider_id}' is not an API-key provider.",
            provider=provider_id,
            code="invalid_provider",
        )

    api_key = ""
    key_source = ""
    api_key, key_source = _resolve_api_key_provider_secret(provider_id, pconfig)

    # No-auth LM Studio: substitute a placeholder so runtime / auxiliary_client
    # see the local server as configured. doctor still reports unconfigured
    # because get_api_key_provider_status uses the raw secret resolver.
    if not api_key and provider_id == "lmstudio":
        api_key = LMSTUDIO_NOAUTH_PLACEHOLDER
        key_source = key_source or "default"

    env_url = ""
    if pconfig.base_url_env_var:
        env_url = os.getenv(pconfig.base_url_env_var, "").strip()

    if provider_id in {"kimi-coding", "kimi-coding-cn"}:
        base_url = _resolve_kimi_base_url(api_key, pconfig.inference_base_url, env_url)
    elif provider_id == "zai":
        base_url = _resolve_zai_base_url(api_key, pconfig.inference_base_url, env_url)
    elif provider_id == "copilot":
        # Resolve the Copilot API base URL from the token-exchange response
        # (endpoints.api, with a proxy-ep fallback), which is authoritative
        # for Enterprise / proxied accounts. Falls back to the registry
        # default and is guarded non-empty below so chat inference never
        # resolves an empty base URL (#50252).
        base_url = env_url.rstrip("/") if env_url else pconfig.inference_base_url
        try:
            from hermes_cli.copilot_auth import (
                resolve_copilot_token,
                get_copilot_api_token,
            )
            raw_token, _ = resolve_copilot_token()
            if raw_token:
                _, resolved = get_copilot_api_token(raw_token)
                resolved = (resolved or "").strip()
                if resolved:
                    base_url = resolved
        except Exception as exc:
            logger.debug("Copilot base URL resolution fell back to default: %s", exc)
    elif env_url:
        base_url = env_url.rstrip("/")
    else:
        base_url = pconfig.inference_base_url

    if provider_id == "lmstudio":
        base_url = _normalize_lmstudio_runtime_base_url(base_url)

    if provider_id == "actual":
        base_url = normalize_actual_base_url(base_url)

    # Last-resort guard: an API-key provider must never hand back an empty
    # base URL (a set-but-empty COPILOT_API_BASE_URL or similar env override
    # otherwise wedges chat inference — #50252).
    if not (isinstance(base_url, str) and base_url.strip()):
        base_url = pconfig.inference_base_url

    if not api_key and provider_id == "actual" and is_actual_local_base_url(base_url):
        api_key = ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
        key_source = key_source or "local-offline"

    return {
        "provider": provider_id,
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "source": key_source or "default",
    }


def resolve_external_process_provider_credentials(provider_id: str) -> Dict[str, Any]:
    """Resolve runtime details for local subprocess-backed providers."""
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "external_process":
        raise AuthError(
            f"Provider '{provider_id}' is not an external-process provider.",
            provider=provider_id,
            code="invalid_provider",
        )

    base_url = os.getenv(pconfig.base_url_env_var, "").strip() if pconfig.base_url_env_var else ""
    if not base_url:
        base_url = pconfig.inference_base_url

    # How to launch the CLI comes from the provider's own profile, so a provider
    # shipped outside this tree describes its binary/args instead of inheriting
    # another vendor's. copilot-acp's values live in its profile, which is why
    # HERMES_COPILOT_ACP_COMMAND / COPILOT_CLI_PATH / HERMES_COPILOT_ACP_ARGS
    # keep working unchanged.
    profile = None
    try:
        from providers import get_provider_profile as _get_provider_profile

        profile = _get_provider_profile(provider_id)
    except Exception:
        profile = None

    command_env_vars = tuple(getattr(profile, "process_command_env_vars", ()) or ())
    default_command = str(getattr(profile, "process_command", "") or "")
    default_args = list(getattr(profile, "process_args", ()) or [])
    args_env_var = str(getattr(profile, "process_args_env_var", "") or "")

    command = ""
    for _var in command_env_vars:
        command = os.getenv(_var, "").strip()
        if command:
            break
    if not command:
        command = default_command

    raw_args = os.getenv(args_env_var, "").strip() if args_env_var else ""
    args = shlex.split(raw_args) if raw_args else list(default_args)

    resolved_command = shutil.which(command) if command else None
    if not resolved_command and not base_url.startswith("acp+tcp://"):
        _hint = (
            " or set " + "/".join(command_env_vars) if command_env_vars else ""
        )
        raise AuthError(
            f"Could not find the '{provider_id}' CLI command "
            f"'{command or '(none configured)'}'. Install it{_hint}.",
            provider=provider_id,
            code="missing_external_process_cli",
        )

    return {
        "provider": provider_id,
        # Placeholder credential: the subprocess owns real auth. Keyed on the
        # provider id so each external-process provider gets a distinct value.
        "api_key": pconfig.id or provider_id,
        "base_url": base_url.rstrip("/"),
        "command": resolved_command or command,
        "args": args,
        "source": "process",
    }


# =============================================================================
# CLI Commands — login / logout
# =============================================================================

def _update_config_for_provider(
    provider_id: str,
    inference_base_url: str,
    default_model: Optional[str] = None,
) -> Path:
    """Update config.yaml and auth.json to reflect the active provider.

    When *default_model* is provided the function also writes it as the
    ``model.default`` value.  This prevents a race condition where the
    gateway (which re-reads config per-message) picks up the new provider
    before the caller has finished model selection, resulting in a
    mismatched model/provider (e.g. ``anthropic/claude-opus-4.6`` sent to
    MiniMax's API).
    """
    # Set active_provider in auth.json so auto-resolution picks this provider
    with _auth_store_lock():
        auth_store = _load_auth_store()
        auth_store["active_provider"] = provider_id
        _save_auth_store(auth_store)

    # Update config.yaml model section
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    require_readable_config_before_write(config_path)

    config = read_raw_config()

    current_model = config.get("model")
    if isinstance(current_model, dict):
        model_cfg = dict(current_model)
    elif isinstance(current_model, str) and current_model.strip():
        model_cfg = {"default": current_model.strip()}
    else:
        model_cfg = {}

    model_cfg["provider"] = provider_id
    if inference_base_url and inference_base_url.strip():
        model_cfg["base_url"] = inference_base_url.rstrip("/")
    else:
        # Clear stale base_url to prevent contamination when switching providers
        model_cfg.pop("base_url", None)

    # Clear stale endpoint credentials left over from a previous custom provider.
    # Built-in providers resolve credentials from env/auth state, not inline
    # model.api_key.
    from hermes_cli.config import clear_model_endpoint_credentials

    clear_model_endpoint_credentials(model_cfg)

    # When switching to a non-OpenRouter provider, ensure model.default is
    # valid for the new provider.  An OpenRouter-formatted name like
    # "anthropic/claude-opus-4.6" will fail on direct-API providers.
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
    if not config:
        return None
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    provider = model.get("provider")
    if not isinstance(provider, str):
        return None
    provider = provider.strip().lower()
    return provider or None


def _config_provider_matches(provider_id: Optional[str]) -> bool:
    """Return True when config.yaml currently selects *provider_id*."""
    if not provider_id:
        return False
    return _get_config_provider() == provider_id.strip().lower()


def _should_reset_config_provider_on_logout(provider_id: Optional[str]) -> bool:
    """Return True when logout should reset the model provider config."""
    if not provider_id:
        return False
    normalized = provider_id.strip().lower()
    return normalized in PROVIDER_REGISTRY and _config_provider_matches(normalized)


def _logout_default_provider_from_config() -> Optional[str]:
    """Fallback logout target when auth.json has no active provider.

    `hermes logout` historically keyed off auth.json.active_provider only.
    That left users stuck when auth state had already been cleared but
    config.yaml still selected an OAuth provider such as openai-codex for the
    agent model: there was no active auth provider to target, so logout printed
    "No provider is currently logged in" and never reset model.provider.
    """
    provider = _get_config_provider()
    if provider in {"nous", "openai-codex", "xai-oauth"}:
        return provider
    return None


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


def _confirm_selection_guards(
    model_id: str,
    *,
    provider: str = "",
    base_url: str = "",
    api_key: str = "",
    include_kinds: Optional[List[str]] = None,
) -> bool:
    """Prompt before saving a model that trips any selection guard.

    Runs the unified guard registry (cost + data-policy + future guards) via
    :mod:`hermes_cli.model_selection_guards` and shows one [y/N] confirm with
    every warning that fired. Returns True to proceed, False to cancel.
    """
    try:
        from hermes_cli.model_selection_guards import (
            combined_message,
            selection_warnings,
        )

        warnings = selection_warnings(
            model_id,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            include_kinds=include_kinds,
        )
    except Exception:
        warnings = []
    if not warnings:
        return True

    print()
    print("=" * 72)
    print(combined_message(warnings))
    print("=" * 72)
    try:
        response = input("Switch anyway? [y/N]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return False
    return response in {"y", "yes"}


def _prompt_model_selection(
    model_ids: List[str],
    current_model: str = "",
    pricing: Optional[Dict[str, Dict[str, str]]] = None,
    unavailable_models: Optional[List[str]] = None,
    portal_url: str = "",
    unavailable_message: str = "",
    confirm_provider: str = "",
    confirm_base_url: str = "",
    confirm_api_key: str = "",
) -> Optional[str]:
    """Interactive model selection. Puts current_model first with a marker. Returns chosen model ID or None.

    If *pricing* is provided (``{model_id: {prompt, completion}}``), a compact
    price indicator is shown next to each model in aligned columns.

    If *unavailable_models* is provided, those models are shown grayed out
    and unselectable, with an upgrade link to *portal_url*.
    """
    from hermes_cli.cli_output import line_input
    from hermes_cli.models import (
        _format_price_per_mtok,
        compute_sale_discount,
    )

    _unavailable = unavailable_models or []
    # Sale chrome (★ / -N% / was) is Nous Portal-only — never for OpenRouter
    # or other providers even if pricing.original is somehow present.
    sale_chrome = (confirm_provider or "").strip().lower() == "nous"

    def _confirmed_selection(mid: str) -> Optional[str]:
        if not mid:
            return None
        # Unified guard registry (hermes_cli.model_selection_guards): the cost
        # guard only runs when a provider is known (pricing lookups need one);
        # id-keyed guards like the data-policy guard always run — they must
        # fire even via a custom endpoint or gateway.
        _kinds = None if confirm_provider else ["data_policy"]
        if not _confirm_selection_guards(
            mid,
            provider=confirm_provider,
            base_url=confirm_base_url,
            api_key=confirm_api_key,
            include_kinds=_kinds,
        ):
            return None
        return mid

    # Reorder: current model first, then the rest (deduplicated)
    ordered = []
    if current_model and current_model in model_ids:
        ordered.append(current_model)
    for mid in model_ids:
        if mid not in ordered:
            ordered.append(mid)

    # All models for column-width computation (selectable + unavailable)
    all_models = list(ordered) + list(_unavailable)

    # Column-aligned labels when pricing is available
    has_pricing = bool(pricing and any(pricing.get(m) for m in all_models))
    # Leave room for a leading "★ " on sale rows (Nous only).
    name_pad = 3 if sale_chrome else 2
    name_col = (
        max((len(m) for m in all_models), default=0) + name_pad
        if has_pricing
        else 0
    )

    # Pre-compute formatted prices and sale chrome.
    # (inp, out, cache, pct|None, was_inp, was_out)
    # Sale chrome is drawn as curses/ANSI segments (yellow % / dim "was"),
    # not baked into a single plain string — curses addnstr would otherwise
    # render escape bytes literally.
    _price_cache: dict[str, tuple[str, str, str, int | None, str, str]] = {}
    price_col = 3  # minimum width
    cache_col = 0  # only set if any model has cache pricing
    has_cache = False
    any_on_sale = False
    _DIM = "\033[2m"
    _RESET = "\033[0m"
    if has_pricing:
        for mid in all_models:
            p = pricing.get(mid)  # type: ignore[union-attr]
            pct: int | None = None
            was_inp = was_out = ""
            if p:
                inp = _format_price_per_mtok(p.get("prompt", ""))
                out = _format_price_per_mtok(p.get("completion", ""))
                cache_read = p.get("input_cache_read", "")
                cache = _format_price_per_mtok(cache_read) if cache_read else ""
                if cache:
                    has_cache = True
                if sale_chrome:
                    sale = compute_sale_discount(
                        p.get("prompt", ""),
                        p.get("completion", ""),
                        p.get("original"),
                    )
                    if sale is not None:
                        any_on_sale = True
                        pct, was_prompt_raw, was_out_raw = sale
                        # Natively-free models (no gateway original) carry
                        # empty was_* raws — leave them empty so the row
                        # shows bare "-100%" with no "was ?/?" suffix.
                        if was_prompt_raw == "" and was_out_raw == "":
                            was_inp = was_out = ""
                        else:
                            was_inp = (
                                _format_price_per_mtok(was_prompt_raw)
                                if was_prompt_raw != ""
                                else "?"
                            )
                            was_out = (
                                _format_price_per_mtok(was_out_raw)
                                if was_out_raw != ""
                                else "?"
                            )
            else:
                inp, out, cache = "", "", ""
            _price_cache[mid] = (inp, out, cache, pct, was_inp, was_out)
            price_col = max(price_col, len(inp), len(out))
            cache_col = max(cache_col, len(cache))
        if has_cache:
            cache_col = max(cache_col, 5)  # minimum: "Cache" header

    def _label_segments(mid):
        """Build a rich radiolist row: yellow ★/% , dim was, plain prices."""
        if not has_pricing:
            segs: list[tuple[str, str | None]] = [(mid, None)]
            if mid == current_model:
                segs.append(("  ← currently in use", None))
            return segs

        inp, out, cache, pct, was_inp, was_out = _price_cache.get(
            mid, ("", "", "", None, "", "")
        )
        on_sale = pct is not None
        # Reserve 2 columns for "★ " so sale and non-sale names share alignment.
        star_w = 2
        if on_sale:
            name_segs: list[tuple[str, str | None]] = [
                ("★ ", "yellow"),
                (f"{mid:<{name_col - star_w}}", None),
            ]
        else:
            name_segs = [(f"{mid:<{name_col}}", None)]

        price_part = f" {inp:>{price_col}}  {out:>{price_col}}"
        if has_cache:
            price_part += f"  {cache:>{cache_col}}"
        segs = [*name_segs, (price_part, None)]
        if on_sale:
            segs.append((f"  -{pct}%", "yellow"))
            if was_inp or was_out:
                segs.append((f"  was {was_inp}/{was_out}", "dim"))
        if mid == current_model:
            segs.append(("  ← currently in use", None))
        return segs

    def _label(mid):
        return "".join(text for text, _style in _label_segments(mid))

    # Default cursor on the current model (index 0 if it was reordered to top)
    default_idx = 0

    # Build a pricing header hint for the menu title
    menu_title = "Select default model:"
    if has_pricing:
        # Align the header with the model column.
        # Each choice is "  {label}" (2 spaces) and we prepend
        # a 3-char cursor region ("-> " or "   "), so content starts at col 5.
        pad = " " * 5
        header = f"\n{pad}{'':>{name_col}} {'In':>{price_col}}  {'Out':>{price_col}}"
        if has_cache:
            header += f"  {'Cache':>{cache_col}}"
        # Legend lives on the column-header line so it reads as a key
        # (★ = on sale), not a fake menu row.
        menu_title += header + "  $/Mtok"
        if any_on_sale:
            menu_title += "  ★ = on sale"

    # Try arrow-key menu first, fall back to number input.
    try:
        from hermes_cli.curses_ui import curses_radiolist

        choices = [_label_segments(mid) for mid in ordered]
        choices.append("Enter custom model name")
        choices.append("Skip (keep current)")

        _upgrade_url = (portal_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
        unavailable_footer = unavailable_message.strip()
        if not unavailable_footer and _unavailable:
            unavailable_footer = f"Upgrade at {_upgrade_url} for paid models"

        # The pricing column header (and any unavailable-models block) is shown
        # as a multi-line description above the list so it survives the curses
        # screen clear. menu_title already embeds the aligned price header.
        desc_lines: list[str] = []
        if has_pricing:
            # menu_title is "Select default model:\n<pad><header>  $/Mtok\n…"
            # Keep only the header/legend portion for the description.
            header_part = menu_title.split("\n", 1)
            if len(header_part) > 1:
                desc_lines.extend(header_part[1].splitlines())
        if _unavailable:
            for mid in _unavailable:
                desc_lines.append(f"   {_label(mid)}")
            desc_lines.append(f"  ── {unavailable_footer} ──")
        description = "\n".join(desc_lines) if desc_lines else None

        # Search haystacks keep pricing labels visible while adding aliases
        # for brand-less wire ids (e.g. Kimi Coding `k3` ↔ query "kimi").
        from hermes_cli.model_search import model_search_text

        model_search_labels = []
        for mid in ordered:
            label = _label(mid)
            haystack = model_search_text(mid)
            # model_search_text always starts with the wire id; only append when
            # aliases add tokens beyond the bare id already in the label.
            model_search_labels.append(
                label if haystack == mid else f"{label} {haystack}"
            )
        model_search_labels.append("Enter custom model name")
        model_search_labels.append("Skip (keep current)")

        idx = curses_radiolist(
            "Select default model:",
            choices,
            selected=default_idx,
            cancel_returns=-1,
            description=description,
            searchable=True,
            search_labels=model_search_labels,
        )
        if idx < 0:
            return None
        print()
        if idx < len(ordered):
            return _confirmed_selection(ordered[idx])
        elif idx == len(ordered):
            try:
                custom = line_input("Enter model name: ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            return _confirmed_selection(custom) if custom else None
        return None
    except (ImportError, NotImplementedError, OSError, subprocess.SubprocessError):
        pass

    # Fallback: numbered list (ANSI colors for sale chrome)
    from hermes_cli.curses_ui import format_radio_item_ansi
    from hermes_cli.colors import Colors, color

    for line in menu_title.splitlines():
        if "★" in line:
            print(line.replace("★", color("★", Colors.YELLOW), 1))
        else:
            print(line)
    num_width = len(str(len(ordered) + 2))
    for i, mid in enumerate(ordered, 1):
        print(f"  {i:>{num_width}}. {format_radio_item_ansi(_label_segments(mid))}")
    n = len(ordered)
    print(f"  {n + 1:>{num_width}}. Enter custom model name")
    print(f"  {n + 2:>{num_width}}. Skip (keep current)")

    if _unavailable:
        _upgrade_url = (portal_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
        unavailable_footer = unavailable_message.strip() or (
            f"Unavailable models (requires paid tier — upgrade at {_upgrade_url})"
        )
        print()
        print(f"  {_DIM}── {unavailable_footer} ──{_RESET}")
        for mid in _unavailable:
            print(f"  {'':>{num_width}}  {_DIM}{_label(mid)}{_RESET}")
    print()

    while True:
        try:
            choice = input(f"Choice [1-{n + 2}] (default: skip): ").strip()
            if not choice:
                return None
            idx = int(choice)
            if 1 <= idx <= n:
                return _confirmed_selection(ordered[idx - 1])
            elif idx == n + 1:
                custom = line_input("Enter model name: ").strip()
                return _confirmed_selection(custom) if custom else None
            elif idx == n + 2:
                return None
            print(f"Please enter 1-{n + 2}")
        except ValueError:
            print("Please enter a number")
        except (KeyboardInterrupt, EOFError):
            return None


def _save_model_choice(model_id: str) -> None:
    """Save the selected model to config.yaml (single source of truth).

    The model is stored in config.yaml only — NOT in .env.  This avoids
    conflicts in multi-agent setups where env vars would stomp each other.
    """
    from hermes_cli.config import save_config, load_config

    config = load_config()
    # Always use dict format so provider/base_url can be stored alongside
    if isinstance(config.get("model"), dict):
        config["model"]["default"] = model_id
    else:
        config["model"] = {"default": model_id}
    save_config(config)


def login_command(args) -> None:
    """Deprecated: use 'hermes model' or 'hermes setup' instead."""
    print("The 'hermes login' command has been removed.")
    print("Use 'hermes auth' to manage credentials,")
    print("'hermes model' to select a provider, or 'hermes setup' for full setup.")
    raise SystemExit(0)


def _login_openai_codex(
    args,
    pconfig: ProviderConfig,
    *,
    force_new_login: bool = False,
) -> None:
    """OpenAI Codex login via device code flow. Tokens stored in ~/.hermes/auth.json."""

    del args, pconfig  # kept for parity with other provider login helpers

    # Check for existing Hermes-owned credentials
    if not force_new_login:
        try:
            existing = resolve_codex_runtime_credentials()
            # Verify the resolved token is actually usable (not expired).
            # resolve_codex_runtime_credentials attempts refresh, so if we get
            # here the token should be valid — but double-check before telling
            # the user "Login successful!".
            _resolved_key = existing.get("api_key", "")
            if isinstance(_resolved_key, str) and _resolved_key and not _codex_access_token_is_expiring(_resolved_key, 60):
                print("Existing Codex credentials found in Hermes auth store.")
                try:
                    reuse = input("Use existing credentials? [Y/n]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    reuse = "y"
                if reuse in {"", "y", "yes"}:
                    config_path = _update_config_for_provider("openai-codex", existing.get("base_url", DEFAULT_CODEX_BASE_URL))
                    print()
                    print("Login successful!")
                    print(f"  Config updated: {config_path} (model.provider=openai-codex)")
                    return
            else:
                print("Existing Codex credentials are expired. Starting fresh login...")
        except AuthError:
            pass

    # Check for existing Codex CLI tokens we can import
    if not force_new_login:
        cli_tokens = _import_codex_cli_tokens()
        if cli_tokens:
            print("Found existing Codex CLI credentials at ~/.codex/auth.json")
            print("Hermes will create its own session to avoid conflicts with Codex CLI / VS Code.")
            try:
                do_import = input("Import these credentials? (a separate login is recommended) [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                do_import = "n"
            if do_import in {"y", "yes"}:
                _save_codex_tokens(cli_tokens)
                base_url = os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/") or DEFAULT_CODEX_BASE_URL
                config_path = _update_config_for_provider("openai-codex", base_url)
                print()
                print("Credentials imported. Note: if Codex CLI refreshes its token,")
                print("Hermes will keep working independently with its own session.")
                print(f"  Config updated: {config_path} (model.provider=openai-codex)")
                return

    # Run a fresh device code flow — Hermes gets its own OAuth session
    print()
    print("Signing in to OpenAI Codex...")
    print("(Hermes creates its own session — won't affect Codex CLI or VS Code)")
    print()

    creds = _codex_device_code_login()

    # Save tokens to Hermes auth store
    _save_codex_tokens(creds["tokens"], creds.get("last_refresh"))
    config_path = _update_config_for_provider("openai-codex", creds.get("base_url", DEFAULT_CODEX_BASE_URL))
    print()
    print("Login successful!")
    from hermes_constants import display_hermes_home as _dhh
    print(f"  Auth state: {_dhh()}/auth.json")
    print(f"  Config updated: {config_path} (model.provider=openai-codex)")


def _login_xai_oauth(
    args,
    pconfig: ProviderConfig,
    *,
    force_new_login: bool = False,
) -> None:
    del pconfig

    if not force_new_login:
        try:
            existing = resolve_xai_oauth_runtime_credentials()
            api_key = existing.get("api_key", "")
            if isinstance(api_key, str) and api_key and not _xai_access_token_is_expiring(api_key, 60):
                print("Existing xAI OAuth credentials found in Hermes auth store.")
                try:
                    reuse = input("Use existing credentials? [Y/n]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    reuse = "y"
                if reuse in {"", "y", "yes"}:
                    config_path = _update_config_for_provider(
                        "xai-oauth",
                        existing.get("base_url", DEFAULT_XAI_OAUTH_BASE_URL),
                    )
                    print()
                    print("Login successful!")
                    print(f"  Config updated: {config_path} (model.provider=xai-oauth)")
                    return
        except AuthError:
            pass

    print()
    print("Signing in to xAI Grok OAuth (SuperGrok / Premium+)...")
    print("(Hermes creates its own local OAuth session)")
    print()

    timeout_seconds = float(getattr(args, "timeout", None) or 20.0)
    open_browser = not getattr(args, "no_browser", False)
    if _is_remote_session():
        open_browser = False

    creds = _xai_oauth_device_code_login(
        timeout_seconds=timeout_seconds,
        open_browser=open_browser,
    )
    _save_xai_oauth_tokens(
        creds["tokens"],
        discovery=creds.get("discovery"),
        redirect_uri=creds.get("redirect_uri", ""),
        last_refresh=creds.get("last_refresh"),
        auth_mode="oauth_device_code",
    )
    # An explicit interactive re-login is a strong signal the user wants the
    # xAI credential re-enabled. ``hermes auth remove xai-oauth`` leaves a
    # ``device_code`` suppression marker that otherwise stops the singleton
    # seed from re-creating the pool entry, so ``hermes auth list`` would show
    # nothing even though the agent still works via the singleton fallback.
    # Clear it here (same helper ``auth_add_command`` uses). This is kept OUT
    # of ``_save_xai_oauth_tokens`` on purpose — that helper is shared with the
    # refresh hot path, which must never mutate suppression state.
    unsuppress_credential_source("xai-oauth", "device_code")
    config_path = _update_config_for_provider("xai-oauth", creds.get("base_url", DEFAULT_XAI_OAUTH_BASE_URL))
    print()
    print("Login successful!")
    from hermes_constants import display_hermes_home as _dhh
    print(f"  Auth state: {_dhh()}/auth.json")
    print(f"  Config updated: {config_path} (model.provider=xai-oauth)")


def _xai_oauth_request_device_code(
    client: httpx.Client,
    *,
    scope: str = XAI_OAUTH_SCOPE,
) -> Dict[str, Any]:
    response = client.post(
        XAI_OAUTH_DEVICE_CODE_URL,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        data={
            "client_id": XAI_OAUTH_CLIENT_ID,
            "scope": scope,
        },
    )
    if response.status_code != 200:
        raise AuthError(
            f"xAI device-code request failed (HTTP {response.status_code})."
            + (f" Response: {response.text.strip()}" if response.text else ""),
            provider="xai-oauth",
            code="device_code_request_failed",
        )
    payload = response.json()
    required = (
        "device_code",
        "user_code",
        "verification_uri",
        "verification_uri_complete",
        "expires_in",
        "interval",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise AuthError(
            f"xAI device-code response missing fields: {', '.join(missing)}",
            provider="xai-oauth",
            code="device_code_invalid",
        )
    return payload


def _xai_oauth_poll_device_token(
    client: httpx.Client,
    *,
    token_endpoint: str,
    device_code: str,
    expires_in: int,
    poll_interval: int,
) -> Dict[str, Any]:
    deadline = time.monotonic() + max(1, int(expires_in))
    current_interval = max(1, int(poll_interval))
    while time.monotonic() < deadline:
        response = client.post(
            token_endpoint,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": XAI_OAUTH_CLIENT_ID,
                "device_code": device_code,
            },
        )
        if response.status_code == 200:
            payload = response.json()
            if not payload.get("access_token"):
                raise AuthError(
                    "xAI device-code token response did not include an access_token.",
                    provider="xai-oauth",
                    code="xai_device_token_invalid",
                )
            if not payload.get("refresh_token"):
                raise AuthError(
                    "xAI device-code token response did not include a refresh_token.",
                    provider="xai-oauth",
                    code="xai_device_token_invalid",
                )
            return payload

        try:
            error_payload = response.json()
        except Exception:
            response.raise_for_status()
            raise AuthError(
                "xAI device-code token polling returned a non-JSON error response.",
                provider="xai-oauth",
                code="xai_device_token_failed",
            )
        error_code = str(error_payload.get("error") or "")
        if error_code == "authorization_pending":
            time.sleep(current_interval)
            continue
        if error_code == "slow_down":
            current_interval = min(current_interval + 1, 30)
            time.sleep(current_interval)
            continue
        description = (
            error_payload.get("error_description")
            or error_payload.get("error")
            or response.text
        )
        raise AuthError(
            f"xAI device-code token polling failed: {description}",
            provider="xai-oauth",
            code="xai_device_token_failed",
        )
    raise AuthError(
        "Timed out waiting for xAI device authorization.",
        provider="xai-oauth",
        code="device_code_timeout",
    )


def _xai_oauth_device_code_login(
    *,
    timeout_seconds: float = 20.0,
    open_browser: bool = True,
) -> Dict[str, Any]:
    discovery = _xai_oauth_discovery(timeout_seconds)
    token_endpoint = discovery["token_endpoint"]
    timeout = httpx.Timeout(max(20.0, timeout_seconds))
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        device_data = _xai_oauth_request_device_code(client)
        verification_url = str(
            device_data.get("verification_uri_complete")
            or device_data["verification_uri"]
        )
        user_code = str(device_data["user_code"])
        expires_in = int(device_data["expires_in"])
        interval = int(device_data["interval"])

        print()
        print("To continue:")
        print(f"  1. Open: {verification_url}")
        print(f"  2. If prompted, enter code: {user_code}")
        if open_browser and not _is_remote_session() and _can_open_graphical_browser():
            try:
                opened = webbrowser.open(verification_url)
            except Exception:
                opened = False
            if opened:
                print("  (Opened browser for verification)")
            else:
                print("  Could not open browser automatically -- use the URL above.")
        print(f"Waiting for approval (polling every {max(1, interval)}s)...")

        payload = _xai_oauth_poll_device_token(
            client,
            token_endpoint=token_endpoint,
            device_code=str(device_data["device_code"]),
            expires_in=expires_in,
            poll_interval=interval,
        )

    access_token = str(payload.get("access_token", "") or "").strip()
    refresh_token = str(payload.get("refresh_token", "") or "").strip()
    if not access_token or not refresh_token:
        raise AuthError(
            "xAI device-code token response was missing required tokens.",
            provider="xai-oauth",
            code="xai_device_token_invalid",
        )
    base_url = _xai_validate_inference_base_url(
        os.getenv("HERMES_XAI_BASE_URL", "").strip().rstrip("/")
        or os.getenv("XAI_BASE_URL", "").strip().rstrip("/"),
        fallback=DEFAULT_XAI_OAUTH_BASE_URL,
    )
    return {
        "tokens": {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "id_token": str(payload.get("id_token", "") or "").strip(),
            "expires_in": payload.get("expires_in"),
            "token_type": str(payload.get("token_type") or "Bearer").strip() or "Bearer",
        },
        "discovery": discovery,
        "redirect_uri": "",
        "base_url": base_url,
        "last_refresh": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "oauth-device-code",
    }


def _codex_device_code_login() -> Dict[str, Any]:
    """Run the OpenAI device code login flow and return credentials dict."""
    import time as _time

    issuer = "https://auth.openai.com"
    client_id = CODEX_OAUTH_CLIENT_ID

    # Step 1: Request device code. OpenAI's auth endpoint rate-limits this
    # request (HTTP 429) when login is attempted too often from the same
    # IP/account — retry with capped backoff (honoring ``Retry-After``)
    # before surfacing a clear, actionable message instead of a bare status.
    resp = None
    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        try:
            with _codex_http_client(timeout=httpx.Timeout(15.0)) as client:
                resp = client.post(
                    f"{issuer}/api/accounts/deviceauth/usercode",
                    json={"client_id": client_id},
                    headers={"Content-Type": "application/json"},
                )
        except Exception as exc:
            raise AuthError(
                f"Failed to request device code: {exc}",
                provider="openai-codex", code="device_code_request_failed",
            )

        if resp.status_code != 429:
            break

        if attempt < max_attempts:
            retry_after = _parse_retry_after_seconds(
                getattr(resp, "headers", None)
            )
            # Exponential backoff (2s, 4s, 8s) capped, preferring the
            # server-provided Retry-After when present.
            delay = retry_after if retry_after is not None else 2 ** attempt
            delay = max(1, min(int(delay), 60))
            print(
                "OpenAI is rate-limiting login requests "
                f"(429); retrying in {delay}s..."
            )
            _time.sleep(delay)

    if resp is not None and resp.status_code == 429:
        retry_after = _parse_retry_after_seconds(getattr(resp, "headers", None))
        wait_hint = (
            f" Try again in about {retry_after}s."
            if retry_after is not None
            else " Wait a minute and run the login again."
        )
        raise AuthError(
            "OpenAI is rate-limiting Codex login requests (HTTP 429). "
            "This is a temporary throttle on OpenAI's side, not a credential "
            f"problem.{wait_hint}",
            provider="openai-codex", code=CODEX_RATE_LIMITED_CODE,
        )

    if resp is None or resp.status_code != 200:
        status = resp.status_code if resp is not None else "unknown"
        raise AuthError(
            f"Device code request returned status {status}.",
            provider="openai-codex", code="device_code_request_error",
        )

    device_data = resp.json()
    user_code = device_data.get("user_code", "")
    device_auth_id = device_data.get("device_auth_id", "")
    poll_interval = max(3, int(device_data.get("interval", "5")))

    if not user_code or not device_auth_id:
        raise AuthError(
            "Device code response missing required fields.",
            provider="openai-codex", code="device_code_incomplete",
        )

    # Step 2: Show user the code
    print("To continue, follow these steps:\n")
    print("  1. Open this URL in your browser:")
    print(f"     \033[94m{issuer}/codex/device\033[0m\n")
    print("  2. Enter this code:")
    print(f"     \033[94m{user_code}\033[0m\n")
    print("Waiting for sign-in... (press Ctrl+C to cancel)")

    # Step 3: Poll for authorization code
    max_wait = 15 * 60  # 15 minutes
    start = _time.monotonic()
    code_resp = None

    try:
        with _codex_http_client(timeout=httpx.Timeout(15.0)) as client:
            while _time.monotonic() - start < max_wait:
                _time.sleep(poll_interval)
                poll_resp = client.post(
                    f"{issuer}/api/accounts/deviceauth/token",
                    json={"device_auth_id": device_auth_id, "user_code": user_code},
                    headers={"Content-Type": "application/json"},
                )

                if poll_resp.status_code == 200:
                    code_resp = poll_resp.json()
                    break
                elif poll_resp.status_code in {403, 404}:
                    continue  # User hasn't completed login yet
                else:
                    raise AuthError(
                        f"Device auth polling returned status {poll_resp.status_code}.",
                        provider="openai-codex", code="device_code_poll_error",
                    )
    except KeyboardInterrupt:
        print("\nLogin cancelled.")
        raise SystemExit(130)

    if code_resp is None:
        raise AuthError(
            "Login timed out after 15 minutes.",
            provider="openai-codex", code="device_code_timeout",
        )

    # Step 4: Exchange authorization code for tokens
    authorization_code = code_resp.get("authorization_code", "")
    code_verifier = code_resp.get("code_verifier", "")
    redirect_uri = f"{issuer}/deviceauth/callback"

    if not authorization_code or not code_verifier:
        raise AuthError(
            "Device auth response missing authorization_code or code_verifier.",
            provider="openai-codex", code="device_code_incomplete_exchange",
        )

    try:
        with _codex_http_client(timeout=httpx.Timeout(15.0)) as client:
            token_resp = client.post(
                CODEX_OAUTH_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": authorization_code,
                    "redirect_uri": redirect_uri,
                    "client_id": client_id,
                    "code_verifier": code_verifier,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
    except Exception as exc:
        raise AuthError(
            f"Token exchange failed: {exc}",
            provider="openai-codex", code="token_exchange_failed",
        )

    if token_resp.status_code == 429:
        retry_after = _parse_retry_after_seconds(
            getattr(token_resp, "headers", None)
        )
        wait_hint = (
            f" Try again in about {retry_after}s."
            if retry_after is not None
            else " Wait a minute and run the login again."
        )
        raise AuthError(
            "OpenAI is rate-limiting Codex login requests (HTTP 429) during "
            "token exchange. This is a temporary throttle on OpenAI's side, "
            f"not a credential problem.{wait_hint}",
            provider="openai-codex", code=CODEX_RATE_LIMITED_CODE,
        )

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
