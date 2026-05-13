"""
Multi-provider authentication system for Hermes Agent.

Supports OAuth device code flows (Nous Portal, future: OpenAI Codex) and
traditional API key providers (OpenRouter, custom endpoints). Auth state
is persisted in ~/.hermes/auth.json with cross-process file locking.

Architecture:
- ProviderConfig registry defines known OAuth providers
- Auth store (auth.json) holds per-provider credential state
- resolve_provider() picks the active provider via priority chain
- resolve_*_runtime_credentials() handles token refresh and key minting
- logout_command() is the CLI entry point for clearing auth
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import shlex
import ssl
import stat
import sys
import base64
import hashlib
import subprocess
import threading
import time
import uuid
import webbrowser
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import yaml

from hermes_cli.config import get_hermes_home, get_config_path, read_raw_config
from hermes_constants import OPENROUTER_BASE_URL
from utils import atomic_replace, atomic_yaml_write, is_truthy_value

logger = logging.getLogger(__name__)

try:
    import fcntl
except Exception:
    fcntl = None
try:
    import msvcrt
except Exception:
    msvcrt = None

# =============================================================================
# Constants
# =============================================================================

AUTH_STORE_VERSION = 1
AUTH_LOCK_TIMEOUT_SECONDS = 15.0

# Nous Portal defaults
DEFAULT_NOUS_PORTAL_URL = "https://portal.nousresearch.com"
DEFAULT_NOUS_INFERENCE_URL = "https://inference-api.nousresearch.com/v1"
DEFAULT_NOUS_CLIENT_ID = "hermes-cli"
DEFAULT_NOUS_SCOPE = "inference:mint_agent_key"
DEFAULT_AGENT_KEY_MIN_TTL_SECONDS = 30 * 60  # 30 minutes
ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120       # refresh 2 min before expiry
DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS = 1     # poll at most every 1s
DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
MINIMAX_OAUTH_CLIENT_ID = "78257093-7e40-4613-99e0-527b14b39113"
MINIMAX_OAUTH_SCOPE = "group_id profile model.completion"
MINIMAX_OAUTH_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:user_code"
MINIMAX_OAUTH_GLOBAL_BASE = "https://api.minimax.io"
MINIMAX_OAUTH_CN_BASE = "https://api.minimaxi.com"
MINIMAX_OAUTH_GLOBAL_INFERENCE = "https://api.minimax.io/anthropic"
MINIMAX_OAUTH_CN_INFERENCE = "https://api.minimaxi.com/anthropic"
MINIMAX_OAUTH_REFRESH_SKEW_SECONDS = 60
DEFAULT_QWEN_BASE_URL = "https://portal.qwen.ai/v1"
DEFAULT_GITHUB_MODELS_BASE_URL = "https://api.githubcopilot.com"
DEFAULT_COPILOT_ACP_BASE_URL = "acp://copilot"
DEFAULT_OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"
STEPFUN_STEP_PLAN_INTL_BASE_URL = "https://api.stepfun.ai/step_plan/v1"
STEPFUN_STEP_PLAN_CN_BASE_URL = "https://api.stepfun.com/step_plan/v1"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120
QWEN_OAUTH_CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"
QWEN_OAUTH_TOKEN_URL = "https://chat.qwen.ai/api/v1/oauth2/token"
QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120
DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL = "https://accounts.spotify.com"
DEFAULT_SPOTIFY_API_BASE_URL = "https://api.spotify.com/v1"
DEFAULT_SPOTIFY_REDIRECT_URI = "http://127.0.0.1:43827/spotify/callback"
SPOTIFY_DOCS_URL = "https://hermes-agent.nousresearch.com/docs/user-guide/features/spotify"
SPOTIFY_DASHBOARD_URL = "https://developer.spotify.com/dashboard"
SPOTIFY_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120
DEFAULT_SPOTIFY_SCOPE = " ".join((
    "user-modify-playback-state",
    "user-read-playback-state",
    "user-read-currently-playing",
    "user-read-recently-played",
    "playlist-read-private",
    "playlist-read-collaborative",
    "playlist-modify-public",
    "playlist-modify-private",
    "user-library-read",
    "user-library-modify",
))
SERVICE_PROVIDER_NAMES: Dict[str, str] = {
    "spotify": "Spotify",
}

# Google Gemini OAuth (google-gemini-cli provider, Cloud Code Assist backend)
DEFAULT_GEMINI_CLOUDCODE_BASE_URL = "cloudcode-pa://google"
GEMINI_OAUTH_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 60  # refresh 60s before expiry

# LM Studio's default no-auth mode still requires *some* non-empty bearer for
# the API-key code paths (auxiliary_client, runtime resolver) to treat the
# provider as configured. This sentinel is sent only to LM Studio, never to
# any remote service.
LMSTUDIO_NOAUTH_PLACEHOLDER = "dummy-lm-api-key"


# =============================================================================
# Provider Registry
# =============================================================================

@dataclass
class ProviderConfig:
    """Describes a known inference provider."""
    id: str
    name: str
    auth_type: str  # "oauth_device_code", "oauth_external", "oauth_minimax", or "api_key"
    portal_base_url: str = ""
    inference_base_url: str = ""
    client_id: str = ""
    scope: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    # For API-key providers: env vars to check (in priority order)
    api_key_env_vars: tuple = ()
    # Optional env var for base URL override
    base_url_env_var: str = ""


PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {
    "nous": ProviderConfig(
        id="nous",
        name="Nous Portal",
        auth_type="oauth_device_code",
        portal_base_url=DEFAULT_NOUS_PORTAL_URL,
        inference_base_url=DEFAULT_NOUS_INFERENCE_URL,
        client_id=DEFAULT_NOUS_CLIENT_ID,
        scope=DEFAULT_NOUS_SCOPE,
    ),
    "openai-codex": ProviderConfig(
        id="openai-codex",
        name="OpenAI Codex",
        auth_type="oauth_external",
        inference_base_url=DEFAULT_CODEX_BASE_URL,
    ),
    "qwen-oauth": ProviderConfig(
        id="qwen-oauth",
        name="Qwen OAuth",
        auth_type="oauth_external",
        inference_base_url=DEFAULT_QWEN_BASE_URL,
    ),
    "google-gemini-cli": ProviderConfig(
        id="google-gemini-cli",
        name="Google Gemini (OAuth)",
        auth_type="oauth_external",
        inference_base_url=DEFAULT_GEMINI_CLOUDCODE_BASE_URL,
    ),
    "lmstudio": ProviderConfig(
        id="lmstudio",
        name="LM Studio",
        auth_type="api_key",
        inference_base_url="http://127.0.0.1:1234/v1",
        api_key_env_vars=("LM_API_KEY",),
        base_url_env_var="LM_BASE_URL",
    ),
    "copilot": ProviderConfig(
        id="copilot",
        name="GitHub Copilot",
        auth_type="api_key",
        inference_base_url=DEFAULT_GITHUB_MODELS_BASE_URL,
        api_key_env_vars=("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"),
        base_url_env_var="COPILOT_API_BASE_URL",
    ),
    "copilot-acp": ProviderConfig(
        id="copilot-acp",
        name="GitHub Copilot ACP",
        auth_type="external_process",
        inference_base_url=DEFAULT_COPILOT_ACP_BASE_URL,
        base_url_env_var="COPILOT_ACP_BASE_URL",
    ),
    "gemini": ProviderConfig(
        id="gemini",
        name="Google AI Studio",
        auth_type="api_key",
        inference_base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key_env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        base_url_env_var="GEMINI_BASE_URL",
    ),
    "zai": ProviderConfig(
        id="zai",
        name="Z.AI / GLM",
        auth_type="api_key",
        inference_base_url="https://api.z.ai/api/paas/v4",
        api_key_env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
        base_url_env_var="GLM_BASE_URL",
    ),
    "kimi-coding": ProviderConfig(
        id="kimi-coding",
        name="Kimi / Moonshot",
        auth_type="api_key",
        # Legacy platform.moonshot.ai keys use this endpoint (OpenAI-compat).
        # sk-kimi- (Kimi Code) keys are auto-redirected to api.kimi.com/coding
        # by _resolve_kimi_base_url() below.
        inference_base_url="https://api.moonshot.ai/v1",
        api_key_env_vars=("KIMI_API_KEY", "KIMI_CODING_API_KEY"),
        base_url_env_var="KIMI_BASE_URL",
    ),
    "kimi-coding-cn": ProviderConfig(
        id="kimi-coding-cn",
        name="Kimi / Moonshot (China)",
        auth_type="api_key",
        inference_base_url="https://api.moonshot.cn/v1",
        api_key_env_vars=("KIMI_CN_API_KEY",),
    ),
    "stepfun": ProviderConfig(
        id="stepfun",
        name="StepFun Step Plan",
        auth_type="api_key",
        inference_base_url=STEPFUN_STEP_PLAN_INTL_BASE_URL,
        api_key_env_vars=("STEPFUN_API_KEY",),
        base_url_env_var="STEPFUN_BASE_URL",
    ),
    "arcee": ProviderConfig(
        id="arcee",
        name="Arcee AI",
        auth_type="api_key",
        inference_base_url="https://api.arcee.ai/api/v1",
        api_key_env_vars=("ARCEEAI_API_KEY",),
        base_url_env_var="ARCEE_BASE_URL",
    ),
    "gmi": ProviderConfig(
        id="gmi",
        name="GMI Cloud",
        auth_type="api_key",
        inference_base_url="https://api.gmi-serving.com/v1",
        api_key_env_vars=("GMI_API_KEY",),
        base_url_env_var="GMI_BASE_URL",
    ),
    "minimax": ProviderConfig(
        id="minimax",
        name="MiniMax",
        auth_type="api_key",
        inference_base_url="https://api.minimax.io/anthropic",
        api_key_env_vars=("MINIMAX_API_KEY",),
        base_url_env_var="MINIMAX_BASE_URL",
    ),
    "minimax-oauth": ProviderConfig(
        id="minimax-oauth",
        name="MiniMax (OAuth \u00b7 minimax.io)",
        auth_type="oauth_minimax",
        portal_base_url=MINIMAX_OAUTH_GLOBAL_BASE,
        inference_base_url=MINIMAX_OAUTH_GLOBAL_INFERENCE,
        client_id=MINIMAX_OAUTH_CLIENT_ID,
        scope=MINIMAX_OAUTH_SCOPE,
        extra={"region": "global", "cn_portal_base_url": MINIMAX_OAUTH_CN_BASE,
               "cn_inference_base_url": MINIMAX_OAUTH_CN_INFERENCE},
    ),
    "anthropic": ProviderConfig(
        id="anthropic",
        name="Anthropic",
        auth_type="api_key",
        inference_base_url="https://api.anthropic.com",
        api_key_env_vars=("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
        base_url_env_var="ANTHROPIC_BASE_URL",
    ),
    "alibaba": ProviderConfig(
        id="alibaba",
        name="Qwen Cloud",
        auth_type="api_key",
        inference_base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_env_vars=("DASHSCOPE_API_KEY",),
        base_url_env_var="DASHSCOPE_BASE_URL",
    ),
    "alibaba-coding-plan": ProviderConfig(
        id="alibaba-coding-plan",
        name="Alibaba Cloud (Coding Plan)",
        auth_type="api_key",
        inference_base_url="https://coding-intl.dashscope.aliyuncs.com/v1",
        api_key_env_vars=("ALIBABA_CODING_PLAN_API_KEY", "DASHSCOPE_API_KEY"),
        base_url_env_var="ALIBABA_CODING_PLAN_BASE_URL",
    ),
    "minimax-cn": ProviderConfig(
        id="minimax-cn",
        name="MiniMax (China)",
        auth_type="api_key",
        inference_base_url="https://api.minimaxi.com/anthropic",
        api_key_env_vars=("MINIMAX_CN_API_KEY",),
        base_url_env_var="MINIMAX_CN_BASE_URL",
    ),
    "deepseek": ProviderConfig(
        id="deepseek",
        name="DeepSeek",
        auth_type="api_key",
        inference_base_url="https://api.deepseek.com/v1",
        api_key_env_vars=("DEEPSEEK_API_KEY",),
        base_url_env_var="DEEPSEEK_BASE_URL",
    ),
    "xai": ProviderConfig(
        id="xai",
        name="xAI",
        auth_type="api_key",
        inference_base_url="https://api.x.ai/v1",
        api_key_env_vars=("XAI_API_KEY",),
        base_url_env_var="XAI_BASE_URL",
    ),
    "nvidia": ProviderConfig(
        id="nvidia",
        name="NVIDIA NIM",
        auth_type="api_key",
        inference_base_url="https://integrate.api.nvidia.com/v1",
        api_key_env_vars=("NVIDIA_API_KEY",),
        base_url_env_var="NVIDIA_BASE_URL",
    ),
    "ai-gateway": ProviderConfig(
        id="ai-gateway",
        name="Vercel AI Gateway",
        auth_type="api_key",
        inference_base_url="https://ai-gateway.vercel.sh/v1",
        api_key_env_vars=("AI_GATEWAY_API_KEY",),
        base_url_env_var="AI_GATEWAY_BASE_URL",
    ),
    "opencode-zen": ProviderConfig(
        id="opencode-zen",
        name="OpenCode Zen",
        auth_type="api_key",
        inference_base_url="https://opencode.ai/zen/v1",
        api_key_env_vars=("OPENCODE_ZEN_API_KEY",),
        base_url_env_var="OPENCODE_ZEN_BASE_URL",
    ),
    "opencode-go": ProviderConfig(
        id="opencode-go",
        name="OpenCode Go",
        auth_type="api_key",
        # OpenCode Go mixes API surfaces by model:
        # - GLM / Kimi use OpenAI-compatible chat completions under /v1
        # - MiniMax models use Anthropic Messages under /v1/messages
        # Keep the provider base at /v1 and select api_mode per-model.
        inference_base_url="https://opencode.ai/zen/go/v1",
        api_key_env_vars=("OPENCODE_GO_API_KEY",),
        base_url_env_var="OPENCODE_GO_BASE_URL",
    ),
    "kilocode": ProviderConfig(
        id="kilocode",
        name="Kilo Code",
        auth_type="api_key",
        inference_base_url="https://api.kilo.ai/api/gateway",
        api_key_env_vars=("KILOCODE_API_KEY",),
        base_url_env_var="KILOCODE_BASE_URL",
    ),
    "huggingface": ProviderConfig(
        id="huggingface",
        name="Hugging Face",
        auth_type="api_key",
        inference_base_url="https://router.huggingface.co/v1",
        api_key_env_vars=("HF_TOKEN",),
        base_url_env_var="HF_BASE_URL",
    ),
    "xiaomi": ProviderConfig(
        id="xiaomi",
        name="Xiaomi MiMo",
        auth_type="api_key",
        inference_base_url="https://api.xiaomimimo.com/v1",
        api_key_env_vars=("XIAOMI_API_KEY",),
        base_url_env_var="XIAOMI_BASE_URL",
    ),
    "tencent-tokenhub": ProviderConfig(
        id="tencent-tokenhub",
        name="Tencent TokenHub",
        auth_type="api_key",
        inference_base_url="https://tokenhub.tencentmaas.com/v1",
        api_key_env_vars=("TOKENHUB_API_KEY",),
        base_url_env_var="TOKENHUB_BASE_URL",
    ),
    "ollama-cloud": ProviderConfig(
        id="ollama-cloud",
        name="Ollama Cloud",
        auth_type="api_key",
        inference_base_url=DEFAULT_OLLAMA_CLOUD_BASE_URL,
        api_key_env_vars=("OLLAMA_API_KEY",),
        base_url_env_var="OLLAMA_BASE_URL",
    ),
    "bedrock": ProviderConfig(
        id="bedrock",
        name="AWS Bedrock",
        auth_type="aws_sdk",
        inference_base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        api_key_env_vars=(),
        base_url_env_var="BEDROCK_BASE_URL",
    ),
    "azure-foundry": ProviderConfig(
        id="azure-foundry",
        name="Azure Foundry",
        auth_type="api_key",
        inference_base_url="",  # User-provided endpoint
        api_key_env_vars=("AZURE_FOUNDRY_API_KEY",),
        base_url_env_var="AZURE_FOUNDRY_BASE_URL",
    ),
}

# Auto-extend PROVIDER_REGISTRY with any api-key provider registered in
# providers/ that is not already declared above.  New providers only need a
# plugins/model-providers/<name>/ plugin — no edits to this file required.
try:
    from providers import list_providers as _list_providers_for_registry
    for _pp in _list_providers_for_registry():
        if _pp.name in PROVIDER_REGISTRY:
            continue
        if _pp.auth_type != "api_key" or not _pp.env_vars:
            continue
        # Skip providers that need custom token resolution or are special-cased
        # in resolve_provider() (copilot/kimi/zai have bespoke token refresh;
        # openrouter/custom are aggregator/user-supplied and handled outside
        # the registry — adding them here breaks runtime_provider resolution
        # that relies on `openrouter not in PROVIDER_REGISTRY`).
        if _pp.name in {"copilot", "kimi-coding", "kimi-coding-cn", "zai", "openrouter", "custom"}:
            continue
        _api_key_vars = tuple(v for v in _pp.env_vars if not v.endswith("_BASE_URL") and not v.endswith("_URL"))
        _base_url_var = next((v for v in _pp.env_vars if v.endswith("_BASE_URL") or v.endswith("_URL")), None)
        PROVIDER_REGISTRY[_pp.name] = ProviderConfig(
            id=_pp.name,
            name=_pp.display_name or _pp.name,
            auth_type="api_key",
            inference_base_url=_pp.base_url,
            api_key_env_vars=_api_key_vars or _pp.env_vars,
            base_url_env_var=_base_url_var or "",
        )
        # Also register aliases so resolve_provider() resolves them
        for _alias in _pp.aliases:
            if _alias not in PROVIDER_REGISTRY:
                PROVIDER_REGISTRY[_alias] = PROVIDER_REGISTRY[_pp.name]
except Exception:
    pass


# =============================================================================
# Anthropic Key Helper
# =============================================================================

def get_anthropic_key() -> str:
    """Return the first usable Anthropic credential, or ``""``.

    Checks both the ``.env`` file (via ``get_env_value``) and the process
    environment (``os.getenv``).  The fallback order mirrors the
    ``PROVIDER_REGISTRY["anthropic"].api_key_env_vars`` tuple:

        ANTHROPIC_API_KEY -> ANTHROPIC_TOKEN -> CLAUDE_CODE_OAUTH_TOKEN
    """
    from hermes_cli.config import get_env_value

    for var in PROVIDER_REGISTRY["anthropic"].api_key_env_vars:
        value = get_env_value(var) or os.getenv(var, "")
        if value:
            return value
    return ""


# =============================================================================
# Kimi Code Endpoint Detection
# =============================================================================

# Kimi Code (kimi.com/code) issues keys prefixed "sk-kimi-" that only work
# on api.kimi.com/coding.  Legacy keys from platform.moonshot.ai work on
# api.moonshot.ai/v1 (the old default).  Auto-detect when user hasn't set
# KIMI_BASE_URL explicitly.
#
# Note: the base URL intentionally has NO /v1 suffix.  The /coding endpoint
# speaks the Anthropic Messages protocol, and the anthropic SDK appends
# "/v1/messages" internally — so "/coding" + SDK suffix → "/coding/v1/messages"
# (the correct target). Using "/coding/v1" here would produce
# "/coding/v1/v1/messages" (a 404).
KIMI_CODE_BASE_URL = "https://api.kimi.com/coding"


def _resolve_kimi_base_url(api_key: str, default_url: str, env_override: str) -> str:
    """Return the correct Kimi base URL based on the API key prefix.

    If the user has explicitly set KIMI_BASE_URL, that always wins.
    Otherwise, sk-kimi- prefixed keys route to api.kimi.com/coding/v1.
    """
    if env_override:
        return env_override
    # No key → nothing to infer from.  Return default without inspecting.
    if not api_key:
        return default_url
    if api_key.startswith("sk-kimi-"):
        return KIMI_CODE_BASE_URL
    return default_url



_PLACEHOLDER_SECRET_VALUES = {
    "*",
    "**",
    "***",
    "changeme",
    "your_api_key",
    "your-api-key",
    "placeholder",
    "example",
    "dummy",
    "null",
    "none",
}


def has_usable_secret(value: Any, *, min_length: int = 4) -> bool:
    """Return True when a configured secret looks usable, not empty/placeholder."""
    if not isinstance(value, str):
        return False
    cleaned = value.strip()
    if len(cleaned) < min_length:
        return False
    if cleaned.lower() in _PLACEHOLDER_SECRET_VALUES:
        return False
    return True


def _resolve_api_key_provider_secret(
    provider_id: str, pconfig: ProviderConfig
) -> tuple[str, str]:
    """Resolve an API-key provider's token and indicate where it came from."""
    if provider_id == "copilot":
        # Use the dedicated copilot auth module for proper token validation
        try:
            from hermes_cli.copilot_auth import resolve_copilot_token, get_copilot_api_token
            token, source = resolve_copilot_token()
            if token:
                return get_copilot_api_token(token), source
        except ValueError as exc:
            logger.warning("Copilot token validation failed: %s", exc)
        except Exception:
            pass
        return "", ""

    from hermes_cli.config import get_env_value
    for env_var in pconfig.api_key_env_vars:
        # Check both os.environ and ~/.hermes/.env file
        val = (get_env_value(env_var) or "").strip()
        if has_usable_secret(val):
            return val, env_var

    # Fallback: try credential pool (e.g. zai key stored via auth.json)
    try:
        from agent.credential_pool import load_pool
        pool = load_pool(provider_id)
        if pool and pool.has_credentials():
            entry = pool.peek()
            if entry:
                key = getattr(entry, "access_token", "") or getattr(entry, "runtime_api_key", "")
                key = str(key).strip()
                if has_usable_secret(key):
                    return key, f"credential_pool:{provider_id}"
    except Exception:
        pass

    return "", ""


# =============================================================================
# Z.AI Endpoint Detection
# =============================================================================

# Z.AI has separate billing for general vs coding plans, and global vs China
# endpoints.  A key that works on one may return "Insufficient balance" on
# another.  We probe at setup time and store the working endpoint.
# Each entry lists candidate models to try in order — newer coding plan accounts
# may only have access to recent models (glm-5.1, glm-5v-turbo) while older
# ones still use glm-4.7.

ZAI_ENDPOINTS = [
    # (id, base_url, probe_models, label)
    ("global",        "https://api.z.ai/api/paas/v4",        ["glm-5"],   "Global"),
    ("cn",            "https://open.bigmodel.cn/api/paas/v4", ["glm-5"],   "China"),
    ("coding-global", "https://api.z.ai/api/coding/paas/v4",  ["glm-5.1", "glm-5v-turbo", "glm-4.7"], "Global (Coding Plan)"),
    ("coding-cn",     "https://open.bigmodel.cn/api/coding/paas/v4", ["glm-5.1", "glm-5v-turbo", "glm-4.7"], "China (Coding Plan)"),
]


def detect_zai_endpoint(api_key: str, timeout: float = 8.0) -> Optional[Dict[str, str]]:
    """Probe z.ai endpoints to find one that accepts this API key.

    Returns {"id": ..., "base_url": ..., "model": ..., "label": ...} for the
    first working endpoint, or None if all fail.  For endpoints with multiple
    candidate models, tries each in order and returns the first that succeeds.
    """
    for ep_id, base_url, probe_models, label in ZAI_ENDPOINTS:
        for model in probe_models:
            try:
                resp = httpx.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "stream": False,
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "ping"}],
                    },
                    timeout=timeout,
                )
                if resp.status_code == 200:
                    logger.debug("Z.AI endpoint probe: %s (%s) model=%s OK", ep_id, base_url, model)
                    return {
                        "id": ep_id,
                        "base_url": base_url,
                        "model": model,
                        "label": label,
                    }
                logger.debug("Z.AI endpoint probe: %s model=%s returned %s", ep_id, model, resp.status_code)
            except Exception as exc:
                logger.debug("Z.AI endpoint probe: %s model=%s failed: %s", ep_id, model, exc)
    return None


def _resolve_zai_base_url(api_key: str, default_url: str, env_override: str) -> str:
    """Return the correct Z.AI base URL by probing endpoints.

    If the user has explicitly set GLM_BASE_URL, that always wins.
    Otherwise, probe the candidate endpoints to find one that accepts the
    key.  The detected endpoint is cached in provider state (auth.json) keyed
    on a hash of the API key so subsequent starts skip the probe.
    """
    if env_override:
        return env_override

    # No API key set → don't probe (would fire N×M HTTPS requests with an
    # empty Bearer token, all returning 401).  This path is hit during
    # auxiliary-client auto-detection when the user has no Z.AI credentials
    # at all — the caller discards the result immediately, so the probe is
    # pure latency for every AIAgent construction.
    if not api_key:
        return default_url

    # Check provider-state cache for a previously-detected endpoint.
    auth_store = _load_auth_store()
    state = _load_provider_state(auth_store, "zai") or {}
    cached = state.get("detected_endpoint")
    if isinstance(cached, dict) and cached.get("base_url"):
        key_hash = cached.get("key_hash", "")
        if key_hash == hashlib.sha256(api_key.encode()).hexdigest()[:16]:
            logger.debug("Z.AI: using cached endpoint %s", cached["base_url"])
            return cached["base_url"]

    # Probe — may take up to ~8s per endpoint.
    detected = detect_zai_endpoint(api_key)
    if detected and detected.get("base_url"):
        # Persist the detection result keyed on the API key hash.
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        state["detected_endpoint"] = {
            "base_url": detected["base_url"],
            "endpoint_id": detected.get("id", ""),
            "model": detected.get("model", ""),
            "label": detected.get("label", ""),
            "key_hash": key_hash,
        }
        _save_provider_state(auth_store, "zai", state)
        logger.info("Z.AI: auto-detected endpoint %s (%s)", detected["label"], detected["base_url"])
        return detected["base_url"]

    logger.debug("Z.AI: probe failed, falling back to default %s", default_url)
    return default_url


# =============================================================================
# Error Types
# =============================================================================

class AuthError(RuntimeError):
    """Structured auth error with UX mapping hints."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        code: Optional[str] = None,
        relogin_required: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.relogin_required = relogin_required


def format_auth_error(error: Exception) -> str:
    """Map auth failures to concise user-facing guidance."""
    if not isinstance(error, AuthError):
        return str(error)

    if error.relogin_required:
        return f"{error} Run `hermes model` to re-authenticate."

    if error.code == "subscription_required":
        return (
            "No active paid subscription found on Nous Portal. "
            "Please purchase/activate a subscription, then retry."
        )

    if error.code == "insufficient_credits":
        return (
            "Subscription credits are exhausted. "
            "Top up/renew credits in Nous Portal, then retry."
        )

    if error.code == "temporarily_unavailable":
        return f"{error} Please retry in a few seconds."

    return str(error)


def _token_fingerprint(token: Any) -> Optional[str]:
    """Return a short hash fingerprint for telemetry without leaking token bytes."""
    if not isinstance(token, str):
        return None
    cleaned = token.strip()
    if not cleaned:
        return None
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12]


def _oauth_trace_enabled() -> bool:
    raw = os.getenv("HERMES_OAUTH_TRACE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _oauth_trace(event: str, *, sequence_id: Optional[str] = None, **fields: Any) -> None:
    if not _oauth_trace_enabled():
        return
    payload: Dict[str, Any] = {"event": event}
    if sequence_id:
        payload["sequence_id"] = sequence_id
    payload.update(fields)
    logger.info("oauth_trace %s", json.dumps(payload, sort_keys=True, ensure_ascii=False))


# =============================================================================
# Auth Store — persistence layer for ~/.hermes/auth.json
# =============================================================================

def _auth_file_path() -> Path:
    path = get_hermes_home() / "auth.json"
    # Seat belt: if pytest is running and HERMES_HOME resolves to the real
    # user's auth store, refuse rather than silently corrupt it. This catches
    # tests that forgot to monkeypatch HERMES_HOME, tests invoked without the
    # hermetic conftest, or sandbox escapes via threads/subprocesses. In
    # production (no PYTEST_CURRENT_TEST) this is a single dict lookup.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        real_home_auth = (Path.home() / ".hermes" / "auth.json").resolve(strict=False)
        try:
            resolved = path.resolve(strict=False)
        except Exception:
            resolved = path
        if resolved == real_home_auth:
            raise RuntimeError(
                f"Refusing to touch real user auth store during test run: {path}. "
                "Set HERMES_HOME to a tmp_path in your test fixture, or run "
                "via scripts/run_tests.sh for hermetic CI-parity env."
            )
    return path


def _global_auth_file_path() -> Optional[Path]:
    """Return the global-root auth.json when the process is in profile mode.

    Returns ``None`` when the profile and global root resolve to the same
    directory (classic mode, or custom HERMES_HOME that is not a profile).
    Used by read-only fallback paths so providers authed at the root are
    visible to profile processes that haven't configured them locally.

    See issue #18594 follow-up (credential_pool shadowing).
    """
    try:
        from hermes_constants import get_default_hermes_root
        global_root = get_default_hermes_root()
    except Exception:
        return None
    profile_home = get_hermes_home()
    try:
        if profile_home.resolve(strict=False) == global_root.resolve(strict=False):
            return None
    except Exception:
        if profile_home == global_root:
            return None
    # No pytest seat belt here: this is a pure read-only path, and
    # ``_load_global_auth_store()`` wraps the read in a try/except so an
    # unreadable global file can never break the profile process.  The
    # write-side seat belt still lives on ``_auth_file_path()`` where it
    # belongs (that's what protects the real user's auth store from being
    # corrupted by a mis-configured test).
    return global_root / "auth.json"


def _load_global_auth_store() -> Dict[str, Any]:
    """Load the global-root auth store (read-only fallback).

    Returns an empty dict when no global fallback exists (classic mode,
    or the global auth.json is absent). Never raises on missing file.

    Seat belt: under pytest, refuses to read the real user's
    ``~/.hermes/auth.json`` even when HERMES_HOME is set to a profile
    path. The hermetic conftest does not redirect ``HOME``, so
    ``get_default_hermes_root()`` for a profile-shaped HERMES_HOME can
    still resolve to the real user's home on a dev machine. That would
    leak real credentials into tests. This guard uses the unmodified
    ``HOME`` env var (what ``os.path.expanduser('~')`` would resolve to),
    not ``Path.home()``, because ``Path.home`` is sometimes monkeypatched
    by fixtures that want to relocate the global root to a tmp path.
    """
    global_path = _global_auth_file_path()
    if global_path is None or not global_path.exists():
        return {}
    if os.environ.get("PYTEST_CURRENT_TEST"):
        real_home_env = os.environ.get("HOME", "")
        if real_home_env:
            real_root = Path(real_home_env) / ".hermes" / "auth.json"
            try:
                if global_path.resolve(strict=False) == real_root.resolve(strict=False):
                    return {}
            except Exception:
                pass
    try:
        return _load_auth_store(global_path)
    except Exception:
        # A malformed global store must not break profile reads. The
        # profile's own auth store is still authoritative.
        return {}


def _auth_lock_path() -> Path:
    return _auth_file_path().with_suffix(".lock")


_auth_lock_holder = threading.local()


@contextmanager
def _file_lock(
    lock_path: Path,
    holder: threading.local,
    timeout_seconds: float,
    timeout_message: str,
):
    """Cross-process advisory flock helper.

    Reentrant per-thread via ``holder.depth``. Falls back to a depth-only
    guard when neither ``fcntl`` nor ``msvcrt`` is available (rare).
    Callers supply their own ``threading.local`` so independent locks
    (e.g. profile auth.json vs shared Nous store) don't share reentrancy
    state — that would let one lock's reentrant acquisition silently skip
    the other's kernel-level flock.
    """
    if getattr(holder, "depth", 0) > 0:
        holder.depth += 1
        try:
            yield
        finally:
            holder.depth -= 1
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if fcntl is None and msvcrt is None:
        holder.depth = 1
        try:
            yield
        finally:
            holder.depth = 0
        return

    # On Windows, msvcrt.locking needs the file to have content and the
    # file pointer at position 0. Ensure the lock file has at least 1 byte.
    if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
        lock_path.write_text(" ", encoding="utf-8")

    with lock_path.open("r+" if msvcrt else "a+", encoding="utf-8") as lock_file:
        deadline = time.monotonic() + max(1.0, timeout_seconds)
        while True:
            try:
                if fcntl:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
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
            if fcntl:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            elif msvcrt:
                try:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass


@contextmanager
def _auth_store_lock(timeout_seconds: float = AUTH_LOCK_TIMEOUT_SECONDS):
    """Cross-process advisory lock for auth.json reads+writes.  Reentrant.

    Lock ordering invariant: when this lock is held together with
    ``_nous_shared_store_lock``, acquire ``_auth_store_lock`` FIRST
    (outer) and the shared Nous lock SECOND (inner). All runtime
    refresh paths follow this order; violating it risks deadlock
    against a concurrent import on the shared store.
    """
    with _file_lock(
        _auth_lock_path(),
        _auth_lock_holder,
        timeout_seconds,
        "Timed out waiting for auth store lock",
    ):
        yield


def _load_auth_store(auth_file: Optional[Path] = None) -> Dict[str, Any]:
    auth_file = auth_file or _auth_file_path()
    if not auth_file.exists():
        return {"version": AUTH_STORE_VERSION, "providers": {}}

    try:
        raw = json.loads(auth_file.read_text())
    except Exception as exc:
        corrupt_path = auth_file.with_suffix(".json.corrupt")
        try:
            import shutil
            shutil.copy2(auth_file, corrupt_path)
        except Exception:
            pass
        logger.warning(
            "auth: failed to parse %s (%s) — starting with empty store. "
            "Corrupt file preserved at %s",
            auth_file, exc, corrupt_path,
        )
        return {"version": AUTH_STORE_VERSION, "providers": {}}

    if isinstance(raw, dict) and (
        isinstance(raw.get("providers"), dict)
        or isinstance(raw.get("credential_pool"), dict)
    ):
        raw.setdefault("providers", {})
        return raw

    # Migrate from PR's "systems" format if present
    if isinstance(raw, dict) and isinstance(raw.get("systems"), dict):
        systems = raw["systems"]
        providers = {}
        if "nous_portal" in systems:
            providers["nous"] = systems["nous_portal"]
        return {"version": AUTH_STORE_VERSION, "providers": providers,
                "active_provider": "nous" if providers else None}

    return {"version": AUTH_STORE_VERSION, "providers": {}}


def _save_auth_store(auth_store: Dict[str, Any]) -> Path:
    auth_file = _auth_file_path()
    auth_file.parent.mkdir(parents=True, exist_ok=True)
    # Tighten parent dir to 0o700 so siblings can't traverse to creds.
    # No-op on Windows (POSIX mode bits not enforced); ignore failures.
    try:
        os.chmod(auth_file.parent, 0o700)
    except OSError:
        pass
    auth_store["version"] = AUTH_STORE_VERSION
    auth_store["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(auth_store, indent=2) + "\n"
    tmp_path = auth_file.with_name(f"{auth_file.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        # Create with 0o600 atomically via os.open(O_EXCL) + fdopen to close
        # the TOCTOU window where default umask (often 0o644) briefly exposed
        # OAuth tokens to other local users between open() and chmod().
        # Mirrors agent/google_oauth.py (#19673) and tools/mcp_oauth.py (#21148).
        fd = os.open(
            str(tmp_path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        atomic_replace(tmp_path, auth_file)
        try:
            dir_fd = os.open(str(auth_file.parent), os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
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
    # Restrict file permissions to owner only
    try:
        auth_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return auth_file


def _load_provider_state(auth_store: Dict[str, Any], provider_id: str) -> Optional[Dict[str, Any]]:
    providers = auth_store.get("providers")
    if not isinstance(providers, dict):
        return None
    state = providers.get(provider_id)
    return dict(state) if isinstance(state, dict) else None


def _save_provider_state(auth_store: Dict[str, Any], provider_id: str, state: Dict[str, Any]) -> None:
    providers = auth_store.setdefault("providers", {})
    if not isinstance(providers, dict):
        auth_store["providers"] = {}
        providers = auth_store["providers"]
    providers[provider_id] = state
    auth_store["active_provider"] = provider_id


def _store_provider_state(
    auth_store: Dict[str, Any],
    provider_id: str,
    state: Dict[str, Any],
    *,
    set_active: bool = True,
) -> None:
    providers = auth_store.setdefault("providers", {})
    if not isinstance(providers, dict):
        auth_store["providers"] = {}
        providers = auth_store["providers"]
    providers[provider_id] = state
    if set_active:
        auth_store["active_provider"] = provider_id


def is_known_auth_provider(provider_id: str) -> bool:
    normalized = (provider_id or "").strip().lower()
    return normalized in PROVIDER_REGISTRY or normalized in SERVICE_PROVIDER_NAMES


def get_auth_provider_display_name(provider_id: str) -> str:
    normalized = (provider_id or "").strip().lower()
    if normalized in PROVIDER_REGISTRY:
        return PROVIDER_REGISTRY[normalized].name
    return SERVICE_PROVIDER_NAMES.get(normalized, provider_id)


def read_credential_pool(provider_id: Optional[str] = None) -> Dict[str, Any]:
    """Return the persisted credential pool, or one provider slice.

    In profile mode, the profile's credential pool is authoritative. If a
    provider has no entries in the profile, entries from the global-root
    ``auth.json`` are used as a read-only fallback — so workers spawned in a
    profile can see providers that were only authenticated at global scope.

    Profile entries always win: the global fallback only applies per-provider
    when the profile has zero entries for that provider. Once the user runs
    ``hermes auth add <provider>`` inside the profile, profile entries
    fully shadow global for that provider on the next read.

    Writes always go to the profile (``write_credential_pool`` is unchanged).
    See issue #18594 follow-up.
    """
    auth_store = _load_auth_store()
    pool = auth_store.get("credential_pool")
    if not isinstance(pool, dict):
        pool = {}

    global_pool: Dict[str, Any] = {}
    global_store = _load_global_auth_store()
    maybe_global_pool = global_store.get("credential_pool") if global_store else None
    if isinstance(maybe_global_pool, dict):
        global_pool = maybe_global_pool

    if provider_id is None:
        merged = dict(pool)
        for gp_key, gp_entries in global_pool.items():
            if not isinstance(gp_entries, list) or not gp_entries:
                continue
            # Per-provider shadowing: profile wins whenever it has ANY entries.
            existing = merged.get(gp_key)
            if isinstance(existing, list) and existing:
                continue
            merged[gp_key] = list(gp_entries)
        return merged

    provider_entries = pool.get(provider_id)
    if isinstance(provider_entries, list) and provider_entries:
        return list(provider_entries)
    # Profile has no entries for this provider — fall back to global.
    global_entries = global_pool.get(provider_id)
    return list(global_entries) if isinstance(global_entries, list) else []


def write_credential_pool(provider_id: str, entries: List[Dict[str, Any]]) -> Path:
    """Persist one provider's credential pool under auth.json."""
    with _auth_store_lock():
        auth_store = _load_auth_store()
        pool = auth_store.get("credential_pool")
        if not isinstance(pool, dict):
            pool = {}
            auth_store["credential_pool"] = pool
        pool[provider_id] = list(entries)
        return _save_auth_store(auth_store)


def suppress_credential_source(provider_id: str, source: str) -> None:
    """Mark a credential source as suppressed so it won't be re-seeded."""
    with _auth_store_lock():
        auth_store = _load_auth_store()
        suppressed = auth_store.setdefault("suppressed_sources", {})
        provider_list = suppressed.setdefault(provider_id, [])
        if source not in provider_list:
            provider_list.append(source)
        _save_auth_store(auth_store)


def is_source_suppressed(provider_id: str, source: str) -> bool:
    """Check if a credential source has been suppressed by the user."""
    try:
        auth_store = _load_auth_store()
        suppressed = auth_store.get("suppressed_sources", {})
        return source in suppressed.get(provider_id, [])
    except Exception:
        return False


def unsuppress_credential_source(provider_id: str, source: str) -> bool:
    """Clear a suppression marker so the source will be re-seeded on the next load.

    Returns True if a marker was cleared, False if no marker existed.
    """
    with _auth_store_lock():
        auth_store = _load_auth_store()
        suppressed = auth_store.get("suppressed_sources")
        if not isinstance(suppressed, dict):
            return False
        provider_list = suppressed.get(provider_id)
        if not isinstance(provider_list, list) or source not in provider_list:
            return False
        provider_list.remove(source)
        if not provider_list:
            suppressed.pop(provider_id, None)
        if not suppressed:
            auth_store.pop("suppressed_sources", None)
        _save_auth_store(auth_store)
        return True


def get_provider_auth_state(provider_id: str) -> Optional[Dict[str, Any]]:
    """Return persisted auth state for a provider, or None.

    In profile mode, falls back to the global-root ``auth.json`` when the
    profile has no state for this provider. Profile state always wins when
    present. Writes (``_save_auth_store`` / ``persist_*_credentials``) are
    unchanged — they still target the profile only. This mirrors
    ``read_credential_pool``'s per-provider shadowing semantics so that
    ``_seed_from_singletons`` can reseed a profile's credential pool from
    global-scope provider state (e.g. a globally-authenticated Anthropic
    OAuth or Nous device-code session). See issue #18594 follow-up.
    """
    auth_store = _load_auth_store()
    state = _load_provider_state(auth_store, provider_id)
    if state is not None:
        return state
    global_store = _load_global_auth_store()
    if not global_store:
        return None
    return _load_provider_state(global_store, provider_id)


def get_active_provider() -> Optional[str]:
    """Return the currently active provider ID from auth store."""
    auth_store = _load_auth_store()
    return auth_store.get("active_provider")


def is_provider_explicitly_configured(provider_id: str) -> bool:
    """Return True only if the user has explicitly configured this provider.

    Checks:
      1. active_provider in auth.json matches
      2. model.provider in config.yaml matches
      3. Provider-specific env vars are set (e.g. ANTHROPIC_API_KEY)

    This is used to gate auto-discovery of external credentials (e.g.
    Claude Code's ~/.claude/.credentials.json) so they are never used
    without the user's explicit choice.  See PR #4210 for the same
    pattern applied to the setup wizard gate.
    """
    normalized = (provider_id or "").strip().lower()

    # 1. Check auth.json active_provider
    try:
        auth_store = _load_auth_store()
        active = (auth_store.get("active_provider") or "").strip().lower()
        if active and active == normalized:
            return True
    except Exception:
        pass

    # 2. Check config.yaml model.provider
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get("model")
        if isinstance(model_cfg, dict):
            cfg_provider = (model_cfg.get("provider") or "").strip().lower()
            if cfg_provider == normalized:
                return True
    except Exception:
        pass

    # 3. Check provider-specific env vars
    # Exclude CLAUDE_CODE_OAUTH_TOKEN — it's set by Claude Code itself,
    # not by the user explicitly configuring anthropic in Hermes.
    _IMPLICIT_ENV_VARS = {"CLAUDE_CODE_OAUTH_TOKEN"}
    pconfig = PROVIDER_REGISTRY.get(normalized)
    if pconfig and pconfig.auth_type == "api_key":
        for env_var in pconfig.api_key_env_vars:
            if env_var in _IMPLICIT_ENV_VARS:
                continue
            if has_usable_secret(os.getenv(env_var, "")):
                return True

    return False


def clear_provider_auth(provider_id: Optional[str] = None) -> bool:
    """
    Clear auth state for a provider. Used by `hermes logout`.
    If provider_id is None, clears the active provider.
    Returns True if something was cleared.
    """
    with _auth_store_lock():
        auth_store = _load_auth_store()
        target = provider_id or auth_store.get("active_provider")
        if not target:
            return False

        providers = auth_store.get("providers", {})
        if not isinstance(providers, dict):
            providers = {}
            auth_store["providers"] = providers

        pool = auth_store.get("credential_pool")
        if not isinstance(pool, dict):
            pool = {}
            auth_store["credential_pool"] = pool

        cleared = False
        if target in providers:
            del providers[target]
            cleared = True
        if target in pool:
            del pool[target]
            cleared = True

        if auth_store.get("active_provider") == target:
            auth_store["active_provider"] = None
            cleared = True

        if not cleared:
            return False
        _save_auth_store(auth_store)
    return True


def deactivate_provider() -> None:
    """
    Clear active_provider in auth.json without deleting credentials.
    Used when the user switches to a non-OAuth provider (OpenRouter, custom)
    so auto-resolution doesn't keep picking the OAuth provider.
    """
    with _auth_store_lock():
        auth_store = _load_auth_store()
        auth_store["active_provider"] = None
        _save_auth_store(auth_store)


# =============================================================================
# Provider Resolution — picks which provider to use
# =============================================================================


def _get_config_hint_for_unknown_provider(provider_name: str) -> str:
    """Return a helpful hint string when provider resolution fails.

    Checks for common config.yaml mistakes (malformed custom_providers, etc.)
    and returns a human-readable diagnostic, or empty string if nothing found.
    """
    try:
        from hermes_cli.config import validate_config_structure
        issues = validate_config_structure()
        if not issues:
            return ""

        lines = ["Config issue detected — run 'hermes doctor' for full diagnostics:"]
        for ci in issues:
            prefix = "ERROR" if ci.severity == "error" else "WARNING"
            lines.append(f"  [{prefix}] {ci.message}")
            # Show first line of hint
            first_hint = ci.hint.splitlines()[0] if ci.hint else ""
            if first_hint:
                lines.append(f"    → {first_hint}")
        return "\n".join(lines)
    except Exception:
        return ""


def resolve_provider(
    requested: Optional[str] = None,
    *,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> str:
    """
    Determine which inference provider to use.

    Priority (when requested="auto" or None):
    1. active_provider in auth.json with valid credentials
    2. Explicit CLI api_key/base_url -> "openrouter"
    3. OPENAI_API_KEY or OPENROUTER_API_KEY env vars -> "openrouter"
    4. Provider-specific API keys (GLM, Kimi, MiniMax) -> that provider
    5. Fallback: "openrouter"
    """
    normalized = (requested or "auto").strip().lower()

    # Normalize provider aliases
    _PROVIDER_ALIASES = {
        "glm": "zai", "z-ai": "zai", "z.ai": "zai", "zhipu": "zai",
        "google": "gemini", "google-gemini": "gemini", "google-ai-studio": "gemini",
        "x-ai": "xai", "x.ai": "xai", "grok": "xai",
        "kimi": "kimi-coding", "kimi-for-coding": "kimi-coding", "moonshot": "kimi-coding",
        "kimi-cn": "kimi-coding-cn", "moonshot-cn": "kimi-coding-cn",
        "step": "stepfun", "stepfun-coding-plan": "stepfun",
        "arcee-ai": "arcee", "arceeai": "arcee",
        "gmi-cloud": "gmi", "gmicloud": "gmi",
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
        "qwen-portal": "qwen-oauth", "qwen-cli": "qwen-oauth", "qwen-oauth": "qwen-oauth", "google-gemini-cli": "google-gemini-cli", "gemini-cli": "google-gemini-cli", "gemini-oauth": "google-gemini-cli",
        "hf": "huggingface", "hugging-face": "huggingface", "huggingface-hub": "huggingface",
        "mimo": "xiaomi", "xiaomi-mimo": "xiaomi",
        "tencent": "tencent-tokenhub", "tokenhub": "tencent-tokenhub",
        "tencent-cloud": "tencent-tokenhub", "tencentmaas": "tencent-tokenhub",
        "aws": "bedrock", "aws-bedrock": "bedrock", "amazon-bedrock": "bedrock", "amazon": "bedrock",
        "go": "opencode-go", "opencode-go-sub": "opencode-go",
        "kilo": "kilocode", "kilo-code": "kilocode", "kilo-gateway": "kilocode",
        "lmstudio": "lmstudio", "lm-studio": "lmstudio", "lm_studio": "lmstudio",
        # Local server aliases — route through the generic custom provider
        "ollama": "custom", "ollama_cloud": "ollama-cloud",
        "vllm": "custom", "llamacpp": "custom",
        "llama.cpp": "custom", "llama-cpp": "custom",
    }
    # Extend with aliases declared in plugins/model-providers/<name>/ that aren't already mapped.
    # This keeps providers/ as the single source for new aliases while the
    # hardcoded dict above remains authoritative for existing ones.
    try:
        from providers import list_providers as _lp
        for _pp in _lp():
            for _alias in _pp.aliases:
                if _alias not in _PROVIDER_ALIASES:
                    _PROVIDER_ALIASES[_alias] = _pp.name
    except Exception:
        pass
    normalized = _PROVIDER_ALIASES.get(normalized, normalized)

    if normalized == "openrouter":
        return "openrouter"
    if normalized == "custom":
        return "custom"
    if normalized in PROVIDER_REGISTRY:
        return normalized
    if normalized != "auto":
        # Check for common config.yaml issues that cause this error
        _config_hint = _get_config_hint_for_unknown_provider(normalized)
        msg = f"Unknown provider '{normalized}'."
        if _config_hint:
            msg += f"\n\n{_config_hint}"
        else:
            msg += " Check 'hermes model' for available providers, or run 'hermes doctor' to diagnose config issues."
        raise AuthError(msg, code="invalid_provider")

    # Explicit one-off CLI creds always mean openrouter/custom
    if explicit_api_key or explicit_base_url:
        return "openrouter"

    # Check auth store for an active OAuth provider
    try:
        auth_store = _load_auth_store()
        active = auth_store.get("active_provider")
        if active and active in PROVIDER_REGISTRY:
            status = get_auth_status(active)
            if status.get("logged_in"):
                return active
    except Exception as e:
        logger.debug("Could not detect active auth provider: %s", e)

    if has_usable_secret(os.getenv("OPENAI_API_KEY")) or has_usable_secret(os.getenv("OPENROUTER_API_KEY")):
        return "openrouter"

    # Auto-detect API-key providers by checking their env vars
    for pid, pconfig in PROVIDER_REGISTRY.items():
        if pconfig.auth_type != "api_key":
            continue
        # GitHub tokens are commonly present for repo/tool access but should not
        # hijack inference auto-selection unless the user explicitly chooses
        # Copilot/GitHub Models as the provider. LM Studio is a local server
        # whose availability isn't implied by LM_API_KEY presence (it may be
        # offline, and the no-auth setup uses a placeholder value), so it
        # also requires explicit selection.
        if pid in {"copilot", "lmstudio"}:
            continue
        for env_var in pconfig.api_key_env_vars:
            if has_usable_secret(os.getenv(env_var, "")):
                return pid

    # AWS Bedrock — detect via boto3 credential chain (IAM roles, SSO, env vars).
    # This runs after API-key providers so explicit keys always win.
    try:
        from agent.bedrock_adapter import has_aws_credentials
        if has_aws_credentials():
            return "bedrock"
    except ImportError:
        pass  # boto3 not installed — skip Bedrock auto-detection

    raise AuthError(
        "No inference provider configured. Run 'hermes model' to choose a "
        "provider and model, or set an API key (OPENROUTER_API_KEY, "
        "OPENAI_API_KEY, etc.) in ~/.hermes/.env.",
        code="no_provider_configured",
    )


# =============================================================================
# Timestamp / TTL helpers
# =============================================================================

def _parse_iso_timestamp(value: Any) -> Optional[float]:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
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
    if expires_epoch is None:
        return True
    return expires_epoch <= (time.time() + skew_seconds)


def _coerce_ttl_seconds(expires_in: Any) -> int:
    try:
        ttl = int(expires_in)
    except Exception:
        ttl = 0
    return max(0, ttl)


def _optional_base_url(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().rstrip("/")
    return cleaned if cleaned else None


def _decode_jwt_claims(token: Any) -> Dict[str, Any]:
    if not isinstance(token, str) or token.count(".") != 2:
        return {}
    payload = token.split(".")[1]
    payload += "=" * ((4 - len(payload) % 4) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload.encode("utf-8"))
        claims = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return claims if isinstance(claims, dict) else {}


def _codex_access_token_is_expiring(access_token: Any, skew_seconds: int) -> bool:
    claims = _decode_jwt_claims(access_token)
    exp = claims.get("exp")
    if not isinstance(exp, (int, float)):
        return False
    return float(exp) <= (time.time() + max(0, int(skew_seconds)))


def _qwen_cli_auth_path() -> Path:
    return Path.home() / ".qwen" / "oauth_creds.json"


def _read_qwen_cli_tokens() -> Dict[str, Any]:
    auth_path = _qwen_cli_auth_path()
    if not auth_path.exists():
        raise AuthError(
            "Qwen CLI credentials not found. Run 'qwen auth qwen-oauth' first.",
            provider="qwen-oauth",
            code="qwen_auth_missing",
        )
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuthError(
            f"Failed to read Qwen CLI credentials from {auth_path}: {exc}",
            provider="qwen-oauth",
            code="qwen_auth_read_failed",
        ) from exc
    if not isinstance(data, dict):
        raise AuthError(
            f"Invalid Qwen CLI credentials in {auth_path}.",
            provider="qwen-oauth",
            code="qwen_auth_invalid",
        )
    return data


def _save_qwen_cli_tokens(tokens: Dict[str, Any]) -> Path:
    auth_path = _qwen_cli_auth_path()
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(auth_path.parent, 0o700)
    except OSError:
        pass
    # Per-process random temp suffix avoids collisions between concurrent
    # writers and stale leftovers from a crashed prior write.
    tmp_path = auth_path.with_name(f"{auth_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    # Create with 0o600 atomically via os.open(O_EXCL) — closes the TOCTOU
    # window where write_text() + post-write chmod briefly exposed tokens
    # at process umask (typically 0o644). See #19673, #21148.
    fd = os.open(
        str(tmp_path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        stat.S_IRUSR | stat.S_IWUSR,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(tokens, indent=2, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        atomic_replace(tmp_path, auth_path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
    return auth_path


def _qwen_access_token_is_expiring(expiry_date_ms: Any, skew_seconds: int = QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS) -> bool:
    try:
        expiry_ms = int(expiry_date_ms)
    except Exception:
        return True
    return (time.time() + max(0, int(skew_seconds))) * 1000 >= expiry_ms


def _refresh_qwen_cli_tokens(tokens: Dict[str, Any], timeout_seconds: float = 20.0) -> Dict[str, Any]:
    refresh_token = str(tokens.get("refresh_token", "") or "").strip()
    if not refresh_token:
        raise AuthError(
            "Qwen OAuth refresh token missing. Re-run 'qwen auth qwen-oauth'.",
            provider="qwen-oauth",
            code="qwen_refresh_token_missing",
        )

    try:
        response = httpx.post(
            QWEN_OAUTH_TOKEN_URL,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": QWEN_OAUTH_CLIENT_ID,
            },
            timeout=timeout_seconds,
        )
    except Exception as exc:
        raise AuthError(
            f"Qwen OAuth refresh failed: {exc}",
            provider="qwen-oauth",
            code="qwen_refresh_failed",
        ) from exc

    if response.status_code >= 400:
        body = response.text.strip()
        raise AuthError(
            "Qwen OAuth refresh failed. Re-run 'qwen auth qwen-oauth'."
            + (f" Response: {body}" if body else ""),
            provider="qwen-oauth",
            code="qwen_refresh_failed",
        )

    try:
        payload = response.json()
    except Exception as exc:
        raise AuthError(
            f"Qwen OAuth refresh returned invalid JSON: {exc}",
            provider="qwen-oauth",
            code="qwen_refresh_invalid_json",
        ) from exc

    if not isinstance(payload, dict) or not str(payload.get("access_token", "") or "").strip():
        raise AuthError(
            "Qwen OAuth refresh response missing access_token.",
            provider="qwen-oauth",
            code="qwen_refresh_invalid_response",
        )

    expires_in = payload.get("expires_in")
    try:
        expires_in_seconds = int(expires_in)
    except Exception:
        expires_in_seconds = 6 * 60 * 60

    refreshed = {
        "access_token": str(payload.get("access_token", "") or "").strip(),
        "refresh_token": str(payload.get("refresh_token", refresh_token) or refresh_token).strip(),
        "token_type": str(payload.get("token_type", tokens.get("token_type", "Bearer")) or "Bearer").strip() or "Bearer",
        "resource_url": str(payload.get("resource_url", tokens.get("resource_url", "portal.qwen.ai")) or "portal.qwen.ai").strip(),
        "expiry_date": int(time.time() * 1000) + max(1, expires_in_seconds) * 1000,
    }
    _save_qwen_cli_tokens(refreshed)
    return refreshed


def resolve_qwen_runtime_credentials(
    *,
    force_refresh: bool = False,
    refresh_if_expiring: bool = True,
    refresh_skew_seconds: int = QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> Dict[str, Any]:
    tokens = _read_qwen_cli_tokens()
    access_token = str(tokens.get("access_token", "") or "").strip()
    should_refresh = bool(force_refresh)
    if not should_refresh and refresh_if_expiring:
        should_refresh = _qwen_access_token_is_expiring(tokens.get("expiry_date"), refresh_skew_seconds)
    if should_refresh:
        tokens = _refresh_qwen_cli_tokens(tokens)
        access_token = str(tokens.get("access_token", "") or "").strip()
    if not access_token:
        raise AuthError(
            "Qwen OAuth access token missing. Re-run 'qwen auth qwen-oauth'.",
            provider="qwen-oauth",
            code="qwen_access_token_missing",
        )

    base_url = os.getenv("HERMES_QWEN_BASE_URL", "").strip().rstrip("/") or DEFAULT_QWEN_BASE_URL
    return {
        "provider": "qwen-oauth",
        "base_url": base_url,
        "api_key": access_token,
        "source": "qwen-cli",
        "expires_at_ms": tokens.get("expiry_date"),
        "auth_file": str(_qwen_cli_auth_path()),
    }


def get_qwen_auth_status() -> Dict[str, Any]:
    auth_path = _qwen_cli_auth_path()
    try:
        creds = resolve_qwen_runtime_credentials(refresh_if_expiring=False)
        return {
            "logged_in": True,
            "auth_file": str(auth_path),
            "source": creds.get("source"),
            "api_key": creds.get("api_key"),
            "expires_at_ms": creds.get("expires_at_ms"),
        }
    except AuthError as exc:
        return {
            "logged_in": False,
            "auth_file": str(auth_path),
            "error": str(exc),
        }


# =============================================================================
# Google Gemini OAuth (google-gemini-cli) — PKCE flow + Cloud Code Assist.
#
# Tokens live in ~/.hermes/auth/google_oauth.json (managed by agent.google_oauth).
# The `base_url` here is the marker "cloudcode-pa://google" that run_agent.py
# uses to construct a GeminiCloudCodeClient instead of the default OpenAI SDK.
# Actual HTTP traffic goes to https://cloudcode-pa.googleapis.com/v1internal:*.
# =============================================================================

def resolve_gemini_oauth_runtime_credentials(
    *,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Resolve runtime OAuth creds for google-gemini-cli."""
    try:
        from agent.google_oauth import (
            GoogleOAuthError,
            _credentials_path,
            get_valid_access_token,
            load_credentials,
        )
    except ImportError as exc:
        raise AuthError(
            f"agent.google_oauth is not importable: {exc}",
            provider="google-gemini-cli",
            code="google_oauth_module_missing",
        ) from exc

    try:
        access_token = get_valid_access_token(force_refresh=force_refresh)
    except GoogleOAuthError as exc:
        raise AuthError(
            str(exc),
            provider="google-gemini-cli",
            code=exc.code,
        ) from exc

    creds = load_credentials()
    base_url = DEFAULT_GEMINI_CLOUDCODE_BASE_URL
    return {
        "provider": "google-gemini-cli",
        "base_url": base_url,
        "api_key": access_token,
        "source": "google-oauth",
        "expires_at_ms": (creds.expires_ms if creds else None),
        "auth_file": str(_credentials_path()),
        "email": (creds.email if creds else "") or "",
        "project_id": (creds.project_id if creds else "") or "",
    }


def get_gemini_oauth_auth_status() -> Dict[str, Any]:
    """Return a status dict for `hermes auth list` / `hermes status`."""
    try:
        from agent.google_oauth import _credentials_path, load_credentials
    except ImportError:
        return {"logged_in": False, "error": "agent.google_oauth unavailable"}
    auth_path = _credentials_path()
    creds = load_credentials()
    if creds is None or not creds.access_token:
        return {
            "logged_in": False,
            "auth_file": str(auth_path),
            "error": "not logged in",
        }
    return {
        "logged_in": True,
        "auth_file": str(auth_path),
        "source": "google-oauth",
        "api_key": creds.access_token,
        "expires_at_ms": creds.expires_ms,
        "email": creds.email,
        "project_id": creds.project_id,
    }
# Spotify auth — PKCE tokens stored in ~/.hermes/auth.json
# =============================================================================


def _spotify_scope_list(raw_scope: Optional[str] = None) -> List[str]:
    scope_text = (raw_scope or DEFAULT_SPOTIFY_SCOPE).strip()
    scopes = [part for part in scope_text.split() if part]
    seen: set[str] = set()
    ordered: List[str] = []
    for scope in scopes:
        if scope not in seen:
            seen.add(scope)
            ordered.append(scope)
    return ordered


def _spotify_scope_string(raw_scope: Optional[str] = None) -> str:
    return " ".join(_spotify_scope_list(raw_scope))


def _spotify_client_id(
    explicit: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> str:
    from hermes_cli.config import get_env_value

    candidates = (
        explicit,
        get_env_value("HERMES_SPOTIFY_CLIENT_ID"),
        get_env_value("SPOTIFY_CLIENT_ID"),
        state.get("client_id") if isinstance(state, dict) else None,
    )
    for candidate in candidates:
        cleaned = str(candidate or "").strip()
        if cleaned:
            return cleaned
    raise AuthError(
        "Spotify client_id is required. Set HERMES_SPOTIFY_CLIENT_ID or pass --client-id.",
        provider="spotify",
        code="spotify_client_id_missing",
    )


def _spotify_redirect_uri(
    explicit: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> str:
    from hermes_cli.config import get_env_value

    candidates = (
        explicit,
        get_env_value("HERMES_SPOTIFY_REDIRECT_URI"),
        get_env_value("SPOTIFY_REDIRECT_URI"),
        state.get("redirect_uri") if isinstance(state, dict) else None,
        DEFAULT_SPOTIFY_REDIRECT_URI,
    )
    for candidate in candidates:
        cleaned = str(candidate or "").strip()
        if cleaned:
            return cleaned
    return DEFAULT_SPOTIFY_REDIRECT_URI


def _spotify_api_base_url(state: Optional[Dict[str, Any]] = None) -> str:
    from hermes_cli.config import get_env_value

    candidates = (
        get_env_value("HERMES_SPOTIFY_API_BASE_URL"),
        state.get("api_base_url") if isinstance(state, dict) else None,
        DEFAULT_SPOTIFY_API_BASE_URL,
    )
    for candidate in candidates:
        cleaned = str(candidate or "").strip().rstrip("/")
        if cleaned:
            return cleaned
    return DEFAULT_SPOTIFY_API_BASE_URL


def _spotify_accounts_base_url(state: Optional[Dict[str, Any]] = None) -> str:
    from hermes_cli.config import get_env_value

    candidates = (
        get_env_value("HERMES_SPOTIFY_ACCOUNTS_BASE_URL"),
        state.get("accounts_base_url") if isinstance(state, dict) else None,
        DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL,
    )
    for candidate in candidates:
        cleaned = str(candidate or "").strip().rstrip("/")
        if cleaned:
            return cleaned
    return DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL


def _spotify_code_verifier(length: int = 64) -> str:
    raw = base64.urlsafe_b64encode(os.urandom(length)).decode("ascii")
    return raw.rstrip("=")[:128]


def _spotify_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _spotify_build_authorize_url(
    *,
    client_id: str,
    redirect_uri: str,
    scope: str,
    state: str,
    code_challenge: str,
    accounts_base_url: str,
) -> str:
    query = urlencode({
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
        "code_challenge_method": "S256",
        "code_challenge": code_challenge,
    })
    return f"{accounts_base_url}/authorize?{query}"


def _spotify_validate_redirect_uri(redirect_uri: str) -> tuple[str, int, str]:
    parsed = urlparse(redirect_uri)
    if parsed.scheme != "http":
        raise AuthError(
            "Spotify PKCE redirect_uri must use http://localhost or http://127.0.0.1.",
            provider="spotify",
            code="spotify_redirect_invalid",
        )
    host = parsed.hostname or ""
    if host not in {"127.0.0.1", "localhost"}:
        raise AuthError(
            "Spotify PKCE redirect_uri must point to localhost or 127.0.0.1.",
            provider="spotify",
            code="spotify_redirect_invalid",
        )
    if not parsed.port:
        raise AuthError(
            "Spotify PKCE redirect_uri must include an explicit localhost port.",
            provider="spotify",
            code="spotify_redirect_invalid",
        )
    return host, parsed.port, parsed.path or "/"


def _make_spotify_callback_handler(expected_path: str) -> tuple[type[BaseHTTPRequestHandler], dict[str, Any]]:
    result: dict[str, Any] = {
        "code": None,
        "state": None,
        "error": None,
        "error_description": None,
    }

    class _SpotifyCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != expected_path:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found.")
                return

            params = parse_qs(parsed.query)
            result["code"] = params.get("code", [None])[0]
            result["state"] = params.get("state", [None])[0]
            result["error"] = params.get("error", [None])[0]
            result["error_description"] = params.get("error_description", [None])[0]

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if result["error"]:
                body = "<html><body><h1>Spotify authorization failed.</h1>You can close this tab.</body></html>"
            else:
                body = "<html><body><h1>Spotify authorization received.</h1>You can close this tab.</body></html>"
            self.wfile.write(body.encode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return _SpotifyCallbackHandler, result


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
            state = _refresh_spotify_oauth_state(state)
            _store_provider_state(auth_store, "spotify", state, set_active=False)
            _save_auth_store(auth_store)

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

    try:
        raw = input("Spotify Client ID: ").strip()
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

    if open_browser and not _is_remote_session():
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
# SSH / remote session detection
# =============================================================================

def _is_remote_session() -> bool:
    """Detect if running in an SSH session where webbrowser.open() won't work."""
    return bool(os.getenv("SSH_CLIENT") or os.getenv("SSH_TTY"))


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


def _save_codex_tokens(tokens: Dict[str, str], last_refresh: str = None) -> None:
    """Save Codex OAuth tokens to Hermes auth store (~/.hermes/auth.json)."""
    if last_refresh is None:
        last_refresh = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with _auth_store_lock():
        auth_store = _load_auth_store()
        state = _load_provider_state(auth_store, "openai-codex") or {}
        state["tokens"] = tokens
        state["last_refresh"] = last_refresh
        state["auth_mode"] = "chatgpt"
        _save_provider_state(auth_store, "openai-codex", state)
        _save_auth_store(auth_store)


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
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        response = client.post(
            CODEX_OAUTH_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_OAUTH_CLIENT_ID,
            },
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
    refreshed = refresh_codex_oauth_pure(
        str(tokens.get("access_token", "") or ""),
        str(tokens.get("refresh_token", "") or ""),
        timeout_seconds=timeout_seconds,
    )
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
        payload = json.loads(auth_path.read_text())
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
    """Resolve runtime credentials from Hermes's own Codex token store."""
    data = _read_codex_tokens()
    tokens = dict(data["tokens"])
    access_token = str(tokens.get("access_token", "") or "").strip()
    refresh_timeout_seconds = float(os.getenv("HERMES_CODEX_REFRESH_TIMEOUT_SECONDS", "20"))

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

    raise TimeoutError("Timed out waiting for device authorization")


# =============================================================================
# Nous Portal — token refresh, agent key minting, model discovery
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
    the entire refresh+mint cycle so concurrent imports on sibling
    profiles can't race on the single-use shared refresh token; that
    helper must NOT be called with ``_auth_store_lock`` already held.
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

    We deliberately omit the short-lived ``agent_key`` (24h TTL, profile-
    specific) — only the long-lived OAuth tokens are cross-profile useful.
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
            try:
                os.chmod(path.parent, 0o700)
            except OSError:
                pass
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
        payload = json.loads(path.read_text())
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


def _try_import_shared_nous_state(
    *,
    timeout_seconds: float = 15.0,
    min_key_ttl_seconds: int = 5 * 60,
) -> Optional[Dict[str, Any]]:
    """Attempt to rehydrate Nous OAuth state from the shared store.

    Reads the shared file (if present), runs a forced refresh+mint using
    the stored refresh_token to produce a fresh access_token + agent_key
    scoped to this profile, and returns the full auth_state dict ready
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
            # for this profile; force_mint=True gets us a fresh agent_key.
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

            refreshed = refresh_nous_oauth_from_state(
                state,
                min_key_ttl_seconds=min_key_ttl_seconds,
                timeout_seconds=timeout_seconds,
                force_refresh=True,
                force_mint=True,
            )
            _write_shared_nous_state(refreshed)
    except AuthError as exc:
        _oauth_trace(
            "nous_shared_import_failed",
            error_type=type(exc).__name__,
            error_code=getattr(exc, "code", None),
        )
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
    relogin = code in {"invalid_grant", "invalid_token"}

    # Detect the OAuth 2.1 "refresh token reuse" signal from the Nous portal
    # server and surface an actionable message.  This fires when an external
    # process (health-check script, monitoring tool, custom self-heal hook)
    # called POST /api/oauth/token with Hermes's refresh_token without
    # persisting the rotated token back to auth.json — the server then
    # retires the original RT, Hermes's next refresh uses it, and the whole
    # session chain gets revoked as a token-theft signal (#15099).
    lowered = description.lower()
    if "reuse" in lowered or "reuse detected" in lowered:
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

    raise AuthError(description, provider="nous", code=code, relogin_required=relogin)


def _mint_agent_key(
    *,
    client: httpx.Client,
    portal_base_url: str,
    access_token: str,
    min_ttl_seconds: int,
) -> Dict[str, Any]:
    """Mint (or reuse) a short-lived inference API key."""
    response = client.post(
        f"{portal_base_url}/api/oauth/agent-key",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"min_ttl_seconds": max(60, int(min_ttl_seconds))},
    )

    if response.status_code == 200:
        payload = response.json()
        if "api_key" not in payload:
            raise AuthError("Mint response missing api_key",
                            provider="nous", code="server_error")
        return payload

    try:
        error_payload = response.json()
    except Exception as exc:
        raise AuthError("Agent key mint request failed",
                        provider="nous", code="server_error") from exc

    code = str(error_payload.get("error", "server_error"))
    description = str(error_payload.get("error_description") or "Agent key mint request failed")
    relogin = code in {"invalid_token", "invalid_grant"}
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
    return not _is_expiring(state.get("agent_key_expires_at"), min_ttl_seconds)


def resolve_nous_access_token(
    *,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    refresh_skew_seconds: int = ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> str:
    """Resolve a refresh-aware Nous Portal access token for managed tool gateways."""
    with _auth_store_lock():
        auth_store = _load_auth_store()
        state = _load_provider_state(auth_store, "nous")

        if not state:
            raise AuthError(
                "Hermes is not logged into Nous Portal.",
                provider="nous",
                relogin_required=True,
            )

        portal_base_url = (
            _optional_base_url(state.get("portal_base_url"))
            or os.getenv("HERMES_PORTAL_BASE_URL")
            or os.getenv("NOUS_PORTAL_BASE_URL")
            or DEFAULT_NOUS_PORTAL_URL
        ).rstrip("/")
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
                    _save_provider_state(auth_store, "nous", state)
                    _save_auth_store(auth_store)
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
                refreshed = _refresh_access_token(
                    client=client,
                    portal_base_url=portal_base_url,
                    client_id=client_id,
                    refresh_token=refresh_token,
                )

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
            _save_provider_state(auth_store, "nous", state)
            _save_auth_store(auth_store)
            _write_shared_nous_state(state)
            return state["access_token"]


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
    min_key_ttl_seconds: int = DEFAULT_AGENT_KEY_MIN_TTL_SECONDS,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    force_refresh: bool = False,
    force_mint: bool = False,
) -> Dict[str, Any]:
    """Refresh Nous OAuth state without mutating auth.json."""
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
        if force_refresh or _is_expiring(state.get("expires_at"), ACCESS_TOKEN_REFRESH_SKEW_SECONDS):
            refreshed = _refresh_access_token(
                client=client,
                portal_base_url=state["portal_base_url"],
                client_id=state["client_id"],
                refresh_token=state["refresh_token"],
            )
            now = datetime.now(timezone.utc)
            access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
            state["access_token"] = refreshed["access_token"]
            state["refresh_token"] = refreshed.get("refresh_token") or state["refresh_token"]
            state["token_type"] = refreshed.get("token_type") or state.get("token_type") or "Bearer"
            state["scope"] = refreshed.get("scope") or state.get("scope")
            refreshed_url = _optional_base_url(refreshed.get("inference_base_url"))
            if refreshed_url:
                state["inference_base_url"] = refreshed_url
            state["obtained_at"] = now.isoformat()
            state["expires_in"] = access_ttl
            state["expires_at"] = datetime.fromtimestamp(
                now.timestamp() + access_ttl, tz=timezone.utc
            ).isoformat()

        if force_mint or not _agent_key_is_usable(state, max(60, int(min_key_ttl_seconds))):
            mint_payload = _mint_agent_key(
                client=client,
                portal_base_url=state["portal_base_url"],
                access_token=state["access_token"],
                min_ttl_seconds=min_key_ttl_seconds,
            )
            now = datetime.now(timezone.utc)
            state["agent_key"] = mint_payload.get("api_key")
            state["agent_key_id"] = mint_payload.get("key_id")
            state["agent_key_expires_at"] = mint_payload.get("expires_at")
            state["agent_key_expires_in"] = mint_payload.get("expires_in")
            state["agent_key_reused"] = bool(mint_payload.get("reused", False))
            state["agent_key_obtained_at"] = now.isoformat()
            minted_url = _optional_base_url(mint_payload.get("inference_base_url"))
            if minted_url:
                state["inference_base_url"] = minted_url

    return state


def refresh_nous_oauth_from_state(
    state: Dict[str, Any],
    *,
    min_key_ttl_seconds: int = DEFAULT_AGENT_KEY_MIN_TTL_SECONDS,
    timeout_seconds: float = 15.0,
    force_refresh: bool = False,
    force_mint: bool = False,
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
        min_key_ttl_seconds=min_key_ttl_seconds,
        timeout_seconds=timeout_seconds,
        insecure=tls.get("insecure"),
        ca_bundle=tls.get("ca_bundle"),
        force_refresh=force_refresh,
        force_mint=force_mint,
    )


NOUS_DEVICE_CODE_SOURCE = "device_code"


def persist_nous_credentials(
    creds: Dict[str, Any],
    *,
    label: Optional[str] = None,
):
    """Persist minted Nous OAuth credentials as the singleton provider state
    and ensure the credential pool is in sync.

    Nous credentials are read at runtime from two independent locations:

    - ``providers.nous``: singleton state read by
      ``resolve_nous_runtime_credentials()`` during 401 recovery and by
      ``_seed_from_singletons()`` during pool load.
    - ``credential_pool.nous``: used by the runtime ``pool.select()`` path.

    Historically ``hermes auth add nous`` wrote a ``manual:device_code`` pool
    entry only, skipping ``providers.nous``.  When the 24h agent_key TTL
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


def resolve_nous_runtime_credentials(
    *,
    min_key_ttl_seconds: int = DEFAULT_AGENT_KEY_MIN_TTL_SECONDS,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    force_mint: bool = False,
) -> Dict[str, Any]:
    """
    Resolve Nous inference credentials for runtime use.

    Ensures access_token is valid (refreshes if needed) and a short-lived
    inference key is present with minimum TTL (mints/reuses as needed).
    Concurrent processes coordinate through the auth store file lock.

    Returns dict with: provider, base_url, api_key, key_id, expires_at,
    expires_in, source ("cache" or "portal").
    """
    min_key_ttl_seconds = max(60, int(min_key_ttl_seconds))
    sequence_id = uuid.uuid4().hex[:12]

    with _auth_store_lock():
        auth_store = _load_auth_store()
        state = _load_provider_state(auth_store, "nous")

        if not state:
            raise AuthError("Hermes is not logged into Nous Portal.",
                            provider="nous", relogin_required=True)

        portal_base_url = (
            _optional_base_url(state.get("portal_base_url"))
            or os.getenv("HERMES_PORTAL_BASE_URL")
            or os.getenv("NOUS_PORTAL_BASE_URL")
            or DEFAULT_NOUS_PORTAL_URL
        ).rstrip("/")
        inference_base_url = (
            _optional_base_url(state.get("inference_base_url"))
            or os.getenv("NOUS_INFERENCE_BASE_URL")
            or DEFAULT_NOUS_INFERENCE_URL
        ).rstrip("/")
        client_id = str(state.get("client_id") or DEFAULT_NOUS_CLIENT_ID)

        def _persist_state(reason: str) -> None:
            try:
                _save_provider_state(auth_store, "nous", state)
                _save_auth_store(auth_store)
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
            force_mint=bool(force_mint),
            min_key_ttl_seconds=min_key_ttl_seconds,
            refresh_token_fp=_token_fingerprint(state.get("refresh_token")),
        )

        with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
            access_token = state.get("access_token")
            refresh_token = state.get("refresh_token")

            if not isinstance(access_token, str) or not access_token:
                raise AuthError("No access token found for Nous Portal login.",
                                provider="nous", relogin_required=True)

            # Step 1: refresh access token if expiring
            if _is_expiring(state.get("expires_at"), ACCESS_TOKEN_REFRESH_SKEW_SECONDS):
                with _nous_shared_store_lock(timeout_seconds=max(timeout_seconds + 5.0, AUTH_LOCK_TIMEOUT_SECONDS)):
                    if _merge_shared_nous_oauth_state(state):
                        access_token = state.get("access_token")
                        refresh_token = state.get("refresh_token")
                        _persist_state("post_shared_merge_access_expiring")

                    if _is_expiring(state.get("expires_at"), ACCESS_TOKEN_REFRESH_SKEW_SECONDS):
                        if not isinstance(refresh_token, str) or not refresh_token:
                            raise AuthError("Session expired and no refresh token is available.",
                                            provider="nous", relogin_required=True)

                        _oauth_trace(
                            "refresh_start",
                            sequence_id=sequence_id,
                            reason="access_expiring",
                            refresh_token_fp=_token_fingerprint(refresh_token),
                        )
                        refreshed = _refresh_access_token(
                            client=client, portal_base_url=portal_base_url,
                            client_id=client_id, refresh_token=refresh_token,
                        )
                        now = datetime.now(timezone.utc)
                        access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
                        previous_refresh_token = refresh_token
                        state["access_token"] = refreshed["access_token"]
                        state["refresh_token"] = refreshed.get("refresh_token") or refresh_token
                        state["token_type"] = refreshed.get("token_type") or state.get("token_type") or "Bearer"
                        state["scope"] = refreshed.get("scope") or state.get("scope")
                        refreshed_url = _optional_base_url(refreshed.get("inference_base_url"))
                        if refreshed_url:
                            inference_base_url = refreshed_url
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
                            reason="access_expiring",
                            previous_refresh_token_fp=_token_fingerprint(previous_refresh_token),
                            new_refresh_token_fp=_token_fingerprint(refresh_token),
                        )
                        # Persist immediately so downstream mint failures cannot drop rotated refresh tokens.
                        _persist_state("post_refresh_access_expiring")

            # Step 2: mint agent key if missing/expiring
            used_cached_key = False
            mint_payload: Optional[Dict[str, Any]] = None

            if not force_mint and _agent_key_is_usable(state, min_key_ttl_seconds):
                used_cached_key = True
                _oauth_trace("agent_key_reuse", sequence_id=sequence_id)
            else:
                try:
                    _oauth_trace(
                        "mint_start",
                        sequence_id=sequence_id,
                        access_token_fp=_token_fingerprint(access_token),
                    )
                    mint_payload = _mint_agent_key(
                        client=client, portal_base_url=portal_base_url,
                        access_token=access_token, min_ttl_seconds=min_key_ttl_seconds,
                    )
                except AuthError as exc:
                    _oauth_trace(
                        "mint_error",
                        sequence_id=sequence_id,
                        code=exc.code,
                    )
                    # Retry path: access token may be stale server-side despite local checks
                    latest_refresh_token = state.get("refresh_token")
                    if (
                        exc.code in {"invalid_token", "invalid_grant"}
                        and isinstance(latest_refresh_token, str)
                        and latest_refresh_token
                    ):
                        with _nous_shared_store_lock(timeout_seconds=max(timeout_seconds + 5.0, AUTH_LOCK_TIMEOUT_SECONDS)):
                            if _merge_shared_nous_oauth_state(state):
                                access_token = state.get("access_token")
                                latest_refresh_token = state.get("refresh_token")
                                _persist_state("post_shared_merge_mint_retry")
                            else:
                                _oauth_trace(
                                    "refresh_start",
                                    sequence_id=sequence_id,
                                    reason="mint_retry_after_invalid_token",
                                    refresh_token_fp=_token_fingerprint(latest_refresh_token),
                                )
                                refreshed = _refresh_access_token(
                                    client=client, portal_base_url=portal_base_url,
                                    client_id=client_id, refresh_token=latest_refresh_token,
                                )
                                now = datetime.now(timezone.utc)
                                access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
                                state["access_token"] = refreshed["access_token"]
                                state["refresh_token"] = refreshed.get("refresh_token") or latest_refresh_token
                                state["token_type"] = refreshed.get("token_type") or state.get("token_type") or "Bearer"
                                state["scope"] = refreshed.get("scope") or state.get("scope")
                                refreshed_url = _optional_base_url(refreshed.get("inference_base_url"))
                                if refreshed_url:
                                    inference_base_url = refreshed_url
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
                                    reason="mint_retry_after_invalid_token",
                                    previous_refresh_token_fp=_token_fingerprint(latest_refresh_token),
                                    new_refresh_token_fp=_token_fingerprint(refresh_token),
                                )
                                # Persist retry refresh immediately for crash safety and cross-process visibility.
                                _persist_state("post_refresh_mint_retry")

                        mint_payload = _mint_agent_key(
                            client=client, portal_base_url=portal_base_url,
                            access_token=access_token, min_ttl_seconds=min_key_ttl_seconds,
                        )
                    else:
                        raise

            if mint_payload is not None:
                now = datetime.now(timezone.utc)
                state["agent_key"] = mint_payload.get("api_key")
                state["agent_key_id"] = mint_payload.get("key_id")
                state["agent_key_expires_at"] = mint_payload.get("expires_at")
                state["agent_key_expires_in"] = mint_payload.get("expires_in")
                state["agent_key_reused"] = bool(mint_payload.get("reused", False))
                state["agent_key_obtained_at"] = now.isoformat()
                minted_url = _optional_base_url(mint_payload.get("inference_base_url"))
                if minted_url:
                    inference_base_url = minted_url
                _oauth_trace(
                    "mint_success",
                    sequence_id=sequence_id,
                    reused=bool(mint_payload.get("reused", False)),
                )

            # Persist routing and TLS metadata for non-interactive refresh/mint
            state["portal_base_url"] = portal_base_url
            state["inference_base_url"] = inference_base_url
            state["client_id"] = client_id
            state["tls"] = {
                "insecure": verify is False,
                "ca_bundle": verify if isinstance(verify, str) else None,
            }

        _persist_state("resolve_nous_runtime_credentials_final")

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
        "source": "cache" if used_cached_key else "portal",
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
    }


def _snapshot_nous_pool_status() -> Dict[str, Any]:
    """Best-effort status from the credential pool.

    This is a fallback only. The auth-store provider state is the runtime source
    of truth because it is what ``resolve_nous_runtime_credentials()`` refreshes
    and mints against.
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
        access_token = (
            getattr(entry, "access_token", None)
            or getattr(entry, "runtime_api_key", "")
        )
        if not access_token:
            return _empty_nous_auth_status()

        return {
            "logged_in": True,
            "portal_base_url": getattr(entry, "portal_base_url", None)
            or getattr(entry, "base_url", None),
            "inference_base_url": getattr(entry, "inference_base_url", None)
            or getattr(entry, "base_url", None),
            "access_token": access_token,
            "access_expires_at": getattr(entry, "expires_at", None),
            "agent_key_expires_at": getattr(entry, "agent_key_expires_at", None),
            "has_refresh_token": bool(getattr(entry, "refresh_token", None)),
            "source": f"pool:{getattr(entry, 'label', 'unknown')}",
        }
    except Exception:
        return _empty_nous_auth_status()


def get_nous_auth_status() -> Dict[str, Any]:
    """Status snapshot for Nous auth.

    Prefer the auth-store provider state, because that is the live source of
    truth for refresh + mint operations. When provider state exists, validate it
    by resolving runtime credentials so revoked refresh sessions do not show up
    as a healthy login. If provider state is absent, fall back to the credential
    pool for the just-logged-in / not-yet-promoted case.
    """
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
            "source": "auth_store",
        }
        try:
            creds = resolve_nous_runtime_credentials(min_key_ttl_seconds=60)
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


def get_api_key_provider_status(provider_id: str) -> Dict[str, Any]:
    """Status snapshot for API-key providers (z.ai, Kimi, MiniMax)."""
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "api_key":
        return {"configured": False}

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

    return {
        "configured": bool(api_key),
        "provider": provider_id,
        "name": pconfig.name,
        "key_source": key_source,
        "base_url": base_url,
        "logged_in": bool(api_key),  # compat with OAuth status shape
    }


def get_external_process_provider_status(provider_id: str) -> Dict[str, Any]:
    """Status snapshot for providers that run a local subprocess."""
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
    return {
        "configured": bool(resolved_command or base_url.startswith("acp+tcp://")),
        "provider": provider_id,
        "name": pconfig.name,
        "command": command,
        "args": args,
        "resolved_command": resolved_command,
        "base_url": base_url,
        "logged_in": bool(resolved_command or base_url.startswith("acp+tcp://")),
    }


def get_auth_status(provider_id: Optional[str] = None) -> Dict[str, Any]:
    """Generic auth status dispatcher."""
    target = provider_id or get_active_provider()
    if target == "spotify":
        return get_spotify_auth_status()
    if target == "nous":
        return get_nous_auth_status()
    if target == "openai-codex":
        return get_codex_auth_status()
    if target == "qwen-oauth":
        return get_qwen_auth_status()
    if target == "google-gemini-cli":
        return get_gemini_oauth_auth_status()
    if target == "minimax-oauth":
        return get_minimax_oauth_auth_status()
    if target == "copilot-acp":
        return get_external_process_provider_status(target)
    # API-key providers
    pconfig = PROVIDER_REGISTRY.get(target)
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
    elif env_url:
        base_url = env_url.rstrip("/")
    else:
        base_url = pconfig.inference_base_url

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

    command = (
        os.getenv("HERMES_COPILOT_ACP_COMMAND", "").strip()
        or os.getenv("COPILOT_CLI_PATH", "").strip()
        or "copilot"
    )
    raw_args = os.getenv("HERMES_COPILOT_ACP_ARGS", "").strip()
    args = shlex.split(raw_args) if raw_args else ["--acp", "--stdio"]
    resolved_command = shutil.which(command) if command else None
    if not resolved_command and not base_url.startswith("acp+tcp://"):
        raise AuthError(
            f"Could not find the Copilot CLI command '{command}'. "
            "Install GitHub Copilot CLI or set HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH.",
            provider=provider_id,
            code="missing_copilot_cli",
        )

    return {
        "provider": provider_id,
        "api_key": "copilot-acp",
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

    # Clear stale api_key/api_mode left over from a previous custom provider.
    # When the user switches from e.g. a MiniMax custom endpoint
    # (api_mode=anthropic_messages, api_key=mxp-...) to a built-in provider
    # (e.g. OpenRouter), the stale api_key/api_mode would override the new
    # provider's credentials and transport choice.  Built-in providers that
    # need a specific api_mode (copilot, xai) set it at request-resolution
    # time via `_copilot_runtime_api_mode` / `_detect_api_mode_for_url`, so
    # removing the persisted value here is safe.
    model_cfg.pop("api_key", None)
    model_cfg.pop("api_mode", None)

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
    if provider in {"nous", "openai-codex"}:
        return provider
    return None


def _reset_config_provider() -> Path:
    """Reset config.yaml provider back to auto after logout."""
    config_path = get_config_path()
    if not config_path.exists():
        return config_path

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


def _prompt_model_selection(
    model_ids: List[str],
    current_model: str = "",
    pricing: Optional[Dict[str, Dict[str, str]]] = None,
    unavailable_models: Optional[List[str]] = None,
    portal_url: str = "",
) -> Optional[str]:
    """Interactive model selection. Puts current_model first with a marker. Returns chosen model ID or None.

    If *pricing* is provided (``{model_id: {prompt, completion}}``), a compact
    price indicator is shown next to each model in aligned columns.

    If *unavailable_models* is provided, those models are shown grayed out
    and unselectable, with an upgrade link to *portal_url*.
    """
    from hermes_cli.models import _format_price_per_mtok

    _unavailable = unavailable_models or []

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
    name_col = max((len(m) for m in all_models), default=0) + 2 if has_pricing else 0

    # Pre-compute formatted prices and dynamic column widths
    _price_cache: dict[str, tuple[str, str, str]] = {}
    price_col = 3  # minimum width
    cache_col = 0  # only set if any model has cache pricing
    has_cache = False
    if has_pricing:
        for mid in all_models:
            p = pricing.get(mid)  # type: ignore[union-attr]
            if p:
                inp = _format_price_per_mtok(p.get("prompt", ""))
                out = _format_price_per_mtok(p.get("completion", ""))
                cache_read = p.get("input_cache_read", "")
                cache = _format_price_per_mtok(cache_read) if cache_read else ""
                if cache:
                    has_cache = True
            else:
                inp, out, cache = "", "", ""
            _price_cache[mid] = (inp, out, cache)
            price_col = max(price_col, len(inp), len(out))
            cache_col = max(cache_col, len(cache))
        if has_cache:
            cache_col = max(cache_col, 5)  # minimum: "Cache" header

    def _label(mid):
        if has_pricing:
            inp, out, cache = _price_cache.get(mid, ("", "", ""))
            price_part = f" {inp:>{price_col}}  {out:>{price_col}}"
            if has_cache:
                price_part += f"  {cache:>{cache_col}}"
            base = f"{mid:<{name_col}}{price_part}"
        else:
            base = mid
        if mid == current_model:
            base += "  ← currently in use"
        return base

    # Default cursor on the current model (index 0 if it was reordered to top)
    default_idx = 0

    # Build a pricing header hint for the menu title
    menu_title = "Select default model:"
    if has_pricing:
        # Align the header with the model column.
        # Each choice is "  {label}" (2 spaces) and simple_term_menu prepends
        # a 3-char cursor region ("-> " or "   "), so content starts at col 5.
        pad = " " * 5
        header = f"\n{pad}{'':>{name_col}} {'In':>{price_col}}  {'Out':>{price_col}}"
        if has_cache:
            header += f"  {'Cache':>{cache_col}}"
        menu_title += header + "  /Mtok"

    # ANSI escape for dim text
    _DIM = "\033[2m"
    _RESET = "\033[0m"

    # Try arrow-key menu first, fall back to number input
    try:
        from simple_term_menu import TerminalMenu

        choices = [f"  {_label(mid)}" for mid in ordered]
        choices.append("  Enter custom model name")
        choices.append("  Skip (keep current)")

        # Print the unavailable block BEFORE the menu via regular print().
        # simple_term_menu pads title lines to terminal width (causes wrapping),
        # so we keep the title minimal and use stdout for the static block.
        # clear_screen=False means our printed output stays visible above.
        _upgrade_url = (portal_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
        if _unavailable:
            print(menu_title)
            print()
            for mid in _unavailable:
                print(f"{_DIM}     {_label(mid)}{_RESET}")
            print()
            print(f"{_DIM}  ── Upgrade at {_upgrade_url} for paid models ──{_RESET}")
            print()
            effective_title = "Available free models:"
        else:
            effective_title = menu_title

        menu = TerminalMenu(
            choices,
            cursor_index=default_idx,
            menu_cursor="-> ",
            menu_cursor_style=("fg_green", "bold"),
            menu_highlight_style=("fg_green",),
            cycle_cursor=True,
            clear_screen=False,
            title=effective_title,
        )
        idx = menu.show()
        from hermes_cli.curses_ui import flush_stdin
        flush_stdin()
        if idx is None:
            return None
        print()
        if idx < len(ordered):
            return ordered[idx]
        elif idx == len(ordered):
            custom = input("Enter model name: ").strip()
            return custom if custom else None
        return None
    except (ImportError, NotImplementedError, OSError, subprocess.SubprocessError):
        pass

    # Fallback: numbered list
    print(menu_title)
    num_width = len(str(len(ordered) + 2))
    for i, mid in enumerate(ordered, 1):
        print(f"  {i:>{num_width}}. {_label(mid)}")
    n = len(ordered)
    print(f"  {n + 1:>{num_width}}. Enter custom model name")
    print(f"  {n + 2:>{num_width}}. Skip (keep current)")

    if _unavailable:
        _upgrade_url = (portal_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
        print()
        print(f"  {_DIM}── Unavailable models (requires paid tier — upgrade at {_upgrade_url}) ──{_RESET}")
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
                return ordered[idx - 1]
            elif idx == n + 1:
                custom = input("Enter model name: ").strip()
                return custom if custom else None
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


def _codex_device_code_login() -> Dict[str, Any]:
    """Run the OpenAI device code login flow and return credentials dict."""
    import time as _time

    issuer = "https://auth.openai.com"
    client_id = CODEX_OAUTH_CLIENT_ID

    # Step 1: Request device code
    try:
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
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

    if resp.status_code != 200:
        raise AuthError(
            f"Device code request returned status {resp.status_code}.",
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
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
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
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
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

    if token_resp.status_code != 200:
        raise AuthError(
            f"Token exchange returned status {token_resp.status_code}.",
            provider="openai-codex", code="token_exchange_error",
        )

    tokens = token_resp.json()
    access_token = tokens.get("access_token", "")
    refresh_token = tokens.get("refresh_token", "")

    if not access_token:
        raise AuthError(
            "Token exchange did not return an access_token.",
            provider="openai-codex", code="token_exchange_no_access_token",
        )

    # Return tokens for the caller to persist (no longer writes to ~/.codex/)
    base_url = (
        os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/")
        or DEFAULT_CODEX_BASE_URL
    )

    return {
        "tokens": {
            "access_token": access_token,
            "refresh_token": refresh_token,
        },
        "base_url": base_url,
        "last_refresh": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "auth_mode": "chatgpt",
        "source": "device-code",
    }


# ==================== MiniMax Portal OAuth ====================

def _minimax_pkce_pair() -> tuple:
    """Generate (code_verifier, code_challenge_S256, state) for MiniMax OAuth."""
    import secrets
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode().rstrip("=")
    state = secrets.token_urlsafe(16)
    return verifier, challenge, state


def _minimax_request_user_code(
    client: httpx.Client, *, portal_base_url: str, client_id: str,
    code_challenge: str, state: str,
) -> Dict[str, Any]:
    response = client.post(
        f"{portal_base_url}/oauth/code",
        data={
            "response_type": "code",
            "client_id": client_id,
            "scope": MINIMAX_OAUTH_SCOPE,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
        },
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "x-request-id": str(uuid.uuid4()),
        },
    )
    if response.status_code != 200:
        raise AuthError(
            f"MiniMax OAuth authorization failed: {response.text or response.reason_phrase}",
            provider="minimax-oauth", code="authorization_failed",
        )
    payload = response.json()
    for field in ("user_code", "verification_uri", "expired_in"):
        if field not in payload:
            raise AuthError(
                f"MiniMax OAuth response missing field: {field}",
                provider="minimax-oauth", code="authorization_incomplete",
            )
    if payload.get("state") != state:
        raise AuthError(
            "MiniMax OAuth state mismatch (possible CSRF).",
            provider="minimax-oauth", code="state_mismatch",
        )
    return payload


def _minimax_expired_in_looks_like_unix_ms(expired_in: int, *, now_ms: int) -> bool:
    """True if ``expired_in`` is plausibly a unix-ms absolute time (vs TTL seconds)."""
    return int(expired_in) > (now_ms // 2)


def _minimax_resolve_token_expiry_unix(expired_in: int, *, now: datetime) -> float:
    """Return access-token expiry as unix seconds (MiniMax uses ms epoch or TTL seconds)."""
    raw = int(expired_in)
    now_ms = int(now.timestamp() * 1000)
    if _minimax_expired_in_looks_like_unix_ms(raw, now_ms=now_ms):
        return raw / 1000.0
    return now.timestamp() + max(1, raw)


def _minimax_poll_token(
    client: httpx.Client, *, portal_base_url: str, client_id: str,
    user_code: str, code_verifier: str, expired_in: int, interval_ms: Optional[int],
) -> Dict[str, Any]:
    # OpenClaw treats expired_in as a unix-ms timestamp (Date.now() < expireTimeMs).
    # Defensive parsing: if it's small enough to be a duration, treat as seconds.
    import time as _time
    now_ms = int(_time.time() * 1000)
    raw = int(expired_in)
    if _minimax_expired_in_looks_like_unix_ms(raw, now_ms=now_ms):
        deadline = raw / 1000.0
    else:
        deadline = _time.time() + max(1, raw)
    interval = max(2.0, (interval_ms or 2000) / 1000.0)

    while _time.time() < deadline:
        response = client.post(
            f"{portal_base_url}/oauth/token",
            data={
                "grant_type": MINIMAX_OAUTH_GRANT_TYPE,
                "client_id": client_id,
                "user_code": user_code,
                "code_verifier": code_verifier,
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
        )
        try:
            payload = response.json() if response.text else {}
        except Exception:
            payload = {}

        if response.status_code != 200:
            msg = (payload.get("base_resp", {}) or {}).get("status_msg") or response.text
            raise AuthError(
                f"MiniMax OAuth error: {msg or 'unknown'}",
                provider="minimax-oauth", code="token_exchange_failed",
            )

        status = payload.get("status")
        if status == "error":
            raise AuthError(
                "MiniMax OAuth reported an error. Please try again later.",
                provider="minimax-oauth", code="authorization_denied",
            )
        if status == "success":
            if not all(payload.get(k) for k in ("access_token", "refresh_token", "expired_in")):
                raise AuthError(
                    "MiniMax OAuth success payload missing required token fields.",
                    provider="minimax-oauth", code="token_incomplete",
                )
            return payload
        # "pending" or any other status -> keep polling
        _time.sleep(interval)

    raise AuthError(
        "MiniMax OAuth timed out before authorization completed.",
        provider="minimax-oauth", code="timeout",
    )


def _minimax_save_auth_state(auth_state: Dict[str, Any]) -> None:
    """Persist MiniMax OAuth state to Hermes auth store (~/.hermes/auth.json)."""
    with _auth_store_lock():
        auth_store = _load_auth_store()
        _save_provider_state(auth_store, "minimax-oauth", auth_state)
        _save_auth_store(auth_store)


def _minimax_oauth_login(
    *, region: str = "global", open_browser: bool = True,
    timeout_seconds: float = 15.0,
) -> Dict[str, Any]:
    """Run MiniMax OAuth flow, persist tokens, return auth state dict."""
    pconfig = PROVIDER_REGISTRY["minimax-oauth"]
    if region == "cn":
        portal_base_url = pconfig.extra["cn_portal_base_url"]
        inference_base_url = pconfig.extra["cn_inference_base_url"]
    else:
        portal_base_url = pconfig.portal_base_url
        inference_base_url = pconfig.inference_base_url

    verifier, challenge, state = _minimax_pkce_pair()

    if _is_remote_session():
        open_browser = False

    print(f"Starting Hermes login via MiniMax ({region}) OAuth...")
    print(f"Portal: {portal_base_url}")

    with httpx.Client(timeout=httpx.Timeout(timeout_seconds),
                      headers={"Accept": "application/json"},
                      follow_redirects=True) as client:
        code_data = _minimax_request_user_code(
            client, portal_base_url=portal_base_url,
            client_id=pconfig.client_id,
            code_challenge=challenge, state=state,
        )
        verification_url = str(code_data["verification_uri"])
        user_code = str(code_data["user_code"])

        print()
        print("To continue:")
        print(f"  1. Open: {verification_url}")
        print(f"  2. If prompted, enter code: {user_code}")
        if open_browser:
            if webbrowser.open(verification_url):
                print("  (Opened browser for verification)")
            else:
                print("  Could not open browser automatically -- use the URL above.")

        interval_raw = code_data.get("interval")
        interval_ms = int(interval_raw) if interval_raw is not None else None
        print("Waiting for approval...")

        token_data = _minimax_poll_token(
            client, portal_base_url=portal_base_url,
            client_id=pconfig.client_id,
            user_code=user_code, code_verifier=verifier,
            expired_in=int(code_data["expired_in"]),
            interval_ms=interval_ms,
        )

    now = datetime.now(timezone.utc)
    expires_at_unix = _minimax_resolve_token_expiry_unix(
        int(token_data["expired_in"]), now=now,
    )
    expires_in_s = max(0, int(expires_at_unix - now.timestamp()))

    auth_state = {
        "provider": "minimax-oauth",
        "region": region,
        "portal_base_url": portal_base_url,
        "inference_base_url": inference_base_url,
        "client_id": pconfig.client_id,
        "scope": MINIMAX_OAUTH_SCOPE,
        "token_type": token_data.get("token_type", "Bearer"),
        "access_token": token_data["access_token"],
        "refresh_token": token_data["refresh_token"],
        "resource_url": token_data.get("resource_url"),
        "obtained_at": now.isoformat(),
        "expires_at": datetime.fromtimestamp(expires_at_unix, tz=timezone.utc).isoformat(),
        "expires_in": expires_in_s,
    }

    _minimax_save_auth_state(auth_state)
    print("\u2713 MiniMax OAuth login successful.")
    if msg := token_data.get("notification_message"):
        print(f"Note from MiniMax: {msg}")
    return auth_state


def _refresh_minimax_oauth_state(
    state: Dict[str, Any], *, timeout_seconds: float = 15.0,
    force: bool = False,
) -> Dict[str, Any]:
    """Refresh MiniMax OAuth access token if close to expiry (or forced)."""
    if not state.get("refresh_token"):
        raise AuthError(
            "MiniMax OAuth state has no refresh_token; please re-login.",
            provider="minimax-oauth", code="no_refresh_token", relogin_required=True,
        )
    try:
        expires_at = datetime.fromisoformat(state.get("expires_at", "")).timestamp()
    except Exception:
        expires_at = 0.0
    now = time.time()
    if not force and (expires_at - now) > MINIMAX_OAUTH_REFRESH_SKEW_SECONDS:
        return state

    portal_base_url = state["portal_base_url"]
    with httpx.Client(timeout=httpx.Timeout(timeout_seconds),
                      follow_redirects=True) as client:
        response = client.post(
            f"{portal_base_url}/oauth/token",
            data={
                "grant_type": "refresh_token",
                "client_id": state["client_id"],
                "refresh_token": state["refresh_token"],
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
        )
    if response.status_code != 200:
        body = response.text.lower()
        relogin = any(m in body for m in
                      ("invalid_grant", "refresh_token_reused", "invalid_refresh_token"))
        raise AuthError(
            f"MiniMax OAuth refresh failed: {response.text or response.reason_phrase}",
            provider="minimax-oauth", code="refresh_failed",
            relogin_required=relogin,
        )
    payload = response.json()
    if payload.get("status") != "success":
        raise AuthError(
            "MiniMax OAuth refresh did not return success.",
            provider="minimax-oauth", code="refresh_failed",
            relogin_required=True,
        )
    now_dt = datetime.now(timezone.utc)
    expires_at_unix = _minimax_resolve_token_expiry_unix(
        int(payload["expired_in"]), now=now_dt,
    )
    expires_in_s = max(0, int(expires_at_unix - now_dt.timestamp()))
    new_state = dict(state)
    new_state.update({
        "access_token": payload["access_token"],
        "refresh_token": payload.get("refresh_token", state["refresh_token"]),
        "obtained_at": now_dt.isoformat(),
        "expires_at": datetime.fromtimestamp(expires_at_unix, tz=timezone.utc).isoformat(),
        "expires_in": expires_in_s,
    })
    _minimax_save_auth_state(new_state)
    return new_state


def resolve_minimax_oauth_runtime_credentials(
    *, min_token_ttl_seconds: int = MINIMAX_OAUTH_REFRESH_SKEW_SECONDS,
) -> Dict[str, Any]:
    """Return {provider, api_key, base_url, source} for minimax-oauth."""
    state = get_provider_auth_state("minimax-oauth")
    if not state or not state.get("access_token"):
        raise AuthError(
            "Not logged into MiniMax OAuth. Run `hermes model` and select "
            "MiniMax (OAuth).",
            provider="minimax-oauth", code="not_logged_in", relogin_required=True,
        )
    state = _refresh_minimax_oauth_state(state)
    return {
        "provider": "minimax-oauth",
        "api_key": state["access_token"],
        "base_url": state["inference_base_url"].rstrip("/"),
        "source": "oauth",
    }


def get_minimax_oauth_auth_status() -> Dict[str, Any]:
    """Return auth status dict for MiniMax OAuth provider."""
    state = get_provider_auth_state("minimax-oauth")
    if not state or not state.get("access_token"):
        return {"logged_in": False, "provider": "minimax-oauth"}
    try:
        expires_at = datetime.fromisoformat(state.get("expires_at", "")).timestamp()
        token_valid = (expires_at - time.time()) > 0
    except Exception:
        token_valid = bool(state.get("access_token"))
    return {
        "logged_in": token_valid,
        "provider": "minimax-oauth",
        "region": state.get("region", "global"),
        "expires_at": state.get("expires_at"),
    }


def _login_minimax_oauth(args, pconfig: ProviderConfig) -> None:
    """CLI entry for MiniMax OAuth login."""
    region = getattr(args, "region", None) or "global"
    open_browser = not getattr(args, "no_browser", False)
    timeout = getattr(args, "timeout", None) or 15.0
    try:
        _minimax_oauth_login(
            region=region, open_browser=open_browser, timeout_seconds=timeout,
        )
    except AuthError as exc:
        print(format_auth_error(exc))
        raise SystemExit(1)


def _nous_device_code_login(
    *,
    portal_base_url: Optional[str] = None,
    inference_base_url: Optional[str] = None,
    client_id: Optional[str] = None,
    scope: Optional[str] = None,
    open_browser: bool = True,
    timeout_seconds: float = 15.0,
    insecure: bool = False,
    ca_bundle: Optional[str] = None,
    min_key_ttl_seconds: int = 5 * 60,
) -> Dict[str, Any]:
    """Run the Nous device-code flow and return full OAuth state without persisting."""
    pconfig = PROVIDER_REGISTRY["nous"]
    portal_base_url = (
        portal_base_url
        or os.getenv("HERMES_PORTAL_BASE_URL")
        or os.getenv("NOUS_PORTAL_BASE_URL")
        or pconfig.portal_base_url
    ).rstrip("/")
    requested_inference_url = (
        inference_base_url
        or os.getenv("NOUS_INFERENCE_BASE_URL")
        or pconfig.inference_base_url
    ).rstrip("/")
    client_id = client_id or pconfig.client_id
    scope = scope or pconfig.scope
    timeout = httpx.Timeout(timeout_seconds)
    verify: bool | str = False if insecure else (ca_bundle if ca_bundle else True)

    if _is_remote_session():
        open_browser = False

    print(f"Starting Hermes login via {pconfig.name}...")
    print(f"Portal: {portal_base_url}")
    if insecure:
        print("TLS verification: disabled (--insecure)")
    elif ca_bundle:
        print(f"TLS verification: custom CA bundle ({ca_bundle})")

    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
        device_data = _request_device_code(
            client=client,
            portal_base_url=portal_base_url,
            client_id=client_id,
            scope=scope,
        )

        verification_url = str(device_data["verification_uri_complete"])
        user_code = str(device_data["user_code"])
        expires_in = int(device_data["expires_in"])
        interval = int(device_data["interval"])

        print()
        print("To continue:")
        print(f"  1. Open: {verification_url}")
        print(f"  2. If prompted, enter code: {user_code}")

        if open_browser:
            opened = webbrowser.open(verification_url)
            if opened:
                print("  (Opened browser for verification)")
            else:
                print("  Could not open browser automatically — use the URL above.")

        effective_interval = max(1, min(interval, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS))
        print(f"Waiting for approval (polling every {effective_interval}s)...")

        token_data = _poll_for_token(
            client=client,
            portal_base_url=portal_base_url,
            client_id=client_id,
            device_code=str(device_data["device_code"]),
            expires_in=expires_in,
            poll_interval=interval,
        )

    now = datetime.now(timezone.utc)
    token_expires_in = _coerce_ttl_seconds(token_data.get("expires_in", 0))
    expires_at = now.timestamp() + token_expires_in
    resolved_inference_url = (
        _optional_base_url(token_data.get("inference_base_url"))
        or requested_inference_url
    )
    if resolved_inference_url != requested_inference_url:
        print(f"Using portal-provided inference URL: {resolved_inference_url}")

    auth_state = {
        "portal_base_url": portal_base_url,
        "inference_base_url": resolved_inference_url,
        "client_id": client_id,
        "scope": token_data.get("scope") or scope,
        "token_type": token_data.get("token_type", "Bearer"),
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "obtained_at": now.isoformat(),
        "expires_at": datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
        "expires_in": token_expires_in,
        "tls": {
            "insecure": verify is False,
            "ca_bundle": verify if isinstance(verify, str) else None,
        },
        "agent_key": None,
        "agent_key_id": None,
        "agent_key_expires_at": None,
        "agent_key_expires_in": None,
        "agent_key_reused": None,
        "agent_key_obtained_at": None,
    }
    try:
        return refresh_nous_oauth_from_state(
            auth_state,
            min_key_ttl_seconds=min_key_ttl_seconds,
            timeout_seconds=timeout_seconds,
            force_refresh=False,
            force_mint=True,
        )
    except AuthError as exc:
        if exc.code == "subscription_required":
            portal_url = auth_state.get(
                "portal_base_url", DEFAULT_NOUS_PORTAL_URL
            ).rstrip("/")
            print()
            print("Your Nous Portal account does not have an active subscription.")
            print(f"  Subscribe here: {portal_url}/billing")
            print()
            print("After subscribing, run `hermes model` again to finish setup.")
            raise SystemExit(1)
        raise


def _login_nous(args, pconfig: ProviderConfig) -> None:
    """Nous Portal device authorization flow."""
    timeout_seconds = getattr(args, "timeout", None) or 15.0
    insecure = bool(getattr(args, "insecure", False))
    ca_bundle = (
        getattr(args, "ca_bundle", None)
        or os.getenv("HERMES_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE")
    )

    try:
        auth_state = None

        # Codex-style auto-import: before launching a fresh device-code
        # flow, check the shared store for an existing Nous credential
        # from any other profile. If present, offer to rehydrate it.
        shared = _read_shared_nous_state()
        if shared:
            try:
                shared_path = _nous_shared_store_path()
            except RuntimeError:
                shared_path = None
            print()
            if shared_path:
                print(f"Found existing Nous OAuth credentials at {shared_path}")
            else:
                print("Found existing shared Nous OAuth credentials")
            try:
                do_import = input("Import these credentials? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                do_import = "y"
            if do_import in {"", "y", "yes"}:
                print("Rehydrating Nous session from shared credentials...")
                auth_state = _try_import_shared_nous_state(
                    timeout_seconds=timeout_seconds,
                    min_key_ttl_seconds=5 * 60,
                )
                if auth_state is None:
                    print("Could not refresh shared credentials — falling back to device-code login.")

        if auth_state is None:
            auth_state = _nous_device_code_login(
                portal_base_url=getattr(args, "portal_url", None),
                inference_base_url=getattr(args, "inference_url", None),
                client_id=getattr(args, "client_id", None) or pconfig.client_id,
                scope=getattr(args, "scope", None) or pconfig.scope,
                open_browser=not getattr(args, "no_browser", False),
                timeout_seconds=timeout_seconds,
                insecure=insecure,
                ca_bundle=ca_bundle,
                min_key_ttl_seconds=5 * 60,
            )

        inference_base_url = auth_state["inference_base_url"]

        # Snapshot the prior active_provider BEFORE _save_provider_state
        # overwrites it to "nous".  If the user picks "Skip (keep current)"
        # during model selection below, we restore this so the user's previous
        # provider (e.g. openrouter) is preserved.
        with _auth_store_lock():
            _prior_store = _load_auth_store()
            prior_active_provider = _prior_store.get("active_provider")

        with _auth_store_lock():
            auth_store = _load_auth_store()
            _save_provider_state(auth_store, "nous", auth_state)
            saved_to = _save_auth_store(auth_store)

        # Mirror to the shared store so other profiles can one-tap import
        # these credentials. Best-effort: any I/O failure is logged and
        # swallowed inside the helper.
        _write_shared_nous_state(auth_state)

        print()
        print("Login successful!")
        print(f"  Auth state: {saved_to}")

        # Resolve model BEFORE writing provider to config.yaml so we never
        # leave the config in a half-updated state (provider=nous but model
        # still set to the previous provider's model, e.g. opus from
        # OpenRouter).  The auth.json active_provider was already set above.
        selected_model = None
        try:
            runtime_key = auth_state.get("agent_key") or auth_state.get("access_token")
            if not isinstance(runtime_key, str) or not runtime_key:
                raise AuthError(
                    "No runtime API key available to fetch models",
                    provider="nous",
                    code="invalid_token",
                )

            from hermes_cli.models import (
                get_curated_nous_model_ids, get_pricing_for_provider,
                check_nous_free_tier, partition_nous_models_by_tier,
                union_with_portal_free_recommendations,
                union_with_portal_paid_recommendations,
            )
            model_ids = get_curated_nous_model_ids()

            print()
            unavailable_models: list = []
            if model_ids:
                pricing = get_pricing_for_provider("nous")
                free_tier = check_nous_free_tier()
                _portal_for_recs = auth_state.get("portal_base_url", "")
                if free_tier:
                    # The Portal's freeRecommendedModels endpoint is the
                    # source of truth for what's free *right now*. Augment
                    # the curated list with anything new the Portal flags
                    # as free so users on older Hermes builds still see
                    # newly-launched free models without a CLI release.
                    model_ids, pricing = union_with_portal_free_recommendations(
                        model_ids, pricing, _portal_for_recs,
                    )
                    model_ids, unavailable_models = partition_nous_models_by_tier(
                        model_ids, pricing, free_tier=True,
                    )
                else:
                    # Paid-tier mirror: pull paidRecommendedModels so newly
                    # launched paid models surface in the picker even if
                    # the in-repo curated list and docs-hosted manifest
                    # haven't caught up yet.
                    model_ids, pricing = union_with_portal_paid_recommendations(
                        model_ids, pricing, _portal_for_recs,
                    )
            _portal = auth_state.get("portal_base_url", "")
            if model_ids:
                print(f"Showing {len(model_ids)} curated models — use \"Enter custom model name\" for others.")
                selected_model = _prompt_model_selection(
                    model_ids, pricing=pricing,
                    unavailable_models=unavailable_models,
                    portal_url=_portal,
                )
            elif unavailable_models:
                _url = (_portal or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
                print("No free models currently available.")
                print(f"Upgrade at {_url} to access paid models.")
            else:
                print("No curated models available for Nous Portal.")
        except Exception as exc:
            message = format_auth_error(exc) if isinstance(exc, AuthError) else str(exc)
            print()
            print(f"Login succeeded, but could not fetch available models. Reason: {message}")

        # Write provider + model atomically so config is never mismatched.
        # If no model was selected (user picked "Skip (keep current)",
        # model list fetch failed, or no curated models were available),
        # preserve the user's previous provider — don't silently switch
        # them to Nous with a mismatched model.  The Nous OAuth tokens
        # stay saved for future use.
        if not selected_model:
            # Restore the prior active_provider that _save_provider_state
            # overwrote to "nous".  config.yaml model.provider is left
            # untouched, so the user's previous provider is fully preserved.
            with _auth_store_lock():
                auth_store = _load_auth_store()
                if prior_active_provider:
                    auth_store["active_provider"] = prior_active_provider
                else:
                    auth_store.pop("active_provider", None)
                _save_auth_store(auth_store)
            print()
            print("No provider change. Nous credentials saved for future use.")
            print("  Run `hermes model` again to switch to Nous Portal.")
            return

        config_path = _update_config_for_provider(
            "nous", inference_base_url, default_model=selected_model,
        )
        if selected_model:
            _save_model_choice(selected_model)
            print(f"Default model set to: {selected_model}")
        print(f"  Config updated: {config_path} (model.provider=nous)")

    except KeyboardInterrupt:
        print("\nLogin cancelled.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"Login failed: {exc}")
        raise SystemExit(1)


def logout_command(args) -> None:
    """Clear auth state for a provider."""
    provider_id = getattr(args, "provider", None)

    if provider_id and not is_known_auth_provider(provider_id):
        print(f"Unknown provider: {provider_id}")
        raise SystemExit(1)

    active = get_active_provider()
    target = provider_id or active or _logout_default_provider_from_config()

    if not target:
        print("No provider is currently logged in.")
        return

    should_reset_config = _should_reset_config_provider_on_logout(target)
    provider_name = get_auth_provider_display_name(target)

    if clear_provider_auth(target) or should_reset_config:
        if should_reset_config:
            _reset_config_provider()
        print(f"Logged out of {provider_name}.")
        if should_reset_config and os.getenv("OPENROUTER_API_KEY"):
            print("Hermes will use OpenRouter for inference.")
        elif should_reset_config:
            print("Run `hermes model` or configure an API key to use Hermes.")
        else:
            print("Model provider configuration was unchanged.")
    else:
        print(f"No auth state found for {provider_name}.")
