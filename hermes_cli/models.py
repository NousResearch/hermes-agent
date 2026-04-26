"""
Canonical model catalogs and lightweight validation helpers.

Add, remove, or reorder entries here — both `hermes setup` and
`hermes` provider-selection will pick up the change automatically.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
import time
from difflib import get_close_matches
from pathlib import Path
from typing import Any, NamedTuple, Optional

from hermes_cli import __version__ as _HERMES_VERSION

# Identify ourselves so endpoints fronted by Cloudflare's Browser Integrity
# Check (error 1010) don't reject the default ``Python-urllib/*`` signature.
_HERMES_USER_AGENT = f"hermes-cli/{_HERMES_VERSION}"

COPILOT_BASE_URL = "https://api.githubcopilot.com"
COPILOT_MODELS_URL = f"{COPILOT_BASE_URL}/models"
COPILOT_EDITOR_VERSION = "vscode/1.104.1"
COPILOT_REASONING_EFFORTS_GPT5 = ["minimal", "low", "medium", "high"]
COPILOT_REASONING_EFFORTS_O_SERIES = ["low", "medium", "high"]


# Fallback OpenRouter snapshot used when the live catalog is unavailable.
# (model_id, display description shown in menus)
OPENROUTER_MODELS: list[tuple[str, str]] = [
    ("moonshotai/kimi-k2.6",            "recommended"),
    ("anthropic/claude-opus-4.7",       ""),
    ("anthropic/claude-opus-4.6",       ""),
    ("anthropic/claude-sonnet-4.6",     ""),
    ("qwen/qwen3.6-plus",               ""),
    ("anthropic/claude-sonnet-4.5",     ""),
    ("anthropic/claude-haiku-4.5",      ""),
    ("openrouter/elephant-alpha",       "free"),
    ("openai/gpt-5.5",                  ""),
    ("openai/gpt-5.4-mini",             ""),
    ("xiaomi/mimo-v2.5-pro",             ""),
    ("xiaomi/mimo-v2.5",                 ""),
    ("openai/gpt-5.3-codex",            ""),
    ("google/gemini-3-pro-image-preview", ""),
    ("google/gemini-3-flash-preview",   ""),
    ("google/gemini-3.1-pro-preview",     ""),
    ("google/gemini-3.1-flash-lite-preview",   ""),
    ("qwen/qwen3.5-plus-02-15",         ""),
    ("qwen/qwen3.5-35b-a3b",            ""),
    ("stepfun/step-3.5-flash",          ""),
    ("minimax/minimax-m2.7",            ""),
    ("minimax/minimax-m2.5",            ""),
    ("minimax/minimax-m2.5:free",       "free"),
    ("z-ai/glm-5.1",                    ""),
    ("z-ai/glm-5v-turbo",               ""),
    ("z-ai/glm-5-turbo",                ""),
    ("x-ai/grok-4.20",                  ""),
    ("nvidia/nemotron-3-super-120b-a12b",      ""),
    ("nvidia/nemotron-3-super-120b-a12b:free", "free"),
    ("arcee-ai/trinity-large-preview:free", "free"),
    ("arcee-ai/trinity-large-thinking",  ""),
    ("openai/gpt-5.5-pro",              ""),
    ("openai/gpt-5.4-nano",             ""),
]

_openrouter_catalog_cache: list[tuple[str, str]] | None = None


# Fallback Vercel AI Gateway snapshot used when the live catalog is unavailable.
# OSS / open-weight models prioritized first, then closed-source by family.
# Slugs match Vercel's actual /v1/models catalog (e.g. alibaba/ for Qwen,
# zai/ and xai/ without hyphens).
VERCEL_AI_GATEWAY_MODELS: list[tuple[str, str]] = [
    ("moonshotai/kimi-k2.6",                 "recommended"),
    ("alibaba/qwen3.6-plus",                 ""),
    ("zai/glm-5.1",                          ""),
    ("minimax/minimax-m2.7",                 ""),
    ("anthropic/claude-sonnet-4.6",          ""),
    ("anthropic/claude-opus-4.7",            ""),
    ("anthropic/claude-opus-4.6",            ""),
    ("anthropic/claude-haiku-4.5",           ""),
    ("openai/gpt-5.4",                       ""),
    ("openai/gpt-5.4-mini",                  ""),
    ("openai/gpt-5.3-codex",                 ""),
    ("google/gemini-3.1-pro-preview",        ""),
    ("google/gemini-3-flash",                ""),
    ("google/gemini-3.1-flash-lite-preview", ""),
    ("xai/grok-4.20-reasoning",              ""),
]

_ai_gateway_catalog_cache: list[tuple[str, str]] | None = None


def _codex_curated_models() -> list[str]:
    """Derive the openai-codex curated list from codex_models.py.

    Single source of truth: DEFAULT_CODEX_MODELS + forward-compat synthesis.
    This keeps the gateway /model picker in sync with the CLI `hermes model`
    flow without maintaining a separate static list.
    """
    from hermes_cli.codex_models import DEFAULT_CODEX_MODELS, _add_forward_compat_models
    return _add_forward_compat_models(list(DEFAULT_CODEX_MODELS))


_PROVIDER_MODELS: dict[str, list[str]] = {
    "nous": [
        "moonshotai/kimi-k2.6",
        "xiaomi/mimo-v2.5-pro",
        "xiaomi/mimo-v2.5",
        "anthropic/claude-opus-4.7",
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-haiku-4.5",
        "openai/gpt-5.5",
        "openai/gpt-5.4-mini",
        "openai/gpt-5.3-codex",
        "google/gemini-3-pro-preview",
        "google/gemini-3-flash-preview",
        "google/gemini-3.1-pro-preview",
        "google/gemini-3.1-flash-lite-preview",
        "qwen/qwen3.5-plus-02-15",
        "qwen/qwen3.5-35b-a3b",
        "stepfun/step-3.5-flash",
        "minimax/minimax-m2.7",
        "minimax/minimax-m2.5",
        "minimax/minimax-m2.5:free",
        "z-ai/glm-5.1",
        "z-ai/glm-5v-turbo",
        "z-ai/glm-5-turbo",
        "x-ai/grok-4.20-beta",
        "nvidia/nemotron-3-super-120b-a12b",
        "arcee-ai/trinity-large-thinking",
        "openai/gpt-5.5-pro",
        "openai/gpt-5.4-nano",
    ],
    # Native OpenAI Chat Completions (api.openai.com). Used by /model counts and
    # provider_model_ids fallback when /v1/models is unavailable.
    "openai": [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5-mini",
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "openai-codex": _codex_curated_models(),
    "copilot-acp": [
        "copilot-acp",
    ],
    "copilot": [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5-mini",
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
        "claude-sonnet-4.6",
        "claude-sonnet-4",
        "claude-sonnet-4.5",
        "claude-haiku-4.5",
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "grok-code-fast-1",
    ],
    "gemini": [
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
    ],
    "google-gemini-cli": [
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
    ],
    "zai": [
        "glm-5.1",
        "glm-5",
        "glm-5v-turbo",
        "glm-5-turbo",
        "glm-4.7",
        "glm-4.5",
        "glm-4.5-flash",
    ],
    "xai": [
        "grok-4.20-reasoning",
        "grok-4-1-fast-reasoning",
    ],
    "nvidia": [
        # NVIDIA flagship reasoning models
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/nemotron-3-nano-30b-a3b",
        "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        # Third-party agentic models hosted on build.nvidia.com
        # (map to OpenRouter defaults — users get familiar picks on NIM)
        "qwen/qwen3.5-397b-a17b",
        "deepseek-ai/deepseek-v3.2",
        "moonshotai/kimi-k2.6",
        "minimaxai/minimax-m2.5",
        "z-ai/glm5",
        "openai/gpt-oss-120b",
    ],
    "kimi-coding": [
        "kimi-k2.6",
        "kimi-k2.5",
        "kimi-for-coding",
        "kimi-k2-thinking",
        "kimi-k2-thinking-turbo",
        "kimi-k2-turbo-preview",
        "kimi-k2-0905-preview",
    ],
    "kimi-coding-cn": [
        "kimi-k2.6",
        "kimi-k2.5",
        "kimi-k2-thinking",
        "kimi-k2-turbo-preview",
        "kimi-k2-0905-preview",
    ],
    "stepfun": [
        "step-3.5-flash",
        "step-3.5-flash-2603",
    ],
    "moonshot": [
        "kimi-k2.6",
        "kimi-k2.5",
        "kimi-k2-thinking",
        "kimi-k2-turbo-preview",
        "kimi-k2-0905-preview",
    ],
    "minimax": [
        "MiniMax-M2.7",
        "MiniMax-M2.5",
        "MiniMax-M2.1",
        "MiniMax-M2",
    ],
    "minimax-cn": [
        "MiniMax-M2.7",
        "MiniMax-M2.5",
        "MiniMax-M2.1",
        "MiniMax-M2",
    ],
    "anthropic": [
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-haiku-4-5-20251001",
    ],
    "deepseek": [
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-chat",
        "deepseek-reasoner",
    ],
    "xiaomi": [
        "mimo-v2.5-pro",
        "mimo-v2.5",
        "mimo-v2-pro",
        "mimo-v2-omni",
        "mimo-v2-flash",
    ],
    "arcee": [
        "trinity-large-thinking",
        "trinity-large-preview",
        "trinity-mini",
    ],
    "opencode-zen": [
        "kimi-k2.5",
        "gpt-5.4-pro",
        "gpt-5.4",
        "gpt-5.3-codex",
        "gpt-5.2",
        "gpt-5.2-codex",
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini",
        "gpt-5",
        "gpt-5-codex",
        "gpt-5-nano",
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-opus-4-1",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-sonnet-4",
        "claude-haiku-4-5",
        "claude-3-5-haiku",
        "gemini-3.1-pro",
        "gemini-3-pro",
        "gemini-3-flash",
        "minimax-m2.7",
        "minimax-m2.5",
        "minimax-m2.5-free",
        "minimax-m2.1",
        "glm-5",
        "glm-4.7",
        "glm-4.6",
        "kimi-k2-thinking",
        "kimi-k2",
        "qwen3-coder",
        "big-pickle",
    ],
    "opencode-go": [
        "kimi-k2.6",
        "kimi-k2.5",
        "glm-5.1",
        "glm-5",
        "mimo-v2.5-pro",
        "mimo-v2.5",
        "mimo-v2-pro",
        "mimo-v2-omni",
        "minimax-m2.7",
        "minimax-m2.5",
        "qwen3.6-plus",
        "qwen3.5-plus",
    ],
    "kilocode": [
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
        "google/gemini-3-pro-preview",
        "google/gemini-3-flash-preview",
    ],
    # Alibaba DashScope Coding platform (coding-intl) — default endpoint.
    # Supports Qwen models + third-party providers (GLM, Kimi, MiniMax).
    # Users with classic DashScope keys should override DASHSCOPE_BASE_URL
    # to https://dashscope-intl.aliyuncs.com/compatible-mode/v1 (OpenAI-compat)
    # or https://dashscope-intl.aliyuncs.com/apps/anthropic (Anthropic-compat).
    "alibaba": [
        "kimi-k2.5",
        "qwen3.5-plus",
        "qwen3-coder-plus",
        "qwen3-coder-next",
        # Third-party models available on coding-intl
        "glm-5",
        "glm-4.7",
        "MiniMax-M2.5",
    ],
    # Curated HF model list — only agentic models that map to OpenRouter defaults.
    "huggingface": [
        "moonshotai/Kimi-K2.5",
        "Qwen/Qwen3.5-397B-A17B",
        "Qwen/Qwen3.5-35B-A3B",
        "deepseek-ai/DeepSeek-V3.2",
        "MiniMaxAI/MiniMax-M2.5",
        "zai-org/GLM-5",
        "XiaomiMiMo/MiMo-V2-Flash",
        "moonshotai/Kimi-K2-Thinking",
        "moonshotai/Kimi-K2.6",
    ],
    # AWS Bedrock — static fallback list used when dynamic discovery is
    # unavailable (no boto3, no credentials, or API error).  The agent
    # prefers live discovery via ListFoundationModels + ListInferenceProfiles.
    # Use inference profile IDs (us.*) since most models require them.
    "bedrock": [
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-opus-4-6-v1",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.amazon.nova-pro-v1:0",
        "us.amazon.nova-lite-v1:0",
        "us.amazon.nova-micro-v1:0",
        "deepseek.v3.2",
        "us.meta.llama4-maverick-17b-instruct-v1:0",
        "us.meta.llama4-scout-17b-instruct-v1:0",
    ],
    # Azure Foundry: user-provided endpoint and model.
    # Empty list because models depend on the endpoint configuration.
    "azure-foundry": [],
}

# Vercel AI Gateway: derive the bare-model-id catalog from the curated
# ``VERCEL_AI_GATEWAY_MODELS`` snapshot so both the picker (tuples with descriptions)
# and the static fallback catalog (bare ids) stay in sync from a single
# source of truth.
_PROVIDER_MODELS["ai-gateway"] = [mid for mid, _ in VERCEL_AI_GATEWAY_MODELS]

# ---------------------------------------------------------------------------
# Nous Portal free-model helper
# ---------------------------------------------------------------------------
# The Nous Portal models endpoint is the source of truth for which models
# are currently offered (free or paid). We trust whatever it returns and
# surface it to users as-is — no local allowlist filtering.


def _is_model_free(model_id: str, pricing: dict[str, dict[str, str]]) -> bool:
    """Return True if *model_id* has zero-cost prompt AND completion pricing."""
    p = pricing.get(model_id)
    if not p:
        return False
    try:
        return float(p.get("prompt", "1")) == 0 and float(p.get("completion", "1")) == 0
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Nous Portal account tier detection
# ---------------------------------------------------------------------------

def fetch_nous_account_tier(access_token: str, portal_base_url: str = "") -> dict[str, Any]:
    """Fetch the user's Nous Portal account/subscription info.

    Calls ``<portal>/api/oauth/account`` with the OAuth access token.

    Returns the parsed JSON dict on success, e.g.::

        {
            "subscription": {
                "plan": "Plus",
                "tier": 2,
                "monthly_charge": 20,
                "credits_remaining": 1686.60,
                ...
            },
            ...
        }

    Returns an empty dict on any failure (network, auth, parse).
    """
    base = (portal_base_url or "https://portal.nousresearch.com").rstrip("/")
    url = f"{base}/api/oauth/account"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return {}


def is_nous_free_tier(account_info: dict[str, Any]) -> bool:
    """Return True if the account info indicates a free (unpaid) tier.

    Checks ``subscription.monthly_charge == 0``.  Returns False when
    the field is missing or unparseable (assumes paid — don't block users).
    """
    sub = account_info.get("subscription")
    if not isinstance(sub, dict):
        return False
    charge = sub.get("monthly_charge")
    if charge is None:
        return False
    try:
        return float(charge) == 0
    except (TypeError, ValueError):
        return False


def partition_nous_models_by_tier(
    model_ids: list[str],
    pricing: dict[str, dict[str, str]],
    free_tier: bool,
) -> tuple[list[str], list[str]]:
    """Split Nous models into (selectable, unavailable) based on user tier.

    For paid-tier users: all models are selectable, none unavailable.

    For free-tier users: only free models are selectable; paid models
    are returned as unavailable (shown grayed out in the menu).
    """
    if not free_tier:
        return (model_ids, [])

    if not pricing:
        return (model_ids, [])  # can't determine, show everything

    selectable: list[str] = []
    unavailable: list[str] = []
    for mid in model_ids:
        if _is_model_free(mid, pricing):
            selectable.append(mid)
        else:
            unavailable.append(mid)
    return (selectable, unavailable)


# ---------------------------------------------------------------------------
# TTL cache for free-tier detection — avoids repeated API calls within a
# session while still picking up upgrades quickly.
# ---------------------------------------------------------------------------
_FREE_TIER_CACHE_TTL: int = 180  # seconds (3 minutes)
_free_tier_cache: tuple[bool, float] | None = None  # (result, timestamp)


def check_nous_free_tier() -> bool:
    """Check if the current Nous Portal user is on a free (unpaid) tier.

    Results are cached for ``_FREE_TIER_CACHE_TTL`` seconds to avoid
    hitting the Portal API on every call.  The cache is short-lived so
    that an account upgrade is reflected within a few minutes.

    Returns False (assume paid) on any error — never blocks paying users.
    """
    global _free_tier_cache
    now = time.monotonic()
    if _free_tier_cache is not None:
        cached_result, cached_at = _free_tier_cache
        if now - cached_at < _FREE_TIER_CACHE_TTL:
            return cached_result

    try:
        from hermes_cli.auth import get_provider_auth_state, resolve_nous_runtime_credentials

        # Ensure we have a fresh token (triggers refresh if needed)
        resolve_nous_runtime_credentials(min_key_ttl_seconds=60)

        state = get_provider_auth_state("nous")
        if not state:
            _free_tier_cache = (False, now)
            return False
        access_token = state.get("access_token", "")
        portal_url = state.get("portal_base_url", "")
        if not access_token:
            _free_tier_cache = (False, now)
            return False

        account_info = fetch_nous_account_tier(access_token, portal_url)
        result = is_nous_free_tier(account_info)
        _free_tier_cache = (result, now)
        return result
    except Exception:
        _free_tier_cache = (False, now)
        return False  # default to paid on error — don't block users


# ---------------------------------------------------------------------------
# Nous Portal recommended models
#
# The Portal publishes a curated list of suggested models (separated into
# paid and free tiers) plus dedicated recommendations for compaction (text
# summarisation / auxiliary) and vision tasks. We fetch it once per process
# with a TTL cache so callers can ask "what's the best aux model right now?"
# without hitting the network on every lookup.
#
# Shape of the response (fields we care about):
#   {
#     "paidRecommendedModels":     [ {modelName, ...}, ... ],
#     "freeRecommendedModels":     [ {modelName, ...}, ... ],
#     "paidRecommendedCompactionModel":  {modelName, ...} | null,
#     "paidRecommendedVisionModel":      {modelName, ...} | null,
#     "freeRecommendedCompactionModel":  {modelName, ...} | null,
#     "freeRecommendedVisionModel":      {modelName, ...} | null,
#   }
# ---------------------------------------------------------------------------

NOUS_RECOMMENDED_MODELS_PATH = "/api/nous/recommended-models"
_NOUS_RECOMMENDED_CACHE_TTL: int = 600  # seconds (10 minutes)
# (result_dict, timestamp) keyed by portal_base_url so staging vs prod don't collide.
_nous_recommended_cache: dict[str, tuple[dict[str, Any], float]] = {}


def fetch_nous_recommended_models(
    portal_base_url: str = "",
    timeout: float = 5.0,
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Fetch the Nous Portal's curated recommended-models payload.

    Hits ``<portal>/api/nous/recommended-models``. The endpoint is public —
    no auth is required. Results are cached per portal URL for
    ``_NOUS_RECOMMENDED_CACHE_TTL`` seconds; pass ``force_refresh=True`` to
    bypass the cache.

    Returns the parsed JSON dict on success, or ``{}`` on any failure
    (network, parse, non-2xx). Callers must treat missing/null fields as
    "no recommendation" and fall back to their own default.
    """
    base = (portal_base_url or "https://portal.nousresearch.com").rstrip("/")
    now = time.monotonic()
    cached = _nous_recommended_cache.get(base)
    if not force_refresh and cached is not None:
        payload, cached_at = cached
        if now - cached_at < _NOUS_RECOMMENDED_CACHE_TTL:
            return payload

    url = f"{base}{NOUS_RECOMMENDED_MODELS_PATH}"
    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}

    _nous_recommended_cache[base] = (data, now)
    return data


def _resolve_nous_portal_url() -> str:
    """Best-effort lookup of the Portal base URL the user is authed against."""
    try:
        from hermes_cli.auth import (
            DEFAULT_NOUS_PORTAL_URL,
            get_provider_auth_state,
        )
        state = get_provider_auth_state("nous") or {}
        portal = str(state.get("portal_base_url") or "").strip()
        if portal:
            return portal.rstrip("/")
        return str(DEFAULT_NOUS_PORTAL_URL).rstrip("/")
    except Exception:
        return "https://portal.nousresearch.com"


def _extract_model_name(entry: Any) -> Optional[str]:
    """Pull the ``modelName`` field from a recommended-model entry, else None."""
    if not isinstance(entry, dict):
        return None
    model_name = entry.get("modelName")
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()
    return None


def get_nous_recommended_aux_model(
    *,
    vision: bool = False,
    free_tier: Optional[bool] = None,
    portal_base_url: str = "",
    force_refresh: bool = False,
) -> Optional[str]:
    """Return the Portal's recommended model name for an auxiliary task.

    Picks the best field from the Portal's recommended-models payload:

    * ``vision=True``  → ``paidRecommendedVisionModel``  (paid tier) or
                         ``freeRecommendedVisionModel``  (free tier)
    * ``vision=False`` → ``paidRecommendedCompactionModel`` or
                         ``freeRecommendedCompactionModel``

    When ``free_tier`` is ``None`` (default) the user's tier is auto-detected
    via :func:`check_nous_free_tier`. Pass an explicit bool to bypass the
    detection — useful for tests or when the caller already knows the tier.

    For paid-tier users we prefer the paid recommendation but gracefully fall
    back to the free recommendation if the Portal returned ``null`` for the
    paid field (common during the staged rollout of new paid models).

    Returns ``None`` when every candidate is missing, null, or the fetch
    fails — callers should fall back to their own default (currently
    ``google/gemini-3-flash-preview``).
    """
    base = portal_base_url or _resolve_nous_portal_url()
    payload = fetch_nous_recommended_models(base, force_refresh=force_refresh)
    if not payload:
        return None

    if free_tier is None:
        try:
            free_tier = check_nous_free_tier()
        except Exception:
            # On any detection error, assume paid — paid users see both fields
            # anyway so this is a safe default that maximises model quality.
            free_tier = False

    if vision:
        paid_key, free_key = "paidRecommendedVisionModel", "freeRecommendedVisionModel"
    else:
        paid_key, free_key = "paidRecommendedCompactionModel", "freeRecommendedCompactionModel"

    # Preference order:
    #   free tier  → free only
    #   paid tier  → paid, then free (if paid field is null)
    candidates = [free_key] if free_tier else [paid_key, free_key]
    for key in candidates:
        name = _extract_model_name(payload.get(key))
        if name:
            return name
    return None


# ---------------------------------------------------------------------------
# Canonical provider list — single source of truth for provider identity.
# Every code path that lists, displays, or iterates providers derives from
# this list:  hermes model, /model, list_authenticated_providers.
#
# Fields:
#   slug        — internal provider ID (used in config.yaml, --provider flag)
#   label       — short display name
#   tui_desc    — longer description for the `hermes model` interactive picker
# ---------------------------------------------------------------------------

class ProviderEntry(NamedTuple):
    slug: str
    label: str
    tui_desc: str   # detailed description for `hermes model` TUI


CANONICAL_PROVIDERS: list[ProviderEntry] = [
    ProviderEntry("nous",           "Nous Portal",              "Nous Portal (Nous Research subscription)"),
    ProviderEntry("openrouter",     "OpenRouter",               "OpenRouter (100+ models, pay-per-use)"),
    ProviderEntry("ai-gateway",     "Vercel AI Gateway",        "Vercel AI Gateway (200+ models, $5 free credit, no markup)"),
    ProviderEntry("anthropic",      "Anthropic",                "Anthropic (Claude models — API key or Claude Code)"),
    ProviderEntry("openai-codex",   "OpenAI Codex",             "OpenAI Codex"),
    ProviderEntry("xiaomi",         "Xiaomi MiMo",              "Xiaomi MiMo (MiMo-V2.5 and V2 models — pro, omni, flash)"),
    ProviderEntry("nvidia",         "NVIDIA NIM",               "NVIDIA NIM (Nemotron models — build.nvidia.com or local NIM)"),
    ProviderEntry("qwen-oauth",     "Qwen OAuth (Portal)",      "Qwen OAuth (reuses local Qwen CLI login)"),
    ProviderEntry("copilot",        "GitHub Copilot",           "GitHub Copilot (uses GITHUB_TOKEN or gh auth token)"),
    ProviderEntry("copilot-acp",    "GitHub Copilot ACP",       "GitHub Copilot ACP (spawns `copilot --acp --stdio`)"),
    ProviderEntry("huggingface",    "Hugging Face",             "Hugging Face Inference Providers (20+ open models)"),
    ProviderEntry("gemini",         "Google AI Studio",         "Google AI Studio (Gemini models — native Gemini API)"),
    ProviderEntry("google-gemini-cli", "Google Gemini (OAuth)",   "Google Gemini via OAuth + Code Assist (free tier supported; no API key needed)"),
    ProviderEntry("deepseek",       "DeepSeek",                 "DeepSeek (DeepSeek-V3, R1, coder — direct API)"),
    ProviderEntry("xai",            "xAI",                      "xAI (Grok models — direct API)"),
    ProviderEntry("zai",            "Z.AI / GLM",               "Z.AI / GLM (Zhipu AI direct API)"),
    ProviderEntry("kimi-coding",    "Kimi / Kimi Coding Plan",  "Kimi Coding Plan (api.kimi.com) & Moonshot API"),
    ProviderEntry("kimi-coding-cn", "Kimi / Moonshot (China)",  "Kimi / Moonshot China (Moonshot CN direct API)"),
    ProviderEntry("stepfun",        "StepFun Step Plan",       "StepFun Step Plan (agent/coding models via Step Plan API)"),
    ProviderEntry("minimax",        "MiniMax",                  "MiniMax (global direct API)"),
    ProviderEntry("minimax-cn",     "MiniMax (China)",          "MiniMax China (domestic direct API)"),
    ProviderEntry("alibaba",        "Alibaba Cloud (DashScope)","Alibaba Cloud / DashScope Coding (Qwen + multi-provider)"),
    ProviderEntry("ollama-cloud",   "Ollama Cloud",             "Ollama Cloud (cloud-hosted open models — ollama.com)"),
    ProviderEntry("arcee",          "Arcee AI",                 "Arcee AI (Trinity models — direct API)"),
    ProviderEntry("kilocode",       "Kilo Code",                "Kilo Code (Kilo Gateway API)"),
    ProviderEntry("opencode-zen",   "OpenCode Zen",             "OpenCode Zen (35+ curated models, pay-as-you-go)"),
    ProviderEntry("opencode-go",    "OpenCode Go",              "OpenCode Go (open models, $10/month subscription)"),
    ProviderEntry("bedrock",        "AWS Bedrock",              "AWS Bedrock (Claude, Nova, Llama, DeepSeek — IAM or API key)"),
    ProviderEntry("azure-foundry",  "Azure Foundry",            "Azure Foundry (OpenAI-style or Anthropic-style endpoint — your Azure AI deployment)"),
]

# Derived dicts — used throughout the codebase
_PROVIDER_LABELS = {p.slug: p.label for p in CANONICAL_PROVIDERS}
_PROVIDER_LABELS["custom"] = "Custom endpoint"  # special case: not a named provider


_PROVIDER_ALIASES = {
    "glm": "zai",
    "z-ai": "zai",
    "z.ai": "zai",
    "zhipu": "zai",
    "github": "copilot",
    "github-copilot": "copilot",
    "github-models": "copilot",
    "github-model": "copilot",
    "github-copilot-acp": "copilot-acp",
    "copilot-acp-agent": "copilot-acp",
    "google": "gemini",
    "google-gemini": "gemini",
    "google-ai-studio": "gemini",
    "kimi": "kimi-coding",
    "moonshot": "kimi-coding",
    "kimi-cn": "kimi-coding-cn",
    "moonshot-cn": "kimi-coding-cn",
    "step": "stepfun",
    "stepfun-coding-plan": "stepfun",
    "arcee-ai": "arcee",
    "arceeai": "arcee",
    "minimax-china": "minimax-cn",
    "minimax_cn": "minimax-cn",
    "claude": "anthropic",
    "claude-code": "anthropic",
    "deep-seek": "deepseek",
    "opencode": "opencode-zen",
    "zen": "opencode-zen",
    "go": "opencode-go",
    "opencode-go-sub": "opencode-go",
    "aigateway": "ai-gateway",
    "vercel": "ai-gateway",
    "vercel-ai-gateway": "ai-gateway",
    "kilo": "kilocode",
    "kilo-code": "kilocode",
    "kilo-gateway": "kilocode",
    "dashscope": "alibaba",
    "aliyun": "alibaba",
    "qwen": "alibaba",
    "alibaba-cloud": "alibaba",
    "qwen-portal": "qwen-oauth",
    "gemini-cli": "google-gemini-cli",
    "gemini-oauth": "google-gemini-cli",
    "hf": "huggingface",
    "hugging-face": "huggingface",
    "huggingface-hub": "huggingface",
    "mimo": "xiaomi",
    "xiaomi-mimo": "xiaomi",
    "aws": "bedrock",
    "aws-bedrock": "bedrock",
    "amazon-bedrock": "bedrock",
    "amazon": "bedrock",
    "grok": "xai",
    "x-ai": "xai",
    "x.ai": "xai",
    "nim": "nvidia",
    "nvidia-nim": "nvidia",
    "build-nvidia": "nvidia",
    "nemotron": "nvidia",
    "ollama": "custom",  # bare "ollama" = local; use "ollama-cloud" for cloud
    "ollama_cloud": "ollama-cloud",
}


def get_default_model_for_provider(provider: str) -> str:
    """Return the default model for a provider, or empty string if unknown.

    Uses the first entry in _PROVIDER_MODELS as the default.  This is the
    model a user would be offered first in the ``hermes model`` picker.

    Used as a fallback when the user has configured a provider but never
    selected a model (e.g. ``hermes auth add openai-codex`` without
    ``hermes model``).
    """
    models = _PROVIDER_MODELS.get(provider, [])
    return models[0] if models else ""


def _openrouter_model_is_free(pricing: Any) -> bool:
    """Return True when both prompt and completion pricing are zero."""
    if not isinstance(pricing, dict):
        return False
    try:
        return float(pricing.get("prompt", "0")) == 0 and float(pricing.get("completion", "0")) == 0
    except (TypeError, ValueError):
        return False


def _openrouter_model_supports_tools(item: Any) -> bool:
    """Return True when the model's ``supported_parameters`` advertise tool calling.

    hermes-agent is tool-calling-first — every provider path assumes the model
    can invoke tools. Models that don't advertise ``tools`` in their
    ``supported_parameters`` (e.g. image-only or completion-only models) cannot
    be driven by the agent loop and would fail at the first tool call.

    **Permissive when the field is missing.** Some OpenRouter-compatible gateways
    (Nous Portal, private mirrors, older catalog snapshots) don't populate
    ``supported_parameters`` at all. Treat that as "unknown capability → allow"
    so the picker doesn't silently empty for those users. Only hide models
    whose ``supported_parameters`` is an explicit list that omits ``tools``.

    Ported from Kilo-Org/kilocode#9068.
    """
    if not isinstance(item, dict):
        return True
    params = item.get("supported_parameters")
    if not isinstance(params, list):
        # Field absent / malformed / None — be permissive.
        return True
    return "tools" in params


def fetch_openrouter_models(
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> list[tuple[str, str]]:
    """Return the curated OpenRouter picker list, refreshed from the live catalog when possible."""
    global _openrouter_catalog_cache

    if _openrouter_catalog_cache is not None and not force_refresh:
        return list(_openrouter_catalog_cache)

    # Prefer the remotely-hosted catalog manifest; fall back to the in-repo
    # snapshot when the manifest is unreachable. Both are curated lists that
    # drive the picker; the OpenRouter live /v1/models filter (tool support,
    # free pricing) is applied on top either way.
    try:
        from hermes_cli.model_catalog import get_curated_openrouter_models
        remote = get_curated_openrouter_models()
    except Exception:
        remote = None
    fallback = list(remote) if remote else list(OPENROUTER_MODELS)
    preferred_ids = [mid for mid, _ in fallback]

    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return list(_openrouter_catalog_cache or fallback)

    live_items = payload.get("data", [])
    if not isinstance(live_items, list):
        return list(_openrouter_catalog_cache or fallback)

    live_by_id: dict[str, dict[str, Any]] = {}
    for item in live_items:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        if not mid:
            continue
        live_by_id[mid] = item

    curated: list[tuple[str, str]] = []
    for preferred_id in preferred_ids:
        live_item = live_by_id.get(preferred_id)
        if live_item is None:
            continue
        # Hide models that don't advertise tool-calling support — hermes-agent
        # requires it and surfacing them leads to immediate runtime failures
        # when the user selects them. Ported from Kilo-Org/kilocode#9068.
        if not _openrouter_model_supports_tools(live_item):
            continue
        desc = "free" if _openrouter_model_is_free(live_item.get("pricing")) else ""
        curated.append((preferred_id, desc))

    if not curated:
        return list(_openrouter_catalog_cache or fallback)

    first_id, _ = curated[0]
    curated[0] = (first_id, "recommended")
    _openrouter_catalog_cache = curated
    return list(curated)


def model_ids(*, force_refresh: bool = False) -> list[str]:
    """Return just the OpenRouter model-id strings."""
    return [mid for mid, _ in fetch_openrouter_models(force_refresh=force_refresh)]


def get_curated_nous_model_ids() -> list[str]:
    """Return the curated Nous Portal model-id list.

    Prefers the remotely-hosted catalog manifest (published under
    ``website/static/api/model-catalog.json``); falls back to the in-repo
    snapshot in ``_PROVIDER_MODELS["nous"]`` when the manifest is
    unreachable. Always returns a list (never None).
    """
    try:
        from hermes_cli.model_catalog import get_curated_nous_models
        remote = get_curated_nous_models()
    except Exception:
        remote = None
    if remote:
        return list(remote)
    return list(_PROVIDER_MODELS.get("nous", []))


def _ai_gateway_model_is_free(pricing: Any) -> bool:
    """Return True if an AI Gateway model has $0 input AND output pricing."""
    if not isinstance(pricing, dict):
        return False
    try:
        return float(pricing.get("input", "0")) == 0 and float(pricing.get("output", "0")) == 0
    except (TypeError, ValueError):
        return False


def fetch_ai_gateway_models(
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> list[tuple[str, str]]:
    """Return the curated AI Gateway picker list, refreshed from the live catalog when possible."""
    global _ai_gateway_catalog_cache

    if _ai_gateway_catalog_cache is not None and not force_refresh:
        return list(_ai_gateway_catalog_cache)

    from hermes_constants import AI_GATEWAY_BASE_URL

    fallback = list(VERCEL_AI_GATEWAY_MODELS)
    preferred_ids = [mid for mid, _ in fallback]

    try:
        req = urllib.request.Request(
            f"{AI_GATEWAY_BASE_URL.rstrip('/')}/models",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return list(_ai_gateway_catalog_cache or fallback)

    live_items = payload.get("data", [])
    if not isinstance(live_items, list):
        return list(_ai_gateway_catalog_cache or fallback)

    live_by_id: dict[str, dict[str, Any]] = {}
    for item in live_items:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        if not mid:
            continue
        live_by_id[mid] = item

    curated: list[tuple[str, str]] = []
    for preferred_id in preferred_ids:
        live_item = live_by_id.get(preferred_id)
        if live_item is None:
            continue
        desc = "free" if _ai_gateway_model_is_free(live_item.get("pricing")) else ""
        curated.append((preferred_id, desc))

    if not curated:
        return list(_ai_gateway_catalog_cache or fallback)

    # If the live catalog offers a free Moonshot model, auto-promote it to
    # position #1 as "recommended" — dynamic discovery without a PR.
    free_moonshot = next(
        (
            mid
            for mid, item in live_by_id.items()
            if mid.startswith("moonshotai/")
            and _ai_gateway_model_is_free(item.get("pricing"))
        ),
        None,
    )
    if free_moonshot:
        curated = [(mid, desc) for mid, desc in curated if mid != free_moonshot]
        curated.insert(0, (free_moonshot, "recommended"))
    else:
        first_id, _ = curated[0]
        curated[0] = (first_id, "recommended")

    _ai_gateway_catalog_cache = curated
    return list(curated)


def ai_gateway_model_ids(*, force_refresh: bool = False) -> list[str]:
    """Return just the AI Gateway model-id strings."""
    return [mid for mid, _ in fetch_ai_gateway_models(force_refresh=force_refresh)]




# ---------------------------------------------------------------------------
# Pricing helpers — fetch live pricing from OpenRouter-compatible /v1/models
# ---------------------------------------------------------------------------

# Cache: maps model_id → {"prompt": str, "completion": str} per endpoint
_pricing_cache: dict[str, dict[str, dict[str, str]]] = {}


def _format_price_per_mtok(per_token_str: str) -> str:
    """Convert a per-token price string to a human-friendly $/Mtok string.

    Always uses 2 decimal places so that prices align vertically when
    right-justified in a column (the decimal point stays in the same position).

    Examples:
        "0.000003"   → "$3.00"      (per million tokens)
        "0.00003"    → "$30.00"
        "0.00000015" → "$0.15"
        "0.0000001"  → "$0.10"
        "0.00018"    → "$180.00"
        "0"          → "free"
    """
    try:
        val = float(per_token_str)
    except (TypeError, ValueError):
        return "?"
    if val == 0:
        return "free"
    per_m = val * 1_000_000
    return f"${per_m:.2f}"


def format_model_pricing_table(
    models: list[tuple[str, str]],
    pricing_map: dict[str, dict[str, str]],
    current_model: str = "",
    indent: str = "      ",
) -> list[str]:
    """Build a column-aligned model+pricing table for terminal display.

    Returns a list of pre-formatted lines ready to print.
    *models* is ``[(model_id, description), ...]``.
    """
    if not models:
        return []

    # Build rows: (model_id, input_price, output_price, cache_price, is_current)
    rows: list[tuple[str, str, str, str, bool]] = []
    has_cache = False
    for mid, _desc in models:
        is_cur = mid == current_model
        p = pricing_map.get(mid)
        if p:
            inp = _format_price_per_mtok(p.get("prompt", ""))
            out = _format_price_per_mtok(p.get("completion", ""))
            cache_read = p.get("input_cache_read", "")
            cache = _format_price_per_mtok(cache_read) if cache_read else ""
            if cache:
                has_cache = True
        else:
            inp, out, cache = "", "", ""
        rows.append((mid, inp, out, cache, is_cur))

    name_col = max(len(r[0]) for r in rows) + 2
    # Compute price column widths from the actual data so decimals align
    price_col = max(
        max((len(r[1]) for r in rows if r[1]), default=4),
        max((len(r[2]) for r in rows if r[2]), default=4),
        3,  # minimum: "In" / "Out" header
    )
    cache_col = max(
        max((len(r[3]) for r in rows if r[3]), default=4),
        5,  # minimum: "Cache" header
    ) if has_cache else 0
    lines: list[str] = []

    # Header
    if has_cache:
        lines.append(f"{indent}{'Model':<{name_col}} {'In':>{price_col}}  {'Out':>{price_col}}  {'Cache':>{cache_col}}  /Mtok")
        lines.append(f"{indent}{'-' * name_col} {'-' * price_col}  {'-' * price_col}  {'-' * cache_col}")
    else:
        lines.append(f"{indent}{'Model':<{name_col}} {'In':>{price_col}}  {'Out':>{price_col}}  /Mtok")
        lines.append(f"{indent}{'-' * name_col} {'-' * price_col}  {'-' * price_col}")

    for mid, inp, out, cache, is_cur in rows:
        marker = "  ← current" if is_cur else ""
        if has_cache:
            lines.append(f"{indent}{mid:<{name_col}} {inp:>{price_col}}  {out:>{price_col}}  {cache:>{cache_col}}{marker}")
        else:
            lines.append(f"{indent}{mid:<{name_col}} {inp:>{price_col}}  {out:>{price_col}}{marker}")

    return lines


def fetch_models_with_pricing(
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai/api",
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, str]]:
    """Fetch ``/v1/models`` and return ``{model_id: {prompt, completion}}`` pricing.

    Results are cached per *base_url* so repeated calls are free.
    Works with any OpenRouter-compatible endpoint (OpenRouter, Nous Portal).
    """
    cache_key = (base_url or "").rstrip("/")
    if not force_refresh and cache_key in _pricing_cache:
        return _pricing_cache[cache_key]

    url = cache_key.rstrip("/") + "/v1/models"
    headers: dict[str, str] = {
        "Accept": "application/json",
        "User-Agent": _HERMES_USER_AGENT,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        _pricing_cache[cache_key] = {}
        return {}

    result: dict[str, dict[str, str]] = {}
    for item in payload.get("data", []):
        mid = item.get("id")
        pricing = item.get("pricing")
        if mid and isinstance(pricing, dict):
            entry: dict[str, str] = {
                "prompt": str(pricing.get("prompt", "")),
                "completion": str(pricing.get("completion", "")),
            }
            if pricing.get("input_cache_read"):
                entry["input_cache_read"] = str(pricing["input_cache_read"])
            if pricing.get("input_cache_write"):
                entry["input_cache_write"] = str(pricing["input_cache_write"])
            result[mid] = entry

    _pricing_cache[cache_key] = result
    return result


def fetch_ai_gateway_pricing(
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, str]]:
    """Fetch Vercel AI Gateway /v1/models and return hermes-shaped pricing.

    Vercel uses ``input`` / ``output`` field names; hermes's picker expects
    ``prompt`` / ``completion``. This translates. Cache read/write field names
    already match.
    """
    from hermes_constants import AI_GATEWAY_BASE_URL

    cache_key = AI_GATEWAY_BASE_URL.rstrip("/")
    if not force_refresh and cache_key in _pricing_cache:
        return _pricing_cache[cache_key]

    try:
        req = urllib.request.Request(
            f"{cache_key}/models",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        _pricing_cache[cache_key] = {}
        return {}

    result: dict[str, dict[str, str]] = {}
    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        pricing = item.get("pricing")
        if not (mid and isinstance(pricing, dict)):
            continue
        entry: dict[str, str] = {
            "prompt": str(pricing.get("input", "")),
            "completion": str(pricing.get("output", "")),
        }
        if pricing.get("input_cache_read"):
            entry["input_cache_read"] = str(pricing["input_cache_read"])
        if pricing.get("input_cache_write"):
            entry["input_cache_write"] = str(pricing["input_cache_write"])
        result[mid] = entry

    _pricing_cache[cache_key] = result
    return result


def _resolve_openrouter_api_key() -> str:
    """Best-effort OpenRouter API key for pricing fetch."""
    return os.getenv("OPENROUTER_API_KEY", "").strip()


def _resolve_nous_pricing_credentials() -> tuple[str, str]:
    """Return ``(api_key, base_url)`` for Nous Portal pricing, or empty strings."""
    try:
        from hermes_cli.auth import resolve_nous_runtime_credentials
        creds = resolve_nous_runtime_credentials()
        if creds:
            return (creds.get("api_key", ""), creds.get("base_url", ""))
    except Exception:
        pass
    return ("", "")


def get_pricing_for_provider(provider: str, *, force_refresh: bool = False) -> dict[str, dict[str, str]]:
    """Return live pricing for providers that support it (openrouter, nous, ai-gateway)."""
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return fetch_models_with_pricing(
            api_key=_resolve_openrouter_api_key(),
            base_url="https://openrouter.ai/api",
            force_refresh=force_refresh,
        )
    if normalized == "ai-gateway":
        return fetch_ai_gateway_pricing(force_refresh=force_refresh)
    if normalized == "nous":
        api_key, base_url = _resolve_nous_pricing_credentials()
        if base_url:
            # Nous base_url typically looks like https://inference-api.nousresearch.com/v1
            # We need the part before /v1 for our fetch function
            stripped = base_url.rstrip("/")
            if stripped.endswith("/v1"):
                stripped = stripped[:-3]
            return fetch_models_with_pricing(
                api_key=api_key,
                base_url=stripped,
                force_refresh=force_refresh,
            )
    return {}


# All provider IDs and aliases that are valid for the provider:model syntax.
_KNOWN_PROVIDER_NAMES: set[str] = (
    set(_PROVIDER_LABELS.keys())
    | set(_PROVIDER_ALIASES.keys())
    | {"openrouter", "custom"}
)


def list_available_providers() -> list[dict[str, str]]:
    """Return info about all providers the user could use with ``provider:model``.

    Each dict has ``id``, ``label``, and ``aliases``.
    Checks which providers have valid credentials configured.

    Derives the provider list from :data:`CANONICAL_PROVIDERS` (single
    source of truth shared with ``hermes model``, ``/model``, etc.).
    """
    # Derive display order from canonical list + custom
    provider_order = [p.slug for p in CANONICAL_PROVIDERS] + ["custom"]

    # Build reverse alias map
    aliases_for: dict[str, list[str]] = {}
    for alias, canonical in _PROVIDER_ALIASES.items():
        aliases_for.setdefault(canonical, []).append(alias)

    result = []
    for pid in provider_order:
        label = _PROVIDER_LABELS.get(pid, pid)
        alias_list = aliases_for.get(pid, [])
        # Check if this provider has credentials available
        has_creds = False
        try:
            from hermes_cli.auth import get_auth_status, has_usable_secret
            if pid == "custom":
                custom_base_url = _get_custom_base_url() or ""
                has_creds = bool(custom_base_url.strip())
            elif pid == "openrouter":
                has_creds = has_usable_secret(os.getenv("OPENROUTER_API_KEY", ""))
            else:
                status = get_auth_status(pid)
                has_creds = bool(status.get("logged_in") or status.get("configured"))
        except Exception:
            pass
        result.append({
            "id": pid,
            "label": label,
            "aliases": alias_list,
            "authenticated": has_creds,
        })
    return result


def parse_model_input(raw: str, current_provider: str) -> tuple[str, str]:
    """Parse ``/model`` input into ``(provider, model)``.

    Supports ``provider:model`` syntax to switch providers at runtime::

        openrouter:anthropic/claude-sonnet-4.5  →  ("openrouter", "anthropic/claude-sonnet-4.5")
        nous:hermes-3                           →  ("nous", "hermes-3")
        anthropic/claude-sonnet-4.5             →  (current_provider, "anthropic/claude-sonnet-4.5")
        gpt-5.4                                 →  (current_provider, "gpt-5.4")

    The colon is only treated as a provider delimiter if the left side is a
    recognized provider name or alias.  This avoids misinterpreting model names
    that happen to contain colons (e.g. ``anthropic/claude-3.5-sonnet:beta``).

    Returns ``(provider, model)`` where *provider* is either the explicit
    provider from the input or *current_provider* if none was specified.
    """
    stripped = raw.strip()
    colon = stripped.find(":")
    if colon > 0:
        provider_part = stripped[:colon].strip().lower()
        model_part = stripped[colon + 1:].strip()
        if provider_part and model_part and provider_part in _KNOWN_PROVIDER_NAMES:
            # Support custom:name:model triple syntax for named custom
            # providers.  ``custom:local:qwen`` → ("custom:local", "qwen").
            # Single colon ``custom:qwen`` → ("custom", "qwen") as before.
            if provider_part == "custom" and ":" in model_part:
                second_colon = model_part.find(":")
                custom_name = model_part[:second_colon].strip()
                actual_model = model_part[second_colon + 1:].strip()
                if custom_name and actual_model:
                    return (f"custom:{custom_name}", actual_model)
            return (normalize_provider(provider_part), model_part)
    return (current_provider, stripped)


def _get_custom_base_url() -> str:
    """Get the custom endpoint base_url from config.yaml."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        model_cfg = config.get("model", {})
        if isinstance(model_cfg, dict):
            return str(model_cfg.get("base_url", "")).strip()
    except Exception:
        pass
    return ""


def curated_models_for_provider(
    provider: Optional[str],
    *,
    force_refresh: bool = False,
) -> list[tuple[str, str]]:
    """Return ``(model_id, description)`` tuples for a provider's model list.

    Tries to fetch the live model list from the provider's API first,
    falling back to the static ``_PROVIDER_MODELS`` catalog if the API
    is unreachable.
    """
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return fetch_openrouter_models(force_refresh=force_refresh)

    # Try live API first (Codex, Nous, etc. all support /models)
    live = provider_model_ids(normalized)
    if live:
        return [(m, "") for m in live]

    # Fallback to static catalog
    models = _PROVIDER_MODELS.get(normalized, [])
    return [(m, "") for m in models]


def _provider_keys(provider: str) -> set[str]:
    key = (provider or "").strip().lower()
    normalized = normalize_provider(provider)
    return {k for k in (key, normalized) if k}


def _model_in_provider_catalog(name_lower: str, providers: set[str]) -> bool:
    return any(
        name_lower == model.lower()
        for provider in providers
        for model in _PROVIDER_MODELS.get(provider, [])
    )


_AGGREGATOR_PROVIDERS = frozenset(
    {"nous", "openrouter", "ai-gateway", "copilot", "kilocode"}
)


def _resolve_static_model_alias(
    name_lower: str,
    current_keys: set[str],
) -> Optional[tuple[str, str]]:
    """Resolve short aliases (e.g. sonnet/opus) using static catalogs only."""
    try:
        from hermes_cli.model_switch import MODEL_ALIASES
    except Exception:
        return None

    identity = MODEL_ALIASES.get(name_lower)
    if identity is None:
        return None

    vendor = identity.vendor
    family = identity.family

    def _match(provider: str) -> Optional[str]:
        models = _PROVIDER_MODELS.get(provider, [])
        if not models:
            return None
        prefix = (
            f"{vendor}/{family}"
            if provider in _AGGREGATOR_PROVIDERS
            else family
        ).lower()
        for model in models:
            if model.lower().startswith(prefix):
                return model
        return None

    for provider in current_keys:
        if matched := _match(provider):
            return provider, matched

    for provider in _PROVIDER_MODELS:
        if provider in current_keys or provider in _AGGREGATOR_PROVIDERS:
            continue
        if matched := _match(provider):
            return provider, matched

    for provider in _AGGREGATOR_PROVIDERS:
        if provider in current_keys and (matched := _match(provider)):
            return provider, matched

    return None


def detect_static_provider_for_model(
    model_name: str,
    current_provider: str,
) -> Optional[tuple[str, str]]:
    """Auto-detect a provider from static catalogs only.

    Returns ``(provider_id, model_name)``. The model name may be remapped
    when a static alias or bare provider name resolves to a catalog default.
    Returns ``None`` when no confident match is found.
    """
    name = (model_name or "").strip()
    if not name:
        return None

    name_lower = name.lower()
    current_keys = _provider_keys(current_provider)

    alias_match = _resolve_static_model_alias(name_lower, current_keys)
    if alias_match:
        return alias_match

    # --- Step 0: bare provider name typed as model ---
    # If someone types `/model nous` or `/model anthropic`, treat it as a
    # provider switch and pick the first model from that provider's catalog.
    # Skip "custom" and "openrouter" — custom has no model catalog, and
    # openrouter requires an explicit model name to be useful.
    resolved_provider = _PROVIDER_ALIASES.get(name_lower, name_lower)
    if resolved_provider not in {"custom", "openrouter"}:
        default_models = _PROVIDER_MODELS.get(resolved_provider, [])
        if (
            resolved_provider in _PROVIDER_LABELS
            and default_models
            and resolved_provider not in current_keys
        ):
            return (resolved_provider, default_models[0])

    # Aggregators list other providers' models — never auto-switch TO them
    # If the model belongs to the current provider's catalog, don't suggest switching
    if _model_in_provider_catalog(name_lower, current_keys):
        return None

    # --- Step 1: check static provider catalogs for a direct match ---
    for pid, models in _PROVIDER_MODELS.items():
        if pid in current_keys or pid in _AGGREGATOR_PROVIDERS:
            continue
        if any(name_lower == m.lower() for m in models):
            return (pid, name)

    return None


def detect_provider_for_model(
    model_name: str,
    current_provider: str,
) -> Optional[tuple[str, str]]:
    """Auto-detect the best provider for a model name.

    Returns ``(provider_id, model_name)`` — the model name may be remapped
    (e.g. bare ``deepseek-chat`` → ``deepseek/deepseek-chat`` for OpenRouter).
    Returns ``None`` when no confident match is found.

    Priority:
    0. Bare provider name → switch to that provider's default model
    1. Direct provider static catalog match
    2. OpenRouter catalog match
    """
    name = (model_name or "").strip()
    if not name:
        return None

    static_match = detect_static_provider_for_model(name, current_provider)
    if static_match:
        return static_match
    if _model_in_provider_catalog(name.lower(), _provider_keys(current_provider)):
        return None

    # --- Step 2: check OpenRouter catalog ---
    # First try exact match (handles provider/model format)
    or_slug = _find_openrouter_slug(name)
    if or_slug:
        if current_provider != "openrouter":
            return ("openrouter", or_slug)
        # Already on openrouter, just return the resolved slug
        if or_slug != name:
            return ("openrouter", or_slug)
        return None  # already on openrouter with matching name

    return None


def _find_openrouter_slug(model_name: str) -> Optional[str]:
    """Find the full OpenRouter model slug for a bare or partial model name.

    Handles:
    - Exact match: ``anthropic/claude-opus-4.6`` → as-is
    - Bare name: ``deepseek-chat`` → ``deepseek/deepseek-chat``
    - Bare name: ``claude-opus-4.6`` → ``anthropic/claude-opus-4.6``
    """
    name_lower = model_name.strip().lower()
    if not name_lower:
        return None

    # Exact match (already has provider/ prefix)
    for mid in model_ids():
        if name_lower == mid.lower():
            return mid

    # Try matching just the model part (after the /)
    for mid in model_ids():
        if "/" in mid:
            _, model_part = mid.split("/", 1)
            if name_lower == model_part.lower():
                return mid

    return None


def normalize_provider(provider: Optional[str]) -> str:
    """Normalize provider aliases to Hermes' canonical provider ids.

    Note: ``"auto"`` passes through unchanged — use
    ``hermes_cli.auth.resolve_provider()`` to resolve it to a concrete
    provider based on credentials and environment.
    """
    normalized = (provider or "openrouter").strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def provider_label(provider: Optional[str]) -> str:
    """Return a human-friendly label for a provider id or alias."""
    original = (provider or "openrouter").strip()
    normalized = original.lower()
    if normalized == "auto":
        return "Auto"
    normalized = normalize_provider(normalized)
    return _PROVIDER_LABELS.get(normalized, original or "OpenRouter")


# Models that support OpenAI Priority Processing (service_tier="priority").
# See https://openai.com/api-priority-processing/ for the canonical list.
# Only the bare model slug is stored (no vendor prefix).
_PRIORITY_PROCESSING_MODELS: frozenset[str] = frozenset({
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "o3",
    "o4-mini",
})

# Models that support Anthropic Fast Mode (speed="fast").
# See https://platform.claude.com/docs/en/build-with-claude/fast-mode
# Currently only Claude Opus 4.6.  Both hyphen and dot variants are stored
# to handle native Anthropic (claude-opus-4-6) and OpenRouter (claude-opus-4.6).
_ANTHROPIC_FAST_MODE_MODELS: frozenset[str] = frozenset({
    "claude-opus-4-6",
    "claude-opus-4.6",
})


def _strip_vendor_prefix(model_id: str) -> str:
    """Strip vendor/ prefix from a model ID (e.g. 'anthropic/claude-opus-4-6' -> 'claude-opus-4-6')."""
    raw = str(model_id or "").strip().lower()
    if "/" in raw:
        raw = raw.split("/", 1)[1]
    return raw


def model_supports_fast_mode(model_id: Optional[str]) -> bool:
    """Return whether Hermes should expose the /fast toggle for this model."""
    raw = _strip_vendor_prefix(str(model_id or ""))
    if raw in _PRIORITY_PROCESSING_MODELS:
        return True
    # Anthropic fast mode — strip date suffixes (e.g. claude-opus-4-6-20260401)
    # and OpenRouter variant tags (:fast, :beta) for matching.
    base = raw.split(":")[0]
    return base in _ANTHROPIC_FAST_MODE_MODELS


def _is_anthropic_fast_model(model_id: Optional[str]) -> bool:
    """Return True if the model supports Anthropic's fast mode (speed='fast')."""
    raw = _strip_vendor_prefix(str(model_id or ""))
    base = raw.split(":")[0]
    return base in _ANTHROPIC_FAST_MODE_MODELS


def resolve_fast_mode_overrides(model_id: Optional[str]) -> dict[str, Any] | None:
    """Return request_overrides for fast/priority mode, or None if unsupported.

    Returns provider-appropriate overrides:
    - OpenAI models: ``{"service_tier": "priority"}`` (Priority Processing)
    - Anthropic models: ``{"speed": "fast"}`` (Anthropic Fast Mode beta)

    The overrides are injected into the API request kwargs by
    ``_build_api_kwargs`` in run_agent.py — each API path handles its own
    keys (service_tier for OpenAI/Codex, speed for Anthropic Messages).
    """
    if not model_supports_fast_mode(model_id):
        return None
    if _is_anthropic_fast_model(model_id):
        return {"speed": "fast"}
    return {"service_tier": "priority"}


def _resolve_copilot_catalog_api_key() -> str:
    """Best-effort GitHub token for fetching the Copilot model catalog."""
    try:
        from hermes_cli.auth import resolve_api_key_provider_credentials

        creds = resolve_api_key_provider_credentials("copilot")
        return str(creds.get("api_key") or "").strip()
    except Exception:
        return ""


# Providers where models.dev is treated as authoritative: curated static
# lists are kept only as an offline fallback and to capture custom additions
# the registry doesn't publish yet. Adding a provider here causes its
# curated list to be merged with fresh models.dev entries (fresh first, any
# curated-only names appended) for both the CLI and the gateway /model picker.
#
# DELIBERATELY EXCLUDED:
#   - "openrouter": curated list is already a hand-picked agentic subset of
#     OpenRouter's 400+ catalog. Blindly merging would dump everything.
#   - "nous": curated list and Portal /models endpoint are the source of
#     truth for the subscription tier.
# Also excluded: providers that already have dedicated live-endpoint
# branches below (copilot, anthropic, ai-gateway, ollama-cloud, custom,
# stepfun, openai-codex) — those paths handle freshness themselves.
_MODELS_DEV_PREFERRED: frozenset[str] = frozenset({
    "opencode-go",
    "opencode-zen",
    "deepseek",
    "kilocode",
    "fireworks",
    "mistral",
    "togetherai",
    "cohere",
    "perplexity",
    "groq",
    "nvidia",
    "huggingface",
    "zai",
    "gemini",
    "google",
})


def _merge_with_models_dev(provider: str, curated: list[str]) -> list[str]:
    """Merge curated list with fresh models.dev entries for a preferred provider.

    Returns models.dev entries first (in models.dev order), then any
    curated-only entries appended. Preserves case for curated fallbacks
    (e.g. ``MiniMax-M2.7``) while trusting models.dev for newer variants.

    If models.dev is unreachable or returns nothing, the curated list is
    returned unchanged — this is the offline/CI fallback path.
    """
    try:
        from agent.models_dev import list_agentic_models
        mdev = list_agentic_models(provider)
    except Exception:
        mdev = []

    if not mdev:
        return list(curated)

    # Case-insensitive dedup while preserving order and curated casing.
    seen_lower: set[str] = set()
    merged: list[str] = []
    for mid in mdev:
        key = str(mid).lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        merged.append(mid)
    for mid in curated:
        key = str(mid).lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        merged.append(mid)
    return merged


def provider_model_ids(provider: Optional[str], *, force_refresh: bool = False) -> list[str]:
    """Return the best known model catalog for a provider.

    Tries live API endpoints for providers that support them (Codex, Nous),
    falling back to static lists. For providers in ``_MODELS_DEV_PREFERRED``
    (opencode-go/zen, xiaomi, deepseek, smaller inference providers, etc.),
    models.dev entries are merged on top of curated so new models released
    on the platform appear in ``/model`` without a Hermes release.
    """
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return model_ids(force_refresh=force_refresh)
    if normalized == "openai-codex":
        from hermes_cli.codex_models import get_codex_model_ids

        # Pass the live OAuth access token so the picker matches whatever
        # ChatGPT lists for this account right now (new models appear without
        # a Hermes release). Falls back to the hardcoded catalog if no token
        # or the endpoint is unreachable.
        access_token = None
        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials(refresh_if_expiring=True)
            access_token = creds.get("api_key")
        except Exception:
            access_token = None
        return get_codex_model_ids(access_token=access_token)
    if normalized in {"copilot", "copilot-acp"}:
        try:
            live = _fetch_github_models(_resolve_copilot_catalog_api_key())
            if live:
                return live
        except Exception:
            pass
        if normalized == "copilot-acp":
            return list(_PROVIDER_MODELS.get("copilot", []))
    if normalized == "nous":
        # Try live Nous Portal /models endpoint
        try:
            from hermes_cli.auth import fetch_nous_models, resolve_nous_runtime_credentials
            creds = resolve_nous_runtime_credentials()
            if creds:
                live = fetch_nous_models(api_key=creds.get("api_key", ""), inference_base_url=creds.get("base_url", ""))
                if live:
                    return live
        except Exception:
            pass
    if normalized == "stepfun":
        try:
            from hermes_cli.auth import resolve_api_key_provider_credentials

            creds = resolve_api_key_provider_credentials("stepfun")
            api_key = str(creds.get("api_key") or "").strip()
            base_url = str(creds.get("base_url") or "").strip()
            if api_key and base_url:
                live = fetch_api_models(api_key, base_url)
                if live:
                    return live
        except Exception:
            pass
    if normalized == "anthropic":
        live = _fetch_anthropic_models()
        if live:
            return live
    if normalized == "ai-gateway":
        live = _fetch_ai_gateway_models()
        if live:
            return live
    if normalized == "ollama-cloud":
        live = fetch_ollama_cloud_models(force_refresh=force_refresh)
        if live:
            return live
    if normalized == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            base_raw = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
            base = base_raw or "https://api.openai.com/v1"
            try:
                live = fetch_api_models(api_key, base)
                if live:
                    return live
            except Exception:
                pass
    if normalized == "custom":
        base_url = _get_custom_base_url()
        if base_url:
            # Try common API key env vars for custom endpoints
            api_key = (
                os.getenv("CUSTOM_API_KEY", "")
                or os.getenv("OPENAI_API_KEY", "")
                or os.getenv("OPENROUTER_API_KEY", "")
            )
            live = fetch_api_models(api_key, base_url)
            if live:
                return live
    curated_static = list(_PROVIDER_MODELS.get(normalized, []))
    if normalized in _MODELS_DEV_PREFERRED:
        return _merge_with_models_dev(normalized, curated_static)
    return curated_static


def _fetch_anthropic_models(timeout: float = 5.0) -> Optional[list[str]]:
    """Fetch available models from the Anthropic /v1/models endpoint.

    Uses resolve_anthropic_token() to find credentials (env vars or
    Claude Code auto-discovery).  Returns sorted model IDs or None.
    """
    try:
        from agent.anthropic_adapter import resolve_anthropic_token, _is_oauth_token
    except ImportError:
        return None

    token = resolve_anthropic_token()
    if not token:
        return None

    headers: dict[str, str] = {"anthropic-version": "2023-06-01"}
    if _is_oauth_token(token):
        headers["Authorization"] = f"Bearer {token}"
        from agent.anthropic_adapter import _COMMON_BETAS, _OAUTH_ONLY_BETAS
        headers["anthropic-beta"] = ",".join(_COMMON_BETAS + _OAUTH_ONLY_BETAS)
    else:
        headers["x-api-key"] = token

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/models",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            models = [m["id"] for m in data.get("data", []) if m.get("id")]
            # Sort: latest/largest first (opus > sonnet > haiku, higher version first)
            return sorted(models, key=lambda m: (
                "opus" not in m,      # opus first
                "sonnet" not in m,    # then sonnet
                "haiku" not in m,     # then haiku
                m,                    # alphabetical within tier
            ))
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("Failed to fetch Anthropic models: %s", e)
        return None


def _payload_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        data = payload.get("data", [])
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    return []


def copilot_default_headers() -> dict[str, str]:
    """Standard headers for Copilot API requests.

    Includes Openai-Intent and x-initiator headers that opencode and the
    Copilot CLI send on every request.
    """
    try:
        from hermes_cli.copilot_auth import copilot_request_headers
        return copilot_request_headers(is_agent_turn=True)
    except ImportError:
        return {
            "Editor-Version": COPILOT_EDITOR_VERSION,
            "User-Agent": "HermesAgent/1.0",
            "Openai-Intent": "conversation-edits",
            "x-initiator": "agent",
        }


def _copilot_catalog_item_is_text_model(item: dict[str, Any]) -> bool:
    model_id = str(item.get("id") or "").strip()
    if not model_id:
        return False

    if item.get("model_picker_enabled") is False:
        return False

    capabilities = item.get("capabilities")
    if isinstance(capabilities, dict):
        model_type = str(capabilities.get("type") or "").strip().lower()
        if model_type and model_type != "chat":
            return False

    supported_endpoints = item.get("supported_endpoints")
    if isinstance(supported_endpoints, list):
        normalized_endpoints = {
            str(endpoint).strip()
            for endpoint in supported_endpoints
            if str(endpoint).strip()
        }
        if normalized_endpoints and not normalized_endpoints.intersection(
            {"/chat/completions", "/responses", "/v1/messages"}
        ):
            return False

    return True


def fetch_github_model_catalog(
    api_key: Optional[str] = None, timeout: float = 5.0
) -> Optional[list[dict[str, Any]]]:
    """Fetch the live GitHub Copilot model catalog for this account."""
    attempts: list[dict[str, str]] = []
    if api_key:
        attempts.append({
            **copilot_default_headers(),
            "Authorization": f"Bearer {api_key}",
        })
    attempts.append(copilot_default_headers())

    for headers in attempts:
        req = urllib.request.Request(COPILOT_MODELS_URL, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                items = _payload_items(data)
                models: list[dict[str, Any]] = []
                seen_ids: set[str] = set()
                for item in items:
                    if not _copilot_catalog_item_is_text_model(item):
                        continue
                    model_id = str(item.get("id") or "").strip()
                    if not model_id or model_id in seen_ids:
                        continue
                    seen_ids.add(model_id)
                    models.append(item)
                if models:
                    return models
        except Exception:
            continue
    return None


# ─── Copilot catalog context-window helpers ─────────────────────────────────

# Module-level cache: {model_id: max_prompt_tokens}
_copilot_context_cache: dict[str, int] = {}
_copilot_context_cache_time: float = 0.0
_COPILOT_CONTEXT_CACHE_TTL = 3600  # 1 hour


def get_copilot_model_context(model_id: str, api_key: Optional[str] = None) -> Optional[int]:
    """Look up max_prompt_tokens for a Copilot model from the live /models API.

    Results are cached in-process for 1 hour to avoid repeated API calls.
    Returns the token limit or None if not found.
    """
    global _copilot_context_cache, _copilot_context_cache_time

    # Serve from cache if fresh
    if _copilot_context_cache and (time.time() - _copilot_context_cache_time < _COPILOT_CONTEXT_CACHE_TTL):
        if model_id in _copilot_context_cache:
            return _copilot_context_cache[model_id]
        # Cache is fresh but model not in it — don't re-fetch
        return None

    # Fetch and populate cache
    catalog = fetch_github_model_catalog(api_key=api_key)
    if not catalog:
        return None

    cache: dict[str, int] = {}
    for item in catalog:
        mid = str(item.get("id") or "").strip()
        if not mid:
            continue
        caps = item.get("capabilities") or {}
        limits = caps.get("limits") or {}
        max_prompt = limits.get("max_prompt_tokens")
        if isinstance(max_prompt, int) and max_prompt > 0:
            cache[mid] = max_prompt

    _copilot_context_cache = cache
    _copilot_context_cache_time = time.time()

    return cache.get(model_id)


def _is_github_models_base_url(base_url: Optional[str]) -> bool:
    normalized = (base_url or "").strip().rstrip("/").lower()
    return (
        normalized.startswith(COPILOT_BASE_URL)
        or normalized.startswith("https://models.github.ai/inference")
    )


def _fetch_github_models(api_key: Optional[str] = None, timeout: float = 5.0) -> Optional[list[str]]:
    catalog = fetch_github_model_catalog(api_key=api_key, timeout=timeout)
    if not catalog:
        return None
    return [item.get("id", "") for item in catalog if item.get("id")]


_COPILOT_MODEL_ALIASES = {
    "openai/gpt-5": "gpt-5-mini",
    "openai/gpt-5-chat": "gpt-5-mini",
    "openai/gpt-5-mini": "gpt-5-mini",
    "openai/gpt-5-nano": "gpt-5-mini",
    "openai/gpt-4.1": "gpt-4.1",
    "openai/gpt-4.1-mini": "gpt-4.1",
    "openai/gpt-4.1-nano": "gpt-4.1",
    "openai/gpt-4o": "gpt-4o",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "openai/o1": "gpt-5.2",
    "openai/o1-mini": "gpt-5-mini",
    "openai/o1-preview": "gpt-5.2",
    "openai/o3": "gpt-5.3-codex",
    "openai/o3-mini": "gpt-5-mini",
    "openai/o4-mini": "gpt-5-mini",
    "anthropic/claude-opus-4.6": "claude-opus-4.6",
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4.6",
    "anthropic/claude-sonnet-4": "claude-sonnet-4",
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5": "claude-haiku-4.5",
    # Dash-notation fallbacks: Hermes' default Claude IDs elsewhere use
    # hyphens (anthropic native format), but Copilot's API only accepts
    # dot-notation.  Accept both so users who configure copilot + a
    # default hyphenated Claude model don't hit HTTP 400
    # "model_not_supported".  See issue #6879.
    "claude-opus-4-6": "claude-opus-4.6",
    "claude-sonnet-4-6": "claude-sonnet-4.6",
    "claude-sonnet-4-0": "claude-sonnet-4",
    "claude-sonnet-4-5": "claude-sonnet-4.5",
    "claude-haiku-4-5": "claude-haiku-4.5",
    "anthropic/claude-opus-4-6": "claude-opus-4.6",
    "anthropic/claude-sonnet-4-6": "claude-sonnet-4.6",
    "anthropic/claude-sonnet-4-0": "claude-sonnet-4",
    "anthropic/claude-sonnet-4-5": "claude-sonnet-4.5",
    "anthropic/claude-haiku-4-5": "claude-haiku-4.5",
}


def _copilot_catalog_ids(
    catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> set[str]:
    if catalog is None and api_key:
        catalog = fetch_github_model_catalog(api_key=api_key)
    if not catalog:
        return set()
    return {
        str(item.get("id") or "").strip()
        for item in catalog
        if str(item.get("id") or "").strip()
    }


def normalize_copilot_model_id(
    model_id: Optional[str],
    *,
    catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> str:
    raw = str(model_id or "").strip()
    if not raw:
        return ""

    catalog_ids = _copilot_catalog_ids(catalog=catalog, api_key=api_key)
    alias = _COPILOT_MODEL_ALIASES.get(raw)
    if alias:
        return alias

    candidates = [raw]
    if "/" in raw:
        candidates.append(raw.split("/", 1)[1].strip())

    if raw.endswith("-mini"):
        candidates.append(raw[:-5])
    if raw.endswith("-nano"):
        candidates.append(raw[:-5])
    if raw.endswith("-chat"):
        candidates.append(raw[:-5])

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate in _COPILOT_MODEL_ALIASES:
            return _COPILOT_MODEL_ALIASES[candidate]
        if candidate in catalog_ids:
            return candidate

    if "/" in raw:
        return raw.split("/", 1)[1].strip()
    return raw


def _github_reasoning_efforts_for_model_id(model_id: str) -> list[str]:
    raw = (model_id or "").strip().lower()
    if raw.startswith(("openai/o1", "openai/o3", "openai/o4", "o1", "o3", "o4")):
        return list(COPILOT_REASONING_EFFORTS_O_SERIES)
    normalized = normalize_copilot_model_id(model_id).lower()
    if normalized.startswith("gpt-5"):
        return list(COPILOT_REASONING_EFFORTS_GPT5)
    return []


def _should_use_copilot_responses_api(model_id: str) -> bool:
    """Decide whether a Copilot model should use the Responses API.

    Replicates opencode's ``shouldUseCopilotResponsesApi`` logic:
    GPT-5+ models use Responses API, except ``gpt-5-mini`` which uses
    Chat Completions.  All non-GPT models (Claude, Gemini, etc.) use
    Chat Completions.
    """
    import re

    match = re.match(r"^gpt-(\d+)", model_id)
    if not match:
        return False
    major = int(match.group(1))
    return major >= 5 and not model_id.startswith("gpt-5-mini")


def copilot_model_api_mode(
    model_id: Optional[str],
    *,
    catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> str:
    """Determine the API mode for a Copilot model.

    Uses the model ID pattern (matching opencode's approach) as the
    primary signal.  Falls back to the catalog's ``supported_endpoints``
    only for models not covered by the pattern check.
    """
    # Fetch the catalog once so normalize + endpoint check share it
    # (avoids two redundant network calls for non-GPT-5 models).
    if catalog is None and api_key:
        catalog = fetch_github_model_catalog(api_key=api_key)

    normalized = normalize_copilot_model_id(model_id, catalog=catalog, api_key=api_key)
    if not normalized:
        return "chat_completions"

    # Primary: model ID pattern (matches opencode's shouldUseCopilotResponsesApi)
    if _should_use_copilot_responses_api(normalized):
        return "codex_responses"

    # Secondary: check catalog for non-GPT-5 models (Claude via /v1/messages, etc.)
    if catalog:
        catalog_entry = next((item for item in catalog if item.get("id") == normalized), None)
        if isinstance(catalog_entry, dict):
            supported_endpoints = {
                str(endpoint).strip()
                for endpoint in (catalog_entry.get("supported_endpoints") or [])
                if str(endpoint).strip()
            }
            # For non-GPT-5 models, check if they only support messages API
            if "/v1/messages" in supported_endpoints and "/chat/completions" not in supported_endpoints:
                return "anthropic_messages"

    return "chat_completions"


def normalize_opencode_model_id(provider_id: Optional[str], model_id: Optional[str]) -> str:
    """Normalize OpenCode config IDs to the bare model slug used in API requests."""
    provider = normalize_provider(provider_id)
    current = str(model_id or "").strip()
    if not current or provider not in {"opencode-zen", "opencode-go"}:
        return current

    prefix = f"{provider}/"
    if current.lower().startswith(prefix):
        return current[len(prefix):]
    return current


def opencode_model_api_mode(provider_id: Optional[str], model_id: Optional[str]) -> str:
    """Determine the API mode for an OpenCode Zen / Go model.

    OpenCode routes different models behind different API surfaces:

    - GPT-5 / Codex models on Zen use ``/v1/responses``
    - Claude models on Zen use ``/v1/messages``
    - MiniMax models on Go use ``/v1/messages``
    - GLM / Kimi on Go use ``/v1/chat/completions``
    - Other Zen models (Gemini, GLM, Kimi, MiniMax, Qwen, etc.) use
      ``/v1/chat/completions``

    This follows the published OpenCode docs for Zen and Go endpoints.
    """
    provider = normalize_provider(provider_id)
    normalized = normalize_opencode_model_id(provider_id, model_id).lower()
    if not normalized:
        return "chat_completions"

    if provider == "opencode-go":
        if normalized.startswith("minimax-"):
            return "anthropic_messages"
        return "chat_completions"

    if provider == "opencode-zen":
        if normalized.startswith("claude-"):
            return "anthropic_messages"
        if normalized.startswith("gpt-"):
            return "codex_responses"
        return "chat_completions"

    return "chat_completions"


def github_model_reasoning_efforts(
    model_id: Optional[str],
    *,
    catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> list[str]:
    """Return supported reasoning-effort levels for a Copilot-visible model."""
    normalized = normalize_copilot_model_id(model_id, catalog=catalog, api_key=api_key)
    if not normalized:
        return []

    catalog_entry = None
    if catalog is not None:
        catalog_entry = next((item for item in catalog if item.get("id") == normalized), None)
    elif api_key:
        fetched_catalog = fetch_github_model_catalog(api_key=api_key)
        if fetched_catalog:
            catalog_entry = next((item for item in fetched_catalog if item.get("id") == normalized), None)

    if catalog_entry is not None:
        capabilities = catalog_entry.get("capabilities")
        if isinstance(capabilities, dict):
            supports = capabilities.get("supports")
            if isinstance(supports, dict):
                efforts = supports.get("reasoning_effort")
                if isinstance(efforts, list):
                    normalized_efforts = [
                        str(effort).strip().lower()
                        for effort in efforts
                        if str(effort).strip()
                    ]
                    return list(dict.fromkeys(normalized_efforts))
            return []
        legacy_capabilities = {
            str(capability).strip().lower()
            for capability in catalog_entry.get("capabilities", [])
            if str(capability).strip()
        }
        if "reasoning" not in legacy_capabilities:
            return []

    return _github_reasoning_efforts_for_model_id(str(model_id or normalized))


def probe_api_models(
    api_key: Optional[str],
    base_url: Optional[str],
    timeout: float = 5.0,
    api_mode: Optional[str] = None,
) -> dict[str, Any]:
    """Probe a ``/models`` endpoint with light URL heuristics.

    For ``anthropic_messages`` mode, uses ``x-api-key`` and
    ``anthropic-version`` headers (Anthropic's native auth) instead of
    ``Authorization: Bearer``.  The response shape (``data[].id``) is
    identical, so the same parser works for both.
    """
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return {
            "models": None,
            "probed_url": None,
            "resolved_base_url": "",
            "suggested_base_url": None,
            "used_fallback": False,
        }

    if _is_github_models_base_url(normalized):
        models = _fetch_github_models(api_key=api_key, timeout=timeout)
        return {
            "models": models,
            "probed_url": COPILOT_MODELS_URL,
            "resolved_base_url": COPILOT_BASE_URL,
            "suggested_base_url": None,
            "used_fallback": False,
        }

    if normalized.endswith("/v1"):
        alternate_base = normalized[:-3].rstrip("/")
    else:
        alternate_base = normalized + "/v1"

    candidates: list[tuple[str, bool]] = [(normalized, False)]
    if alternate_base and alternate_base != normalized:
        candidates.append((alternate_base, True))

    tried: list[str] = []
    headers: dict[str, str] = {"User-Agent": _HERMES_USER_AGENT}
    if api_key and api_mode == "anthropic_messages":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if normalized.startswith(COPILOT_BASE_URL):
        headers.update(copilot_default_headers())

    for candidate_base, is_fallback in candidates:
        url = candidate_base.rstrip("/") + "/models"
        tried.append(url)
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                return {
                    "models": [m.get("id", "") for m in data.get("data", [])],
                    "probed_url": url,
                    "resolved_base_url": candidate_base.rstrip("/"),
                    "suggested_base_url": alternate_base if alternate_base != candidate_base else normalized,
                    "used_fallback": is_fallback,
                }
        except Exception:
            continue

    return {
        "models": None,
        "probed_url": tried[0] if tried else normalized.rstrip("/") + "/models",
        "resolved_base_url": normalized,
        "suggested_base_url": alternate_base if alternate_base != normalized else None,
        "used_fallback": False,
    }


def _fetch_ai_gateway_models(timeout: float = 5.0) -> Optional[list[str]]:
    """Fetch available language models with tool-use from AI Gateway."""
    api_key = os.getenv("AI_GATEWAY_API_KEY", "").strip()
    if not api_key:
        return None
    base_url = os.getenv("AI_GATEWAY_BASE_URL", "").strip()
    if not base_url:
        from hermes_constants import AI_GATEWAY_BASE_URL
        base_url = AI_GATEWAY_BASE_URL

    url = base_url.rstrip("/") + "/models"
    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": _HERMES_USER_AGENT,
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return [
                m["id"]
                for m in data.get("data", [])
                if m.get("id")
                and m.get("type") == "language"
                and "tool-use" in (m.get("tags") or [])
            ]
    except Exception:
        return None


def fetch_api_models(
    api_key: Optional[str],
    base_url: Optional[str],
    timeout: float = 5.0,
    api_mode: Optional[str] = None,
) -> Optional[list[str]]:
    """Fetch the list of available model IDs from the provider's ``/models`` endpoint.

    Returns a list of model ID strings, or ``None`` if the endpoint could not
    be reached (network error, timeout, auth failure, etc.).
    """
    return probe_api_models(api_key, base_url, timeout=timeout, api_mode=api_mode).get("models")


# ---------------------------------------------------------------------------
# Ollama Cloud — merged model discovery with disk cache
# ---------------------------------------------------------------------------



_OLLAMA_CLOUD_CACHE_TTL = 3600  # 1 hour


def _ollama_cloud_cache_path() -> Path:
    """Return the path for the Ollama Cloud model cache."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "ollama_cloud_models_cache.json"


def _load_ollama_cloud_cache(*, ignore_ttl: bool = False) -> Optional[dict]:
    """Load cached Ollama Cloud models from disk.

    Args:
        ignore_ttl: If True, return data even if the TTL has expired (stale fallback).
    """
    try:
        cache_path = _ollama_cloud_cache_path()
        if not cache_path.exists():
            return None
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        models = data.get("models")
        if not (isinstance(models, list) and models):
            return None
        if not ignore_ttl:
            cached_at = data.get("cached_at", 0)
            if (time.time() - cached_at) > _OLLAMA_CLOUD_CACHE_TTL:
                return None  # stale
        return data
    except Exception:
        pass
    return None


def _save_ollama_cloud_cache(models: list[str]) -> None:
    """Persist the merged Ollama Cloud model list to disk."""
    try:
        from utils import atomic_json_write
        cache_path = _ollama_cloud_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(cache_path, {"models": models, "cached_at": time.time()}, indent=None)
    except Exception:
        pass


def fetch_ollama_cloud_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    *,
    force_refresh: bool = False,
) -> list[str]:
    """Fetch Ollama Cloud models by merging live API + models.dev, with disk cache.

    Resolution order:
      1. Disk cache (if fresh, < 1 hour, and not force_refresh)
      2. Live ``/v1/models`` endpoint (primary — freshest source)
      3. models.dev registry (secondary — fills gaps for unlisted models)
      4. Merge: live models first, then models.dev additions (deduped)

    Returns a list of model IDs (never None — empty list on total failure).
    """
    # 1. Check disk cache
    if not force_refresh:
        cached = _load_ollama_cloud_cache()
        if cached is not None:
            return cached["models"]

    # 2. Live API probe
    if not api_key:
        api_key = os.getenv("OLLAMA_API_KEY", "")
    if not base_url:
        base_url = os.getenv("OLLAMA_BASE_URL", "") or "https://ollama.com/v1"

    live_models: list[str] = []
    if api_key:
        result = fetch_api_models(api_key, base_url, timeout=8.0)
        if result:
            live_models = result

    # 3. models.dev registry
    mdev_models: list[str] = []
    try:
        from agent.models_dev import list_agentic_models
        mdev_models = list_agentic_models("ollama-cloud")
    except Exception:
        pass

    # 4. Merge: live first, then models.dev additions (deduped, order-preserving)
    if live_models or mdev_models:
        seen: set[str] = set()
        merged: list[str] = []
        for m in live_models:
            if m and m not in seen:
                seen.add(m)
                merged.append(m)
        for m in mdev_models:
            if m and m not in seen:
                seen.add(m)
                merged.append(m)
        if merged:
            _save_ollama_cloud_cache(merged)
            return merged

    # Total failure — return stale cache if available (ignore TTL)
    stale = _load_ollama_cloud_cache(ignore_ttl=True)
    if stale is not None:
        return stale["models"]

    return []


def validate_requested_model(
    model_name: str,
    provider: Optional[str],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
) -> dict[str, Any]:
    """
    Validate a ``/model`` value for the active provider.

    Performs format checks first, then probes the live API to confirm
    the model actually exists.

    Returns a dict with:
      - accepted: whether the CLI should switch to the requested model now
      - persist: whether it is safe to save to config
      - recognized: whether it matched a known provider catalog
      - message: optional warning / guidance for the user
    """
    requested = (model_name or "").strip()
    normalized = normalize_provider(provider)
    if normalized == "openrouter" and base_url and "openrouter.ai" not in base_url:
        normalized = "custom"
    requested_for_lookup = requested
    if normalized == "copilot":
        requested_for_lookup = normalize_copilot_model_id(
            requested,
            api_key=api_key,
        ) or requested

    if not requested:
        return {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": "Model name cannot be empty.",
        }

    if any(ch.isspace() for ch in requested):
        return {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": "Model names cannot contain spaces.",
        }

    if normalized == "custom":
        # Try probing with correct auth for the api_mode.
        if api_mode == "anthropic_messages":
            probe = probe_api_models(api_key, base_url, api_mode=api_mode)
        else:
            probe = probe_api_models(api_key, base_url)
        api_models = probe.get("models")
        if api_models is not None:
            if requested_for_lookup in set(api_models):
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }

            # Auto-correct if the top match is very similar (e.g. typo)
            auto = get_close_matches(requested_for_lookup, api_models, n=1, cutoff=0.9)
            if auto:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }

            suggestions = get_close_matches(requested, api_models, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)

            message = (
                f"Note: `{requested}` was not found in this custom endpoint's model listing "
                f"({probe.get('probed_url')}). It may still work if the server supports hidden or aliased models."
                f"{suggestion_text}"
            )
            if probe.get("used_fallback"):
                message += (
                    f"\n  Endpoint verification succeeded after trying `{probe.get('resolved_base_url')}`. "
                    f"Consider saving that as your base URL."
                )

            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": message,
            }

        message = (
            f"Note: could not reach this custom endpoint's model listing at `{probe.get('probed_url')}`. "
            f"Hermes will still save `{requested}`, but the endpoint should expose `/models` for verification."
        )
        if api_mode == "anthropic_messages":
            message += (
                "\n  Many Anthropic-compatible proxies do not implement the Models API "
                "(GET /v1/models).  The model name has been accepted without verification."
            )
        if probe.get("suggested_base_url"):
            message += f"\n  If this server expects `/v1`, try base URL: `{probe.get('suggested_base_url')}`"

        return {
            "accepted": api_mode == "anthropic_messages",
            "persist": True,
            "recognized": False,
            "message": message,
        }

    # OpenAI Codex has its own catalog path; /v1/models probing is not the right validation path.
    if normalized == "openai-codex":
        try:
            codex_models = provider_model_ids("openai-codex")
        except Exception:
            codex_models = []
        if codex_models:
            if requested_for_lookup in set(codex_models):
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }
            # Auto-correct if the top match is very similar (e.g. typo)
            auto = get_close_matches(requested_for_lookup, codex_models, n=1, cutoff=0.9)
            if auto:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }
            suggestions = get_close_matches(requested_for_lookup, codex_models, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)
            return {
                "accepted": False,
                "persist": False,
                "recognized": False,
                "message": (
                    f"Model `{requested}` was not found in the OpenAI Codex model listing."
                    f"{suggestion_text}"
                ),
            }

    # MiniMax providers don't expose a /models endpoint — validate against
    # the static catalog instead, similar to openai-codex.
    if normalized in ("minimax", "minimax-cn"):
        try:
            catalog_models = provider_model_ids(normalized)
        except Exception:
            catalog_models = []
        if catalog_models:
            # Case-insensitive lookup (catalog uses mixed case like MiniMax-M2.7)
            catalog_lower = {m.lower(): m for m in catalog_models}
            if requested_for_lookup.lower() in catalog_lower:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }
            # Auto-correct close matches (case-insensitive)
            catalog_lower_list = list(catalog_lower.keys())
            auto = get_close_matches(requested_for_lookup.lower(), catalog_lower_list, n=1, cutoff=0.9)
            if auto:
                corrected = catalog_lower[auto[0]]
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": corrected,
                    "message": f"Auto-corrected `{requested}` → `{corrected}`",
                }
            suggestions = get_close_matches(requested_for_lookup.lower(), catalog_lower_list, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{catalog_lower[s]}`" for s in suggestions)
            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": (
                    f"Note: `{requested}` was not found in the MiniMax catalog."
                    f"{suggestion_text}"
                    "\n  MiniMax does not expose a /models endpoint, so Hermes cannot verify the model name."
                    "\n  The model may still work if it exists on the server."
                ),
            }

    # Native Anthropic provider: /v1/models requires x-api-key (or Bearer for
    # OAuth) plus anthropic-version headers.  The generic OpenAI-style probe
    # below uses plain Bearer auth and 401s against Anthropic, so dispatch to
    # the native fetcher which handles both API keys and Claude-Code OAuth
    # tokens.  (The api_mode=="anthropic_messages" branch below handles the
    # Messages-API transport case separately.)
    if normalized == "anthropic":
        anthropic_models = _fetch_anthropic_models()
        if anthropic_models is not None:
            if requested_for_lookup in set(anthropic_models):
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }
            auto = get_close_matches(requested_for_lookup, anthropic_models, n=1, cutoff=0.9)
            if auto:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }
            suggestions = get_close_matches(requested, anthropic_models, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)
            # Accept anyway — Anthropic sometimes gates newer/preview models
            # (e.g. snapshot IDs, early-access releases) behind accounts
            # even though they aren't listed on /v1/models.
            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": (
                    f"Note: `{requested}` was not found in Anthropic's /v1/models listing. "
                    f"It may still work if you have early-access or snapshot IDs."
                    f"{suggestion_text}"
                ),
            }
        # _fetch_anthropic_models returned None — no token resolvable or
        # network failure.  Fall through to the generic warning below.

    # Anthropic Messages API: many proxies don't implement /v1/models.
    # Try probing with correct auth; if it fails, accept with a warning.
    if api_mode == "anthropic_messages":
        api_models = fetch_api_models(api_key, base_url, api_mode=api_mode)
        if api_models is not None:
            if requested_for_lookup in set(api_models):
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }
            auto = get_close_matches(requested_for_lookup, api_models, n=1, cutoff=0.9)
            if auto:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }
        # Probe failed or model not found — accept anyway (proxy likely
        # doesn't implement the Anthropic Models API).
        return {
            "accepted": True,
            "persist": True,
            "recognized": False,
            "message": (
                f"Note: could not verify `{requested}` against this endpoint's "
                f"model listing.  Many Anthropic-compatible proxies do not "
                f"implement GET /v1/models.  The model name has been accepted "
                f"without verification."
            ),
        }

    # Probe the live API to check if the model actually exists
    api_models = fetch_api_models(api_key, base_url)

    if api_models is not None:
        # Gemini's OpenAI-compat /v1beta/openai/models endpoint returns IDs
        # prefixed with "models/" (e.g. "models/gemini-2.5-flash") — native
        # Gemini-API convention.  Our curated list and user input both use
        # the bare ID, so a direct set-membership check drops every known
        # Gemini model.  Strip the prefix before comparison.  See #12532.
        if normalized == "gemini":
            api_models = [
                m[len("models/"):] if isinstance(m, str) and m.startswith("models/") else m
                for m in api_models
            ]
        if requested_for_lookup in set(api_models):
            # API confirmed the model exists
            return {
                "accepted": True,
                "persist": True,
                "recognized": True,
                "message": None,
            }
        else:
            # API responded but model is not listed.  Accept anyway —
            # the user may have access to models not shown in the public
            # listing (e.g. Z.AI Pro/Max plans can use glm-5 on coding
            # endpoints even though it's not in /models).  Warn but allow.

            # Auto-correct if the top match is very similar (e.g. typo)
            auto = get_close_matches(requested_for_lookup, api_models, n=1, cutoff=0.9)
            if auto:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }

            suggestions = get_close_matches(requested, api_models, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)

        return {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": (
                f"Model `{requested}` was not found in this provider's model listing."
                f"{suggestion_text}"
            ),
        }

    # api_models is None — couldn't reach API.  Accept and persist,
    # but warn so typos don't silently break things.

    # Bedrock: use our own discovery instead of HTTP /models endpoint.
    # Bedrock's bedrock-runtime URL doesn't support /models — it uses the
    # AWS SDK control plane (ListFoundationModels + ListInferenceProfiles).
    if normalized == "bedrock":
        try:
            from agent.bedrock_adapter import discover_bedrock_models, resolve_bedrock_region
            region = resolve_bedrock_region()
            discovered = discover_bedrock_models(region)
            discovered_ids = {m["id"] for m in discovered}
            if requested in discovered_ids:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }
            # Not in discovered list — still accept (user may have custom
            # inference profiles or cross-account access), but warn.
            suggestions = get_close_matches(requested, list(discovered_ids), n=3, cutoff=0.4)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)
            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": (
                    f"Note: `{requested}` was not found in Bedrock model discovery for {region}. "
                    f"It may still work with custom inference profiles or cross-account access."
                    f"{suggestion_text}"
                ),
            }
        except Exception:
            pass  # Fall through to generic warning

    # Static-catalog fallback: when the /models probe was unreachable,
    # validate against the curated list from provider_model_ids() — same
    # pattern as the openai-codex and minimax branches above.  This fixes
    # /model switches in the gateway for providers like opencode-go and
    # opencode-zen whose /models endpoint returns 404 against the HTML
    # marketing site.  Without this block, validate_requested_model would
    # reject every model on such providers, switch_model() would return
    # success=False, and the gateway would never write to
    # _session_model_overrides.
    provider_label = _PROVIDER_LABELS.get(normalized, normalized)
    try:
        catalog_models = provider_model_ids(normalized)
    except Exception:
        catalog_models = []

    if catalog_models:
        catalog_lower = {m.lower(): m for m in catalog_models}
        if requested_for_lookup.lower() in catalog_lower:
            return {
                "accepted": True,
                "persist": True,
                "recognized": True,
                "message": None,
            }
        catalog_lower_list = list(catalog_lower.keys())
        auto = get_close_matches(
            requested_for_lookup.lower(), catalog_lower_list, n=1, cutoff=0.9
        )
        if auto:
            corrected = catalog_lower[auto[0]]
            return {
                "accepted": True,
                "persist": True,
                "recognized": True,
                "corrected_model": corrected,
                "message": f"Auto-corrected `{requested}` → `{corrected}`",
            }
        suggestions = get_close_matches(
            requested_for_lookup.lower(), catalog_lower_list, n=3, cutoff=0.5
        )
        suggestion_text = ""
        if suggestions:
            suggestion_text = "\n  Similar models: " + ", ".join(
                f"`{catalog_lower[s]}`" for s in suggestions
            )
        return {
            "accepted": True,
            "persist": True,
            "recognized": False,
            "message": (
                f"Note: `{requested}` was not found in the {provider_label} curated catalog "
                f"and the /models endpoint was unreachable.{suggestion_text}"
                f"\n  The model may still work if it exists on the provider."
            ),
        }

    # No catalog available — accept with a warning, matching the comment's
    # stated intent ("Accept and persist, but warn").
    return {
        "accepted": True,
        "persist": True,
        "recognized": False,
        "message": (
            f"Note: could not reach the {provider_label} API to validate `{requested}`. "
            f"If the service isn't down, this model may not be valid."
        ),
    }
