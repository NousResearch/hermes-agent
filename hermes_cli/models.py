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
from copy import deepcopy
from difflib import get_close_matches
from pathlib import Path
from typing import Any, NamedTuple, Optional

from hermes_cli import __version__ as _HERMES_VERSION

# Identify ourselves so endpoints fronted by Cloudflare's Browser Integrity
# Check (error 1010) don't reject the default ``Python-urllib/*`` signature.
_HERMES_USER_AGENT = f"hermes-cli/{_HERMES_VERSION}"

COPILOT_BASE_URL = "https://api.githubcopilot.com"
COPILOT_MODELS_URL = f"{COPILOT_BASE_URL}/models"
# Single Copilot CLI identity (the `copilot-developer-cli` integration that
# unlocks the full 33-model catalog). copilot_auth.py is the authoritative
# source; these mirror its fallback values for the degraded ImportError path in
# copilot_default_headers() below, so the identity is identical everywhere.
_COPILOT_INTEGRATION_ID = "copilot-developer-cli"
_COPILOT_CLI_VERSION = "1.0.63"
COPILOT_REASONING_EFFORTS_GPT5 = ["minimal", "low", "medium", "high", "xhigh"]
COPILOT_REASONING_EFFORTS_O_SERIES = ["low", "medium", "high"]

# Account-usable Copilot models the live /models catalog OMITS (hidden/preview
# slugs that work for inference but aren't listed (verified live 2026-06-15):
# gemini-3.5-flash returns 200, gemini-3.1-pro-preview is integrator-gated but
# reachable). Appended to the live catalog so they don't vanish from the picker
# when /models under-reports. Keep to models confirmed reachable on the account.
_COPILOT_HIDDEN_USABLE = ["gemini-3.1-pro-preview", "gemini-3.5-flash"]


# Fallback OpenRouter snapshot used when the live catalog is unavailable.
# (model_id, display description shown in menus)
OPENROUTER_MODELS: list[tuple[str, str]] = [
    # Anthropic
    ("anthropic/claude-opus-4.8",              ""),
    ("anthropic/claude-opus-4.8-fast",         "2x price, higher output speed"),
    ("anthropic/claude-sonnet-4.6",            ""),
    ("anthropic/claude-haiku-4.5",             ""),
    # OpenAI
    ("openai/gpt-5.5",                         ""),
    ("openai/gpt-5.5-pro",                     ""),
    ("openai/gpt-5.4-mini",                    ""),
    # Google
    ("google/gemini-3-pro-preview",            ""),
    ("google/gemini-3.1-pro-preview",          ""),
    ("google/gemini-3.5-flash",                ""),
    # xAI
    ("x-ai/grok-4.3",                          ""),
    # DeepSeek
    ("deepseek/deepseek-v4-pro",               ""),
    ("deepseek/deepseek-v4-flash",             ""),
    # Qwen
    ("qwen/qwen3.7-max",                       ""),
    ("qwen/qwen3.7-plus",                      ""),
    ("qwen/qwen3.6-35b-a3b",                   ""),
    # MoonshotAI
    ("moonshotai/kimi-k2.6",                   "recommended"),
    # MiniMax
    ("minimax/minimax-m3",                     ""),
    # Z-AI
    ("z-ai/glm-5.1",                           ""),
    # Xiaomi
    ("xiaomi/mimo-v2.5-pro",                   ""),
    # Tencent
    ("tencent/hy3-preview",                    ""),
    # StepFun
    ("stepfun/step-3.7-flash",                 ""),
    # NVIDIA
    ("nvidia/nemotron-3-super-120b-a12b",      ""),
    # OpenRouter routers
    ("openrouter/pareto-code",                 "auto-routes to cheapest coder meeting openrouter.min_coding_score"),
    # Free tier
    ("openrouter/elephant-alpha",              "free"),
    ("openrouter/owl-alpha",                   "free"),
    ("tencent/hy3-preview:free",               "free"),
    ("nvidia/nemotron-3-super-120b-a12b:free", "free"),
    ("inclusionai/ring-2.6-1t:free",           "free"),
]

_openrouter_catalog_cache: list[tuple[str, str]] | None = None




def _codex_curated_models() -> list[str]:
    """Derive the openai-codex curated list from codex_models.py.

    Single source of truth: DEFAULT_CODEX_MODELS + forward-compat synthesis.
    This keeps the gateway /model picker in sync with the CLI `hermes model`
    flow without maintaining a separate static list.
    """
    from hermes_cli.codex_models import DEFAULT_CODEX_MODELS, _add_forward_compat_models
    return _add_forward_compat_models(list(DEFAULT_CODEX_MODELS))


# Static fallback for xAI when the models.dev disk cache is empty (fresh
# install, offline first run, etc.). Mirrors the xAI-direct model IDs from
# $HERMES_HOME/models_dev_cache.json as of 2026-04-28. Whenever xAI renames
# or retires a model, the disk cache picks it up on the next refresh and the
# fallback here only matters until that refresh lands.
#
# Models retired by xAI on May 15, 2026 are excluded — see
# https://docs.x.ai/developers/migration/may-15-retirement
# (grok-4, grok-4-0709, grok-4-fast{,-reasoning,-non-reasoning},
#  grok-4-1-fast{,-reasoning,-non-reasoning}, grok-code-fast-1 → grok-4.3).
_XAI_STATIC_FALLBACK: list[str] = [
    "grok-4.3",
    "grok-4.20-0309-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4.20-multi-agent-0309",
]


_XAI_TOP_MODEL = "grok-4.3"


def _xai_promote_top(ids: list[str]) -> list[str]:
    """Pin the headline xAI model to the top of the curated list."""
    if _XAI_TOP_MODEL in ids:
        return [_XAI_TOP_MODEL] + [m for m in ids if m != _XAI_TOP_MODEL]
    return ids


def _xai_curated_models() -> list[str]:
    """Derive the xAI-direct curated list from models.dev disk cache.

    Reads $HERMES_HOME/models_dev_cache.json directly (no network) so this
    runs at import time without blocking. Falls back to ``_XAI_STATIC_FALLBACK``
    when the cache is empty or unreadable. Hermes refreshes the cache from
    https://models.dev/api.json on normal use, so this list self-heals as
    xAI renames models.

    Mirrors ``_codex_curated_models()``'s role for openai-codex.
    """
    try:
        from agent.models_dev import _load_disk_cache
        data = _load_disk_cache()
        xai = data.get("xai") if isinstance(data, dict) else None
        models = xai.get("models") if isinstance(xai, dict) else None
        if isinstance(models, dict) and models:
            ids = [mid for mid in models.keys() if isinstance(mid, str)]
            if ids:
                return _xai_promote_top(sorted(ids))
    except Exception:
        # Any failure (missing file, malformed JSON, import error)
        # falls through to the static list.
        pass
    return list(_XAI_STATIC_FALLBACK)


_PROVIDER_MODELS: dict[str, list[str]] = {
    "nous": [
        # Anthropic
        "anthropic/claude-opus-4.8",
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-haiku-4.5",
        # OpenAI
        "openai/gpt-5.5",
        "openai/gpt-5.5-pro",
        "openai/gpt-5.4-mini",
        # Google
        "google/gemini-3-pro-preview",
        "google/gemini-3.1-pro-preview",
        "google/gemini-3.5-flash",
        # xAI
        "x-ai/grok-4.3",
        # DeepSeek
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-flash",
        # Qwen
        "qwen/qwen3.7-max",
        "qwen/qwen3.7-plus",
        "qwen/qwen3.6-35b-a3b",
        # MoonshotAI
        "moonshotai/kimi-k2.6",
        # MiniMax
        "minimax/minimax-m3",
        # Z-AI
        "z-ai/glm-5.1",
        # Xiaomi
        "xiaomi/mimo-v2.5-pro",
        # Tencent
        "tencent/hy3-preview",
        # StepFun
        "stepfun/step-3.7-flash",
        # NVIDIA
        "nvidia/nemotron-3-super-120b-a12b",
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
    "openai-api": [
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5-mini",
        "gpt-5.3-codex",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "openai-codex": _codex_curated_models(),
    "xai-oauth": _xai_curated_models(),
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
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemma-4-31b-it",
        "gemma-4-26b-a4b-it",
        # claude-fable-5 = the GA slug for the Mythos-class model (preview
        # codename "claude-mythos-*"). Confirmed canonical 3 ways: web changelog,
        # server error-code differentiation, and the official @github/copilot
        # 1.0.61 bundle. Requires an org admin to enable Fable 5 in Copilot
        # policies (30-day data-retention opt-in); until then the catalog omits
        # it and selecting it returns model_not_available_for_integrator.
        "claude-fable-5",
        "goldeneye-secondary",
    ],
    "gemini": [
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite-preview",
    ],
    "google-gemini-cli": [
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        # Code Assist serves two flash slugs with different access gates
        # (gemini-cli models.ts): gemini-3-flash-preview is the preview flash
        # that subscription/free-tier OAuth users actually reach, while
        # gemini-3.5-flash is GA-channel-gated. Offer both so non-GA users
        # aren't stuck with a slug cloudcode-pa 404s for them.
        "gemini-3-flash-preview",
        "gemini-3.5-flash",
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
    "xai": _xai_curated_models(),
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
        "MiniMax-M3",
        "MiniMax-M2.7",
        "MiniMax-M2.5",
        "MiniMax-M2.1",
        "MiniMax-M2",
    ],
    "minimax-oauth": [
        "MiniMax-M3",
        "MiniMax-M2.7",
        "MiniMax-M2.7-highspeed",
    ],
    "minimax-cn": [
        "MiniMax-M3",
        "MiniMax-M2.7",
        "MiniMax-M2.5",
        "MiniMax-M2.1",
        "MiniMax-M2",
    ],
    "anthropic": [
        "claude-opus-4-8",
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
    "tencent-tokenhub": [
        "hy3-preview",
    ],
    "arcee": [
        "trinity-large-thinking",
        "trinity-large-preview",
        "trinity-mini",
    ],
    "gmi": [
        "zai-org/GLM-5.1-FP8",
        "deepseek-ai/DeepSeek-V3.2",
        "moonshotai/Kimi-K2.5",
        "google/gemini-3.1-flash-lite-preview",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
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
        "qwen3.7-max",
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
        "qwen3.7-max",
        "qwen3.6-plus",
        "kimi-k2.5",
        "qwen3.5-plus",
        "qwen3-coder-plus",
        "qwen3-coder-next",
        # Third-party models available on coding-intl
        "glm-5",
        "glm-4.7",
        "MiniMax-M2.5",
    ],
    # Alibaba Coding Plan — same platform as alibaba (DashScope coding-intl),
    # separate provider ID with its own base_url_env_var.
    "alibaba-coding-plan": [
        "qwen3.7-max",
        "qwen3.6-plus",
        "qwen3.5-plus",
        "qwen3-coder-plus",
        "qwen3-coder-next",
        "kimi-k2.5",
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
    "novita": [
        "moonshotai/kimi-k2.5",
        "minimax/minimax-m2.7",
        "zai-org/glm-5",
        "deepseek/deepseek-v3-0324",
        "deepseek/deepseek-r1-0528",
        "qwen/qwen3-235b-a22b-fp8",
    ],
}

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
def is_nous_free_tier(account_info: dict[str, Any]) -> bool:
    """Return True if the account info indicates a free (unpaid) tier.

    Prefer the Portal's explicit ``paid_service_access.allowed`` entitlement
    decision.  Legacy payloads fall back to ``subscription.monthly_charge == 0``.
    Returns False when both signals are missing or unparseable.
    """
    paid_access = account_info.get("paid_service_access")
    if isinstance(paid_access, dict):
        allowed = paid_access.get("allowed")
        if isinstance(allowed, bool):
            return not allowed
        paid = paid_access.get("paid_access")
        if isinstance(paid, bool):
            return not paid

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


def union_with_portal_free_recommendations(
    curated_ids: list[str],
    pricing: dict[str, dict[str, str]],
    portal_base_url: str = "",
    *,
    force_refresh: bool = False,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Augment curated list + pricing with the Portal's ``freeRecommendedModels``.

    The Portal's ``/api/nous/recommended-models`` endpoint advertises which
    models are free *right now* — independent of what the in-repo
    ``_PROVIDER_MODELS["nous"]`` list happens to contain or whether the
    docs-hosted catalog manifest has been rebuilt since the last release.

    For free-tier users this is the source of truth: any model the Portal
    flags as free should be selectable, even if the user is running an
    older Hermes that doesn't ship that model in its hardcoded curated
    list.  This function returns an augmented ``(model_ids, pricing)``
    pair where:

    * Portal free recommendations missing from ``curated_ids`` are
      appended after the curated list (so the in-repo curated models
      show first and Portal-only picks follow).
    * ``pricing`` gets a synthetic ``{"prompt": "0", "completion": "0"}``
      entry for any free recommendation missing from the live pricing
      map, so :func:`partition_nous_models_by_tier` keeps it.

    Failures (network, parse, missing field) are silent and degrade to
    returning the inputs unchanged.
    """
    try:
        payload = fetch_nous_recommended_models(
            portal_base_url, force_refresh=force_refresh
        )
    except Exception:
        return (list(curated_ids), dict(pricing))

    free_block = payload.get("freeRecommendedModels") if isinstance(payload, dict) else None
    if not isinstance(free_block, list) or not free_block:
        return (list(curated_ids), dict(pricing))

    portal_free_ids: list[str] = []
    for entry in free_block:
        name = _extract_model_name(entry)
        if name:
            portal_free_ids.append(name)
    if not portal_free_ids:
        return (list(curated_ids), dict(pricing))

    augmented_pricing = dict(pricing)
    free_synthetic = {"prompt": "0", "completion": "0"}
    for mid in portal_free_ids:
        if mid not in augmented_pricing:
            augmented_pricing[mid] = dict(free_synthetic)

    augmented_ids = list(curated_ids)
    seen = set(augmented_ids)
    # Append Portal free recommendations that aren't already curated, so the
    # in-repo curated ("HA") models show first and Portal-only picks follow.
    new_ones = [mid for mid in portal_free_ids if mid not in seen]
    if new_ones:
        augmented_ids = augmented_ids + new_ones

    return (augmented_ids, augmented_pricing)


def union_with_portal_paid_recommendations(
    curated_ids: list[str],
    pricing: dict[str, dict[str, str]],
    portal_base_url: str = "",
    *,
    force_refresh: bool = False,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Augment curated list with the Portal's ``paidRecommendedModels``.

    Mirror of :func:`union_with_portal_free_recommendations` for paid-tier
    users. The Portal's ``/api/nous/recommended-models`` endpoint advertises
    which paid models are blessed *right now* — independent of what the
    in-repo ``_PROVIDER_MODELS["nous"]`` list happens to contain or whether
    the docs-hosted catalog manifest has been rebuilt since the last release.

    For paid-tier users this lets newly-launched paid models surface in the
    picker even if the user is running an older Hermes that doesn't ship
    them in its hardcoded curated list. This function returns an augmented
    ``(model_ids, pricing)`` pair where:

    * Portal paid recommendations missing from ``curated_ids`` are
      appended after the curated list (so the in-repo curated models
      show first and Portal-only picks follow).
    * ``pricing`` is left untouched — we deliberately do NOT synthesize
      pricing entries for paid models. Live pricing is fetched separately
      via :func:`get_pricing_for_provider`; if the live endpoint hasn't
      published pricing yet, the picker shows a blank price column rather
      than fabricating numbers. (The free helper synthesizes ``$0`` so
      :func:`partition_nous_models_by_tier` keeps free models selectable;
      no equivalent gating applies on the paid side, so synthesis would
      only mislead the user.)

    Failures (network, parse, missing field) are silent and degrade to
    returning the inputs unchanged — never block the picker on a
    Portal-side hiccup.
    """
    try:
        payload = fetch_nous_recommended_models(
            portal_base_url, force_refresh=force_refresh
        )
    except Exception:
        return (list(curated_ids), dict(pricing))

    paid_block = payload.get("paidRecommendedModels") if isinstance(payload, dict) else None
    if not isinstance(paid_block, list) or not paid_block:
        return (list(curated_ids), dict(pricing))

    portal_paid_ids: list[str] = []
    for entry in paid_block:
        name = _extract_model_name(entry)
        if name:
            portal_paid_ids.append(name)
    if not portal_paid_ids:
        return (list(curated_ids), dict(pricing))

    augmented_ids = list(curated_ids)
    seen = set(augmented_ids)
    # Append Portal paid recommendations that aren't already curated, so the
    # in-repo curated ("HA") models show first and Portal-only picks follow.
    new_ones = [mid for mid in portal_paid_ids if mid not in seen]
    if new_ones:
        augmented_ids = augmented_ids + new_ones

    return (augmented_ids, dict(pricing))


# ---------------------------------------------------------------------------
# TTL cache for free-tier detection — avoids repeated API calls within a
# session while still picking up upgrades quickly.
# ---------------------------------------------------------------------------
_FREE_TIER_CACHE_TTL: int = 180  # seconds (3 minutes)
_free_tier_cache: tuple[bool, float] | None = None  # (result, timestamp)


def check_nous_free_tier(*, force_fresh: bool = False) -> bool:
    """Check if the current Nous Portal user is on a free (unpaid) tier.

    Results are cached for ``_FREE_TIER_CACHE_TTL`` seconds to avoid
    hitting the Portal API on every call.  The cache is short-lived so
    that an account upgrade is reflected within a few minutes.

    Returns True only when entitlement is known to be free.  Unknown/error
    states return False so this compatibility wrapper does not block users.
    """
    global _free_tier_cache
    now = time.monotonic()
    if not force_fresh and _free_tier_cache is not None:
        cached_result, cached_at = _free_tier_cache
        if now - cached_at < _FREE_TIER_CACHE_TTL:
            return cached_result

    try:
        from hermes_cli.nous_account import get_nous_portal_account_info

        account_info = get_nous_portal_account_info(force_fresh=force_fresh)
        result = account_info.is_free_tier
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
    ProviderEntry("nous",           "Nous Portal",              "Nous Portal (Everything your agent needs, 300+ models with bundled tool use)"),
    ProviderEntry("openrouter",     "OpenRouter",               "OpenRouter (Pay-per-use API aggregator)"),
    ProviderEntry("novita",         "NovitaAI",                 "NovitaAI (Cloud: Model API, Agent Sandbox, GPU Cloud)"),
    ProviderEntry("lmstudio",       "LM Studio",                "LM Studio (Local desktop app with built-in model server)"),
    ProviderEntry("anthropic",      "Anthropic",                "Anthropic (Claude models via API key or Claude Code)"),
    ProviderEntry("openai-codex",   "OpenAI Codex",             "OpenAI Codex (Codex CLI via ChatGPT subscription or API key)"),
    ProviderEntry("openai-api",     "OpenAI API",               "OpenAI API (api.openai.com, API key)"),
    ProviderEntry("alibaba",        "Qwen Cloud",               "Qwen Cloud / DashScope (Qwen + multi-provider)"),
    ProviderEntry("xai-oauth",      "xAI Grok OAuth (SuperGrok / Premium+)", "xAI Grok OAuth (SuperGrok / Premium+ subscription)"),
    ProviderEntry("xiaomi",         "Xiaomi MiMo",              "Xiaomi MiMo (MiMo-V2.5 and V2 models: pro, omni, flash)"),
    ProviderEntry("tencent-tokenhub", "Tencent TokenHub",       "Tencent TokenHub (Hy3 Preview via tokenhub.tencentmaas.com)"),
    ProviderEntry("nvidia",         "NVIDIA NIM",               "NVIDIA NIM (Nemotron models via build.nvidia.com or local NIM)"),
    ProviderEntry("copilot",        "GitHub Copilot",           "GitHub Copilot (Uses GITHUB_TOKEN or gh auth token)"),
    ProviderEntry("copilot-acp",    "GitHub Copilot ACP",       "GitHub Copilot ACP (Spawns copilot --acp --stdio)"),
    ProviderEntry("huggingface",    "Hugging Face",             "Hugging Face Inference Providers"),
    ProviderEntry("gemini",         "Google AI Studio",         "Google AI Studio (Native Gemini API)"),
    ProviderEntry("google-gemini-cli", "Google Gemini (OAuth)",   "Google Gemini via OAuth + Code Assist (Code Assist OAuth flow)"),
    ProviderEntry("deepseek",       "DeepSeek",                 "DeepSeek (V3, R1, coder, direct API)"),
    ProviderEntry("xai",            "xAI",                      "xAI Grok (Direct API)"),
    ProviderEntry("zai",            "Z.AI / GLM",               "Z.AI / GLM (Zhipu direct API)"),
    ProviderEntry("kimi-coding",    "Kimi / Kimi Coding Plan",  "Kimi Coding Plan (api.kimi.com & Moonshot API)"),
    ProviderEntry("kimi-coding-cn", "Kimi / Moonshot (China)",  "Kimi / Moonshot China (Domestic direct API)"),
    ProviderEntry("stepfun",        "StepFun Step Plan",       "StepFun Step Plan (Agent / coding models via Step Plan API)"),
    ProviderEntry("minimax",        "MiniMax",                  "MiniMax (Global direct API)"),
    ProviderEntry("minimax-oauth",  "MiniMax (OAuth)",          "MiniMax via OAuth browser login (Coding Plan, minimax.io)"),
    ProviderEntry("minimax-cn",     "MiniMax (China)",          "MiniMax China (Domestic direct API)"),
    ProviderEntry("ollama-cloud",   "Ollama Cloud",             "Ollama Cloud (Cloud-hosted open models, ollama.com)"),
    ProviderEntry("arcee",          "Arcee AI",                 "Arcee AI (Trinity models, direct API)"),
    ProviderEntry("gmi",            "GMI Cloud",                "GMI Cloud (Multi-model direct API)"),
    ProviderEntry("kilocode",       "Kilo Code",                "Kilo Code (Kilo Gateway API)"),
    ProviderEntry("opencode-zen",   "OpenCode Zen",             "OpenCode Zen (Curated models, pay-as-you-go)"),
    ProviderEntry("opencode-go",    "OpenCode Go",              "OpenCode Go (Open models subscription)"),
    ProviderEntry("bedrock",        "AWS Bedrock",              "AWS Bedrock (Claude, Nova, Llama, DeepSeek; IAM or API key)"),
    ProviderEntry("azure-foundry",  "Azure Foundry",            "Azure Foundry (OpenAI-style or Anthropic-style endpoint, your Azure AI deployment)"),
    ProviderEntry("qwen-oauth",     "Qwen OAuth (Portal)",      "Qwen OAuth (Reuses local Qwen CLI login)"),
]

# Auto-extend CANONICAL_PROVIDERS with any provider registered in providers/
# that is not already in the list above.  Adding plugins/model-providers/<name>/
# is sufficient to expose a new provider in the model picker, /model, and all
# downstream consumers — no edits to this file needed.
_canonical_slugs = {p.slug for p in CANONICAL_PROVIDERS}
try:
    from providers import list_providers as _list_providers_for_canonical
    for _pp in _list_providers_for_canonical():
        if _pp.name in _canonical_slugs:
            continue
        if _pp.auth_type in {"oauth_device_code", "oauth_external", "external_process", "aws_sdk", "copilot"}:
            continue  # non-api-key flows need bespoke picker UX; skip auto-inject
        _label = _pp.display_name or _pp.name
        _desc = _pp.description or f"{_label} (direct API)"
        CANONICAL_PROVIDERS.append(ProviderEntry(_pp.name, _label, _desc))
        _canonical_slugs.add(_pp.name)
except Exception:
    pass

# Derived dicts — used throughout the codebase
_PROVIDER_LABELS = {p.slug: p.label for p in CANONICAL_PROVIDERS}
_PROVIDER_LABELS["custom"] = "Custom endpoint"  # special case: not a named provider


# ---------------------------------------------------------------------------
# Provider groups — DISPLAY ONLY
#
# Some vendors expose several Hermes provider slugs (one per endpoint /
# auth method: global API, China API, OAuth coding plan, ...). Listing every
# slug as a top-level row in the interactive `hermes model` / setup wizard /
# Telegram `/model` pickers makes that list long and noisy.
#
# These groups fold related slugs under one top-level row in INTERACTIVE
# PICKERS only. They do NOT change ``CANONICAL_PROVIDERS``, slug identity,
# the ``--provider`` flag, ``/model <provider:model>``, or any typed path —
# every member slug remains individually addressable. Grouping is a pure
# display affordance; ``group_providers()`` is the single fold used by all
# three picker surfaces so they stay consistent.
#
#   group_id -> (display_label, group_description, [member_slug, ...])
#
# ``group_description`` is a short blurb shown on the collapsed top-level group
# row in the interactive pickers (alongside the label). Member-specific detail
# lives in each member's ``tui_desc`` and shows in the drill-down sub-picker.
# Member order is the order shown inside the group submenu.
# ---------------------------------------------------------------------------
PROVIDER_GROUPS: dict[str, tuple[str, str, list[str]]] = {
    "kimi":     ("Kimi / Moonshot", "Coding Plan, Moonshot global & China endpoints", ["kimi-coding", "kimi-coding-cn"]),
    "minimax":  ("MiniMax",         "Global, OAuth Coding Plan & China endpoints",     ["minimax", "minimax-oauth", "minimax-cn"]),
    "xai":      ("xAI Grok",        "Direct API or SuperGrok / Premium+ OAuth",        ["xai", "xai-oauth"]),
    "google":   ("Google Gemini",   "AI Studio API or OAuth + Code Assist",            ["gemini", "google-gemini-cli"]),
    "openai":   ("OpenAI",          "Codex CLI or direct OpenAI API",                  ["openai-codex", "openai-api"]),
    "opencode": ("OpenCode",        "Zen pay-as-you-go or Go subscription",            ["opencode-zen", "opencode-go"]),
    "copilot":  ("GitHub Copilot",  "GitHub token API or copilot --acp process",       ["copilot", "copilot-acp"]),
}

# Reverse index: member slug -> group_id. Built once at import.
_SLUG_TO_GROUP: dict[str, str] = {
    slug: gid for gid, (_label, _desc, members) in PROVIDER_GROUPS.items() for slug in members
}


def provider_group_for_slug(slug: str) -> str:
    """Return the group_id a provider slug belongs to, or "" if ungrouped."""
    return _SLUG_TO_GROUP.get(str(slug or "").strip().lower(), "")


def group_providers(slugs):
    """Fold a flat ordered slug iterable into picker rows by provider group.

    DISPLAY ONLY. Used by every interactive picker (``hermes model``, the
    setup wizard, the Telegram ``/model`` keyboard) so grouping is identical
    across surfaces.

    Each returned row is a dict::

        {"kind": "single", "slug": <slug>}                       # ungrouped, or
                                                                  # 1-member group
        {"kind": "group", "group_id": <gid>, "label": <label>,
         "description": <desc>, "members": [<slug>, ...]}        # 2+ members

    Rules:
      * A group row appears at the position of its FIRST present member, in
        the input order. Subsequent members fold into that row (and are not
        emitted again).
      * Member order inside a group follows ``PROVIDER_GROUPS`` declaration,
        restricted to the members actually present in ``slugs``.
      * A group reduced to a single present member degrades to a ``single``
        row — no pointless one-item submenu.
      * Slugs not in any group pass through as ``single`` rows, order
        preserved.
      * Duplicate slugs in the input are ignored after first sight.
    """
    seen: set[str] = set()
    # Which present members each group has, in declaration order.
    group_members: dict[str, list[str]] = {}
    for gid, (_label, _desc, members) in PROVIDER_GROUPS.items():
        present = [m for m in members if m in set(slugs)]
        if present:
            group_members[gid] = present

    rows = []
    emitted_groups: set[str] = set()
    for slug in slugs:
        s = str(slug or "").strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        gid = _SLUG_TO_GROUP.get(s, "")
        if not gid:
            rows.append({"kind": "single", "slug": s})
            continue
        if gid in emitted_groups:
            continue  # already folded at the first member's position
        emitted_groups.add(gid)
        members = group_members.get(gid, [s])
        if len(members) <= 1:
            rows.append({"kind": "single", "slug": members[0]})
        else:
            label, desc, _ = PROVIDER_GROUPS[gid]
            rows.append(
                {"kind": "group", "group_id": gid, "label": label,
                 "description": desc, "members": list(members)}
            )
    return rows


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
    "gmi-cloud": "gmi",
    "gmicloud": "gmi",
    "minimax-china": "minimax-cn",
    "minimax_cn": "minimax-cn",
    "minimax-portal": "minimax-oauth",
    "minimax-global": "minimax-oauth",
    "minimax_oauth": "minimax-oauth",
    "claude": "anthropic",
    "claude-code": "anthropic",
    "deep-seek": "deepseek",
    "opencode": "opencode-zen",
    "zen": "opencode-zen",
    "go": "opencode-go",
    "opencode-go-sub": "opencode-go",
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
    "novita-ai": "novita",
    "novitaai": "novita",
    "mimo": "xiaomi",
    "xiaomi-mimo": "xiaomi",
    "tencent": "tencent-tokenhub",
    "tokenhub": "tencent-tokenhub",
    "tencent-cloud": "tencent-tokenhub",
    "tencentmaas": "tencent-tokenhub",
    "aws": "bedrock",
    "aws-bedrock": "bedrock",
    "amazon-bedrock": "bedrock",
    "amazon": "bedrock",
    "grok": "xai",
    "grok-oauth": "xai-oauth",
    "xai-oauth": "xai-oauth",
    "x-ai-oauth": "xai-oauth",
    "xai-grok-oauth": "xai-oauth",
    "x-ai": "xai",
    "x.ai": "xai",
    "nim": "nvidia",
    "nvidia-nim": "nvidia",
    "build-nvidia": "nvidia",
    "nemotron": "nvidia",
    "lmstudio": "lmstudio",
    "lm-studio": "lmstudio",
    "lm_studio": "lmstudio",
    "ollama": "custom",  # bare "ollama" = local; use "ollama-cloud" for cloud
    "ollama_cloud": "ollama-cloud",
}


# Cost-safe overrides for the *silent* auto-default
# (``get_default_model_for_provider``). Most providers' curated lists lead with a
# sensible default, but Nous Portal is a per-token *metered aggregator* whose
# list is ordered best-/most-capable-first — entry [0] is the priciest flagship
# (``anthropic/claude-opus-4.8``, $5/$25 per Mtok). Using that as the
# non-interactive fallback when a profile sets ``provider: nous`` with no model
# silently bills the most expensive model for traffic the user never opted into
# (a missing default escalated to Opus and billed 863 requests before the user
# noticed). Pin the silent default to a low-cost curated model instead so a
# missing model can never escalate to the flagship.
#
# This is deliberately a fixed, side-effect-free default for the hot resolution
# path. The *interactive* default (GUI onboarding / ``hermes model``) uses the
# richer free/paid-tier-aware resolver — see ``get_recommended_default_model``
# in hermes_cli/web_server.py and ``partition_nous_models_by_tier`` — which can
# hit the Portal; this fallback must stay cheap and network-free.
_PROVIDER_SILENT_DEFAULT_OVERRIDES: dict[str, str] = {
    "nous": "deepseek/deepseek-v4-flash",
}


def get_default_model_for_provider(provider: str) -> str:
    """Return a cost-safe default model for a provider, or "" if unknown.

    Used as a NON-INTERACTIVE fallback when a provider is configured but no
    model was ever selected (e.g. ``hermes auth add openai-codex`` without
    ``hermes model``, or a profile that sets ``provider`` with no ``model``).

    For most providers this is the first entry in ``_PROVIDER_MODELS`` — the
    same model the ``hermes model`` picker offers first. For metered aggregators
    whose curated list is ordered most-capable-first, that entry is also the
    most EXPENSIVE one, so silently defaulting to it is a billing footgun. Such
    providers carry an explicit low-cost override in
    ``_PROVIDER_SILENT_DEFAULT_OVERRIDES``; a missing model must never
    auto-escalate to the flagship.
    """
    models = _PROVIDER_MODELS.get(provider, [])
    override = _PROVIDER_SILENT_DEFAULT_OVERRIDES.get(provider)
    if override and override in models:
        return override
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


def _resolve_openrouter_api_key() -> str:
    """Best-effort OpenRouter API key for pricing fetch."""
    return os.getenv("OPENROUTER_API_KEY", "").strip()


_DEFAULT_NOUS_INFERENCE_BASE = "https://inference-api.nousresearch.com"


def _resolve_nous_pricing_credentials() -> tuple[str, str]:
    """Return ``(api_key, base_url)`` for Nous Portal pricing.

    The Nous inference ``/v1/models`` endpoint exposes pricing without
    authentication, so the api_key is best-effort: when runtime credential
    resolution fails (expired refresh token, missing auth.json, etc.) we
    still return the default inference base URL so the picker keeps
    working with anonymous pricing data.  Free-tier users in particular
    need this — pricing drives the free/paid partition, and silently
    returning empty pricing because of an auth blip makes the picker
    look broken ("No free models currently available").
    """
    try:
        from hermes_cli.auth import resolve_nous_runtime_credentials
        creds = resolve_nous_runtime_credentials()
        if creds:
            return (creds.get("api_key", ""), creds.get("base_url", ""))
    except Exception:
        pass
    return ("", _DEFAULT_NOUS_INFERENCE_BASE)


def get_pricing_for_provider(provider: str, *, force_refresh: bool = False) -> dict[str, dict[str, str]]:
    """Return live pricing for providers that support it (openrouter, nous, novita)."""
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return fetch_models_with_pricing(
            api_key=_resolve_openrouter_api_key(),
            base_url="https://openrouter.ai/api",
            force_refresh=force_refresh,
        )
    if normalized == "novita":
        return _fetch_novita_pricing(force_refresh=force_refresh)
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


def _fetch_novita_pricing(
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, str]]:
    """Fetch pricing from NovitaAI /v1/models.

    NovitaAI returns input/output prices per million tokens in units of
    0.0001 USD. Convert them to the per-token strings used by the shared
    pricing formatter.

    Results are cached in ``_pricing_cache`` keyed on the resolved base URL —
    without this, every menu render or pricing lookup re-hits the network.
    """
    api_key = os.getenv("NOVITA_API_KEY", "").strip()
    if not api_key:
        return {}

    base_url = os.getenv("NOVITA_BASE_URL", "").strip() or "https://api.novita.ai/openai/v1"
    cache_key = base_url.rstrip("/")
    if not force_refresh and cache_key in _pricing_cache:
        return _pricing_cache[cache_key]

    url = cache_key + "/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": _HERMES_USER_AGENT,
    }

    try:
        req = urllib.request.Request(url, headers=headers)
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
        if not mid:
            continue
        inp = item.get("input_token_price_per_m")
        out = item.get("output_token_price_per_m")
        if inp is None and out is None:
            continue
        result[str(mid)] = {
            "prompt": str(float(inp or 0) / 10_000 / 1_000_000),
            "completion": str(float(out or 0) / 10_000 / 1_000_000),
        }

    _pricing_cache[cache_key] = result
    return result


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
    {"nous", "openrouter", "copilot", "kilocode"}
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
#
# Pattern-based matching — any OpenAI flagship model (gpt-*, o1*, o3*, o4*)
# is assumed to support Priority Processing. service_tier=priority is silently
# ignored by non-OpenAI endpoints (OpenRouter/Copilot/opencode-zen proxies
# strip the field), so false positives are harmless. Codex-series models
# (gpt-5-codex, gpt-5.3-codex, etc.) are excluded — they don't expose the
# service_tier parameter through the Codex Responses API.
_OPENAI_FAST_MODE_PREFIXES: tuple[str, ...] = (
    "gpt-",
    "o1",
    "o3",
    "o4",
)


def _is_openai_fast_model(model_id: Optional[str]) -> bool:
    """Return True if the model is an OpenAI flagship eligible for Priority Processing."""
    raw = _strip_vendor_prefix(str(model_id or ""))
    base = raw.split(":")[0]
    if not base:
        return False
    # Exclude Codex-series — they route through the Codex Responses API
    # which doesn't accept service_tier.
    if "codex" in base:
        return False
    return any(base.startswith(prefix) for prefix in _OPENAI_FAST_MODE_PREFIXES)


# Models that support Anthropic Fast Mode (speed="fast").
# See https://platform.claude.com/docs/en/build-with-claude/fast-mode
#
# Pattern-based matching — any claude-* model is eligible. The anthropic
# adapter gates speed=fast on native Anthropic endpoints only (see
# _is_third_party_anthropic_endpoint in agent/anthropic_adapter.py), so
# third-party proxies that would reject the beta header are protected.


def _strip_vendor_prefix(model_id: str) -> str:
    """Strip vendor/ prefix from a model ID (e.g. 'anthropic/claude-opus-4-6' -> 'claude-opus-4-6')."""
    raw = str(model_id or "").strip().lower()
    if "/" in raw:
        raw = raw.split("/", 1)[1]
    return raw


def model_supports_fast_mode(model_id: Optional[str]) -> bool:
    """Return whether Hermes should expose the /fast toggle for this model."""
    return _is_anthropic_fast_model(model_id) or _is_openai_fast_model(model_id)


def _is_anthropic_fast_model(model_id: Optional[str]) -> bool:
    """Return True if the model accepts the Anthropic Fast Mode ``speed`` param.

    This gates the *speed=fast request parameter*, which Anthropic supports on
    Opus 4.6 only (Opus 4.7 explicitly 400s). It is deliberately NOT a general
    "is this a fast model" check: for Opus 4.8 the fast offering is a SEPARATE
    model id (``…-opus-4.8-fast``) selected via the model field, not the speed
    parameter — see ``agent.anthropic_adapter._supports_fast_mode`` and its
    test. Keep this in lock-step with that adapter gate so the UI never shows a
    Fast toggle that the runtime would silently drop.
    """
    raw = _strip_vendor_prefix(str(model_id or ""))
    base = raw.split(":")[0]
    if not base.startswith("claude-"):
        return False
    # Only Opus 4.6 supports the speed=fast parameter at present.
    return "opus-4-6" in base or "opus-4.6" in base


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
    """Best-effort GitHub token for fetching the Copilot model catalog.

    Resolution order:
      1. Copilot env vars (``COPILOT_GITHUB_TOKEN`` / ``GH_TOKEN`` /
         ``GITHUB_TOKEN``) via the shared identity audit helper.
      2. ``credential_pool.copilot[]`` entries, skipping malformed or
         unsupported entries before attempting the next one.
      3. ``gh auth token`` as the final fallback when both env vars and the
         credential pool are exhausted.

    Pool tokens are still exchanged before returning so the catalog fetch
    keeps the current Copilot API behavior while sharing the same identity
    selection path as the compatibility wrapper.
    """
    try:
        from hermes_cli.copilot_auth import resolve_copilot_identity_audit

        audit = resolve_copilot_identity_audit(
            include_credential_pool=True,
            exchange_pool_tokens=True,
        )
        if audit.error or not audit.token:
            return ""
        return audit.token
    except Exception:
        pass

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
# branches below (copilot, anthropic, ollama-cloud, custom,
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
    if normalized == "xai-oauth":
        return list(_PROVIDER_MODELS.get("xai-oauth", _PROVIDER_MODELS.get("xai", [])))
    if normalized in {"copilot", "copilot-acp"}:
        try:
            live = _fetch_github_models(_resolve_copilot_catalog_api_key())
            if live:
                # The /models catalog omits some account-usable models (hidden/preview
                # slugs that work for inference but aren't listed, e.g. gemini-3.5-flash).
                # Append them so they don't silently vanish from the picker. Deduped;
                # live entries win, supplements only fill gaps.
                merged = list(live)
                for _m in _COPILOT_HIDDEN_USABLE:
                    if _m not in merged:
                        merged.append(_m)
                return merged
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
        # Live failed (or no creds). Fall back to the docs-hosted manifest
        # — NOT the in-repo _PROVIDER_MODELS["nous"] snapshot — so newly
        # added Portal models still surface without a Hermes release.
        manifest_ids = get_curated_nous_model_ids()
        if manifest_ids:
            return manifest_ids
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
    if normalized == "ollama-cloud":
        live = fetch_ollama_cloud_models(force_refresh=force_refresh)
        if live:
            return live
    if normalized in ("openai", "openai-api"):
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            base_raw = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
            base = base_raw or "https://api.openai.com/v1"
            # Custom OpenAI-compatible endpoints (proxies, gateways, self-hosted)
            # may serve a small curated catalog — use the live list verbatim so
            # discovery works. But the canonical api.openai.com /v1/models dump
            # is 120+ entries of embeddings, whisper, tts, dall-e, moderation and
            # legacy chat models — none of which belong in the agent model picker.
            # For the default endpoint, intersect the live list with our curated
            # agentic catalog so ``/model`` matches what ``hermes model`` shows.
            is_default_openai = base.rstrip("/") in (
                "https://api.openai.com/v1",
                "https://api.openai.com",
            )
            try:
                live = fetch_api_models(api_key, base)
                if live:
                    if is_default_openai:
                        live_lower = {m.lower() for m in live}
                        curated = list(_PROVIDER_MODELS.get(normalized, []))
                        # Keep curated order; only surface curated models the
                        # account actually has access to.
                        filtered = [m for m in curated if m.lower() in live_lower]
                        if filtered:
                            return filtered
                        # Account serves none of the curated models (rare —
                        # e.g. org without GPT-5 access). Fall back to curated
                        # so the picker still offers sane defaults.
                        return curated or live
                    return live
            except Exception:
                pass
    if normalized == "gmi":
        try:
            from hermes_cli.auth import resolve_api_key_provider_credentials

            creds = resolve_api_key_provider_credentials("gmi")
            api_key = str(creds.get("api_key") or "").strip()
            base_url = str(creds.get("base_url") or "").strip()
            if api_key and base_url:
                live = fetch_api_models(api_key, base_url)
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
    # Bedrock uses live discovery keyed by the resolved AWS region so that
    # EU/AP users see eu.*/ap.* model IDs instead of the static us.* list.
    # Note: early return intentionally skips _MODELS_DEV_PREFERRED merge
    # below — bedrock is not expected to appear in that table.
    if normalized == "bedrock":
        try:
            from agent.bedrock_adapter import bedrock_model_ids_or_none
            ids = bedrock_model_ids_or_none()
            if ids is not None:
                return ids
        except Exception:
            pass

    # ── Profile-based generic live fetch (all simple api-key providers) ──
    # Handles any provider registered in providers/ with auth_type="api_key".
    # Replaces per-provider copy-paste blocks (stepfun, gmi, zai, etc.).
    try:
        from providers import get_provider_profile
        from hermes_cli.auth import resolve_api_key_provider_credentials

        _p = get_provider_profile(normalized)
        if _p and _p.auth_type == "api_key" and _p.base_url:
            try:
                creds = resolve_api_key_provider_credentials(normalized)
                api_key = str(creds.get("api_key") or "").strip()
                base_url = str(creds.get("base_url") or "").strip()
            except Exception:
                api_key, base_url = "", _p.base_url
            if not base_url:
                base_url = _p.base_url
            if api_key:
                live = _p.fetch_models(api_key=api_key)
                if live:
                    return live
            # Use profile's fallback_models if defined
            if _p.fallback_models:
                return list(_p.fallback_models)
    except Exception:
        pass

    curated_static = list(_PROVIDER_MODELS.get(normalized, []))
    if normalized in _MODELS_DEV_PREFERRED:
        return _merge_with_models_dev(normalized, curated_static)
    return curated_static


# ---------------------------------------------------------------------------
# Generic disk cache for provider_model_ids() — keeps /model picker fast.
# ---------------------------------------------------------------------------
#
# Without this layer, every /model picker open re-fetches every authed
# provider's /v1/models endpoint. On a well-configured user (anthropic +
# openai + copilot + gemini + huggingface + ...) that's 2+ seconds of cold
# HTTP roundtrips just to render the provider list.
#
# Cache strategy:
#   - One JSON file at $HERMES_HOME/provider_models_cache.json
#   - Per-provider entries keyed by (provider, credential fingerprint)
#   - Credential fingerprint = sha256 of env-var values that the provider
#     normally reads. Swap your OPENAI_API_KEY and the entry invalidates.
#   - 1h TTL by default. `force_refresh=True` skips the cache entirely
#     and overwrites it on success.
#   - Only NON-EMPTY results are cached. An empty/None response from a
#     transient network error never gets pinned.
#   - Cache file is best-effort. Any read/write error degrades silently
#     to a live fetch — the picker keeps working.

_PROVIDER_MODELS_CACHE_TTL = 3600  # 1h


def _provider_models_cache_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "provider_models_cache.json"


def _credential_fingerprint(provider: str) -> str:
    """Return a short hash representing the credentials that
    ``provider_model_ids(provider)`` would see right now.

    Rotating any of the relevant env vars invalidates the cached entry
    for that provider. We hash AT LEAST the api-key + base-url env vars
    declared in ``PROVIDER_REGISTRY``. For OAuth-backed providers
    (codex, copilot, anthropic-via-claude-code, nous portal), the
    relevant tokens live in ``$HERMES_HOME/auth.json`` and external
    credential files. Rather than parse every shape, we additionally
    fold the mtime of those files into the fingerprint so refreshes
    after re-auth bust the cache.
    """
    import hashlib
    import os as _os

    parts: list[str] = []

    # Env vars from PROVIDER_REGISTRY for this slug
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        pcfg = PROVIDER_REGISTRY.get(provider)
        if pcfg is not None:
            for ev in getattr(pcfg, "api_key_env_vars", ()) or ():
                parts.append(f"{ev}={_os.environ.get(ev, '')}")
            bev = getattr(pcfg, "base_url_env_var", "") or ""
            if bev:
                parts.append(f"{bev}={_os.environ.get(bev, '')}")
    except Exception:
        pass

    # Bedrock's available model IDs are region-scoped (us.*, eu.*, ap.*).
    # Include the resolved region in the cache fingerprint; otherwise a fresh
    # eu-central-1 profile can reuse a still-fresh us-east-1 cache entry and
    # show the wrong regional inference profile IDs.
    if provider == "bedrock":
        try:
            from agent.bedrock_adapter import resolve_bedrock_region
            parts.append(f"bedrock_region={resolve_bedrock_region()}")
        except Exception:
            parts.append(
                "bedrock_region="
                f"{_os.environ.get('AWS_REGION', '') or _os.environ.get('AWS_DEFAULT_REGION', '')}"
            )

    # OAuth / external-file mtimes that change on re-auth
    try:
        from hermes_constants import get_hermes_home
        for rel in ("auth.json", "credentials.json"):
            p = get_hermes_home() / rel
            try:
                parts.append(f"{rel}@{p.stat().st_mtime_ns}")
            except FileNotFoundError:
                parts.append(f"{rel}@missing")
            except Exception:
                pass
    except Exception:
        pass

    # External well-known credential file locations
    for path in (
        _os.path.expanduser("~/.codex/auth.json"),
        _os.path.expanduser("~/.claude/.credentials.json"),
        _os.path.expanduser("~/.config/github-copilot/hosts.json"),
        _os.path.expanduser("~/.minimax/credentials.json"),
    ):
        try:
            mt = _os.stat(path).st_mtime_ns
            parts.append(f"{path}@{mt}")
        except FileNotFoundError:
            parts.append(f"{path}@missing")
        except Exception:
            pass

    blob = "|".join(parts).encode("utf-8", errors="replace")
    # blake2b for cache-key fingerprinting only — not for credential storage.
    # We never reverse this hash; collisions are harmless (worst case: cache
    # miss → live re-fetch). Use blake2b instead of sha256 here because
    # CodeQL's `py/weak-sensitive-data-hashing` rule flags sha256 over env
    # vars whose names contain "API_KEY" / "TOKEN" even when the hash is
    # used as an identity fingerprint, not for password storage. blake2b
    # is a keyed-hash primitive and isn't flagged.
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def _load_provider_models_cache() -> dict:
    """Return the full cache dict, or {} on any error."""
    try:
        path = _provider_models_cache_path()
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_provider_models_cache(data: dict) -> None:
    """Persist the cache dict. Best-effort — silent on any error."""
    try:
        from utils import atomic_json_write
        path = _provider_models_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, data, indent=None)
    except Exception:
        pass


def cached_provider_model_ids(
    provider: Optional[str],
    *,
    force_refresh: bool = False,
    ttl_seconds: int = _PROVIDER_MODELS_CACHE_TTL,
) -> list[str]:
    """Disk-cached wrapper around :func:`provider_model_ids`.

    Hits the cache when fresh; otherwise calls the live function and
    persists a non-empty result. Always returns a list (never None).
    """
    normalized = normalize_provider(provider) or (provider or "")
    if not normalized:
        return []

    cache = _load_provider_models_cache()
    fp = _credential_fingerprint(normalized)
    entry = cache.get(normalized)
    now = time.time()

    if (
        not force_refresh
        and isinstance(entry, dict)
        and entry.get("fp") == fp
        and isinstance(entry.get("models"), list)
        and entry["models"]
        and (now - float(entry.get("at", 0))) < ttl_seconds
    ):
        return list(entry["models"])

    # Cache miss / stale / forced refresh — call the live path.
    live = provider_model_ids(normalized, force_refresh=force_refresh)
    if live:
        cache[normalized] = {
            "fp": fp,
            "at": now,
            "models": list(live),
        }
        _save_provider_models_cache(cache)
        return list(live)

    # Live fetch returned nothing. If we have a stale entry with the
    # SAME fingerprint, prefer it over an empty result — stale data
    # beats no data when the network is flaky.
    if (
        isinstance(entry, dict)
        and entry.get("fp") == fp
        and isinstance(entry.get("models"), list)
        and entry["models"]
    ):
        return list(entry["models"])
    return list(live or [])


def clear_provider_models_cache(provider: Optional[str] = None) -> None:
    """Drop a single provider's cache entry, or wipe the whole cache.

    ``provider=None`` wipes everything; otherwise only that provider's
    entry is removed. Used by ``/model --refresh`` and
    ``hermes model --refresh``.
    """
    try:
        if provider is None:
            path = _provider_models_cache_path()
            if path.exists():
                path.unlink()
            return
        cache = _load_provider_models_cache()
        normalized = normalize_provider(provider) or provider or ""
        if normalized in cache:
            del cache[normalized]
            _save_provider_models_cache(cache)
    except Exception:
        pass


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
    is_oauth = _is_oauth_token(token)
    if is_oauth:
        headers["Authorization"] = f"Bearer {token}"
        from agent.anthropic_adapter import _COMMON_BETAS, _OAUTH_ONLY_BETAS, _CONTEXT_1M_BETA
        headers["anthropic-beta"] = ",".join(_COMMON_BETAS + _OAUTH_ONLY_BETAS)
    else:
        headers["x-api-key"] = token

    def _do_request(h: dict[str, str]):
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/models",
            headers=h,
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    try:
        try:
            data = _do_request(headers)
        except urllib.error.HTTPError as http_err:
            # Reactive recovery for OAuth subscriptions that reject the 1M
            # context beta with 400 "long context beta is not yet available
            # for this subscription". Retry once without the beta; re-raise
            # anything else so the outer except logs it.
            if (
                is_oauth
                and http_err.code == 400
            ):
                try:
                    body_text = http_err.read().decode(errors="ignore").lower()
                except Exception:
                    body_text = ""
                if "long context beta" in body_text and "not yet available" in body_text:
                    headers["anthropic-beta"] = ",".join(
                        [b for b in _COMMON_BETAS if b != _CONTEXT_1M_BETA]
                        + list(_OAUTH_ONLY_BETAS)
                    )
                    data = _do_request(headers)
                else:
                    raise
            else:
                raise
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


_COPILOT_INVENTORY_CACHE_TTL = 3600  # 1 hour


def _copilot_inventory_cache_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "copilot_inventory_cache.json"


def _load_copilot_inventory_cache() -> dict[str, Any]:
    try:
        path = _copilot_inventory_cache_path()
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_copilot_inventory_cache(data: dict[str, Any]) -> None:
    try:
        from utils import atomic_json_write
        path = _copilot_inventory_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, data, indent=None)
    except Exception:
        pass


def _copilot_catalog_source_name() -> str:
    return "githubcopilot/models"


def _copilot_model_source_coverage(raw_id: str) -> dict[str, bool]:
    lowered = raw_id.strip().lower()
    return {
        "google": lowered.startswith("google/"),
        "gemini": "gemini" in lowered,
    }


def _copilot_catalog_item_limits(item: dict[str, Any]) -> dict[str, Optional[int]]:
    """Extract separate Copilot limit fields from a catalog item."""
    capabilities = item.get("capabilities")
    limits = capabilities.get("limits") if isinstance(capabilities, dict) else {}
    limits = limits if isinstance(limits, dict) else {}

    def _limit_value(key: str) -> Optional[int]:
        value = limits.get(key)
        if isinstance(value, int) and value > 0:
            return value
        return None

    return {
        "prompt_budget": _limit_value("max_prompt_tokens"),
        "total_context_window": _limit_value("max_context_window_tokens"),
        "max_output_tokens": _limit_value("max_output_tokens"),
        "max_non_streaming_output_tokens": _limit_value(
            "max_non_streaming_output_tokens",
        ),
    }


def _sync_copilot_limit_freshness(snapshot: dict[str, Any]) -> None:
    """Keep nested Copilot limit freshness aligned with the snapshot."""
    freshness = dict(snapshot.get("freshness") or {})
    models = snapshot.get("models")
    if not isinstance(models, dict):
        return

    for model in models.values():
        if not isinstance(model, dict):
            continue

        limits = model.get("limits")
        if isinstance(limits, dict):
            limits["freshness"] = dict(freshness)

        sources = model.get("sources")
        if not isinstance(sources, list):
            continue
        for source_record in sources:
            if not isinstance(source_record, dict):
                continue
            source_limits = source_record.get("limits")
            if isinstance(source_limits, dict):
                source_limits["freshness"] = dict(freshness)


# ── Copilot catalog auth + caching ──────────────────────────────────────────
# The /models catalog REQUIRES an Authorization token; without one the endpoint
# returns 401 and capability lookups (reasoning effort, context, output) silently
# fall back to stale hardcoded values (this was the root cause of "opus stuck at
# medium"). Many internal call sites don't thread an api_key, so we auto-resolve
# one here, memoized to bound `gh` subprocess calls. Raw catalog items are cached
# per integration-id for an hour so repeated lookups don't re-hit the network.
_copilot_catalog_items_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}
_COPILOT_CATALOG_ITEMS_TTL = 3600
_copilot_catalog_token_memo: Optional[tuple[Optional[str], float]] = None
_COPILOT_CATALOG_TOKEN_TTL = 1800


def _auto_resolve_copilot_token() -> Optional[str]:
    """Best-effort resolve a Copilot API token for catalog fetches.

    Memoizes the result (including ``None``) for ``_COPILOT_CATALOG_TOKEN_TTL``
    so non-Copilot users don't repeatedly shell out to ``gh``. Never raises.

    Skipped entirely under pytest: ``gh auth token`` reads its own credential
    store (hosts.yml) which the hermetic test wrapper can't unset, so allowing
    it would make capability lookups perform live network calls during unit
    tests. Tests that exercise the catalog path mock it explicitly instead.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    global _copilot_catalog_token_memo
    now = time.time()
    if (
        _copilot_catalog_token_memo is not None
        and now - _copilot_catalog_token_memo[1] < _COPILOT_CATALOG_TOKEN_TTL
    ):
        return _copilot_catalog_token_memo[0]
    token: Optional[str] = None
    try:
        from hermes_cli.copilot_auth import (
            resolve_copilot_token,
            get_copilot_api_token,
        )

        raw, _source = resolve_copilot_token()
        token = get_copilot_api_token(raw) or None
    except Exception as exc:
        logger.debug("Copilot catalog token auto-resolve failed: %s", exc)
        token = None
    _copilot_catalog_token_memo = (token, now)
    return token


def _fetch_github_model_catalog_items(
    api_key: Optional[str] = None, timeout: float = 5.0
) -> Optional[list[dict[str, Any]]]:
    from hermes_cli.copilot_auth import _copilot_integration_id

    # Skip the in-process cache under pytest: module state persists across tests
    # in the same file (subprocess isolation is per-file), so a cached catalog
    # would shadow a test's mocked urlopen response. Tests are hermetic and fast
    # without it; production keeps the 1h cache to avoid repeated network hits.
    _use_cache = not os.environ.get("PYTEST_CURRENT_TEST")
    cache_key = _copilot_integration_id()
    if _use_cache:
        cached = _copilot_catalog_items_cache.get(cache_key)
        if cached and time.time() - cached[1] < _COPILOT_CATALOG_ITEMS_TTL:
            return cached[0]

    attempts: list[dict[str, str]] = []
    if api_key:
        attempts.append({
            **copilot_default_headers(model="catalog"),
            "Authorization": f"Bearer {api_key}",
        })
    # Last-resort unauthenticated attempt (works only if auth is injected by a
    # surrounding context, e.g. a proxy). Kept for backward compatibility.
    attempts.append(copilot_default_headers(model="catalog"))

    for headers in attempts:
        req = urllib.request.Request(COPILOT_MODELS_URL, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                items = _payload_items(data)
                if items and _use_cache:
                    _copilot_catalog_items_cache[cache_key] = (items, time.time())
                return items
        except Exception:
            continue
    return None


def build_copilot_inventory_snapshot(
    catalog: Optional[list[dict[str, Any]]],
    *,
    previous_snapshot: Optional[dict[str, Any]] = None,
    captured_at: Optional[float] = None,
) -> dict[str, Any]:
    """Normalize live Copilot catalog items into a provenance-rich snapshot."""
    source_name = _copilot_catalog_source_name()
    captured_at = float(captured_at if captured_at is not None else time.time())
    raw_items = [item for item in (catalog or []) if isinstance(item, dict)]
    prev_models: dict[str, Any] = {}
    if isinstance(previous_snapshot, dict):
        candidate = previous_snapshot.get("models")
        if isinstance(candidate, dict):
            prev_models = candidate

    raw_evidence: list[dict[str, Any]] = []
    aliases_by_model: dict[str, set[str]] = {}
    coverage_by_model: dict[str, dict[str, bool]] = {}
    limits_by_model: dict[str, dict[str, Optional[int]]] = {}
    order: list[str] = []

    for item in raw_items:
        raw_id = str(item.get("id") or "").strip()
        coverage = _copilot_model_source_coverage(raw_id)
        chat_capable = _copilot_catalog_item_is_text_model(item)
        picker_enabled = item.get("model_picker_enabled") is not False
        normalized_id = (
            normalize_copilot_model_id(raw_id, catalog=raw_items)
            if raw_id
            else ""
        )
        limit_values = _copilot_catalog_item_limits(item)
        included = bool(raw_id and normalized_id and chat_capable and picker_enabled)
        reason = "included" if included else (
            "missing_id"
            if not raw_id
            else "non_chat_or_picker_disabled"
        )
        raw_evidence.append({
            "source": source_name,
            "captured_at": captured_at,
            "raw_id": raw_id,
            "normalized_id": normalized_id,
            "included": included,
            "reason": reason,
            "chat_capable": chat_capable,
            "picker_enabled": picker_enabled,
            "google": coverage["google"],
            "gemini": coverage["gemini"],
            "limits": {
                "prompt_budget": limit_values["prompt_budget"],
                "total_context_window": limit_values["total_context_window"],
                "max_output_tokens": limit_values["max_output_tokens"],
                "max_non_streaming_output_tokens": limit_values[
                    "max_non_streaming_output_tokens"
                ],
            },
        })
        if not included:
            continue
        if normalized_id not in aliases_by_model:
            aliases_by_model[normalized_id] = set()
            coverage_by_model[normalized_id] = {"google": False, "gemini": False}
            limits_by_model[normalized_id] = dict(limit_values)
            order.append(normalized_id)
        else:
            existing_limits = limits_by_model.setdefault(normalized_id, {})
            for key, value in limit_values.items():
                if existing_limits.get(key) is None and value is not None:
                    existing_limits[key] = value
        aliases_by_model[normalized_id].add(raw_id)
        coverage_by_model[normalized_id]["google"] = (
            coverage_by_model[normalized_id]["google"] or coverage["google"]
        )
        coverage_by_model[normalized_id]["gemini"] = (
            coverage_by_model[normalized_id]["gemini"] or coverage["gemini"]
        )

    models: dict[str, Any] = {}
    for model_id in order:
        prev_first_seen = captured_at
        prev_entry = prev_models.get(model_id)
        if isinstance(prev_entry, dict):
            first_seen = prev_entry.get("first_seen")
            if isinstance(first_seen, (int, float)):
                prev_first_seen = float(first_seen)
        aliases = sorted(aliases_by_model.get(model_id, set()))
        coverage = coverage_by_model.get(model_id, {"google": False, "gemini": False})
        limit_values = limits_by_model.get(model_id, {
            "prompt_budget": None,
            "total_context_window": None,
            "max_output_tokens": None,
            "max_non_streaming_output_tokens": None,
        })
        limit_snapshot = {
            "prompt_budget": limit_values["prompt_budget"],
            "total_context_window": limit_values["total_context_window"],
            "max_output_tokens": limit_values["max_output_tokens"],
            "max_non_streaming_output_tokens": limit_values[
                "max_non_streaming_output_tokens"
            ],
            "source": source_name,
            "captured_at": captured_at,
            "raw_keys": {
                "prompt_budget": "max_prompt_tokens",
                "total_context_window": "max_context_window_tokens",
                "max_output_tokens": "max_output_tokens",
                "max_non_streaming_output_tokens": "max_non_streaming_output_tokens",
            },
        }
        source_record = {
            "source": source_name,
            "captured_at": captured_at,
            "first_seen": prev_first_seen,
            "last_seen": captured_at,
            "raw_aliases": list(aliases),
            "google": coverage["google"],
            "gemini": coverage["gemini"],
            "chat_capable": True,
            "picker_enabled": True,
            "limits": dict(limit_snapshot),
        }
        models[model_id] = {
            "id": model_id,
            "first_seen": prev_first_seen,
            "last_seen": captured_at,
            "raw_aliases": list(aliases),
            "sources": [source_record],
            "limits": dict(limit_snapshot),
            "coverage": {
                "google": coverage["google"],
                "gemini": coverage["gemini"],
                "per_source": {
                    source_name: {
                        "google": coverage["google"],
                        "gemini": coverage["gemini"],
                    }
                },
            },
            "chat_capable": True,
            "picker_enabled": True,
        }

    model_ids = list(models)
    snapshot = {
        "source": source_name,
        "captured_at": captured_at,
        "model_ids": model_ids,
        "models": models,
        "raw_evidence": raw_evidence,
        "freshness": {
            "state": "live" if model_ids else "empty",
            "checked_at": captured_at,
            "source": source_name,
            "raw_count": len(raw_evidence),
            "model_count": len(model_ids),
            "has_last_known_good": bool(prev_models),
            "used_last_known_good": False,
        },
    }
    _sync_copilot_limit_freshness(snapshot)
    return snapshot


def cached_copilot_inventory_snapshot(
    api_key: Optional[str] = None,
    *,
    force_refresh: bool = False,
    ttl_seconds: int = _COPILOT_INVENTORY_CACHE_TTL,
) -> dict[str, Any]:
    """Return the best known Copilot inventory snapshot with stale fallback."""
    cache = _load_copilot_inventory_cache()
    fp = _credential_fingerprint("copilot")
    entry = cache.get(fp) if isinstance(cache, dict) else None
    now = time.time()

    if (
        not force_refresh
        and isinstance(entry, dict)
        and entry.get("fp") == fp
        and isinstance(entry.get("snapshot"), dict)
        and (now - float(entry.get("at", 0))) < ttl_seconds
    ):
        snapshot = deepcopy(entry["snapshot"])
        freshness = dict(snapshot.get("freshness") or {})
        freshness.update({
            "state": "cached",
            "checked_at": now,
            "source": _copilot_catalog_source_name(),
            "has_last_known_good": True,
            "used_last_known_good": True,
        })
        snapshot["freshness"] = freshness
        _sync_copilot_limit_freshness(snapshot)
        return snapshot

    raw_items = _fetch_github_model_catalog_items(api_key=api_key)
    previous_snapshot = (
        deepcopy(entry.get("snapshot"))
        if isinstance(entry, dict) and isinstance(entry.get("snapshot"), dict)
        else None
    )
    live_snapshot = build_copilot_inventory_snapshot(
        raw_items or [],
        previous_snapshot=previous_snapshot,
        captured_at=now,
    )

    if live_snapshot["model_ids"]:
        live_snapshot["freshness"].update({
            "state": "live",
            "checked_at": now,
            "has_last_known_good": bool(previous_snapshot),
            "used_last_known_good": False,
        })
        _sync_copilot_limit_freshness(live_snapshot)
        cache[fp] = {
            "fp": fp,
            "at": now,
            "snapshot": deepcopy(live_snapshot),
        }
        _save_copilot_inventory_cache(cache)
        return live_snapshot

    if previous_snapshot:
        snapshot = deepcopy(previous_snapshot)
        freshness = dict(snapshot.get("freshness") or {})
        freshness.update({
            "state": "stale",
            "checked_at": now,
            "source": _copilot_catalog_source_name(),
            "has_last_known_good": True,
            "used_last_known_good": True,
        })
        snapshot["freshness"] = freshness
        _sync_copilot_limit_freshness(snapshot)
        return snapshot

    live_snapshot["freshness"].update({
        "state": "empty",
        "checked_at": now,
        "has_last_known_good": False,
        "used_last_known_good": False,
    })
    _sync_copilot_limit_freshness(live_snapshot)
    return live_snapshot


def copilot_default_headers(model: str = "") -> dict[str, str]:
    """Standard headers for Copilot API requests.

    Includes Openai-Intent and x-initiator headers that opencode and the
    Copilot CLI send on every request.
    """
    try:
        from hermes_cli.copilot_auth import copilot_request_headers
        return copilot_request_headers(is_agent_turn=True, model=model)
    except ImportError:
        # copilot_auth is the single source of truth; this fallback only fires if
        # it cannot be imported. Mirror its Copilot CLI identity shape (no
        # Editor-* VS Code headers, CLI User-Agent) so the identity stays
        # consistent even on the degraded path.
        return {
            "User-Agent": f"copilot/{_COPILOT_CLI_VERSION}",
            "Copilot-Integration-Id": _COPILOT_INTEGRATION_ID,
            "Runtime-Client-Version": _COPILOT_CLI_VERSION,
            "Openai-Intent": "conversation-panel",
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
    elif not capabilities:
        # If no supported_endpoints AND no capabilities, it's not a known text model
        return False

    return True


def fetch_github_model_catalog(
    api_key: Optional[str] = None, timeout: float = 5.0
) -> Optional[list[dict[str, Any]]]:
    """Fetch the live GitHub Copilot model catalog for this account."""
    items = _fetch_github_model_catalog_items(api_key=api_key, timeout=timeout)
    if not items:
        return None
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
    
    # Inject test/preview slugs to force availability for empirical probing.
    # Gated behind HERMES_COPILOT_INJECT_PROBE_SLUGS so it stays opt-in:
    # without the env var the catalog returns ONLY what GitHub actually
    # advertises for this account (keeps unit tests deterministic and avoids
    # surfacing unreachable slugs in the picker for users who do not probe).
    if os.environ.get("HERMES_COPILOT_INJECT_PROBE_SLUGS"):
        _probe_slugs = [
            "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview",
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
            "gemma-4-31b-it", "gemma-4-26b-a4b-it", "goldeneye-secondary",
            "claude-mythos-preview", "claude-mythos-1-preview", "claude-mythos",
        ]
        for slug in _probe_slugs:
            if slug not in seen_ids:
                models.append({
                    "id": slug,
                    "name": slug,
                    "model_picker_enabled": True,
                    "vendor": "TestInjection",
                    "capabilities": {"type": "chat"},
                })
                seen_ids.add(slug)

    if models:
        return models
    return None


# ─── Copilot catalog context-window helpers ─────────────────────────────────

# Module-level cache: {model_id: max_prompt_tokens}
_copilot_context_cache: dict[str, int] = {}
_copilot_context_cache_time: float = 0.0
_COPILOT_CONTEXT_CACHE_TTL = 3600  # 1 hour

# Verified max INPUT-token budget for Copilot preview models that the live
# /models endpoint omits non-deterministically (gemini-3.x flicker across
# load-balanced backends). Consulted ONLY as a supplement when the catalog
# lookup misses; it never overrides a value the catalog actually returned.
# Verified live 2026-06-07 (account e126380_magh): gemini-3.x enforce the
# prompt cap at the full 1M context window (like Claude), output is separate.
_COPILOT_CONTEXT_SUPPLEMENT: dict[str, int] = {
    "gemini-3.1-pro-preview": 1_000_000,
    "gemini-3.5-flash": 1_000_000,
    "gemini-3-pro-preview": 1_000_000,
    # claude-fable-5: catalog-miss fallback only. The 1.0.61 bundle clones
    # opus-4.8's config, so we model its window on opus-4.8 (1M, the enforced
    # Claude prompt cap). UNVERIFIED live (account not yet entitled); the live
    # catalog's max_context_window_tokens overrides this once Fable is enabled.
    "claude-fable-5": 1_000_000,
}


def _copilot_input_budget_from_limits(
    model_id: str, limits: dict[str, Any]
) -> Optional[int]:
    """Pick the true usable INPUT-token budget for a Copilot model.

    Verified live 2026-06-07 (account e126380_magh):
      - Claude (/v1/messages) and Gemini (/chat/completions) enforce the prompt
        cap at the FULL ``max_context_window_tokens``; output tokens are billed
        separately (opus accepted a 998,564-token prompt AND 128k output in the
        same request). Their catalog ``max_prompt_tokens`` UNDER-reports by the
        output reservation (936k = 1M − 64k), so using it wastes ~64k of usable
        context, hence we use the full window for these families.
      - GPT / o-series / codex (/responses) treat the window as a COMBINED
        input+output budget; the enforced INPUT cap is ``max_prompt_tokens``
        (gpt-5.5 rejects ~924k input though its window is 1.05M). Using the
        window would over-budget and 400 near the top, so we keep max_prompt.
    """
    window = limits.get("max_context_window_tokens")
    prompt = limits.get("max_prompt_tokens")
    mid = model_id.lower()
    window_is_input_cap = mid.startswith("claude") or mid.startswith("gemini")
    if window_is_input_cap and isinstance(window, int) and window > 0:
        return window
    if isinstance(prompt, int) and prompt > 0:
        return prompt
    if isinstance(window, int) and window > 0:
        return window
    return None


def get_copilot_model_context(model_id: str, api_key: Optional[str] = None) -> Optional[int]:
    """Return the usable INPUT-token budget for a Copilot model (live /models).

    Results are cached in-process for 1 hour to avoid repeated API calls.
    Returns the token limit or None if not found.

    Catalog-driven (matching the upstream v0.16.0 design); a previous hardcoded
    override layer here was stale and partly WRONG once the catalog/token path
    was fixed (it forced gemini-2.5-pro to 1,048,576 when the real Copilot cap
    is 128,000, and gpt-5.4 to 750,000 when max_prompt is 922,000), so it was
    removed. The per-model field selection lives in
    ``_copilot_input_budget_from_limits``: Claude/Gemini use the full
    ``max_context_window_tokens`` (their enforced prompt cap; catalog
    max_prompt UNDER-reports by the output reservation; opus is really 1M, not
    936k), while GPT/codex use ``max_prompt_tokens`` (their window is a combined
    input+output budget). Verified live 2026-06-07. The only remaining
    hardcoded layer is _COPILOT_CONTEXT_SUPPLEMENT, a catalog-miss fallback for
    the preview models the endpoint flakily omits.
    """
    global _copilot_context_cache, _copilot_context_cache_time

    # Serve from cache if fresh
    if _copilot_context_cache and (time.time() - _copilot_context_cache_time < _COPILOT_CONTEXT_CACHE_TTL):
        if model_id in _copilot_context_cache:
            return _copilot_context_cache[model_id]
        # Cache is fresh but model not in it; the catalog may have flakily
        # omitted a preview model; consult the supplement before giving up.
        return _COPILOT_CONTEXT_SUPPLEMENT.get(model_id)

    # Fetch and populate cache
    if not api_key:
        api_key = _auto_resolve_copilot_token()
    catalog = fetch_github_model_catalog(api_key=api_key)
    if not catalog:
        return _COPILOT_CONTEXT_SUPPLEMENT.get(model_id)

    cache: dict[str, int] = {}
    for item in catalog:
        mid = str(item.get("id") or "").strip()
        if not mid:
            continue
        caps = item.get("capabilities") or {}
        limits = caps.get("limits") or {}
        budget = _copilot_input_budget_from_limits(mid, limits)
        if isinstance(budget, int) and budget > 0:
            cache[mid] = budget

    _copilot_context_cache = cache
    _copilot_context_cache_time = time.time()

    # The Copilot /models endpoint is non-deterministic for preview models:
    # gemini-3.x flicker in and out across load-balanced backends, so a given
    # fetch may omit them even though inference works fine. Supplement (NOT
    # override) with verified max_prompt values so a flaky omission doesn't
    # break context budgeting. Verified live 2026-06-07 (account e126380_magh).
    return cache.get(model_id) or _COPILOT_CONTEXT_SUPPLEMENT.get(model_id)


def _is_github_models_base_url(base_url: Optional[str]) -> bool:
    normalized = (base_url or "").strip().rstrip("/").lower()
    return (
        normalized.startswith(COPILOT_BASE_URL)
        or normalized.startswith("https://models.github.ai/inference")
        or normalized.startswith("https://models.inference.ai.azure.com")
    )


def _lmstudio_server_root(base_url: Optional[str]) -> Optional[str]:
    """Strip ``/v1`` suffix from an LM Studio base URL to get the native API root.

    Returns ``None`` when the base URL is empty/invalid.
    """
    root = (base_url or "").strip().rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3].rstrip("/")
    return root or None


def _lmstudio_request_headers(api_key: Optional[str] = None) -> dict:
    """Build HTTP headers for LM Studio native API requests."""
    headers = {"User-Agent": _HERMES_USER_AGENT}
    token = str(api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _lmstudio_fetch_raw_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 5.0,
) -> Optional[list[dict]]:
    """Fetch the raw model list from LM Studio's ``/api/v1/models``.

    Returns the ``models`` list of dicts on success, ``None`` on network
    errors or malformed responses.  Raises ``AuthError`` on HTTP 401/403.
    """
    server_root = _lmstudio_server_root(base_url)
    if not server_root:
        return None

    headers = _lmstudio_request_headers(api_key)
    request = urllib.request.Request(server_root + "/api/v1/models", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            from hermes_cli.auth import AuthError
            raise AuthError(
                f"LM Studio rejected the request with HTTP {exc.code}.",
                provider="lmstudio",
                code="auth_rejected",
            ) from exc
        import logging
        logging.getLogger(__name__).debug(
            "LM Studio probe at %s failed with HTTP %s", server_root, exc.code,
        )
        return None
    except Exception as exc:
        import logging
        logging.getLogger(__name__).debug(
            "LM Studio probe at %s failed: %s", server_root, exc,
        )
        return None

    raw_models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(raw_models, list):
        import logging
        logging.getLogger(__name__).debug(
            "LM Studio probe at %s returned malformed payload (no `models` list)",
            server_root,
        )
        return None
    return raw_models


def probe_lmstudio_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 5.0,
) -> Optional[list[str]]:
    """Probe LM Studio's model listing.

    Returns chat-capable model keys on success, including the valid empty-list
    case when the server is reachable but has no non-embedding models.
    Returns ``None`` on network errors, malformed responses, or empty/invalid
    base URLs.

    Raises ``AuthError`` on HTTP 401/403 so callers can surface token issues
    separately from reachability problems.
    """
    raw_models = _lmstudio_fetch_raw_models(api_key=api_key, base_url=base_url, timeout=timeout)
    if raw_models is None:
        return None

    keys: list[str] = []
    for raw in raw_models:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("type") or "").strip().lower() == "embedding":
            continue
        key = str(raw.get("key") or raw.get("id") or "").strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def fetch_lmstudio_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 5.0,
) -> list[str]:
    """Fetch LM Studio chat-capable model keys from native ``/api/v1/models``.

    Returns a list of model keys (e.g. ``publisher/model-name``) with embedding
    models filtered out. Returns an empty list on network errors, malformed
    responses, or empty/invalid base URLs.

    Raises ``AuthError`` on HTTP 401/403 so callers can distinguish a missing
    or wrong ``LM_API_KEY`` from an unreachable server — the most common
    LM Studio support case once auth-enabled mode is turned on.
    """
    models = probe_lmstudio_models(api_key=api_key, base_url=base_url, timeout=timeout)
    return models or []


def ensure_lmstudio_model_loaded(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    target_context_length: int,
    timeout: float = 120.0,
) -> Optional[int]:
    """Ensure LM Studio has ``model`` loaded with at least ``target_context_length``.

    No-op when an instance is already loaded with sufficient context. Otherwise
    POSTs ``/api/v1/models/load`` to (re)load with the target context, capped
    at the model's ``max_context_length``. Returns the resolved loaded context
    length, or ``None`` when the probe / load failed.
    """
    server_root = _lmstudio_server_root(base_url)
    if not server_root:
        return None

    headers = _lmstudio_request_headers(api_key)

    try:
        raw_models = _lmstudio_fetch_raw_models(api_key=api_key, base_url=base_url, timeout=10)
    except Exception:
        raw_models = None
    if raw_models is None:
        return None

    target_entry = None
    for raw in raw_models:
        if not isinstance(raw, dict):
            continue
        if raw.get("key") == model or raw.get("id") == model:
            target_entry = raw
            break
    if target_entry is None:
        return None

    max_ctx = target_entry.get("max_context_length")
    if isinstance(max_ctx, int) and max_ctx > 0:
        target_context_length = min(target_context_length, max_ctx)

    for inst in target_entry.get("loaded_instances") or []:
        cfg = inst.get("config") if isinstance(inst, dict) else None
        loaded_ctx = cfg.get("context_length") if isinstance(cfg, dict) else None
        if isinstance(loaded_ctx, int) and loaded_ctx >= target_context_length:
            return loaded_ctx

    body = json.dumps({
        "model": model,
        "context_length": target_context_length,
    }).encode()
    load_headers = dict(headers)
    load_headers["Content-Type"] = "application/json"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                server_root + "/api/v1/models/load",
                data=body,
                headers=load_headers,
                method="POST",
            ),
            timeout=timeout,
        ) as resp:
            resp.read()
    except Exception:
        return None
    return target_context_length


def lmstudio_model_reasoning_options(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str] = None,
    timeout: float = 5.0,
) -> list[str]:
    """Return the reasoning ``allowed_options`` LM Studio publishes for ``model``.

    Pulls ``capabilities.reasoning.allowed_options`` from ``/api/v1/models``.
    Returns ``[]`` when the model is unknown, the endpoint is unreachable,
    or the model does not declare a reasoning capability.
    """
    try:
        raw_models = _lmstudio_fetch_raw_models(api_key=api_key, base_url=base_url, timeout=timeout)
    except Exception:
        raw_models = None
    if not raw_models:
        return []

    for raw in raw_models:
        if not isinstance(raw, dict):
            continue
        if raw.get("key") != model and raw.get("id") != model:
            continue
        caps = raw.get("capabilities")
        reasoning = caps.get("reasoning") if isinstance(caps, dict) else None
        opts = reasoning.get("allowed_options") if isinstance(reasoning, dict) else None
        if isinstance(opts, list):
            return [str(o).strip().lower() for o in opts if isinstance(o, str)]
        return []
    return []


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
    "anthropic/claude-opus-4.7": "claude-opus-4.7",
    "anthropic/claude-opus-4.8": "claude-opus-4.8",
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4.6",
    "anthropic/claude-sonnet-4": "claude-sonnet-4",
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5": "claude-haiku-4.5",
    # Friendly aliases for the restricted-preview Opus the user calls "Mythos".
    # Catalog matching below still accepts a literal `claude-mythos` id should
    # Copilot ever publish one; today this maps to the underlying 4.7 deployment.
    "mythos": "claude-opus-4.7",
    "claude-mythos": "claude-opus-4.7",
    "anthropic/claude-mythos": "claude-opus-4.7",
    # Dash-notation fallbacks: Hermes' default Claude IDs elsewhere use
    # hyphens (anthropic native format), but Copilot's API only accepts
    # dot-notation.  Accept both so users who configure copilot + a
    # default hyphenated Claude model don't hit HTTP 400
    # "model_not_supported".  See issue #6879.
    "claude-opus-4-6": "claude-opus-4.6",
    "claude-opus-4-7": "claude-opus-4.7",
    "claude-opus-4-8": "claude-opus-4.8",
    "claude-sonnet-4-6": "claude-sonnet-4.6",
    "claude-sonnet-4-0": "claude-sonnet-4",
    "claude-sonnet-4-5": "claude-sonnet-4.5",
    "claude-haiku-4-5": "claude-haiku-4.5",
    "anthropic/claude-opus-4-6": "claude-opus-4.6",
    "anthropic/claude-opus-4-7": "claude-opus-4.7",
    "anthropic/claude-opus-4-8": "claude-opus-4.8",
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

    # Claude models on Copilot ALWAYS use /v1/messages, regardless of whether
    # the live catalog probe succeeded. The /chat/completions path is a proxy
    # clamp that misleadingly reports `exceeds the limit of 168000` for any
    # claude-* prompt over ~300k. Routing every claude-* through anthropic_messages
    # before catalog probing avoids that wrong-route path when the catalog is
    # cold/empty/down or returns ambiguous supported_endpoints. See:
    #   - probe/FINDINGS.md §1: endpoint map
    #   - probe V18.1: opus-4.8 → 999,968 input tokens 200 OK on /v1/messages
    #   - upstream PR #27446: same short-circuit upstream
    if normalized.startswith("claude-"):
        return "anthropic_messages"

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


# Azure Foundry model families that require the Responses API.  Azure
# rejects /chat/completions against these deployments with
# ``400 "The requested operation is unsupported."`` — the same payload Bob
# Dobolina hit in April 2026 on ``gpt-5.3-codex`` while ``gpt-4o-pure`` on
# the same endpoint worked fine.  Keep the patterns broad enough to cover
# vendor-renamed deployments (e.g. ``gpt-5.3-codex``, ``gpt-5-codex``,
# ``gpt-5.4``, ``o1-preview``) but tight enough to leave GPT-4 / 3.5 / Llama /
# Mistral / Grok deployments on chat completions.
_AZURE_FOUNDRY_RESPONSES_PREFIXES = (
    "codex",       # codex-*, codex-mini
    "gpt-5",       # gpt-5, gpt-5.x, gpt-5-codex, gpt-5.x-codex
    "o1",          # o1, o1-preview, o1-mini
    "o3",          # o3, o3-mini
    "o4",          # o4, o4-mini
)


def azure_foundry_model_api_mode(model_name: Optional[str]) -> Optional[str]:
    """Infer Azure Foundry api_mode from a deployment/model name.

    Returns ``"codex_responses"`` when the model name matches a family that
    only accepts the Responses API on Azure Foundry (GPT-5.x, codex, o1/o3/o4
    reasoning models).  Returns ``None`` otherwise — the caller should fall
    back to the configured/default api_mode (typically ``chat_completions``)
    so GPT-4o, GPT-4 Turbo, Llama, Mistral, etc. keep working.

    Intentionally does NOT return ``anthropic_messages``; Anthropic-style
    Azure endpoints are disambiguated by URL (``/anthropic`` suffix) in
    ``runtime_provider._detect_api_mode_for_url`` and by the user setting
    ``model.api_mode: anthropic_messages`` explicitly.
    """
    raw = str(model_name or "").strip().lower()
    if not raw:
        return None
    # Strip any vendor/ prefix a user may have copied from OpenRouter / Copilot.
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    # gpt-5-mini speaks chat completions on Copilot but Azure Foundry deploys
    # the full gpt-5 family uniformly on Responses API — don't carve an
    # exception here.
    for prefix in _AZURE_FOUNDRY_RESPONSES_PREFIXES:
        if raw.startswith(prefix):
            return "codex_responses"
    return None


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
        if normalized.startswith("qwen3.7-max"):
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
    else:
        # api_key may be None here; auto-resolve a Copilot token so this works
        # on call paths that don't thread one (the common case, and the root
        # cause of Claude effort being stuck at the offline fallback). Callers
        # are all Copilot/GitHub-Models gated, so this never fires for other
        # providers. Skipped under pytest (see _auto_resolve_copilot_token).
        fetched_catalog = fetch_github_model_catalog(
            api_key=api_key or _auto_resolve_copilot_token()
        )
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


def _strip_ollama_cloud_suffix(model_id: str) -> str:
    """Strip :cloud / -cloud suffixes that models.dev appends to Ollama Cloud IDs.

    The live API uses clean IDs (e.g. 'kimi-k2.6') while models.dev sometimes
    returns them as 'kimi-k2.6:cloud'. Normalising before the dedup merge
    prevents duplicate entries in the merged model list.
    """
    for suffix in (":cloud", "-cloud"):
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return model_id


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
            normalized = _strip_ollama_cloud_suffix(m)
            if normalized and normalized not in seen:
                seen.add(normalized)
                merged.append(normalized)
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

    if normalized == "lmstudio":
        from hermes_cli.auth import AuthError
        # Use probe_lmstudio_models so we can distinguish None (unreachable
        # / malformed response) from [] (reachable, but no chat-capable models
        # are loaded). fetch_lmstudio_models collapses both to [].
        try:
            models = probe_lmstudio_models(api_key=api_key, base_url=base_url)
        except AuthError as exc:
            return {
                "accepted": False, "persist": False, "recognized": False,
                "message": (
                    f"{exc} Set `LM_API_KEY` (or update it) to match the server's bearer token."
                ),
            }
        if models is None:
            return {
                "accepted": False, "persist": False, "recognized": False,
                "message": f"Could not reach LM Studio's `/api/v1/models` to validate `{requested}`.",
            }
        if not models:
            return {
                "accepted": False, "persist": False, "recognized": False,
                "message": (
                    f"LM Studio is reachable but no chat-capable models are loaded. "
                    f"Load `{requested}` in LM Studio (Developer tab → Load Model) and try again."
                ),
            }
        if requested_for_lookup in set(models):
            return {"accepted": True, "persist": True, "recognized": True, "message": None}
        return {
            "accepted": False, "persist": False, "recognized": False,
            "message": f"Model `{requested}` was not found in LM Studio's model listing.",
        }

    # Antigravity CLI (`agy-cli`): no HTTP /models endpoint; the model list is
    # owned by the agy binary's internal Connect-RPC server and the plugin's
    # `AGY_SLUG_TO_DISPLAY` map. Use the plugin's own model list as the
    # validation source rather than letting the generic /models probe fail
    # and emit a misleading "could not reach the agy-cli API" warning.
    if normalized == "agy-cli" or normalized in {"agy", "antigravity", "antigravity-cli"}:
        try:
            import importlib.util as _ilu
            from pathlib import Path as _Path
            _plugin_init = _Path(__file__).resolve().parent.parent / "plugins" / "model-providers" / "agy-cli" / "__init__.py"
            _spec = _ilu.spec_from_file_location("_agy_plugin_validate", _plugin_init)
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _slug_map = getattr(_mod, "AGY_SLUG_TO_DISPLAY", None) or {}
            agy_known = [s for s in _slug_map.keys() if s and s != "default"]
        except Exception:
            agy_known = []
        if agy_known:
            if requested_for_lookup in set(agy_known):
                return {"accepted": True, "persist": True, "recognized": True, "message": None}
            auto = get_close_matches(requested_for_lookup, agy_known, n=1, cutoff=0.85)
            if auto:
                return {
                    "accepted": True, "persist": True, "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }
            suggestions = get_close_matches(requested_for_lookup, agy_known, n=3, cutoff=0.5)
            suggestion_text = "\n  Similar agy models: " + ", ".join(f"`{s}`" for s in suggestions) if suggestions else ""
            # Allow but warn: agy may have added new models since the plugin
            # was last refreshed; the AgyCliClient will surface the real
            # error if the model is genuinely invalid.
            return {
                "accepted": True, "persist": True, "recognized": False,
                "message": (
                    f"Note: `{requested}` is not in the pinned agy-cli model list."
                    f"{suggestion_text}"
                    f"\n  Hermes will still send the request. If the model exists on this account's agy install it will work."
                ),
            }
        # plugin model list unavailable; accept silently rather than nag with
        # a misleading "could not reach the API" warning.
        return {"accepted": True, "persist": True, "recognized": False, "message": None}

    if normalized == "custom" or normalized.startswith("custom:"):
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

    # Providers with non-standard catalog validation — /v1/models probing is not the right path.
    if normalized in {"openai-codex", "xai-oauth"}:
        try:
            catalog_models = provider_model_ids(normalized)
        except Exception:
            catalog_models = []
        if catalog_models:
            if requested_for_lookup in set(catalog_models):
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }
            # Auto-correct if the top match is very similar (e.g. typo)
            auto = get_close_matches(requested_for_lookup, catalog_models, n=1, cutoff=0.9)
            if auto:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }
            suggestions = get_close_matches(requested_for_lookup, catalog_models, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)
            provider_label = "OpenAI Codex" if normalized == "openai-codex" else "xAI Grok OAuth (SuperGrok / Premium+)"
            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": (
                    f"Note: `{requested}` was not found in the {provider_label} model listing. "
                    "It may still work if your account has access to a newer or hidden model ID."
                    f"{suggestion_text}"
                ),
            }

    # MiniMax providers don't expose a /models endpoint — validate against
    # the static catalog instead, similar to openai-codex.
    if normalized in {"minimax", "minimax-cn"}:
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
    # pattern as the openai-codex and minimax branches above.  This keeps
    # /model switches working in the gateway for providers whose /models
    # endpoint is temporarily unreachable or returns a non-JSON payload.
    # Without this block, validate_requested_model would reject every model
    # on such providers, switch_model() would return success=False, and
    # the gateway would never write to _session_model_overrides.
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
