"""Static model catalogue policy tables (data only — no network).

Curated per-provider model lists and model-parser alias policy. Provider
identity and declarations are owned by :mod:`providers`; live picker projection lives in
:mod:`hermes_cli.provider_catalog`.
"""

from __future__ import annotations

# Fallback OpenRouter snapshot used when the live catalog is unavailable, as
# ``(model_id, description shown in menus)``. ``:free`` SKUs are described "free".
_OPENROUTER_DESCRIPTIONS = {
    "anthropic/claude-opus-5-fast": "2x price, higher output speed",
    "anthropic/claude-opus-4.8-fast": "2x price, higher output speed",
    "deepseek/deepseek-v4-pro-0813": "dated snapshot of v4-pro",
    "deepseek/deepseek-v4-flash-0731": "dated snapshot of v4-flash",
    "moonshotai/kimi-k3": "recommended",
    "z-ai/glm-5.2": "default",
    "z-ai/glm-5.3-flashx": "high-speed tier of glm-5.3-flash",
    "openrouter/pareto-code": "auto-routes to cheapest coder meeting openrouter.min_coding_score",
    "openai/gpt-6-astra-fast": "2x price, priority tier",
    "openai/gpt-6-astra-flex": "0.5x price, flex tier",
    "openai/gpt-6-astra-pro-fast": "2x price, priority tier",
    "openai/gpt-6-astra-pro-flex": "0.5x price, flex tier",
    "stealth/union-alpha": "free, stealth model",
}
OPENROUTER_MODELS: list[tuple[str, str]] = [
    (mid, _OPENROUTER_DESCRIPTIONS.get(mid, "free" if mid.endswith(":free") else ""))
    for mid in (
        "anthropic/claude-fable-5.1", "anthropic/claude-fable-5", "anthropic/claude-opus-5.5",
        "anthropic/claude-opus-5", "anthropic/claude-opus-5-fast", "anthropic/claude-opus-4.8", "anthropic/claude-opus-4.8-fast",
        "anthropic/claude-sonnet-5", "anthropic/claude-haiku-4.5", "openai/gpt-6-astra", "openai/gpt-6-astra-fast",
        "openai/gpt-6-astra-flex", "openai/gpt-6-astra-pro", "openai/gpt-6-astra-pro-fast", "openai/gpt-6-astra-pro-flex",
        "openai/gpt-6-sol", "openai/gpt-6-sol-pro",
        "openai/gpt-6-luna", "openai/gpt-6-luna-pro",
        "openai/gpt-5.5", "openai/gpt-5.5-pro", "openai/gpt-5.4-mini", "google/gemini-3.1-pro-preview",
        "google/gemini-3.8-flash", "google/gemini-3.7-flash", "x-ai/grok-4.7", "x-ai/grok-4.6",
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-pro-0813", "deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4-flash-0731",
        "qwen/qwen3.8-max-0902", "qwen/qwen3.8-flash", "moonshotai/kimi-k3", "minimax/minimax-m3", "z-ai/glm-5.3",
        "z-ai/glm-5.3-flash", "z-ai/glm-5.3-flashx", "z-ai/glm-5.2",
        "xiaomi/mimo-v2.6-pro", "xiaomi/mimo-v2.6-flash", "xiaomi/mimo-v2.6-pro-ultraspeed", "xiaomi/mimo-v2.5-pro", "tencent/hy4-preview",
        "tencent/hy3",
        "stepfun/step-3.7-flash", "nvidia/nemotron-3-super-120b-a12b", "meta/muse-spark-1.2",
        "meta/muse-spark-1.2-contributor", "meta/muse-spark-1.3", "meta/muse-spark-1.3-contributor", "sakana/fugu-ultra",
        "openrouter/pareto-code", "thinkingmachines/inkling:free", "thinkingmachines/inkling-small:free",
        "minimax/minimax-m3:free", "z-ai/glm-5.2:free", "poolside/laguna-s-2.1:free", "poolside/laguna-xs-2.1:free",
        "nvidia/nemotron-3-super-120b-a12b:free", "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia/nemotron-3.5-lightning:free", "stealth/union-alpha",
    )
]

# OpenRouter entries the Nous Portal does not carry (routing/fast variants, free tier —
# ``stealth/union-alpha`` is a $0 stealth SKU without the ``:free`` suffix).
_OPENROUTER_ONLY = {
    "anthropic/claude-opus-5-fast", "anthropic/claude-opus-4.8-fast", "meta/muse-spark-1.2",
    "meta/muse-spark-1.2-contributor", "meta/muse-spark-1.3", "meta/muse-spark-1.3-contributor", "openrouter/pareto-code",
    "stealth/union-alpha",
}


# Fallback Vercel AI Gateway snapshot (open-weight first, then closed-source by family). Slugs
# match Vercel's /v1/models catalog (``alibaba/`` for Qwen, ``zai/`` and ``xai/`` without hyphens).
VERCEL_AI_GATEWAY_MODELS: list[tuple[str, str]] = [("moonshotai/kimi-k2.6", "recommended")] + [
    (mid, "") for mid in (
        "alibaba/qwen3.6-plus", "zai/glm-5.1", "minimax/minimax-m2.7", "anthropic/claude-sonnet-4.6",
        "anthropic/claude-opus-4.7", "anthropic/claude-opus-4.6", "anthropic/claude-haiku-4.5",
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.3-codex", "google/gemini-3.1-pro-preview",
        "google/gemini-3-flash", "google/gemini-3.1-flash-lite-preview", "xai/grok-4.20-reasoning",
    )
]


def _codex_curated_models() -> list[str]:
    """Canonical offline Codex catalogue, including forward/context variants."""
    from models.codex_catalog import curated_codex_models

    return curated_codex_models()


# Static xAI fallback when the models.dev disk cache is empty (fresh install, offline first run).
# Mirrors the xAI-direct IDs from $HERMES_HOME/models_dev_cache.json; the cache overrides it on the
# next refresh. Models xAI retired on 2026-05-15 (grok-4*, grok-4-fast*, grok-4-1-fast*,
# grok-code-fast-1) are excluded — see docs.x.ai/developers/migration/may-15-retirement.
_XAI_STATIC_FALLBACK: list[str] = [
    "grok-4.6", "grok-build-0.1", "grok-4.5", "grok-4.3", "grok-4.20-0309-reasoning",
    "grok-4.20-0309-non-reasoning", "grok-4.20-multi-agent-0309",
]

# Callable via xAI OAuth but omitted from models.dev and /v1/models listings. grok-4.6 / grok-4.5
# stay here until the models.dev disk cache refreshes.
_XAI_CURATED_EXTRAS: list[str] = ["grok-4.6", "grok-4.5", "grok-composer-2.5-fast"]

_XAI_TOP_MODEL = "grok-4.6"


def _xai_promote_top(ids: list[str]) -> list[str]:
    """Pin the headline xAI model to the top of the curated list."""
    if _XAI_TOP_MODEL in ids:
        return [_XAI_TOP_MODEL] + [m for m in ids if m != _XAI_TOP_MODEL]
    return ids


def _xai_merge_curated_extras(ids: list[str]) -> list[str]:
    """Append Hermes-curated xAI models missing from models.dev, right after the pinned headline."""
    out = list(ids)
    for extra in _XAI_CURATED_EXTRAS:
        if extra not in out:
            out.insert(1 if out and out[0] == _XAI_TOP_MODEL else len(out), extra)
    return out


def _xai_finalize_catalog(ids: list[str]) -> list[str]:
    return _xai_promote_top(_xai_merge_curated_extras(ids))


def _xai_curated_models() -> list[str]:
    """Offline curated floor for xAI / xAI OAuth pickers: $HERMES_HOME/models_dev_cache.json
    (no network), else ``_XAI_STATIC_FALLBACK``. Any failure falls through to the static list."""
    try:
        from models.models_dev_cache import load_models_dev_disk_cache
        data = load_models_dev_disk_cache()
        xai = data.get("xai") if isinstance(data, dict) else None
        models = xai.get("models") if isinstance(xai, dict) else None
        if isinstance(models, dict) and models:
            ids = [mid for mid in models if isinstance(mid, str)]
            if ids:
                return _xai_finalize_catalog(sorted(ids))
    except Exception:
        pass
    return _xai_finalize_catalog(list(_XAI_STATIC_FALLBACK))


# Native OpenAI Chat Completions (api.openai.com); also the head of the Copilot list.
_OPENAI_CHAT_MODELS = [
    "gpt-5.4", "gpt-5.4-mini", "gpt-5-mini", "gpt-5.3-codex", "gpt-5.2-codex", "gpt-4.1", "gpt-4o", "gpt-4o-mini",
]
_MINIMAX_MODELS = ["MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1", "MiniMax-M2"]
_TENCENT_MODELS = ["hy4-preview", "hy3", "hy3-preview"]
# Alibaba DashScope Coding platform (coding-intl): Qwen + third-party (GLM, Kimi, MiniMax, DeepSeek).
# Classic DashScope keys should override DASHSCOPE_BASE_URL to
# https://dashscope-intl.aliyuncs.com/compatible-mode/v1 (OpenAI-compat) or /apps/anthropic.
_ALIBABA_MODELS = [
    "qwen3.8-max", "qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus", "qwen3.6-flash", "kimi-k2.5",
    "qwen3.5-plus", "qwen3-coder-plus", "qwen3-coder-next", "glm-5.2", "glm-5", "glm-4.7",
    "deepseek-v4-pro", "deepseek-v4-flash-0731", "MiniMax-M2.5",
]
_ALIBABA_CODING_PLAN_MODELS = [
    "qwen3.7-plus", "qwen3.6-plus", "qwen3.5-plus", "qwen3-max-2026-01-23", "qwen3-coder-plus",
    "qwen3-coder-next", "kimi-k2.5", "glm-5", "glm-4.7", "MiniMax-M2.5",
]
# Verified against a live Token Plan subscription (key tier ``sk-sp-...``).
_ALIBABA_TOKEN_PLAN_MODELS = [
    "qwen3.8-max-0902", "qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus", "qwen3.6-flash", "deepseek-v4-pro",
    "deepseek-v4-flash", "deepseek-v3.2", "kimi-k2.7-code", "kimi-k2.6", "kimi-k2.5", "glm-5.2", "glm-5.1", "glm-5",
]
_XAI_MODELS = _xai_curated_models()

# Curated per-provider lists. ``-cn`` twins share the international catalog on a domestic endpoint.
_PROVIDER_MODELS: dict[str, list[str]] = {
    "moa": ["default"],
    "nous": [mid for mid, _ in OPENROUTER_MODELS if mid not in _OPENROUTER_ONLY and not mid.endswith(":free")],
    # Used by /model counts and provider_model_ids fallback when /v1/models is unavailable.
    "openai": list(_OPENAI_CHAT_MODELS),
    "openai-api": [
        "gpt-6-sol", "gpt-6-sol-pro", "gpt-6-luna", "gpt-6-luna-pro",
        "gpt-5.6-sol", "gpt-5.6-sol-pro", "gpt-5.6-terra", "gpt-5.6-terra-pro", "gpt-5.6-luna",
        "gpt-5.6-luna-pro", "gpt-5.5", "gpt-5.5-pro", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano",
        "gpt-5-mini", "gpt-5.3-codex", "gpt-4.1", "gpt-4o", "gpt-4o-mini",
    ],
    "openai-codex": _codex_curated_models(),
    "xai-oauth": list(_XAI_MODELS),
    "copilot-acp": ["copilot-acp"],
    "copilot": _OPENAI_CHAT_MODELS + [
        "claude-sonnet-4.6", "claude-sonnet-5", "claude-sonnet-4", "claude-sonnet-4.5", "claude-haiku-4.5",
        "gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-pro",
    ],
    "gemini": [
        "gemini-3.8-flash", "gemini-3.7-flash",
        "gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-3.6-flash", "gemini-3.1-flash-lite-preview",
    ],
    "zai": [
        "glm-5.3", "glm-5.3-flash", "glm-5.2", "glm-5.1", "glm-5", "glm-5v-turbo", "glm-5-turbo",
        "glm-4.7", "glm-4.5", "glm-4.5-flash",
    ],
    "xai": list(_XAI_MODELS),
    # Nemotron flagships, then third-party agentic models hosted on build.nvidia.com.
    "nvidia": [
        "nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/nemotron-3.5-lightning-30b-a3b", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "z-ai/glm-5.3", "z-ai/glm-5.2", "moonshotai/kimi-k2.6", "minimaxai/minimax-m3",
    ],
    "kimi-coding": [
        "kimi-k3", "kimi-k2.7-code", "kimi-k2.6", "kimi-k2.5", "kimi-for-coding", "kimi-for-coding-highspeed",
        "kimi-k2-thinking", "kimi-k2-thinking-turbo", "kimi-k2-turbo-preview", "kimi-k2-0905-preview",
    ],
    "kimi-coding-cn": [
        "kimi-k3", "kimi-k2.7-code", "kimi-k2.7-code-highspeed", "kimi-k2.6", "kimi-k2.5",
        "kimi-k2-thinking", "kimi-k2-turbo-preview", "kimi-k2-0905-preview",
    ],
    "stepfun": ["step-3.5-flash", "step-3.5-flash-2603"],
    "moonshot": [
        "kimi-k3", "kimi-k2.6", "kimi-k2.5", "kimi-k2-thinking", "kimi-k2-turbo-preview", "kimi-k2-0905-preview",
    ],
    "minimax": list(_MINIMAX_MODELS),
    "minimax-oauth": ["MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed"],
    "minimax-cn": list(_MINIMAX_MODELS),
    "anthropic": [
        "claude-fable-5.1", "claude-fable-5", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5",
        "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
        "claude-sonnet-4-6", "claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929",
        "claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-haiku-4-5-20251001",
    ],
    "deepseek": ["deepseek-flash", "deepseek-v4-pro"],
    "xiaomi": [
        "mimo-v2.6-pro", "mimo-v2.6-flash", "mimo-v2.6-pro-ultraspeed",
        "mimo-v2.5-pro", "mimo-v2.5", "mimo-v2-pro", "mimo-v2-omni", "mimo-v2-flash",
    ],
    "tencent-tokenhub": list(_TENCENT_MODELS),
    "tencent-tokenplan": list(_TENCENT_MODELS),
    "arcee": ["trinity-large-thinking", "trinity-large-preview", "trinity-mini"],
    "gmi": [
        "zai-org/GLM-5.1-FP8", "deepseek-ai/DeepSeek-V3.2", "moonshotai/Kimi-K2.5",
        "google/gemini-3.1-flash-lite-preview", "anthropic/claude-sonnet-5",
        "anthropic/claude-sonnet-4.6", "openai/gpt-5.4",
    ],
    # Synced against opencode.ai/docs/zen + live GET /zen/v1/models. Zen/Go are
    # _LIVE_FIRST_PICKER_PROVIDERS, so this is a discovery floor: live entries lead in the picker
    # and stale curated names never pollute the top. "x-preview-f-free" = "Ox Alpha" stealth model.
    "opencode-zen": [
        "x-preview-f-free", "kimi-k3", "kimi-k2.5", "kimi-k2.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
        "gpt-5.5", "gpt-5.5-pro", "gpt-5.4-pro", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.3-codex",
        "gpt-5.3-codex-spark", "gpt-5.2", "gpt-5.2-codex", "gpt-5.1", "gpt-5.1-codex", "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini", "gpt-5", "gpt-5-codex", "gpt-5-nano", "claude-fable-5", "claude-opus-5",
        "claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5",
        "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-sonnet-4", "claude-haiku-4-5", "gemini-3.8-flash",
        "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-pro", "gemini-3-flash",
        "grok-4.6", "grok-4.5", "grok-build-0.1", "muse-spark-1.2", "minimax-m3", "minimax-m2.7", "minimax-m2.5",
        "glm-5.3", "glm-5.3-flash", "glm-5.2", "glm-5.1", "glm-5", "kimi-k2.7-code", "deepseek-v4-pro",
        "deepseek-v4-flash", "qwen3.6-plus", "qwen3.5-plus", "big-pickle", "mimo-v2.5-free",
        "nemotron-3-ultra-free", "nemotron-3.5-lightning-free",
        "muse-spark-1.2-contributor-free", "muse-spark-1.3-contributor-free",
    ],
    # Synced against opencode.ai/docs/go + live GET /zen/go/v1/models. Known-delisted models are
    # REMOVED (the live-first merge would otherwise keep offering a model that 401s): "ox-alpha-free"
    # — the Go-subscription twin of Zen's Ox Alpha — was delisted 2026-09-09.
    "opencode-go": [
        "kimi-k3", "kimi-k2.7-code", "kimi-k2.6", "kimi-k2.5", "gpt-5.6-luna", "grok-4.5", "glm-5.3",
        "glm-5.3-flash", "glm-5.2", "glm-5.1", "glm-5", "mimo-v2.5-pro", "mimo-v2.5", "mimo-v2-pro",
        "mimo-v2-omni", "minimax-m3", "minimax-m2.7", "minimax-m2.5", "deepseek-v4-pro",
        "deepseek-v4-flash", "qwen3.8-max", "qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus",
        "qwen3.5-plus", "hy3", "hy3-preview", "muse-spark-1.2-contributor", "muse-spark-1.3-contributor",
    ],
    "kilocode": [
        "anthropic/claude-opus-4.6", "anthropic/claude-sonnet-4.6", "openai/gpt-5.4",
        "google/gemini-3-pro-preview", "google/gemini-3-flash-preview",
    ],
    "alibaba": list(_ALIBABA_MODELS),
    "alibaba-cn": list(_ALIBABA_MODELS),
    "alibaba-coding-plan": list(_ALIBABA_CODING_PLAN_MODELS),
    "alibaba-coding-plan-cn": list(_ALIBABA_CODING_PLAN_MODELS),
    "alibaba-token-plan": list(_ALIBABA_TOKEN_PLAN_MODELS),
    "alibaba-token-plan-cn": list(_ALIBABA_TOKEN_PLAN_MODELS),
    # Only agentic HF models that map to OpenRouter defaults.
    "huggingface": [
        "moonshotai/Kimi-K2.5", "Qwen/Qwen3.5-397B-A17B", "Qwen/Qwen3.5-35B-A3B",
        "deepseek-ai/DeepSeek-V3.2", "MiniMaxAI/MiniMax-M2.5", "zai-org/GLM-5",
        "XiaomiMiMo/MiMo-V2-Flash", "moonshotai/Kimi-K2-Thinking", "moonshotai/Kimi-K2.6",
    ],
    # Static fallback when live discovery (ListFoundationModels + ListInferenceProfiles) is
    # unavailable. Inference-profile IDs (us.*) because most models require them.
    "bedrock": [
        # [0] is the provider default (select_provider_default) — keep the cheaper Sonnet there.
        "us.anthropic.claude-sonnet-5", "us.anthropic.claude-opus-5-5", "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-opus-4-6-v1",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0", "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "openai.gpt-5.5", "openai.gpt-5.6-sol", "openai.gpt-5.6-terra", "openai.gpt-5.6-luna",
        "us.amazon.nova-pro-v1:0", "us.amazon.nova-lite-v1:0", "us.amazon.nova-micro-v1:0", "deepseek.v3.2",
        "us.meta.llama4-maverick-17b-instruct-v1:0", "us.meta.llama4-scout-17b-instruct-v1:0",
    ],
    # Azure Foundry models depend on the user's endpoint configuration.
    "azure-foundry": [],
    # Vertex's OpenAI-compatible endpoint has no /models route, so without this the /model picker
    # only shows the configured model. IDs carry the "google/" publisher prefix Vertex expects
    # (see hermes_cli/model_setup_flows.py); validated live against a GCP project (global region).
    "vertex": [
        "google/gemini-3.8-flash", "google/gemini-3.7-flash",
        "google/gemini-3.1-pro-preview", "google/gemini-3-pro-preview", "google/gemini-3.6-flash",
        "google/gemini-3.5-flash", "google/gemini-3.5-flash-lite", "google/gemini-3-flash-preview",
        "google/gemini-3.1-flash-lite-preview", "google/gemini-3.1-flash-lite",
    ],
    "novita": [
        "moonshotai/kimi-k2.5", "minimax/minimax-m2.7", "zai-org/glm-5", "deepseek/deepseek-v3-0324",
        "deepseek/deepseek-r1-0528", "qwen/qwen3-235b-a22b-fp8",
    ],
    # Bare ids derived from the picker snapshot so both stay in sync.
    "ai-gateway": [mid for mid, _ in VERCEL_AI_GATEWAY_MODELS],
}


# Offline/fresh-install fallback for the model Hermes silently lands on when the user never picked
# one (GUI onboarding confirm card, empty ``model.default``, provider-set-but-model-missing). The
# AUTHORITATIVE source is the remote catalog manifest, which labels exactly one entry per provider
# ``"default": true`` (get_default_model_from_cache) so the default rotates without a release; this
# MUST match the labeled entry in website/static/api/model-catalog.json. Deliberately a capable
# low-cost model rather than the curated lists' entry [0]: aggregator lists are ordered
# most-capable-first, so [0] is the priciest Anthropic flagship.
PREFERRED_SILENT_DEFAULT_MODEL = "z-ai/glm-5.2"


# Providers whose *silent* auto-default goes through the cost-safe catalog-labeled default
# (``preferred_silent_default_model``) instead of curated entry [0]. Metered aggregators order
# best-first, so [0] is the priciest flagship; a profile that sets a provider with no model would
# otherwise silently bill the most expensive model (863 Opus requests before one user noticed).
# Network-free (cache-only) on purpose — this is the hot resolution path. The *interactive* default
# (GUI onboarding / ``hermes model``) uses the tier-aware ``get_recommended_default_model`` in
# hermes_cli/web_server.py + ``partition_nous_models_by_tier``, which may hit the Portal.
_SILENT_DEFAULT_PROVIDERS: frozenset[str] = frozenset({"nous", "openrouter"})


def static_provider_model_ids(provider: str) -> tuple[str, ...]:
    """Curated candidate IDs for provider-owned normalization; no discovery or network I/O."""
    from providers import normalize_provider

    return tuple(_PROVIDER_MODELS.get(normalize_provider(provider), ()))


def static_provider_default_preference(provider: str) -> str:
    """Offline preferred default for providers whose curated order is not billing-safe."""
    from providers import normalize_provider

    return (
        PREFERRED_SILENT_DEFAULT_MODEL
        if normalize_provider(provider) in _SILENT_DEFAULT_PROVIDERS
        else ""
    )


def find_static_provider_model_id(provider: str, model_name: str) -> str | None:
    """Match an exact or bare model ID against the provider's offline catalogue."""
    from providers import normalize_provider

    canonical = normalize_provider(provider)
    if canonical == "openrouter":
        ids = tuple(model_id for model_id, _ in OPENROUTER_MODELS)
    else:
        ids = static_provider_model_ids(canonical)
    wanted = str(model_name or "").strip().lower()
    if not wanted:
        return None
    return (
        next((model_id for model_id in ids if wanted == model_id.lower()), None)
        or next(
            (
                model_id
                for model_id in ids
                if "/" in model_id and wanted == model_id.split("/", 1)[1].lower()
            ),
            None,
        )
    )


# Subscription/OAuth providers whose catalogs RE-EXPOSE other vendors' models; tried only as a last
# resort for bare short-alias resolution (after every native-vendor catalog) so they never hijack
# an alias from the model's native vendor. None currently defined.
_BORROWED_MODEL_PROVIDERS: frozenset[str] = frozenset()


# Providers whose live /v1/models is the authoritative catalog: the picker merges live-first (live
# entries lead, curated-only append). Every OTHER provider keeps curated-first so a deliberately
# surfaced newest model stays on top when the live API lags. Zen/Go re-expose dozens of vendors
# and rotate them often, so their stale curated entries must not pollute the top.
_LIVE_FIRST_PICKER_PROVIDERS: frozenset[str] = frozenset({"opencode-zen", "opencode-go", "meta-ai"})


# Models supporting OpenAI Priority Processing (service_tier="priority"; see
# openai.com/api-priority-processing). Pattern-based: any OpenAI flagship (gpt-*, o1*, o3*, o4*).
# Non-OpenAI endpoints (OpenRouter/Copilot/opencode-zen proxies) strip service_tier, so false
# positives are harmless. Codex-series models are excluded — the Codex Responses API doesn't
# expose service_tier.


# Providers where models.dev is authoritative: the curated list is an offline fallback plus custom
# additions the registry lacks, merged fresh-first (curated-only names appended) for both the CLI
# and the gateway /model picker. DELIBERATELY EXCLUDED: "openrouter" (curated list is a hand-picked
# agentic subset of 400+ models — merging would dump everything), "nous" (curated list + Portal
# /models are the subscription-tier source of truth), and providers with dedicated live-endpoint
# branches (copilot, anthropic, ai-gateway, ollama-cloud, custom, stepfun, openai-codex).
_MODELS_DEV_PREFERRED: frozenset[str] = frozenset({
    "opencode-go", "opencode-zen", "kilocode", "fireworks", "mistral", "togetherai", "cohere",
    "perplexity", "groq", "nvidia", "huggingface", "zai", "gemini", "google", "xai", "xai-oauth",
})


# Azure Foundry model families that require the Responses API: Azure rejects /chat/completions
# against them with ``400 "The requested operation is unsupported."`` (seen on gpt-5.3-codex while
# gpt-4o on the same endpoint worked). Broad enough for vendor-renamed deployments (gpt-5.x-codex,
# o1-preview), tight enough to leave GPT-4 / 3.5 / Llama / Mistral / Grok on chat completions.
_AZURE_FOUNDRY_RESPONSES_PREFIXES = ("codex", "gpt-5", "o1", "o3", "o4")
