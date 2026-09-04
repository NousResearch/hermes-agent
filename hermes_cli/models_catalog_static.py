"""Static provider/model catalog tables (data only — no network).

Curated per-provider model lists, the canonical provider registry, display groups and alias
maps. Split out of ``hermes_cli.models``.
"""

from __future__ import annotations

from typing import NamedTuple


# Fallback OpenRouter snapshot used when the live catalog is unavailable, as
# ``(model_id, description shown in menus)``. ``:free`` SKUs are described "free".
_OPENROUTER_DESCRIPTIONS = {
    "anthropic/claude-opus-5-fast": "2x price, higher output speed",
    "anthropic/claude-opus-4.8-fast": "2x price, higher output speed",
    "deepseek/deepseek-v4-pro-0813": "dated snapshot of v4-pro",
    "deepseek/deepseek-v4-flash-0731": "dated snapshot of v4-flash",
    "moonshotai/kimi-k3": "recommended",
    "z-ai/glm-5.2": "default",
    "openrouter/pareto-code": "auto-routes to cheapest coder meeting openrouter.min_coding_score",
}
OPENROUTER_MODELS: list[tuple[str, str]] = [
    (mid, _OPENROUTER_DESCRIPTIONS.get(mid, "free" if mid.endswith(":free") else ""))
    for mid in (
        "anthropic/claude-fable-5.1", "anthropic/claude-fable-5", "anthropic/claude-opus-5",
        "anthropic/claude-opus-5-fast", "anthropic/claude-opus-4.8", "anthropic/claude-opus-4.8-fast",
        "anthropic/claude-sonnet-5", "anthropic/claude-haiku-4.5", "openai/gpt-5.6-sol", "openai/gpt-5.6-sol-pro",
        "openai/gpt-5.6-terra", "openai/gpt-5.6-terra-pro", "openai/gpt-5.6-luna", "openai/gpt-5.6-luna-pro",
        "openai/gpt-5.5", "openai/gpt-5.5-pro", "openai/gpt-5.4-mini", "google/gemini-3.1-pro-preview",
        "google/gemini-3.8-flash", "google/gemini-3.7-flash", "x-ai/grok-4.6", "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-pro-0813", "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-flash-0731",
        "qwen/qwen3.8-max", "qwen/qwen3.8-flash", "moonshotai/kimi-k3", "minimax/minimax-m3", "z-ai/glm-5.3",
        "z-ai/glm-5.3-flash", "z-ai/glm-5.2", "xiaomi/mimo-v2.5-pro", "tencent/hy4-preview", "tencent/hy3",
        "stepfun/step-3.7-flash", "nvidia/nemotron-3-super-120b-a12b", "meta/muse-spark-1.2",
        "meta/muse-spark-1.2-contributor", "meta/muse-spark-1.3", "meta/muse-spark-1.3-contributor", "sakana/fugu-ultra",
        "openrouter/pareto-code", "thinkingmachines/inkling:free", "thinkingmachines/inkling-small:free",
        "minimax/minimax-m3:free", "z-ai/glm-5.2:free", "poolside/laguna-s-2.1:free", "poolside/laguna-xs-2.1:free",
        "nvidia/nemotron-3-super-120b-a12b:free", "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia/nemotron-3.5-lightning:free",
    )
]

# OpenRouter entries the Nous Portal does not carry (routing/fast variants, free tier).
_OPENROUTER_ONLY = {
    "anthropic/claude-opus-5-fast", "anthropic/claude-opus-4.8-fast", "meta/muse-spark-1.2",
    "meta/muse-spark-1.2-contributor", "meta/muse-spark-1.3", "meta/muse-spark-1.3-contributor", "openrouter/pareto-code",
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
    """openai-codex curated list from codex_models.py (DEFAULT_CODEX_MODELS + forward-compat
    synthesis) so the gateway /model picker and the CLI ``hermes model`` flow share one source."""
    from hermes_cli.codex_models import DEFAULT_CODEX_MODELS, _finalize_codex_models
    return _finalize_codex_models(list(DEFAULT_CODEX_MODELS))


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
        from agent.models_dev import _load_disk_cache
        data = _load_disk_cache()
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
    "qwen3.8-max-preview", "qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus", "qwen3.6-flash", "deepseek-v4-pro",
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
        "claude-fable-5", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
        "claude-sonnet-4-6", "claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929",
        "claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-haiku-4-5-20251001",
    ],
    "deepseek": ["deepseek-v4-pro", "deepseek-v4-flash"],
    "xiaomi": ["mimo-v2.5-pro", "mimo-v2.5", "mimo-v2-pro", "mimo-v2-omni", "mimo-v2-flash"],
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
        "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-sonnet-4", "claude-haiku-4-5", "gemini-3.7-flash",
        "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-pro", "gemini-3-flash",
        "grok-4.6", "grok-4.5", "grok-build-0.1", "muse-spark-1.2", "minimax-m3", "minimax-m2.7", "minimax-m2.5",
        "glm-5.3", "glm-5.3-flash", "glm-5.2", "glm-5.1", "glm-5", "kimi-k2.7-code", "deepseek-v4-pro",
        "deepseek-v4-flash", "deepseek-v4-flash-free", "qwen3.6-plus", "qwen3.5-plus", "big-pickle", "mimo-v2.5-free",
        "hy3-free", "laguna-s-2.1-free", "nemotron-3-ultra-free", "nemotron-3.5-lightning-free",
        "muse-spark-1.2-contributor-free", "muse-spark-1.3-contributor-free",
    ],
    # OpenCode keyless free tier — OFFLINE FLOOR only. provider_model_ids("opencode-free")
    # revalidates live against GET /zen/v1/models and filters to the anonymous tier, so this list
    # may lag the relay (intentional). Known-delisted models are REMOVED (the offline fallback must
    # not offer a model that 401s, e.g. x-preview-f-free).
    "opencode-free": [
        "deepseek-v4-flash-free", "hy3-free", "mimo-v2.5-free", "laguna-s-2.1-free",
        "nemotron-3-ultra-free", "nemotron-3.5-lightning-free", "muse-spark-1.2-contributor-free",
        "muse-spark-1.3-contributor-free",
    ],
    # Synced against opencode.ai/docs/go + live GET /zen/go/v1/models. "ox-alpha-free" is the
    # Go-subscription twin of Zen's keyless Ox Alpha (NOT keyless — the Go relay requires a Go key).
    "opencode-go": [
        "kimi-k3", "kimi-k2.7-code", "kimi-k2.6", "kimi-k2.5", "gpt-5.6-luna", "grok-4.5", "glm-5.3",
        "glm-5.3-flash", "glm-5.2", "glm-5.1", "glm-5", "mimo-v2.5-pro", "mimo-v2.5", "mimo-v2-pro",
        "mimo-v2-omni", "minimax-m3", "minimax-m2.7", "minimax-m2.5", "deepseek-v4-pro",
        "deepseek-v4-flash", "qwen3.8-max", "qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus",
        "qwen3.5-plus", "hy3", "hy3-preview", "muse-spark-1.2-contributor", "muse-spark-1.3-contributor",
        "ox-alpha-free",
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
        "us.anthropic.claude-sonnet-5", "us.anthropic.claude-sonnet-4-6", "us.anthropic.claude-opus-4-6-v1",
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


# ---------------------------------------------------------------------------
# Canonical provider list — single source of truth for provider identity. Every code path that
# lists, displays, or iterates providers (hermes model, /model, list_authenticated_providers)
# derives from it. slug = internal ID (config.yaml, --provider); label = short display name;
# tui_desc = longer description for the `hermes model` picker.
# ---------------------------------------------------------------------------

class ProviderEntry(NamedTuple):
    slug: str
    label: str
    tui_desc: str


CANONICAL_PROVIDERS: list[ProviderEntry] = [ProviderEntry(*row) for row in (
    ("nous", "Nous Portal", "Nous Portal (Everything your agent needs, 300+ models with bundled tool use)"),
    ("fireworks", "Fireworks AI", "Fireworks AI (OpenAI-compatible direct model API)"),
    ("openrouter", "OpenRouter", "OpenRouter (Pay-per-use API aggregator)"),
    ("moa", "Mixture of Agents", "Mixture of Agents (named presets; aggregator acts after reference models)"),
    ("novita", "NovitaAI", "NovitaAI (Cloud: Model API, Agent Sandbox, GPU Cloud)"),
    ("lmstudio", "LM Studio", "LM Studio (Local desktop app with built-in model server)"),
    ("anthropic", "Anthropic", "Anthropic (Claude models via API key or Claude Code)"),
    ("openai-codex", "ChatGPT or Codex Subscription", "ChatGPT or Codex Subscription (Sign in with your ChatGPT account, uses Codex models)"),
    ("openai-api", "OpenAI API", "OpenAI API (api.openai.com, API key)"),
    ("alibaba", "Qwen Cloud", "Qwen Cloud / DashScope (Qwen + multi-provider)"),
    ("xai-oauth", "xAI Grok OAuth (SuperGrok / Premium+)", "xAI Grok OAuth (SuperGrok / Premium+ subscription)"),
    ("xiaomi", "Xiaomi MiMo", "Xiaomi MiMo (MiMo-V2.5 and V2 models: pro, omni, flash)"),
    ("tencent-tokenhub", "Tencent TokenHub", "Tencent TokenHub (Hy4 preview via tokenhub.tencentmaas.com)"),
    ("tencent-tokenplan", "Tencent TokenPlan", "Tencent TokenPlan (Hy4 preview via api.lkeap.cloud.tencent.com, Anthropic Messages)"),
    ("nvidia", "NVIDIA NIM", "NVIDIA NIM (Nemotron models via build.nvidia.com or local NIM)"),
    ("copilot", "GitHub Copilot", "GitHub Copilot (Uses GITHUB_TOKEN or gh auth token)"),
    ("copilot-acp", "GitHub Copilot ACP", "GitHub Copilot ACP (Spawns copilot --acp --stdio)"),
    ("huggingface", "Hugging Face", "Hugging Face Inference Providers"),
    ("gemini", "Google AI Studio", "Google AI Studio (Native Gemini API)"),
    ("vertex", "Google Vertex AI", "Google Vertex AI (Gemini via GCP; OAuth2 service account or ADC, GCP billing/quotas)"),
    ("deepseek", "DeepSeek", "DeepSeek (V3, R1, coder, direct API)"), ("xai", "xAI", "xAI Grok (Direct API)"),
    ("zai", "Z.AI / GLM", "Z.AI / GLM (Zhipu direct API)"),
    ("kimi-coding", "Kimi / Kimi Coding Plan", "Kimi Coding Plan (api.kimi.com & Moonshot API)"),
    ("kimi-coding-cn", "Kimi / Moonshot (China)", "Kimi / Moonshot China (Domestic direct API)"),
    ("stepfun", "StepFun Step Plan", "StepFun Step Plan (Agent / coding models via Step Plan API)"),
    ("minimax", "MiniMax", "MiniMax (Global direct API)"),
    ("minimax-oauth", "MiniMax (OAuth)", "MiniMax via OAuth browser login (Coding Plan, minimax.io)"),
    ("minimax-cn", "MiniMax (China)", "MiniMax China (Domestic direct API)"),
    ("ollama-cloud", "Ollama Cloud", "Ollama Cloud (Cloud-hosted open models, ollama.com)"),
    ("arcee", "Arcee AI", "Arcee AI (Trinity models, direct API)"),
    ("gmi", "GMI Cloud", "GMI Cloud (Multi-model direct API)"),
    ("kilocode", "Kilo Code", "Kilo Code (Kilo Gateway API)"),
    ("opencode-zen", "OpenCode Zen", "OpenCode Zen (Curated models, pay-as-you-go)"),
    ("opencode-go", "OpenCode Go", "OpenCode Go (Open models subscription)"),
    ("bedrock", "AWS Bedrock", "AWS Bedrock (Claude, Nova, Llama, DeepSeek; IAM or API key)"),
    ("azure-foundry", "Azure Foundry", "Azure Foundry (OpenAI-style or Anthropic-style endpoint, your Azure AI deployment)"),
    ("ai-gateway", "Vercel AI Gateway", "Vercel AI Gateway (Multi-model aggregator)"),
    ("qwen-oauth", "Qwen OAuth (Portal)", "Qwen OAuth (Reuses local Qwen CLI login)"),
)]


# Auto-extend CANONICAL_PROVIDERS with providers registered under plugins/model-providers/<name>/
# so a new provider reaches the picker, /model and every downstream consumer without edits here.
# Non-api-key flows need bespoke picker UX and are skipped.
_canonical_slugs = {p.slug for p in CANONICAL_PROVIDERS}
try:
    from providers import list_providers as _list_providers_for_canonical
    for _pp in _list_providers_for_canonical():
        if _pp.name in _canonical_slugs or _pp.auth_type in {
            "oauth_device_code", "oauth_external", "external_process", "aws_sdk", "copilot", "vertex",
        }:
            continue
        _label = _pp.display_name or _pp.name
        CANONICAL_PROVIDERS.append(ProviderEntry(_pp.name, _label, _pp.description or f"{_label} (direct API)"))
        _canonical_slugs.add(_pp.name)
except Exception:
    pass


_PROVIDER_LABELS = {p.slug: p.label for p in CANONICAL_PROVIDERS}
_PROVIDER_LABELS["custom"] = "Custom endpoint"  # special case: not a named provider


# ---------------------------------------------------------------------------
# Provider groups — DISPLAY ONLY. Vendors with several slugs (global API, China API, OAuth plan,
# ...) fold under one top-level row in the INTERACTIVE PICKERS (``hermes model``, setup wizard,
# Telegram ``/model``). They do NOT change CANONICAL_PROVIDERS, slug identity, ``--provider``,
# ``/model <provider:model>`` or any typed path — every member slug stays individually addressable.
# ``group_providers()`` is the single fold used by all three surfaces.
#   group_id -> (display_label, group_description shown on the collapsed row, [member_slug, ...])
# Member order is the order shown inside the group submenu; member detail lives in ``tui_desc``.
# ---------------------------------------------------------------------------
PROVIDER_GROUPS: dict[str, tuple[str, str, list[str]]] = {
    "kimi":     ("Kimi / Moonshot", "Coding Plan, Moonshot global & China endpoints", ["kimi-coding", "kimi-coding-cn"]),
    "minimax":  ("MiniMax",         "Global, OAuth Coding Plan & China endpoints",     ["minimax", "minimax-oauth", "minimax-cn"]),
    "xai":      ("xAI Grok",        "Direct API or SuperGrok / Premium+ OAuth",        ["xai", "xai-oauth"]),
    "google":   ("Google Gemini",   "Google AI Studio (API key)",                     ["gemini"]),
    "openai":   ("OpenAI",          "ChatGPT/Codex subscription or direct OpenAI API", ["openai-codex", "openai-api"]),
    "qwen":     ("Qwen",            "Qwen Cloud / DashScope, Coding Plan, Token Plan & Qwen CLI OAuth", ["alibaba", "alibaba-cn", "alibaba-coding-plan", "alibaba-coding-plan-cn", "alibaba-token-plan", "alibaba-token-plan-cn", "qwen-oauth"]),
    "opencode": ("OpenCode",        "Zen pay-as-you-go, Go subscription, or free tier", ["opencode-zen", "opencode-go", "opencode-free"]),
    "copilot":  ("GitHub Copilot",  "GitHub token API or copilot --acp process",       ["copilot", "copilot-acp"]),
    "tencent":  ("Tencent Hy",      "Hy4 / Hy3 via TokenHub & TokenPlan", ["tencent-tokenhub", "tencent-tokenplan"]),
}

# Reverse index: member slug -> group_id.
_SLUG_TO_GROUP: dict[str, str] = {
    slug: gid for gid, (_label, _desc, members) in PROVIDER_GROUPS.items() for slug in members
}


def provider_group_for_slug(slug: str) -> str:
    """Return the group_id a provider slug belongs to, or "" if ungrouped."""
    return _SLUG_TO_GROUP.get(str(slug or "").strip().lower(), "")


def group_providers(slugs):
    """Fold a flat ordered slug iterable into picker rows by provider group (DISPLAY ONLY).

    A group row appears at the position of its FIRST present member, in input order; later
    members fold into it. Member order inside a group follows ``PROVIDER_GROUPS`` declaration,
    restricted to the members present in ``slugs``.
    """
    present = set(slugs)
    group_members = {
        gid: [m for m in members if m in present]
        for gid, (_label, _desc, members) in PROVIDER_GROUPS.items()
    }
    rows = []
    seen: set[str] = set()
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
        members = group_members.get(gid) or [s]
        if len(members) <= 1:
            rows.append({"kind": "single", "slug": members[0]})
        else:
            label, desc, _ = PROVIDER_GROUPS[gid]
            rows.append({"kind": "group", "group_id": gid, "label": label,
                         "description": desc, "members": list(members)})
    return rows


_PROVIDER_ALIASES = dict((
    ("glm", "zai"), ("z-ai", "zai"), ("z.ai", "zai"), ("zhipu", "zai"), ("github", "copilot"),
    ("github-copilot", "copilot"), ("github-models", "copilot"), ("github-model", "copilot"),
    ("github-copilot-acp", "copilot-acp"), ("copilot-acp-agent", "copilot-acp"), ("google", "gemini"),
    ("google-gemini", "gemini"), ("google-ai-studio", "gemini"), ("google-vertex", "vertex"), ("vertex-ai", "vertex"),
    ("gcp-vertex", "vertex"), ("vertexai", "vertex"), ("kimi", "kimi-coding"), ("moonshot", "kimi-coding"),
    ("kimi-cn", "kimi-coding-cn"), ("moonshot-cn", "kimi-coding-cn"), ("step", "stepfun"),
    ("stepfun-coding-plan", "stepfun"), ("arcee-ai", "arcee"), ("arceeai", "arcee"), ("gmi-cloud", "gmi"),
    ("gmicloud", "gmi"), ("fireworks-ai", "fireworks"), ("fw", "fireworks"), ("actual-computer", "actual"),
    ("actualcomputer", "actual"), ("aci", "actual"), ("nebius", "nebius-token-factory"),
    ("nebius-tokenfactory", "nebius-token-factory"), ("nebius-tf", "nebius-token-factory"),
    ("token-factory", "nebius-token-factory"), ("tokenfactory", "nebius-token-factory"),
    ("minimax-china", "minimax-cn"), ("minimax_cn", "minimax-cn"), ("minimax-portal", "minimax-oauth"),
    ("minimax-global", "minimax-oauth"), ("minimax_oauth", "minimax-oauth"), ("claude", "anthropic"),
    ("claude-code", "anthropic"), ("deep-seek", "deepseek"), ("opencode", "opencode-zen"), ("zen", "opencode-zen"),
    ("go", "opencode-go"), ("opencode-go-sub", "opencode-go"), ("free", "opencode-free"),
    ("opencode_free", "opencode-free"), ("aigateway", "ai-gateway"), ("vercel", "ai-gateway"),
    ("vercel-ai-gateway", "ai-gateway"), ("kilo", "kilocode"), ("kilo-code", "kilocode"),
    ("kilo-gateway", "kilocode"), ("dashscope", "alibaba"), ("aliyun", "alibaba"), ("qwen", "alibaba"),
    ("alibaba-cloud", "alibaba"), ("qwen-portal", "qwen-oauth"), ("hf", "huggingface"),
    ("hugging-face", "huggingface"), ("huggingface-hub", "huggingface"), ("novita-ai", "novita"),
    ("novitaai", "novita"), ("mimo", "xiaomi"), ("xiaomi-mimo", "xiaomi"), ("tencent", "tencent-tokenhub"),
    ("tokenhub", "tencent-tokenhub"), ("tencent-cloud", "tencent-tokenhub"), ("tencentmaas", "tencent-tokenhub"),
    ("tokenplan", "tencent-tokenplan"), ("tencent-lkeap", "tencent-tokenplan"), ("aws", "bedrock"),
    ("aws-bedrock", "bedrock"), ("amazon-bedrock", "bedrock"), ("amazon", "bedrock"), ("grok", "xai"),
    ("grok-oauth", "xai-oauth"), ("xai-oauth", "xai-oauth"), ("x-ai-oauth", "xai-oauth"),
    ("xai-grok-oauth", "xai-oauth"), ("x-ai", "xai"), ("x.ai", "xai"), ("nim", "nvidia"), ("nvidia-nim", "nvidia"),
    ("build-nvidia", "nvidia"), ("nemotron", "nvidia"), ("lmstudio", "lmstudio"), ("lm-studio", "lmstudio"),
    ("lm_studio", "lmstudio"),
    ("ollama", "custom"),  # bare "ollama" = local; use "ollama-cloud" for cloud
    ("ollama_cloud", "ollama-cloud"),
))


# Offline/fresh-install fallback for the model Hermes silently lands on when the user never picked
# one (GUI onboarding confirm card, empty ``model.default``, provider-set-but-model-missing). The
# AUTHORITATIVE source is the remote catalog manifest, which labels exactly one entry per provider
# ``"default": true`` (get_default_model_from_cache) so the default rotates without a release; this
# MUST match the labeled entry in website/static/api/model-catalog.json. Deliberately a capable
# low-cost model rather than the curated lists' entry [0]: aggregator lists are ordered
# most-capable-first, so [0] is the priciest Anthropic flagship.
PREFERRED_SILENT_DEFAULT_MODEL = "z-ai/glm-5.2"


# Providers whose *silent* auto-default goes through the cost-safe catalog-labeled default
# (``get_preferred_silent_default_model``) instead of curated entry [0]. Metered aggregators order
# best-first, so [0] is the priciest flagship; a profile that sets a provider with no model would
# otherwise silently bill the most expensive model (863 Opus requests before one user noticed).
# Network-free (cache-only) on purpose — this is the hot resolution path. The *interactive* default
# (GUI onboarding / ``hermes model``) uses the tier-aware ``get_recommended_default_model`` in
# hermes_cli/web_server.py + ``partition_nous_models_by_tier``, which may hit the Portal.
_SILENT_DEFAULT_PROVIDERS: frozenset[str] = frozenset({"nous", "openrouter"})


# Retired model IDs kept for /model auto-detect only — not shown in pickers. DeepSeek cut these
# off; model_normalize remaps them on the wire.
_PROVIDER_RETIRED_ALIASES: dict[str, tuple[str, ...]] = {
    "deepseek": ("deepseek-chat", "deepseek-reasoner"),
}


_AGGREGATOR_PROVIDERS = frozenset({"nous", "openrouter", "ai-gateway", "copilot", "kilocode"})


# OpenRouter request-time routing variants (docs: guides/routing/model-variants): per-request
# modifiers valid on ANY model id (":nitro" throughput sort + priority tier, ":floor" price sort +
# flex tier, ":exacto" quality-first provider sort, ":online" web plugin). Never separate catalog
# entries — /models lists only the base id. NOT here: ":free", ":batch", ":thinking", ":extended"
# — those ARE distinct SKUs that appear in /models when they exist, so absence is authoritative.
_OPENROUTER_VARIANT_SUFFIXES = frozenset({"nitro", "floor", "exacto", "online"})


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
_OPENAI_FAST_MODE_PREFIXES: tuple[str, ...] = ("gpt-", "o1", "o3", "o4")


# Providers where models.dev is authoritative: the curated list is an offline fallback plus custom
# additions the registry lacks, merged fresh-first (curated-only names appended) for both the CLI
# and the gateway /model picker. DELIBERATELY EXCLUDED: "openrouter" (curated list is a hand-picked
# agentic subset of 400+ models — merging would dump everything), "nous" (curated list + Portal
# /models are the subscription-tier source of truth), and providers with dedicated live-endpoint
# branches (copilot, anthropic, ai-gateway, ollama-cloud, custom, stepfun, openai-codex).
_MODELS_DEV_PREFERRED: frozenset[str] = frozenset({
    "opencode-go", "opencode-zen", "deepseek", "kilocode", "fireworks", "mistral", "togetherai", "cohere",
    "perplexity", "groq", "nvidia", "huggingface", "zai", "gemini", "google", "xai", "xai-oauth",
})


# Providers whose catalog is served with NO credential get a constant credential fingerprint in
# the disk cache: the anonymous opencode-free catalog's freshness comes from TTL revalidation,
# so folding in unrelated auth.json mtimes would only bust the SWR cache needlessly.
_KEYLESS_STABLE_CACHE_PROVIDERS = frozenset({"opencode-free"})


# OpenRouter-style ids -> Copilot ids. Dash-notation Claude ids are accepted too: Hermes' default
# Claude IDs use hyphens (Anthropic native) but Copilot's API only accepts dot-notation, so a
# copilot + hyphenated default would otherwise hit HTTP 400 "model_not_supported".
_COPILOT_MODEL_ALIASES = dict((
    ("openai/gpt-5", "gpt-5-mini"), ("openai/gpt-5-chat", "gpt-5-mini"), ("openai/gpt-5-mini", "gpt-5-mini"),
    ("openai/gpt-5-nano", "gpt-5-mini"), ("openai/gpt-4.1", "gpt-4.1"), ("openai/gpt-4.1-mini", "gpt-4.1"),
    ("openai/gpt-4.1-nano", "gpt-4.1"), ("openai/gpt-4o", "gpt-4o"), ("openai/gpt-4o-mini", "gpt-4o-mini"),
    ("openai/o1", "gpt-5.2"), ("openai/o1-mini", "gpt-5-mini"), ("openai/o1-preview", "gpt-5.2"),
    ("openai/o3", "gpt-5.3-codex"), ("openai/o3-mini", "gpt-5-mini"), ("openai/o4-mini", "gpt-5-mini"),
    ("anthropic/claude-opus-4.6", "claude-opus-4.6"), ("anthropic/claude-sonnet-5", "claude-sonnet-5"),
    ("anthropic/claude-sonnet-4.6", "claude-sonnet-4.6"), ("anthropic/claude-sonnet-4", "claude-sonnet-4"),
    ("anthropic/claude-sonnet-4.5", "claude-sonnet-4.5"), ("anthropic/claude-haiku-4.5", "claude-haiku-4.5"),
    ("claude-sonnet-5", "claude-sonnet-5"), ("claude-opus-4-6", "claude-opus-4.6"),
    ("claude-sonnet-4-6", "claude-sonnet-4.6"), ("claude-sonnet-4-0", "claude-sonnet-4"),
    ("claude-sonnet-4-5", "claude-sonnet-4.5"), ("claude-haiku-4-5", "claude-haiku-4.5"),
    ("anthropic/claude-opus-4-6", "claude-opus-4.6"), ("anthropic/claude-sonnet-4-6", "claude-sonnet-4.6"),
    ("anthropic/claude-sonnet-4-0", "claude-sonnet-4"), ("anthropic/claude-sonnet-4-5", "claude-sonnet-4.5"),
    ("anthropic/claude-haiku-4-5", "claude-haiku-4.5"),
))


# Azure Foundry model families that require the Responses API: Azure rejects /chat/completions
# against them with ``400 "The requested operation is unsupported."`` (seen on gpt-5.3-codex while
# gpt-4o on the same endpoint worked). Broad enough for vendor-renamed deployments (gpt-5.x-codex,
# o1-preview), tight enough to leave GPT-4 / 3.5 / Llama / Mistral / Grok on chat completions.
_AZURE_FOUNDRY_RESPONSES_PREFIXES = ("codex", "gpt-5", "o1", "o3", "o4")
