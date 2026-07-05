"""Declarative provider naming table (Phase 5a Apply-1).

This module is the SINGLE SOURCE OF TRUTH for the canonical provider
identifier mapping across the legacy runtime, the Phase 4a Router,
and the Phase 4b Engine adapter registry.

Contract v1 — frozen 2026-06-28 (Phase 5a Apply-1).
Phase 5a declares the table but does NOT change runtime behavior.
The legacy runtime still resolves providers via
``hermes_cli.auth.resolve_provider``; the Phase 4b Engine still
resolves via ``agent.execution_router.PROVIDER_METADATA`` and
``agent.providers.registry``. This module is read-only from both
paths and is intended for migration in Phase 5b.

See ~/.hermes/reports/design/2026-06-28_hermes-phase-5a-apply-1/NAMING_TABLE.md.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple


# Canonical provider ids (the legacy PROVIDER_REGISTRY names).
# These are the names that the legacy runtime emits from
# resolve_provider(); they are the target names for Phase 5b
# normalization.
CANONICAL_PROVIDER_IDS: Tuple[str, ...] = (
    "minimax",
    "minimax-oauth",
    "minimax-cn",
    "openai-codex",
    "openai-api",
    "anthropic",
    "gemini",
    "nvidia",
    "openrouter",
    "deepseek",
    "opencode-zen",
    "opencode-go",
    "huggingface",
    "alibaba",
    "alibaba-coding-plan",
    "kimi-coding",
    "kimi-coding-cn",
    "stepfun",
    "arcee",
    "gmi",
    "kilocode",
    "xiaomi",
    "tencent-tokenhub",
    "ollama-cloud",
    "novita",
    "xai",
    "xai-oauth",
    "qwen-oauth",
    "zai",
    "copilot",
    "copilot-acp",
    "lmstudio",
    "nous",
    "bedrock",
    "azure-foundry",
)


# Mapping: legacy alias (input string) → canonical provider id.
# Mirrors ``hermes_cli.auth._PROVIDER_ALIASES`` (which is local to
# resolve_provider). Phase 5a declares the table here; the legacy
# function remains the active resolution path.
ALIAS_TO_CANONICAL: Mapping[str, str] = {
    "minimax": "minimax",
    "minimax-oauth": "minimax-oauth",
    "minimax-portal": "minimax-oauth",
    "minimax-global": "minimax-oauth",
    "minimax-cn": "minimax-cn",
    "minimax-china": "minimax-cn",
    "openai-codex": "openai-codex",
    "openai-api": "openai-api",
    "openai": "openai-api",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "claude-code": "anthropic",
    "gemini": "gemini",
    "google": "gemini",
    "google-gemini": "gemini",
    "google-ai-studio": "gemini",
    "nvidia": "nvidia",
    "nemotron": "nvidia",
    "openrouter": "openrouter",
    "deepseek": "deepseek",
    "opencode-zen": "opencode-zen",
    "opencode": "opencode-zen",
    "zen": "opencode-zen",
    "opencode-go": "opencode-go",
    "opencode-go-sub": "opencode-go",
    "go": "opencode-go",
    "huggingface": "huggingface",
    "hf": "huggingface",
    "hugging-face": "huggingface",
    "huggingface-hub": "huggingface",
    "alibaba": "alibaba",
    "alibaba-coding-plan": "alibaba-coding-plan",
    "alibaba_coding": "alibaba-coding-plan",
    "kimi-coding": "kimi-coding",
    "kimi": "kimi-coding",
    "kimi-for-coding": "kimi-coding",
    "moonshot": "kimi-coding",
    "kimi-cn": "kimi-coding-cn",
    "kimi-coding-cn": "kimi-coding-cn",
    "moonshot-cn": "kimi-coding-cn",
    "stepfun": "stepfun",
    "step": "stepfun",
    "stepfun-coding-plan": "stepfun",
    "arcee": "arcee",
    "arcee-ai": "arcee",
    "arceeai": "arcee",
    "gmi": "gmi",
    "gmi-cloud": "gmi",
    "gmicloud": "gmi",
    "minimax-cn": "minimax-cn",
    "minimax-china": "minimax-cn",
    "minimax_cn": "minimax-cn",
    "minimax-oauth": "minimax-oauth",
    "minimax_oauth": "minimax-oauth",
    "alibaba-coding-plan": "alibaba-coding-plan",
    "alibaba-coding": "alibaba-coding-plan",
    "alibaba_coding": "alibaba-coding-plan",
    "alibaba_coding_plan": "alibaba-coding-plan",
    "kilocode": "kilocode",
    "kilo": "kilocode",
    "kilo-code": "kilocode",
    "kilo-gateway": "kilocode",
    "xiaomi": "xiaomi",
    "mimo": "xiaomi",
    "xiaomi-mimo": "xiaomi",
    "tencent-tokenhub": "tencent-tokenhub",
    "tokenhub": "tencent-tokenhub",
    "tencent": "tencent-tokenhub",
    "tencent-cloud": "tencent-tokenhub",
    "tencentmaas": "tencent-tokenhub",
    "ollama-cloud": "ollama-cloud",
    "ollama_cloud": "ollama-cloud",
    "ollama": "custom",
    "vllm": "custom",
    "llamacpp": "custom",
    "llama.cpp": "custom",
    "llama-cpp": "custom",
    "novita": "novita",
    "novita-ai": "novita",
    "xai": "xai",
    "x-ai": "xai",
    "x.ai": "xai",
    "grok": "xai",
    "xai-oauth": "xai-oauth",
    "x-ai-oauth": "xai-oauth",
    "grok-oauth": "xai-oauth",
    "xai-grok-oauth": "xai-oauth",
    "qwen-oauth": "qwen-oauth",
    "qwen-portal": "qwen-oauth",
    "qwen-cli": "qwen-oauth",
    "zai": "zai",
    "glm": "zai",
    "z-ai": "zai",
    "z.ai": "zai",
    "zhipu": "zai",
    "copilot": "copilot",
    "github": "copilot",
    "github-copilot": "copilot",
    "github-models": "copilot",
    "github-model": "copilot",
    "copilot-acp": "copilot-acp",
    "copilot-acp-agent": "copilot-acp",
    "github-copilot-acp": "copilot-acp",
    "lmstudio": "lmstudio",
    "lm-studio": "lmstudio",
    "lm_studio": "lmstudio",
    "nous": "nous",
    "bedrock": "bedrock",
    "aws": "bedrock",
    "aws-bedrock": "bedrock",
    "amazon-bedrock": "bedrock",
    "amazon": "bedrock",
    "azure-foundry": "azure-foundry",
}


# Mapping: Phase 4b adapter id → canonical provider id.
# The Phase 4b adapter registry uses shorter, role-oriented names
# (e.g., "codex_auth") than the legacy runtime (e.g., "openai-codex").
# Phase 5b will rename the adapter to match the canonical id; until
# then this table documents the planned mapping for migration.
ADAPTER_TO_CANONICAL: Mapping[str, str] = {
    "minimax": "minimax",
    "codex_auth": "openai-codex",  # Phase 5b will rename adapter
    "fake": "__test_only__",
}


# Mapping: Router Phase 4a id (PROVIDER_METADATA) → canonical provider id.
# Phase 4a Router uses short names ("openai", "codex", "google",
# "nemotron") that don't match the legacy registry's longer names.
# Phase 5b will extend PROVIDER_METADATA; until then this table
# documents the planned mapping.
ROUTER_TO_CANONICAL: Mapping[str, str] = {
    "minimax": "minimax",
    "openai": "openai-api",
    "codex": "openai-codex",
    "nemotron": "nvidia",
    "anthropic": "anthropic",
    "google": "gemini",
    "local_only": "custom",
}


def canonicalize_alias(alias: str) -> str:
    """Return the canonical provider id for a legacy alias, or the
    alias itself if not found."""
    return ALIAS_TO_CANONICAL.get(alias, alias)


def canonical_from_adapter(adapter_id: str) -> str:
    """Return the canonical provider id for a Phase 4b adapter id."""
    return ADAPTER_TO_CANONICAL.get(adapter_id, adapter_id)


def canonical_from_router(router_id: str) -> str:
    """Return the canonical provider id for a Router Phase 4a id."""
    return ROUTER_TO_CANONICAL.get(router_id, router_id)


# Production status (declared, not measured). Aligns with the
# classification from HERMES_MULTI_PROVIDER_COMPARISON_READONLY.
PROVIDER_PRODUCTION_STATUS: Dict[str, str] = {
    "minimax": "Production Ready",
    "minimax-oauth": "Production Ready with Caveats",
    "minimax-cn": "Production Ready",
    "openai-codex": "Production Ready with Caveats",
    "openai-api": "Production Ready",
    "anthropic": "Production Ready",
    "gemini": "Production Ready",
    "nvidia": "Production Ready",
    "openrouter": "Production Ready",
    "bedrock": "Production Ready",
    "azure-foundry": "Production Ready",
    "xai": "Production Ready",
    "xai-oauth": "Production Ready with Caveats",
    "qwen-oauth": "Production Ready with Caveats",
    "nous": "Production Ready",
    "copilot": "Production Ready",
    "copilot-acp": "Experimental",
    "lmstudio": "Production Ready",
    "codex_auth": "Experimental",  # Phase 4b adapter; placeholder URL
    "fake": "Unsupported",  # test-only
}