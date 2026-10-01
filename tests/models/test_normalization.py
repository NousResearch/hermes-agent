from __future__ import annotations

import pytest

from models import normalize_model_id


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("openrouter", "claude-sonnet-4.6", "anthropic/claude-sonnet-4.6"),
        ("nous", "gpt-5.4", "openai/gpt-5.4"),
        ("ai-gateway", "gemini-3.8-flash", "google/gemini-3.8-flash"),
        ("kilocode", "deepseek-v4-pro", "deepseek/deepseek-v4-pro"),
    ],
)
def test_vendor_qualified_aggregators_add_detected_vendor(provider, model, expected):
    assert normalize_model_id(provider, model) == expected


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("zai", "zai/glm-5.1", "glm-5.1"),
        ("zai", "glm:glm-5.1", "glm-5.1"),
        ("minimax", "minimax/minimax-m2.7", "minimax-m2.7"),
        ("qwen-oauth", "qwen/qwen3-coder-plus", "qwen/qwen3-coder-plus"),
        ("qwen-oauth", "qwen-oauth/qwen3-coder-plus", "qwen3-coder-plus"),
        ("arcee", "arcee-ai/trinity-large-preview", "trinity-large-preview"),
        ("custom", "custom:qwen3:8b", "qwen3:8b"),
        ("custom", "ollama/qwen3:8b", "ollama/qwen3:8b"),
        ("gemini", "google:gemini-3.1-pro", "gemini-3.1-pro"),
        ("xai", "grok/grok-4", "grok-4"),
    ],
)
def test_direct_provider_profiles_strip_only_owned_prefixes(provider, model, expected):
    assert normalize_model_id(provider, model) == expected


def test_anthropic_strips_owned_prefix_and_converts_dots():
    assert (
        normalize_model_id("anthropic", "claude/claude-sonnet-4.6")
        == "claude-sonnet-4-6"
    )


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai/gpt-5.4", "gpt-5.4"),
        ("openai:gpt-5.4", "gpt-5.4"),
        ("openai-codex:gpt-5.6-sol", "gpt-5.6-sol"),
        ("anthropic:claude-x", "anthropic:claude-x"),
    ],
)
def test_openai_codex_strips_only_owned_or_openai_prefixes(model, expected):
    assert normalize_model_id("openai-codex", model) == expected


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("opencode-go", "moonshotai/kimi-k2.5", "kimi-k2.5"),
        ("opencode-go", "minimax-m2.7", "minimax-m2.7"),
        ("opencode-zen", "anthropic/claude-sonnet-4.6", "claude-sonnet-4-6"),
        ("opencode-zen", "google/gemini-3.1-pro", "gemini-3.1-pro"),
    ],
)
def test_opencode_profiles_use_flat_native_ids(provider, model, expected):
    assert normalize_model_id(provider, model) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("deepseek-reasoner", "deepseek-flash"),
        ("deepseek-chat", "deepseek-flash"),
        ("deepseek-v4-pro", "deepseek-v4-pro"),
        ("deepseek-next-preview", "deepseek-next-preview"),
    ],
)
def test_deepseek_only_rewrites_retired_aliases(model, expected):
    assert normalize_model_id("deepseek", model) == expected


def test_xiaomi_lowercases_after_owned_prefix_strip():
    assert normalize_model_id("xiaomi", "mimo/MiMo-V2.5-Pro") == "mimo-v2.5-pro"


def test_nvidia_repairs_bare_prefix_only_from_supplied_candidates():
    known = {
        "nvidia/nemotron-3-ultra-550b-a55b",
        "z-ai/glm-5.2",
    }
    assert (
        normalize_model_id(
            "nvidia",
            "nemotron-3-ultra-550b-a55b",
            known_ids=known,
        )
        == "nvidia/nemotron-3-ultra-550b-a55b"
    )
    assert normalize_model_id("nvidia", "glm-5.2", known_ids=known) == "z-ai/glm-5.2"
    assert normalize_model_id("nvidia", "local-model", known_ids=known) == "local-model"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("anthropic/claude-sonnet-4-6", "claude-sonnet-4.6"),
        ("openai/gpt-5", "gpt-5-mini"),
        ("owner/custom/model", "owner/custom/model"),
    ],
)
def test_copilot_uses_provider_owned_aliases_without_catalogue_fetch(model, expected):
    assert normalize_model_id("copilot", model) == expected


def test_copilot_accepts_caller_supplied_live_candidate():
    assert (
        normalize_model_id("copilot", "gpt-5.7-preview", known_ids={"gpt-5.7-preview"})
        == "gpt-5.7-preview"
    )
