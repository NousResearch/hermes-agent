"""Declarative api_mode taxonomy (Phase 5a Apply-1).

This module is the SINGLE SOURCE OF TRUTH for the closed enum of
api_modes used across the legacy runtime and the Phase 4b Engine.

Contract v1 — frozen 2026-06-28 (Phase 5a Apply-1).
Do not add new api_modes here without bumping to v2.
See ~/.hermes/reports/design/2026-06-28_hermes-phase-5a-apply-1/CONTRACT_FREEZE.md.

The taxonomy mirrors ``hermes_cli.runtime_provider._VALID_API_MODES``
but is moved here so that both the legacy runtime and the Phase 4b
Engine can import the same canonical definition without circular
dependencies. The legacy runtime still owns the auto-detection
helper (``_detect_api_mode_for_url``); Phase 5a does NOT change
runtime behavior — only declares the canonical enum.
"""

from __future__ import annotations

from typing import FrozenSet


# Closed enum v1. Mirroring the legacy runtime's _VALID_API_MODES.
API_MODES: FrozenSet[str] = frozenset(
    {
        "chat_completions",
        "codex_responses",
        "anthropic_messages",
        "bedrock_converse",
        # Optional opt-in: hands the turn to a `codex app-server` subprocess.
        # Default is unchanged; opt-in via model.openai_runtime in config.
        "codex_app_server",
    }
)


# Canonical mapping: legacy provider id → primary api_mode.
# "Primary" means the api_mode that the runtime will use unless an
# explicit override is provided. Multiple api_modes can apply to one
# provider (e.g., openai-api supports both chat_completions and
# codex_responses depending on model); this table lists the default.
PROVIDER_DEFAULT_API_MODE: dict = {
    # default operational
    "minimax": "anthropic_messages",
    "minimax-oauth": "anthropic_messages",
    "minimax-cn": "anthropic_messages",
    # fallback validated
    "openai-codex": "codex_responses",
    "openai-api": "codex_responses",
    # api_key legacy providers (chat_completions by default)
    "anthropic": "anthropic_messages",
    "gemini": "chat_completions",
    "nvidia": "chat_completions",
    "openrouter": "chat_completions",
    "deepseek": "chat_completions",
    "opencode-zen": "chat_completions",
    "opencode-go": "chat_completions",
    "huggingface": "chat_completions",
    "alibaba": "chat_completions",
    "alibaba-coding-plan": "chat_completions",
    "kimi-coding": "anthropic_messages",
    "kimi-coding-cn": "anthropic_messages",
    "stepfun": "chat_completions",
    "arcee": "chat_completions",
    "gmi": "chat_completions",
    "kilocode": "chat_completions",
    "xiaomi": "chat_completions",
    "tencent-tokenhub": "chat_completions",
    "ollama-cloud": "chat_completions",
    "novita": "chat_completions",
    "xai": "codex_responses",
    "xai-oauth": "codex_responses",
    "qwen-oauth": "chat_completions",
    "zai": "chat_completions",
    "copilot": "chat_completions",
    "copilot-acp": "codex_app_server",
    "lmstudio": "chat_completions",
    "nous": "chat_completions",
    "bedrock": "bedrock_converse",
    "azure-foundry": "codex_responses",
}


def is_valid_api_mode(mode: str) -> bool:
    """Return True if ``mode`` is in the closed api_mode enum."""
    return isinstance(mode, str) and mode in API_MODES