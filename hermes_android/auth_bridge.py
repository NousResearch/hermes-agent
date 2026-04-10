from __future__ import annotations

from typing import Any

from hermes_cli.config import load_env, save_env_value

PROVIDER_ENV_KEYS = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-codex": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "nous": "NOUS_API_KEY",
    "custom": "OPENAI_API_KEY",
}


def provider_env_key(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    return PROVIDER_ENV_KEYS.get(normalized, normalized.upper().replace("-", "_") + "_API_KEY")


def read_provider_api_key(provider: str) -> str:
    return load_env().get(provider_env_key(provider), "")


def write_provider_api_key(provider: str, api_key: str) -> dict[str, Any]:
    env_key = provider_env_key(provider)
    save_env_value(env_key, api_key)
    return {"provider": provider, "env_key": env_key, "saved": True}
