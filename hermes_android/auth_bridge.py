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
    "gemini": "GOOGLE_API_KEY",
    "chatgpt-web": "CHATGPT_WEB_ACCESS_TOKEN",
}

PROVIDER_AUTH_BUNDLE_KEYS = {
    "chatgpt-web": {
        "api_key": "CHATGPT_WEB_ACCESS_TOKEN",
        "access_token": "CHATGPT_WEB_ACCESS_TOKEN",
        "session_token": "CHATGPT_WEB_SESSION_TOKEN",
    },
    "anthropic": {
        "api_key": "ANTHROPIC_API_KEY",
        "access_token": "ANTHROPIC_TOKEN",
    },
    "gemini": {
        "api_key": "GOOGLE_API_KEY",
    },
}


def provider_env_key(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    return PROVIDER_ENV_KEYS.get(normalized, normalized.upper().replace("-", "_") + "_API_KEY")


def read_provider_api_key(provider: str) -> str:
    return load_env().get(provider_env_key(provider), "")


def read_provider_auth_bundle(provider: str) -> dict[str, Any]:
    normalized = str(provider or "").strip().lower()
    env = load_env()
    keys = dict(PROVIDER_AUTH_BUNDLE_KEYS.get(normalized, {}))
    if "api_key" not in keys:
        keys["api_key"] = provider_env_key(normalized)
    return {
        "provider": normalized,
        "api_key": env.get(keys.get("api_key", ""), ""),
        "access_token": env.get(keys.get("access_token", ""), "") if keys.get("access_token") else "",
        "session_token": env.get(keys.get("session_token", ""), "") if keys.get("session_token") else "",
        "configured": any(
            env.get(env_key, "")
            for env_key in keys.values()
            if env_key
        ),
    }


def write_provider_api_key(provider: str, api_key: str) -> dict[str, Any]:
    env_key = provider_env_key(provider)
    save_env_value(env_key, api_key)
    return {"provider": provider, "env_key": env_key, "saved": True}


def write_provider_auth_bundle(
    provider: str,
    api_key: str = "",
    access_token: str = "",
    session_token: str = "",
) -> dict[str, Any]:
    normalized = str(provider or "").strip().lower()
    keys = dict(PROVIDER_AUTH_BUNDLE_KEYS.get(normalized, {}))
    keys.setdefault("api_key", provider_env_key(normalized))
    api_key_value = api_key or access_token
    if keys.get("api_key"):
        save_env_value(keys["api_key"], api_key_value)
    if keys.get("access_token"):
        save_env_value(keys["access_token"], access_token)
    if keys.get("session_token"):
        save_env_value(keys["session_token"], session_token)
    return {
        "provider": normalized,
        "saved": True,
        "api_key_env": keys.get("api_key", ""),
        "access_token_env": keys.get("access_token", ""),
        "session_token_env": keys.get("session_token", ""),
    }


def clear_provider_auth_bundle(provider: str) -> dict[str, Any]:
    normalized = str(provider or "").strip().lower()
    keys = set(PROVIDER_AUTH_BUNDLE_KEYS.get(normalized, {}).values())
    keys.add(provider_env_key(normalized))
    for env_key in keys:
        if env_key:
            save_env_value(env_key, "")
    return {"provider": normalized, "cleared": True, "keys": sorted(k for k in keys if k)}
