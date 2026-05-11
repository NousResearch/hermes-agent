from __future__ import annotations

import json
import os
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from hermes_android.python_path import prefer_hermes_package_root

prefer_hermes_package_root()

from hermes_cli.config import load_env, save_env_value

DEFAULT_QWEN_BASE_URL = "https://portal.qwen.ai/v1"
QWEN_BASE_URL_ENV = "HERMES_QWEN_BASE_URL"
QWEN_OAUTH_CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"
QWEN_OAUTH_TOKEN_URL = "https://chat.qwen.ai/api/v1/oauth2/token"
QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120


def _qwen_auth_path() -> Path:
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE") or str(Path.home())
    return Path(home).expanduser() / ".qwen" / "oauth_creds.json"

PROVIDER_ENV_KEYS = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-codex": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "nous": "NOUS_API_KEY",
    "custom": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "chatgpt-web": "CHATGPT_WEB_ACCESS_TOKEN",
    "alibaba": "DASHSCOPE_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "zai": "GLM_API_KEY",
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
    "zai": {
        "api_key": "GLM_API_KEY",
    },
    "alibaba": {
        "api_key": "DASHSCOPE_API_KEY",
    },
    "dashscope": {
        "api_key": "DASHSCOPE_API_KEY",
    },
}


def _read_qwen_tokens() -> dict[str, Any]:
    auth_path = _qwen_auth_path()
    if not auth_path.exists():
        return {}
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}



def _save_qwen_tokens(tokens: dict[str, Any]) -> Path:
    auth_path = _qwen_auth_path()
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = auth_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(tokens, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)
    tmp_path.replace(auth_path)
    return auth_path



def _clear_qwen_tokens() -> None:
    try:
        _qwen_auth_path().unlink()
    except FileNotFoundError:
        return



def _qwen_access_token_is_expiring(expiry_date_ms: Any, skew_seconds: int = QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS) -> bool:
    try:
        expiry_ms = int(expiry_date_ms)
    except Exception:
        return True
    return (time.time() + max(0, int(skew_seconds))) * 1000 >= expiry_ms



def _refresh_qwen_tokens(tokens: dict[str, Any], timeout_seconds: float = 20.0) -> dict[str, Any]:
    refresh_token = str(tokens.get("refresh_token", "") or "").strip()
    if not refresh_token:
        raise RuntimeError("Qwen OAuth refresh token missing")
    body = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": QWEN_OAUTH_CLIENT_ID,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        QWEN_OAUTH_TOKEN_URL,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Qwen OAuth refresh failed: HTTP {exc.code} {detail}".strip()) from exc
    if not isinstance(payload, dict) or not str(payload.get("access_token", "") or "").strip():
        raise RuntimeError("Qwen OAuth refresh response missing access_token")
    try:
        expires_in_seconds = int(payload.get("expires_in"))
    except Exception:
        expires_in_seconds = 6 * 60 * 60
    refreshed = {
        "access_token": str(payload.get("access_token", "") or "").strip(),
        "refresh_token": str(payload.get("refresh_token", refresh_token) or refresh_token).strip(),
        "token_type": str(payload.get("token_type", tokens.get("token_type", "Bearer")) or "Bearer").strip() or "Bearer",
        "resource_url": str(payload.get("resource_url", tokens.get("resource_url", "portal.qwen.ai")) or "portal.qwen.ai").strip(),
        "expiry_date": int(time.time() * 1000) + max(1, expires_in_seconds) * 1000,
    }
    _save_qwen_tokens(refreshed)
    return refreshed



def provider_env_key(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    return PROVIDER_ENV_KEYS.get(normalized, normalized.upper().replace("-", "_") + "_API_KEY")



def read_provider_api_key(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized == "qwen-oauth":
        return str(_read_qwen_tokens().get("access_token", "") or "")
    return load_env().get(provider_env_key(normalized), "")



def read_provider_auth_bundle(provider: str) -> dict[str, Any]:
    normalized = str(provider or "").strip().lower()
    env = load_env()
    if normalized == "qwen-oauth":
        tokens = _read_qwen_tokens()
        refresh_error = ""
        if tokens and _qwen_access_token_is_expiring(tokens.get("expiry_date")) and str(tokens.get("refresh_token", "") or "").strip():
            try:
                tokens = _refresh_qwen_tokens(tokens)
            except Exception as exc:
                refresh_error = exc.__class__.__name__
        access_token = str(tokens.get("access_token", "") or "")
        refresh_token = str(tokens.get("refresh_token", "") or "")
        base_url = env.get(QWEN_BASE_URL_ENV, "").strip().rstrip("/") or DEFAULT_QWEN_BASE_URL
        return {
            "provider": normalized,
            "api_key": access_token,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "session_token": "",
            "base_url": base_url,
            "expires_at_ms": tokens.get("expiry_date", 0),
            "refresh_error": refresh_error,
            "configured": bool(access_token or refresh_token),
        }
    keys = dict(PROVIDER_AUTH_BUNDLE_KEYS.get(normalized, {}))
    if "api_key" not in keys:
        keys["api_key"] = provider_env_key(normalized)
    return {
        "provider": normalized,
        "api_key": env.get(keys.get("api_key", ""), ""),
        "access_token": env.get(keys.get("access_token", ""), "") if keys.get("access_token") else "",
        "session_token": env.get(keys.get("session_token", ""), "") if keys.get("session_token") else "",
        "configured": any(env.get(env_key, "") for env_key in keys.values() if env_key),
    }


def read_provider_auth_bundle_json(provider: str) -> str:
    return json.dumps(read_provider_auth_bundle(provider), sort_keys=True)



def write_provider_api_key(provider: str, api_key: str) -> dict[str, Any]:
    normalized = str(provider or "").strip().lower()
    if normalized == "qwen-oauth":
        return write_provider_auth_bundle(normalized, access_token=api_key)
    env_key = provider_env_key(normalized)
    save_env_value(env_key, api_key)
    return {"provider": normalized, "env_key": env_key, "saved": True}



def write_provider_auth_bundle(
    provider: str,
    api_key: str = "",
    access_token: str = "",
    session_token: str = "",
    refresh_token: str = "",
    base_url: str = "",
) -> dict[str, Any]:
    normalized = str(provider or "").strip().lower()
    if normalized == "qwen-oauth":
        existing = _read_qwen_tokens()
        access_value = access_token or api_key
        refresh_value = refresh_token or str(existing.get("refresh_token", "") or "")
        if not refresh_value and session_token:
            refresh_value = session_token
        existing_expiry = existing.get("expiry_date")
        default_ttl_ms = 30 * 24 * 60 * 60 * 1000 if not refresh_value else 6 * 60 * 60 * 1000
        try:
            expiry_date = int(existing_expiry)
        except Exception:
            expiry_date = int(time.time() * 1000) + default_ttl_ms
        if expiry_date <= 0 or (access_value and access_value != str(existing.get("access_token", "") or "")):
            expiry_date = int(time.time() * 1000) + default_ttl_ms
        tokens = {
            "access_token": access_value,
            "refresh_token": refresh_value,
            "token_type": str(existing.get("token_type", "Bearer") or "Bearer"),
            "resource_url": str(existing.get("resource_url", "portal.qwen.ai") or "portal.qwen.ai"),
            "expiry_date": expiry_date,
        }
        _save_qwen_tokens(tokens)
        normalized_base_url = base_url.strip().rstrip("/")
        if normalized_base_url:
            save_env_value(QWEN_BASE_URL_ENV, normalized_base_url)
        return {
            "provider": normalized,
            "saved": True,
            "auth_file": str(_qwen_auth_path()),
            "base_url_env": QWEN_BASE_URL_ENV,
        }

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
    if normalized == "qwen-oauth":
        _clear_qwen_tokens()
        save_env_value(QWEN_BASE_URL_ENV, "")
        return {
            "provider": normalized,
            "cleared": True,
            "keys": [QWEN_BASE_URL_ENV],
            "auth_file": str(_qwen_auth_path()),
        }

    keys = set(PROVIDER_AUTH_BUNDLE_KEYS.get(normalized, {}).values())
    keys.add(provider_env_key(normalized))
    for env_key in keys:
        if env_key:
            save_env_value(env_key, "")
    return {"provider": normalized, "cleared": True, "keys": sorted(k for k in keys if k)}
