"""Read Freebuff auth tokens from the local Codebuff CLI credential store."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

CREDENTIALS_PATH = Path.home() / ".config" / "manicode" / "credentials.json"
ENV_TOKEN = "FREEBUFF_TOKEN"
ENV_PROXY_KEY = "FREEBUFF_PROXY_API_KEY"


def credentials_path() -> Path:
    override = (os.environ.get("FREEBUFF_CREDENTIALS_PATH") or "").strip()
    if override:
        return Path(override).expanduser()
    return CREDENTIALS_PATH


def _read_env_key(name: str) -> str:
    home = get_hermes_home()
    env_path = home / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    return (os.environ.get(name) or "").strip()


def _write_env_key(name: str, value: str) -> None:
    home = get_hermes_home()
    env_path = home / ".env"
    lines: list[str] = []
    replaced = False
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{name}="):
                if not replaced:
                    lines.append(f'{name}="{value}"')
                    replaced = True
                continue
            lines.append(line)
    if not replaced:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f'{name}="{value}"')
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_auth_token_from_credentials() -> str:
    """Return Bearer token from ``~/.config/manicode/credentials.json`` if present."""
    path = credentials_path()
    if not path.is_file():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    default = payload.get("default")
    if isinstance(default, dict):
        token = str(default.get("authToken") or "").strip()
        if token:
            return token
    for value in payload.values():
        if isinstance(value, dict):
            token = str(value.get("authToken") or "").strip()
            if token:
                return token
    return ""


def resolve_upstream_token() -> str:
    """Prefer ``FREEBUFF_TOKEN`` in env/.env, else CLI credentials file."""
    env_token = _read_env_key(ENV_TOKEN)
    if env_token:
        return env_token
    return load_auth_token_from_credentials()


def sync_upstream_token_to_env(*, force: bool = False) -> dict[str, Any]:
    existing = _read_env_key(ENV_TOKEN)
    if existing and not force:
        return {
            "ok": True,
            "skipped": True,
            "source": "env",
            "env_path": str(get_hermes_home() / ".env"),
        }
    token = load_auth_token_from_credentials()
    if not token:
        return {
            "ok": False,
            "error": (
                "No Freebuff auth token found. Run `hermes freebuff run`, complete "
                "GitHub login, or set FREEBUFF_TOKEN in ~/.hermes/.env"
            ),
            "credentials_path": str(credentials_path()),
        }
    _write_env_key(ENV_TOKEN, token)
    return {
        "ok": True,
        "action": "written",
        "env_var": ENV_TOKEN,
        "env_path": str(get_hermes_home() / ".env"),
        "token_prefix": token[:12] + "…" if len(token) > 12 else "set",
    }


def ensure_proxy_api_key(*, force: bool = False) -> dict[str, Any]:
    existing = _read_env_key(ENV_PROXY_KEY)
    if existing and not force:
        return {"ok": True, "skipped": True, "api_key_prefix": existing[:8] + "…"}
    import secrets

    key = f"freebuff-local-{secrets.token_urlsafe(24)}"
    _write_env_key(ENV_PROXY_KEY, key)
    return {
        "ok": True,
        "action": "generated",
        "env_var": ENV_PROXY_KEY,
        "api_key_prefix": key[:16] + "…",
    }


def redact_token(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if len(text) <= 12:
        return "***"
    return f"{text[:8]}…{text[-4:]}"
