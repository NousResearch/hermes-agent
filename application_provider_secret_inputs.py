"""Application-owned scoped credential and user-supplied header inputs.

Credential acquisition is retained in the application until Phase 6. Lower
provider/catalogue domains receive resolved facts, never read profile secrets.
"""
from __future__ import annotations

from typing import Any


def scoped_key_env(name: str) -> str:
    if not name:
        return ""
    try:
        from agent.secret_scope import current_secret_scope, get_secret, is_multiplex_active
        if current_secret_scope() is not None or is_multiplex_active():
            return (get_secret(name, "") or "").strip()
        from agent.credential_pool import get_env_prefer_dotenv
        return (get_env_prefer_dotenv(name) or "").strip()
    except Exception:
        return ""


def extra_headers_from_config(entry: Any) -> dict[str, str]:
    if not isinstance(entry, dict):
        return {}
    from hermes_cli.config import normalize_extra_headers
    return normalize_extra_headers(entry.get("extra_headers"))
