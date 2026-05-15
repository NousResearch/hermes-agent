"""Helpers for deriving a safe, stable binding identity."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def _normalize_text(value: str | None) -> str:
    return str(value or "").strip()


def _redacted_binding(label: str, raw_value: str | None) -> str:
    normalized = _normalize_text(raw_value)
    if not normalized:
        return ""
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    return f"hermes:{label}:{digest}"


def _stable_cli_cwd_key(cwd: str | None = None) -> str:
    raw_cwd = (
        _normalize_text(cwd)
        or _normalize_text(os.getenv("TERMINAL_CWD"))
        or _normalize_text(os.getcwd())
    )
    if not raw_cwd:
        return ""
    try:
        resolved = str(Path(raw_cwd).expanduser().resolve())
    except Exception:
        resolved = raw_cwd
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return f"cli:cwd:{digest}"


def resolve_binding_key(*, session_key: str | None = None, cwd: str | None = None) -> str:
    """Return a redacted binding key safe to expose to skills/tools."""
    explicit_binding = _normalize_text(os.getenv("HERMES_BINDING_KEY"))
    if explicit_binding:
        return explicit_binding

    resolved_session_key = _normalize_text(session_key)
    if not resolved_session_key:
        try:
            from gateway.session_context import get_session_env

            resolved_session_key = _normalize_text(
                get_session_env("HERMES_SESSION_KEY", "")
            )
        except Exception:
            resolved_session_key = ""
    if resolved_session_key:
        return _redacted_binding("sk", resolved_session_key)

    cli_cwd_key = _stable_cli_cwd_key(cwd)
    if cli_cwd_key:
        return _redacted_binding("cwd", cli_cwd_key)

    return "hermes:unknown"
