"""Helpers for resolving public, stable Hermes session identity values.

These helpers deliberately avoid any app-specific logic. They expose a
generic, redacted ``binding key`` that skills/tools can use to key durable
state without leaking raw session identifiers.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path


def _normalize_text(value: str | None) -> str:
    return str(value or "").strip()


def _gateway_session_key() -> str:
    try:
        from gateway.session_context import get_session_env

        return _normalize_text(get_session_env("HERMES_SESSION_KEY", ""))
    except Exception:
        return ""


def _hermes_home() -> Path:
    custom = _normalize_text(os.getenv("HERMES_HOME"))
    if custom:
        return Path(custom).expanduser()
    return Path.home() / ".hermes"


def _redacted_binding(label: str, raw_value: str | None) -> str:
    normalized = _normalize_text(raw_value)
    if not normalized:
        return ""
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    return f"hermes:{label}:{digest}"


def _persistent_cli_identity() -> str:
    state_dir = _hermes_home() / "state"
    identity_file = state_dir / "cli_binding_key"
    try:
        if identity_file.exists():
            return _normalize_text(identity_file.read_text(encoding="utf-8"))
        state_dir.mkdir(parents=True, exist_ok=True)
        token = f"cli-local-{secrets.token_hex(16)}"
        identity_file.write_text(token, encoding="utf-8")
        try:
            identity_file.chmod(0o600)
        except Exception:
            pass
        return token
    except Exception:
        return ""


def _stable_cli_cwd_key(cwd: str | None = None) -> str:
    raw_cwd = _normalize_text(cwd) or _normalize_text(os.getenv("TERMINAL_CWD")) or _normalize_text(os.getcwd())
    if not raw_cwd:
        return ""
    try:
        resolved = str(Path(raw_cwd).expanduser().resolve())
    except Exception:
        resolved = raw_cwd
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return f"cli:cwd:{digest}"


def resolve_binding_key(
    *,
    session_id: str | None = None,
    session_key: str | None = None,
    cwd: str | None = None,
) -> str:
    """Return a stable, redacted binding key suitable for durable ownership.

    Resolution order:
    1. Explicit ``HERMES_BINDING_KEY`` (already-final override)
    2. Provided ``session_key`` or gateway session key
    3. Explicit ``HERMES_CLI_BINDING_KEY`` (classic CLI override)
    4. Persistent machine-local CLI identity
    5. Deterministic per-working-directory CLI key

    ``session_id`` is intentionally not used as a binding source because it is
    transient across new sessions and would create unstable account ownership.
    """
    explicit_binding = _normalize_text(os.getenv("HERMES_BINDING_KEY"))
    if explicit_binding:
        return explicit_binding

    resolved_session_key = _normalize_text(session_key) or _gateway_session_key()
    if resolved_session_key:
        return _redacted_binding("sk", resolved_session_key)

    explicit_cli_binding = _normalize_text(os.getenv("HERMES_CLI_BINDING_KEY"))
    if explicit_cli_binding:
        return _redacted_binding("cli", explicit_cli_binding)

    persistent_cli_identity = _persistent_cli_identity()
    if persistent_cli_identity:
        return _redacted_binding("cli", persistent_cli_identity)

    cli_cwd_key = _stable_cli_cwd_key(cwd)
    if cli_cwd_key:
        return _redacted_binding("cwd", cli_cwd_key)

    return "hermes:unknown"
