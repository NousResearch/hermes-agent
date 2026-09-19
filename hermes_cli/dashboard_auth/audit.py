"""Audit log for dashboard-auth events.

Profile-aware location: ``$HERMES_HOME/logs/dashboard-auth.log``.
Format: one JSON object per line. Token-like fields are stripped before
serialisation to avoid leaking refresh tokens or JWTs to disk.

Unlike a plain append, writes go through a ``RotatingFileHandler`` so the
log cannot grow without bound from an unauthenticated endpoint (upstream
#98338). Rotation honours ``logging.max_size_mb`` / ``logging.backup_count``
from ``config.yaml`` (defaults 5 MB and 3 backups).

This module deliberately keeps a minimal dependency surface — no imports
from ``hermes_constants`` or other hermes_cli modules — so it can be
imported safely from middleware code that loads early in the startup
sequence. Config reads use lazy imports inside the function, mirroring
``_resolve_log_path``.
"""
from __future__ import annotations

import datetime as _dt
import enum
import json
import logging
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)
_write_lock = threading.Lock()

# Default rotation policy when config.yaml is absent or unreadable.
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024
_DEFAULT_BACKUP_COUNT = 3

# Field names that must never appear in the log raw; matching kwargs are dropped.
_REDACTED_FIELDS: frozenset = frozenset({
    "access_token", "refresh_token", "code", "code_verifier",
    "state", "ticket", "cookie", "Authorization", "authorization"})

# Cache of {resolved_log_path: RotatingFileHandler}. Rebuilt lazily so a
# HERMES_HOME / profile change in the process picks up the new path (and
# rotation policy) on the next write.
_handlers: dict[Path, RotatingFileHandler] = {}


class AuditEvent(enum.Enum):
    """Event types; values are the literal ``event`` field on the JSON line."""
    LOGIN_START = "login_start"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    REFRESH_SUCCESS = "refresh_success"
    REFRESH_FAILURE = "refresh_failure"
    REVOKE = "revoke"
    SESSION_VERIFY_FAILURE = "session_verify_failure"
    WS_TICKET_MINTED = "ws_ticket_minted"
    WS_TICKET_REJECTED = "ws_ticket_rejected"
    TOKEN_AUTH_SUCCESS = "token_auth_success"
    TOKEN_AUTH_FAILURE = "token_auth_failure"
    # RFC 8252 native-app (system-browser + loopback + PKCE) flow.
    NATIVE_AUTHORIZE_START = "native_authorize_start"
    NATIVE_CODE_ISSUED = "native_code_issued"
    NATIVE_TOKEN_SUCCESS = "native_token_success"
    NATIVE_TOKEN_FAILURE = "native_token_failure"


def _resolve_log_path() -> Path:
    """Lazy leaf import: honours profile overrides + the native-Windows ``%LOCALAPPDATA%`` fallback."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "logs" / "dashboard-auth.log"


def _rotation_policy() -> tuple[int, int]:
    """Return ``(max_bytes, backup_count)`` from ``logging`` config.

    Reads ``logging.max_size_mb`` / ``logging.backup_count`` with a lazy
    import (load_config can raise on malformed YAML / IO), falling back to
    the module defaults. ``max_size_mb`` is floored at 1 so a 0/negative
    value cannot disable rotation silently.
    """
    max_bytes, backup_count = _DEFAULT_MAX_BYTES, _DEFAULT_BACKUP_COUNT
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception:  # noqa: BLE001 — robust to config being unavailable early
        return (max_bytes, _DEFAULT_BACKUP_COUNT)

    if not isinstance(cfg, dict):
        return (max_bytes, _DEFAULT_BACKUP_COUNT)

    logging_cfg = cfg.get("logging")
    if not isinstance(logging_cfg, dict):
        return (max_bytes, _DEFAULT_BACKUP_COUNT)

    try:
        size_mb = int(logging_cfg.get("max_size_mb", 5))
        max_bytes = max(1, size_mb) * 1024 * 1024
    except (TypeError, ValueError):
        max_bytes = _DEFAULT_MAX_BYTES

    try:
        backup_count = int(logging_cfg.get("backup_count", 3))
        if backup_count < 0:
            backup_count = _DEFAULT_BACKUP_COUNT
    except (TypeError, ValueError):
        backup_count = _DEFAULT_BACKUP_COUNT

    return (max_bytes, backup_count)


def _get_handler(path: Path) -> RotatingFileHandler:
    """Return a cached RotatingFileHandler for *path*, re-resolving when the
    path or rotation policy changes (e.g. HERMES_HOME moved in-process)."""
    handler = _handlers.get(path)
    if handler is not None:
        max_bytes, _ = _rotation_policy()
        # Rebuild if the configured max size changed (backup_count changing
        # alone is cosmetic; rebuilds are cheap and only on writes).
        if handler.maxBytes == max_bytes:
            return handler
        handler.close()
        _handlers.pop(path, None)

    max_bytes, backup_count = _rotation_policy()
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        str(path),
        mode="a",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    # Keep audit records out of the standard-library root logger graph; they
    # are written directly as raw lines to the file.
    handler.setLevel(logging.INFO)
    # Evict handlers for a DIFFERENT resolved path (HERMES_HOME / profile
    # moved in-process): leaving them in the cache would leak their fds.
    # Path-change rotations are rare, so a linear sweep on rebuild is fine.
    for old_path, old_handler in list(_handlers.items()):
        if old_path != path:
            old_handler.close()
            _handlers.pop(old_path, None)
    _handlers[path] = handler
    return handler


def audit_log(event: AuditEvent, **fields: Any) -> None:
    """Append one event; token-like fields dropped, log dir created. Write failures are logged at
    WARNING but never raise — auth must not fail because the audit logger broke."""
    safe_fields = {
        k: v for k, v in fields.items()
        if k not in _REDACTED_FIELDS
    }
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "event": event.value,
        **safe_fields,
    }
    line = json.dumps(entry, separators=(",", ":"))  # RotatingFileHandler appends "\n"
    try:
        path = _resolve_log_path()
        with _write_lock:
            handler = _get_handler(path)
            handler.emit(
                logging.LogRecord(
                    name=__name__,
                    level=logging.INFO,
                    pathname=__file__,
                    lineno=0,
                    msg="%s",
                    args=(line,),
                    exc_info=None,
                )
            )
    except Exception as e:
        _log.warning("dashboard-auth audit log write failed: %s", e)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
