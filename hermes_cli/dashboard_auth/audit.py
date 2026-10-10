"""Audit log for dashboard-auth events: ``$HERMES_HOME/logs/dashboard-auth.log``, one JSON object
per line. Token-like fields are stripped before serialisation so refresh tokens / JWTs never
reach disk. Minimal import surface (no ``hermes_constants`` at import time) so early-loading
middleware can import it."""
from __future__ import annotations

import datetime as _dt
import enum
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)
_write_lock = threading.Lock()
_MAX_FIELD_LENGTH = 256
_TRUNCATION_MARKER = "...[truncated]"

# Size-based rotation matching hermes_logging's agent.log / errors.log convention
# (maxBytes=5MB, backupCount=3, stdlib RotatingFileHandler ``.1``/``.2``/``.3`` naming).
# Implemented locally (check-and-rename, stdlib only) rather than importing
# hermes_logging/_ManagedRotatingFileHandler: that would pull in hermes_constants
# (and lazily hermes_cli.config) and break this module's import-light guarantee —
# it is imported by middleware that runs very early in startup.
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3

# Field names that must never appear in the log raw; matching kwargs are dropped.
_REDACTED_FIELDS: frozenset = frozenset({
    "access_token", "refresh_token", "code", "code_verifier",
    "state", "ticket", "cookie", "Authorization", "authorization"})


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
    SESSION_REJECTED = "session_rejected"
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


def _bounded_value(value: Any) -> Any:
    """Bound strings, including nested values and non-JSON object representations."""
    if isinstance(value, dict):
        return {k: _bounded_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bounded_value(v) for v in value]
    if not isinstance(value, (str, int, float, bool, type(None))):
        value = repr(value)
    if isinstance(value, str) and len(value) > _MAX_FIELD_LENGTH:
        return value[:_MAX_FIELD_LENGTH - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER
    return value


def _rotate_if_needed(path: Path, pending_bytes: int) -> None:
    """Rotate ``path`` to ``.1``/…/``.<_BACKUP_COUNT>`` once appending *pending_bytes* would
    reach ``_MAX_BYTES``. Mirrors stdlib ``RotatingFileHandler``'s naming; must be called
    with ``_write_lock`` held. Rotation errors must not break the write path."""
    try:
        if path.stat().st_size + pending_bytes < _MAX_BYTES:
            return
    except OSError:
        return  # missing/vanished mid-check; the append itself surfaces real errors
    try:
        for i in range(_BACKUP_COUNT - 1, 0, -1):
            src = path.with_name(f"{path.name}.{i}")
            if src.exists():
                os.replace(src, path.with_name(f"{path.name}.{i + 1}"))
        os.replace(path, path.with_name(f"{path.name}.1"))
    except OSError:
        pass  # a log that cannot be rotated is still a working (if overlong) log


def audit_log(event: AuditEvent, **fields: Any) -> None:
    """Append one event; token-like fields dropped, log dir created, size-capped rotation
    (``_MAX_BYTES`` with ``_BACKUP_COUNT`` backups). Write failures are logged at WARNING
    but never raise — auth must not fail because the audit logger broke."""
    try:
        entry = {
            "ts": _dt.datetime.now(_dt.UTC).isoformat(),
            "event": event.value,
            **{k: _bounded_value(v) for k, v in fields.items() if k not in _REDACTED_FIELDS}}
        line = json.dumps(entry, separators=(",", ":")) + "\n"
        line_bytes = line.encode("utf-8")
        path = _resolve_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            _rotate_if_needed(path, len(line_bytes))
            with open(path, "ab") as f:
                f.write(line_bytes)
    except Exception as e:
        _log.warning("dashboard-auth audit log write failed: %s", e)
