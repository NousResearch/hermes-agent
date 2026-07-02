"""Audit log for dashboard-auth events.

Profile-aware location: ``$HERMES_HOME/logs/dashboard-auth.log``.
Format: one JSON object per line. Token-like fields are stripped before
serialisation to avoid leaking refresh tokens or JWTs to disk.

This module deliberately keeps a minimal dependency surface — no imports
from ``hermes_constants`` or other hermes_cli modules — so it can be
imported safely from middleware code that loads early in the startup
sequence.
"""
from __future__ import annotations

import datetime as _dt
import enum
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)
_write_lock = threading.Lock()

# Field names that must never appear in the log raw. Any kwarg matching
# these is silently dropped.
_REDACTED_FIELDS: frozenset = frozenset({
    "access_token", "refresh_token", "code", "code_verifier",
    "state", "ticket", "cookie", "Authorization", "authorization",
})


class AuditEvent(enum.Enum):
    """Event types written to dashboard-auth.log.

    Values are the literal ``event`` field on the JSON line.
    """

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


def _resolve_log_path() -> Path:
    """``$HERMES_HOME/logs/dashboard-auth.log`` with the correct platform-aware default.

    Mirrors ``hermes_constants.get_hermes_home`` semantics: env var wins,
    else platform-native path (``%LOCALAPPDATA%/hermes`` on Windows,
    ``~/.hermes`` on POSIX). A local copy avoids an import cycle with the
    middleware which lives below ``hermes_cli``.
    """
    hermes_home_env = os.environ.get("HERMES_HOME")
    if hermes_home_env:
        home = hermes_home_env
    elif sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        if local_appdata:
            home = str(Path(local_appdata) / "hermes")
        else:
            home = str(Path.home() / "AppData" / "Local" / "hermes")
    else:
        home = str(Path.home() / ".hermes")
    return Path(home) / "logs" / "dashboard-auth.log"


def audit_log(event: AuditEvent, **fields: Any) -> None:
    """Append one event to the audit log.

    Token-like fields are dropped. Missing log directory is created.
    Write failures are logged at WARNING but never raise — auth must not
    fail because the audit logger broke.
    """
    safe_fields = {
        k: v for k, v in fields.items()
        if k not in _REDACTED_FIELDS
    }
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "event": event.value,
        **safe_fields,
    }
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    path = _resolve_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as e:
        _log.warning("dashboard-auth audit log write failed: %s", e)
