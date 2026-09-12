"""Audit log for dashboard-auth events: ``$HERMES_HOME/logs/dashboard-auth.log``, one JSON object
per line. Token-like fields are stripped before serialisation so refresh tokens / JWTs never
reach disk. Minimal import surface (no ``hermes_constants`` at import time) so early-loading
middleware can import it."""
from __future__ import annotations

import datetime as _dt
import enum
import json
import logging
import threading
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)
_write_lock = threading.Lock()

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


def resolve_log_path() -> Path:
    """Public path to the dashboard-auth audit log (same resolution as writers use)."""
    return _resolve_log_path()


def audit_log(event: AuditEvent, **fields: Any) -> None:
    """Append one event; token-like fields dropped, log dir created. Write failures are logged at
    WARNING but never raise — auth must not fail because the audit logger broke."""
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "event": event.value,
        **{k: v for k, v in fields.items() if k not in _REDACTED_FIELDS}}
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    path = _resolve_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock, open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        _log.warning("dashboard-auth audit log write failed: %s", e)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
