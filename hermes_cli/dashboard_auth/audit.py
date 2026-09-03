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

# ``ws_ticket_minted`` fires on every dashboard WebSocket (re)connect and,
# with an aggressive UI reconnect loop, dominates dashboard-auth.log without
# carrying information beyond the first mint after an idle gap (#57749 —
# unbounded log growth; ~600KB/day observed on a single-dashboard install).
# Sample it: keep the first mint of every batch of _TICKET_MINT_SAMPLE_EVERY.
# Rejections are security-relevant and are never sampled.
_TICKET_MINT_SAMPLE_EVERY = 50
_ticket_mint_count = 0
_ticket_mint_lock = threading.Lock()
_ticket_mint_since_last_logged = 0


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
    # RFC 8252 native-app (system-browser + loopback + PKCE) flow.
    NATIVE_AUTHORIZE_START = "native_authorize_start"
    NATIVE_CODE_ISSUED = "native_code_issued"
    NATIVE_TOKEN_SUCCESS = "native_token_success"
    NATIVE_TOKEN_FAILURE = "native_token_failure"


def _resolve_log_path() -> Path:
    """``$HERMES_HOME/logs/dashboard-auth.log``.

    Uses ``hermes_constants.get_hermes_home()`` (a leaf module — no import
    cycle) so profile overrides and the native-Windows ``%LOCALAPPDATA%``
    fallback are honored.
    """
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "logs" / "dashboard-auth.log"


def audit_log(event: AuditEvent, **fields: Any) -> None:
    """Append one event to the audit log.

    Token-like fields are dropped. ``ws_ticket_minted`` is sampled (first of
    every ``_TICKET_MINT_SAMPLE_EVERY``) — see the constant's comment for why.
    Missing log directory is created. Write failures are logged at WARNING
    but never raise — auth must not fail because the audit logger broke.
    """
    global _ticket_mint_count, _ticket_mint_since_last_logged
    if event is AuditEvent.WS_TICKET_MINTED:
        with _ticket_mint_lock:
            _ticket_mint_count += 1
            _ticket_mint_since_last_logged += 1
            current_count = _ticket_mint_count
            suppressed_since_last = _ticket_mint_since_last_logged - 1
            should_log = (current_count - 1) % _TICKET_MINT_SAMPLE_EVERY == 0
            if should_log:
                _ticket_mint_since_last_logged = 0
        if not should_log:
            return
        # Add suppressed count to the logged event so operators can see burst patterns
        fields = {**fields, "suppressed_mints_since_last_logged": suppressed_since_last}
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
