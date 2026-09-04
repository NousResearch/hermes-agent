"""HMAC verification of Telegram Mini App ``initData``.

Implements Telegram's documented Mini App validation algorithm:

  1. Parse ``initData`` (a URL-encoded query string) into fields.
  2. Build the "data-check-string": every field except ``hash`` itself
     (``signature``, the newer ed25519 third-party-validation field, is
     NOT excluded — it participates in the check string like any other
     field; see ``_data_check_string``), formatted ``key=value``, sorted
     alphabetically by key, joined with ``\\n``.
  3. ``secret_key = HMAC_SHA256(key=b"WebAppData", msg=bot_token)``.
  4. ``expected_hash = HMAC_SHA256(key=secret_key, msg=data_check_string)``.
  5. Constant-time compare ``expected_hash`` (hex) against the ``hash``
     field.
  6. Reject if ``auth_date`` is missing, malformed, or outside the replay
     window (default 60 minutes each direction — also rejects a suspiciously
     future-dated ``auth_date``, which a valid Telegram client never sends).
     The window is a SESSION-lifetime bound, not a per-request one: a Mini
     App is handed ONE ``initData`` string when it opens (Telegram's
     ``WebApp.initData``) and reuses that same string on every API call for
     the entire time it stays open — there is no client-side way to refresh
     it mid-session. So the window has to cover the longest a user might
     realistically keep the app open, or every action silently starts
     failing partway through the session (an original 60s default broke
     every call ~a minute after open, which is what surfaced this). 60
     minutes is a deliberate middle ground: long enough for a normal check-
     the-dashboard session, short enough to bound a captured token's
     usefulness meaningfully tighter than a full day. The real transport
     protection is TLS (the Mini App is served over HTTPS, which Telegram
     requires); this window is defense-in-depth on top of that, not the
     primary defense. Tune via ``dashboard.telegram_miniapp.max_age_seconds``
     (smaller = tighter but risks cutting off longer sessions; larger =
     longer-lived tokens).

NOTE (spec §8): this assumes the Mini App frontend transports ``initData``
as the raw bearer credential (``Authorization: Bearer <initData>``), reusing
the existing dashboard token-auth seam's ``Bearer`` scheme unmodified rather
than adding a new ``tma`` scheme. Telegram's own docs show ``Authorization:
tma <initData>`` as a convention in some SDKs. Confirm which the actual Mini
App frontend sends before wiring this provider into a live deployment — if
it sends ``tma``, ``hermes_cli/dashboard_auth/token_auth.py:extract_bearer_token``
needs a matching scheme addition; this module's verification logic is
unaffected either way, since it just gets handed whatever string followed
the scheme.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Optional
from urllib.parse import parse_qsl

DEFAULT_MAX_AGE_SECONDS = 3600  # 60 minutes — see module docstring (session-lifetime bound)


def _secret_key(bot_token: str) -> bytes:
    return hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()


def _data_check_string(fields: dict) -> str:
    """All received fields except ``hash`` itself, sorted, ``key=value``, newline-joined.

    ONLY ``hash`` is excluded. ``signature`` (the newer Ed25519 field for
    third-party validation) stays IN the check string — per Telegram's docs
    ("a chain of all received fields, sorted alphabetically" minus hash) and
    confirmed empirically against live initData from real Android and
    Windows Telegram clients during rollout: an earlier version of this
    function also excluded ``signature`` (misreading its separate-validation
    role as exclusion from this one), which made every real token fail HMAC
    verification while all synthetic test tokens — minted without a
    ``signature`` field at all — passed.
    """
    return "\n".join(
        f"{key}={value}"
        for key, value in sorted(fields.items())
        if key != "hash"
    )


def verify_init_data(
    init_data: str,
    *,
    bot_token: str,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    now: Optional[int] = None,
) -> Optional[dict]:
    """Verify *init_data* against *bot_token*.

    Returns the parsed field dict (``auth_date``, ``user``, ``query_id``, …)
    on success, or ``None`` on any failure: empty input, malformed query
    string, missing/mismatched hash, missing/malformed ``auth_date``, or
    ``auth_date`` outside the replay window. Never raises — every failure
    mode collapses to ``None`` so the caller (``TelegramMiniAppProvider.
    verify_token``) can fail closed uniformly, matching every other
    ``verify_token`` implementation's contract in this codebase.
    """
    if not init_data or not bot_token:
        return None
    try:
        fields = dict(parse_qsl(init_data, keep_blank_values=True, strict_parsing=True))
    except ValueError:
        return None

    provided_hash = fields.get("hash", "")
    if not provided_hash:
        return None

    check_string = _data_check_string(fields)
    secret_key = _secret_key(bot_token)
    expected_hash = hmac.new(
        secret_key, check_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected_hash, provided_hash):
        return None

    auth_date_raw = fields.get("auth_date", "")
    try:
        auth_date = int(auth_date_raw)
    except ValueError:
        return None

    current = int(time.time()) if now is None else now
    if current - auth_date > max_age_seconds:
        return None  # too old — replay window expired
    if auth_date - current > max_age_seconds:
        return None  # implausibly future-dated — reject rather than trust clock skew

    return fields


def extract_user_id(fields: dict) -> Optional[str]:
    """Pull the Telegram numeric user id out of initData's JSON-encoded ``user`` field."""
    raw_user = fields.get("user", "")
    if not raw_user:
        return None
    try:
        user = json.loads(raw_user)
    except (ValueError, TypeError):
        return None
    if not isinstance(user, dict):
        return None
    user_id = user.get("id")
    if user_id is None:
        return None
    return str(user_id)
