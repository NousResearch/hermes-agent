"""Telegram WebApp initData verification for the Mini App sidecar."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import re
import time
from typing import Iterable
from urllib.parse import unquote_plus


_BAD_PERCENT_RE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class InitDataAuthError(ValueError):
    """Safe auth error whose message never includes raw secrets/payloads."""

    def __init__(self, message: str, *, status_code: int = 401):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class TelegramMiniAppUser:
    id: str
    username: str | None = None
    first_name: str | None = None
    last_name: str | None = None


@dataclass(frozen=True)
class VerifiedInitData:
    user: TelegramMiniAppUser
    auth_date: int
    fingerprint: str
    # Contract guard: callers must not retain raw initData in sessions.
    raw_init_data: None = None


def _strict_unquote(value: str) -> str:
    if _BAD_PERCENT_RE.search(value):
        raise InitDataAuthError("Malformed initData encoding", status_code=400)
    try:
        return unquote_plus(value, encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise InitDataAuthError("Malformed initData encoding", status_code=400) from exc


def _parse_query(init_data: str) -> dict[str, str]:
    if not init_data:
        raise InitDataAuthError("Missing initData", status_code=400)

    fields: dict[str, str] = {}
    for part in init_data.split("&"):
        if not part:
            continue
        raw_key, sep, raw_value = part.partition("=")
        if not sep:
            raise InitDataAuthError("Malformed initData", status_code=400)
        key = _strict_unquote(raw_key)
        value = _strict_unquote(raw_value)
        if key in fields:
            raise InitDataAuthError("Duplicate initData key", status_code=400)
        fields[key] = value
    return fields


def _coerce_allowed_users(allowed_users: Iterable[str | int]) -> set[str]:
    return {str(user_id).strip() for user_id in allowed_users if str(user_id).strip()}


def verify_init_data(
    init_data: str,
    *,
    bot_token: str,
    allowed_users: Iterable[str | int],
    now: int | float | None = None,
    ttl_seconds: int = 300,
    future_skew_seconds: int = 60,
) -> VerifiedInitData:
    """Verify Telegram WebApp initData and return a safe identity.

    Implements Telegram's WebApp HMAC contract. The returned object stores only
    a SHA-256 fingerprint of the raw initData; raw payload/hash/bot token are
    intentionally not retained.
    """
    if not bot_token:
        raise InitDataAuthError("Telegram Mini App auth is not configured", status_code=503)

    fields = _parse_query(init_data)
    provided_hash = fields.get("hash")
    if provided_hash is None:
        raise InitDataAuthError("Missing initData hash", status_code=400)
    if not _HASH_RE.match(provided_hash):
        raise InitDataAuthError("Invalid initData hash", status_code=400)

    signed_fields = {key: value for key, value in fields.items() if key != "hash"}
    data_check_string = "\n".join(f"{key}={signed_fields[key]}" for key in sorted(signed_fields))
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    expected_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_hash, provided_hash.lower()):
        raise InitDataAuthError("Invalid initData signature", status_code=401)

    try:
        auth_date = int(signed_fields["auth_date"])
    except KeyError as exc:
        raise InitDataAuthError("Missing auth_date", status_code=400) from exc
    except ValueError as exc:
        raise InitDataAuthError("Invalid auth_date", status_code=400) from exc

    current = int(time.time() if now is None else now)
    if auth_date < current - int(ttl_seconds):
        raise InitDataAuthError("Expired initData", status_code=401)
    if auth_date > current + int(future_skew_seconds):
        raise InitDataAuthError("initData auth_date is in the future", status_code=401)

    try:
        user_payload = json.loads(signed_fields["user"])
    except KeyError as exc:
        raise InitDataAuthError("Missing user", status_code=400) from exc
    except json.JSONDecodeError as exc:
        raise InitDataAuthError("Malformed user", status_code=400) from exc
    if not isinstance(user_payload, dict):
        raise InitDataAuthError("Malformed user", status_code=400)

    raw_user_id = user_payload.get("id")
    if isinstance(raw_user_id, bool) or raw_user_id is None:
        raise InitDataAuthError("Invalid user id", status_code=400)
    if not isinstance(raw_user_id, (str, int)):
        raise InitDataAuthError("Invalid user id", status_code=400)
    user_id = str(raw_user_id).strip()
    if not user_id:
        raise InitDataAuthError("Invalid user id", status_code=400)

    allowlist = _coerce_allowed_users(allowed_users)
    if not allowlist or user_id not in allowlist:
        raise InitDataAuthError("Telegram user is not allowed", status_code=403)

    user = TelegramMiniAppUser(
        id=user_id,
        username=user_payload.get("username") if isinstance(user_payload.get("username"), str) else None,
        first_name=user_payload.get("first_name") if isinstance(user_payload.get("first_name"), str) else None,
        last_name=user_payload.get("last_name") if isinstance(user_payload.get("last_name"), str) else None,
    )
    return VerifiedInitData(
        user=user,
        auth_date=auth_date,
        fingerprint=hashlib.sha256(init_data.encode()).hexdigest(),
    )
