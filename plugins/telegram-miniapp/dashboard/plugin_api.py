"""Backend for the Telegram Mini App dashboard, mounted under ``/api/plugins/telegram-miniapp/``:
the Telegram allowlist (admin tier) and ``me`` (the caller's tier).

The Mini App UI is its own React root served at ``/miniapp`` (``web/src/miniapp``); the
``dashboard_auth/telegram_miniapp`` provider verifies its ``initData`` and registers these routes
for token auth. Scoped narrowly to ``TELEGRAM_ALLOWED_USERS`` rather than exposing the generic ``/api/env`` to a Mini
App bearer token. Admin and requester-scope checks live in ``hermes_cli.web_server`` and are resolved
late at call time (cycle-safe).
"""

import asyncio
import logging
import os
import re
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from hermes_cli.web_deps import late

router = APIRouter()

_dashboard_requester_scope = late("_dashboard_requester_scope")
_require_dashboard_admin = late("_require_dashboard_admin")


# ---------------------------------------------------------------------------
# Telegram allowlist endpoints — scoped narrowly to TELEGRAM_ALLOWED_USERS,
# for the Mini App's Users tab. Deliberately NOT the generic GET/PUT /api/env
# (which reads/writes arbitrary keys, including API keys) exposed to a Mini
# App bearer token: that would be a far broader admin-tier surface than
# "manage who's allowed to DM the bot", the one thing this tab does. All
# three are admin-tier only (_require_dashboard_admin) -- the paired/"member"
# tier never sees this tab in the first place (Cron/Users are hidden
# entirely for non-admin in the Mini App shell), so nothing below needs a
# per-row ownership shape, only the flat gate.
#
# A user_id can come from two independent sources that this endpoint merges
# for display, matching the union `is_authorized()` already reads (env
# allowlist OR pairing store):
#   - PairingStore's approved list (gateway/pairing.py) -- has user_name and
#     approved_at, populated by the code-based pairing flow.
#   - The raw TELEGRAM_ALLOWED_USERS env var -- a bare numeric id with no
#     metadata, e.g. someone added directly via `hermes setup` or hand-edited
#     .env, never paired at all.
# An id present in both is reported once, preferring the pairing store's
# richer metadata. For any entry still missing a username/name after that,
# _resolve_telegram_profiles() makes a best-effort Bot API getChat call:
# Telegram's Bot API has no *generic* id->profile lookup, but getChat DOES
# return the profile for a user the bot has interacted with -- and an
# allowlisted user has almost always messaged the bot (that's typically how
# they got allowlisted). It's best-effort: cached, short-timeout, and any
# id it can't resolve (never messaged the bot, privacy settings, network
# blip) simply keeps username/name null and the frontend's "name
# unavailable" fallback, exactly as before.
# ---------------------------------------------------------------------------


def _split_allowlist_ids(raw: str) -> list[str]:
    return [uid.strip() for uid in raw.split(",") if uid.strip()]


# Cache getChat results (positive AND negative) so the Users tab doesn't
# re-hit the Bot API for every id on every load. {uid: (fetched_at, profile)}
# where profile is {"username": str|None, "name": str|None}.
_TELEGRAM_PROFILE_CACHE: dict[str, tuple[float, dict]] = {}
_TELEGRAM_PROFILE_TTL = 3600.0  # 1h — a user who messages the bot resolves within the hour


async def _resolve_telegram_profiles(user_ids: list[str]) -> dict[str, dict]:
    """Best-effort {uid: {"username", "name"}} via the Bot API's getChat.

    Reads the bot token from the machine ``.env`` file (not ``os.environ``,
    which may have the token scrubbed post-startup). Concurrent, short
    per-call timeout, every failure swallowed to an absent entry -- this
    only ever ADDS names it can resolve; it never blocks or errors the
    allowlist response.
    """
    now = time.time()
    resolved: dict[str, dict] = {}
    to_fetch: list[str] = []
    for uid in user_ids:
        cached = _TELEGRAM_PROFILE_CACHE.get(uid)
        if cached and now - cached[0] < _TELEGRAM_PROFILE_TTL:
            resolved[uid] = cached[1]
        else:
            to_fetch.append(uid)
    if not to_fetch:
        return resolved

    try:
        from hermes_cli.config import load_env
        bot_token = (load_env().get("TELEGRAM_BOT_TOKEN") or "").strip()
    except (OSError, ValueError):
        bot_token = ""
    if not bot_token:
        return resolved

    import httpx

    # httpx logs every request URL at INFO, and the bot token is part of a Bot API URL. Floor its
    # logger at WARNING for this call (restored in ``finally``) instead of relying on the app-wide
    # noisy-logger list, which exists for noise, not secrecy. Overlapping calls can interleave the
    # restore; that only matters if someone lowered the httpx level below WARNING on purpose.
    _httpx_logger = logging.getLogger("httpx")
    _prev_httpx_level = _httpx_logger.level
    if _httpx_logger.level == logging.NOTSET or _httpx_logger.level < logging.WARNING:
        _httpx_logger.setLevel(logging.WARNING)

    async def _fetch(client: "httpx.AsyncClient", uid: str) -> None:
        profile = {"username": None, "name": None}
        try:
            r = await client.get(
                f"https://api.telegram.org/bot{bot_token}/getChat",
                params={"chat_id": uid},
            )
            if r.status_code == 200:
                data = r.json()
                if data.get("ok"):
                    res = data.get("result") or {}
                    name = " ".join(
                        p for p in (res.get("first_name"), res.get("last_name")) if p
                    ).strip()
                    profile = {"username": res.get("username"), "name": name or None}
        except (httpx.HTTPError, ValueError):
            # Keep the null profile (negative-cached below). Not logged: httpx errors carry the URL,
            # which contains the bot token.
            pass
        _TELEGRAM_PROFILE_CACHE[uid] = (now, profile)
        resolved[uid] = profile

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(4.0)) as client:
            await asyncio.gather(*(_fetch(client, uid) for uid in to_fetch))
    finally:
        _httpx_logger.setLevel(_prev_httpx_level)
    return resolved


@router.get("/allowlist")
async def get_telegram_allowlist(request: Request):
    _require_dashboard_admin(request)
    from gateway.pairing import PairingStore

    store = PairingStore()
    by_id: dict[str, dict] = {}
    for entry in store.list_approved("telegram"):
        uid = str(entry.get("user_id", ""))
        if not uid:
            continue
        by_id[uid] = {
            "user_id": uid,
            "username": None,  # PairingStore doesn't separately track @handle
            "name": entry.get("user_name") or None,
            "added_at": entry.get("approved_at"),
            "source": "pairing",
        }

    raw = os.environ.get("TELEGRAM_ALLOWED_USERS", "").strip()
    for uid in _split_allowlist_ids(raw):
        if uid == "*" or uid in by_id:
            continue
        by_id[uid] = {
            "user_id": uid,
            "username": None,
            "name": None,
            "added_at": None,
            "source": "env",
        }

    # Best-effort fill in username/display name for any entry still missing
    # both (getChat; see _resolve_telegram_profiles). Never fails the
    # response -- an id it can't resolve keeps its null fields.
    unresolved = [uid for uid, e in by_id.items() if not e["username"] and not e["name"]]
    if unresolved:
        profiles = await _resolve_telegram_profiles(unresolved)
        for uid, profile in profiles.items():
            entry = by_id.get(uid)
            if entry is None:
                continue
            if profile.get("username"):
                entry["username"] = profile["username"]
            if profile.get("name") and not entry["name"]:
                entry["name"] = profile["name"]

    return {"allowlist": list(by_id.values())}


class TelegramAllowlistAdd(BaseModel):
    user_id: str


_TELEGRAM_USER_ID_RE = re.compile(r"^\d{5,15}$")


@router.post("/allowlist")
async def add_telegram_allowlist_entry(request: Request, body: TelegramAllowlistAdd):
    _require_dashboard_admin(request)
    uid = (body.user_id or "").strip()
    if not _TELEGRAM_USER_ID_RE.match(uid):
        raise HTTPException(
            status_code=400,
            detail="user_id must be a numeric Telegram user id (5-15 digits).",
        )

    raw = os.environ.get("TELEGRAM_ALLOWED_USERS", "").strip()
    ids = _split_allowlist_ids(raw)
    if uid in ids or "*" in ids:
        return {"ok": True, "already_present": True}

    # Unconditional append -- unlike gateway/pairing.py's _sync_allowlist_add
    # (which no-ops when the var is unset, to avoid a passive pairing
    # approval silently locking down a previously-open gateway), this is an
    # explicit admin action from the Users tab: the admin typed an id and
    # tapped Add, so writing it must always take effect, empty-var case
    # included -- an unset TELEGRAM_ALLOWED_USERS should not silently
    # swallow a deliberate add.
    ids.append(uid)
    from hermes_cli.config import save_env_value

    save_env_value("TELEGRAM_ALLOWED_USERS", ",".join(ids))
    return {"ok": True, "user_id": uid}


@router.delete("/allowlist/{user_id}")
async def remove_telegram_allowlist_entry(request: Request, user_id: str):
    _require_dashboard_admin(request)
    from gateway.pairing import PairingStore

    store = PairingStore()
    # revoke() also mirrors the removal into TELEGRAM_ALLOWED_USERS when a
    # pairing-derived entry has one (gateway/pairing.py's _sync_allowlist_remove),
    # so this half handles anyone who came through pairing.
    store.revoke("telegram", user_id)

    # Independently strip the id from the raw env var too, for the
    # env-only-source case revoke() never touches (added directly, never
    # paired) -- a no-op if it's already gone.
    raw = os.environ.get("TELEGRAM_ALLOWED_USERS", "").strip()
    ids = _split_allowlist_ids(raw)
    remaining = [i for i in ids if i != user_id]
    if len(remaining) != len(ids):
        from hermes_cli.config import remove_env_value, save_env_value

        if remaining:
            save_env_value("TELEGRAM_ALLOWED_USERS", ",".join(remaining))
        else:
            remove_env_value("TELEGRAM_ALLOWED_USERS")

    return {"ok": True}


@router.get("/me")
async def get_miniapp_me(request: Request):
    """Tells the Mini App frontend its own tier on mount.

    ``/api/auth/me`` (hermes_cli/dashboard_auth/routes.py) only recognizes a
    cookie session and 401s for a bearer-token caller, so it can't serve
    this purpose — a Mini App request never carries a cookie. Registered as
    a Mini App token route (required=False) so a cookie caller (e.g. the
    desktop dashboard previewing the Mini App views) also gets a sensible
    answer here instead of needing a separate code path.

    Reuses _dashboard_requester_scope rather than re-deriving tier — same
    reason as _require_dashboard_admin: this is the read-only counterpart,
    not a fourth independent trust classification.
    """
    scope, requester_user_id = _dashboard_requester_scope(request)
    if scope in (None, "admin"):
        # Cookie/session desktop operator (scope None) and admin-scoped
        # Mini App tokens both get the same unrestricted tier.
        return {"tier": "admin", "user_id": None}
    if requester_user_id:
        return {"tier": "paired", "user_id": requester_user_id}
    # scope == "own" with no usable id: deny-by-default per
    # _dashboard_requester_scope's own contract (e.g. a token provider this
    # scoping logic doesn't recognize) -- not a real paired principal.
    return {"tier": None, "user_id": None}
