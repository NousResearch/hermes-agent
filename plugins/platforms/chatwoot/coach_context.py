"""CRWD Coach context: surface the member's CRWD ``users._id`` to the agent.

On each turn, inject a short context line naming the current Chatwoot member's
CRWD user id so the coach can call ``crwd_db`` ``get_user_gigs`` /
``get_user_receipts`` / ``get_user_products`` **directly** — no ``get_user``
round-trip, and no reliance on the member's email/phone reaching the prompt.

Resolution is **synchronous and self-contained** (``pre_llm_call`` hooks run
sync, like app-chatbot's ``_prefetch_context``), mirroring the ``crwd_handoff``
tool's direct-Chatwoot-API style:

  1. ``contact_id`` = the Chatwoot sender id (from the hook kwargs); account id
     from ``HERMES_SESSION_CHAT_ID`` (``account:conversation``).
  2. ``GET /accounts/{acct}/contacts/{contact_id}`` → ``custom_attributes.
     joincrwd_user_id`` (written by the enrichment pipeline).
  3. Fallback: resolve from CRWD Mongo by the contact's email/phone via
     ``enrichment.fetch_user`` — enrichment is fire-and-forget, so the attribute
     may not be populated on the very first message.
  4. Cache the result per contact id (short TTL) to keep it to one lookup.

Best-effort throughout: any failure returns ``None`` and the coach falls back to
today's ``get_user`` path.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_TIMEOUT_S = 6
_CACHE_TTL_S = 600.0
_CACHE_MAX = 2048
# contact_id -> (crwd_user_id_or_None, monotonic_ts)
_cache: "OrderedDict[str, Tuple[Optional[str], float]]" = OrderedDict()


# --- Chatwoot creds / platform gate -----------------------------------------

def _chatwoot_creds() -> Tuple[str, str]:
    """(base_url, token) for reading a contact.

    The Chatwoot Contacts API is **not authorized for Agent Bots** (HTTP 401), so
    prefer the agent/user token (``CHATWOOT_AGENT_TOKEN``); fall back to the bot
    token only if that's all that's configured.
    """
    base = os.getenv("CHATWOOT_BASE_URL", "").strip().rstrip("/")
    token = (os.getenv("CHATWOOT_AGENT_TOKEN", "") or os.getenv("CHATWOOT_TOKEN", "")).strip()
    return base, token


def _is_chatwoot(platform: Any) -> bool:
    if str(platform or "").strip().lower() == "chatwoot":
        return True
    try:
        from gateway.session_context import get_session_env

        return (get_session_env("HERMES_SESSION_PLATFORM", "") or "").strip().lower() == "chatwoot"
    except Exception:
        return False


def _account_id() -> Optional[str]:
    """Account id for the current conversation (chat id is ``account:conversation``)."""
    try:
        from gateway.session_context import get_session_env
    except Exception:
        return None
    chat_id = (get_session_env("HERMES_SESSION_CHAT_ID", "") or "").strip()
    default_account = os.getenv("CHATWOOT_ACCOUNT_ID", "").strip()
    if ":" in chat_id:
        account = chat_id.partition(":")[0].strip()
        return account or default_account or None
    return default_account or None


# --- Cache ------------------------------------------------------------------

def _cache_get(contact_id: str) -> Tuple[bool, Optional[str]]:
    """Return ``(hit, value)``. ``hit`` is False when absent or expired."""
    entry = _cache.get(contact_id)
    if entry is None:
        return False, None
    value, ts = entry
    if (time.monotonic() - ts) > _CACHE_TTL_S:
        _cache.pop(contact_id, None)
        return False, None
    _cache.move_to_end(contact_id)
    return True, value


def _cache_put(contact_id: str, value: Optional[str]) -> None:
    _cache[contact_id] = (value, time.monotonic())
    _cache.move_to_end(contact_id)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


def _reset_cache() -> None:
    """Test helper — clear the per-contact cache."""
    _cache.clear()


# --- Chatwoot contact read --------------------------------------------------

def _get_contact(account_id: str, contact_id: str) -> Optional[Dict[str, Any]]:
    base, token = _chatwoot_creds()
    if not (base and token):
        return None
    url = f"{base}/api/v1/accounts/{account_id}/contacts/{contact_id}"
    req = urllib.request.Request(url, method="GET", headers={"api_access_token": token})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            if not (200 <= resp.status < 300):
                return None
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, TimeoutError, OSError) as exc:
        logger.debug("[crwd-coach-ctx] get_contact %s failed: %s", contact_id, exc)
        return None
    if isinstance(data, dict):
        # Chatwoot wraps the record under "payload".
        rec = data.get("payload", data)
        return rec if isinstance(rec, dict) else None
    return None


# --- Resolution -------------------------------------------------------------

def resolve_member_crwd_id(contact_id: str) -> Optional[str]:
    """Resolve the current Chatwoot member's CRWD ``users._id``, or ``None``."""
    contact_id = str(contact_id or "").strip()
    if not contact_id:
        return None

    hit, cached = _cache_get(contact_id)
    if hit:
        return cached

    result: Optional[str] = None
    account_id = _account_id()
    contact = _get_contact(account_id, contact_id) if account_id else None

    if contact:
        attrs = contact.get("custom_attributes") or {}
        cid = str(attrs.get("joincrwd_user_id") or "").strip()
        if cid:
            result = cid
        else:
            # Enrichment hasn't populated the attribute yet — resolve from Mongo
            # by the contact's email/phone (same source enrichment uses).
            email = str(contact.get("email") or "").strip() or None
            phone = str(contact.get("phone_number") or "").strip() or None
            if email or phone:
                try:
                    from plugins.platforms.chatwoot import enrichment

                    user = enrichment.fetch_user(email, phone)
                    if user and user.get("_id") is not None:
                        result = str(user["_id"])
                except Exception as exc:
                    logger.debug("[crwd-coach-ctx] mongo fallback failed: %s", exc)

    _cache_put(contact_id, result)
    return result


# --- pre_llm_call hook ------------------------------------------------------

def member_context_hook(**kwargs: Any) -> Optional[Dict[str, str]]:
    """``pre_llm_call`` hook: inject the member's CRWD user id into the prompt."""
    try:
        if not _is_chatwoot(kwargs.get("platform")):
            return None
        if not os.getenv("CRWD_MONGO_URI"):
            return None
        contact_id = str(kwargs.get("sender_id") or "").strip()
        if not contact_id:
            return None
        crwd_id = resolve_member_crwd_id(contact_id)
        if not crwd_id:
            return None
        context = (
            f"[CRWD member] This member's CRWD user_id is {crwd_id}. Use it directly as the "
            "user_id for crwd_db get_user_gigs / get_user_receipts / get_user_products. "
            "Only call get_user when you need to look up a different person."
        )
        return {"context": context}
    except Exception as exc:  # never break a turn over context injection
        logger.debug("[crwd-coach-ctx] hook failed: %s", exc)
        return None
