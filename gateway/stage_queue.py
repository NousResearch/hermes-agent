"""
Message Staging Queue — builtin gateway hook.

Intercepts messages arriving while a session is active, persists them to
Supabase `message_stage_queue`, and drains one-at-a-time on turn completion.
Telegram reactions: 🕐 staged → ▶️ released → cleared on answer.

Phase 1: Telegram lane only (DEV cudbvjvbkifrnszpybns).
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger("hermes.gateway.stage_queue")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_STAGING_ENABLED = os.getenv("HERMES_STAGING_QUEUE", "").lower() in ("1", "true", "yes")
_QUEUE_DEPTH_ALARM = int(os.getenv("HERMES_STAGING_ALARM_DEPTH", "8"))
_QUEUE_AGE_ALARM_S = int(os.getenv("HERMES_STAGING_ALARM_AGE_S", "600"))

# ---------------------------------------------------------------------------
# Supabase rail (service_role — the ONLY writer per E-RLS)
# ---------------------------------------------------------------------------

def _sb_url() -> str:
    return os.getenv("SUPABASE_URL", "")

def _sb_key() -> str:
    return os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

async def _sb_rpc(method: str, table: str, payload: Optional[dict] = None,
                   query: Optional[str] = None) -> dict:
    """Minimal Supabase PostgREST call. Returns parsed JSON or raises."""
    base = _sb_url().rstrip("/")
    key = _sb_key()
    if not base or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set")
    url = f"{base}/rest/v1/{table}"
    if query:
        url += f"?{query}"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await getattr(client, method)(url, json=payload, headers=headers)
    if resp.status_code >= 400:
        logger.error("[stage-queue] Supabase %s %s → %s %s", method.upper(), table, resp.status_code, resp.text[:200])
        resp.raise_for_status()
    return resp.json() if resp.text.strip() else {}


# ---------------------------------------------------------------------------
# Stage: persist inbound message
# ---------------------------------------------------------------------------

async def stage_message(
    platform: str,
    chat_id: str,
    session_key: str,
    message_id: str,
    sender_id: str,
    sender_name: str,
    event_json: str,
    bot_name: str = "",
) -> dict:
    """INSERT a message into the staging queue. Returns the inserted row."""
    row = {
        "session_key": session_key,
        "platform": platform,
        "chat_id": chat_id,
        "bot_name": bot_name,
        "message_id": message_id or "",
        "sender_id": sender_id or "",
        "sender_name": sender_name or "",
        "status": "staged",
        "attempts": 0,
        "event": json.loads(event_json) if isinstance(event_json, str) else event_json,
    }
    result = await _sb_rpc("post", "message_stage_queue", payload=row)
    logger.info("[stage-queue] staged msg_id=%s session=%s depth=%s",
                message_id, session_key, await _queue_depth(session_key))
    return result


# ---------------------------------------------------------------------------
# Drain: pop next staged message for a session
# ---------------------------------------------------------------------------

async def drain_next(session_key: str) -> Optional[dict]:
    """Pop the oldest staged message for this session (FIFO by seq/created_at).
    Sets status='processing' atomically. Returns the row or None."""
    # Fetch oldest staged
    query = quote(f"session_key=eq.{session_key}&status=eq.staged&order=seq.asc,created_at.asc&limit=1")
    rows = await _sb_rpc("get", "message_stage_queue", query=query)
    if not rows:
        return None
    row = rows[0]
    # CAS: update status to 'processing' WHERE id=row.id AND status='staged'
    cas_query = quote(f"id=eq.{row['id']}&status=eq.staged")
    update = {"status": "processing", "updated_at": datetime.now(timezone.utc).isoformat()}
    updated = await _sb_rpc("patch", "message_stage_queue", payload=update, query=cas_query)
    if not updated:
        # Lost the CAS race — another worker grabbed it
        logger.debug("[stage-queue] CAS race lost for id=%s, retrying", row["id"])
        return await drain_next(session_key)
    return updated[0] if isinstance(updated, list) and updated else row


# ---------------------------------------------------------------------------
# Mark done / dead-letter
# ---------------------------------------------------------------------------

async def mark_restaged(msg_id: str) -> None:
    """Re-stage a processing message (session was still active or start rejected)."""
    query = quote(f"id=eq.{msg_id}")
    await _sb_rpc("patch", "message_stage_queue",
                  payload={"status": "staged", "updated_at": datetime.now(timezone.utc).isoformat()},
                  query=query)

async def mark_done(msg_id: str) -> None:
    query = quote(f"id=eq.{msg_id}")
    await _sb_rpc("patch", "message_stage_queue",
                  payload={"status": "done", "updated_at": datetime.now(timezone.utc).isoformat()},
                  query=query)

async def mark_dead(msg_id: str, error: str) -> None:
    query = quote(f"id=eq.{msg_id}")
    await _sb_rpc("patch", "message_stage_queue",
                  payload={"status": "dead", "last_error": error[:500],
                           "updated_at": datetime.now(timezone.utc).isoformat()},
                  query=query)

async def bump_attempt(msg_id: str) -> int:
    """Increment attempts counter, return new value."""
    # Read-then-write (not ideal but OK for low-contention single-writer)
    query = quote(f"id=eq.{msg_id}&select=attempts")
    rows = await _sb_rpc("get", "message_stage_queue", query=query)
    cur = (rows[0]["attempts"] if rows else 0) + 1
    await _sb_rpc("patch", "message_stage_queue",
                  payload={"attempts": cur, "updated_at": datetime.now(timezone.utc).isoformat()},
                  query=quote(f"id=eq.{msg_id}"))
    return cur


# ---------------------------------------------------------------------------
# Depth / age queries (for alarms and /queue command)
# ---------------------------------------------------------------------------

async def _queue_depth(session_key: str) -> int:
    query = quote(f"session_key=eq.{session_key}&status=eq.staged&select=id")
    rows = await _sb_rpc("get", "message_stage_queue", query=query)
    return len(rows) if isinstance(rows, list) else 0

async def queue_depth_all() -> Dict[str, int]:
    """Fleet-wide depth: {session_key: count} for all staged rows."""
    query = quote("status=eq.staged&select=session_key")
    rows = await _sb_rpc("get", "message_stage_queue", query=query)
    counts: Dict[str, int] = {}
    for r in (rows if isinstance(rows, list) else []):
        sk = r.get("session_key", "")
        counts[sk] = counts.get(sk, 0) + 1
    return counts

async def oldest_staged_age(session_key: str) -> Optional[float]:
    """Seconds since the oldest staged message was created, or None if empty."""
    query = quote(f"session_key=eq.{session_key}&status=eq.staged&order=created_at.asc&limit=1&select=created_at")
    rows = await _sb_rpc("get", "message_stage_queue", query=query)
    if not rows:
        return None
    created = datetime.fromisoformat(rows[0]["created_at"].replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - created).total_seconds()


# ---------------------------------------------------------------------------
# Telegram reaction cues
# ---------------------------------------------------------------------------

_CLOCK = "🕐"
_PLAY = "▶️"
_CHECK = "✅"

async def set_staged_reaction(adapter, chat_id: str, message_id: str) -> None:
    add = getattr(adapter, "_add_reaction", None)  # type: ignore[misc]
    if callable(add):
        try:
            await add(chat_id, message_id, _CLOCK)  # type: ignore[misc]
        except Exception:
            pass

async def set_released_reaction(adapter, chat_id: str, message_id: str) -> None:
    add = getattr(adapter, "_add_reaction", None)  # type: ignore[misc]
    remove = getattr(adapter, "_remove_reaction", None)  # type: ignore[misc]
    if callable(add) and callable(remove):
        try:
            await remove(chat_id, message_id)  # type: ignore[misc]
            await add(chat_id, message_id, _PLAY)  # type: ignore[misc]
        except Exception:
            pass

async def clear_reaction(adapter, chat_id: str, message_id: str) -> None:
    remove = getattr(adapter, "_remove_reaction", None)  # type: ignore[misc]
    if callable(remove):
        try:
            await remove(chat_id, message_id)  # type: ignore[misc]
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Alarm (depth or age threshold → ONE alert, no autonomous action)
# ---------------------------------------------------------------------------

_alarm_fired: Dict[str, float] = {}  # session_key -> last alarm timestamp

async def check_alarm(session_key: str) -> Optional[str]:
    """Return alarm message if threshold breached, else None. Fires at most once per session per 30min."""
    depth = await _queue_depth(session_key)
    age = await oldest_staged_age(session_key) or 0
    if depth >= _QUEUE_DEPTH_ALARM or age >= _QUEUE_AGE_ALARM_S:
        now = time.monotonic()
        if now - _alarm_fired.get(session_key, 0) > 1800:
            _alarm_fired[session_key] = now
            return f"⚠️ Staging queue alarm: session={session_key} depth={depth} oldest={age:.0f}s"
    return None


# ---------------------------------------------------------------------------
# Enqueue helper — called from the adapter's _handle_message_while_active
# ---------------------------------------------------------------------------

async def enqueue_from_event(adapter, event, session_key: str) -> bool:
    """Persist the event to the staging queue + add clock reaction.
    Returns True if staged successfully."""
    src = getattr(event, "source", None)
    platform = getattr(src, "platform", "unknown") if src else "unknown"
    chat_id = str(getattr(src, "chat_id", "")) if src else ""
    message_id = str(getattr(event, "message_id", ""))
    sender_id = str(getattr(src, "user_id", "")) if src else ""
    sender_name = str(getattr(src, "user_name", "")) if src else ""

    # Serialize event for later reconstruction
    event_data = {}
    if hasattr(event, "text"):
        event_data["text"] = event.text
    if hasattr(event, "metadata"):
        event_data["metadata"] = event.metadata
    if src:
        for attr in ("platform", "chat_id", "chat_type", "user_id", "user_id_alt", "user_name"):
            if hasattr(src, attr):
                event_data[f"source_{attr}"] = str(getattr(src, attr))

    try:
        await stage_message(
            platform=platform, chat_id=chat_id, session_key=session_key,
            message_id=message_id, sender_id=sender_id, sender_name=sender_name,
            event_json=json.dumps(event_data), bot_name=getattr(adapter, "name", ""),
        )
        await set_staged_reaction(adapter, chat_id, message_id)
        return True
    except Exception as e:
        logger.error("[stage-queue] enqueue failed: %s", e, exc_info=True)
        return False
