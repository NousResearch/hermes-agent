"""Disk-backed Teams conversation references for proactive send.

Bot Framework proactive send is
``POST {serviceUrl}v3/conversations/{id}/activities`` with the bot's own
app token. The live adapter only kept ConversationReferences in memory,
so a restart (or never-seen inbound) had no send path.

This module loads, classifies, and persists slim personal refs and POSTs
from the bot app identity. Callers inject ``poster`` so tests never
touch a live host. Live sends still go through the adapter's Bot
Framework allowlist.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional
from urllib.parse import quote, urlparse


Poster = Callable[[str, Mapping[str, str], Dict[str, Any]], Awaitable[tuple[int, Any]]]

_REQUIRED = ("kind", "bot_app_id", "service_url", "conversation_id", "tenant_id")
_CONV_ID_RE = re.compile(r"^[A-Za-z0-9:@\-_.]+$")
_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")
_GROUP_ADDRESSED_VIA = frozenset({"mention", "reply_to_own"})
_GROUP_HEARD_VIA = frozenset({"mention", "reply_to_own", "unmentioned"})


class StoredRefError(ValueError):
    """Stored reference is missing, the wrong kind, or policy-blocked."""


def group_inbound_addresses_bot(
    *,
    bot_app_id: str,
    entities: Any = None,
    reply_to_id: Optional[str] = None,
    own_activity_ids: Any = None,
) -> Optional[str]:
    """Return how a group inbound addressed this bot, or None.

    A group message addresses the bot only when it @mentions the bot app
    id (``28:{bot_app_id}``) or replies to one of this bot's own
    activity ids. Unmentioned lines are still heard; they do not count
    as an address.
    """
    bot = str(bot_app_id or "").strip()
    if not bot:
        return None
    bot_ids = {bot, f"28:{bot}"}
    for ent in entities or []:
        if isinstance(ent, Mapping):
            typ = ent.get("type")
            mentioned = ent.get("mentioned")
        else:
            typ = getattr(ent, "type", None)
            mentioned = getattr(ent, "mentioned", None)
        if str(typ or "").lower() != "mention":
            continue
        if isinstance(mentioned, Mapping):
            mid = mentioned.get("id")
        else:
            mid = getattr(mentioned, "id", None) if mentioned is not None else None
        if str(mid or "") in bot_ids:
            return "mention"
    reply = str(reply_to_id or "").strip()
    own = {str(x) for x in (own_activity_ids or []) if str(x).strip()}
    if reply and reply in own:
        return "reply_to_own"
    return None


def group_inbound_should_reply(
    addressed_via: Optional[str] = None,
    *,
    decide_speak: Optional[Callable[[], bool]] = None,
) -> bool:
    """Whether a delivered group inbound should produce a send.

    Mention or reply-to-own always replies. Unmentioned lines default
    silent; ``decide_speak`` may opt in (fail closed on exception).
    """
    via = str(addressed_via or "").strip()
    if via in _GROUP_ADDRESSED_VIA:
        return True
    if decide_speak is None:
        return False
    try:
        return bool(decide_speak())
    except Exception:
        return False


def classify_stored_ref(
    ref: Mapping[str, Any],
    *,
    expected_bot_app_id: Optional[str] = None,
) -> None:
    missing = [k for k in _REQUIRED if not str(ref.get(k) or "").strip()]
    if missing:
        raise StoredRefError(f"stored ref missing fields: {', '.join(missing)}")
    kind = str(ref.get("kind") or "").strip()
    if kind in ("group", "groupChat"):
        via = str(ref.get("addressed_via") or "").strip()
        if via not in _GROUP_HEARD_VIA:
            raise StoredRefError(
                "group inbound must record mention, reply-to-own, or unmentioned"
            )
        if not str(ref.get("addressed_by") or "").strip():
            raise StoredRefError("group ref requires inbound addresser")
    elif kind != "personal":
        raise StoredRefError(f"stored ref kind must be personal or groupChat, got {kind!r}")
    conv_id = str(ref["conversation_id"])
    if not _CONV_ID_RE.match(conv_id):
        raise StoredRefError("stored ref conversation_id is not a Bot Framework id")
    bot = str(ref["bot_app_id"]).strip()
    if expected_bot_app_id and bot != expected_bot_app_id:
        raise StoredRefError("stored ref bot_app_id does not match this adapter")
    policy = str(ref.get("outbound_policy") or "")
    if "reply_only" in policy:
        raise StoredRefError("stored ref outbound_policy is reply_only")
    parsed = urlparse(str(ref["service_url"]))
    if not parsed.scheme or not parsed.netloc:
        raise StoredRefError("stored ref service_url is not an absolute URL")


def activity_post_url(ref: Mapping[str, Any]) -> str:
    classify_stored_ref(ref)
    base = str(ref["service_url"]).rstrip("/") + "/"
    conv = quote(str(ref["conversation_id"]), safe=":@-_.")
    return f"{base}v3/conversations/{conv}/activities"


def _ref_recency(path: Path, data: Mapping[str, Any]) -> int:
    stamp = data.get("persisted_at")
    if isinstance(stamp, bool):
        stamp = None
    if isinstance(stamp, (int, float)):
        return int(stamp)
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def load_stored_refs(directory: Path) -> Dict[str, Dict[str, Any]]:
    loaded: Dict[str, Dict[str, Any]] = {}
    recency: Dict[str, int] = {}
    if not directory.is_dir():
        return loaded
    for path in directory.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        try:
            classify_stored_ref(data)
        except StoredRefError:
            continue
        conv = str(data["conversation_id"])
        rank = _ref_recency(path, data)
        if conv in recency and recency[conv] >= rank:
            continue
        recency[conv] = rank
        loaded[conv] = data
    return loaded


def _ref_filename_stem(
    *,
    conversation_id: str,
    kind: str,
    filename_stem: Optional[str],
    person: Optional[str],
    aad_object_id: Optional[str],
    user_id: Optional[str],
) -> str:
    """Conversation-qualified stem. Never key on person alone."""
    label = filename_stem or person or aad_object_id or user_id or "ref"
    raw = f"{label}-{kind}-{conversation_id}"
    stem = _STEM_RE.sub("-", raw).strip("-._") or "ref"
    return stem[:180]


def persist_inbound_ref(
    directory: Path,
    *,
    conversation_id: str,
    conversation_type: str,
    service_url: str,
    tenant_id: str,
    bot_app_id: str,
    aad_object_id: Optional[str] = None,
    user_id: Optional[str] = None,
    person: Optional[str] = None,
    filename_stem: Optional[str] = None,
    inbound_activity_id: Optional[str] = None,
    addressed_via: Optional[str] = None,
) -> Path:
    if conversation_type == "personal":
        kind = "personal"
    elif conversation_type in ("group", "groupChat"):
        kind = "groupChat"
    else:
        raise StoredRefError(f"cannot persist kind {conversation_type!r}")
    ref: Dict[str, Any] = {
        "kind": kind,
        "conversation_id": conversation_id,
        "service_url": service_url if service_url.endswith("/") else service_url + "/",
        "tenant_id": tenant_id,
        "bot_app_id": bot_app_id,
    }
    if person:
        ref["person"] = person
    if aad_object_id:
        ref["aad_object_id"] = aad_object_id
    if user_id:
        ref["user_id"] = user_id
    if kind == "groupChat":
        via = str(addressed_via or "").strip() or "unmentioned"
        if via not in _GROUP_HEARD_VIA:
            raise StoredRefError(
                "group inbound must record mention, reply-to-own, or unmentioned"
            )
        addresser = user_id or aad_object_id
        if not addresser:
            raise StoredRefError("group ref requires inbound addresser")
        ref["addressed_via"] = via
        ref["addressed_by"] = str(addresser)
        if inbound_activity_id:
            ref["last_inbound_activity_id"] = str(inbound_activity_id)
    ref["persisted_at"] = time.time_ns()
    classify_stored_ref(ref, expected_bot_app_id=bot_app_id)
    directory.mkdir(parents=True, exist_ok=True)
    for existing in directory.glob("*.json"):
        try:
            prev = json.loads(existing.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(prev, dict):
            continue
        if prev.get("conversation_id") != conversation_id:
            continue
        if "reply_only" in str(prev.get("outbound_policy") or ""):
            raise StoredRefError("will not unlock reply_only ref")
    stem = _ref_filename_stem(
        conversation_id=conversation_id,
        kind=kind,
        filename_stem=filename_stem,
        person=person,
        aad_object_id=aad_object_id,
        user_id=user_id,
    )
    dest = directory / f"{stem}.json"
    dest.write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
    dest.chmod(0o600)
    return dest


async def send_from_stored_ref(
    ref: Mapping[str, Any],
    text: str,
    *,
    poster: Poster,
    expected_bot_app_id: Optional[str] = None,
    token: str,
    reply_to: Optional[str] = None,
    decide_speak: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    if not text or not str(text).strip():
        return {"error": "stored-ref send: empty text"}
    if not token:
        return {"error": "stored-ref send: missing token"}
    try:
        classify_stored_ref(ref, expected_bot_app_id=expected_bot_app_id)
    except StoredRefError as exc:
        return {"error": f"stored-ref send: {exc}"}
    url = activity_post_url(ref)
    bot = str(ref["bot_app_id"]).strip()
    body: Dict[str, Any] = {
        "type": "message",
        "text": text,
        "textFormat": "markdown",
        "from": {"id": f"28:{bot}"},
    }
    kind = str(ref.get("kind") or "").strip()
    thread_id = str(reply_to or ref.get("last_inbound_activity_id") or "").strip()
    if kind in ("group", "groupChat"):
        if not thread_id:
            return {"error": "stored-ref send: group send is never a first post"}
        via = str(ref.get("addressed_via") or "").strip()
        if not group_inbound_should_reply(via, decide_speak=decide_speak):
            return {
                "silent": True,
                "error": "stored-ref send: unmentioned group inbound is silent by default",
            }
        body["replyToId"] = thread_id
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        status, payload = await poster(url, headers, body)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return {"error": f"stored-ref send failed: {type(exc).__name__}"}
    if status not in (200, 201, 202):
        return {"error": f"stored-ref send: activity post failed ({status})"}
    if not isinstance(payload, dict) or not payload.get("id"):
        return {"error": "stored-ref send: connector accepted without activity id"}
    return {"success": True, "message_id": payload["id"], "http_status": status}
