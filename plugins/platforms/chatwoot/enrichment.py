"""CRWD contact enrichment: MongoDB → Chatwoot contact + Honcho peer.

When a CRWD user messages the bot through Chatwoot, this module looks them up
in the CRWD MongoDB (by email, falling back to phone) and hydrates the two
downstream systems that would otherwise start cold:

  - the **Chatwoot contact** gets profile fields (name, phone, location,
    socials, avatar) plus interest **labels**;
  - the user's **Honcho peer** gets a peer card + demographic conclusions.

Design constraints:
  - **Live, per-conversation** — driven from the inbound message path, not a
    batch job.
  - **Idempotent** — a ``crwd_synced_at`` custom attribute on the contact (plus
    a small in-process cache) prevents re-syncing on every message.
  - **Non-fatal** — every failure is swallowed and logged; enrichment must
    never break message handling.
  - **Secret-safe** — Mongo reads use an explicit projection that excludes
    password/token/otp fields.

Mongo access reuses ``tools.crwd_db_tool._db`` (``CRWD_MONGO_URI`` /
``CRWD_MONGO_DB``). Honcho writes reuse ``HonchoSessionManager`` so the peer-id
derivation and observation-mode semantics match exactly what the live agent
reads back.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Config -----------------------------------------------------------------

_DEFAULT_TTL_DAYS = 30
_HONCHO_ANSWER_CAP = 25         # max useranswers → Honcho conclusions
_LRU_MAX = 4096                 # in-process "recently synced" cache size
_MONGO_MAX_TIME_MS = 5000

# Explicit projection — never pull whole user docs; this list omits every
# password/token/otp/secret field by construction.
_ENRICH_USER_FIELDS = {
    "_id": 1, "first_name": 1, "last_name": 1, "full_name": 1,
    "email": 1, "phone": 1, "bio": 1, "gender": 1, "dob": 1, "status": 1,
    "city": 1, "state": 1, "country": 1, "postal_code": 1, "address": 1,
    "profile_pic": 1, "avatar": 1,
    "insta_url": 1, "twitter_url": 1, "tiktok_url": 1,
}

# in-proc negative+positive cache: contact_id -> monotonic ts of last sync.
_recent_syncs: "OrderedDict[str, float]" = OrderedDict()


def _enabled() -> bool:
    if not os.getenv("CRWD_MONGO_URI"):
        return False
    flag = os.getenv("CRWD_ENRICH_ENABLED", "true").strip().lower()
    return flag not in ("0", "false", "no", "off")


def _ttl_seconds() -> float:
    try:
        days = float(os.getenv("CRWD_ENRICH_TTL_DAYS", str(_DEFAULT_TTL_DAYS)))
    except (TypeError, ValueError):
        days = _DEFAULT_TTL_DAYS
    return max(0.0, days) * 86400.0


def _asset_base_url() -> str:
    return os.getenv("CRWD_ASSET_BASE_URL", "").strip().rstrip("/")


# --- In-process idempotency cache -------------------------------------------

def _recently_synced(contact_id: str) -> bool:
    ts = _recent_syncs.get(contact_id)
    if ts is None:
        return False
    if (time.monotonic() - ts) > _ttl_seconds():
        _recent_syncs.pop(contact_id, None)
        return False
    _recent_syncs.move_to_end(contact_id)
    return True


def _mark_synced(contact_id: str) -> None:
    _recent_syncs[contact_id] = time.monotonic()
    _recent_syncs.move_to_end(contact_id)
    while len(_recent_syncs) > _LRU_MAX:
        _recent_syncs.popitem(last=False)


def _synced_at_is_fresh(contact: Dict[str, Any]) -> bool:
    """True if the contact's persisted crwd_synced_at is within the TTL."""
    attrs = contact.get("custom_attributes") or {}
    raw = attrs.get("crwd_synced_at")
    if not raw:
        return False
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - dt).total_seconds()
    return age < _ttl_seconds()


# --- Mongo readers (reuse crwd_db connection) -------------------------------

def _db():
    from tools.crwd_db_tool import _db as _crwd_db  # lazy: avoid pymongo import at load
    return _crwd_db()


def _oid_candidates(user_id: Any) -> list:
    """Match a users._id stored as either ObjectId or string."""
    from bson import ObjectId
    sid = str(user_id)
    out: list = [sid]
    if re.fullmatch(r"[a-fA-F0-9]{24}", sid):
        try:
            out.append(ObjectId(sid))
        except Exception:
            pass
    return out


def fetch_user(email: Optional[str], phone: Optional[str]) -> Optional[Dict[str, Any]]:
    """Look up a CRWD user by email, falling back to phone. Read-only."""
    coll = _db()["users"]
    for query in _match_queries(email, phone):
        doc = coll.find_one(query, _ENRICH_USER_FIELDS, max_time_ms=_MONGO_MAX_TIME_MS)
        if doc:
            return doc
    return None


def _match_queries(email: Optional[str], phone: Optional[str]) -> List[Dict[str, Any]]:
    """Ordered match queries: exact email (case-insensitive), then phone."""
    queries: List[Dict[str, Any]] = []
    email = (email or "").strip()
    phone = (phone or "").strip()
    if email:
        queries.append({"email": re.compile(f"^{re.escape(email)}$", re.IGNORECASE)})
    if phone:
        queries.append({"phone": phone})
    return queries


def fetch_interests(user_id: Any) -> List[str]:
    """Return cleaned interest titles (emoji stripped) for a user."""
    cur = _db()["interests"].find(
        {"user_id": {"$in": _oid_candidates(user_id)}, "isDeleted": {"$ne": True}},
        {"title": 1},
        max_time_ms=_MONGO_MAX_TIME_MS,
    )
    titles = []
    for d in cur:
        cleaned = _clean_interest_title(d.get("title", ""))
        if cleaned:
            titles.append(cleaned)
    return titles


def fetch_answers(user_id: Any, cap: int = _HONCHO_ANSWER_CAP) -> List[Tuple[str, str]]:
    """Return (label, answer) survey pairs for a user, capped.

    ``label`` is the question text (``useranswers.question_id`` →
    ``questionnaires.question``) when resolvable, else the capitalized
    category — so a bare answer like "No" becomes a meaningful fact.
    """
    db = _db()
    rows = list(
        db["useranswers"]
        .find(
            {"user_id": {"$in": _oid_candidates(user_id)}, "isDeleted": {"$ne": True}},
            {"answer": 1, "category": 1, "question_id": 1},
            max_time_ms=_MONGO_MAX_TIME_MS,
        )
        .limit(max(1, cap))
    )
    qids = [r["question_id"] for r in rows if r.get("question_id") is not None]
    qmap: Dict[Any, str] = {}
    if qids:
        for q in db["questionnaires"].find(
            {"_id": {"$in": qids}}, {"question": 1}, max_time_ms=_MONGO_MAX_TIME_MS
        ):
            text = str(q.get("question") or "").strip()
            if text:
                qmap[q["_id"]] = text
    pairs: List[Tuple[str, str]] = []
    for r in rows:
        ans = str(r.get("answer") or "").strip()
        if not ans:
            continue
        label = qmap.get(r.get("question_id")) or str(
            r.get("category") or "survey"
        ).strip().capitalize()
        pairs.append((label, ans))
    return pairs


# --- Pure mapping helpers (unit-tested) -------------------------------------

_EMOJI_PREFIX = re.compile(r"^[^\w]+", re.UNICODE)


def _clean_interest_title(title: str) -> str:
    """'🎵 Music' -> 'Music'."""
    return _EMOJI_PREFIX.sub("", str(title or "")).strip()


def _slug_label(title: str) -> str:
    """'🎵 Music' -> 'music' (Chatwoot label-safe: lowercase, hyphenated)."""
    cleaned = _clean_interest_title(title).lower()
    slug = re.sub(r"[^a-z0-9]+", "-", cleaned).strip("-")
    return slug


def _clean_handle(handle: str) -> str:
    """Normalize a social handle: strip @, whitespace, surrounding slashes."""
    h = str(handle or "").strip()
    if not h:
        return ""
    h = h.lstrip("@").strip().strip("/")
    # If someone stored a full URL, keep only the last path segment.
    if "/" in h:
        h = h.rsplit("/", 1)[-1]
    return h


def _social_url(handle: str, kind: str) -> Optional[str]:
    h = _clean_handle(handle)
    if not h:
        return None
    if kind == "instagram":
        return f"https://instagram.com/{h}"
    if kind == "twitter":
        return f"https://x.com/{h}"
    if kind == "tiktok":
        return f"https://tiktok.com/@{h}"
    return None


def _avatar_url(profile_pic: str, base_url: str) -> Optional[str]:
    pic = str(profile_pic or "").strip()
    if not pic:
        return None
    if pic.startswith(("http://", "https://")):
        return pic
    if not base_url:
        return None
    return f"{base_url}/{pic.lstrip('/')}"


def _nonempty(value: Any) -> Optional[str]:
    s = str(value or "").strip()
    return s or None


def build_contact_fields(
    user: Dict[str, Any], asset_base_url: str, *, synced_at: Optional[str] = None
) -> Dict[str, Any]:
    """Map a CRWD user doc to a Chatwoot contact update payload."""
    additional: Dict[str, Any] = {}
    for src, dst in (("city", "city"), ("country", "country"), ("state", "state")):
        val = _nonempty(user.get(src))
        if val:
            additional[dst] = val

    custom: Dict[str, Any] = {"joincrwd_user_id": str(user.get("_id"))}
    custom["crwd_synced_at"] = synced_at or datetime.now(timezone.utc).isoformat()
    for key in ("bio", "gender", "dob", "status", "postal_code"):
        val = _nonempty(user.get(key))
        if val:
            custom[f"crwd_{key}"] = val
    insta = _social_url(user.get("insta_url", ""), "instagram")
    twitter = _social_url(user.get("twitter_url", ""), "twitter")
    tiktok = _social_url(user.get("tiktok_url", ""), "tiktok")
    if insta:
        custom["crwd_instagram"] = insta
    if twitter:
        custom["crwd_twitter"] = twitter
    if tiktok:
        custom["crwd_tiktok"] = tiktok

    fields: Dict[str, Any] = {
        "additional_attributes": additional,
        "custom_attributes": custom,
    }
    name = _nonempty(user.get("full_name")) or _nonempty(
        " ".join(x for x in (user.get("first_name"), user.get("last_name")) if x)
    )
    if name:
        fields["name"] = name
    email = _nonempty(user.get("email"))
    if email:
        fields["email"] = email
    phone = _nonempty(user.get("phone"))
    if phone:
        fields["phone_number"] = phone
    avatar = _avatar_url(user.get("profile_pic", ""), asset_base_url)
    if avatar:
        fields["avatar_url"] = avatar
    return fields


def build_interest_labels(interests: List[str]) -> List[str]:
    """Interest titles → deduped, Chatwoot-safe label slugs."""
    seen: "OrderedDict[str, None]" = OrderedDict()
    for title in interests:
        slug = _slug_label(title)
        if slug:
            seen.setdefault(slug, None)
    return list(seen.keys())


def build_peer_card(
    user: Dict[str, Any], interests: List[str], answers: List[Tuple[str, str]]
) -> List[str]:
    """Build a concise Honcho peer card (list of fact strings)."""
    card: List[str] = []
    name = _nonempty(user.get("full_name"))
    if name:
        card.append(f"Name: {name}")
    loc = ", ".join(
        x for x in (
            _nonempty(user.get("city")),
            _nonempty(user.get("state")),
            _nonempty(user.get("country")),
        ) if x
    )
    if loc:
        card.append(f"Location: {loc}")
    if interests:
        card.append("Interests: " + ", ".join(interests))
    # A few survey highlights inline on the card.
    for label, ans in answers[:5]:
        card.append(f"{label}: {ans}")
    return card


def build_conclusions(answers: List[Tuple[str, str]]) -> List[str]:
    """Survey (label, answer) pairs → Honcho conclusion strings."""
    out: List[str] = []
    for label, ans in answers:
        out.append(f"{label} — {ans}")
    return out


# --- Honcho writer (reuse HonchoSessionManager) -----------------------------

def _write_honcho(
    contact_id: str,
    session_key: str,
    card: List[str],
    conclusions: List[str],
) -> None:
    """Set the peer card + write conclusions for this contact's Honcho peer.

    Runs synchronously (Honcho SDK is blocking); the caller offloads it to a
    thread. Reuses ``HonchoSessionManager`` so peer-id derivation and the
    observer/target observation-mode logic match the live agent path exactly.
    """
    try:
        from plugins.memory.honcho.client import HonchoClientConfig, get_honcho_client
        from plugins.memory.honcho.session import HonchoSessionManager
    except Exception as exc:  # honcho not installed / not active
        logger.debug("[crwd-enrich] honcho unavailable, skipping peer write: %s", exc)
        return

    try:
        cfg = HonchoClientConfig.from_global_config()
    except Exception:
        cfg = None
    if cfg is None or not (getattr(cfg, "api_key", None) or getattr(cfg, "base_url", None)):
        logger.debug("[crwd-enrich] honcho not configured, skipping peer write")
        return

    try:
        client = get_honcho_client(cfg)
        mgr = HonchoSessionManager(
            honcho=client, config=cfg, runtime_user_peer_name=str(contact_id)
        )
        mgr.get_or_create(session_key)
        if card:
            mgr.set_peer_card(session_key, card, peer="user")
        for c in conclusions:
            mgr.create_conclusion(session_key, c, peer="user")
        logger.info(
            "[crwd-enrich] honcho peer updated for contact=%s (%d facts, %d conclusions)",
            contact_id, len(card), len(conclusions),
        )
    except Exception as exc:
        logger.warning("[crwd-enrich] honcho peer write failed for %s: %s", contact_id, exc)


# --- Orchestrator -----------------------------------------------------------

def _parse_event(event: Any) -> Optional[Dict[str, Optional[str]]]:
    """Pull the ids + contact hints we need out of the webhook payload."""
    payload = getattr(event, "raw_message", None)
    if not isinstance(payload, dict):
        return None
    sender = payload.get("sender") if isinstance(payload.get("sender"), dict) else {}
    account = payload.get("account") if isinstance(payload.get("account"), dict) else {}
    conv = payload.get("conversation") if isinstance(payload.get("conversation"), dict) else {}
    contact_id = _nonempty(sender.get("id"))
    account_id = _nonempty(account.get("id"))
    if not account_id:
        # Fall back to the chat_id (account:conversation) on the event source.
        chat_id = getattr(getattr(event, "source", None), "chat_id", "") or ""
        if ":" in str(chat_id):
            account_id = str(chat_id).split(":", 1)[0]
    conversation_id = _nonempty(conv.get("id")) or _nonempty(conv.get("display_id"))
    if not (contact_id and account_id):
        return None
    return {
        "contact_id": contact_id,
        "account_id": account_id,
        "conversation_id": conversation_id,
        "email": _nonempty(sender.get("email")),
        "phone": _nonempty(sender.get("phone_number")),
    }


async def enrich(adapter: Any, event: Any) -> None:
    """Best-effort: hydrate the Chatwoot contact + Honcho peer from Mongo.

    Safe to call on every inbound message — it self-gates via an in-process
    cache and the contact's persisted ``crwd_synced_at``.
    """
    if not _enabled():
        return
    try:
        ctx = _parse_event(event)
        if ctx is None:
            return
        contact_id = ctx["contact_id"]
        account_id = ctx["account_id"]

        if _recently_synced(contact_id):
            return

        # One GET serves both the freshness gate and authoritative email/phone.
        contact = await adapter.get_contact(account_id, contact_id)
        if contact and _synced_at_is_fresh(contact):
            _mark_synced(contact_id)
            return

        email = ctx["email"] or (_nonempty(contact.get("email")) if contact else None)
        phone = ctx["phone"] or (
            _nonempty(contact.get("phone_number")) if contact else None
        )
        if not (email or phone):
            _mark_synced(contact_id)  # nothing to match on; don't retry each message
            return

        import asyncio

        user = await asyncio.to_thread(fetch_user, email, phone)
        if not user:
            logger.info("[crwd-enrich] no Mongo user for contact=%s", contact_id)
            _mark_synced(contact_id)  # negative cache; avoid re-querying every message
            return

        user_id = user.get("_id")
        interests = await asyncio.to_thread(fetch_interests, user_id)
        answers = await asyncio.to_thread(fetch_answers, user_id)

        # --- Chatwoot: profile + labels ---
        fields = build_contact_fields(user, _asset_base_url())
        # Validate the avatar before pinning it — staging serves an HTML page
        # (not a 404) for missing files, which would become a broken avatar.
        avatar = fields.pop("avatar_url", None)
        if avatar and await adapter.url_is_image(avatar):
            fields["avatar_url"] = avatar
        elif avatar:
            logger.info("[crwd-enrich] skipping avatar (not an image): %s", avatar)
        await adapter.update_contact(account_id, contact_id, fields)
        labels = build_interest_labels(interests)
        if labels:
            await adapter.add_contact_labels(account_id, contact_id, labels)

        # --- Honcho: peer card + conclusions ---
        card = build_peer_card(user, interests, answers)
        conclusions = build_conclusions(answers)
        session_key = f"chatwoot:{account_id}:{ctx['conversation_id'] or contact_id}"
        await asyncio.to_thread(_write_honcho, contact_id, session_key, card, conclusions)

        _mark_synced(contact_id)
        logger.info(
            "[crwd-enrich] enriched contact=%s user=%s (%d interests, %d answers)",
            contact_id, user_id, len(interests), len(answers),
        )
    except Exception:
        logger.warning("[crwd-enrich] enrichment failed", exc_info=True)
