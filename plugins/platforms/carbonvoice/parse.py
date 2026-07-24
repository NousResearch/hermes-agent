"""Pure parsing helpers for Carbon Voice payloads.

No I/O, no state, no async — everything here is a deterministic function
of the input dict. Keeps the rest of the plugin free of payload-shape knowledge.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .constants import AGENT_NAME_HEADER, AGENT_NAME_VALUE, USER_AGENT


def auth_headers(pat: str) -> Dict[str, str]:
    """Carbon Voice accepts PATs via Bearer auth and other keys via x-api-key."""
    trimmed = pat.strip()
    if trimmed.lower().startswith("cv_pat_"):
        return {"Authorization": f"Bearer {trimmed}"}
    return {"x-api-key": trimmed}


def client_headers(pat: str) -> Dict[str, str]:
    """Default headers for every Carbon Voice API client: auth plus the
    source tags the backend uses to attribute traffic to Hermes (the
    User-Agent is the one its request logger captures today)."""
    return {
        **auth_headers(pat),
        AGENT_NAME_HEADER: AGENT_NAME_VALUE,
        "user-agent": USER_AGENT,
    }


def first_str(*vals: Any) -> Optional[str]:
    """Return the first non-empty string in *vals*, or None."""
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_utc() -> "datetime":
    """Timezone-aware current UTC time (for :func:`message_age_seconds`)."""
    return datetime.now(timezone.utc)


def extract_transcript(msg: Dict[str, Any]) -> str:
    """Pull the human-readable transcript from a CV message payload.

    Shape compatibility — checked in order so the V5 source-of-truth
    payload wins, with the older shapes kept as fallback for the brief
    window between socket signal and the v5 GET enrichment (and for
    webhook callers that haven't migrated yet):

      - **V5 / GET ``/v5/messages/:id``**: top-level ``transcript`` string.
      - **V2 (socket push, ``/v3/messages/recent``)**: ``text_models[]``
        with one entry of ``type == "transcript"`` carrying either a
        joined ``timecodes[].t`` walk or a ``value`` string.
      - **Webhook**: ``transcript_txt`` or ``ai_summary_txt`` flat
        strings.

    When the message is still being transcribed all of these are empty;
    callers must treat an empty return as "not ready yet" and retry.
    """
    # V5 — preferred. Single source of truth per cv-api design.
    v5_transcript = msg.get("transcript")
    if isinstance(v5_transcript, str) and v5_transcript.strip():
        return v5_transcript.strip()
    # V2 — socket / v3-poll fallback.
    text_models = msg.get("text_models") or []
    if isinstance(text_models, list):
        for m in text_models:
            if not isinstance(m, dict):
                continue
            if m.get("type") in ("transcript_with_timecode", "transcript"):
                timecodes = m.get("timecodes") or []
                if isinstance(timecodes, list):
                    joined = " ".join(
                        tc.get("t", "")
                        for tc in timecodes
                        if isinstance(tc, dict) and isinstance(tc.get("t"), str)
                    ).strip()
                    if joined:
                        return joined
                value = m.get("value")
                if isinstance(value, str) and value.strip():
                    return value.strip()
    # Webhook-style payloads use different field names — accept those too.
    fallback = first_str(msg.get("transcript_txt"), msg.get("ai_summary_txt"))
    return fallback or ""


def extract_message_id(msg: Dict[str, Any]) -> Optional[str]:
    # V5 uses ``id``; V2 uses ``message_id``; legacy uses ``_id``.
    return first_str(msg.get("id"), msg.get("message_id"), msg.get("_id"))


def extract_channel_id(msg: Dict[str, Any]) -> Optional[str]:
    # V5 uses ``conversation_id`` (singular); V2 uses ``channel_ids[0]``;
    # webhook payloads use ``channel_id`` / ``channel_guid``.
    v5 = first_str(msg.get("conversation_id"))
    if v5:
        return v5
    channel_ids = msg.get("channel_ids")
    if isinstance(channel_ids, list) and channel_ids:
        first = channel_ids[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    return first_str(msg.get("channel_id"), msg.get("channel_guid"))


def extract_creator_id(msg: Dict[str, Any]) -> Optional[str]:
    # Same field across V2 and V5.
    return first_str(msg.get("creator_id"), msg.get("creator_guid"))


def extract_share_link_id(msg: Dict[str, Any]) -> Optional[str]:
    """Share-link id marking a *forwarded* message, or None.

    When a user forwards a message, cv-api creates a MessageForward record
    and stamps its id onto the new (wrapper) message as ``share_link_id``
    (and the deprecated alias ``forward_id`` — both are set to the same
    value by ``addForwardToMessage``). Present on V2 and V5 payloads. The
    original message's content is NOT on the wrapper; it must be fetched
    via ``GET /v3/message-sharelinks/{share_link_id}`` (the same flow
    cv-claude-channels uses).
    """
    return first_str(msg.get("share_link_id"), msg.get("forward_id"))


def message_age_seconds(msg: Dict[str, Any], now: "datetime") -> Optional[float]:
    """Seconds between a message's ``created_at`` and ``now``.

    Returns ``None`` when the payload carries no parseable timestamp (so
    callers can fall back to age-agnostic behavior). ``created_at`` is an
    ISO-8601 string across V2/V5 (``2026-06-05T19:36:44.437Z``); the
    trailing ``Z`` is normalized to ``+00:00`` for :meth:`fromisoformat`.
    """
    raw = first_str(msg.get("created_at"), msg.get("created"))
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds()


def extract_attachments(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return normalized inbound attachments as a list of dicts.

    Walks ``msg['attachments']`` and returns one dict per attachment
    with these keys (mirrors the field names CV uses on the wire):

        - ``_id``: server-assigned attachment id (used by
          :meth:`CarbonVoiceAPI.get_attachment_download_url` to resolve
          a pre-signed S3 GET URL)
        - ``link``: canonical S3 URL (auth-gated — don't try to GET it
          without going through the signedurl endpoint first)
        - ``filename``: as uploaded; often a UUID, not a friendly name
        - ``mime_type``: e.g. ``"image/png"``, ``"application/pdf"``
        - ``length_in_bytes``: int or None (CV sometimes leaves it null)
        - ``type``: typically ``"file"``; other AttachmentType values
          (``link``, ``location``, ...) are rare on inbound
        - ``status``: upload state (``Uploaded`` / ``Uploading`` /
          ``Initializing`` / ``Failed``) or ``""`` when absent — callers
          should treat a missing status as Uploaded (older payloads
          predate the status field)

    Entries missing both ``_id`` and ``link`` are dropped (defensive —
    CV's responses occasionally include legacy/null rows). Voice memos
    are NOT included here; their audio + transcript live in
    ``audio_models[]`` and ``text_models[]`` respectively, surfaced via
    :func:`extract_transcript` and inbound media handling on the audio
    side (separate path, future PR).
    """
    out: List[Dict[str, Any]] = []
    for att in (msg.get("attachments") or []):
        if not isinstance(att, dict):
            continue
        aid = first_str(att.get("_id"), att.get("id"))
        link = first_str(att.get("link"), att.get("url"))
        if not aid and not link:
            continue
        out.append({
            "_id": aid or "",
            "link": link or "",
            "filename": att.get("filename") or "",
            "mime_type": att.get("mime_type") or "",
            "length_in_bytes": att.get("length_in_bytes"),
            "type": att.get("type") or "file",
            "status": att.get("status") or "",
        })
    return out


def is_user_mentioned(msg: Dict[str, Any], user_id: Optional[str]) -> bool:
    """Return True when *user_id* is tagged in *msg*.

    Detection is **exclusively** via the structured ``tagged_user_ids``
    field. cv-api #243 exposes it on the message DTOs; the Flutter client
    populates it on the POST body for text messages and via the batch
    ``PUT /messages/:id/tagged-users`` for voice (cv-api #271 / #278).
    The transcript no longer carries the old ``@[name](guid)`` inline
    markup — the Flutter composer strips mentions to plain ``@Name``
    before send, so there is nothing to parse out of the text.

    Voice is the reason the field is authoritative: a voice memo tags
    Hermes *after* the audio is recorded, so the tag lands on a later
    ``message:updated`` rather than at create time. The gate's
    ``revisitable`` rejection (leaves the message out of the dedup cache)
    plus the ``get_message_v5`` enrichment guarantee that updated payload
    is re-evaluated with the now-populated array.
    """
    if not user_id:
        return False
    tagged = msg.get("tagged_user_ids")
    return isinstance(tagged, list) and user_id in tagged


def bot_has_reacted(
    msg: Dict[str, Any], bot_id: Optional[str], reaction_id: Optional[str]
) -> bool:
    """True when *bot_id* already reacted to *msg* with *reaction_id*.

    Reads ``msg['reaction_summary']['top_user_reactions']`` — the
    server-side record of who reacted with what. The adapter uses this as
    a **persistent "already processed" marker**: it puts an ack reaction
    on every accepted message, so a message that already carries the bot's
    ack was already handled and must not be re-dispatched.

    Unlike the in-memory ``SeenCache`` (lost on restart, 5-min TTL), the
    reaction lives in Carbon Voice, so this dedup survives gateway
    restarts and breaks the ``use_last_updated`` re-capture loop — the ack
    (and the bot's in-thread reply) bump ``updated_at``, which would
    otherwise make the poller re-fetch and re-process the same message
    indefinitely.

    Defensive against field-name variants and missing/oddly-shaped
    summaries; returns ``False`` on anything it can't positively match.
    """
    if not bot_id or not reaction_id:
        return False
    summary = msg.get("reaction_summary")
    if not isinstance(summary, dict):
        return False
    entries = summary.get("top_user_reactions")
    if not isinstance(entries, list):
        return False
    for e in entries:
        if not isinstance(e, dict):
            continue
        uid = first_str(e.get("user_id"), e.get("user_guid"), e.get("creator_id"))
        rid = first_str(e.get("reaction_id"), e.get("id"))
        if uid == bot_id and rid == reaction_id:
            return True
    return False


def reactors_for(
    msg: Dict[str, Any], reaction_ids: "set[str]"
) -> "set[str]":
    """Return the set of user_ids who reacted to *msg* with any id in
    *reaction_ids*.

    Reads the same ``reaction_summary.top_user_reactions`` shape as
    :func:`bot_has_reacted`, but generalized: instead of asking "did THIS
    user react with THIS id", it returns "who reacted with one of these
    ids". Used for one-tap owner approval — the adapter checks whether the
    owner is among the reactors with the approve/reject reaction on its
    pending prompt. Returns an empty set on anything it can't parse.
    """
    out: "set[str]" = set()
    if not reaction_ids:
        return out
    summary = msg.get("reaction_summary")
    if not isinstance(summary, dict):
        return out
    entries = summary.get("top_user_reactions")
    if not isinstance(entries, list):
        return out
    for e in entries:
        if not isinstance(e, dict):
            continue
        rid = first_str(e.get("reaction_id"), e.get("id"))
        if rid in reaction_ids:
            uid = first_str(
                e.get("user_id"), e.get("user_guid"), e.get("creator_id")
            )
            if uid:
                out.add(uid)
    return out


def chat_type_from_channel(channel: Optional[Dict[str, Any]]) -> str:
    """Map a Carbon Voice channel payload to Hermes ``chat_type``.

    Returns ``"dm"`` for one-to-one direct messages, ``"group"`` for every
    other channel kind (workspace channels, customer conversations, async
    meetings). Defaults to ``"dm"`` when the payload is missing so the
    adapter degrades to the prior single-tier behavior rather than dropping
    messages on a transient channel-lookup failure.

    Discriminator priority:
      1. ``type == "directMessage"`` — explicit type from PersonalizedChannel.
      2. ``dm_hash`` non-null — present only on DM channels (1:1 fingerprint
         used by the merge service); a reliable fallback if ``type`` is
         absent from older payloads.
    """
    if not channel:
        return "dm"
    ch_type = channel.get("type")
    if isinstance(ch_type, str) and ch_type.strip():
        return "dm" if ch_type == "directMessage" else "group"
    if channel.get("dm_hash"):
        return "dm"
    # Unknown/partial payload — preserve the prior "bot responds always"
    # behavior by defaulting to DM until we gain a positive signal.
    return "dm"


def _collaborator_name(p: Dict[str, Any]) -> str:
    """Best display name for one ``json_collaborators`` entry.

    Prefers ``first_name [last_name]``; falls back to the flat
    ``display_name`` / ``name`` / ``username`` shapes some payloads use.
    Returns ``""`` when nothing usable is present.
    """
    first = str(p.get("first_name") or "").strip()
    last = str(p.get("last_name") or "").strip()
    full = (first + " " + last).strip()
    if full:
        return full
    return first_str(p.get("display_name"), p.get("name"), p.get("username")) or ""


def extract_roster(channel: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Map ``user_guid`` → display name from a channel's collaborators.

    Carbon Voice's ``GET /channel/{id}`` returns ``json_collaborators`` —
    one entry per participant with ``user_guid`` + ``first_name`` /
    ``last_name``. This is the canonical place to resolve names: the
    standalone ``GET /v3/users/{id}`` endpoint is dead (404), and the
    collaborator list is already on the channel payload the adapter
    fetches for chat-type resolution, so names cost zero extra calls.

    Returns ``{}`` on a missing/partial payload. Entries without a guid or
    a usable name are skipped.
    """
    out: Dict[str, str] = {}
    if not channel:
        return out
    for p in (channel.get("json_collaborators") or []):
        if not isinstance(p, dict):
            continue
        guid = first_str(p.get("user_guid"), p.get("guid"), p.get("id"))
        if not guid:
            continue
        name = _collaborator_name(p)
        if name:
            out[guid] = name
    return out


def extract_reply_anchor(msg: Dict[str, Any]) -> Optional[str]:
    """The message_id to thread *next* replies under.

    Resolves to ``parent_message_id`` (the thread root) when the inbound
    message is a reply, else the message's own id. Mirrors the
    ``parent_message_id ?? message_id`` pattern in the TypeScript client.

    As of cv-api PR #277 (CV-13155) the backend resolves the thread root
    server-side (``resolveRootParentMessageId``): sending a non-root id as
    ``reply_to_message_id`` no longer returns ``400 You cannot reply to a
    message that is a reply`` — it is normalized to the root. The only
    remaining reply error is cross-conversation. Anchoring to the root
    here is therefore belt-and-suspenders, not a hard requirement.
    """
    parent = first_str(
        msg.get("parent_message_id"), msg.get("parent_message_guid")
    )
    return parent or extract_message_id(msg)
