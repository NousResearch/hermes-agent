"""Where ``PluginContext.post_card`` lands: a card in a conversation of the plugin's own profile.

A card is display-only data, never a prompt. It is admitted into the target session's durable FIFO
as a producer wake (``session_automation.admit_route_wake``, the path a local cron result takes),
and at the FIFO head it is always written as a note (``session_completion_notes``): no model call,
no tools, and the session's background review and forks keep running. The row's ``content`` is one
bounded core-authored line (``model_line``) that tells the next turn a card was posted, built from
the validated plugin id and card id only: no plugin-authored text reaches the model. The card
itself (text, directive name and attrs) rides ``display_metadata`` and the live ``plugin.card``
event, which only renderers read. Core sends nothing to a messaging platform: the plugin sends its
own native card there, and the row records that it did.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections import deque
from typing import Any, Mapping, Optional

from gateway.config import Platform

logger = logging.getLogger(__name__)

DISPLAY_KIND = "plugin_card"
PRODUCER = "plugin-card"
TEXT_MAX = 2000  # session_automation._DISPLAY_TEXT_LIMIT
ATTRS_MAX_KEYS = 16
ATTRS_MAX_CHARS = 2048  # Desktop's DIRECTIVE_BODY_MAX
RATE_LIMIT = (6, 60.0)  # cards per (plugin, session) per window seconds

_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
# A registry id (``manifest_key``: a bare name or ``<prefix>/<dir>``), identifier characters only: it
# is the one plugin-supplied value the model line carries.
_PLUGIN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@:-]{0,127}")
_NAME_RE = re.compile(r"[a-z][a-z0-9-]{0,63}")  # transcript-directives.ts SEGMENT_RE
_ATTR_RE = re.compile(r"[a-z][A-Za-z0-9_-]{0,63}")
_recent: dict[tuple[str, str], deque] = {}
_posting: dict[tuple[str, str], Any] = {}  # one asyncio.Lock per (plugin, session): posts to one session serialize


def _err(code: str, detail: str = "") -> dict:
    return {"ok": False, "error": code, **({"detail": detail} if detail else {})}


def _directive(raw: Any) -> Optional[dict]:
    if not isinstance(raw, Mapping) or set(raw) - {"name", "attrs"}:
        return None
    name, attrs = raw.get("name"), raw.get("attrs", {})
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name) or not isinstance(attrs, Mapping):
        return None
    if (len(attrs) > ATTRS_MAX_KEYS
            or not all(isinstance(k, str) and _ATTR_RE.fullmatch(k) and isinstance(v, str) for k, v in attrs.items())
            or len(json.dumps(dict(attrs), ensure_ascii=False)) > ATTRS_MAX_CHARS):
        return None
    return {"name": name, "attrs": dict(attrs)}


def card_descriptor(raw: Any) -> Optional[dict]:
    """The stored card (``plugin_id``, ``card_id``, ``directive``?) when *raw* is well formed, else
    None. Re-run on admission, as every producer field is."""
    if not isinstance(raw, Mapping) or set(raw) - {"plugin_id", "card_id", "directive"}:
        return None
    plugin_id, card_id = raw.get("plugin_id"), raw.get("card_id")
    if not isinstance(plugin_id, str) or not _PLUGIN_ID_RE.fullmatch(plugin_id):
        return None
    if not isinstance(card_id, str) or not _ID_RE.fullmatch(card_id):
        return None
    out = {"plugin_id": plugin_id, "card_id": card_id}
    if raw.get("directive") is not None:
        directive = _directive(raw["directive"])
        if directive is None:
            return None
        out["directive"] = directive
    return out


def validate(card: Any, plugin_id: str) -> tuple[Optional[dict], Optional[str]]:
    """``(normalized card, None)`` or ``(None, detail)`` for an ``invalid_card`` refusal."""
    if not isinstance(card, Mapping) or set(card) - {"id", "text", "directive"}:
        return None, "a card is a mapping of text, directive and id"
    text = card.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > TEXT_MAX:
        return None, f"text must be 1 to {TEXT_MAX} characters"
    card_id = card.get("id") or uuid.uuid4().hex[:16]
    stored = card_descriptor({"plugin_id": plugin_id, "card_id": card_id,
                              **({"directive": card["directive"]} if card.get("directive") is not None else {})})
    if stored is None:
        return None, "id or directive is malformed"
    return {**stored, "text": text}, None


def model_line(plugin_id: str, card_id: str) -> str:
    """What the model reads: that *plugin_id* posted card *card_id*. Both are validated identifiers
    (``card_descriptor``), so the line is bounded and carries no plugin-authored text, let alone the
    card's text, directive or attrs."""
    return f"[{plugin_id} posted a card {card_id}]"


def _rate_window(plugin_id: str, session_key: str) -> deque:
    """The (plugin, session) window's live stamps. The caller appends one when, and only when, its
    admission created a row: an exact replay spends nothing."""
    stamps = _recent.setdefault((plugin_id, session_key), deque())
    now = time.monotonic()
    while stamps and now - stamps[0] > RATE_LIMIT[1]:
        stamps.popleft()
    return stamps


def _posting_lock(plugin_id: str, session_key: str):
    lock = _posting.get((plugin_id, session_key))
    if lock is None:
        lock = _posting[(plugin_id, session_key)] = asyncio.Lock()
    return lock


def _resolve(runner, *, session_key: str, origin, plugin_home, plugin_id: str):
    """``(source, session_key, entry, None)`` for a conversation of the plugin's own profile, or
    ``(None, None, None, error)``. The same checks ``inject_message`` runs."""
    from gateway.session_identity import replace_source
    if origin is not None:
        source = runner._plugin_origin_source(origin, plugin_home=plugin_home, plugin_id=plugin_id)
        if source is None:
            return None, None, None, _err("no_conversation", "the origin is not a chat of this plugin's profile")
        session_key = runner._session_key_for_source(source)
    entry = runner.session_store.lookup_by_session_key(session_key) if session_key else None
    if entry is None or entry.origin is None:
        return None, None, None, _err("no_conversation", f"session {session_key!r} is not known to this gateway")
    source = replace_source(runner._restored_source(entry))
    if source.platform == Platform.LOCAL:
        live = runner._plugin_local_session(source, session_key, plugin_home=plugin_home, plugin_id=plugin_id)
        if live is None:
            return None, None, None, _err("no_conversation", "the local session is not live in this plugin's profile")
        return live.source, session_key, entry, None
    if not runner._plugin_messaging_session_owned(session_key, plugin_home=plugin_home, plugin_id=plugin_id):
        return None, None, None, _err("no_conversation", "the session belongs to another profile")
    if source.chat_type != "dm":
        return None, None, None, _err("group_not_allowed", "cards go to private chats only")
    return source, session_key, entry, None


async def post(runner, card: Any, *, plugin_id: str, plugin_home, session_key: str = "",
               origin: Optional[Mapping[str, Any]] = None) -> dict:
    """Admit *card* into the target conversation's FIFO. Runs on the gateway loop; never raises."""
    normalized, detail = validate(card, plugin_id)
    if normalized is None:
        return _err("invalid_card", detail)
    source, key, entry, refusal = await asyncio.to_thread(
        _resolve, runner, session_key=session_key, origin=origin, plugin_home=plugin_home, plugin_id=plugin_id)
    if refusal is not None:
        return refusal
    from gateway.session_authorities import authority_for_home
    from gateway.session_automation import AdmissionReplay, admit_route_wake
    from gateway.session_persistence import authority_persistence
    from persistence.core.errors import AuthorityRefusal
    authority = authority_for_home(runner, plugin_home) if plugin_home is not None else None
    if authority is None:
        return _err("no_conversation", "no session authority serves this plugin's profile")
    request_id = f"{plugin_id}:{normalized['card_id']}"
    stored = {k: v for k, v in normalized.items() if k != "text"}
    metadata = {"display_kind": DISPLAY_KIND, "display_text": normalized["text"], "plugin_card": stored}
    adapter = (runner._adapter_for_source(source) if source.platform == Platform.LOCAL
               else runner._delivery_adapter_for(source))
    replay = AdmissionReplay()
    async with _posting_lock(plugin_id, key):
        stamps = _rate_window(plugin_id, key)
        if len(stamps) >= RATE_LIMIT[0]:
            # A full window refuses a new card. An exact replay of a card already admitted is not new:
            # it goes on to the admission, which answers it as a duplicate without spending anything.
            identity = json.dumps([PRODUCER, key, request_id], separators=(",", ":"))
            seen = await asyncio.to_thread(authority_persistence(authority).admissions.find,
                                           principal_id="automation:" + key,
                                           session_id=authority.logical_owner(entry.session_id), request_id=identity)
            if seen is None:
                return _err("rate_limited", f"more than {RATE_LIMIT[0]} cards in {RATE_LIMIT[1]:.0f}s for this session")
        try:
            async with runner._plugin_injection_scope(source, plugin_home):
                admitted = await admit_route_wake(runner, adapter, source, model_line(plugin_id, normalized["card_id"]),
                                                  PRODUCER, metadata, request_id=request_id, replay=replay)
        except AuthorityRefusal as exc:
            if exc.reason == "admission_conflict":
                return _err("card_conflict", f"card {normalized['card_id']!r} was posted with different content")
            logger.warning("plugin_card plugin=%s session=%s refused: %s", plugin_id, key, exc.reason)
            return _err("refused", exc.reason)
        if not admitted:
            return _err("no_conversation", "the conversation has no committed turn to post into yet")
        if replay.created:
            stamps.append(time.monotonic())  # the one call that queued the row spends the slot
    # The receipt is the admission's own created-versus-existing outcome, never a read before it.
    status = "queued" if replay.created else "duplicate"
    logger.info("plugin_card plugin=%s session=%s card=%s %s", plugin_id, key, normalized["card_id"], status)
    return {"ok": True, "card_id": normalized["card_id"], "status": status}
