"""thread_ownership — local Slack thread coordination (no lock service, no LLM).

Problem: when several Hermes agents share a Slack channel, Hermes' "once
@-mentioned in a thread, auto-follow every later message"
(plugins/platforms/slack/adapter.py ``_mentioned_threads``) makes every bot open
a turn on every thread message. An LLM agent that opens a turn MUST emit
something — so a bot that isn't the intended recipient blurts out "this isn't
for me, staying out", which is noise.

Solution: a `pre_gateway_dispatch` hook (runs BEFORE the turn) implementing
"the most-recently @-mentioned bot owns the thread". The owner answers non-@
follow-ups; everyone else stays silent (we return a skip, so the turn never runs
and nothing is emitted).

Why no lock service / no cluster roster is needed: thread ownership is
"is the LAST <@id> mention in this message me?", which every bot computes
independently from the same message text — so they all converge on the same
owner with zero coordination. Every bot following the thread receives every
message (Hermes auto-follow), so they stay in sync. Only the bot's OWN Slack
user_id is needed (auth.test); we never need to know who the other bots are.

Decision per Slack thread message (this bot = me, my_id = my Slack user_id):
  1. Non-Slack / no channel        → allow.
  2. DM (channel id starts "D")    → allow (1:1; every message is for me).
  3. Top-level msg (no thread_ts)  → allow (Slack's root @-gating handles it).
  4. my_id unresolved              → allow (never mute a misconfigured bot).
  5. Message has <@id> mention(s):
       owner := (last <@id> == me);  I reply ⟺ I'm in the mention list.
       I'm mentioned → allow; not mentioned → skip (yield to whoever was @-ed).
  6. No mention (plain follow-up):
       I'm the current owner → allow; otherwise → skip.

Ownership state is in-process memory ({channel:thread -> am I owner}); each bot
keeps its own. A restart drops it (the bot goes quiet until @-mentioned again) —
same as Hermes' own auto-follow state, which is also in-memory. Acceptable.

Env: RAILWAY_SERVICE_NAME / HOSTNAME (identity, platform-provided) and
SLACK_BOT_TOKEN (auth.test → own user_id; the Slack platform already requires
it). No knobs.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Per-process per-thread ownership: f"{channel_id}:{thread_ts}" -> am I the owner.
_owns: dict[str, bool] = {}
_owns_lock = threading.Lock()
# Cap memory growth on a long-running process with many threads; drop the
# oldest half when exceeded.
_OWNS_MAX = 5000

# Slack canonical mention: <@U0AB12CD> or <@U0AB12CD|name>. Capture user ids in
# order of appearance. (Special mentions like <!here>/<!channel> use "<!" and
# don't match; lowercase tokens don't match the uppercase id charset.)
_MENTION_RE = re.compile(r"<@([A-Z0-9]+)(?:\|[^>]*)?>")

# --- identity resolution (platform-provided env only) -----------------------
_bot_user_id_cache: Optional[str] = None
_bot_user_id_cache_lock = threading.Lock()
_BOT_USER_ID_SENTINEL = "<unresolved>"


def _resolve_agent_id() -> str:
    for var in ("RAILWAY_SERVICE_NAME", "HOSTNAME"):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    return ""


def _resolve_bot_user_id() -> Optional[str]:
    """Resolve this bot's own Slack user_id via auth.test, cached forever
    (failures cached too, so we don't hammer Slack)."""
    global _bot_user_id_cache
    if _bot_user_id_cache is not None:
        return None if _bot_user_id_cache == _BOT_USER_ID_SENTINEL else _bot_user_id_cache
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token:
        with _bot_user_id_cache_lock:
            _bot_user_id_cache = _BOT_USER_ID_SENTINEL
        return None
    with _bot_user_id_cache_lock:
        if _bot_user_id_cache is not None:
            return None if _bot_user_id_cache == _BOT_USER_ID_SENTINEL else _bot_user_id_cache
        try:
            resp = httpx.post(
                "https://slack.com/api/auth.test",
                headers={"Authorization": f"Bearer {token}"},
                timeout=3.0,
            )
            data = resp.json()
            if data.get("ok"):
                uid = (data.get("user_id") or "").strip()
                _bot_user_id_cache = uid or _BOT_USER_ID_SENTINEL
                if uid:
                    logger.info("thread_ownership: detected bot_user_id=%s", uid)
                return uid or None
            logger.warning("thread_ownership: auth.test not-ok: %s", data)
        except Exception as e:
            logger.warning("thread_ownership: auth.test failed (%s)", e)
        _bot_user_id_cache = _BOT_USER_ID_SENTINEL
        return None


def _slack_fields(event: Any) -> Optional[tuple[str, Optional[str], str]]:
    """Return (channel_id, thread_ts, text) or None if not a Slack message."""
    source = getattr(event, "source", None)
    if source is None:
        return None
    if "slack" not in str(getattr(source, "platform", "")).lower():
        return None
    channel_id = getattr(source, "chat_id", "") or ""
    if not channel_id:
        return None
    thread_ts = (
        getattr(source, "thread_id", None)
        or getattr(event, "reply_to_message_id", None)
        or None
    )
    # CRITICAL: Hermes strips THIS bot's own <@id> from event.text before the hook
    # runs (plugins/platforms/slack/adapter.py does ``text.replace(f"<@{bot_uid}>",
    # "")`` on a direct @-mention). That would hide our OWN mention from the parser,
    # so a direct "@me ..." looks like a no-mention follow-up and we'd wrongly skip
    # it. Parse the ORIGINAL Slack text from raw_message["text"] (the untouched event
    # dict) so we see our own mention AND the full ordered mention list; fall back to
    # event.text only if the raw text is unavailable.
    text = ""
    raw = getattr(event, "raw_message", None)
    if isinstance(raw, dict):
        text = raw.get("text") or ""
    if not text:
        text = getattr(event, "text", "") or ""
    return channel_id, thread_ts, text


# --- ownership state --------------------------------------------------------
def _set_owns(key: str, owns: bool) -> None:
    with _owns_lock:
        _owns[key] = owns
        if len(_owns) > _OWNS_MAX:
            for k in list(_owns)[: _OWNS_MAX // 2]:
                _owns.pop(k, None)


def _get_owns(key: str) -> bool:
    with _owns_lock:
        return _owns.get(key, False)


# --- the hook ---------------------------------------------------------------
def _on_pre_gateway_dispatch(**kwargs: Any) -> Optional[dict[str, Any]]:
    event = kwargs.get("event")
    if event is None:
        return None
    fields = _slack_fields(event)
    if fields is None:
        return None
    channel_id, thread_ts, text = fields

    # DM: 1:1 — every message is for me.
    if channel_id.startswith("D"):
        return None
    # Top-level message: Slack's root @-gating handles it; we only gate threads.
    if not thread_ts:
        return None
    # No identity → safe no-op (never mute a misconfigured bot).
    if not _resolve_agent_id():
        return None
    my_id = _resolve_bot_user_id()
    if not my_id:
        return None  # can't tell if I'm addressed → fail open

    key = f"{channel_id}:{thread_ts}"
    mentions = _MENTION_RE.findall(text)

    if mentions:
        # An @-message: the LAST mention owns the thread; I reply iff I'm in the
        # mention list (any position). Each bot computes the same owner.
        _set_owns(key, mentions[-1] == my_id)
        if my_id in mentions:
            return None
        reason = f"thread {key}: addressed to another participant — staying silent"
        logger.info("thread_ownership: %s", reason)
        return {"action": "skip", "reason": reason}

    # Plain follow-up (no mention): only the current owner replies.
    if _get_owns(key):
        return None
    reason = f"thread {key}: not the current owner — staying silent"
    logger.info("thread_ownership: %s", reason)
    return {"action": "skip", "reason": reason}


def register(ctx) -> None:
    """Hermes plugin entry point. Wires the hook into the gateway."""
    ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch)
    agent_id = _resolve_agent_id() or "<unset>"
    logger.info(
        "thread_ownership: registered as agent_id=%r (local last-mention ownership)",
        agent_id,
    )
