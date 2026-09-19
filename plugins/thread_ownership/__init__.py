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
message (Hermes auto-follow), so they stay in sync. We only need to know THIS
bot's own Slack user_id; we never need to know who the other bots are.

Identity is workspace-scoped. The Slack adapter authenticates each configured
bot token — comma-separated SLACK_BOT_TOKEN and OAuth-added workspaces — and
keeps team_id → bot_user_id in ``_team_bot_user_ids``
(plugins/platforms/slack/adapter.py). We read that live map keyed by the event's
own workspace, so a multi-workspace deployment resolves the right identity per
message instead of assuming a single token. An unknown/unresolved workspace
fails open (see decision 4) — never mute a bot we can't identify. The adapter is
reached via the ``gateway`` kwarg the hook already receives.

Decision per Slack thread message (this bot = me, my_id = my Slack user_id):
  1. Non-Slack / no channel        → allow.
  2. DM (channel id starts "D")    → allow (1:1; every message is for me).
  3. Top-level msg (no thread_ts)  → allow (Slack's root @-gating handles it).
  4. my_id unresolved              → allow (never mute a bot we can't identify).
  5. Message has <@id> mention(s):
       owner := (last <@id> == me);  I reply ⟺ I'm in the mention list.
       I'm mentioned → allow; not mentioned → skip (yield to whoever was @-ed).
  6. No mention (plain follow-up):
       I'm the current owner → allow; otherwise → skip.

Ownership state is in-process memory ({channel:thread -> am I owner}); each bot
keeps its own. A restart drops it (the bot goes quiet until @-mentioned again) —
same as Hermes' own auto-follow state, which is also in-memory. Acceptable.

No env knobs: identity comes from the live Slack adapter, not the environment.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Any, Optional

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

# --- identity resolution (from the live Slack adapter) ----------------------
def _resolve_my_id(gateway: Any, team_id: str, channel_id: str) -> Optional[str]:
    """This bot's Slack user_id for the event's workspace, or None (fail open).

    Reads the Slack adapter's ``_team_bot_user_ids`` (team_id → bot_user_id),
    the authoritative per-workspace map it builds by auth-testing every
    configured token. Keying by the event's own workspace is what makes this
    correct under the adapter's multi-workspace support (comma-separated
    SLACK_BOT_TOKEN / OAuth-added workspaces); assuming one token would fail
    open or pick the wrong identity there.
    """
    adapters = getattr(gateway, "adapters", None)
    if not isinstance(adapters, dict):
        return None
    # Exactly one Slack platform (Platform.SLACK = "slack"); match by key.
    adapter = next((a for k, a in adapters.items() if "slack" in str(k).lower()), None)
    if adapter is None:
        return None
    team_map = getattr(adapter, "_team_bot_user_ids", None)
    if not isinstance(team_map, dict):
        return None
    if not team_id:
        # Event omitted the workspace; the adapter tracks channel → team.
        team_id = getattr(adapter, "_channel_team", {}).get(channel_id, "")
    my_id = team_map.get(team_id)
    if my_id:
        return my_id
    # Single workspace: the sole bot id is unambiguous even when the event's
    # team key differs (e.g. enterprise-grid shapes). Multiple workspaces: an
    # unknown team must NOT fall back to another workspace's id (the wrong
    # identity) — stay unresolved and let the caller fail open.
    if len(team_map) == 1:
        return next(iter(team_map.values()))
    return None


def _slack_fields(event: Any) -> Optional[tuple[str, Optional[str], str, str]]:
    """Return (channel_id, thread_ts, text, team_id) or None if not Slack."""
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
    team_id = ""
    raw = getattr(event, "raw_message", None)
    if isinstance(raw, dict):
        text = raw.get("text") or ""
        # Workspace of the message. The adapter resolves per-workspace identity
        # from the same field (adapter.py: ``event.get("team") or
        # event.get("team_id")``); mirror it so our id lookup keys match.
        team_id = raw.get("team") or raw.get("team_id") or ""
    if not text:
        text = getattr(event, "text", "") or ""
    return channel_id, thread_ts, text, team_id


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
    channel_id, thread_ts, text, team_id = fields

    # DM: 1:1 — every message is for me.
    if channel_id.startswith("D"):
        return None
    # Top-level message: Slack's root @-gating handles it; we only gate threads.
    if not thread_ts:
        return None
    # My Slack user_id for THIS message's workspace (from the live adapter).
    my_id = _resolve_my_id(kwargs.get("gateway"), team_id, channel_id)
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
    logger.info(
        "thread_ownership: registered (per-workspace last-mention ownership)",
    )
