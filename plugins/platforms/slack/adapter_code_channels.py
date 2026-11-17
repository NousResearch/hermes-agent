"""Slack Code channels: a channel that is one agent session (https://docs.slack.dev/ai/slack-code/).

Opt-in with ``platforms.slack.extra.code_channels: true``; off, every method is a no-op and the
adapter behaves exactly as before (no extra ``conversations.info`` lookups). On, in a code channel:
people talk to the agent without an @mention, the whole channel is one shared session, replies go
top-level, the session status is channel-level (``agents.sessions.setStatus`` without
``thread_ts``) and Slack's Stop button (``agent_session_stopped``) runs ``/stop`` for the session.
A thread a user starts under a message keeps its own session and stays threaded.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("plugins.platforms.slack.adapter")

_ROOTS_MAX = 5000
# Message subtypes that are a person talking; joins, edits and bot posts never become roots.
_PLAIN_SUBTYPES = (None, "file_share", "thread_broadcast")


def _bounded_add(entries: dict[str, None], key: str) -> None:
    entries[key] = None
    while len(entries) > _ROOTS_MAX:
        entries.pop(next(iter(entries)))


class SlackCodeChannels:
    """Per-adapter code-channel state. Channel ids are workspace-global in Slack, so the caches key
    by channel id alone."""

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        # channel id -> is a code channel; a failed lookup is not cached, so the next message retries.
        self._kinds: dict[str, bool] = {}
        # Top-level message ts seen in code channels: replies anchored on them go top-level.
        self._roots: dict[str, None] = {}
        # Roots a person replied under: that thread is theirs, so replies there stay threaded.
        self._threaded: dict[str, None] = {}
        # Channels currently shown as ``processing`` (sent once per turn; _keep_typing refreshes often).
        self._busy: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self._adapter._extra_flag("code_channels")

    def is_code_channel(self, channel_id: str) -> bool:
        return bool(channel_id) and self._kinds.get(str(channel_id)) is True

    async def note_inbound(self, event: dict) -> None:
        """Before a message is handled: recognise its channel and remember thread anchors."""
        channel = str(event.get("channel") or "")
        if not self.enabled or not channel or channel[:1] not in ("C", "G"):
            return
        if channel not in self._kinds:
            await self._detect(channel, event)
        if not self.is_code_channel(channel) or event.get("subtype") not in _PLAIN_SUBTYPES:
            return
        ts, thread_ts = str(event.get("ts") or ""), str(event.get("thread_ts") or "")
        if ts and (not thread_ts or thread_ts == ts):
            _bounded_add(self._roots, ts)
        elif thread_ts:
            _bounded_add(self._threaded, thread_ts)

    async def _detect(self, channel: str, event: dict) -> None:
        """The documented signal is ``properties.record_channel.record_type == "agent_channel"``;
        ``properties.code_channel`` can be ``{}`` on a new channel, so it is not used."""
        client = self._adapter._get_client(channel, team_id=event.get("team") or None)
        try:
            resp = await client.conversations_info(channel=channel)
        except Exception as exc:  # inbound must never block on this lookup; retried next message
            logger.debug("[Slack] code-channel lookup for %s failed: %s", channel, exc, exc_info=True)
            return
        props = (resp.get("channel") or {}).get("properties") or {}
        self._kinds[channel] = (props.get("record_channel") or {}).get("record_type") == "agent_channel"

    def share(self, source: Any) -> Any:
        """A code channel is one session for everyone in it, whatever ``group_sessions_per_user``
        says; set on every top-level source built for the channel (messages, slashes, Stop) so they
        key alike. A user-started thread keeps the normal thread rules."""
        if not source.thread_id and self.is_code_channel(source.chat_id):
            source.shared_session = True
        return source

    def is_top_level(self, event: dict, ts: str) -> bool:
        """A code-channel message that is not a reply: it belongs to the channel's one session."""
        thread_ts = event.get("thread_ts")
        return self.is_code_channel(event.get("channel", "")) and not (thread_ts and thread_ts != ts)

    def reply_is_flat(self, reply_to: Optional[str], metadata: Optional[dict[str, Any]]) -> bool:
        """A reply to a top-level code-channel message goes top-level. Inbound stamps the message's
        own ts as ``metadata.thread_id`` and the streaming path passes no ``reply_to``, so key on
        either."""
        if not self._roots:
            return False
        md = metadata or {}
        thread = md.get("thread_id") or md.get("thread_ts")
        anchor = thread or reply_to
        return (anchor in self._roots and anchor not in self._threaded
                and (not thread or not reply_to or thread == reply_to))

    async def show_working(self, chat_id: str, metadata: Optional[dict[str, Any]]) -> bool:
        """``processing`` for a top-level code-channel turn; True when handled here (the thread
        status path does not apply)."""
        md = metadata or {}
        if not self.is_code_channel(chat_id) or self._adapter._resolve_thread_ts(md.get("message_id"), md):
            return False
        if chat_id not in self._busy:
            self._busy.add(chat_id)
            await self._set_status(chat_id, "processing")
        return True

    async def turn_ended(self, chat_id: str) -> None:
        """``active`` once the turn ends (``closed`` would mean the agent stops responding)."""
        if chat_id in self._busy:
            self._busy.discard(chat_id)
            await self._set_status(chat_id, "active")

    async def _set_status(self, chat_id: str, status: str) -> None:
        try:
            await self._adapter._get_client(chat_id).api_call(
                "agents.sessions.setStatus", json={"channel_id": chat_id, "status": status})
        except Exception as exc:  # status is cosmetic; the turn itself is unaffected
            logger.debug("[Slack] agents.sessions.setStatus %s in %s failed: %s", status, chat_id, exc,
                         exc_info=True)

    async def on_session_stopped(self, event: dict, body: Optional[dict] = None) -> None:
        """Stop pressed in a code channel: the same ``/stop`` a typed slash runs (channel gates and
        user authorization included), then move the session off ``processing``; Slack does not."""
        channel = str((event or {}).get("channel") or "")
        if not self.is_code_channel(channel):
            return
        adapter, user_id = self._adapter, str(event.get("user") or "")
        thread_ts = str(event.get("thread_ts") or "") or None
        # A click from a gated channel or an unauthorized user changes nothing, status included.
        if adapter._slash_channel_gated(channel) or adapter._early_reject_unauthorized(
                user_id, channel, False, thread_id=thread_ts):
            return
        command = {"command": "/stop", "text": "", "user_id": user_id, "channel_id": channel,
                   "team_id": adapter._event_team_id(event, body)}
        if thread_ts:
            command["thread_ts"] = thread_ts
        logger.info("[Slack] agent_session_stopped in code channel %s: running /stop", channel)
        was_busy = channel in self._busy
        await adapter._handle_slash_command(command)
        # The interrupted turn's stop_typing usually sent ``active`` already. Untracked (a restart
        # since the turn began) Slack still shows ``processing``, so send it then too.
        if channel in self._busy or not was_busy:
            self._busy.add(channel)
            await self.turn_ended(channel)
