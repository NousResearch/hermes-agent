"""Slack Code channels: a channel that is one agent session (https://docs.slack.dev/ai/slack-code/).

Opt-in with ``platforms.slack.extra.code_channels: true``; off, every method is a no-op and the
adapter behaves exactly as before (no extra ``conversations.info`` lookups). On, in a code channel:
people talk to the agent without an @mention, the whole channel is one shared session, replies go
top-level, the session status is channel-level (``agents.sessions.setStatus`` without
``thread_ts``) and Slack's Stop button (``agent_session_stopped``) runs ``/stop`` for the session.
A thread a user starts under a message keeps its own session and stays threaded.

The channel session carries no ``thread_id``; the reply destination rides with each send. With the
default ``reply_in_thread`` the gateway still stamps the triggering message's ts as a synthetic
``thread_id`` on progress / stream metadata, next to ``message_id`` (or ``reply_to``) naming the same
message. So a send is for the channel session when it names no thread, or when its thread is a
recorded top-level code-channel message AND the message it answers is that same message. A turn in a
user-started thread answers a reply (``message_id`` != thread), so it stays threaded, and a thread
opened later under a message cannot move that message's running answer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger("plugins.platforms.slack.adapter")

_ROOTS_MAX = 5000


def _thread_of(metadata: Optional[dict[str, Any]]) -> Optional[str]:
    md = metadata or {}
    return md.get("thread_id") or md.get("thread_ts") or None


def _answered(reply_to: Optional[str], metadata: Optional[dict[str, Any]]) -> Optional[str]:
    """The message a send answers: ``reply_to``, else the ``message_id`` the gateway stamps."""
    return reply_to or (metadata or {}).get("message_id") or None


class SlackCodeChannels:
    """Per-adapter code-channel state. Channel ids are workspace-global in Slack, so the caches key
    by channel id alone."""

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        # channel id -> is a code channel; a failed lookup is not cached, so the next event retries.
        self._kinds: dict[str, bool] = {}
        # In-flight lookups: concurrent events for one channel share it, so they resume in arrival
        # order (two slashes must not overtake each other on the lookup).
        self._pending: dict[str, asyncio.Future[None]] = {}
        # Top-level code-channel message ts, recorded on arrival (bounded, oldest first).
        self._roots: dict[str, None] = {}
        # Channels whose channel-session turn shows ``processing`` (sent once per turn;
        # _keep_typing refreshes often).
        self._busy: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self._adapter._extra_flag("code_channels")

    def is_code_channel(self, channel_id: str) -> bool:
        return bool(channel_id) and self._kinds.get(str(channel_id)) is True

    async def note_inbound(self, event: dict, body: Optional[dict] = None) -> None:
        """Before a message is handled: recognise its channel in the workspace that delivered it."""
        adapter = self._adapter
        team_id = adapter._event_team_id(event, body, install_set=getattr(adapter, "_team_clients", None))
        await self.recognise(str(event.get("channel") or ""), team_id)
        ts = str(event.get("ts") or "")
        if ts and self.is_top_level(event, ts):
            self._roots[ts] = None
            while len(self._roots) > _ROOTS_MAX:
                self._roots.pop(next(iter(self._roots)))

    async def recognise(self, channel: str, team_id: str = "") -> None:
        """One ``conversations.info`` per channel (messages and slash commands both call this). The
        documented signal is ``properties.record_channel.record_type == "agent_channel"``;
        ``properties.code_channel`` can be ``{}`` on a new channel, so it is not used."""
        if not self.enabled or not channel or channel[:1] not in ("C", "G") or channel in self._kinds:
            return
        pending = self._pending.get(channel)
        if pending is None:
            pending = self._pending[channel] = asyncio.ensure_future(self._lookup(channel, team_id))
            pending.add_done_callback(lambda _f: self._pending.pop(channel, None))
        await asyncio.shield(pending)

    async def _lookup(self, channel: str, team_id: str) -> None:
        client = self._adapter._get_client(channel, team_id=team_id or None)
        try:
            resp = await client.conversations_info(channel=channel)
        except Exception as exc:  # inbound must never block on this lookup; retried next event
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

    def _channel_turn(self, chat_id: str, reply_to: Optional[str], metadata: Optional[dict[str, Any]]) -> bool:
        """Is this send for the channel session (see the module docstring)?"""
        if not self.is_code_channel(chat_id):
            return False
        thread = _thread_of(metadata)
        return not thread or (thread in self._roots and _answered(reply_to, metadata) == thread)

    def reply_is_flat(self, reply_to: Optional[str], metadata: Optional[dict[str, Any]]) -> bool:
        """A channel-session send answering a top-level code-channel message goes top-level instead
        of opening a thread under it. Roots are only recorded in code channels, so no channel id is
        needed here."""
        answered, thread = _answered(reply_to, metadata), _thread_of(metadata)
        return bool(answered) and answered in self._roots and (not thread or thread == answered)

    async def show_working(self, chat_id: str, metadata: Optional[dict[str, Any]]) -> bool:
        """``processing`` for a channel-session turn; True when handled here (the thread status path
        does not apply)."""
        if not self._channel_turn(chat_id, None, metadata):
            return False
        if chat_id not in self._busy:
            self._busy.add(chat_id)
            await self._set_status(chat_id, "processing")
        return True

    async def turn_ended(self, chat_id: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """``active`` once the channel-session turn ends (``closed`` would mean the agent stops
        responding). A user-started thread's turn owns no channel status, so it leaves it alone."""
        if chat_id in self._busy and self._channel_turn(chat_id, None, metadata):
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
        if not thread_ts and (channel in self._busy or not was_busy):
            self._busy.add(channel)
            await self.turn_ended(channel)
