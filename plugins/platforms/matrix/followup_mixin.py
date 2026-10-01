"""Matrix reaction watches and their delivery and admission lifecycle."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from plugins.platforms.matrix.followup_context import LogicalReplyContext, REPLY_EXCERPT_CHARS
from plugins.platforms.matrix.turn_context import MatrixTurnContext
from plugins.platforms.matrix.reaction_followups import (
    FinalDeliveryEvents, PendingFollowupReactions, ReactionWatchStore,
    REGISTRATION_REPLAY_SECONDS, reaction_follows_delivery,
)

logger = logging.getLogger("plugins.platforms.matrix.adapter")


@dataclass(frozen=True)
class _MatrixFollowupChoice:
    turn_id: str
    emoji_filter: tuple[str, ...]
    room_id: str
    requester: str
    thread_id: str
    profile: str
    session_id: str
    pending: PendingFollowupReactions = field(default_factory=PendingFollowupReactions)


class MatrixFollowupMixin:
    def _followup_store_path(self) -> Path:
        return (self._store_dir or self._resolve_store_dir()).parent / "reaction-followups.sqlite"

    def _followup_store(self) -> ReactionWatchStore:
        store = getattr(self, "_reaction_watch_store", None)
        if store is None:
            store = self._reaction_watch_store = ReactionWatchStore(self._followup_store_path())
        return store

    def _purge_expired_watches(self) -> None:
        """Delete expired reaction watches, then run again when the next watch expires."""
        handle = getattr(self, "_watch_purge_handle", None)
        if handle is not None:
            handle.cancel()
        self._watch_purge_handle = None
        store = self._followup_store()
        try:
            next_expiry = store.purge_expired()
        except sqlite3.Error as exc:
            logger.warning("Matrix: could not purge expired reaction watches: %s", exc)
            return
        if next_expiry is not None:
            self._watch_purge_handle = asyncio.get_running_loop().call_later(
                max(0.0, next_expiry - store.clock()), self._purge_expired_watches)

    async def configure_reaction_followups(
        self, session_key: str, enabled: bool, emoji_filter: tuple[str, ...],
        *, room_id: str, requester: str, thread_id: str, profile: str, session_id: str,
    ) -> bool:
        if session_key not in self._active_sessions or not session_id:
            return False
        self._discard_followup_action(session_key)
        if enabled:
            self._reaction_followup_actions[session_key] = _MatrixFollowupChoice(
                uuid.uuid4().hex, emoji_filter, room_id, requester, thread_id, profile, session_id,
            )
        return True

    def requires_edit_finalize(self, chat_id: str) -> bool:
        """A pending follow-up anchors reaction order on a final edit, even for unchanged text."""
        return any(
            action.room_id == chat_id
            for action in getattr(self, "_reaction_followup_actions", {}).values()
        )

    def _discard_followup_action(self, session_key: str) -> None:
        action = self._reaction_followup_actions.pop(session_key, None)
        if action is not None:
            action.pending.clear()

    async def send_final_ledgered(
        self, event: MessageEvent, session_key: str, text_content: str,
        metadata: Dict[str, Any], *, reply_to: Optional[str],
        is_ephemeral_response: bool = False,
    ) -> tuple[SendResult, BasePlatformAdapter]:
        try:
            result, delivery_adapter = await super().send_final_ledgered(
                event, session_key, text_content, metadata, reply_to=reply_to,
                is_ephemeral_response=is_ephemeral_response)
            if result.success and result.message_id:
                ids = (*result.continuation_message_ids, result.message_id)
                replay = self._register_followup_delivery(
                    event.source, session_key, ids, text_content, delivery_adapter)
                if replay is not None:
                    await replay
            return result, delivery_adapter
        finally:
            self._discard_followup_action(session_key)

    def on_streamed_final_delivery(
        self, source: SessionSource, session_key: str, ids: tuple[str, ...], text_content: str,
    ) -> Awaitable[None] | None:
        return self._register_followup_delivery(source, session_key, ids, text_content, self)

    def _register_followup_delivery(
        self, source: SessionSource, session_key: str, ids: tuple[str, ...],
        text_content: str, delivery_adapter: BasePlatformAdapter,
    ) -> Awaitable[None] | None:
        action = self._reaction_followup_actions.get(session_key)
        if not action or source.chat_id != action.room_id:
            return
        if not text_content or not ids:
            self._discard_followup_action(session_key)
            return
        if action.pending.registered:
            return
        if not isinstance(delivery_adapter, MatrixFollowupMixin):
            self._discard_followup_action(session_key)
            return
        delivery_event_id = (
            delivery_adapter._followup_delivery_events.latest(action.room_id, ids)
            if hasattr(delivery_adapter, "_followup_delivery_events") else ids[-1])
        if not delivery_event_id:
            self._discard_followup_action(session_key)
            return
        delivery_adapter._followup_store().arm(
            action.turn_id, ids, profile=action.profile, room_id=action.room_id,
            thread_id=action.thread_id, session_key=session_key, session_id=action.session_id,
            requester=action.requester, source=source.to_dict(),
            emoji_filter=action.emoji_filter, text_content=text_content,
            delivery_event_id=delivery_event_id,
            reply_excerpt=delivery_adapter._followup_delivery_events.excerpt(
                action.room_id, ids, text_content
            ) if hasattr(delivery_adapter, "_followup_delivery_events") else None,
            target_digests=delivery_adapter._followup_delivery_events.target_digests(
                action.room_id, ids
            ) if hasattr(delivery_adapter, "_followup_delivery_events") else None,
        )
        delivery_adapter._purge_expired_watches()
        action.pending.registered = True
        if action.pending.events:
            return self._replay_followup_reactions(session_key, action, ids, delivery_adapter)
        self._discard_followup_action(session_key)

    async def _replay_followup_reactions(
        self, session_key: str, action: _MatrixFollowupChoice, ids: tuple[str, ...],
        delivery_adapter: MatrixFollowupMixin,
    ) -> None:
        try:
            async with asyncio.timeout(REGISTRATION_REPLAY_SECONDS):
                for reaction in tuple(action.pending.events.values()):
                    if reaction.target_event_id not in ids or not action.pending.eligible(reaction.event_id):
                        continue
                    await delivery_adapter._dispatch_reaction(
                        action.room_id, reaction.target_event_id, reaction.emoji,
                        reaction.sender, reaction.event_id, pending=action.pending,
                    )
        except TimeoutError:
            logger.debug("Matrix: reaction registration replay timed out for session %s", session_key)
        except Exception:
            logger.warning("Matrix: reaction registration replay failed for session %s", session_key, exc_info=True)
        finally:
            action.pending.clear()
            if self._reaction_followup_actions.get(session_key) is action:
                self._discard_followup_action(session_key)

    def _remember_followup_delivery(
        self, room_id: str, event_id: str, content: Dict[str, Any], *, finalize: bool,
    ) -> None:
        if not hasattr(self, "_followup_delivery_events"):
            self._followup_delivery_events = FinalDeliveryEvents()
        self._followup_delivery_events.remember(room_id, event_id, content, finalize=finalize)

    async def _handle_followup_reaction(
        self, room_id: str, target_event_id: str, emoji: str, sender: str,
        reaction_event_id: str, *, pending: PendingFollowupReactions | None = None,
    ) -> bool | None:
        store = self._followup_store()
        candidate = store.candidate(room_id, target_event_id)
        if candidate is None:
            for action in getattr(self, "_reaction_followup_actions", {}).values():
                if (action.room_id == room_id and action.requester == sender
                        and (not action.emoji_filter or emoji in action.emoji_filter)):
                    action.pending.remember(target_event_id, emoji, sender, reaction_event_id)
            return
        if candidate["requester"] != sender:
            return
        if (self._is_system_or_bridge_sender(sender)
                or any(pattern.search(sender) for pattern in self._ignored_user_patterns)
                or not await self._is_allowed_matrix_room_event(room_id)):
            return

        saved = candidate["source"]
        identity = await self._resolve_room_identity(room_id)
        source = self.build_source(
            chat_id=room_id, chat_name=identity.display_name,
            chat_type=saved.get("chat_type", "group"), user_id=sender,
            user_name=await self._get_display_name(room_id, sender),
            thread_id=candidate["thread_id"] or None, chat_topic=identity.room_topic,
            scope_id=saved.get("scope_id"), parent_chat_id=saved.get("parent_chat_id"),
            message_id=reaction_event_id,
        )
        if source.profile_route_rejected or self._source_session_key(source) != candidate["session_key"]:
            return
        if (source.profile or "") != candidate["profile"]:
            return
        if self._is_sender_authorized(
            sender, chat_type=source.chat_type, chat_id=room_id,
            thread_id=source.thread_id,
        ) is not True:
            return
        session_store = getattr(self, "_session_store", None)
        if (not candidate["session_id"] or session_store is None
                or await asyncio.to_thread(
                    session_store.peek_session_id, candidate["session_key"]
                ) != candidate["session_id"]):
            return
        if not await reaction_follows_delivery(
            getattr(self, "_client", None), room_id,
            candidate["delivery_event_id"], reaction_event_id,
        ):
            return
        if pending is not None and not pending.eligible(reaction_event_id):
            return
        if getattr(self, "_message_handler", None) is None:
            return False
        claimed = store.claim(
            source.profile or "", room_id, target_event_id, sender, emoji,
            verified_delivery_event_id=candidate["delivery_event_id"])
        if claimed is None:
            return
        reply_text = claimed["text_content"]
        if not reply_text:
            target = await self._event_context_cache.resolve(
                getattr(self, "_client", None), room_id, target_event_id)
            reply_text = target.text if target else ""
        context = (f"Matrix reaction by {sender}: {emoji} on reply {target_event_id} "
                   f"(reaction event {reaction_event_id}).")
        followup = MessageEvent(
            text=context, source=source, message_id=reaction_event_id,
            raw_message={"m.relates_to": {"rel_type": "m.annotation",
                                          "event_id": target_event_id, "key": emoji}},
            reply_to_message_id=target_event_id,
            reply_to_text=reply_text[:REPLY_EXCERPT_CHARS] or None,
            reply_to_is_own_message=True,
            user_id=sender, allow_gateway_control=False, defer_until_idle=True,
            metadata={
                "gateway_session_key": claimed["session_key"],
                "gateway_session_id": claimed["session_id"],
                "gateway_session_strict": True,
            },
        )
        if claimed["reply_excerpt"] is not None:
            snapshot = MatrixTurnContext.capture(self, followup)
            snapshot.logical_reply = LogicalReplyContext.capture(
                self, room_id, reply_text[:REPLY_EXCERPT_CHARS], claimed["reply_excerpt"]
            )
            followup._inbound_context_dependencies = (snapshot,)
        return await self._admit(followup)

