"""Reactions and generic card clicks routed as synthetic agent turns for ``FeishuAdapter``.

Both must clear the chat's admission gate before they become a turn (#105157).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional

from hermes_agent.gateway.platforms.event import MessageEvent, MessageType

if TYPE_CHECKING:
    from hermes_agent.plugins.platforms.feishu.adapter import RejectReason

logger = logging.getLogger(__name__)


class FeishuSyntheticEventsMixin:
    """Needs the adapter's client, chat-info, sender-profile, admission and guard helpers."""

    async def _handle_reaction_event(self, event_type: str, data: Any) -> None:
        """Fetch the reacted-to message; if it was sent by this bot, emit a synthetic text event."""
        if not self._client:
            return
        event = getattr(data, "event", None)
        message_id = str(getattr(event, "message_id", "") or "")
        if not message_id:
            return
        # Fetch the target message to verify it was sent by us and to obtain chat context.
        try:
            request = self._build_get_message_request(message_id)
            response = await self._run_blocking(self._client.im.v1.message.get, request)
            if not self._response_succeeded(response):
                return
            items = getattr(getattr(response, "data", None), "items", None) or []
            msg = items[0] if items else None
            if not msg:
                return
            # GET im/v1/messages reports sender.id=app_id for bot messages — peer bots share
            # sender_type="app" with us but differ on app_id. Only route our own messages.
            sender = getattr(msg, "sender", None)
            if str(getattr(sender, "id", "") or "") != self._app_id:
                return
            chat_id = str(getattr(msg, "chat_id", "") or "")
            if not chat_id:
                return
        except Exception:
            logger.debug("[Feishu] Failed to fetch message for reaction routing", exc_info=True)
            return
        user_id_obj = getattr(event, "user_id", None)
        reaction_type_obj = getattr(event, "reaction_type", None)
        emoji_type = str(getattr(reaction_type_obj, "emoji_type", "") or "UNKNOWN")
        action = "added" if "created" in event_type else "removed"
        synthetic_text = f"reaction:{action}:{emoji_type}"

        # Reactions become agent turns, so they must clear the same admission gate as text
        # from this operator in this chat; otherwise anyone who can see a bot message in a
        # group wakes the agent by reacting, even under FEISHU_GROUP_POLICY=allowlist.
        # GET im/v1/messages items carry no chat_type, so the chat lookup decides.
        chat_type_raw = await self._synthetic_chat_type(chat_id)
        reject = self._admit_synthetic(user_id_obj, chat_id=chat_id, chat_type=chat_type_raw)
        if reject is not None:
            logger.debug(
                "[Feishu] dropping reaction %s:%s on %s: %s",
                action, emoji_type, message_id, reject,
            )
            return
        logger.info("[Feishu] Routing reaction %s:%s on bot message %s as synthetic event", action, emoji_type, message_id)
        await self._dispatch_synthetic_event(
            text=synthetic_text, message_type=MessageType.TEXT, chat_id=chat_id, sender_id=user_id_obj,
            event_chat_type=chat_type_raw, raw_message=data, message_id=message_id,
        )

    async def _handle_card_action_event(self, data: Any) -> None:
        """Route Feishu interactive card button clicks as synthetic COMMAND events."""
        event = getattr(data, "event", None)
        token = str(getattr(event, "token", "") or "")
        if token and self._is_card_action_duplicate(token):
            logger.debug("[Feishu] Dropping duplicate card action token: %s", token)
            return
        context = getattr(event, "context", None)
        chat_id = str(getattr(context, "open_chat_id", "") or "")
        operator = getattr(event, "operator", None)
        open_id = str(getattr(operator, "open_id", "") or "")
        if not chat_id or not open_id:
            logger.debug("[Feishu] Card action missing chat_id or operator open_id, dropping")
            return
        action = getattr(event, "action", None)
        action_tag = str(getattr(action, "tag", "") or "button")
        action_value = getattr(action, "value", {}) or {}
        synthetic_text = f"/card {action_tag}"
        if action_value:
            try:
                synthetic_text += f" {json.dumps(action_value, ensure_ascii=False)}"
            except (TypeError, ValueError):  # non-JSON value: route the bare tag
                logger.debug("[Feishu] Card action value is not JSON-serializable", exc_info=True)
        # Keep every operator id tier: allowlists may name the tenant user_id or union_id.
        sender_id = SimpleNamespace(
            open_id=open_id,
            user_id=str(getattr(operator, "user_id", "") or "") or None,
            union_id=str(getattr(operator, "union_id", "") or "") or None,
        )
        # Same gate as reactions; the callback carries no chat type either.
        chat_type = await self._synthetic_chat_type(chat_id)
        reject = self._admit_synthetic(sender_id, chat_id=chat_id, chat_type=chat_type)
        if reject is not None:
            logger.debug("[Feishu] dropping card action %r from %s in %s: %s", action_tag, open_id, chat_id, reject)
            return
        logger.info("[Feishu] Routing card action %r from %s in %s as synthetic command", action_tag, open_id, chat_id)
        await self._dispatch_synthetic_event(
            text=synthetic_text, message_type=MessageType.COMMAND, chat_id=chat_id,
            sender_id=sender_id, event_chat_type=chat_type,
            raw_message=data, message_id=token or str(uuid.uuid4()),
        )

    async def _synthetic_chat_type(self, chat_id: str) -> str:
        """"p2p" only when the (cached) chat lookup reports chat_mode p2p; else "group".

        Unknown fails closed to group: a DM reaction/click whose lookup fails is judged by
        group policy (and dropped under the pairing default) rather than skipping it.
        """
        return "p2p" if (await self.get_chat_info(chat_id)).get("raw_mode") == "p2p" else "group"

    async def _dispatch_synthetic_event(
        self, *, text: str, message_type: MessageType, chat_id: str, sender_id: Any, event_chat_type: str,
        raw_message: Any, message_id: str,
    ) -> None:
        """Wrap a reaction/card click as a MessageEvent and run it through the guarded pipeline."""
        sender_profile = await self._resolve_sender_profile(sender_id)
        chat_info = await self.get_chat_info(chat_id)
        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_info.get("name") or chat_id or "Feishu Chat",
            chat_type=self._resolve_source_chat_type(chat_info=chat_info, event_chat_type=event_chat_type),
            user_id=sender_profile["user_id"],
            user_name=sender_profile["user_name"],
            thread_id=None,
            user_id_alt=sender_profile["user_id_alt"],
        )
        synthetic_event = MessageEvent(
            text=text, message_type=message_type, source=source, raw_message=raw_message,
            message_id=message_id, channel_prompt=self._resolve_channel_prompt(chat_id),
            timestamp=datetime.now(),
        )
        await self._handle_message_with_guards(synthetic_event)

    def _admit_synthetic(self, user_id_obj: Any, *, chat_id: str, chat_type: str) -> Optional[RejectReason]:
        """Admission for a reaction / card click that would become a synthetic agent turn.

        Same ``_admit`` as text (DM allowlist / pairing intake, self-echo, group policy,
        per-chat rules, admins) minus the mention requirement: a reaction or click carries
        no @ payload.
        """
        sender = SimpleNamespace(sender_type="user", sender_id=user_id_obj)
        message = SimpleNamespace(chat_type=chat_type, chat_id=chat_id, mentions=None, content="", message_type="text")
        reject = self._admit(sender, message, require_mention=False)
        if reject == "group_policy_rejected":
            self._warn_once_empty_allowlist_deny(chat_id)
        return reject
