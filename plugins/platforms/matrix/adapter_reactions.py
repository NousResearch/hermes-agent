"""Matrix reaction answers (``matrix.reaction_triggers``, off by default).

A listed emoji reaction from an authorized user on a message the bot itself sent becomes a
``reaction:added:<emoji>`` turn replying to that message (the Slack/Photon convention), so a
👍/👎 on a yes/no question reaches the agent as the answer to that question.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# A reaction key is free text, so only short emoji route, and the variation selectors / skin tones
# clients add (Element sends 👍️) are dropped before matching.
_REACTION_TRIGGERS_DEFAULT = frozenset({"\U0001f44d", "\U0001f44e"})  # 👍 👎
_REACTION_KEY_MAX_CHARS = 8
_REACTION_KEY_MODIFIERS = re.compile("[︎️\U0001f3fb-\U0001f3ff]")
_TARGET_FETCH_TIMEOUT_S = 10.0
_REACTION_ANSWER_MARK = "_hermes_reaction_answer"  # on the synthetic turn's raw_message


def normalize_reaction_key(key: Any) -> str:
    """The bare emoji of a reaction key, or "" when the key is not a short emoji (text, spaces,
    letters/digits, control or format characters)."""
    key = _REACTION_KEY_MODIFIERS.sub("", str(key or "")).strip()
    if not key or len(key) > _REACTION_KEY_MAX_CHARS or any(
            ch.isspace() or ch.isalnum() or not ch.isprintable() for ch in key):
        return ""
    return key


def parse_reaction_triggers(raw: Any) -> Optional[frozenset]:
    """None = off (default); true/on/all = 👍 and 👎; otherwise the listed emoji (YAML list or
    comma/space separated). A list without a single valid emoji is off."""
    if raw is None or raw is False:
        return None
    if raw is True:
        return _REACTION_TRIGGERS_DEFAULT
    if isinstance(raw, (list, tuple, set, frozenset)):
        items = [str(i) for i in raw]
    else:
        text = str(raw).strip()
        if text.lower() in {"", "false", "0", "no", "off"}:
            return None
        if text.lower() in {"true", "1", "yes", "on", "all", "*"}:
            return _REACTION_TRIGGERS_DEFAULT
        items = re.split(r"[,\s]+", text)
    keys = frozenset(k for k in (normalize_reaction_key(i) for i in items) if k)
    return keys or None


def _target_text(content: dict) -> str:
    """What the reacted-to message said, for the ``[Replying to your previous message: ...]`` note."""
    from plugins.platforms.matrix import adapter as matrix

    text = content.get("body")
    text = matrix._strip_reply_fallback(text).strip() if isinstance(text, str) else ""
    msgtype = str(content.get("msgtype") or "")
    if msgtype and msgtype not in ("m.text", "m.notice"):
        kind = msgtype.removeprefix("m.")
        return f"[{kind}: {text}]" if text else f"[{kind}]"
    return text


class MatrixReactionAnswersMixin:
    """Inbound reactions for :class:`MatrixAdapter`; every other attribute used here lives there."""

    _reaction_triggers: Optional[frozenset] = None  # class default keeps object.__new__ doubles off

    @staticmethod
    def _lifecycle_reaction_target(event: Any) -> Optional[str]:
        """The event the 👀/✅/❌ processing reactions annotate: none for a reaction answer, whose
        message_id is the user's reaction event, not a message."""
        raw = getattr(event, "raw_message", None)
        return None if isinstance(raw, dict) and raw.get(_REACTION_ANSWER_MARK) else event.message_id

    async def _on_reaction(self, event: Any) -> None:
        """m.reaction entry point: approval and picker prompts first, then (opt-in) reaction answers."""
        sender = str(getattr(event, "sender", ""))
        if self._is_self_sender(sender):
            return
        event_id = str(getattr(event, "event_id", ""))
        if self._is_duplicate_event(event_id):
            return
        room_id = str(getattr(event, "room_id", ""))
        content = getattr(event, "content", None)
        if not content:
            return
        relates_to = (content.get("m.relates_to", {}) if isinstance(content, dict)
                      else getattr(content, "relates_to", {}))
        reacts_to = key = ""
        if isinstance(relates_to, dict):
            reacts_to = relates_to.get("event_id", "")
            key = relates_to.get("key", "")
        elif hasattr(relates_to, "event_id"):
            reacts_to = str(getattr(relates_to, "event_id", ""))
            key = str(getattr(relates_to, "key", ""))
        logger.info("Matrix: reaction %s from %s on %s in %s", key, sender, reacts_to, room_id)
        for handler in (self._handle_approval_reaction, self._handle_model_picker_reaction,
                        self._handle_choice_picker_reaction):
            if await handler(room_id, reacts_to, key, sender):
                return
        if self._reaction_triggers:
            await self._route_reaction_answer(event, room_id, event_id, sender, reacts_to, key)

    async def _load_room_event(self, room_id: str, event_id: str) -> Any:
        """Fetch one event as the homeserver holds it, decrypted with the client's own Olm machine
        when encrypted (fetch and decrypt share one deadline). Raises on failure; None when the event
        is encrypted and no crypto is loaded."""
        from plugins.platforms.matrix import adapter as matrix

        async def _load():
            evt = await self._client.get_event(matrix.RoomID(room_id), matrix.EventID(event_id))
            if str(getattr(evt, "type", "")) == str(matrix.EventType.ROOM_ENCRYPTED):
                crypto = getattr(self._client, "crypto", None)
                if crypto is None:
                    return None
                evt = await crypto.decrypt_megolm_event(evt)
            return evt

        return await asyncio.wait_for(_load(), timeout=_TARGET_FETCH_TIMEOUT_S)

    async def _reaction_answer_allowed(self, event: Any, room_id: str, sender: str, reacts_to: str) -> bool:
        """The gates a typed message passes: bridge/ignored senders, room allowlist, live user
        authorization, startup-sync replay."""
        from plugins.platforms.matrix import adapter as matrix

        if self._is_system_or_bridge_sender(sender) or any(
                pattern.search(sender or "") for pattern in self._ignored_user_patterns):
            return False
        if not await self._is_allowed_matrix_room_event(room_id):
            return False
        if not self._is_authorized_user(sender, room_id):
            logger.info("Matrix: ignoring reaction answer from unauthorized user %s on %s", sender, reacts_to)
            return False
        event_ts = matrix._matrix_event_timestamp_seconds(event)
        return not (event_ts and event_ts < self._startup_ts - matrix._STARTUP_GRACE_SECONDS)

    async def _route_reaction_answer(
            self, event: Any, room_id: str, event_id: str, sender: str, reacts_to: str, key: str) -> None:
        """A listed emoji from an authorized user on the bot's OWN message becomes a
        ``reaction:added:<emoji>`` turn replying to that message, in that message's thread (an
        unthreaded message becomes the thread root when policy would thread a reply anyway).

        The reacted-to sender is read from the homeserver, never from the reaction. A reaction key is
        free text, so only short listed emoji route, never arbitrary text into the prompt. Reactions
        on other people's messages do nothing. A reaction is an answer, not a question: the turn is
        marked ``reply_expected=False`` so the agent may act on it without replying."""
        from plugins.platforms.matrix import adapter as matrix

        emoji = normalize_reaction_key(key)
        if not reacts_to or emoji not in self._reaction_triggers:
            return
        if not await self._reaction_answer_allowed(event, room_id, sender, reacts_to):
            return
        try:
            target = await self._load_room_event(room_id, reacts_to)
        except Exception:  # fetch/decrypt boundary: any failure only drops this one reaction
            logger.warning("Matrix: could not load reacted-to event %s", reacts_to, exc_info=True)
            return
        if target is None or self._is_self_sender(sender) or not self._is_self_sender(
                str(getattr(target, "sender", "") or "")):
            return
        content = getattr(target, "content", None)
        if hasattr(content, "serialize"):
            content = content.serialize()
        content = content if isinstance(content, dict) else {}
        target_thread = matrix._thread_root(content.get("m.relates_to") or {})
        if not target_thread and self._wants_synthetic_thread(await self._is_dm_room(room_id), True):
            target_thread = reacts_to
        body = f"reaction:added:{emoji}"
        relates_to: dict = {"m.in_reply_to": {"event_id": reacts_to}}
        if target_thread:
            relates_to.update({"rel_type": "m.thread", "event_id": target_thread})
        source_content = {"msgtype": "m.text", "body": body, "m.relates_to": relates_to, _REACTION_ANSWER_MARK: True}
        # mention_claimed: a reaction on the bot's own message is addressed to the bot.
        ctx = await self._resolve_message_context(
            room_id, sender, event_id, body, source_content, relates_to, mention_claimed=True)
        if ctx is None:
            return
        msg_event = await self._build_inbound_event(
            room_id, sender, event_id, body, source_content, relates_to, ctx=ctx)
        if msg_event is None:
            return
        msg_event.reply_to_text = _target_text(content) or None
        msg_event.reply_to_author_id = str(getattr(target, "sender", "") or "") or None
        msg_event.reply_to_is_own_message = True
        msg_event.reply_expected = False
        await self.handle_message(msg_event)
