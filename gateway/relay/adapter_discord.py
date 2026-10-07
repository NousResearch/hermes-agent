"""Discord interaction lane of the RelayAdapter: a forwarded raw interaction (slash command,
component, modal) decoded to the same ``MessageEvent`` shape the relay text lane delivers, plus the
text lane's chat labels the interaction body does not carry."""

from __future__ import annotations

import asyncio
import json
import logging
import re

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource

# Log-record parity with the adapter module.
logger = logging.getLogger("gateway.relay.adapter")

# Connector promptCodec.decodePromptCallback id alphabet ([A-Za-z0-9_.-], <=32).
_PROMPT_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,32}$")


class DiscordInteractionMixin:
    """RelayAdapter methods for the Discord interaction lane (``_on_passthrough``) and the text
    lane's label record it reads."""

    _DISCORD_LABELS_MAX = 2048

    async def _remember_discord_labels(self, source) -> None:
        """Keep the text lane's Discord chat labels for the interaction lane: the pinned
        session-context prompt renders them, so a slash turn without them re-rendered the cached
        prefix and the next message rendered it back."""
        # A message carrying no labels is an observation too (a topic or name removed upstream):
        # the interaction must render what the text lane last rendered, not an older label.
        if getattr(source, "platform", None) != Platform.DISCORD or not source.chat_id:
            return
        key, labels = (str(source.scope_id or ""), str(source.chat_id)), (source.chat_name, source.chat_topic)
        self._discord_chat_labels[key] = labels
        self._evict_oldest(self._discord_chat_labels, self._DISCORD_LABELS_MAX)
        store = getattr(self, "_session_store", None)
        if store is None or self._discord_labels_recorded.get(key) == labels:
            return
        # Recorded when observed, so the interaction lane has them after a restart whatever order
        # the chat's sessions were created or reset in. Awaited before dispatch (the reader delivers
        # events one at a time) but off the loop: it is a disk write. One writer thread keeps records
        # in observation order: a cancelled caller's write keeps running, and on a fresh thread the
        # replay's newer write could land first and be overwritten with the older labels.
        try:
            recorded = await asyncio.get_running_loop().run_in_executor(
                self._discord_labels_writer, store.record_chat_labels, source)
        except Exception:
            logger.debug("relay: Discord chat labels not recorded", exc_info=True)
            recorded = False
        if not recorded:
            # Not written (store fault, or no database handle right now): leave the key out so the
            # next message with these labels tries again.
            self._discord_labels_recorded.pop(key, None)
            return
        self._discord_labels_recorded[key] = labels
        self._evict_oldest(self._discord_labels_recorded, self._DISCORD_LABELS_MAX)

    async def _discord_chat_labels_for(self, scope: str, chat_id: str) -> tuple:
        """The text lane's last (chat_name, chat_topic) for a chat. After a restart (or eviction) the
        map is empty until the text lane speaks again, so ask the session store for what that lane
        last recorded; otherwise a first interaction re-renders the prompt the cache still holds."""
        key = (scope, chat_id)
        if key in self._discord_chat_labels:
            return self._discord_chat_labels[key]
        store = getattr(self, "_session_store", None)
        try:
            # Off the loop: the read takes the database's writer lock, which a write may be holding.
            recorded = (await asyncio.to_thread(store.chat_labels, Platform.DISCORD, scope, chat_id)
                        if store is not None else None)
        except Exception:
            # Labels only keep the prompt cache warm; a store fault must not drop the interaction.
            # Not cached either: the next interaction asks the store again.
            logger.debug("relay: recorded Discord chat labels unreadable", exc_info=True)
            return (None, None)
        if recorded is not None:
            self._discord_labels_recorded[key] = recorded
            self._evict_oldest(self._discord_labels_recorded, self._DISCORD_LABELS_MAX)
        self._discord_chat_labels[key] = recorded or (None, None)
        self._evict_oldest(self._discord_chat_labels, self._DISCORD_LABELS_MAX)
        return self._discord_chat_labels[key]

    def _discord_interaction_to_event(self, forward):
        """Convert a forwarded Discord interaction body to a MessageEvent, or None for
        an unusable body (a PING is answered at the edge and never forwarded). The
        session source mirrors the connector's ``interactionSessionSource`` so the
        session key matches the one the follow-up capability was bound under."""
        try:
            payload = json.loads(bytes(getattr(forward, "body", b"")).decode("utf-8"))
        except Exception:  # health: allow BLE001 -- moved from adapter.py unchanged; any undecodable body is unusable
            return None
        if not isinstance(payload, dict):
            return None
        # type 2 = APPLICATION_COMMAND; 3 = MESSAGE_COMPONENT; 5 = MODAL_SUBMIT.
        itype = payload.get("type")
        data = payload.get("data") or {}
        message_type = MessageType.TEXT
        if itype == 2:
            # Normalize to a leading-slash command string ("/name arg…"), the
            # shape the dispatcher and the connector's Slack slash lane expect.
            text = ("/" + str(data.get("name") or "")).rstrip("/") or ""
            if text:
                parts = [text] + self._render_interaction_options(data.get("options"))
                text = " ".join(parts).strip()
                message_type = MessageType.COMMAND
        elif itype == 3:
            text = str(data.get("custom_id") or "")
        else:
            text = ""
        member = payload.get("member") or {}
        user = (member.get("user") if isinstance(member, dict) else None) or payload.get("user") or {}
        if not isinstance(user, dict):
            user = {}
        guild_id = payload.get("guild_id")
        # The text lane's user_display_name is the native author.display_name: guild nick, else
        # global name, else username. The interaction carries all three as of now, so derive the
        # same name from it; a remembered one would outlive a nickname change.
        user_name = next(
            (str(v) for v in ((member.get("nick") if isinstance(member, dict) else None),
                              user.get("global_name"), user.get("username")) if v), None)
        chat_id = str(payload.get("channel_id") or "")
        # The text lane keys a message inside a thread on chat_type "thread" + thread_id (both session-key
        # fields), so an interaction sent there must carry the same or it lands in a per-user "group"
        # session beside the thread's. The partial channel object marks a thread by type (10 announcement,
        # 11 public, 12 private: what discord.py's Thread covers) and names the parent.
        channel = payload.get("channel") if isinstance(payload.get("channel"), dict) else {}
        is_thread = bool(guild_id) and channel.get("type") in (10, 11, 12)
        source = SessionSource(
            # The LOGICAL platform, not RELAY: session keys must match the connector's
            # capability binding (platform="discord"), /sethome must file under the
            # logical platform, and _capture_scope skips the generic "relay".
            platform=Platform.DISCORD,
            chat_id=chat_id,
            # "group", not "channel": both the connector's capability binding and the
            # native Discord adapter key guild channels as "group".
            chat_type="thread" if is_thread else ("group" if guild_id else "dm"),
            thread_id=chat_id if is_thread else None,
            parent_chat_id=str(channel["parent_id"]) if is_thread and channel.get("parent_id") else None,
            user_id=str(user["id"]) if user.get("id") else None,
            user_name=user_name,
            # chat_name / chat_topic: filled by _on_passthrough from what the text lane recorded.
            scope_id=str(guild_id) if guild_id else None,
            message_id=str(payload.get("id")) if payload.get("id") else None,
            # Same upstream-trust marker the relay text lane stamps. Set locally, never
            # read off the wire (engages /sethome's via_relay guard).
            delivered_via_upstream_relay=True,
            # Profile routing (multiplex mode), mirroring _event_from_wire.
            # The HERMES profile this interaction is routed to (multiplex mode) — mirrors _event_from_wire's
            # profile stamping for plain relayed messages (#60586). Without this, a Team-Gateway's Discord
            # slash-command/button/modal always fell back to the legacy agent:main namespace even when the
            # connector resolved a specific profile for it.
            profile=getattr(forward, "profile", None),
        )
        event = MessageEvent(text=text, message_type=message_type, source=source)
        if itype == 3:
            # A component press whose custom_id is a Hermes prompt token
            # (hp1:<prompt_id>:<option_id>) becomes a STRUCTURED prompt answer;
            # foreign custom_ids keep the best-effort TEXT shape.
            decoded = self._decode_prompt_token(text)
            if decoded:
                prompt_id, option_id = decoded
                msg = payload.get("message") or {}
                prompt_message_id = str(msg["id"]) if isinstance(msg, dict) and msg.get("id") else None
                event.prompt_response = {
                    "prompt_id": prompt_id,
                    "option_id": option_id,
                    "prompt_message_id": prompt_message_id,
                }
                event.text = f"/{option_id}"
                event.message_type = MessageType.COMMAND
        return event

    @staticmethod
    def _decode_prompt_token(token: str):
        """Decode an hp1:<prompt_id>:<option_id> callback token, or None (mirrors the connector's promptCodec)."""
        parts = (token or "").split(":")
        if len(parts) != 3 or parts[0] != "hp1":
            return None
        if not _PROMPT_ID_RE.match(parts[1]) or not _PROMPT_ID_RE.match(parts[2]):
            return None
        return parts[1], parts[2]

    @staticmethod
    def _render_interaction_options(options) -> list:
        """Render Discord interaction options to text parts: scalars contribute their
        value (native ``f"/model {name}"`` shape); SUB_COMMAND (1) / SUB_COMMAND_GROUP
        (2) contribute their name then recurse into nested options."""
        parts: list = []
        if not isinstance(options, list):
            return parts
        for opt in options:
            if not isinstance(opt, dict):
                continue
            if opt.get("type") in (1, 2):
                sub_name = str(opt.get("name") or "").strip()
                if sub_name:
                    parts.append(sub_name)
                parts.extend(DiscordInteractionMixin._render_interaction_options(opt.get("options")))
            else:
                value = opt.get("value")
                if value is not None and str(value).strip():
                    parts.append(str(value).strip())
        return parts
