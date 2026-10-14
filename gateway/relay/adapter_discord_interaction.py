"""Forwarded Discord interaction -> ``MessageEvent`` normalization for the relay adapter.

``RelayAdapter`` mixes this in; the passthrough path (``RelayAdapter._on_passthrough``) calls
``_discord_interaction_to_event`` for every connector-forwarded Discord interaction body.
"""

from __future__ import annotations

import json
import re

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource

# Connector promptCodec.decodePromptCallback id alphabet ([A-Za-z0-9_.-], <=32).
_PROMPT_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,32}$")


class RelayDiscordInteractionMixin:
    """Decode connector-forwarded Discord interactions into normalized inbound events."""

    def _discord_interaction_to_event(self, forward):
        """Convert a forwarded Discord interaction body to a MessageEvent, or None for
        an unusable body (a PING is answered at the edge and never forwarded). The
        session source mirrors the connector's ``interactionSessionSource`` so the
        session key matches the one the follow-up capability was bound under."""
        try:
            payload = json.loads(bytes(getattr(forward, "body", b"")).decode("utf-8"))
        except Exception:
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
        source = SessionSource(
            # The LOGICAL platform, not RELAY: session keys must match the connector's
            # capability binding (platform="discord"), /sethome must file under the
            # logical platform, and _capture_scope skips the generic "relay".
            platform=Platform.DISCORD,
            chat_id=str(payload.get("channel_id") or ""),
            # "group", not "channel": both the connector's capability binding and the
            # native Discord adapter key guild channels as "group".
            chat_type="group" if guild_id else "dm",
            user_id=str(user["id"]) if user.get("id") else None,
            user_name=str(user["username"]) if user.get("username") else None,
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
                parts.extend(RelayDiscordInteractionMixin._render_interaction_options(opt.get("options")))
            else:
                value = opt.get("value")
                if value is not None and str(value).strip():
                    parts.append(str(value).strip())
        return parts
