"""Turn-loop glue: caller speech → gateway agent turn → spoken reply.

A final inbound transcript becomes a normal gateway ``MessageEvent`` (so
voice gets sessions, history, tools, and memory like every other platform);
the agent's reply comes back through ``adapter.send()`` →
``runtime.speak_for_chat()`` → carrier TTS. The platform_hint registered by
the adapter asks for short spoken-style plain text; :func:`strip_for_speech`
removes whatever markup slips through before it reaches the caller's ear.
"""

import logging
import re
from typing import TYPE_CHECKING

from .events import CallRecord

if TYPE_CHECKING:  # pragma: no cover
    from .runtime import VoiceCallRuntime

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_BARE_URL_RE = re.compile(r"https?://\S+")
_MD_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{1,3}|~~)(\S(?:.*?\S)?)\1")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)


def strip_for_speech(text: str, max_chars: int = 1000) -> str:
    """Reduce agent output to something a TTS voice can say naturally."""
    out = _CODE_FENCE_RE.sub(" code omitted. ", text or "")
    out = _INLINE_CODE_RE.sub(r"\1", out)
    out = _MD_LINK_RE.sub(r"\1", out)
    out = _BARE_URL_RE.sub("a link", out)
    out = _MD_EMPHASIS_RE.sub(r"\2", out)
    out = _MD_HEADING_RE.sub("", out)
    out = _MD_BULLET_RE.sub("", out)
    out = re.sub(r"\s+", " ", out).strip()
    if len(out) > max_chars:
        cut = out[:max_chars]
        # Prefer ending on a sentence boundary.
        for stop in (". ", "! ", "? "):
            idx = cut.rfind(stop)
            if idx > max_chars // 2:
                return cut[: idx + 1].strip()
        out = cut.rstrip() + "…"
    return out


async def dispatch_transcript(
    runtime: "VoiceCallRuntime", record: CallRecord, text: str
) -> None:
    """Route a final caller utterance into the gateway as a MessageEvent."""
    adapter = runtime.adapter
    if adapter is None:
        logger.warning(
            "voice_call: transcript on %s but no gateway adapter is attached "
            "(headless runtime?) — dropping", record.call_id,
        )
        return

    from gateway.platforms.base import MessageEvent, MessageType

    peer = record.peer_number or "unknown"
    thread_id = (
        record.call_id if runtime.config.session_scope == "per-call" else None
    )
    source = adapter.build_source(
        chat_id=peer,
        chat_name=f"Call with {peer}",
        chat_type="dm",
        user_id=peer,
        user_name=peer,
        thread_id=thread_id,
    )
    event = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id=f"{record.call_id}:{len(record.transcript)}",
        raw_message={"call_id": record.call_id, "provider": record.provider},
    )
    await adapter.handle_message(event)
