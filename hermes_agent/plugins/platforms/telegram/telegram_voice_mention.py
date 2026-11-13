"""Spoken wake words for group voice notes (``voice_mention_patterns``, #102330).

``require_mention`` + ``mention_patterns`` gate a group message on its text, and a captionless voice
note has none until speech-to-text runs, which is after the gate: saying the wake word never woke
the bot. With ``voice_mention_patterns: true``, a group voice note that only a wake word could admit
is transcribed at the gate and admitted when the transcript matches; the turn reuses that transcript.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from hermes_agent.gateway.platforms.base import cache_audio_from_bytes_async
from hermes_agent.gateway.platforms.helpers import spoken_mention_clips, voice_mention_gate_enabled

if TYPE_CHECKING:
    from hermes_agent.gateway.platforms.event import MessageEvent

logger = logging.getLogger("hermes_agent.plugins.platforms.telegram.adapter")


async def admit_media(adapter: Any, msg: Any) -> Optional[list[str]]:
    """``[]`` when the media message passes the group gate as usual, the voice clip when only its
    spoken wake word does, else ``None`` (dropped or observed)."""
    if adapter._should_process_message(msg):
        return []
    voice = getattr(msg, "voice", None)
    if (voice is None or getattr(msg, "caption", None) or not adapter._mention_patterns
            or not voice_mention_gate_enabled(adapter.config.extra)
            or not adapter._should_process_message(msg, spoken_mention=True)):
        return None
    allowed, _note = adapter._telegram_media_size_allowed(voice, "voice message")
    if not allowed:
        return None
    from hermes_agent.plugins.platforms.telegram.telegram_media_fetch import fetch_telegram_file
    try:
        _file, data = await fetch_telegram_file(voice, "user voice")
        path = await cache_audio_from_bytes_async(data, ext=".ogg")
    except Exception as exc:  # health: allow BLE001 -- a failed fetch leaves the note gated, as before
        logger.warning("[%s] Could not fetch a group voice note for the spoken mention check: %s", adapter.name, exc)
        return None
    from hermes_agent.gateway.platforms.event import MessageType
    source = adapter._build_message_event(msg, MessageType.VOICE).source
    clips = await spoken_mention_clips([path], adapter._mention_patterns, adapter=adapter, source=source)
    logger.info("[%s] Group voice note %s: spoken wake word %s", adapter.name, getattr(msg, "message_id", None),
                "matched" if clips else "not found")
    return clips or None


async def cache_inbound_voice(adapter: Any, msg: Any, event: MessageEvent, clips: list[str]) -> None:
    """Attach the voice note: the clip the spoken-mention gate already fetched and transcribed (a
    wake word addressed the bot, so a reply is expected), else download it as usual."""
    if not clips:
        await adapter._cache_inbound_av(msg, event, msg.voice, "voice message", "voice", ".ogg", "audio/ogg")
        return
    event.media_urls, event.media_types = list(clips), ["audio/ogg"] * len(clips)
    event.reply_expected = True
