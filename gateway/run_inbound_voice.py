"""Voice transcription and pending transcript delivery for GatewayRunner."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from gateway.run_common import _UNSET
from gateway.platforms.base_pending import pending_dispatch_revision, pending_dispatch_withdrawn

if TYPE_CHECKING:
    from gateway.run import GatewayRunner
    from gateway.platforms.event import MessageEvent

logger = logging.getLogger("gateway.run")


@dataclass(frozen=True)
class VoiceClipTranscript:
    path: str
    text: str


@dataclass(frozen=True)
class VoiceTranscriptPart:
    text: str
    clip: VoiceClipTranscript | None = None


@dataclass(frozen=True)
class VoiceTranscription:
    text: str
    clips: tuple[VoiceClipTranscript, ...] = ()
    parts: tuple[VoiceTranscriptPart, ...] = ()

    def transcripts(self) -> list[str]:
        return [clip.text for clip in self.clips]

    def authored_text(self, user_text: str) -> str:
        return "\n\n".join(text for text in (user_text, *(clip.text for clip in self.clips)) if text).strip()

    def render(self, user_text: str) -> str:
        notes = "\n\n".join(
            f'"{part.clip.text}"' if part.clip is not None else part.text
            for part in self.parts
        )
        return GatewayInboundVoiceMixin._prepend_media_prefix(notes, user_text) if notes else user_text


class GatewayInboundVoiceMixin:
    """Prepare voice input and deliver transcript echoes."""

    if TYPE_CHECKING:
        _delivery_adapter_for = GatewayRunner._delivery_adapter_for
        _session_key_for_source = GatewayRunner._session_key_for_source


    _EMPTY_TEXT_PLACEHOLDER = "(The user sent a message with no text content)"

    @classmethod
    def _prepend_media_prefix(cls, prefix: str, user_text: str) -> str:
        """``prefix`` + the user's text; the Discord empty-content placeholder is dropped as redundant."""
        if user_text and user_text.strip() != cls._EMPTY_TEXT_PLACEHOLDER:
            return f"{prefix}\n\n{user_text}"
        return prefix

    @staticmethod
    def _untranscribed_audio_note(path: str) -> str:
        """One minimal neutral marker for every STT failure. Never mention "no STT provider" or setup
        steps — persisted in history they make the model keep volunteering STT-setup advice."""
        from tools.credential_files import to_agent_visible_cache_path
        agent_path = to_agent_visible_cache_path(os.path.abspath(path))
        return f"[voice message could not be transcribed automatically; the audio is available at: {agent_path}]"

    async def _transcribe_one_clip(self, path: str, transcribe_audio, transcribe_audio_local_fallback) -> tuple[Optional[str], str]:
        """``(transcript_or_None, note)`` for one clip via configured STT with local fallback."""
        result = await asyncio.to_thread(transcribe_audio, path, None, "gateway")
        if not result.get("success"):
            fallback = await asyncio.to_thread(transcribe_audio_local_fallback, path)
            if fallback.get("success"):
                logger.info("Configured STT failed for %s; recovered with local STT", path)
                result = fallback
        if not result["success"]:
            logger.info("Voice transcription failed for %s: %s", path, result.get("error", "unknown error"))
            return None, self._untranscribed_audio_note(path)
        transcript = result["transcript"]
        # STT may return success=True with an empty/whitespace transcript (silence, cut-off);
        # empty quotes make the agent reply to nothing and can loop, so emit a sentinel note.
        # See #41603.
        if not (transcript or "").strip():
            return None, (
                "[The user sent a voice message but it came through "
                "empty or inaudible — speech-to-text returned no "
                "words. Do not guess at the content; ask the user "
                "to resend or type it out.]"
            )
        # Plain quoted line: a "The user sent a voice message..." wrapper read as a meta-instruction
        # and made the LLM comment on voice mode instead.
        return transcript, f'"{transcript}"'

    async def _enrich_message_with_transcription(
        self, user_text: str, audio_paths: list[str], *, event: MessageEvent | None = None,
    ) -> tuple[str, list[str]]:
        result = await self._transcribe_voice_clips(user_text, audio_paths)
        if event is not None:
            setattr(event, "_gateway_pending_stt_input", result)
        return result.text, result.transcripts()

    async def _transcribe_voice_clips(
        self, user_text: str, audio_paths: list[str]
    ) -> VoiceTranscription:
        from gateway.run import _probe_audio_duration
        audio_paths = list(dict.fromkeys(audio_paths))
        if not getattr(self.config, "stt_enabled", True):
            notes = []
            for path in audio_paths:
                abs_path = os.path.abspath(path)
                duration_str = await _probe_audio_duration(abs_path)
                suffix = f" (duration: {duration_str})" if duration_str else ""
                notes.append(f"[The user sent a voice message: {abs_path}{suffix}]")
            return VoiceTranscription(
                self._prepend_media_prefix("\n\n".join(notes), user_text) if notes else user_text,
                parts=tuple(VoiceTranscriptPart(note) for note in notes),
            )

        try:
            from tools.transcription_tools import (
                transcribe_audio, transcribe_audio_local_fallback
            )
        except ModuleNotFoundError as e:
            logger.error("Transcription module unavailable: %s", e)
            note = "[voice message could not be transcribed]"
            return VoiceTranscription(self._prepend_media_prefix(note, user_text), parts=(VoiceTranscriptPart(note),))

        enriched_parts = []
        clips: list[VoiceClipTranscript] = []
        parts: list[VoiceTranscriptPart] = []
        for path in audio_paths:
            try:
                logger.debug("Transcribing user voice: %s", path)
                transcript, note = await self._transcribe_one_clip(
                    path, transcribe_audio, transcribe_audio_local_fallback,
                )
                clip = VoiceClipTranscript(path, transcript) if transcript is not None else None
                if clip is not None:
                    clips.append(clip)
                parts.append(VoiceTranscriptPart(note, clip))
                enriched_parts.append(note)
            except Exception as e:
                logger.error("Transcription error: %s", e)
                note = self._untranscribed_audio_note(path)
                parts.append(VoiceTranscriptPart(note))
                enriched_parts.append(note)

        text = self._prepend_media_prefix("\n\n".join(enriched_parts), user_text) if enriched_parts else user_text
        return VoiceTranscription(text, tuple(clips), tuple(parts))

    def _pending_event_audio_paths(self, event) -> list[str]:
        """Return STT-eligible paths from a pending voice message."""
        from gateway.run import _event_media_is_stt_input
        return [
            path for i, path in enumerate(getattr(event, "media_urls", None) or [])
            if _event_media_is_stt_input(event, i)
        ]

    async def _transcribe_pending_audio_event_once(
        self, event, user_text: Optional[str] = None
    ) -> tuple[str | None, list[str]]:
        adapter = self._delivery_adapter_for(event.source)
        key = self._session_key_for_source(event.source)
        if pending_dispatch_withdrawn(adapter, key, event):
            return None, []
        if hasattr(event, "_gateway_pending_stt_text"):
            return event._gateway_pending_stt_text, list(getattr(event, "_gateway_pending_stt_transcripts", []) or [])
        while True:
            if pending_dispatch_withdrawn(adapter, key, event):
                return None, []
            revision = pending_dispatch_revision(adapter, key, event)
            audio_paths = self._pending_event_audio_paths(event)
            if not audio_paths:
                return user_text if user_text is not None else (getattr(event, "text", None) or None), []
            text = user_text if user_text is not None else (getattr(event, "text", "") or "")
            result = await self._transcribe_voice_clips(text, audio_paths)
            if pending_dispatch_withdrawn(adapter, key, event):
                return None, []
            if pending_dispatch_revision(adapter, key, event) == revision:
                break
            user_text = event.text
        event._gateway_pending_stt_text = result.text
        event._gateway_pending_stt_transcripts = result.transcripts()
        event._gateway_pending_stt_clips = result.clips
        setattr(event, "_gateway_pending_stt_input", result)
        return result.text, result.transcripts()

    async def _echo_pending_stt_transcripts_once(
        self, event, adapter, source, transcripts: list[str], *, metadata=None,
        log_context: str = "Transcript",
    ) -> None:
        if not transcripts or not self._should_echo_stt_transcripts() or adapter is None:
            return
        echoed = set(getattr(event, "_gateway_pending_stt_echoed_paths", ()))
        clips = getattr(event, "_gateway_pending_stt_clips", ())
        key = self._session_key_for_source(event.source)
        for clip in clips:
            if pending_dispatch_withdrawn(adapter, key, event):
                return
            if clip.path in echoed or clip.path not in event.media_urls:
                continue
            echoed.add(clip.path)
            event._gateway_pending_stt_echoed_paths = set(echoed)
            await self._echo_stt_transcripts(
                adapter, source, [clip.text], metadata=metadata, log_context=log_context,
            )

    async def _transcribe_and_echo_pending_voice(
        self, event, adapter, source, text: str, *, log_context: str, metadata=_UNSET
    ) -> tuple[str, list[str]]:
        """Transcribe a pending voice event and echo transcripts once → ``(enriched_text,
        transcripts)`` for ``agent.interrupt()`` or the pending-drain flow; ``(text, [])`` when there
        is no STT-eligible media (caller owns the ``_build_media_placeholder`` fallback)."""
        if not self._pending_event_audio_paths(event):
            return text, []
        try:
            key = self._session_key_for_source(event.source)
            while True:
                if pending_dispatch_withdrawn(adapter, key, event):
                    return "", []
                revision = pending_dispatch_revision(adapter, key, event)
                enriched_text, transcripts = await self._transcribe_pending_audio_event_once(event, text)
                echo_metadata = (self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))
                                 if metadata is _UNSET else metadata)
                await self._echo_pending_stt_transcripts_once(
                    event, adapter, source, transcripts, metadata=echo_metadata, log_context=log_context,
                )
                if pending_dispatch_withdrawn(adapter, key, event):
                    return "", []
                if pending_dispatch_revision(adapter, key, event) == revision:
                    return enriched_text or text, transcripts
                text = event.text or ""
        except Exception as trans_exc:
            logger.warning("%s transcription failed: %s", log_context, trans_exc)
            return text, []
