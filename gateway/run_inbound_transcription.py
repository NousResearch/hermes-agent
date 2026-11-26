"""Voice-clip transcription for inbound messages (moved out of ``gateway/run_inbound.py``)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
from collections import OrderedDict
from typing import Optional

from gateway.run_common import _UNSET

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


def _dedupe_audio_paths(audio_paths: list[str]) -> list[str]:
    """Order-preserving dedupe by path, then by content: one clip can reach a turn through two
    routes (Telegram's current voice + ``reply_to_message.voice``) under two cache paths, and
    each would be transcribed and billed. Only same-size files are hashed; an unreadable path
    keeps path identity so a failed probe never drops a distinct clip."""
    paths = list(dict.fromkeys(audio_paths))
    by_size: dict[int, list[str]] = {}
    for path in paths:
        try:
            by_size.setdefault(os.path.getsize(path), []).append(path)
        except OSError:
            pass
    duplicates: set[str] = set()
    for group in by_size.values():
        seen: set[bytes] = set()
        for path in group if len(group) > 1 else ():
            try:
                with open(path, "rb") as fh:
                    digest = hashlib.file_digest(fh, "sha256").digest()
            except OSError:
                continue
            if digest in seen:
                logger.info("Skipping duplicate audio attachment before STT: %s", path)
                duplicates.add(path)
            seen.add(digest)
    return [path for path in paths if path not in duplicates]


def stt_enabled_for_turn(runner) -> bool:
    """Whether the turn transcribes inbound voice: ``GatewayConfig.stt_enabled``, which also honours
    the legacy ``gateway.json`` ``stt_enabled`` key. The spoken-wake-word gate reads the same value."""
    return getattr(getattr(runner, "config", None), "stt_enabled", True)


# Transcript TEXT the adapters' spoken-wake-word gate already paid for, by profile and clip content:
# native admission re-captures the clip under a new path and keeps no adapter attributes, so the turn
# finds it by bytes and builds its note from its own path. Profiles never share an entry (their STT
# provider and language can differ for the same forwarded clip).
_GATE_TRANSCRIPTS: OrderedDict[tuple[str, bytes], str] = OrderedDict()
_GATE_TRANSCRIPTS_MAX = 32
_GATE_TRANSCRIPTS_LOCK = threading.Lock()


def _clip_digest(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as fh:
            return hashlib.file_digest(fh, "sha256").digest()
    except OSError:
        return None


def _gate_key(path: str) -> Optional[tuple[str, bytes]]:
    from hermes_constants import hermes_home_key
    digest = _clip_digest(path)
    return None if digest is None else (hermes_home_key(), digest)


def remember_gate_transcript(path: str, transcript: str) -> None:
    if (key := _gate_key(path)) is None:
        return
    with _GATE_TRANSCRIPTS_LOCK:
        _GATE_TRANSCRIPTS[key] = transcript
        while len(_GATE_TRANSCRIPTS) > _GATE_TRANSCRIPTS_MAX:
            _GATE_TRANSCRIPTS.popitem(last=False)


def _take_gate_transcript(path: str) -> Optional[str]:
    if not _GATE_TRANSCRIPTS or (key := _gate_key(path)) is None:
        return None
    with _GATE_TRANSCRIPTS_LOCK:
        return _GATE_TRANSCRIPTS.pop(key, None)


# Clips a busy follow-up already transcribed before native admission, by the MESSAGE that carried
# them (profile, platform, chat, message id) and clip content → (transcript, whether the chat was
# shown its echo). The FIFO turn restores a fresh event (a re-captured clip, no
# ``_gateway_pending_stt_*``) but keeps the chat and message id, so it reuses that transcript and
# skips a shown echo (#152457). The same clip forwarded into another chat is another message: it is
# transcribed and echoed there as usual and never consumes this one's record.
_HANDED_OFF: OrderedDict[tuple, tuple[str, bool]] = OrderedDict()


def _echo_key(event, path: str) -> Optional[tuple]:
    source = getattr(event, "source", None)
    message_id = getattr(event, "message_id", None)
    if source is None or not message_id or (key := _gate_key(path)) is None:
        return None  # no message identity: never hand a transcript or an echo to another turn
    platform = getattr(getattr(source, "platform", None), "value", None)
    return (key[0], platform, str(source.chat_id), str(message_id), key[1])


def hand_prepared_voice_to_turn(event) -> None:
    """Hand a busy follow-up's per-clip transcripts (``_gateway_pending_stt_clips``) to the turn
    native admission restores for the same message, noting whether the chat was shown each."""
    clip_memo = getattr(event, "_gateway_pending_stt_clips", None) or {}
    done = [(path, transcript) for path, (transcript, _note) in clip_memo.items() if transcript is not None]
    # Transcript indices whose busy echo the platform accepted (``done`` follows the transcripts' order).
    shown = getattr(event, "_gateway_pending_stt_shown", None) or ()
    records = [(key, (transcript, index in shown)) for index, (path, transcript) in enumerate(done)
               if (key := _echo_key(event, path)) is not None]
    with _GATE_TRANSCRIPTS_LOCK:
        for key, record in records:
            _HANDED_OFF[key] = record
            while len(_HANDED_OFF) > _GATE_TRANSCRIPTS_MAX:
                _HANDED_OFF.popitem(last=False)


def take_handed_off_clips(event, paths: list[str]) -> dict[str, tuple[str, bool]]:
    """The *paths* of *event* its busy pass already transcribed → ``(transcript, echo shown)``
    (consumed: one turn reuses them)."""
    if not _HANDED_OFF:
        return {}
    keys = {path: _echo_key(event, path) for path in paths}
    found = {}
    with _GATE_TRANSCRIPTS_LOCK:
        for path, key in keys.items():
            if key is not None and key in _HANDED_OFF:
                found[path] = _HANDED_OFF[key]
        for path in found:
            _HANDED_OFF.pop(keys[path], None)
    return found


def transcript_note(transcript: str, path: str) -> str:
    """The turn's note for a transcribed clip at *path* (the clip the turn holds, never the gate's).
    A plain quoted line: a "The user sent a voice message..." wrapper read as a meta-instruction and
    made the LLM comment on voice mode instead. The path follows on its own line, as every failure
    branch keeps it: ``content`` is the only record of where the audio lives, in
    ``_build_media_placeholder``'s marker grammar so a consumer parsing media markers finds it."""
    return f'"{transcript}"\n[User sent audio: {GatewayInboundTranscriptionMixin._agent_audio_path(path)}]'


class GatewayInboundTranscriptionMixin:
    """STT enrichment and the once-only echo of pending voice transcripts."""

    async def _enrich_message_with_transcription(
        self, user_text: str, audio_paths: list[str], clip_memo: Optional[dict] = None
    ) -> tuple[str, list[str]]:
        """Transcribe voice clips with the configured STT provider and prepend the transcripts →
        ``(enriched_text, successful_transcripts)``; the transcripts (input order; empty if every clip
        failed or STT is disabled) let callers echo them back before the agent loop. ``clip_memo``
        (path → ``_transcribe_one_clip`` result) skips clips an earlier pass already transcribed."""
        from gateway.run import _probe_audio_duration
        audio_paths = await asyncio.to_thread(_dedupe_audio_paths, audio_paths)
        if not stt_enabled_for_turn(self):
            notes = []
            for path in audio_paths:
                abs_path = os.path.abspath(path)
                duration_str = await _probe_audio_duration(abs_path)
                suffix = f" (duration: {duration_str})" if duration_str else ""
                notes.append(f"[The user sent a voice message: {abs_path}{suffix}]")
            return (self._prepend_media_prefix("\n\n".join(notes), user_text) if notes else user_text), []

        try:
            from tools.transcription_tools import (
                transcribe_audio, transcribe_audio_local_fallback
            )
        except ModuleNotFoundError as e:
            logger.error("Transcription module unavailable: %s", e)
            return self._prepend_media_prefix("[voice message could not be transcribed]", user_text), []

        enriched_parts = []
        successful_transcripts: list[str] = []
        for path in audio_paths:
            try:
                if clip_memo is not None and path in clip_memo:
                    transcript, note = clip_memo[path]
                elif transcript := await asyncio.to_thread(_take_gate_transcript, path):
                    note = transcript_note(transcript, path)
                    if clip_memo is not None:
                        clip_memo[path] = (transcript, note)
                else:
                    logger.info("Transcribing user voice: %s", path)
                    transcript, note = await self._transcribe_one_clip(
                        path, transcribe_audio, transcribe_audio_local_fallback,
                    )
                    if clip_memo is not None:
                        clip_memo[path] = (transcript, note)
                if transcript is not None:
                    successful_transcripts.append(transcript)
                enriched_parts.append(note)
            except Exception as e:
                logger.error("Transcription error: %s", e)
                enriched_parts.append(self._untranscribed_audio_note(path))

        if enriched_parts:
            user_text = self._prepend_media_prefix("\n\n".join(enriched_parts), user_text)
        return user_text, successful_transcripts

    async def _enrich_inbound_voice(
        self, event, source, message_text: str, audio_paths: list[str]
    ) -> str:
        handed = await asyncio.to_thread(take_handed_off_clips, event, audio_paths)  # paid when it arrived busy
        shown = {path for path, (_transcript, was_shown) in handed.items() if was_shown}
        clips: dict = {path: (transcript, transcript_note(transcript, path)) for path, (transcript, _s) in handed.items()}
        message_text, _ = await self._enrich_message_with_transcription(message_text, audio_paths, clips)
        # ``clips`` holds each clip once, in order (enrichment dedupes repeated paths and bytes).
        _successful_transcripts = [transcript for path, (transcript, _note) in clips.items()
                                   if transcript is not None and path not in shown]
        # Echo each successful transcript back immediately when configured so users can verify STT
        # quality in real time. On transcription failure do NOT send a hardcoded notice: that
        # bypassed the LLM and produced two replies; enrichment leaves one neutral marker instead.
        if _successful_transcripts and self._should_echo_stt_transcripts():
            _echo_adapter = self._delivery_adapter_for(source)
            if _echo_adapter:
                _echo_meta = self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))
                await self._echo_stt_transcripts(_echo_adapter, source, _successful_transcripts, metadata=_echo_meta)
        return message_text

    @staticmethod
    def _busy_interrupt_text(event) -> Optional[str]:
        """What ``agent.interrupt()`` carries for a busy follow-up: its caption, else the media
        placeholder. Never the transcript: awaiting STT first held the running turn and the busy
        ack for the clip's whole transcription (#143961). The queued event keeps its audio, so the
        follow-up turn transcribes (and echoes) it once."""
        from gateway.run import _build_media_placeholder
        if event.text or not getattr(event, "media_urls", None):
            return event.text
        return _build_media_placeholder(event)

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
        """Transcribe a pending audio event once and cache the result on the event: the interrupt
        monitor and the pending-drain path both need it — one STT call and one echo per message.
        ``user_text`` is a deliberate override of the caption; ``None`` reads ``event.text`` live."""
        if hasattr(event, "_gateway_pending_stt_text"):
            return event._gateway_pending_stt_text, list(getattr(event, "_gateway_pending_stt_transcripts", []) or [])
        audio_paths = self._pending_event_audio_paths(event)
        if not audio_paths:
            return user_text if user_text is not None else (getattr(event, "text", None) or None), []
        text = user_text if user_text is not None else (getattr(event, "text", "") or "")
        # Per-clip results outlive the merged-event cache below (a merge drops only that), so every
        # re-run — this loop's or the drain path's — transcribes just the newly merged clips.
        clip_memo = getattr(event, "_gateway_pending_stt_clips", None)
        if clip_memo is None:  # a restored turn that lands on a busy session again reuses its hand-off
            handed = await asyncio.to_thread(take_handed_off_clips, event, audio_paths)
            clip_memo = {path: (tx, transcript_note(tx, path)) for path, (tx, _shown) in handed.items()}
            event._gateway_pending_stt_clips = clip_memo
        enriched_text, successful_transcripts = await self._enrich_message_with_transcription(text, audio_paths, clip_memo)
        # A voice note merged in during the await dropped a cache that did not exist yet; re-run on
        # the merged media rather than publish a transcript that silently omits the new note.
        while self._pending_event_audio_paths(event) != audio_paths:
            audio_paths = self._pending_event_audio_paths(event)
            text = user_text if user_text is not None else (getattr(event, "text", "") or "")
            enriched_text, successful_transcripts = await self._enrich_message_with_transcription(
                text, audio_paths, clip_memo,
            )
        event._gateway_pending_stt_text = enriched_text
        event._gateway_pending_stt_transcripts = list(successful_transcripts)
        return enriched_text, successful_transcripts

    async def _echo_pending_stt_transcripts_once(
        self, event, adapter, source, transcripts: list[str], *, metadata=None,
        log_context: str = "Transcript",
    ) -> None:
        """Echo pending-event STT transcripts to the chat at most once. Tracked as a COUNT (not a
        set — identical transcripts are distinct deliveries): ``merge_pending_message_event`` can
        append a second voice note and invalidate the cache; the re-run returns earlier transcripts
        as a prefix, so only the unsent tail is echoed."""
        if not transcripts or not self._should_echo_stt_transcripts() or adapter is None:
            return
        already_echoed = int(getattr(event, "_gateway_pending_stt_echoed", 0) or 0)
        tail = transcripts[already_echoed:]
        event._gateway_pending_stt_echoed = max(already_echoed, len(transcripts))  # claimed before the await
        # A refused echo (flood control, dead transport) was never shown: record each accepted one, so
        # the turn native admission restores echoes exactly the clips still owed. Every clip is tried:
        # a steered message has no later turn to make up a skipped echo.
        shown = event._gateway_pending_stt_shown = set(getattr(event, "_gateway_pending_stt_shown", None) or ())
        for index, transcript in enumerate(tail, start=already_echoed):
            if await self._echo_stt_transcripts(
                adapter, source, [transcript], metadata=metadata, log_context=log_context,
            ):
                shown.add(index)

    async def _transcribe_and_echo_pending_voice(
        self, event, adapter, source, text: str, *, log_context: str, metadata=_UNSET
    ) -> tuple[str, list[str]]:
        """Transcribe a pending voice event and echo transcripts once → ``(enriched_text,
        transcripts)`` for ``agent.interrupt()`` or the pending-drain flow; ``(text, [])`` when there
        is no STT-eligible media (caller owns the ``_build_media_placeholder`` fallback). ``text`` is
        only that fallback: the caption is read live from ``event.text`` so one merged in during
        the STT await is kept."""
        if not self._pending_event_audio_paths(event):
            return text, []
        try:
            enriched_text, transcripts = await self._transcribe_pending_audio_event_once(event)
            if metadata is _UNSET:
                metadata = self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))
            await self._echo_pending_stt_transcripts_once(
                event, adapter, source, transcripts, metadata=metadata, log_context=log_context
            )
            return enriched_text or text, transcripts
        except Exception as trans_exc:
            logger.warning("%s transcription failed: %s", log_context, trans_exc)
            return text, []

    _EMPTY_TEXT_PLACEHOLDER = "(The user sent a message with no text content)"

    @classmethod
    def _prepend_media_prefix(cls, prefix: str, user_text: str) -> str:
        """``prefix`` + the user's text; the Discord empty-content placeholder is dropped as redundant."""
        if user_text and user_text.strip() != cls._EMPTY_TEXT_PLACEHOLDER:
            return f"{prefix}\n\n{user_text}"
        return prefix

    @staticmethod
    def _agent_audio_path(path: str) -> str:
        from tools.credential_files import to_agent_visible_cache_path
        return to_agent_visible_cache_path(os.path.abspath(path))

    @classmethod
    def _untranscribed_audio_note(cls, path: str) -> str:
        """One minimal neutral marker for every STT failure. Never mention "no STT provider" or setup
        steps — persisted in history they make the model keep volunteering STT-setup advice."""
        return f"[voice message could not be transcribed automatically; the audio is available at: {cls._agent_audio_path(path)}]"

    async def _transcribe_one_clip(self, path: str, transcribe_audio, transcribe_audio_local_fallback) -> tuple[Optional[str], str]:
        return await transcribe_one_clip(path, transcribe_audio, transcribe_audio_local_fallback)


async def transcribe_one_clip(path: str, transcribe_audio=None, transcribe_audio_local_fallback=None) -> tuple[Optional[str], str]:
    """``(transcript_or_None, note)`` for one clip via configured STT with local fallback. Also the
    adapters' group mention gate (``gateway.platforms.helpers.spoken_mention_clips``)."""
    if transcribe_audio is None:
        from tools.transcription_tools import transcribe_audio, transcribe_audio_local_fallback
    result = await asyncio.to_thread(transcribe_audio, path, None, "gateway")
    if not result.get("success"):
        fallback = await asyncio.to_thread(transcribe_audio_local_fallback, path)
        if fallback.get("success"):
            logger.info("Configured STT failed for %s; recovered with local STT", path)
            result = fallback
    if not result["success"]:
        logger.info("Voice transcription failed for %s: %s", path, result.get("error", "unknown error"))
        return None, GatewayInboundTranscriptionMixin._untranscribed_audio_note(path)
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
    return transcript, transcript_note(transcript, path)
