"""run_inbound_enrich — the _enrich* methods split out of GatewayInboundMixin (mechanical extraction)."""

from __future__ import annotations
import logging
import asyncio
import json
import os
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from typing import List
logger = logging.getLogger("gateway.run")

class GatewayInboundEnrichMixin:
    async def _enrich_inbound_images(
        self, source: SessionSource, session_key: str, message_text: str, image_paths: list[str]
    ) -> str:
        """Route images natively (attach pixels at run_conversation) or pre-analyze them into text."""
        # See agent/image_routing.py. Offloaded to a thread: the decision does blocking network I/O
        # (models.dev fetch on cache miss, Ollama /api/show probe) that would stall the event loop.
        _img_mode = await asyncio.to_thread(
            self._decide_image_input_mode, source=source, session_key=session_key,
        )
        if _img_mode == "native":
            self._session_state(session_key).persistent.native_image_paths = list(image_paths)
            logger.info(
                "Image routing: native (model supports vision). %d image(s) will be attached inline.",
                len(image_paths),
            )
            return message_text
        logger.info(
            "Image routing: text (mode=%s). Pre-analyzing %d image(s) via vision_analyze.",
            _img_mode, len(image_paths),
        )
        # Vision enrichment runs before AIAgent.run_conversation(), so bind this session's resolved
        # runtime explicitly rather than consulting process-global compatibility mirrors.
        vision_runtime = None
        try:
            turn_model, runtime_kwargs = self._resolve_session_agent_runtime(
                source=source, session_key=session_key,
            )
            vision_runtime = {**(runtime_kwargs or {}), "model": turn_model}
        except Exception:
            logger.debug("vision enrichment: session runtime resolution failed", exc_info=True)

        from agent.auxiliary_client import scoped_runtime_main

        with scoped_runtime_main(vision_runtime):
            return await self._enrich_message_with_vision(message_text, image_paths)

    async def _enrich_inbound_voice(
        self, event: MessageEvent, source: SessionSource, message_text: str, audio_paths: list[str]
    ) -> str:
        message_text, _successful_transcripts = await self._enrich_message_with_transcription(
            message_text, audio_paths,
        )
        # Echo each successful transcript back immediately when configured so users can verify STT
        # quality in real time. On transcription failure do NOT send a hardcoded notice: that
        # bypassed the LLM and produced two replies; enrichment leaves one neutral marker instead.
        if _successful_transcripts and self._should_echo_stt_transcripts():
            _echo_adapter = self._adapter_for_source(source)
            if _echo_adapter:
                _echo_meta = self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))
                await self._echo_stt_transcripts(_echo_adapter, source, _successful_transcripts, metadata=_echo_meta)
        return message_text

    async def _enrich_message_with_vision(self, user_text: str, image_paths: List[str]) -> str:
        """Auto-analyze user-attached images with the vision tool and prepend the descriptions.
        Description *and* local cache path are injected so the model understands the image without
        a tool call and can re-examine it with vision_analyze."""
        from tools.vision_tools import vision_analyze_tool
        from agent.memory_manager import sanitize_context

        analysis_prompt = (
            "Concisely describe this image in 2-4 sentences "
            "(~200 Chinese characters or ~150 English words). "
            "Cover the main subject, key visible text/data/code, and overall context. "
            "If it is a chart, diagram, or scientific figure, include the important "
            "labels, legend, and key values. Skip decorative details."
        )
        enriched_parts = []
        for path in image_paths:
            try:
                logger.debug("Auto-analyzing user image: %s", path)
                result = json.loads(await vision_analyze_tool(image_url=path, user_prompt=analysis_prompt))
                if result.get("success"):
                    description = sanitize_context(result.get("analysis", ""))
                    note = (
                        f"[The user sent an image~ Here's what I can see:\n{description}]\n"
                        f"[If you need a closer look, use vision_analyze with "
                        f"image_url: {path} ~]"
                    )
                else:
                    note = (
                        "[The user sent an image but I couldn't quite see it "
                        "this time (>_<) You can try looking at it yourself "
                        f"with vision_analyze using image_url: {path}]"
                    )
            except Exception as e:
                logger.error("Vision auto-analysis error: %s", e)
                note = (
                    f"[The user sent an image but something went wrong when I "
                    f"tried to look at it~ You can try examining it yourself "
                    f"with vision_analyze using image_url: {path}]"
                )
            enriched_parts.append(note)
        if not enriched_parts:
            return user_text
        prefix = "\n\n".join(enriched_parts)
        return f"{prefix}\n\n{user_text}" if user_text else prefix

    async def _enrich_message_with_transcription(
        self, user_text: str, audio_paths: List[str]
    ) -> tuple[str, List[str]]:
        """Transcribe voice clips with the configured STT provider and prepend the transcripts →
        ``(enriched_text, successful_transcripts)``; the transcripts (input order; empty if every clip
        failed or STT is disabled) let callers echo them back before the agent loop."""
        from gateway.run import _probe_audio_duration
        audio_paths = list(dict.fromkeys(audio_paths))
        if not getattr(self.config, "stt_enabled", True):
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
        successful_transcripts: List[str] = []
        for path in audio_paths:
            try:
                logger.debug("Transcribing user voice: %s", path)
                transcript, note = await self._transcribe_one_clip(
                    path, transcribe_audio, transcribe_audio_local_fallback,
                )
                if transcript is not None:
                    successful_transcripts.append(transcript)
                enriched_parts.append(note)
            except Exception as e:
                logger.error("Transcription error: %s", e)
                enriched_parts.append(self._untranscribed_audio_note(path))

        if enriched_parts:
            user_text = self._prepend_media_prefix("\n\n".join(enriched_parts), user_text)
        return user_text, successful_transcripts
