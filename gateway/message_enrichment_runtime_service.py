"""Vision and STT message enrichment for gateway turns.

Promoted from ``GatewayRunner._enrich_message_with_vision`` and
``_enrich_message_with_transcription``. Runner methods remain thin delegates.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger_default = logging.getLogger(__name__)


async def _probe_audio_duration(path: str) -> Optional[str]:
    from gateway.run import _probe_audio_duration as _fn
    return await _fn(path)


async def enrich_message_with_vision(
    *,
    runner: Any,
    user_text: str,
    image_paths: List[str],
    source: Any = None,
    logger: Any = None,
) -> str:
    """Opportunistically analyze user-attached images (production semantics)."""
    if logger is None:
        logger = logger_default

    runner._ensure_auto_vision_state()
    analysis_prompt = (
        "Describe everything visible in this image in thorough detail. "
        "Include any text, code, data, objects, people, layout, colors, "
        "and any other notable visual information."
    )
    analysis_timeout = runner._auto_vision_analysis_timeout_seconds()
    inline_wait = runner._auto_vision_inline_wait_seconds(
        source,
        has_user_text=bool(str(user_text or "").strip()),
    )
    inline_deadline = time.monotonic() + inline_wait if inline_wait > 0 else None

    enriched_parts = []
    pending_tasks: List[tuple[str, str, asyncio.Task]] = []
    for path in image_paths:
        cache_key = runner._auto_vision_cache_key(path)
        cached = runner._get_auto_vision_cache_entry(cache_key)
        if cached:
            if cached.get("status") == "success":
                description = str(cached.get("analysis") or "").strip()
                if description:
                    enriched_parts.append(
                        f"[The user sent an image~ Here's what I can see:\n{description}]"
                    )
                    continue
            elif cached.get("status") == "error":
                enriched_parts.append(runner._auto_vision_degraded_note(path, pending=False))
                continue

        remaining, reason = runner._auto_vision_cooldown_remaining()
        if remaining > 0:
            logger.debug(
                "Skipping vision auto-analysis for %.1fs after %s (image=%s)",
                remaining,
                reason or "recent_failure",
                path,
            )
            enriched_parts.append(runner._auto_vision_degraded_note(path, pending=False))
            continue

        task = runner._start_auto_vision_task(
            cache_key=cache_key,
            path=path,
            analysis_prompt=analysis_prompt,
            analysis_timeout=analysis_timeout,
        )
        if task is None:
            enriched_parts.append(runner._auto_vision_degraded_note(path, pending=False))
            continue
        pending_tasks.append((path, cache_key, task))

    completed_tasks: set[asyncio.Task] = set()
    if pending_tasks:
        if inline_deadline is not None:
            remaining_inline = max(0.0, inline_deadline - time.monotonic())
        else:
            remaining_inline = 0.0
        if remaining_inline > 0:
            done, _pending = await asyncio.wait(
                [task for _, _, task in pending_tasks],
                timeout=remaining_inline,
            )
            completed_tasks = set(done)

    for path, cache_key, task in pending_tasks:
        task_entry: Optional[Dict[str, Any]] = None
        if task in completed_tasks or task.done():
            try:
                task_entry = task.result()
            except Exception as exc:
                logger.debug("Auto vision background task failed for %s: %s", path, exc)
                task_entry = runner._get_auto_vision_cache_entry(cache_key)

        if task_entry and task_entry.get("status") == "success":
            description = str(task_entry.get("analysis") or "").strip()
            if description:
                enriched_parts.append(
                    f"[The user sent an image~ Here's what I can see:\n{description}]"
                )
                continue

        if task_entry and task_entry.get("status") == "error":
            enriched_parts.append(runner._auto_vision_degraded_note(path, pending=False))
            continue

        enriched_parts.append(runner._auto_vision_degraded_note(path, pending=True))

    # Combine: vision descriptions first, then the user's original text
    if enriched_parts:
        prefix = "\n\n".join(enriched_parts)
        if user_text:
            return f"{prefix}\n\n{user_text}"
        return prefix
    return user_text




async def enrich_message_with_transcription(
    *,
    runner: Any,
    user_text: str,
    audio_paths: List[str],
    logger: Any = None,
) -> Tuple[str, List[str]]:
    """Transcribe voice attachments and enrich user text (production semantics)."""
    if logger is None:
        logger = logger_default

    if not getattr(runner.config, "stt_enabled", True):
        notes = []
        for path in audio_paths:
            abs_path = os.path.abspath(path)
            duration_str = await _probe_audio_duration(abs_path)
            if duration_str:
                notes.append(
                    f"[The user sent a voice message: {abs_path} (duration: {duration_str})]"
                )
            else:
                notes.append(f"[The user sent a voice message: {abs_path}]")
        if not notes:
            return user_text, []
        prefix = "\n\n".join(notes)
        _placeholder = "(The user sent a message with no text content)"
        if user_text and user_text.strip() == _placeholder:
            return prefix, []
        if user_text:
            return f"{prefix}\n\n{user_text}", []
        return prefix, []

    from tools.transcription_tools import transcribe_audio

    enriched_parts = []
    successful_transcripts: List[str] = []
    for path in audio_paths:
        try:
            logger.debug("Transcribing user voice: %s", path)
            result = await asyncio.to_thread(transcribe_audio, path)
            if result["success"]:
                transcript = result["transcript"]
                successful_transcripts.append(transcript)
                # Pass the transcript through as a plain quoted line. The
                # earlier wording ("The user sent a voice message~ Here's
                # what they said: ...") read as a meta-instruction and made
                # the LLM volunteer commentary about voice mode rather than
                # reply to the content.
                enriched_parts.append(f'"{transcript}"')
            else:
                error = result.get("error", "unknown error")
                # All failure branches: a single, minimal, neutral marker.
                # Do NOT mention "no STT provider configured", "setup
                # instructions", or the "hermes-agent-setup" skill, and do
                # NOT claim a direct message was sent — those phrases get
                # persisted in conversation history and poison every later
                # turn, so the model keeps volunteering STT-setup advice
                # even after transcription starts working. The cause is
                # logged for operator diagnosis but kept out of the
                # LLM-visible prompt.
                logger.info("Voice transcription failed for %s: %s", path, error)
                enriched_parts.append("[voice message could not be transcribed]")
        except Exception as e:
            logger.error("Transcription error: %s", e)
            enriched_parts.append("[voice message could not be transcribed]")

    if enriched_parts:
        prefix = "\n\n".join(enriched_parts)
        # Strip the empty-content placeholder from the Discord adapter
        # when we successfully transcribed the audio — it's redundant.
        _placeholder = "(The user sent a message with no text content)"
        if user_text and user_text.strip() == _placeholder:
            return prefix, successful_transcripts
        if user_text:
            return f"{prefix}\n\n{user_text}", successful_transcripts
        return prefix, successful_transcripts
    return user_text, successful_transcripts

