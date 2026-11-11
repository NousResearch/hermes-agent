"""Best-effort attachment speech owned independently of the conversation guard."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
AUDIO_DEADLINE_SECONDS = 120.0
NOTICE_TIMEOUT_SECONDS = 5.0
DEADLINE_NOTICE = "Audio missed its deadline. The full text reply is above."
FAILURE_NOTICE = "Audio could not be delivered. The full text reply is above."


MAX_PENDING_AUDIO = 2


def _safe_exc_info(exc):
    # Keep stack locations without logging arbitrary provider text, secrets or chains.
    safe = RuntimeError("Auto-TTS error details omitted")
    return RuntimeError, safe, exc.__traceback__


def _existing_tts_paths(tts_data):
    paths = tts_data.get("file_paths") or [tts_data.get("file_path")]
    return [str(path) for path in paths if path and Path(path).exists()]


def _profile_scope(adapter, source) -> contextlib.AbstractAsyncContextManager:
    """Re-enter the serving profile's secret scope after the text handler returns."""
    resolve = getattr(getattr(adapter, "gateway_runner", None), "_async_profile_scope_for_source", None)
    if not callable(resolve) or source is None:
        return contextlib.nullcontext()
    return resolve(source)


class AutoTTSMixin:
    def background_auto_tts(self, chat_id: str) -> bool:
        """Whether attachment speech may outlive the text turn (live playback must not)."""
        return False

    async def _synthesize_auto_tts(self, text_content: str, source=None, *, output_dir: Optional[str] = None) -> tuple[list[str], Optional[str]]:
        """Synthesize auto-TTS audio -> ``(existing_paths, requested_path)``; empty/None on failure
        (logged, never raised). Path built platform-aware HERE: HERMES_SESSION_PLATFORM is cleared
        post-handler."""
        from gateway.platforms.base import build_auto_tts_output_path

        paths: list[str] = []
        requested_path = None
        try:
            from tools.tts_tool import text_to_speech_tool, check_tts_requirements
            async with _profile_scope(self, source):
                if output_dir is not None or check_tts_requirements():
                    import json as _json
                    speech_text = self.prepare_tts_text(text_content)
                    if not speech_text:
                        raise ValueError("Empty text after markdown cleanup")
                    requested_path = build_auto_tts_output_path(self.platform)
                    if output_dir is not None:
                        from gateway.platforms.base_auto_tts_worker import synthesize
                        requested_path = str(Path(output_dir) / Path(requested_path).name)
                        tts_data = await synthesize(speech_text, requested_path)
                    else:
                        tts_data = _json.loads(await asyncio.to_thread(
                            text_to_speech_tool, text=speech_text, output_path=requested_path))
                    if tts_data.get("success", True):
                        paths = await asyncio.to_thread(_existing_tts_paths, tts_data)
        except Exception as tts_err:
            logger.warning("[%s] Auto-TTS failed (%s)", self.name, type(tts_err).__name__,
                           exc_info=_safe_exc_info(tts_err))
        return paths, requested_path

    def _wants_auto_tts(self, event: MessageEvent, session_key: str, interrupt_event: asyncio.Event,
                        text_content: str, media_files: list) -> bool:
        """Auto-TTS on voice input (voice-first), gated by /voice or voice.auto_tts;
        skipped when streaming TTS already delivered audio this turn."""
        generation = getattr(interrupt_event, "_hermes_run_generation", None)
        return bool(
            self._should_auto_tts_for_chat(event.source.chat_id)
            and event.message_type == MessageType.VOICE and text_content and not media_files
            and not self._streaming_tts_turn_completed(session_key, generation, event=event))

    async def _play_tts_file(
        self, event: MessageEvent, text_content: str, tts_path: str, first: bool,
        metadata: dict[str, Any], record_delivery: Callable) -> bool:
        """Play one synthesized TTS file. Returns True when the ORIGINAL reply text rode
        along as a Telegram caption (first file, ≤1024 chars) so the text send is skipped."""
        caption = None
        if first and self.platform == Platform.TELEGRAM and text_content and text_content[:1024] == text_content:
            caption = text_content
        tts_result = await self.play_tts(
            chat_id=event.source.chat_id, audio_path=tts_path, caption=caption, metadata=metadata)
        record_delivery(tts_result)
        return bool(caption and getattr(tts_result, "success", False))


async def schedule_auto_tts(adapter, event, session_key, text, reply_to, metadata):
    # No waiting synthesis queue: queued long replies could otherwise consume
    # unbounded memory/processes while the conversation continues normally.
    pending = sum(not task.done() and task.get_name().startswith("auto-tts:")
                  for task in adapter._background_tasks)
    if pending >= MAX_PENDING_AUDIO:
        task = asyncio.create_task(
            _notice(adapter, event, reply_to, metadata,
                    "Audio skipped because other audio jobs are still pending."),
            name=f"auto-tts-notice:{session_key}")
    else:
        task = asyncio.create_task(
            _deliver(adapter, event, text, reply_to, metadata), name=f"auto-tts:{session_key}")
    adapter._background_tasks.add(task)
    task.add_done_callback(adapter._background_tasks.discard)
    runner_tasks = getattr(adapter.gateway_runner, "_background_tasks", None)
    if isinstance(runner_tasks, set):
        runner_tasks.add(task)
        task.add_done_callback(runner_tasks.discard)


async def cancel_auto_tts(tasks, session_key):
    owned = [task for task in list(tasks) if not task.done()
             and task.get_name() in {f"auto-tts:{session_key}", f"auto-tts-notice:{session_key}"}]
    for task in owned:
        task.cancel()
    if owned:
        await asyncio.gather(*owned, return_exceptions=True)
    return bool(owned)


async def _notice(adapter, event, reply_to, metadata, notice):
    # One attempt only: retrying an ambiguous transport ACK risks duplicate notices.
    try:
        with adapter._media_delivery_scope(event.source):
            await asyncio.wait_for(adapter.send(
                event.source.chat_id, notice, reply_to=reply_to, metadata=metadata),
                NOTICE_TIMEOUT_SECONDS)
    except Exception as exc:
        logger.warning("Could not deliver auto-TTS notice (%s)", type(exc).__name__,
                       exc_info=_safe_exc_info(exc))


async def _deliver(adapter, event, text, reply_to, metadata):
    paths, requested = [], None
    with adapter._media_delivery_scope(event.source):
        try:
            cache = get_hermes_home() / "cache" / "auto_tts"
            cache.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="job-", dir=cache) as directory:
                async with asyncio.timeout(AUDIO_DEADLINE_SECONDS):
                    paths, requested = await adapter._synthesize_auto_tts(text, output_dir=directory)
                    if not paths:
                        raise RuntimeError("Synthesis failed")
                    for path in paths:
                        result = await adapter.send_voice(
                            chat_id=event.source.chat_id, audio_path=path,
                            reply_to=reply_to, metadata=metadata, notify_failure=False)
                        if not result.success:
                            raise RuntimeError("Audio delivery failed")
        except Exception as exc:
            logger.warning("Auto-TTS attachment failed (%s)", type(exc).__name__,
                           exc_info=_safe_exc_info(exc))
            notice = DEADLINE_NOTICE if isinstance(exc, TimeoutError) else FAILURE_NOTICE
            await _notice(adapter, event, reply_to, metadata, notice)
        finally:
            for path in {requested, *paths} - {None}:
                with contextlib.suppress(OSError):
                    os.remove(path)
