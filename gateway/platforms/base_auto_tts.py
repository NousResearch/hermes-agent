"""Best-effort attachment speech owned independently of the conversation guard."""

import asyncio
import contextlib
import logging
import os
import tempfile

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
AUDIO_DEADLINE_SECONDS = 120.0
NOTICE_TIMEOUT_SECONDS = 5.0
DEADLINE_NOTICE = "Audio missed its deadline. The full text reply is above."
FAILURE_NOTICE = "Audio could not be delivered. The full text reply is above."


MAX_PENDING_AUDIO = 2


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
    except Exception:
        logger.warning("Could not deliver auto-TTS notice", exc_info=True)


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
            logger.warning("Auto-TTS attachment failed (%s)", type(exc).__name__)
            notice = DEADLINE_NOTICE if isinstance(exc, TimeoutError) else FAILURE_NOTICE
            await _notice(adapter, event, reply_to, metadata, notice)
        finally:
            for path in {requested, *paths} - {None}:
                with contextlib.suppress(OSError):
                    os.remove(path)
