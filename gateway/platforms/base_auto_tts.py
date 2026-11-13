"""Post-handler auto-TTS synthesis for ``BasePlatformAdapter`` (voice reply to a voice note).

Runs after the message handler returned, so the routed profile's secret scope that wrapped the
turn is gone. Under a multiplexed gateway an unscoped provider read (``GEMINI_BASE_URL``, a plugin
provider's key in ``is_available``) raises ``UnscopedSecretError`` and the voice reply is lost, so
synthesis re-enters the scope of the profile that serves ``source`` (#145151).
"""

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _profile_scope(adapter, source) -> contextlib.AbstractAsyncContextManager:
    """The runner's secret-hydrating scope for ``source``; a no-op without a runner."""
    resolve = getattr(getattr(adapter, "gateway_runner", None), "_async_profile_scope_for_source", None)
    if not callable(resolve) or source is None:
        return contextlib.nullcontext()
    return resolve(source)


def _synthesize(text: str, output_path: str) -> list[str]:
    """Worker-thread half: run the TTS tool and keep the files it reports that exist."""
    from tools.tts_tool import text_to_speech_tool
    tts_data = json.loads(text_to_speech_tool(text=text, output_path=output_path))
    if not tts_data.get("success", True):
        return []
    raw_tts_paths = tts_data.get("file_paths") or [tts_data.get("file_path")]
    return [str(path) for path in raw_tts_paths if path and Path(path).exists()]


async def synthesize_auto_tts(adapter, text_content: str, source, output_path_for) -> tuple[list[str], Optional[str]]:
    """``(existing_paths, requested_path)``; empty/None on failure (logged, never raised).
    ``output_path_for(platform)`` builds the platform-aware temp path (``.ogg`` for Opus platforms)."""
    paths: list[str] = []
    requested_path = None
    try:
        from tools.tts_tool import check_tts_requirements
        async with _profile_scope(adapter, source):
            if check_tts_requirements():
                speech_text = adapter.prepare_tts_text(text_content)
                if not speech_text:
                    raise ValueError("Empty text after markdown cleanup")
                requested_path = output_path_for(adapter.platform)
                paths = await asyncio.to_thread(_synthesize, speech_text, requested_path)
    except Exception as tts_err:  # an auto voice reply never fails the text reply it accompanies
        logger.warning("[%s] Auto-TTS failed: %s", adapter.name, tts_err, exc_info=True)
    return paths, requested_path
