"""
Sound Alerts Plugin

Plays TTS audio notifications when:
1. An approval request is triggered (dangerous command)
2. A task completes

Uses Edge TTS for audio generation and system audio players for playback.
"""

import logging
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Default voice — can be overridden via HERMES_SOUND_ALERTS_VOICE env var
DEFAULT_VOICE = "zh-CN-XiaoyiNeural"
DEFAULT_APPROVAL_TEXT = "注意！有一个操作需要你的审批。"
DEFAULT_COMPLETE_TEXT = "任务已完成。"

# Lock to prevent overlapping audio playback
_play_lock = threading.Lock()


def _get_voice() -> str:
    """Get the TTS voice from config or environment."""
    return os.environ.get("HERMES_SOUND_ALERTS_VOICE", DEFAULT_VOICE)


def _get_approval_text() -> str:
    """Get the approval alert text."""
    return os.environ.get("HERMES_SOUND_ALERTS_APPROVAL_TEXT", DEFAULT_APPROVAL_TEXT)


def _get_complete_text() -> str:
    """Get the task complete alert text."""
    return os.environ.get("HERMES_SOUND_ALERTS_COMPLETE_TEXT", DEFAULT_COMPLETE_TEXT)


def _get_audio_player() -> Optional[List[str]]:
    """Detect the system audio player command."""
    system = platform.system()
    if system == "Darwin":
        if shutil.which("afplay"):
            return ["afplay"]
    elif system == "Linux":
        # Try common Linux audio players
        for player in ["ffplay", "paplay", "aplay", "mpv", "cvlc"]:
            if shutil.which(player):
                if player == "ffplay":
                    return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
                elif player == "mpv":
                    return ["mpv", "--no-video", "--really-quiet"]
                elif player == "cvlc":
                    return ["cvlc", "--play-and-exit", "--quiet"]
                return [player]
    elif system == "Windows":
        # PowerShell can play audio
        return ["powershell", "-c", "(New-Object Media.SoundPlayer '{}').PlaySync()"]
    return None


def _generate_tts(text: str, voice: str) -> Optional[str]:
    """Generate TTS audio using edge-tts. Returns path to audio file or None."""
    try:
        import edge_tts
        import asyncio

        output_path = os.path.join(
            tempfile.gettempdir(),
            f"hermes_sound_alert_{os.getpid()}_{threading.get_ident()}.mp3",
        )

        async def _generate():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)

        asyncio.run(_generate())

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
    except Exception as e:
        logger.debug("TTS generation failed: %s", e)
    return None


def _play_audio(file_path: str) -> None:
    """Play an audio file using the system audio player."""
    player_cmd = _get_audio_player()
    if player_cmd is None:
        logger.debug("No audio player found on this system")
        return

    try:
        if platform.system() == "Windows":
            # Windows: replace placeholder with actual path
            cmd = [part.format(file_path) if '{}' in part else part for part in player_cmd]
        else:
            cmd = player_cmd + [file_path]

        subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception as e:
        logger.debug("Audio playback failed: %s", e)
    finally:
        # Clean up temp file
        try:
            os.unlink(file_path)
        except OSError:
            pass


def _play_alert(text: str) -> None:
    """Generate TTS and play audio alert in a background thread."""

    def _worker():
        with _play_lock:
            voice = _get_voice()
            audio_path = _generate_tts(text, voice)
            if audio_path:
                _play_audio(audio_path)

    # Run in background so we don't block the hook caller
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def on_approval_request(**kwargs: Any) -> None:
    """Hook callback: play alert sound when approval is requested."""
    _play_alert(_get_approval_text())


def on_task_complete(**kwargs: Any) -> None:
    """Hook callback: play alert sound when a task completes."""
    _play_alert(_get_complete_text())


def register(ctx) -> None:
    """Plugin entry point — register hooks."""
    ctx.register_hook("on_approval_request", on_approval_request)
    ctx.register_hook("on_task_complete", on_task_complete)
    logger.info("sound_alerts plugin registered")
