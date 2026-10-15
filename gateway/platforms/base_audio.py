"""Gateway audio-output path generation."""
import os
import tempfile
import uuid

def build_auto_tts_output_path(platform) -> str:
    """Unique temp output path for gateway auto-TTS: ``.ogg`` for ``OPUS_VOICE_PLATFORMS``
    (the tool's ``_repair_ogg_container`` then guarantees real Opus bytes), else ``.mp3``.
    Platform-awareness lives HERE because ``_clear_session_env`` wipes the TTS tool's
    ``HERMES_SESSION_PLATFORM`` contextvar before the post-handler auto-TTS block runs.

    Platforms whose native voice bubbles require Ogg/Opus (``tools.tts_tool.OPUS_VOICE_PLATFORMS`` — the
    single source of truth) get an explicit ``.ogg`` path; the tool's central container repair
    (``_repair_ogg_container``) then guarantees real Ogg/Opus bytes for every provider, including MP3-only
    backends like Edge TTS. Everything else keeps the MP3 default. See #36685, #57049.
    """
    from tools.tts_tool import OPUS_VOICE_PLATFORMS
    from gateway.platforms.base import _platform_name
    ext = "ogg" if _platform_name(platform) in OPUS_VOICE_PLATFORMS else "mp3"
    audio_path = os.path.join(
        tempfile.gettempdir(), "hermes_voice", f"tts_reply_{uuid.uuid4().hex[:12]}.{ext}")
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    return audio_path
