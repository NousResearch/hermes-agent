"""kenseivoice — local CPU TTS provider for the mesh bots.

Per-persona voice is the configured ``tts.voice`` in the form
``"<engine>:<voice>"`` — e.g. ``kokoro:af_alloy`` or ``pocket:azelma``.
A bare voice id (no prefix) defaults to Kokoro.

Engines load lazily and stay process-cached, so a gateway only pulls in
the engine its persona actually uses. Pinned to 4 threads (Pocket is
fastest there on this ARM box; more threads regress on small models).
"""

from __future__ import annotations

import os
import logging
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tts_provider import TTSProvider

logger = logging.getLogger(__name__)

_HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
_MODELS_DIR = Path(
    os.environ.get(
        "KENSEIVOICE_MODELS_DIR",
        str(_HERMES_HOME / "plugins" / "kenseivoice" / "models"),
    )
)
_THREADS = 4

# Built-in catalogues (English only). Used for list_voices + validation.
_KOKORO_VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]
_POCKET_VOICES = [
    "alba", "anna", "azelma", "bill_boerst", "caro_davy", "charles", "cosette",
    "eponine", "eve", "fantine", "george", "jane", "javert", "jean", "marius",
    "mary", "michael", "paul", "peter_yearsley", "stuart_bell", "vera",
]


class _Engines:
    """Lazy, process-cached model holders guarded by a lock.

    ONNX/Torch inference is not reliably thread-safe; the gateway can
    call synthesize concurrently, so every inference is serialised.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._kokoro = None
        self._pocket = None
        self._pocket_states: Dict[str, Any] = {}

    def kokoro(self):
        if self._kokoro is None:
            os.environ.setdefault("OMP_NUM_THREADS", str(_THREADS))
            from kokoro_onnx import Kokoro
            self._kokoro = Kokoro(
                str(_MODELS_DIR / "kokoro-v1.0.onnx"),
                str(_MODELS_DIR / "voices-v1.0.bin"),
            )
        return self._kokoro

    def pocket(self):
        if self._pocket is None:
            import torch
            torch.set_num_threads(_THREADS)
            from pocket_tts import TTSModel
            self._pocket = TTSModel.load_model()
        return self._pocket

    def pocket_state(self, voice: str):
        st = self._pocket_states.get(voice)
        if st is None:
            st = self.pocket().get_state_for_audio_prompt(voice)
            self._pocket_states[voice] = st
        return st


_ENGINES = _Engines()


def _to_wav(engine: str, voice: str, text: str, wav_path: str, speed: float) -> None:
    import soundfile as sf
    with _ENGINES._lock:
        if engine == "kokoro":
            k = _ENGINES.kokoro()
            lang = "en-gb" if voice[:1] == "b" else "en-us"
            samples, sr = k.create(text, voice=voice, speed=speed, lang=lang)
            sf.write(wav_path, samples, sr)
        elif engine == "pocket":
            m = _ENGINES.pocket()
            state = _ENGINES.pocket_state(voice)
            wav = m.generate_audio(state, text)
            sf.write(wav_path, wav.flatten().cpu().numpy(), m.sample_rate)
        else:
            raise ValueError(f"unknown TTS engine {engine!r} (use kokoro|pocket)")


def _transcode(wav_path: str, output_path: str, fmt: str) -> None:
    """ffmpeg wav -> requested container. opus/ogg both go to libopus/ogg."""
    codec = {"mp3": ["-c:a", "libmp3lame"],
             "ogg": ["-c:a", "libopus"],
             "opus": ["-c:a", "libopus"],
             "flac": ["-c:a", "flac"]}.get(fmt, ["-c:a", "libmp3lame"])
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, *codec, output_path],
        check=True, capture_output=True,
    )


class KenseiVoiceProvider(TTSProvider):
    @property
    def name(self) -> str:
        return "kenseivoice"

    @property
    def display_name(self) -> str:
        return "KENSEI Voice (local Kokoro/Pocket)"

    @property
    def voice_compatible(self) -> bool:
        return True

    def is_available(self) -> bool:
        return (_MODELS_DIR / "kokoro-v1.0.onnx").exists() or True

    def list_voices(self) -> List[Dict[str, Any]]:
        out = [{"id": f"kokoro:{v}", "language": "en-GB" if v[:1] == "b" else "en-US"}
               for v in _KOKORO_VOICES]
        out += [{"id": f"pocket:{v}", "language": "en"} for v in _POCKET_VOICES]
        return out

    def default_voice(self) -> Optional[str]:
        return "kokoro:bf_emma"

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,
        format: str = "mp3",
        **extra: Any,
    ) -> str:
        voice = voice or self.default_voice()
        engine, _, vid = voice.partition(":")
        if not vid:  # bare voice id -> default to kokoro
            engine, vid = "kokoro", engine
        spd = float(speed) if speed else 1.0
        fmt = (format or "mp3").lower()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav_path = tf.name
        try:
            _to_wav(engine, vid, text, wav_path, spd)
            if fmt == "wav":
                os.replace(wav_path, output_path)
            else:
                _transcode(wav_path, output_path, fmt)
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
        return output_path


def register(ctx) -> None:
    """Plugin entry point — wire KenseiVoiceProvider into the registry."""
    ctx.register_tts_provider(KenseiVoiceProvider())
