"""
VTuber TTS - Streamlined voice synthesis for Cosmos VTuber streaming.

Single voice (Cosmos), F5-TTS voice cloning (primary) with XTTS v2 and Edge TTS fallbacks.
Pre-loads model at startup for zero-latency first generation.

Priority: F5-TTS (fast, high quality) → XTTS v2 → Edge TTS (no cloning)
"""

import asyncio
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Optional
from loguru import logger

# Check available providers at import time
F5TTS_AVAILABLE = False
XTTS_AVAILABLE = False
EDGE_TTS_AVAILABLE = False

try:
    from f5_tts.api import F5TTS as F5TTSModel
    F5TTS_AVAILABLE = True
except ImportError:
    pass

try:
    from TTS.api import TTS
    XTTS_AVAILABLE = True
except ImportError:
    pass

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    pass


class VTuberTTS:
    """Streamlined TTS for VTuber streaming - Cosmos voice only.

    Priority chain:
    1. F5-TTS - Fast zero-shot voice cloning (2-3s generation)
    2. XTTS v2 - Slower voice cloning fallback (5-25s generation)
    3. Edge TTS - No cloning, but fast and reliable
    """

    def __init__(
        self,
        reference_audio: str = "",
        cache_dir: str = "/tmp/vtuber_tts_cache",
        device: str = "cuda:0",
    ):
        self._reference_audio = reference_audio
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._device = device

        # F5-TTS (primary)
        self._f5tts_model = None
        self._f5tts_ready = False
        self._ref_text = ""  # Cached transcription of reference audio

        # XTTS v2 (secondary fallback)
        self._xtts_model = None
        self._xtts_ready = False

        self._generation_lock = asyncio.Lock()

        # Edge TTS voice for final fallback
        self._edge_voice = "en-US-GuyNeural"

    async def initialize(self) -> bool:
        """Pre-load TTS models at startup. Call this once."""
        has_ref = self._reference_audio and os.path.exists(self._reference_audio)

        if not has_ref:
            logger.warning(f"Reference audio not found: {self._reference_audio}")
            logger.warning("Will use Edge TTS only (no voice cloning)")
            return EDGE_TTS_AVAILABLE

        # Try F5-TTS first (primary)
        if F5TTS_AVAILABLE:
            try:
                await self._init_f5tts()
            except Exception as e:
                logger.error(f"F5-TTS initialization failed: {e}")
                self._f5tts_ready = False

        # Try XTTS as fallback
        if not self._f5tts_ready and XTTS_AVAILABLE:
            try:
                await self._init_xtts()
            except Exception as e:
                logger.error(f"XTTS initialization failed: {e}")
                self._xtts_ready = False

        ready = self._f5tts_ready or self._xtts_ready or EDGE_TTS_AVAILABLE
        if self._f5tts_ready:
            logger.info("VTuberTTS ready: F5-TTS (primary)")
        elif self._xtts_ready:
            logger.info("VTuberTTS ready: XTTS v2 (fallback)")
        elif EDGE_TTS_AVAILABLE:
            logger.info("VTuberTTS ready: Edge TTS only (no voice cloning)")
        return ready

    async def _init_f5tts(self):
        """Initialize F5-TTS model."""
        loop = asyncio.get_event_loop()

        def load_model():
            logger.info("Loading F5-TTS v1 model...")
            model = F5TTSModel(
                model="F5TTS_v1_Base",
                device=self._device if "cuda" in self._device else None,
            )
            logger.info("F5-TTS v1 loaded")
            return model

        self._f5tts_model = await loop.run_in_executor(None, load_model)

        if self._f5tts_model is not None:
            # Pre-transcribed reference text (Cosmos voice clips)
            # Avoids needing Whisper model download at runtime
            self._ref_text = (
                "Any more ridiculous ideas? Are you all right? Bad news, everyone. "
                "The creature is a shapeshifter. It knocked me out and took my form "
                "so it could prey on poor Hermes. Damn! Have you ever dissected a yeti before?"
            )
            logger.info(f"Using pre-transcribed reference text ({len(self._ref_text)} chars)")

            # Warm up with a short generation
            logger.info("Warming up F5-TTS...")
            warmup_path = self._cache_dir / "warmup_f5.wav"
            ok = await self._generate_f5tts("Hello, testing.", warmup_path)
            if warmup_path.exists():
                warmup_path.unlink()
            if ok:
                self._f5tts_ready = True
                logger.info("F5-TTS warm-up complete - ready for streaming")
            else:
                logger.warning("F5-TTS warm-up failed")

    async def _init_xtts(self):
        """Initialize XTTS v2 model."""
        loop = asyncio.get_event_loop()

        def load_model():
            import torch
            logger.info("Loading XTTS v2 model...")
            model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            if torch.cuda.is_available():
                model = model.to(self._device)
            logger.info("XTTS v2 loaded on GPU")
            return model

        self._xtts_model = await loop.run_in_executor(None, load_model)
        self._xtts_ready = self._xtts_model is not None

        if self._xtts_ready:
            logger.info("Warming up XTTS with reference audio...")
            warmup_path = self._cache_dir / "warmup.wav"
            await self._generate_xtts("Hello.", warmup_path)
            if warmup_path.exists():
                warmup_path.unlink()
            logger.info("XTTS warm-up complete")

    async def generate(self, text: str) -> Optional[str]:
        """Generate speech audio for the given text.

        Returns path to WAV file, or None on failure.
        All speech uses Cosmos's cloned voice.
        """
        if not text or not text.strip():
            return None

        text = self._clean_text(text)
        if not text:
            return None

        # Check cache
        cache_path = self._get_cache_path(text)
        if cache_path.exists():
            return str(cache_path)

        async with self._generation_lock:
            # Double-check cache after acquiring lock
            if cache_path.exists():
                return str(cache_path)

            # Try F5-TTS first (fast voice cloning, 2-3s)
            if self._f5tts_ready:
                try:
                    result = await asyncio.wait_for(
                        self._generate_f5tts(text, cache_path),
                        timeout=30.0,
                    )
                    if result and cache_path.exists():
                        return str(cache_path)
                    logger.warning("F5-TTS generation failed, trying XTTS fallback")
                except asyncio.TimeoutError:
                    logger.warning("F5-TTS generation timed out (30s), trying XTTS")

            # Try XTTS v2 (slower voice cloning)
            if self._xtts_ready:
                try:
                    result = await asyncio.wait_for(
                        self._generate_xtts(text, cache_path),
                        timeout=30.0,
                    )
                    if result and cache_path.exists():
                        return str(cache_path)
                    logger.warning("XTTS generation failed, falling back to Edge TTS")
                except asyncio.TimeoutError:
                    logger.warning("XTTS generation timed out (30s), using Edge TTS")

            # Fallback: Edge TTS (no cloning but fast and reliable)
            if EDGE_TTS_AVAILABLE:
                result = await self._generate_edge(text, cache_path)
                if result:
                    return str(cache_path)

            logger.error("All TTS providers failed")
            return None

    async def _generate_f5tts(self, text: str, output_path: Path) -> bool:
        """Generate with F5-TTS voice cloning."""
        try:
            loop = asyncio.get_event_loop()

            def _run():
                self._f5tts_model.infer(
                    ref_file=self._reference_audio,
                    ref_text=self._ref_text,
                    gen_text=text,
                    file_wave=str(output_path),
                    show_info=logger.debug,
                    nfe_step=32,
                    speed=1.0,
                )
                return True

            return await loop.run_in_executor(None, _run)
        except Exception as e:
            logger.error(f"F5-TTS error: {e}")
            return False

    async def _generate_xtts(self, text: str, output_path: Path) -> bool:
        """Generate with XTTS v2 voice cloning."""
        try:
            loop = asyncio.get_event_loop()

            def _run():
                self._xtts_model.tts_to_file(
                    text=text,
                    speaker_wav=self._reference_audio,
                    language="en",
                    file_path=str(output_path),
                )
                return True

            return await loop.run_in_executor(None, _run)
        except Exception as e:
            logger.error(f"XTTS error: {e}")
            return False

    async def _generate_edge(self, text: str, output_path: Path) -> bool:
        """Generate with Edge TTS (fast fallback, no voice cloning)."""
        try:
            mp3_path = str(output_path).replace(".wav", ".mp3")
            tts = edge_tts.Communicate(text, voice=self._edge_voice)
            await tts.save(mp3_path)

            # Convert to WAV for stream compatibility
            subprocess.run(
                ["ffmpeg", "-y", "-i", mp3_path, "-ar", "24000", "-ac", "1",
                 str(output_path)],
                capture_output=True, timeout=10,
            )
            # Clean up MP3
            try:
                os.unlink(mp3_path)
            except OSError:
                pass

            if output_path.exists():
                return True

            # If ffmpeg conversion failed, use MP3 directly
            if os.path.exists(mp3_path):
                os.rename(mp3_path, str(output_path))
                return True

            return False
        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            return False

    def _clean_text(self, text: str) -> str:
        """Clean text for TTS."""
        import re

        # Remove markdown
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'#{1,6}\s*', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # Remove emojis and special chars
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'[═─│┌┐└┘├┤┬┴┼]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _get_cache_path(self, text: str) -> Path:
        """Get deterministic cache path for text."""
        text_hash = hashlib.md5(f"cosmos:{text}".encode()).hexdigest()
        return self._cache_dir / f"cosmos_{text_hash}.wav"

    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of an audio file in seconds."""
        try:
            import soundfile as sf
            data, sr = sf.read(audio_path)
            return len(data) / sr
        except Exception:
            try:
                import wave
                with wave.open(audio_path, 'r') as f:
                    return f.getnframes() / f.getframerate()
            except Exception:
                return 5.0  # Default estimate

    async def cleanup(self):
        """Release GPU memory."""
        self._f5tts_model = None
        self._f5tts_ready = False
        self._xtts_model = None
        self._xtts_ready = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("VTuberTTS cleaned up")
