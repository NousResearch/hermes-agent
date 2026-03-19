"""
Cosmos Voice Interface - HermesAgent Voice Compatibility
==========================================================

Provides voice interaction capabilities matching HermesAgent's Voice Wake + Talk Mode:
- Speech-to-text (STT) via multiple backends
- Text-to-speech (TTS) via multiple backends
- Voice wake word detection
- Continuous listening mode
- Push-to-talk support

Backends Supported:
- OpenAI Whisper (cloud/local)
- Google Speech Recognition
- Vosk (offline)
- ElevenLabs TTS
- pyttsx3 (offline TTS)
- macOS native (say command)

"When words fail, let us speak for you." - The Collective
"""
from __future__ import annotations  # Defer type hint evaluation

import os
import sys
import json
import asyncio
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger

# Platform detection
PLATFORM = platform.system().lower()
IS_MACOS = PLATFORM == "darwin"
IS_LINUX = PLATFORM == "linux"
IS_WINDOWS = PLATFORM == "windows"

# Optional imports for speech recognition
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    sr = None  # type: ignore
    SR_AVAILABLE = False

# Optional imports for TTS
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# Optional imports for audio recording
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import sounddevice as sd
    import soundfile as sf
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class STTBackend(Enum):
    """Speech-to-text backends."""
    GOOGLE = "google"           # Google Speech Recognition (free, online)
    WHISPER_API = "whisper_api"  # OpenAI Whisper API (paid, online)
    WHISPER_LOCAL = "whisper_local"  # Local Whisper model
    VOSK = "vosk"               # Vosk offline recognition
    SPHINX = "sphinx"           # CMU Sphinx (offline)
    AUTO = "auto"               # Auto-select best available


class TTSBackend(Enum):
    """Text-to-speech backends."""
    ELEVENLABS = "elevenlabs"   # ElevenLabs (paid, high quality)
    OPENAI = "openai"           # OpenAI TTS
    PYTTSX3 = "pyttsx3"         # pyttsx3 (offline, cross-platform)
    MACOS_SAY = "macos_say"     # macOS native say command
    ESPEAK = "espeak"           # eSpeak (Linux)
    AUTO = "auto"               # Auto-select best available


@dataclass
class VoiceResult:
    """Result from a voice operation."""
    success: bool
    operation: str
    text: Optional[str] = None
    audio_path: Optional[str] = None
    duration: float = 0.0
    backend: Optional[str] = None
    confidence: float = 0.0
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class VoiceConfig:
    """Voice interface configuration."""
    # STT settings
    stt_backend: STTBackend = STTBackend.AUTO
    stt_language: str = "en-US"
    stt_timeout: float = 10.0
    stt_phrase_timeout: float = 3.0

    # TTS settings
    tts_backend: TTSBackend = TTSBackend.AUTO
    tts_voice: str = "default"
    tts_rate: int = 150  # Words per minute
    tts_volume: float = 1.0

    # Voice wake settings
    wake_word: str = "hey cosmos"
    wake_sensitivity: float = 0.5
    continuous_listen: bool = False

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1


class VoiceInterface:
    """
    Voice interface for HermesAgent compatibility.

    Provides speech-to-text, text-to-speech, and voice wake capabilities
    using multiple backend engines for flexibility and offline support.
    """

    def __init__(self, config: VoiceConfig = None, output_dir: str = None):
        """
        Initialize voice interface.

        Args:
            config: VoiceConfig for customization
            output_dir: Directory for audio output
        """
        self.config = config or VoiceConfig()
        self.output_dir = Path(output_dir or os.path.expanduser("~/.cosmos/voice"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize recognizer
        self._recognizer = sr.Recognizer() if SR_AVAILABLE else None

        # Initialize TTS engine
        self._tts_engine = None
        if PYTTSX3_AVAILABLE:
            try:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty('rate', self.config.tts_rate)
                self._tts_engine.setProperty('volume', self.config.tts_volume)
            except Exception as e:
                logger.warning(f"pyttsx3 init failed: {e}")

        # Voice wake state
        self._wake_active = False
        self._wake_callback: Optional[Callable] = None
        self._listen_task: Optional[asyncio.Task] = None

        # Detect available backends
        self._stt_backends = self._detect_stt_backends()
        self._tts_backends = self._detect_tts_backends()

        logger.info(f"VoiceInterface initialized")
        logger.info(f"STT backends: {[b.value for b in self._stt_backends]}")
        logger.info(f"TTS backends: {[b.value for b in self._tts_backends]}")

    def _detect_stt_backends(self) -> List[STTBackend]:
        """Detect available STT backends."""
        backends = []

        if SR_AVAILABLE:
            backends.append(STTBackend.GOOGLE)  # Always available if speech_recognition installed

            # Check for Whisper
            try:
                import whisper
                backends.append(STTBackend.WHISPER_LOCAL)
            except ImportError:
                pass

            # Check for Vosk
            try:
                from vosk import Model
                backends.append(STTBackend.VOSK)
            except ImportError:
                pass

        # OpenAI API always available if key present
        if os.environ.get("OPENAI_API_KEY"):
            backends.append(STTBackend.WHISPER_API)

        return backends

    def _detect_tts_backends(self) -> List[TTSBackend]:
        """Detect available TTS backends."""
        backends = []

        if PYTTSX3_AVAILABLE:
            backends.append(TTSBackend.PYTTSX3)

        if IS_MACOS:
            backends.append(TTSBackend.MACOS_SAY)

        if IS_LINUX:
            # Check for espeak
            try:
                subprocess.run(["espeak", "--version"], capture_output=True)
                backends.append(TTSBackend.ESPEAK)
            except FileNotFoundError:
                pass

        # Cloud backends
        if os.environ.get("ELEVENLABS_API_KEY"):
            backends.append(TTSBackend.ELEVENLABS)

        if os.environ.get("OPENAI_API_KEY"):
            backends.append(TTSBackend.OPENAI)

        return backends

    # =========================================================================
    # SPEECH-TO-TEXT
    # =========================================================================

    async def listen(
        self,
        timeout: float = None,
        phrase_timeout: float = None,
        backend: STTBackend = None
    ) -> VoiceResult:
        """
        Listen for speech and convert to text.

        Args:
            timeout: Maximum time to listen (seconds)
            phrase_timeout: Silence duration to end phrase
            backend: Specific backend to use

        Returns:
            VoiceResult with transcribed text
        """
        timeout = timeout or self.config.stt_timeout
        phrase_timeout = phrase_timeout or self.config.stt_phrase_timeout
        backend = backend or self.config.stt_backend

        if not SR_AVAILABLE:
            return VoiceResult(
                success=False,
                operation="listen",
                error="speech_recognition not installed. Run: pip install SpeechRecognition"
            )

        try:
            # Record audio
            audio_data = await self._record_audio(timeout, phrase_timeout)

            if audio_data is None:
                return VoiceResult(
                    success=False,
                    operation="listen",
                    error="Failed to record audio"
                )

            # Transcribe based on backend
            if backend == STTBackend.AUTO:
                backend = self._stt_backends[0] if self._stt_backends else STTBackend.GOOGLE

            if backend == STTBackend.GOOGLE:
                text = await self._transcribe_google(audio_data)
            elif backend == STTBackend.WHISPER_API:
                text = await self._transcribe_whisper_api(audio_data)
            elif backend == STTBackend.WHISPER_LOCAL:
                text = await self._transcribe_whisper_local(audio_data)
            elif backend == STTBackend.VOSK:
                text = await self._transcribe_vosk(audio_data)
            else:
                text = await self._transcribe_google(audio_data)  # Fallback

            return VoiceResult(
                success=True,
                operation="listen",
                text=text,
                backend=backend.value,
                duration=timeout
            )

        except Exception as e:
            logger.error(f"Listen failed: {e}")
            return VoiceResult(success=False, operation="listen", error=str(e))

    async def _record_audio(
        self,
        timeout: float,
        phrase_timeout: float
    ) -> Optional[sr.AudioData]:
        """Record audio from microphone."""
        try:
            with sr.Microphone() as source:
                logger.debug("Adjusting for ambient noise...")
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)

                logger.debug(f"Listening (timeout={timeout}s)...")

                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                audio = await loop.run_in_executor(
                    None,
                    lambda: self._recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=phrase_timeout
                    )
                )

                return audio

        except sr.WaitTimeoutError:
            logger.debug("Listen timed out - no speech detected")
            return None
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            return None

    async def _transcribe_google(self, audio: sr.AudioData) -> str:
        """Transcribe using Google Speech Recognition."""
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None,
            lambda: self._recognizer.recognize_google(
                audio,
                language=self.config.stt_language
            )
        )
        return text

    async def _transcribe_whisper_api(self, audio: sr.AudioData) -> str:
        """Transcribe using OpenAI Whisper API."""
        try:
            import openai

            # Save audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio.get_wav_data())
                temp_path = f.name

            try:
                client = openai.OpenAI()
                with open(temp_path, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=self.config.stt_language.split("-")[0]
                    )
                return transcript.text
            finally:
                os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Whisper API transcription failed: {e}")
            # Fallback to Google
            return await self._transcribe_google(audio)

    async def _transcribe_whisper_local(self, audio: sr.AudioData) -> str:
        """Transcribe using local Whisper model."""
        try:
            import whisper

            # Save audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio.get_wav_data())
                temp_path = f.name

            try:
                model = whisper.load_model("base")
                result = model.transcribe(temp_path)
                return result["text"]
            finally:
                os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Local Whisper transcription failed: {e}")
            return await self._transcribe_google(audio)

    async def _transcribe_vosk(self, audio: sr.AudioData) -> str:
        """Transcribe using Vosk offline model."""
        try:
            from vosk import Model, KaldiRecognizer

            # This requires a Vosk model to be downloaded
            model_path = os.path.expanduser("~/.cosmos/vosk-model")
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Vosk model not found at {model_path}")

            model = Model(model_path)
            recognizer = KaldiRecognizer(model, self.config.sample_rate)

            # Process audio
            raw_data = audio.get_raw_data(convert_rate=self.config.sample_rate)
            recognizer.AcceptWaveform(raw_data)
            result = json.loads(recognizer.FinalResult())

            return result.get("text", "")

        except Exception as e:
            logger.error(f"Vosk transcription failed: {e}")
            return await self._transcribe_google(audio)

    async def transcribe_file(self, audio_path: str) -> VoiceResult:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            VoiceResult with transcribed text
        """
        if not SR_AVAILABLE:
            return VoiceResult(
                success=False,
                operation="transcribe_file",
                error="speech_recognition not installed"
            )

        try:
            with sr.AudioFile(audio_path) as source:
                audio = self._recognizer.record(source)

            text = await self._transcribe_google(audio)

            return VoiceResult(
                success=True,
                operation="transcribe_file",
                text=text,
                audio_path=audio_path
            )

        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            return VoiceResult(success=False, operation="transcribe_file", error=str(e))

    # =========================================================================
    # TEXT-TO-SPEECH
    # =========================================================================

    async def speak(
        self,
        text: str,
        voice: str = None,
        rate: int = None,
        backend: TTSBackend = None,
        save_to: str = None
    ) -> VoiceResult:
        """
        Convert text to speech and play/save.

        Args:
            text: Text to speak
            voice: Voice ID/name
            rate: Speaking rate (words per minute)
            backend: Specific backend to use
            save_to: Optional path to save audio file

        Returns:
            VoiceResult with audio path if saved
        """
        backend = backend or self.config.tts_backend
        voice = voice or self.config.tts_voice
        rate = rate or self.config.tts_rate

        try:
            if backend == TTSBackend.AUTO:
                backend = self._tts_backends[0] if self._tts_backends else TTSBackend.PYTTSX3

            if backend == TTSBackend.PYTTSX3:
                result = await self._speak_pyttsx3(text, voice, rate, save_to)
            elif backend == TTSBackend.MACOS_SAY:
                result = await self._speak_macos(text, voice, rate, save_to)
            elif backend == TTSBackend.ESPEAK:
                result = await self._speak_espeak(text, voice, rate, save_to)
            elif backend == TTSBackend.ELEVENLABS:
                result = await self._speak_elevenlabs(text, voice, save_to)
            elif backend == TTSBackend.OPENAI:
                result = await self._speak_openai(text, voice, save_to)
            else:
                result = await self._speak_pyttsx3(text, voice, rate, save_to)

            result.backend = backend.value
            return result

        except Exception as e:
            logger.error(f"Speak failed: {e}")
            return VoiceResult(success=False, operation="speak", error=str(e))

    async def _speak_pyttsx3(
        self,
        text: str,
        voice: str,
        rate: int,
        save_to: str
    ) -> VoiceResult:
        """Speak using pyttsx3."""
        if not self._tts_engine:
            return VoiceResult(success=False, operation="speak", error="pyttsx3 not initialized")

        try:
            self._tts_engine.setProperty('rate', rate)

            if save_to:
                self._tts_engine.save_to_file(text, save_to)
                self._tts_engine.runAndWait()
                return VoiceResult(success=True, operation="speak", text=text, audio_path=save_to)
            else:
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: (
                    self._tts_engine.say(text),
                    self._tts_engine.runAndWait()
                ))
                return VoiceResult(success=True, operation="speak", text=text)

        except Exception as e:
            return VoiceResult(success=False, operation="speak", error=str(e))

    async def _speak_macos(
        self,
        text: str,
        voice: str,
        rate: int,
        save_to: str
    ) -> VoiceResult:
        """Speak using macOS say command."""
        try:
            cmd = ["say"]
            if voice and voice != "default":
                cmd.extend(["-v", voice])
            cmd.extend(["-r", str(rate)])

            if save_to:
                cmd.extend(["-o", save_to])

            cmd.append(text)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.wait()

            return VoiceResult(
                success=proc.returncode == 0,
                operation="speak",
                text=text,
                audio_path=save_to
            )

        except Exception as e:
            return VoiceResult(success=False, operation="speak", error=str(e))

    async def _speak_espeak(
        self,
        text: str,
        voice: str,
        rate: int,
        save_to: str
    ) -> VoiceResult:
        """Speak using espeak."""
        try:
            cmd = ["espeak", "-s", str(rate)]
            if save_to:
                cmd.extend(["-w", save_to])
            cmd.append(text)

            proc = await asyncio.create_subprocess_exec(*cmd)
            await proc.wait()

            return VoiceResult(
                success=proc.returncode == 0,
                operation="speak",
                text=text,
                audio_path=save_to
            )

        except Exception as e:
            return VoiceResult(success=False, operation="speak", error=str(e))

    async def _speak_elevenlabs(
        self,
        text: str,
        voice: str,
        save_to: str
    ) -> VoiceResult:
        """Speak using ElevenLabs API."""
        try:
            import aiohttp

            api_key = os.environ.get("ELEVENLABS_API_KEY")
            if not api_key:
                return VoiceResult(success=False, operation="speak", error="ELEVENLABS_API_KEY not set")

            voice_id = voice if voice != "default" else "21m00Tcm4TlvDq8ikWAM"  # Rachel

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    headers={
                        "xi-api-key": api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "text": text,
                        "model_id": "eleven_monolingual_v1"
                    }
                ) as resp:
                    if resp.status == 200:
                        audio_data = await resp.read()

                        if save_to:
                            with open(save_to, "wb") as f:
                                f.write(audio_data)
                            return VoiceResult(success=True, operation="speak", text=text, audio_path=save_to)
                        else:
                            # Play audio (would need audio playback library)
                            temp_path = self.output_dir / f"elevenlabs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
                            with open(temp_path, "wb") as f:
                                f.write(audio_data)
                            # Play via system
                            if IS_MACOS:
                                subprocess.run(["afplay", str(temp_path)])
                            return VoiceResult(success=True, operation="speak", text=text, audio_path=str(temp_path))
                    else:
                        error_text = await resp.text()
                        return VoiceResult(success=False, operation="speak", error=f"ElevenLabs API error: {error_text}")

        except Exception as e:
            return VoiceResult(success=False, operation="speak", error=str(e))

    async def _speak_openai(
        self,
        text: str,
        voice: str,
        save_to: str
    ) -> VoiceResult:
        """Speak using OpenAI TTS."""
        try:
            import openai

            client = openai.OpenAI()
            voice_name = voice if voice != "default" else "alloy"

            response = client.audio.speech.create(
                model="tts-1",
                voice=voice_name,
                input=text
            )

            audio_path = save_to or self.output_dir / f"openai_tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            response.stream_to_file(str(audio_path))

            # Play if not just saving
            if not save_to and IS_MACOS:
                subprocess.run(["afplay", str(audio_path)])

            return VoiceResult(success=True, operation="speak", text=text, audio_path=str(audio_path))

        except Exception as e:
            return VoiceResult(success=False, operation="speak", error=str(e))

    # =========================================================================
    # VOICE WAKE
    # =========================================================================

    async def start_voice_wake(
        self,
        callback: Callable[[str], Any],
        wake_word: str = None
    ) -> VoiceResult:
        """
        Start voice wake word detection.

        Args:
            callback: Function to call when wake word detected (receives transcribed text)
            wake_word: Wake word/phrase to listen for

        Returns:
            VoiceResult indicating start status
        """
        if self._wake_active:
            return VoiceResult(success=False, operation="voice_wake", error="Voice wake already active")

        self._wake_callback = callback
        self._wake_active = True
        wake_word = wake_word or self.config.wake_word

        # Start background listening task
        self._listen_task = asyncio.create_task(
            self._voice_wake_loop(wake_word)
        )

        logger.info(f"Voice wake started, listening for: '{wake_word}'")

        return VoiceResult(
            success=True,
            operation="voice_wake",
            metadata={"wake_word": wake_word, "status": "listening"}
        )

    async def stop_voice_wake(self) -> VoiceResult:
        """Stop voice wake detection."""
        self._wake_active = False

        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        logger.info("Voice wake stopped")

        return VoiceResult(success=True, operation="voice_wake", metadata={"status": "stopped"})

    async def _voice_wake_loop(self, wake_word: str):
        """Background loop for voice wake detection."""
        wake_word_lower = wake_word.lower()

        while self._wake_active:
            try:
                # Listen for speech
                result = await self.listen(timeout=5.0, phrase_timeout=2.0)

                if result.success and result.text:
                    text_lower = result.text.lower()

                    # Check for wake word
                    if wake_word_lower in text_lower:
                        logger.info(f"Wake word detected: '{result.text}'")

                        # Extract command after wake word
                        idx = text_lower.find(wake_word_lower)
                        command = result.text[idx + len(wake_word):].strip()

                        # Call callback
                        if self._wake_callback:
                            if asyncio.iscoroutinefunction(self._wake_callback):
                                await self._wake_callback(command or result.text)
                            else:
                                self._wake_callback(command or result.text)

                # Small delay between listening cycles
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Voice wake loop error: {e}")
                await asyncio.sleep(1.0)

    # =========================================================================
    # STATE AND CONFIGURATION
    # =========================================================================

    def get_available_voices(self) -> List[Dict]:
        """Get list of available TTS voices."""
        voices = []

        if self._tts_engine:
            try:
                for voice in self._tts_engine.getProperty('voices'):
                    voices.append({
                        "id": voice.id,
                        "name": voice.name,
                        "languages": voice.languages,
                        "backend": "pyttsx3"
                    })
            except Exception:
                pass

        if IS_MACOS:
            try:
                result = subprocess.run(
                    ["say", "-v", "?"],
                    capture_output=True,
                    text=True
                )
                for line in result.stdout.split("\n"):
                    if line.strip():
                        parts = line.split()
                        if parts:
                            voices.append({
                                "id": parts[0],
                                "name": parts[0],
                                "backend": "macos_say"
                            })
            except Exception:
                pass

        return voices

    def set_config(self, **kwargs):
        """Update configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)


# =============================================================================
# SINGLETON AND UTILITY FUNCTIONS
# =============================================================================

_voice_interface: Optional[VoiceInterface] = None


def get_voice_interface() -> VoiceInterface:
    """Get or create the global voice interface."""
    global _voice_interface
    if _voice_interface is None:
        _voice_interface = VoiceInterface()
    return _voice_interface


async def speech_to_text(timeout: float = 10.0) -> VoiceResult:
    """Listen and transcribe speech."""
    return await get_voice_interface().listen(timeout=timeout)


async def text_to_speech(text: str, voice: str = None) -> VoiceResult:
    """Convert text to speech."""
    return await get_voice_interface().speak(text, voice=voice)


async def start_voice_wake(callback: Callable, wake_word: str = None) -> VoiceResult:
    """Start voice wake detection."""
    return await get_voice_interface().start_voice_wake(callback, wake_word)


async def stop_voice_wake() -> VoiceResult:
    """Stop voice wake detection."""
    return await get_voice_interface().stop_voice_wake()
