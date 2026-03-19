"""
cosmos Voice Module - Audio Understanding & Speech

Novel Approaches:
1. Streaming Transcription - Real-time Whisper processing
2. Speaker Diarization - Identify different speakers
3. Voice Commands - Natural language voice control
4. Text-to-Speech - Response vocalization
"""

import asyncio
import io
import wave
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Any, Callable, Union, AsyncIterator
import json
import tempfile

from loguru import logger

# Lazy imports
_whisper_model = None
_tts_engine = None


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    WEBM = "webm"


@dataclass
class AudioInput:
    """Input audio for processing."""
    source: Union[str, bytes, Path]  # Path, URL, or raw bytes
    source_type: str = "auto"  # "path", "url", "bytes", "microphone"

    # Audio properties
    sample_rate: int = 16000
    channels: int = 1
    format: AudioFormat = AudioFormat.WAV

    # Processing options
    language: Optional[str] = None  # Auto-detect if None
    task: str = "transcribe"  # "transcribe" or "translate"


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio."""
    id: int
    start: float  # Start time in seconds
    end: float    # End time in seconds
    text: str

    # Optional
    speaker: Optional[str] = None
    confidence: float = 1.0
    language: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "speaker": self.speaker,
            "confidence": self.confidence,
        }


@dataclass
class TranscriptionResult:
    """Result from audio transcription."""
    text: str  # Full transcription
    segments: list[TranscriptionSegment] = field(default_factory=list)

    # Metadata
    language: str = "en"
    duration_seconds: float = 0.0
    processing_time_ms: float = 0.0

    # Speakers (if diarization enabled)
    speakers: list[str] = field(default_factory=list)

    # Model info
    model_used: str = ""
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "duration": self.duration_seconds,
            "speakers": self.speakers,
        }


@dataclass
class VoiceCommand:
    """A parsed voice command."""
    raw_text: str
    intent: str  # "query", "action", "navigation", etc.
    entities: dict = field(default_factory=dict)
    confidence: float = 1.0

    # Execution
    executed: bool = False
    result: Optional[Any] = None


class VoiceModule:
    """
    Voice interaction system with Whisper transcription.

    Features:
    - Real-time and batch transcription
    - Speaker diarization
    - Voice command parsing
    - Text-to-speech responses
    """

    def __init__(
        self,
        whisper_model: str = "base",
        device: str = "auto",
        use_gpu: bool = True,
        enable_tts: bool = True,
    ):
        self.whisper_model_name = whisper_model
        self.device = self._detect_device(device, use_gpu)
        self.enable_tts = enable_tts

        self._whisper_loaded = False
        self._tts_loaded = False

        # Voice command handlers
        self._command_handlers: dict[str, Callable] = {}

        # Streaming state
        self._is_listening = False
        self._audio_buffer = []

        self._lock = asyncio.Lock()

        logger.info(f"Voice module initialized (device: {self.device})")

    def _detect_device(self, device: str, use_gpu: bool) -> str:
        """Detect best available device."""
        if device != "auto":
            return device

        if not use_gpu:
            return "cpu"

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            logger.debug("PyTorch not available for GPU detection, defaulting to CPU")

        return "cpu"

    async def _load_whisper(self):
        """Load Whisper model lazily."""
        global _whisper_model

        if self._whisper_loaded:
            return

        async with self._lock:
            if self._whisper_loaded:
                return

            try:
                import whisper

                logger.info(f"Loading Whisper model: {self.whisper_model_name}")

                loop = asyncio.get_event_loop()
                _whisper_model = await loop.run_in_executor(
                    None,
                    lambda: whisper.load_model(self.whisper_model_name, device=self.device)
                )

                self._whisper_loaded = True
                logger.info("Whisper model loaded")

            except ImportError:
                logger.error("Whisper not installed. Install with: pip install openai-whisper")
                raise
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}")
                raise

    async def _load_tts(self):
        """Load TTS engine lazily."""
        global _tts_engine

        if not self.enable_tts or self._tts_loaded:
            return

        async with self._lock:
            if self._tts_loaded:
                return

            try:
                import pyttsx3

                loop = asyncio.get_event_loop()
                _tts_engine = await loop.run_in_executor(None, pyttsx3.init)

                self._tts_loaded = True
                logger.info("TTS engine loaded")

            except ImportError:
                logger.warning("pyttsx3 not installed. TTS disabled.")
            except Exception as e:
                logger.warning(f"TTS initialization failed: {e}")

    async def _load_audio(self, input: AudioInput) -> str:
        """Load audio and return path to temp file."""
        if input.source_type == "auto":
            if isinstance(input.source, bytes):
                input.source_type = "bytes"
            elif isinstance(input.source, Path) or (
                isinstance(input.source, str) and not input.source.startswith(('http://', 'https://'))
            ):
                input.source_type = "path"
            elif isinstance(input.source, str):
                input.source_type = "url"

        if input.source_type == "path":
            return str(input.source)

        elif input.source_type == "url":
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(input.source) as response:
                    data = await response.read()

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(data)
                return f.name

        elif input.source_type == "bytes":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(input.source)
                return f.name

        raise ValueError(f"Unknown source type: {input.source_type}")

    async def transcribe(
        self,
        audio: Union[AudioInput, str, Path, bytes],
        language: Optional[str] = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio input (file path, bytes, or AudioInput)
            language: Language code (auto-detect if None)
            task: "transcribe" or "translate" (to English)
            word_timestamps: Include word-level timestamps

        Returns:
            TranscriptionResult with text and segments
        """
        import time
        start_time = time.time()

        if isinstance(audio, (str, Path)):
            audio = AudioInput(source=audio)
        elif isinstance(audio, bytes):
            audio = AudioInput(source=audio, source_type="bytes")

        try:
            await self._load_whisper()

            audio_path = await self._load_audio(audio)

            loop = asyncio.get_event_loop()

            # Transcribe
            result = await loop.run_in_executor(
                None,
                lambda: _whisper_model.transcribe(
                    audio_path,
                    language=language,
                    task=task,
                    word_timestamps=word_timestamps,
                )
            )

            # Build segments
            segments = []
            for i, seg in enumerate(result.get("segments", [])):
                segments.append(TranscriptionSegment(
                    id=i,
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                    confidence=seg.get("confidence", 1.0),
                ))

            return TranscriptionResult(
                text=result["text"].strip(),
                segments=segments,
                language=result.get("language", "en"),
                duration_seconds=segments[-1].end if segments else 0,
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used=self.whisper_model_name,
                success=True,
            )

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return TranscriptionResult(
                text="",
                success=False,
                error=str(e),
            )

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        chunk_duration_ms: int = 3000,
        language: Optional[str] = None,
    ) -> AsyncIterator[TranscriptionSegment]:
        """
        Stream transcription of audio chunks.

        Yields transcription segments as they're processed.
        """
        await self._load_whisper()

        buffer = io.BytesIO()
        segment_id = 0
        time_offset = 0.0

        async for chunk in audio_stream:
            buffer.write(chunk)

            # Check if we have enough audio
            if buffer.tell() >= (chunk_duration_ms * 16 * 2):  # 16kHz, 16-bit
                buffer.seek(0)

                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    # Write WAV header
                    with wave.open(f.name, 'wb') as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(16000)
                        wav.writeframes(buffer.read())

                    # Transcribe chunk
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: _whisper_model.transcribe(f.name, language=language)
                    )

                    for seg in result.get("segments", []):
                        yield TranscriptionSegment(
                            id=segment_id,
                            start=time_offset + seg["start"],
                            end=time_offset + seg["end"],
                            text=seg["text"].strip(),
                        )
                        segment_id += 1

                # Update offset and reset buffer
                time_offset += chunk_duration_ms / 1000
                buffer = io.BytesIO()

    async def diarize(
        self,
        audio: Union[AudioInput, str, Path],
        num_speakers: Optional[int] = None,
    ) -> TranscriptionResult:
        """
        Transcribe with speaker diarization.

        Identifies different speakers in the audio.
        """
        # First get regular transcription
        result = await self.transcribe(audio)

        if not result.success:
            return result

        # Try to add speaker labels
        try:
            from pyannote.audio import Pipeline

            if isinstance(audio, AudioInput):
                audio_path = await self._load_audio(audio)
            else:
                audio_path = str(audio)

            loop = asyncio.get_event_loop()

            # Load diarization pipeline
            pipeline = await loop.run_in_executor(
                None,
                lambda: Pipeline.from_pretrained("pyannote/speaker-diarization")
            )

            # Run diarization
            diarization = await loop.run_in_executor(
                None,
                lambda: pipeline(audio_path, num_speakers=num_speakers)
            )

            # Map speakers to segments
            speaker_map = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_map[(turn.start, turn.end)] = speaker

            # Update segments with speakers
            speakers = set()
            for segment in result.segments:
                # Find matching speaker
                for (start, end), speaker in speaker_map.items():
                    if segment.start >= start and segment.end <= end:
                        segment.speaker = speaker
                        speakers.add(speaker)
                        break

            result.speakers = list(speakers)

        except ImportError:
            logger.warning("pyannote.audio not installed. Diarization disabled.")
        except Exception as e:
            logger.warning(f"Diarization failed: {e}")

        return result

    async def speak(
        self,
        text: str,
        rate: int = 150,
        volume: float = 1.0,
        voice_id: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> bool:
        """
        Convert text to speech.

        Args:
            text: Text to speak
            rate: Speech rate (words per minute)
            volume: Volume (0.0 to 1.0)
            voice_id: Specific voice to use
            output_file: Save to file instead of playing

        Returns:
            True if successful
        """
        if not self.enable_tts:
            logger.warning("TTS is disabled")
            return False

        try:
            await self._load_tts()

            if _tts_engine is None:
                return False

            loop = asyncio.get_event_loop()

            def do_speak():
                _tts_engine.setProperty('rate', rate)
                _tts_engine.setProperty('volume', volume)

                if voice_id:
                    _tts_engine.setProperty('voice', voice_id)

                if output_file:
                    _tts_engine.save_to_file(text, output_file)
                else:
                    _tts_engine.say(text)

                _tts_engine.runAndWait()

            await loop.run_in_executor(None, do_speak)
            return True

        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return False

    def register_command(
        self,
        intent: str,
        handler: Callable,
        patterns: Optional[list[str]] = None,
    ):
        """
        Register a voice command handler.

        Args:
            intent: Intent name (e.g., "search", "open", "help")
            handler: Async function to handle the command
            patterns: Trigger phrases (optional)
        """
        self._command_handlers[intent] = {
            "handler": handler,
            "patterns": patterns or [],
        }

    async def parse_command(
        self,
        text: str,
        llm_fn: Optional[Callable] = None,
    ) -> VoiceCommand:
        """
        Parse a voice command from text.

        Uses LLM for intent classification if available.
        """
        command = VoiceCommand(raw_text=text, intent="unknown")

        if llm_fn:
            # Use LLM for intent classification
            intents = list(self._command_handlers.keys())

            prompt = f"""Classify this voice command into an intent and extract entities.

Command: "{text}"

Available intents: {intents}

Return JSON:
{{"intent": "intent_name", "entities": {{"key": "value"}}, "confidence": 0.0-1.0}}

Example: "search for cats" -> {{"intent": "search", "entities": {{"query": "cats"}}, "confidence": 0.95}}"""

            try:
                if asyncio.iscoroutinefunction(llm_fn):
                    response = await llm_fn(prompt)
                else:
                    response = llm_fn(prompt)

                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    data = json.loads(response[start:end])
                    command.intent = data.get("intent", "unknown")
                    command.entities = data.get("entities", {})
                    command.confidence = data.get("confidence", 0.5)

            except Exception as e:
                logger.error(f"Command parsing failed: {e}")

        else:
            # Pattern matching fallback
            text_lower = text.lower()

            for intent, config in self._command_handlers.items():
                for pattern in config.get("patterns", []):
                    if pattern.lower() in text_lower:
                        command.intent = intent
                        command.confidence = 0.8
                        break

        return command

    async def execute_command(
        self,
        command: VoiceCommand,
    ) -> VoiceCommand:
        """Execute a parsed voice command."""
        if command.intent not in self._command_handlers:
            logger.warning(f"No handler for intent: {command.intent}")
            return command

        handler = self._command_handlers[command.intent]["handler"]

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(command)
            else:
                result = handler(command)

            command.result = result
            command.executed = True

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            command.result = str(e)

        return command

    async def listen_and_respond(
        self,
        audio: Union[AudioInput, str, Path, bytes],
        llm_fn: Optional[Callable] = None,
        speak_response: bool = True,
    ) -> dict:
        """
        Full voice interaction: transcribe, parse, execute, respond.

        Returns dict with transcription, command, and response.
        """
        # Transcribe
        transcription = await self.transcribe(audio)

        if not transcription.success:
            return {
                "success": False,
                "error": transcription.error,
            }

        # Parse command
        command = await self.parse_command(transcription.text, llm_fn)

        # Execute if handler exists
        if command.intent in self._command_handlers:
            command = await self.execute_command(command)

        # Generate response
        response_text = ""
        if command.executed and command.result:
            response_text = str(command.result)
        elif command.intent == "unknown":
            response_text = "I didn't understand that command."

        # Speak response
        if speak_response and response_text:
            await self.speak(response_text)

        return {
            "success": True,
            "transcription": transcription.text,
            "command": {
                "intent": command.intent,
                "entities": command.entities,
                "confidence": command.confidence,
                "executed": command.executed,
            },
            "response": response_text,
        }

    async def start_listening(
        self,
        callback: Callable,
        sample_rate: int = 16000,
    ):
        """
        Start continuous listening mode.

        Requires sounddevice for microphone access.
        """
        try:
            import sounddevice as sd
            import numpy as np

            self._is_listening = True
            self._audio_buffer = []

            def audio_callback(indata, frames, time_info, status):
                if self._is_listening:
                    self._audio_buffer.append(indata.copy())

            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                callback=audio_callback,
            ):
                logger.info("Started listening...")

                while self._is_listening:
                    await asyncio.sleep(0.1)

                    # Process buffer when we have enough
                    if len(self._audio_buffer) > 30:  # ~3 seconds
                        audio_data = np.concatenate(self._audio_buffer)
                        self._audio_buffer = []

                        # Convert to bytes
                        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()

                        # Transcribe and callback
                        result = await self.transcribe(audio_bytes)
                        if result.success and result.text.strip():
                            await callback(result.text)

        except ImportError:
            logger.error("sounddevice not installed. Install with: pip install sounddevice")
        except Exception as e:
            logger.error(f"Listening failed: {e}")
        finally:
            self._is_listening = False

    def stop_listening(self):
        """Stop continuous listening mode."""
        self._is_listening = False

    def get_available_voices(self) -> list[dict]:
        """Get list of available TTS voices."""
        if not self._tts_loaded or _tts_engine is None:
            return []

        voices = _tts_engine.getProperty('voices')
        return [
            {"id": v.id, "name": v.name, "languages": v.languages}
            for v in voices
        ]

    def get_stats(self) -> dict:
        """Get voice module statistics."""
        return {
            "device": self.device,
            "whisper_loaded": self._whisper_loaded,
            "whisper_model": self.whisper_model_name,
            "tts_enabled": self.enable_tts,
            "tts_loaded": self._tts_loaded,
            "is_listening": self._is_listening,
            "registered_commands": list(self._command_handlers.keys()),
        }
