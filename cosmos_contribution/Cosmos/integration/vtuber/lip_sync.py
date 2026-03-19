"""
Lip Sync Engine - Real-time viseme generation from audio/text
Supports: Rhubarb, phoneme-based, amplitude-based, neural (MuseTalk)
"""

import asyncio
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, AsyncGenerator
from enum import Enum
import time
import subprocess
import json
import tempfile
import os
from pathlib import Path
from loguru import logger

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


class Viseme(Enum):
    """Preston Blair phoneme set - standard for animation"""
    SIL = "sil"   # Silence
    PP = "PP"     # P, B, M (closed lips)
    FF = "FF"     # F, V (teeth on lip)
    TH = "TH"     # Th (tongue between teeth)
    DD = "DD"     # T, D, N, L (tongue on teeth ridge)
    KK = "kk"     # K, G, NG (back of tongue up)
    CH = "CH"     # Ch, J, Sh, Zh (lips forward)
    SS = "SS"     # S, Z (teeth together)
    NN = "nn"     # N, L (tongue up)
    RR = "RR"     # R (lips slightly rounded)
    AA = "aa"     # A (wide open)
    E = "E"       # E (slightly open, wide)
    IH = "ih"     # I (small opening)
    OH = "oh"     # O (round, medium open)
    OU = "ou"     # U, OO (tight round)


@dataclass
class VisemeEvent:
    """A single viseme with timing"""
    viseme: str
    start_time: float  # seconds
    end_time: float
    intensity: float = 1.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class LipSyncData:
    """Complete lip sync data for an audio clip"""
    duration: float
    visemes: List[VisemeEvent]
    audio_path: Optional[str] = None

    def get_viseme_at(self, time: float) -> Tuple[str, float]:
        """Get the viseme and intensity at a specific time"""
        for v in self.visemes:
            if v.start_time <= time < v.end_time:
                return v.viseme, v.intensity
        return "sil", 0.0


class LipSyncMethod(Enum):
    """Available lip sync methods"""
    AMPLITUDE = "amplitude"      # Simple audio amplitude
    RHUBARB = "rhubarb"         # Rhubarb Lip Sync tool
    PHONEME = "phoneme"         # Text-based phoneme mapping
    NEURAL = "neural"           # MuseTalk / neural lip sync
    TIMESTAMPS = "timestamps"    # TTS with timestamps (ElevenLabs style)


# Phoneme to viseme mapping (CMU/ARPABET to Preston Blair)
PHONEME_TO_VISEME = {
    # Consonants
    'P': 'PP', 'B': 'PP', 'M': 'PP',
    'F': 'FF', 'V': 'FF',
    'TH': 'TH', 'DH': 'TH',
    'T': 'DD', 'D': 'DD', 'N': 'DD', 'L': 'DD',
    'K': 'kk', 'G': 'kk', 'NG': 'kk',
    'CH': 'CH', 'JH': 'CH', 'SH': 'CH', 'ZH': 'CH',
    'S': 'SS', 'Z': 'SS',
    'R': 'RR', 'ER': 'RR',
    'W': 'ou', 'Y': 'ih',
    'HH': 'sil',

    # Vowels
    'AA': 'aa', 'AE': 'aa', 'AH': 'aa', 'AO': 'aa', 'AW': 'aa',
    'AY': 'aa',
    'EH': 'E', 'EY': 'E',
    'IH': 'ih', 'IY': 'ih',
    'OW': 'oh', 'OY': 'oh',
    'UH': 'ou', 'UW': 'ou',
}

# Simple text character to phoneme approximation
CHAR_TO_PHONEME = {
    'a': 'AA', 'e': 'EH', 'i': 'IH', 'o': 'OW', 'u': 'UW',
    'b': 'B', 'c': 'K', 'd': 'D', 'f': 'F', 'g': 'G',
    'h': 'HH', 'j': 'JH', 'k': 'K', 'l': 'L', 'm': 'M',
    'n': 'N', 'p': 'P', 'q': 'K', 'r': 'R', 's': 'S',
    't': 'T', 'v': 'V', 'w': 'W', 'x': 'K', 'y': 'Y', 'z': 'Z',
}


class LipSyncEngine:
    """
    Multi-method lip sync engine for Cosmos VTuber

    Generates viseme sequences from:
    - Audio files (amplitude or Rhubarb)
    - Text (phoneme approximation)
    - TTS timestamps
    - Real-time audio stream
    """

    def __init__(self, method: LipSyncMethod = LipSyncMethod.AMPLITUDE):
        self.method = method
        self._rhubarb_path: Optional[str] = None
        self._check_rhubarb()

        # Real-time state
        self._current_viseme = "sil"
        self._current_intensity = 0.0

        logger.info(f"LipSyncEngine initialized with method: {method}")

    def _check_rhubarb(self):
        """Check if Rhubarb Lip Sync is available"""
        # Check common installation paths
        rhubarb_paths = [
            '/workspace/Rhubarb-Lip-Sync-1.13.0-Linux/rhubarb',
            '/workspace/rhubarb/rhubarb',
            '/usr/local/bin/rhubarb',
            'rhubarb'  # PATH fallback
        ]

        for path in rhubarb_paths:
            try:
                result = subprocess.run([path, '--version'],
                                       capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    self._rhubarb_path = path
                    logger.info(f"Rhubarb Lip Sync found at {path}")
                    return
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        logger.warning("Rhubarb Lip Sync not found - using amplitude fallback")

    async def generate_from_audio(self, audio_path: str,
                                  transcript: Optional[str] = None) -> LipSyncData:
        """Generate lip sync data from audio file"""
        if self.method == LipSyncMethod.RHUBARB and self._rhubarb_path:
            return await self._generate_rhubarb(audio_path, transcript)
        else:
            return await self._generate_amplitude(audio_path)

    async def _generate_rhubarb(self, audio_path: str,
                                transcript: Optional[str] = None) -> LipSyncData:
        """Use Rhubarb Lip Sync for accurate phoneme detection"""
        transcript_path = None
        try:
            cmd = [self._rhubarb_path, audio_path, '-f', 'json']

            # Add transcript for better accuracy
            if transcript:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(transcript)
                    transcript_path = f.name
                cmd.extend(['-d', transcript_path])

            logger.debug(f"Running Rhubarb: {' '.join(cmd)}")

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                result.communicate(),
                timeout=60.0  # 60 second timeout
            )

            if result.returncode != 0:
                logger.error(f"Rhubarb failed: {stderr.decode()}")
                return await self._generate_amplitude(audio_path)

            # Parse JSON output (skip progress lines)
            output = stdout.decode()
            json_start = output.find('{')
            if json_start >= 0:
                data = json.loads(output[json_start:])
            else:
                logger.error("No JSON output from Rhubarb")
                return await self._generate_amplitude(audio_path)

            visemes = []
            for cue in data.get('mouthCues', []):
                start = cue['start']
                end = cue['end']
                shape = cue['value']

                # Keep Rhubarb shapes (A-X) directly - avatar controller will map them
                # Also provide intensity based on mouth openness
                intensity_map = {
                    'X': 0.0,   # Silence
                    'A': 0.1,   # Closed (M, B, P)
                    'B': 0.3,   # Slightly open
                    'C': 0.6,   # Open (E, EH)
                    'D': 0.9,   # Wide open (AA)
                    'E': 0.5,   # Round (OH)
                    'F': 0.4,   # Pucker (OO)
                    'G': 0.3,   # Teeth (F, V)
                    'H': 0.5,   # Tongue (L, TH)
                }

                visemes.append(VisemeEvent(
                    viseme=shape,  # Keep original Rhubarb shape (A-X)
                    start_time=start,
                    end_time=end,
                    intensity=intensity_map.get(shape, 0.5)
                ))

            logger.info(f"Rhubarb generated {len(visemes)} visemes for {data.get('metadata', {}).get('duration', 0):.1f}s audio")

            return LipSyncData(
                duration=data.get('metadata', {}).get('duration', 0),
                visemes=visemes,
                audio_path=audio_path
            )

        except asyncio.TimeoutError:
            logger.error("Rhubarb timed out")
            return await self._generate_amplitude(audio_path)
        except Exception as e:
            logger.error(f"Rhubarb processing failed: {e}")
            return await self._generate_amplitude(audio_path)
        finally:
            # Cleanup transcript file
            if transcript_path:
                try:
                    os.unlink(transcript_path)
                except OSError:
                    pass

    async def _generate_amplitude(self, audio_path: str) -> LipSyncData:
        """Generate lip sync from audio amplitude (simpler fallback)"""
        if not HAS_SOUNDFILE:
            logger.error("soundfile not installed for amplitude analysis")
            return LipSyncData(duration=0, visemes=[])

        try:
            # Load audio
            audio, sample_rate = sf.read(audio_path)

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            duration = len(audio) / sample_rate

            # Analyze amplitude in chunks
            chunk_size = int(sample_rate * 0.05)  # 50ms chunks
            visemes = []

            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                if len(chunk) == 0:
                    continue

                # Calculate RMS amplitude
                rms = np.sqrt(np.mean(chunk ** 2))

                # Normalize to 0-1 range (assuming typical speech levels)
                intensity = min(rms * 10, 1.0)

                start_time = i / sample_rate
                end_time = min((i + chunk_size) / sample_rate, duration)

                # Map amplitude to viseme
                if intensity < 0.1:
                    viseme = 'sil'
                elif intensity < 0.3:
                    viseme = 'PP'
                elif intensity < 0.5:
                    viseme = 'E'
                elif intensity < 0.7:
                    viseme = 'aa'
                else:
                    viseme = 'oh'

                visemes.append(VisemeEvent(
                    viseme=viseme,
                    start_time=start_time,
                    end_time=end_time,
                    intensity=intensity
                ))

            return LipSyncData(
                duration=duration,
                visemes=visemes,
                audio_path=audio_path
            )

        except Exception as e:
            logger.error(f"Amplitude analysis failed: {e}")
            return LipSyncData(duration=0, visemes=[])

    async def generate_from_text(self, text: str,
                                 words_per_minute: float = 150) -> LipSyncData:
        """Generate lip sync from text using phoneme approximation"""
        # Calculate timing
        words = text.split()
        total_duration = len(words) / (words_per_minute / 60)
        char_duration = total_duration / max(len(text), 1)

        visemes = []
        current_time = 0.0

        for char in text.lower():
            if char in CHAR_TO_PHONEME:
                phoneme = CHAR_TO_PHONEME[char]
                viseme = PHONEME_TO_VISEME.get(phoneme, 'sil')
                duration = char_duration * (2 if char in 'aeiou' else 1)
            elif char == ' ':
                viseme = 'sil'
                duration = char_duration * 0.5
            else:
                viseme = 'sil'
                duration = char_duration * 0.3

            visemes.append(VisemeEvent(
                viseme=viseme,
                start_time=current_time,
                end_time=current_time + duration,
                intensity=1.0 if viseme != 'sil' else 0.0
            ))

            current_time += duration

        # Merge consecutive same visemes
        merged = []
        for v in visemes:
            if merged and merged[-1].viseme == v.viseme:
                merged[-1] = VisemeEvent(
                    viseme=v.viseme,
                    start_time=merged[-1].start_time,
                    end_time=v.end_time,
                    intensity=max(merged[-1].intensity, v.intensity)
                )
            else:
                merged.append(v)

        return LipSyncData(duration=current_time, visemes=merged)

    async def generate_from_timestamps(self, text: str,
                                       word_timestamps: List[Dict]) -> LipSyncData:
        """Generate lip sync from TTS word timestamps (ElevenLabs style)"""
        visemes = []

        for word_data in word_timestamps:
            word = word_data.get('word', '')
            start = word_data.get('start', 0)
            end = word_data.get('end', start + 0.1)

            # Calculate per-character timing
            if len(word) > 0:
                char_duration = (end - start) / len(word)
                current_time = start

                for char in word.lower():
                    if char in CHAR_TO_PHONEME:
                        phoneme = CHAR_TO_PHONEME[char]
                        viseme = PHONEME_TO_VISEME.get(phoneme, 'sil')
                    else:
                        viseme = 'sil'

                    visemes.append(VisemeEvent(
                        viseme=viseme,
                        start_time=current_time,
                        end_time=current_time + char_duration,
                        intensity=1.0 if viseme != 'sil' else 0.0
                    ))

                    current_time += char_duration

            # Add silence between words
            if word_data != word_timestamps[-1]:
                next_start = word_timestamps[word_timestamps.index(word_data) + 1].get('start', end)
                if next_start > end:
                    visemes.append(VisemeEvent(
                        viseme='sil',
                        start_time=end,
                        end_time=next_start,
                        intensity=0.0
                    ))

        total_duration = word_timestamps[-1].get('end', 0) if word_timestamps else 0

        return LipSyncData(duration=total_duration, visemes=visemes)

    async def stream_visemes(self, lip_sync_data: LipSyncData,
                            start_time: float = 0.0) -> AsyncGenerator[Tuple[str, float], None]:
        """Stream visemes in real-time for playback synchronization"""
        if not lip_sync_data.visemes:
            yield 'sil', 0.0
            return

        playback_start = time.time() - start_time

        for viseme in lip_sync_data.visemes:
            # Wait until viseme start time
            target_time = playback_start + viseme.start_time
            now = time.time()

            if target_time > now:
                await asyncio.sleep(target_time - now)

            # Yield viseme
            yield viseme.viseme, viseme.intensity

            self._current_viseme = viseme.viseme
            self._current_intensity = viseme.intensity

        # End with silence
        yield 'sil', 0.0
        self._current_viseme = 'sil'
        self._current_intensity = 0.0

    def process_audio_chunk(self, audio_chunk: np.ndarray,
                           sample_rate: int = 16000) -> Tuple[str, float]:
        """Process a chunk of audio for real-time lip sync"""
        if len(audio_chunk) == 0:
            return 'sil', 0.0

        # Calculate RMS amplitude
        rms = np.sqrt(np.mean(audio_chunk.astype(float) ** 2))

        # Normalize (assuming 16-bit audio)
        intensity = min(rms / 5000, 1.0)

        # Map to viseme based on intensity and zero-crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_chunk)))) / 2
        zcr = zero_crossings / len(audio_chunk)

        if intensity < 0.05:
            viseme = 'sil'
            intensity = 0.0
        elif zcr > 0.2:
            # High ZCR suggests fricatives (s, f, sh)
            viseme = 'SS' if intensity < 0.3 else 'FF'
        elif intensity < 0.3:
            viseme = 'PP'
        elif intensity < 0.5:
            viseme = 'E'
        elif intensity < 0.7:
            viseme = 'aa'
        else:
            viseme = 'oh'

        self._current_viseme = viseme
        self._current_intensity = intensity

        return viseme, intensity

    @property
    def current_viseme(self) -> str:
        return self._current_viseme

    @property
    def current_intensity(self) -> float:
        return self._current_intensity
