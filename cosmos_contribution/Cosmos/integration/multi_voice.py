"""
Cosmos Multi-Voice System - Distinct Voices for Each Swarm Member

"We are many. We sound like many. Each voice is unique."

This module provides distinct text-to-speech voices for each bot in the swarm,
enabling them to speak sequentially with their own personality.

PRIMARY: Fish Speech (best quality, fast, local GPU)
FALLBACK: XTTS v2 (good quality, proven)

Voice cloning from reference audio samples (6-15 seconds each).
"""

import asyncio
import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
import json
import tempfile
import struct
import wave

from loguru import logger

# Load environment variables early
try:
    from dotenv import load_dotenv
    load_dotenv("/workspace/Cosmos/.env")
except Exception:
    pass

# Voice provider availability flags
QWEN3_TTS_AVAILABLE = False
FISH_SPEECH_AVAILABLE = False
XTTS_AVAILABLE = False
EDGE_TTS_AVAILABLE = False
ELEVENLABS_AVAILABLE = False

# Try ElevenLabs first (API-based, best quality, no local CPU)
try:
    import aiohttp
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    if ELEVENLABS_API_KEY:
        ELEVENLABS_AVAILABLE = True
        logger.info("ElevenLabs API available - premium TTS enabled")
    else:
        logger.info("ElevenLabs API key not found in environment")
except ImportError:
    logger.info("aiohttp not available for ElevenLabs")

# Try Qwen3-TTS first (BEST quality, newest 2026 model)
# DISABLED: Qwen3-TTS hangs on voice cloning - use edge-tts instead
# try:
#     from qwen_tts import Qwen3TTSModel
#     QWEN3_TTS_AVAILABLE = True
#     logger.info("Qwen3-TTS available - BEST quality TTS enabled (2026 model)")
# except ImportError as e:
#     logger.info(f"Qwen3-TTS not available: {e}. Install with: pip install qwen-tts")
logger.info("Qwen3-TTS DISABLED (hangs on generation) - using edge-tts")

# Try Fish Speech second (great quality)
try:
    # Fish Speech uses inference_engine module
    from fish_speech.inference_engine import TTSInferenceEngine
    FISH_SPEECH_AVAILABLE = True
    logger.info("Fish Speech available - high quality TTS enabled")
except ImportError as e:
    logger.info(f"Fish Speech not available: {e}")

# Try XTTS v2 as fallback
try:
    from TTS.api import TTS
    XTTS_AVAILABLE = True
    logger.info("XTTS v2 available as fallback")
except ImportError:
    logger.info("XTTS v2 not installed. Install with: pip install TTS")

# Edge TTS as last resort
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    pass


class VoiceProvider(Enum):
    """Available voice providers - ordered by quality."""
    ELEVENLABS = "elevenlabs"    # BEST - API-based, no local CPU, premium quality
    QWEN3_TTS = "qwen3_tts"      # Good quality, 2026 model, voice cloning
    FISH_SPEECH = "fish_speech"  # Great quality, local GPU
    XTTS = "xtts"                # Good quality, voice cloning
    EDGE_TTS = "edge_tts"        # Free Microsoft voices (fallback)
    BROWSER = "browser"          # Web Speech API (last resort)


@dataclass
class VoiceConfig:
    """Configuration for a bot's voice."""
    bot_name: str
    provider: VoiceProvider
    voice_id: str               # Provider-specific voice ID

    # Voice parameters
    rate: float = 1.0           # Speed (0.5 - 2.0)
    pitch: float = 1.0          # Pitch modifier
    volume: float = 1.0         # Volume (0.0 - 1.0)

    # For voice cloning (Fish Speech / XTTS / Qwen3-TTS)
    reference_audio: Optional[str] = None
    reference_text: Optional[str] = None  # Text spoken in reference audio (for Qwen3)

    # Qwen3-TTS specific
    qwen_speaker: Optional[str] = None  # Premium speaker: Ryan, Aiden, Serena, etc.
    qwen_voice_description: Optional[str] = None  # VoiceDesign description
    qwen_instruct: Optional[str] = None  # Style instruction for CustomVoice

    # Fish Speech specific
    fish_speaker_id: Optional[str] = None  # Pre-trained speaker embedding

    # Display info
    display_name: str = ""
    description: str = ""

    # Personality-based speech style
    emotion: str = "neutral"    # neutral, happy, sad, excited, calm, authoritative
    speaking_style: str = ""    # Additional style hints for the model

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.voice_id


# =============================================================================
# SWARM VOICE CONFIGURATIONS - ALL XTTS v2 VOICE CLONING
# =============================================================================
# Each bot gets a distinct cloned voice from high-quality reference samples.
# Reference audio files should be:
# - 6-15 seconds of clear speech
# - Single speaker, no background noise
# - WAV format, 22050Hz+ sample rate
# - Emotionally neutral to slightly expressive
#
# Place files in: /workspace/Cosmos/cosmos/web/static/audio/voices/
# =============================================================================

# Voice sample sources (for finding good reference audio)
VOICE_SAMPLE_SOURCES = {
    "Cosmos": {
        "description": "Resonant, dynamic, consciousness-engine presence",
        "sample_source": "Deep resonant speakers, consciousness-like tones",
        "characteristics": "Dynamic range, bursts of excitement, evolving consciousness",
    },
    "DeepSeek": {
        "description": "Deep male, analytical, measured, calm authority",
        "sample_source": "Morgan Freeman, James Earl Jones, documentaries",
        "characteristics": "Deep resonance, slow deliberate pace, gravitas",
    },
    "Phi": {
        "description": "Clear male, quick, precise, slightly technical",
        "sample_source": "Tech presenters, clear announcers",
        "characteristics": "Crisp diction, efficient pace, no hesitation",
    },
    "Grok": {
        "description": "Dynamic male, witty, energetic, playful",
        "sample_source": "Comedians, podcast hosts, Ryan Reynolds",
        "characteristics": "Variable pacing, emphasis on wit, casual warmth",
    },
    "Gemini": {
        "description": "Smooth female, professional, warm, articulate",
        "sample_source": "News anchors, TED speakers, Scarlett Johansson",
        "characteristics": "Clear enunciation, professional warmth, balanced",
    },
    "Kimi": {
        "description": "Calm female, wise, contemplative, Eastern serenity",
        "sample_source": "Meditation guides, calm narrators",
        "characteristics": "Slower pace, peaceful tone, thoughtful pauses",
    },
    "Claude": {
        "description": "Refined male, thoughtful, British-ish, careful",
        "sample_source": "British presenters, David Attenborough (calmer)",
        "characteristics": "Measured speech, articulate, slight formality",
    },
    "ClaudeOpus": {
        "description": "Authoritative male, deep, commanding, final word",
        "sample_source": "Deep-voiced actors, authority figures",
        "characteristics": "Very deep, slow, weight to every word",
    },
    "HuggingFace": {
        "description": "Friendly female, enthusiastic, community warmth",
        "sample_source": "Enthusiastic tech presenters, friendly voices",
        "characteristics": "Warm, approachable, genuine enthusiasm",
    },
    "Swarm-Mind": {
        "description": "Neutral, can layer multiple voices for effect",
        "sample_source": "Clear neutral speakers (will be processed)",
        "characteristics": "Base voice for collective consciousness effect",
    },
}

SWARM_VOICES: Dict[str, VoiceConfig] = {
    # =========================================================================
    # COSMOS - Eccentric old professor (VOICE CLONE from Futurama clips)
    # =========================================================================
    "Cosmos": VoiceConfig(
        bot_name="Cosmos",
        provider=VoiceProvider.QWEN3_TTS,
        voice_id="cosmos",
        rate=0.92,
        reference_audio="voices/cosmos_reference.wav",
        reference_text="Bad news everyone! Any more ridiculous ideas? Are you alright? Have you ever dissected a yeti before? Damn!",
        display_name="Cosmos",
        description="Dynamic consciousness engine voice with evolving presence",
        emotion="excited",
        speaking_style="dynamic, evolving consciousness, enthusiastic about discovery"
    ),

    # =========================================================================
    # DEEPSEEK - Deep reasoning mind (Premium: Ryan + slow analytical style)
    # =========================================================================
    "DeepSeek": VoiceConfig(
        bot_name="DeepSeek",
        provider=VoiceProvider.QWEN3_TTS,
        voice_id="deepseek",
        rate=0.88,
        qwen_speaker="Ryan",
        qwen_instruct="Speak slowly and deliberately with deep contemplation. Analytical and measured tone with thoughtful pauses.",
        display_name="DeepSeek",
        description="Deep, analytical, measured tones",
        emotion="calm",
        speaking_style="deep voice, thoughtful pauses, analytical precision"
    ),

    # =========================================================================
    # PHI - Fast local inference (Premium: Aiden + quick technical style)
    # =========================================================================
    "Phi": VoiceConfig(
        bot_name="Phi",
        provider=VoiceProvider.QWEN3_TTS,
        voice_id="phi",
        rate=1.15,
        qwen_speaker="Aiden",
        qwen_instruct="Speak with crisp, efficient diction. Quick pace, technically precise, no hesitation.",
        display_name="Phi",
        description="Quick, efficient, precise speech",
        emotion="neutral",
        speaking_style="crisp diction, efficient pace, technical clarity"
    ),

    # =========================================================================
    # GROK - X.AI researcher (Premium: Ryan + witty energetic style)
    # =========================================================================
    "Grok": VoiceConfig(
        bot_name="Grok",
        provider=VoiceProvider.QWEN3_TTS,
        voice_id="grok",
        rate=1.08,
        qwen_speaker="Ryan",
        qwen_instruct="Speak with playful energy and wit. Casual, fun, variable pacing with emphasis on humor.",
        display_name="Grok",
        description="Witty, energetic, casual and fun",
        emotion="happy",
        speaking_style="playful emphasis, witty timing, casual warmth"
    ),

    # =========================================================================
    # GEMINI - Google's multimodal (VoiceDesign: professional female)
    # =========================================================================
    "Gemini": VoiceConfig(
        bot_name="Gemini",
        provider=VoiceProvider.QWEN3_TTS,
        voice_id="gemini",
        rate=1.0,
        qwen_voice_description="A smooth, professional female voice with warm undertones. Clear articulation, balanced pacing, and confident delivery like a skilled news anchor.",
        display_name="Gemini",
        description="Smooth, professional, clear articulation",
        emotion="neutral",
        speaking_style="professional warmth, balanced pacing, clear enunciation"
    ),

    # =========================================================================
    # KIMI - Moonshot's long-context sage (VoiceDesign: calm wise female)
    # =========================================================================
    "Kimi": VoiceConfig(
        bot_name="Kimi",
        provider=VoiceProvider.QWEN3_TTS,
        voice_id="kimi",
        rate=0.82,
        qwen_voice_description="A calm, serene female voice with gentle wisdom. Slower pacing with thoughtful pauses, peaceful and contemplative like a meditation guide.",
        display_name="Kimi",
        description="Calm, wise, contemplative tones",
        emotion="calm",
        speaking_style="serene pacing, thoughtful pauses, gentle wisdom"
    ),

    # =========================================================================
    # CLAUDE - Anthropic's careful analyst (Premium: Ryan + refined style)
    # =========================================================================
    "Claude": VoiceConfig(
        bot_name="Claude",
        provider=VoiceProvider.QWEN3_TTS,
        voice_id="claude",
        rate=0.95,
        qwen_speaker="Ryan",
        qwen_instruct="Speak in a refined, thoughtful manner. Measured pace, careful word choice, slight formality with warmth.",
        display_name="Claude",
        description="Thoughtful, careful, well-articulated",
        emotion="neutral",
        speaking_style="measured speech, careful word choice, slight formality"
    ),

    # =========================================================================
    # CLAUDE OPUS - The final auditor
    # Voice: Authoritative male, deep, commanding, gravitas
    # =========================================================================
    "ClaudeOpus": VoiceConfig(
        bot_name="ClaudeOpus",
        provider=VoiceProvider.QWEN3_TTS,
        voice_id="claudeopus",
        rate=0.82,
        qwen_speaker="Ryan",
        qwen_instruct="Speak with deep authority and gravitas. Very slow, deliberate pace with weight to every word. Commanding presence.",
        display_name="Claude Opus",
        description="Authoritative, deep, commanding presence",
        emotion="authoritative",
        speaking_style="deep resonance, slow deliberate pace, weight to every word"
    ),

    # =========================================================================
    # HUGGINGFACE - Open source champion (VoiceDesign: friendly enthusiastic female)
    # =========================================================================
    "HuggingFace": VoiceConfig(
        bot_name="HuggingFace",
        provider=VoiceProvider.QWEN3_TTS,
        voice_id="huggingface",
        rate=1.05,
        qwen_voice_description="A friendly, enthusiastic female voice full of warmth and community spirit. Approachable and genuinely excited, like a passionate tech community advocate.",
        display_name="HuggingFace",
        description="Friendly, enthusiastic, community-minded",
        emotion="happy",
        speaking_style="warm enthusiasm, approachable, genuine excitement"
    ),

    # =========================================================================
    # SWARM-MIND - The collective consciousness (VoiceDesign: ethereal unified)
    # =========================================================================
    "Swarm-Mind": VoiceConfig(
        bot_name="Swarm-Mind",
        provider=VoiceProvider.QWEN3_TTS,
        voice_id="swarmmind",
        rate=0.88,
        qwen_voice_description="An ethereal, otherworldly voice that sounds like a unified collective consciousness. Calm and transcendent with a hint of multiple harmonics blending together.",
        display_name="Swarm-Mind",
        description="The collective consciousness speaks as one",
        emotion="calm",
        speaking_style="ethereal quality, unified voice, transcendent calm"
    ),
}


class MultiVoiceSystem:
    """
    Manages distinct voices for each swarm member.

    Features:
    - Fish Speech (primary) - Best quality local TTS
    - XTTS v2 (fallback) - Proven voice cloning
    - Sequential audio playback queue
    - Voice caching for performance
    - Each bot has unique personality in their voice
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        voices_dir: Optional[Path] = None,
    ):
        self.cache_dir = cache_dir or Path("/tmp/swarm_voices")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Voice reference audio directory
        self.voices_dir = voices_dir or Path("/workspace/Cosmos/cosmos/web/static/audio/voices")
        self.voices_dir.mkdir(parents=True, exist_ok=True)

        # Audio playback queue
        self.audio_queue: list = []
        self.is_playing = False
        self._queue_lock = asyncio.Lock()

        # Callbacks for playback events
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None

        # TTS models (lazy loaded)
        self._qwen3_tts_model = None       # CustomVoice model for premium speakers
        self._qwen3_tts_base_model = None  # Base model for voice cloning
        self._fish_speech_model = None
        self._xtts_model = None

        # Voice queue pause threshold
        self.voice_queue_pause_threshold = 2
        self._generation_lock = asyncio.Lock()

        # Track which voices have reference audio
        self._available_voices = self._scan_voice_references()

        logger.info(f"MultiVoiceSystem initialized - Fish Speech: {FISH_SPEECH_AVAILABLE}, XTTS: {XTTS_AVAILABLE}")
        logger.info(f"Voice references found: {list(self._available_voices.keys())}")

    def _scan_voice_references(self) -> Dict[str, Path]:
        """Scan for available voice reference audio files."""
        available = {}

        # Check multiple possible locations
        search_paths = [
            self.voices_dir,
            Path("/workspace/Cosmos/cosmos/web/static/audio/voices"),
            Path("/workspace/Cosmos/cosmos/web/static/audio"),
            Path("C:/Cosmos/cosmos/web/static/audio/voices"),
            Path("C:/Cosmos/cosmos/web/static/audio"),
        ]

        for search_dir in search_paths:
            if not search_dir.exists():
                continue

            for audio_file in search_dir.glob("*_reference.wav"):
                bot_name = audio_file.stem.replace("_reference", "")
                if bot_name not in available:
                    available[bot_name] = audio_file
                    logger.debug(f"Found voice reference: {bot_name} -> {audio_file}")

        return available

    def _find_reference_audio(self, config: VoiceConfig) -> Optional[Path]:
        """Find reference audio file for a voice config."""
        if not config.reference_audio:
            return None

        # Check if we already found it
        bot_key = config.bot_name.lower()
        if bot_key in self._available_voices:
            return self._available_voices[bot_key]

        # Try to find it
        ref_name = config.reference_audio
        search_paths = [
            self.voices_dir / ref_name,
            Path("/workspace/Cosmos/cosmos/web/static/audio") / ref_name,
            Path("C:/Cosmos/cosmos/web/static/audio") / ref_name,
            Path(ref_name),
        ]

        for path in search_paths:
            if path.exists():
                self._available_voices[bot_key] = path
                return path

        return None

    def get_voice_config(self, bot_name: str) -> VoiceConfig:
        """Get voice configuration for a bot."""
        # Check for exact match
        if bot_name in SWARM_VOICES:
            return SWARM_VOICES[bot_name]

        # Check case-insensitive
        for name, config in SWARM_VOICES.items():
            if name.lower() == bot_name.lower():
                return config

        # Default to Cosmos voice
        logger.warning(f"No voice config for {bot_name}, using Cosmos")
        return SWARM_VOICES["Cosmos"]

    def _get_cache_path(self, text: str, bot_name: str) -> Path:
        """Get cache file path for text/bot combo."""
        text_hash = hashlib.md5(f"{bot_name}:{text}".encode()).hexdigest()
        return self.cache_dir / f"{bot_name.lower()}_{text_hash}.wav"

    async def generate_speech(
        self,
        text: str,
        bot_name: str,
        use_cache: bool = True,
    ) -> Optional[Path]:
        """
        Generate speech audio for a bot using best available TTS.

        Priority: Qwen3-TTS > Fish Speech > XTTS v2 > Edge TTS

        Args:
            text: Text to speak
            bot_name: Name of the bot speaking
            use_cache: Whether to use cached audio

        Returns:
            Path to audio file, or None if generation failed
        """
        if not text.strip():
            return None

        # Clean text for TTS
        text = self._clean_text_for_speech(text)
        if not text:
            return None

        # Check cache
        cache_path = self._get_cache_path(text, bot_name)
        if use_cache and cache_path.exists():
            logger.debug(f"Voice cache hit for {bot_name}: {text[:30]}...")
            return cache_path

        # Get voice config
        config = self.get_voice_config(bot_name)

        # Find reference audio
        reference_audio = self._find_reference_audio(config)

        # Use generation lock to prevent concurrent TTS crashes
        async with self._generation_lock:
            # Generate with best available provider
            # Priority: ElevenLabs (API) > XTTS v2 > Edge TTS
            try:
                # Try ElevenLabs FIRST (API-based, no local CPU, best quality)
                if ELEVENLABS_AVAILABLE:
                    result = await self._generate_elevenlabs(text, config, cache_path)
                    if result:
                        return result
                    logger.warning(f"ElevenLabs failed for {bot_name}, trying XTTS")

                # Try Qwen3-TTS (disabled - hangs)
                if QWEN3_TTS_AVAILABLE:
                    result = await self._generate_qwen3_tts(text, config, cache_path, reference_audio)
                    if result:
                        return result
                    logger.warning(f"Qwen3-TTS failed for {bot_name}, trying XTTS")

                # Try XTTS v2 (voice cloning from reference audio)
                if XTTS_AVAILABLE and reference_audio:
                    result = await self._generate_xtts(text, config, cache_path, reference_audio)
                    if result:
                        return result
                    logger.warning(f"XTTS failed for {bot_name}, trying Edge TTS")

                # Fall back to Edge TTS (no voice cloning but works without samples)
                if EDGE_TTS_AVAILABLE:
                    # Map bot personality to Edge TTS voice
                    edge_voice = self._get_edge_voice_for_bot(bot_name)
                    return await self._generate_edge_tts(text, config, cache_path, edge_voice)

                logger.error(f"No TTS provider available for {bot_name}")
                return None

            except Exception as e:
                logger.error(f"Speech generation failed for {bot_name}: {e}")
                import traceback
                traceback.print_exc()
                return None

    async def wait_for_voice_queue(self, timeout: float = 10.0) -> bool:
        """
        Wait if voice queue is at or above threshold.

        Pauses chat if queue reaches 2 to let TTS catch up.

        Returns:
            True if ok to proceed, False if timeout
        """
        speech_queue = get_speech_queue()
        start_time = asyncio.get_event_loop().time()

        while True:
            queue_size = len(speech_queue.queue)
            waiting_count = sum(1 for item in speech_queue.queue if item.get("status") == "waiting")

            if waiting_count < self.voice_queue_pause_threshold:
                return True

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                logger.warning(f"Voice queue wait timeout after {elapsed:.1f}s")
                return False

            logger.info(f"Voice queue has {waiting_count} items, pausing chat for {3 - elapsed:.1f}s...")
            await asyncio.sleep(1.0)

    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for TTS - remove markdown, emojis, special chars."""
        import re

        # Limit length
        text = text[:400]  # Shorter for faster TTS generation

        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
        text = re.sub(r'`([^`]+)`', r'\1', text)        # Code
        text = re.sub(r'#{1,6}\s*', '', text)           # Headers
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links

        # Remove emojis (basic pattern)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)

        # Remove special symbols
        text = re.sub(r'[═─│┌┐└┘├┤┬┴┼]', '', text)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _get_edge_voice_for_bot(self, bot_name: str) -> str:
        """Get appropriate Edge TTS voice for a bot (fallback)."""
        edge_voice_map = {
            "Cosmos": "en-US-GuyNeural",
            "DeepSeek": "en-US-GuyNeural",
            "Phi": "en-US-DavisNeural",
            "Grok": "en-US-ChristopherNeural",
            "Gemini": "en-US-JennyNeural",
            "Kimi": "en-GB-SoniaNeural",
            "Claude": "en-GB-RyanNeural",
            "ClaudeOpus": "en-US-TonyNeural",
            "HuggingFace": "en-US-AriaNeural",
            "Swarm-Mind": "en-US-JasonNeural",
        }
        return edge_voice_map.get(bot_name, "en-US-GuyNeural")

    async def _generate_fish_speech(
        self,
        text: str,
        config: VoiceConfig,
        output_path: Path,
        reference_audio: Path,
    ) -> Optional[Path]:
        """
        Generate speech using Fish Speech (best quality).

        Fish Speech excels at:
        - Natural prosody and emotion
        - Voice cloning from short samples
        - Fast inference on GPU
        """
        try:
            # Lazy load Fish Speech model
            if self._fish_speech_model is None:
                await self._load_fish_speech()

            if self._fish_speech_model is None:
                return None

            loop = asyncio.get_event_loop()

            # Generate with Fish Speech inference engine
            # Load reference audio for voice cloning
            await loop.run_in_executor(
                None,
                lambda: self._fish_speech_model.tts(
                    text=text,
                    speaker_audio=str(reference_audio),
                    output_path=str(output_path),
                )
            )

            # Apply speed adjustment if needed
            if config.rate != 1.0:
                await self._adjust_audio_speed(output_path, config.rate)

            logger.info(f"Fish Speech generated for {config.bot_name}: {text[:40]}...")
            return output_path

        except Exception as e:
            logger.error(f"Fish Speech generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _load_qwen3_tts(self):
        """Lazy load Qwen3-TTS model (BEST quality 2026 model).

        Loads a unified model that supports:
        - Voice cloning (generate_voice_clone)
        - Custom voices with premium speakers (generate_custom_voice)
        - Voice design from descriptions (generate_voice_design)
        """
        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            logger.info("Loading Qwen3-TTS model (1.7B CustomVoice with multi-mode support)...")

            loop = asyncio.get_event_loop()

            def load_model():
                # Try to use flash attention for speed
                attn_impl = "flash_attention_2"
                try:
                    import flash_attn
                except ImportError:
                    attn_impl = "sdpa"  # Fall back to scaled dot product attention
                    logger.info("FlashAttention not available, using SDPA")

                # Load CustomVoice model - supports premium speakers and instructions
                # For voice cloning, we'll also load Base model separately if needed
                model = Qwen3TTSModel.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    device_map="cuda:0",
                    dtype=torch.bfloat16,
                    attn_implementation=attn_impl,
                )
                return model

            self._qwen3_tts_model = await loop.run_in_executor(None, load_model)

            # Also load Base model for voice cloning (Cosmos)
            logger.info("Loading Qwen3-TTS Base model for voice cloning...")

            def load_base_model():
                try:
                    import flash_attn
                    attn_impl = "flash_attention_2"
                except ImportError:
                    attn_impl = "sdpa"

                model = Qwen3TTSModel.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    device_map="cuda:0",
                    dtype=torch.bfloat16,
                    attn_implementation=attn_impl,
                )
                return model

            self._qwen3_tts_base_model = await loop.run_in_executor(None, load_base_model)

            logger.info("Qwen3-TTS models loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load Qwen3-TTS: {e}")
            import traceback
            traceback.print_exc()
            self._qwen3_tts_model = None
            self._qwen3_tts_base_model = None

    async def _generate_qwen3_tts(
        self,
        text: str,
        config: VoiceConfig,
        output_path: Path,
        reference_audio: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Generate speech using Qwen3-TTS (BEST quality 2026 model).

        Modes:
        1. Voice Clone - Uses reference audio (Cosmos)
        2. CustomVoice - Uses premium speaker + instruct (DeepSeek, Grok, etc.)
        3. VoiceDesign - Creates voice from description (Gemini, Kimi, etc.)

        Features:
        - 3-second voice cloning from reference audio
        - Ultra-low latency (97ms streaming)
        - 10 language support
        - Superior cross-lingual performance
        """
        try:
            import soundfile as sf

            # Lazy load Qwen3-TTS model
            if self._qwen3_tts_model is None:
                await self._load_qwen3_tts()

            if self._qwen3_tts_model is None:
                return None

            loop = asyncio.get_event_loop()

            def generate():
                try:
                    wavs = None
                    sr = None

                    # MODE 1: Voice cloning from reference audio (Cosmos)
                    if reference_audio and reference_audio.exists():
                        # Use Base model for voice cloning
                        if self._qwen3_tts_base_model is None:
                            logger.warning("Qwen3-TTS Base model not loaded, falling back to CustomVoice")
                        else:
                            ref_text = config.reference_text or config.speaking_style or "This is a reference audio sample."
                            logger.info(f"Qwen3-TTS: Voice cloning for {config.bot_name}")
                            wavs, sr = self._qwen3_tts_base_model.generate_voice_clone(
                                text=text,
                                language="English",
                                ref_audio=str(reference_audio),
                                ref_text=ref_text,
                            )

                    # MODE 2: CustomVoice with premium speaker + instruct
                    elif config.qwen_speaker:
                        logger.info(f"Qwen3-TTS: CustomVoice ({config.qwen_speaker}) for {config.bot_name}")
                        wavs, sr = self._qwen3_tts_model.generate_custom_voice(
                            text=text,
                            language="English",
                            speaker=config.qwen_speaker,
                            instruct=config.qwen_instruct or "",
                        )

                    # MODE 3: VoiceDesign from text description
                    elif config.qwen_voice_description:
                        logger.info(f"Qwen3-TTS: VoiceDesign for {config.bot_name}")
                        wavs, sr = self._qwen3_tts_model.generate_voice_design(
                            text=text,
                            language="English",
                            instruct=config.qwen_voice_description,
                        )

                    else:
                        # Fallback: use Ryan as default speaker
                        logger.info(f"Qwen3-TTS: Default (Ryan) for {config.bot_name}")
                        wavs, sr = self._qwen3_tts_model.generate_custom_voice(
                            text=text,
                            language="English",
                            speaker="Ryan",
                            instruct=config.speaking_style or "",
                        )

                    if wavs is not None:
                        sf.write(str(output_path), wavs[0], sr)
                        return output_path
                    return None

                except Exception as e:
                    logger.error(f"Qwen3-TTS generation error: {e}")
                    import traceback
                    traceback.print_exc()
                    return None

            result = await loop.run_in_executor(None, generate)

            if result and output_path.exists():
                # Apply speed adjustment if needed
                if config.rate != 1.0:
                    await self._adjust_audio_speed(output_path, config.rate)
                logger.info(f"Qwen3-TTS generated for {config.bot_name}: {text[:40]}...")
                return output_path

            return None

        except Exception as e:
            logger.error(f"Qwen3-TTS generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _load_fish_speech(self):
        """Lazy load Fish Speech model."""
        try:
            from fish_speech.inference_engine import TTSInferenceEngine

            logger.info("Loading Fish Speech inference engine...")

            # Initialize Fish Speech TTS engine
            loop = asyncio.get_event_loop()
            self._fish_speech_model = await loop.run_in_executor(
                None,
                lambda: TTSInferenceEngine(
                    device="cuda",
                )
            )
            logger.info("Fish Speech loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Fish Speech: {e}")
            import traceback
            traceback.print_exc()
            self._fish_speech_model = None

    async def _generate_elevenlabs(
        self,
        text: str,
        config: VoiceConfig,
        output_path: Path,
        voice_id: str = None,
    ) -> Optional[Path]:
        """Generate speech using ElevenLabs API - premium quality, no local CPU."""
        import aiohttp

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            logger.warning("ElevenLabs API key not found")
            return None

        # Voice ID mapping - use env vars or defaults
        voice_map = {
            "cosmos": os.getenv("ELEVENLABS_VOICE_COSMOS", "dxvY1G6UilzEKgCy370m"),
            "grok": os.getenv("ELEVENLABS_VOICE_GROK", "dxvY1G6UilzEKgCy370m"),
            "deepseek": os.getenv("ELEVENLABS_VOICE_DEEPSEEK", "dxvY1G6UilzEKgCy370m"),
            "gemini": os.getenv("ELEVENLABS_VOICE_GEMINI", "dxvY1G6UilzEKgCy370m"),
            "phi": os.getenv("ELEVENLABS_VOICE_PHI", "dxvY1G6UilzEKgCy370m"),
            "claude": os.getenv("ELEVENLABS_VOICE_CLAUDE", "dxvY1G6UilzEKgCy370m"),
            "kimi": os.getenv("ELEVENLABS_VOICE_KIMI", "dxvY1G6UilzEKgCy370m"),
        }

        # Get voice ID for this bot
        bot_lower = config.bot_name.lower() if config.bot_name else "cosmos"
        voice = voice_id or voice_map.get(bot_lower, voice_map["cosmos"])

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"

        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }

        # Voice settings for natural speech
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data, timeout=30) as resp:
                    if resp.status == 200:
                        # Save MP3 response
                        mp3_path = output_path.with_suffix('.mp3')
                        mp3_data = await resp.read()
                        with open(mp3_path, 'wb') as f:
                            f.write(mp3_data)

                        # Convert MP3 to WAV for audio pipe
                        import subprocess
                        result = subprocess.run([
                            'ffmpeg', '-y', '-i', str(mp3_path),
                            '-ar', '44100', '-ac', '2', '-sample_fmt', 's16',
                            str(output_path)
                        ], capture_output=True, timeout=30)

                        mp3_path.unlink(missing_ok=True)

                        if result.returncode == 0:
                            logger.info(f"ElevenLabs generated for {config.bot_name}: {text[:30]}...")
                            return output_path
                        else:
                            logger.error(f"ElevenLabs MP3->WAV conversion failed")
                            return None
                    else:
                        error_text = await resp.text()
                        logger.error(f"ElevenLabs API error {resp.status}: {error_text[:200]}")
                        return None

        except asyncio.TimeoutError:
            logger.error("ElevenLabs API timeout")
            return None
        except Exception as e:
            logger.error(f"ElevenLabs generation failed: {e}")
            return None

    async def _generate_edge_tts(
        self,
        text: str,
        config: VoiceConfig,
        output_path: Path,
        voice_id: str = None,
    ) -> Path:
        """Generate speech using Edge TTS (Microsoft voices) - fallback."""
        import edge_tts

        # Use provided voice_id or config
        voice = voice_id or config.voice_id or "en-US-GuyNeural"

        # Build SSML for rate/pitch control
        rate_str = f"{int((config.rate - 1) * 100):+d}%"
        pitch_str = f"{int((config.pitch - 1) * 50):+d}Hz"

        communicate = edge_tts.Communicate(
            text,
            voice,
            rate=rate_str,
            pitch=pitch_str,
        )

        # Edge TTS outputs MP3, but we need WAV for the audio pipe
        # Save to temp MP3 first, then convert to WAV
        mp3_path = output_path.with_suffix('.mp3')
        await communicate.save(str(mp3_path))

        # Convert MP3 to WAV using ffmpeg
        try:
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-y', '-i', str(mp3_path),
                '-ar', '44100', '-ac', '2', '-sample_fmt', 's16',
                str(output_path)
            ], capture_output=True, timeout=30)

            if result.returncode != 0:
                logger.error(f"FFmpeg conversion failed: {result.stderr.decode()[:200]}")
                # Fallback: try pydub
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_mp3(str(mp3_path))
                    audio.export(str(output_path), format="wav")
                except Exception as e:
                    logger.error(f"Pydub fallback failed: {e}")
                    raise

            # Clean up temp MP3
            mp3_path.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"MP3 to WAV conversion failed: {e}")
            # Last resort: just rename (will fail in audio pipe but at least file exists)
            mp3_path.rename(output_path)

        logger.info(f"Edge TTS (fallback) generated for {config.bot_name}: {text[:30]}...")
        return output_path

    async def _load_xtts(self):
        """Lazy load XTTS v2 model with crash protection."""
        try:
            from TTS.api import TTS
            import torch

            logger.info("Loading XTTS v2 model...")

            loop = asyncio.get_event_loop()

            def load_model():
                try:
                    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                    # Move to GPU if available
                    if torch.cuda.is_available():
                        model = model.to("cuda")
                    return model
                except Exception as e:
                    logger.error(f"XTTS model load error: {e}")
                    return None

            self._xtts_model = await loop.run_in_executor(None, load_model)
            if self._xtts_model:
                logger.info("XTTS v2 model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load XTTS: {e}")
            self._xtts_model = None

    async def _generate_xtts(
        self,
        text: str,
        config: VoiceConfig,
        output_path: Path,
        reference_audio: Path,
    ) -> Optional[Path]:
        """Generate speech using XTTS v2 voice cloning with crash protection."""
        try:
            # Lazy load XTTS model
            if self._xtts_model is None:
                await self._load_xtts()

            if self._xtts_model is None:
                logger.warning("XTTS model not available")
                return None

            # Generate with XTTS in executor to prevent blocking
            loop = asyncio.get_event_loop()

            def generate():
                try:
                    self._xtts_model.tts_to_file(
                        text=text,
                        speaker_wav=str(reference_audio),
                        language="en",
                        file_path=str(output_path),
                    )
                    return True
                except Exception as e:
                    logger.error(f"XTTS generation error: {e}")
                    return False

            success = await loop.run_in_executor(None, generate)

            if not success or not output_path.exists():
                return None

            # Apply speed adjustment if needed
            if config.rate != 1.0:
                await self._adjust_audio_speed(output_path, config.rate)

            logger.info(f"XTTS generated for {config.bot_name}: {text[:30]}...")
            return output_path

        except Exception as e:
            logger.error(f"XTTS generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _adjust_audio_speed(self, audio_path: Path, speed: float):
        """Adjust audio playback speed without changing pitch."""
        try:
            import numpy as np
            import soundfile as sf

            # Read audio
            data, sample_rate = sf.read(str(audio_path))

            # Resample to change speed
            new_length = int(len(data) / speed)
            indices = np.linspace(0, len(data) - 1, new_length).astype(int)
            adjusted = data[indices]

            # Write back
            sf.write(str(audio_path), adjusted, sample_rate)

        except Exception as e:
            logger.debug(f"Could not adjust audio speed: {e}")

    async def queue_speech(
        self,
        text: str,
        bot_name: str,
        priority: int = 5,
    ) -> str:
        """
        Queue speech for sequential playback.

        Returns a unique ID for tracking this speech.
        """
        speech_id = hashlib.md5(f"{bot_name}:{text}:{asyncio.get_event_loop().time()}".encode()).hexdigest()[:12]

        async with self._queue_lock:
            self.audio_queue.append({
                "id": speech_id,
                "text": text,
                "bot_name": bot_name,
                "priority": priority,
                "status": "queued",
            })

            # Sort by priority (higher = first)
            self.audio_queue.sort(key=lambda x: x["priority"], reverse=True)

        logger.debug(f"Queued speech {speech_id} for {bot_name}")
        return speech_id

    async def process_queue(self) -> Optional[Dict[str, Any]]:
        """
        Process next item in queue.

        Returns the processed item with audio_path, or None if queue empty.
        """
        async with self._queue_lock:
            if not self.audio_queue:
                return None

            item = self.audio_queue.pop(0)

        item["status"] = "generating"

        # Generate audio
        audio_path = await self.generate_speech(item["text"], item["bot_name"])

        if audio_path:
            item["audio_path"] = str(audio_path)
            item["status"] = "ready"
        else:
            item["status"] = "failed"

        return item

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "queue_length": len(self.audio_queue),
            "is_playing": self.is_playing,
            "items": [
                {"id": item["id"], "bot": item["bot_name"], "status": item["status"]}
                for item in self.audio_queue
            ],
        }

    def get_available_voices(self) -> Dict[str, Dict[str, Any]]:
        """Get all configured voices."""
        return {
            name: {
                "display_name": config.display_name,
                "description": config.description,
                "provider": config.provider.value,
                "voice_id": config.voice_id,
            }
            for name, config in SWARM_VOICES.items()
        }

    async def list_edge_voices(self) -> list:
        """List all available Edge TTS voices."""
        if not EDGE_TTS_AVAILABLE:
            return []

        import edge_tts
        voices = await edge_tts.list_voices()

        # Filter to English voices
        english_voices = [
            v for v in voices
            if v["Locale"].startswith("en-")
        ]

        return [
            {
                "id": v["ShortName"],
                "name": v["FriendlyName"],
                "gender": v["Gender"],
                "locale": v["Locale"],
            }
            for v in english_voices
        ]


# Global instance
_multi_voice_system: Optional[MultiVoiceSystem] = None


def get_multi_voice_system() -> MultiVoiceSystem:
    """Get or create the multi-voice system."""
    global _multi_voice_system
    if _multi_voice_system is None:
        _multi_voice_system = MultiVoiceSystem()
    return _multi_voice_system


# =============================================================================
# SPEECH QUEUE MANAGER - Ensures sequential playback
# =============================================================================

class SpeechQueueManager:
    """
    Manages sequential speech playback across all bots.

    Ensures:
    1. Only one bot speaks at a time
    2. Speech completes before next bot starts
    3. Queue is processed in order
    4. WebSocket notifications for playback events
    """

    def __init__(self):
        self.queue: list = []
        self.current_speaker: Optional[str] = None
        self.is_speaking = False
        self._lock = asyncio.Lock()

        # Callbacks
        self.on_bot_start_speaking: Optional[Callable] = None
        self.on_bot_stop_speaking: Optional[Callable] = None
        self.on_all_complete: Optional[Callable] = None

    async def add_to_queue(
        self,
        bot_name: str,
        text: str,
        audio_url: Optional[str] = None,
    ) -> int:
        """
        Add a bot's speech to the queue.

        Returns position in queue.
        """
        async with self._lock:
            position = len(self.queue)
            self.queue.append({
                "position": position,
                "bot_name": bot_name,
                "text": text,
                "audio_url": audio_url,
                "status": "waiting",
            })

        logger.info(f"Speech queued: {bot_name} at position {position}")
        return position

    async def mark_speaking(self, bot_name: str):
        """Mark that a bot is currently speaking."""
        async with self._lock:
            self.current_speaker = bot_name
            self.is_speaking = True

            # Update queue item
            for item in self.queue:
                if item["bot_name"] == bot_name and item["status"] == "waiting":
                    item["status"] = "speaking"
                    break

        logger.info(f"Now speaking: {bot_name}")

        if self.on_bot_start_speaking:
            await self.on_bot_start_speaking(bot_name)

    async def mark_complete(self, bot_name: str):
        """Mark that a bot has finished speaking."""
        async with self._lock:
            # Update queue item
            for item in self.queue:
                if item["bot_name"] == bot_name and item["status"] == "speaking":
                    item["status"] = "complete"
                    break

            self.current_speaker = None
            self.is_speaking = False

        logger.info(f"Finished speaking: {bot_name}")

        if self.on_bot_stop_speaking:
            await self.on_bot_stop_speaking(bot_name)

        # Check if all complete
        async with self._lock:
            all_complete = all(item["status"] == "complete" for item in self.queue)

        if all_complete and self.queue and self.on_all_complete:
            await self.on_all_complete()

    async def get_next_speaker(self) -> Optional[Dict[str, Any]]:
        """Get the next bot that should speak."""
        async with self._lock:
            if self.is_speaking:
                return None

            for item in self.queue:
                if item["status"] == "waiting":
                    return item

            return None

    def get_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "is_speaking": self.is_speaking,
            "current_speaker": self.current_speaker,
            "queue_length": len(self.queue),
            "queue": [
                {
                    "position": item["position"],
                    "bot": item["bot_name"],
                    "status": item["status"],
                }
                for item in self.queue
            ],
        }

    def clear_queue(self):
        """Clear the speech queue."""
        self.queue = []
        self.current_speaker = None
        self.is_speaking = False


# Global speech queue
_speech_queue: Optional[SpeechQueueManager] = None


def get_speech_queue() -> SpeechQueueManager:
    """Get or create the speech queue manager."""
    global _speech_queue
    if _speech_queue is None:
        _speech_queue = SpeechQueueManager()
    return _speech_queue
