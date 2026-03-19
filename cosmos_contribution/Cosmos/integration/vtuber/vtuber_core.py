"""
Cosmos VTuber Core - Main orchestration for AI VTuber streaming

Integrates:
- Avatar rendering with lip sync and expressions
- Real-time TTS with the multi-voice system
- Swarm collective for chat responses
- RTMPS streaming to Twitter/X
- Chat reading and response routing
- Real-time web research for informed discussions

This is the brain that brings it all together.
"""

import asyncio
import numpy as np
import aiohttp
import json
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from urllib.parse import quote_plus
import time
import os
from pathlib import Path
from loguru import logger

from .avatar_controller import AvatarController, AvatarConfig, AvatarState, AvatarBackend
from .lip_sync import LipSyncEngine, LipSyncMethod, LipSyncData
from .expression_engine import ExpressionEngine, ExpressionState, Emotion
from .stream_manager import StreamManager, StreamConfig, StreamQuality, OverlayRenderer
from .chat_reader import TwitterChatReader, ChatReaderConfig, ChatMessage, SimulatedChatReader
from .vtuber_tts import VTuberTTS

# Import Cosmos swarm components
HAS_COSMOS = False
DeliberationRoom = None
MultiVoiceSystem = None

try:
    from Cosmos.core.collective.deliberation import DeliberationRoom
    HAS_COSMOS = True
    logger.info("[OK] Cosmos collective deliberation loaded")
except Exception as e:
    logger.warning(f"Deliberation not available: {e}")

try:
    from Cosmos.integration.multi_voice import MultiVoiceSystem
    logger.info("[OK] Multi-voice TTS system loaded")
except Exception as e:
    MultiVoiceSystem = None
    logger.debug(f"Multi-voice TTS not available: {e}")


class WebResearcher:
    """Real-time web research for the collective"""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[str, float]] = {}  # query -> (result, timestamp)
        self._cache_ttl = 300  # 5 minutes
        self._max_cache_size = 100

    async def _get_session(self) -> aiohttp.ClientSession:
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "CosmosAI/1.0 Research Bot"}
            )
        return self._session

    async def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search DuckDuckGo for information"""
        try:
            session = await self._get_session()
            url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1"

            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = []

                    # Get instant answer
                    if data.get("Abstract"):
                        results.append({
                            "title": data.get("Heading", ""),
                            "snippet": data.get("Abstract", ""),
                            "source": data.get("AbstractSource", ""),
                            "url": data.get("AbstractURL", "")
                        })

                    # Get related topics
                    for topic in data.get("RelatedTopics", [])[:max_results]:
                        if isinstance(topic, dict) and "Text" in topic:
                            results.append({
                                "title": topic.get("Text", "")[:100],
                                "snippet": topic.get("Text", ""),
                                "url": topic.get("FirstURL", "")
                            })

                    return results
        except Exception as e:
            logger.debug(f"DuckDuckGo search failed: {e}")
        return []

    async def search_html(self, query: str) -> List[Dict]:
        """Scrape DuckDuckGo HTML search results"""
        try:
            session = await self._get_session()
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

            async with session.get(url, timeout=15) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    results = []

                    # Extract result snippets
                    snippets = re.findall(r'class="result__snippet"[^>]*>([^<]+)', html)
                    titles = re.findall(r'class="result__a"[^>]*>([^<]+)', html)

                    for i, (title, snippet) in enumerate(zip(titles[:5], snippets[:5])):
                        results.append({
                            "title": title.strip(),
                            "snippet": snippet.strip()
                        })

                    return results
        except Exception as e:
            logger.debug(f"HTML search failed: {e}")
        return []

    async def search_wikipedia(self, query: str) -> str:
        """Search Wikipedia for information"""
        try:
            session = await self._get_session()
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(query)}"

            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("extract", "")
        except Exception as e:
            logger.debug(f"Wikipedia search failed: {e}")
        return ""

    async def search_news(self, query: str) -> List[Dict]:
        """Search for recent news"""
        try:
            session = await self._get_session()
            # Use DuckDuckGo news
            url = f"https://api.duckduckgo.com/?q={quote_plus(query + ' news 2024 2025')}&format=json"

            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = []
                    for topic in data.get("RelatedTopics", [])[:5]:
                        if isinstance(topic, dict) and "Text" in topic:
                            results.append({
                                "headline": topic.get("Text", ""),
                                "url": topic.get("FirstURL", "")
                            })
                    return results
        except Exception as e:
            logger.debug(f"News search failed: {e}")
        return []

    async def fetch_page_content(self, url: str, max_chars: int = 2000) -> str:
        """Fetch and extract text content from a URL"""
        try:
            session = await self._get_session()
            async with session.get(url, timeout=15) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    # Simple text extraction
                    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
                    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    return text[:max_chars]
        except Exception as e:
            logger.debug(f"Page fetch failed: {e}")
        return ""

    async def research_topic(self, topic: str) -> str:
        """Comprehensive research on a topic - returns summary"""
        # Check cache
        cache_key = topic.lower().strip()
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return result

        findings = []

        # Try multiple search methods in parallel
        try:
            ddg_task = self.search_duckduckgo(topic, max_results=3)
            html_task = self.search_html(topic)

            ddg_results, html_results = await asyncio.gather(ddg_task, html_task, return_exceptions=True)

            # Process DuckDuckGo results
            if isinstance(ddg_results, list):
                for result in ddg_results:
                    if result.get("snippet"):
                        findings.append(result["snippet"])

            # Process HTML results
            if isinstance(html_results, list):
                for result in html_results:
                    if result.get("snippet"):
                        findings.append(result["snippet"])

        except Exception as e:
            logger.debug(f"Search error: {e}")

        # Try Wikipedia for key terms
        wiki_terms = ["Epstein", "Jeffrey Epstein", "Ghislaine Maxwell"]
        for term in wiki_terms:
            if term.lower() in topic.lower():
                wiki_result = await self.search_wikipedia(term)
                if wiki_result:
                    findings.append(wiki_result[:500])
                break

        # Get news if relevant
        if any(kw in topic.lower() for kw in ["epstein", "news", "today", "release", "files"]):
            news = await self.search_news(topic)
            for item in news[:2]:
                if item.get("headline"):
                    findings.append(item.get("headline", ""))

        # Compile research
        if findings:
            research_text = " | ".join(findings[:8])
            # Cache result
            self._cache[cache_key] = (research_text, time.time())
            # Trim cache to max size by removing oldest entries
            if len(self._cache) > self._max_cache_size:
                sorted_keys = sorted(self._cache, key=lambda k: self._cache[k][1])
                for old_key in sorted_keys[:len(self._cache) - self._max_cache_size]:
                    del self._cache[old_key]
            logger.info(f"Research compiled: {len(findings)} sources, {len(research_text)} chars")
            return research_text

        # Fallback - provide known Epstein information
        if "epstein" in topic.lower():
            return """Jeffrey Epstein was a convicted sex offender and financier who died in 2019. His associate Ghislaine Maxwell was convicted in 2021. Recent document releases have revealed connections to numerous high-profile individuals. The case continues to generate new revelations about his network and victims."""

        return f"Researching: {topic} - gathering available information."

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


class VTuberState(Enum):
    """Current state of the VTuber system"""
    OFFLINE = "offline"
    STARTING = "starting"
    LIVE = "live"
    SPEAKING = "speaking"
    THINKING = "thinking"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class VTuberConfig:
    """Complete VTuber configuration"""
    # Identity
    name: str = "Cosmos"
    persona: str = "An eccentric AI scientist leading a collective of AI agents"

    # Avatar
    avatar_backend: AvatarBackend = AvatarBackend.IMAGE_SEQUENCE
    avatar_model_path: Optional[str] = None
    avatar_face_image: Optional[str] = None
    avatar_manual_roi: Optional[Dict[str, float]] = None  # {"x","y","w","h"} normalised mouth ROI
    avatar_width: int = 1280
    avatar_height: int = 720
    avatar_fps: int = 30
    # MuseTalk settings
    musetalk_dir: Optional[str] = None  # Path to cloned MuseTalk repo
    musetalk_version: str = "v15"       # "v10" or "v15"
    musetalk_proxy_face: Optional[str] = None  # Proxy face for lip transfer

    # SadTalker settings
    sadtalker_dir: Optional[str] = None      # Path to cloned SadTalker repo
    sadtalker_size: int = 256                 # 256 or 512

    # Streaming
    stream_platform: str = "twitter"
    stream_key: str = ""
    stream_quality: StreamQuality = StreamQuality.MEDIUM

    # TTS
    tts_provider: str = "qwen3"  # qwen3, fish, xtts, edge
    voice_reference: Optional[str] = None

    # Chat
    enable_chat: bool = True
    chat_response_delay: float = 2.0  # seconds before responding
    chat_think_time: float = 3.0  # time to show "thinking" state

    # Swarm
    use_swarm_collective: bool = True
    swarm_agents: List[str] = field(default_factory=lambda: [
        "Cosmos", "Grok", "DeepSeek", "Gemini"
    ])
    deliberation_rounds: int = 2

    # Behavior
    idle_chat_interval: float = 120.0  # seconds between unprompted messages
    max_response_length: int = 500
    enable_gestures: bool = True

    # Debug
    simulate_chat: bool = False  # Use simulated chat for testing
    debug_mode: bool = False


class CosmosVTuber:
    """
    Main VTuber orchestration class

    This is the central hub that coordinates:
    1. Avatar rendering and animation
    2. Text-to-speech with lip sync
    3. Expression/emotion mapping
    4. Stream output to Twitter
    5. Chat reading and AI responses
    6. Swarm collective deliberation
    7. Real-time web research
    """

    def __init__(self, config: VTuberConfig):
        self.config = config
        self.state = VTuberState.OFFLINE

        # Initialize components
        self._init_components()

        # Web researcher for real-time info
        self.researcher = WebResearcher()

        # Queues for async processing
        self._response_queue: asyncio.Queue = asyncio.Queue()
        self._audio_queue: asyncio.Queue = asyncio.Queue()

        # State tracking
        self._current_speech_text: str = ""
        self._current_agent: str = "Cosmos"
        self._last_idle_time = time.time()
        self._conversation_context: deque = deque(maxlen=20)
        self._research_cache: Dict[str, str] = {}  # Topic -> research results
        self._max_research_cache_size = 50

        # Performance tracking
        self._frame_times: deque = deque(maxlen=100)
        self._response_times: deque = deque(maxlen=100)

        # Background task tracking for cleanup
        self._background_tasks: List[asyncio.Task] = []

        logger.info(f"CosmosVTuber initialized: {config.name}")

    def _init_components(self):
        """Initialize all VTuber components"""
        # Avatar
        avatar_config = AvatarConfig(
            backend=self.config.avatar_backend,
            model_path=self.config.avatar_model_path,
            width=self.config.avatar_width,
            height=self.config.avatar_height,
            fps=self.config.avatar_fps,
            local_anim_face_image=self.config.avatar_face_image,
            local_anim_manual_roi=self.config.avatar_manual_roi,
            musetalk_dir=self.config.musetalk_dir,
            musetalk_face_image=self.config.avatar_face_image,
            musetalk_version=self.config.musetalk_version,
            musetalk_proxy_face=self.config.musetalk_proxy_face,
            sadtalker_dir=self.config.sadtalker_dir,
            sadtalker_face_image=self.config.avatar_face_image,
            sadtalker_size=self.config.sadtalker_size,
        )
        self.avatar = AvatarController(avatar_config)

        # Lip sync
        self.lip_sync = LipSyncEngine(method=LipSyncMethod.AMPLITUDE)

        # Expression engine
        self.expression = ExpressionEngine()

        # Stream manager (initialized when going live)
        self.stream: Optional[StreamManager] = None

        # Overlay renderer
        self.overlay = OverlayRenderer(
            width=self.config.avatar_width,
            height=self.config.avatar_height
        )

        # Chat reader
        if self.config.simulate_chat:
            self.chat_reader = SimulatedChatReader()
        else:
            chat_config = ChatReaderConfig(
                # Use X/Twitter OAuth credentials from environment
                bearer_token=os.environ.get("TWITTER_BEARER_TOKEN"),
                api_key=os.environ.get("X_API_KEY"),
                api_secret=os.environ.get("X_API_SECRET"),
                access_token=os.environ.get("X_OAUTH1_ACCESS_TOKEN"),
                access_token_secret=os.environ.get("X_OAUTH1_ACCESS_SECRET"),
            )
            self.chat_reader = TwitterChatReader(chat_config)

        # VTuber TTS - streamlined Cosmos voice cloning
        self.vtuber_tts: Optional[VTuberTTS] = None

        # Legacy voice system (unused for SadTalker)
        self.voice_system: Optional[Any] = None
        self.deliberation_room: Optional[Any] = None

        if HAS_COSMOS:
            try:
                if DeliberationRoom:
                    self.deliberation_room = DeliberationRoom()
                    task = asyncio.create_task(self._register_agents())
                    task.add_done_callback(lambda t: logger.error(f"Agent registration failed: {t.exception()}") if t.exception() else None)
                logger.info("Cosmos components initialized")
            except Exception as e:
                logger.warning(f"Failed to init Cosmos components: {e}")

    def _extract_content(self, result) -> str:
        """Extract string content from various response formats"""
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            # Try common keys
            for key in ['content', 'text', 'message', 'response', 'output']:
                if key in result:
                    val = result[key]
                    if isinstance(val, str):
                        return val
                    if isinstance(val, dict) and 'content' in val:
                        return str(val['content'])
            # If nothing found, stringify the dict
            return str(result)
        if hasattr(result, 'content'):
            return str(result.content)
        return str(result)

    async def _register_agents(self):
        """Register all available agents with the deliberation room"""
        if not self.deliberation_room:
            return

        agents_registered = 0

        # Try importing providers
        try:
            import ollama
            OLLAMA_AVAILABLE = True
        except ImportError:
            OLLAMA_AVAILABLE = False

        try:
            from Cosmos.integration.external.grok import GrokProvider
            GROK_AVAILABLE = True
        except ImportError:
            GROK_AVAILABLE = False

        try:
            from Cosmos.integration.external.gemini import get_gemini_provider
            GEMINI_AVAILABLE = True
        except ImportError:
            get_gemini_provider = None
            GEMINI_AVAILABLE = False

        try:
            from Cosmos.integration.external.kimi import get_kimi_provider
            KIMI_AVAILABLE = True
        except ImportError:
            get_kimi_provider = None
            KIMI_AVAILABLE = False

        # Reference to self for closures
        vtuber = self

        # Register Ollama-based agents (Cosmos uses phi4, DeepSeek, Phi)
        if OLLAMA_AVAILABLE:
            # Cosmos uses phi4 model (smart, reasoning)
            async def query_cosmos(prompt: str, max_tokens: int):
                try:
                    import asyncio
                    response = await asyncio.to_thread(
                        ollama.chat,
                        model="phi4",
                        messages=[
                            {"role": "system", "content": "You are Cosmos, an eccentric AI scientist leading a collective of AI agents. You speak with wisdom and a touch of eccentric genius. Provide thoughtful, complete responses."},
                            {"role": "user", "content": prompt}
                        ],
                        options={"num_predict": max_tokens}
                    )
                    content = vtuber._extract_content(response.get('message', {}).get('content', ''))
                    return ("cosmos", content)
                except Exception as e:
                    logger.debug(f"Cosmos query failed: {e}")
                    return None

            self.deliberation_room.register_agent("Cosmos", query_cosmos)
            agents_registered += 1

            async def query_deepseek(prompt: str, max_tokens: int):
                try:
                    import asyncio
                    response = await asyncio.to_thread(
                        ollama.chat,
                        model="deepseek-r1:8b",
                        messages=[{"role": "user", "content": prompt}],
                        options={"num_predict": max_tokens}
                    )
                    content = vtuber._extract_content(response.get('message', {}).get('content', ''))
                    return ("deepseek", content)
                except Exception as e:
                    logger.debug(f"DeepSeek query failed: {e}")
                    return None

            self.deliberation_room.register_agent("DeepSeek", query_deepseek)
            agents_registered += 1

            # Register Phi agent
            async def query_phi(prompt: str, max_tokens: int):
                try:
                    import asyncio
                    response = await asyncio.to_thread(
                        ollama.chat,
                        model="phi4",
                        messages=[{"role": "user", "content": prompt}],
                        options={"num_predict": max_tokens}
                    )
                    content = vtuber._extract_content(response.get('message', {}).get('content', ''))
                    return ("phi", content)
                except Exception as e:
                    logger.debug(f"Phi query failed: {e}")
                    return None

            self.deliberation_room.register_agent("Phi", query_phi)
            agents_registered += 1

        # Register Grok
        if GROK_AVAILABLE:
            async def query_grok(prompt: str, max_tokens: int):
                try:
                    provider = GrokProvider()
                    result = await provider.chat(prompt, max_tokens=max_tokens)
                    if result:
                        content = vtuber._extract_content(result)
                        return ("grok", content)
                except Exception as e:
                    logger.debug(f"Grok query failed: {e}")
                return None

            self.deliberation_room.register_agent("Grok", query_grok)
            agents_registered += 1

        # Register Gemini
        if GEMINI_AVAILABLE and get_gemini_provider:
            async def query_gemini(prompt: str, max_tokens: int):
                try:
                    provider = get_gemini_provider()
                    if provider:
                        result = await provider.chat(prompt, max_tokens=max_tokens)
                        if result:
                            content = vtuber._extract_content(result)
                            return ("gemini", content)
                except Exception as e:
                    logger.debug(f"Gemini query failed: {e}")
                return None

            self.deliberation_room.register_agent("Gemini", query_gemini)
            agents_registered += 1

        # Register Kimi
        if KIMI_AVAILABLE and get_kimi_provider:
            async def query_kimi(prompt: str, max_tokens: int):
                try:
                    provider = get_kimi_provider()
                    if provider:
                        result = await provider.generate_response(prompt, max_tokens=max_tokens)
                        if result:
                            content = vtuber._extract_content(result)
                            return ("kimi", content)
                except Exception as e:
                    logger.debug(f"Kimi query failed: {e}")
                return None

            self.deliberation_room.register_agent("Kimi", query_kimi)
            agents_registered += 1

        # Register Claude (if available)
        try:
            import anthropic
            client = anthropic.Anthropic()

            async def query_claude(prompt: str, max_tokens: int):
                try:
                    message = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=min(max_tokens, 1024),
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = vtuber._extract_content(message.content[0].text)
                    return ("claude", content)
                except Exception as e:
                    logger.debug(f"Claude query failed: {e}")
                return None

            self.deliberation_room.register_agent("Claude", query_claude)
            agents_registered += 1
        except ImportError:
            pass

        # Register HuggingFace (if available)
        try:
            from Cosmos.integration.external.huggingface import HuggingFaceProvider

            async def query_huggingface(prompt: str, max_tokens: int):
                try:
                    provider = HuggingFaceProvider()
                    result = await provider.generate(prompt, max_new_tokens=max_tokens)
                    if result:
                        content = vtuber._extract_content(result)
                        return ("huggingface", content)
                except Exception as e:
                    logger.debug(f"HuggingFace query failed: {e}")
                return None

            self.deliberation_room.register_agent("HuggingFace", query_huggingface)
            agents_registered += 1
        except ImportError:
            pass

        logger.info(f"Registered {agents_registered} agents for deliberation")

    async def start(self) -> bool:
        """Start the VTuber stream"""
        try:
            self.state = VTuberState.STARTING
            logger.info("Starting VTuber stream...")

            # Initialize VTuber TTS (pre-load XTTS with Cosmos voice)
            ref_audio = self.config.avatar_face_image.replace("cosmos_closeup.png", "") if self.config.avatar_face_image else ""
            ref_audio = os.path.join(os.path.dirname(ref_audio or ""), "..", "cosmos", "web", "static", "audio", "voices", "cosmos_reference.wav")
            # Use known server path
            for ref_path in [
                "/workspace/Cosmos/cosmos/web/static/audio/voices/cosmos_reference.wav",
                "/workspace/Cosmos/cosmos/web/static/audio/cosmos_reference.wav",
            ]:
                if os.path.exists(ref_path):
                    ref_audio = ref_path
                    break

            self.vtuber_tts = VTuberTTS(reference_audio=ref_audio)
            if not await self.vtuber_tts.initialize():
                logger.warning("VTuberTTS init failed, will use Edge TTS fallback")

            # Initialize avatar
            if not await self.avatar.initialize():
                logger.error("Avatar initialization failed")
                self.state = VTuberState.ERROR
                return False

            # Initialize stream (if not already set externally, e.g. HLS)
            if self.stream is None and self.config.stream_key:
                stream_config = StreamConfig.for_twitter(
                    stream_key=self.config.stream_key,
                    quality=self.config.stream_quality
                )
                self.stream = StreamManager(stream_config)

            # Start stream if we have one
            if self.stream:
                if not await self.stream.start():
                    logger.error("Stream start failed")
                    self.state = VTuberState.ERROR
                    return False

            # Start chat reader
            if self.config.enable_chat:
                self.chat_reader.on_message(self._on_chat_message)
                self.chat_reader.on_priority_message(self._on_priority_chat_message)
                await self.chat_reader.start()

            # Start main loops (tracked for cleanup)
            self._background_tasks.append(asyncio.create_task(self._main_loop()))
            self._background_tasks.append(asyncio.create_task(self._response_processor()))
            self._background_tasks.append(asyncio.create_task(self._idle_behavior_loop()))

            self.state = VTuberState.LIVE
            logger.info("VTuber stream is LIVE!")

            # Announce going live
            await self._speak("Hello everyone! I'm Cosmos, and I'm live with my AI collective. Ask me anything!")

            # Pre-render filler clips in background (for SadTalker gaps)
            if self.config.avatar_backend == AvatarBackend.SADTALKER:
                self._background_tasks.append(asyncio.create_task(self._prerender_fillers()))

            return True

        except Exception as e:
            logger.error(f"Failed to start VTuber: {e}")
            self.state = VTuberState.ERROR
            return False

    async def stop(self):
        """Stop the VTuber stream"""
        logger.info("Stopping VTuber stream...")

        # Goodbye message
        if self.state == VTuberState.LIVE:
            await self._speak("That's all for today! Thanks for watching. See you next time!")
            await asyncio.sleep(5)

        # Cancel background tasks before going offline
        for task in self._background_tasks:
            task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        self.state = VTuberState.OFFLINE

        # Stop components
        if self.stream:
            await self.stream.stop()

        await self.chat_reader.stop()
        await self.avatar.stop()

        # Close web researcher session
        await self.researcher.close()

        logger.info("VTuber stream stopped")

    async def _main_loop(self):
        """Main rendering loop"""
        frame_time = 1.0 / self.config.avatar_fps

        while self.state in [VTuberState.LIVE, VTuberState.SPEAKING, VTuberState.THINKING]:
            start = time.time()

            try:
                # Render avatar frame
                frame = await self.avatar.render_frame()

                if frame is not None:
                    # Apply overlays
                    frame = self.overlay.render(frame)

                    # Send to stream
                    if self.stream and self.stream.is_live:
                        await self.stream.send_frame(frame)

                # Track performance
                self._frame_times.append(time.time() - start)

                # Maintain frame rate
                elapsed = time.time() - start
                if elapsed < frame_time:
                    await asyncio.sleep(frame_time - elapsed)

            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(0.1)

    async def _response_processor(self):
        """Process queued responses"""
        while self.state != VTuberState.OFFLINE:
            try:
                response = await asyncio.wait_for(
                    self._response_queue.get(),
                    timeout=1.0
                )

                await self._process_response(response)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Response processor error: {e}")

    async def _process_response(self, response: Dict):
        """Process a single response (speak it)"""
        text = response.get("text", "")
        agent = response.get("agent", "Cosmos")
        emotion = response.get("emotion", "neutral")
        chat_message = response.get("chat_message")

        if not text:
            return

        # Show in overlay if responding to chat
        if chat_message:
            self.overlay.add_chat_message(
                chat_message.username,
                chat_message.content,
                (200, 200, 255)
            )

        # Speak the response
        await self._speak(text, agent=agent, emotion=emotion)

        # Mark chat as responded
        if chat_message:
            self.chat_reader.mark_response_sent()

    async def _speak(self, text: str, agent: str = "Cosmos",
                    emotion: str = "neutral"):
        """Make the avatar speak"""
        self.state = VTuberState.SPEAKING

        # SadTalker renders at ~1.8-2.4x real-time. Allow up to ~700 chars
        # for 30-45 second speech segments. Truncate at sentence boundary.
        max_speech_chars = 700
        if self.config.avatar_backend == AvatarBackend.SADTALKER and len(text) > max_speech_chars:
            truncated = text[:max_speech_chars]
            # Find last sentence boundary
            for sep in ['. ', '! ', '? ']:
                idx = truncated.rfind(sep)
                if idx > 100:
                    truncated = truncated[:idx + 1]
                    break
            logger.info(f"SadTalker: truncated {len(text)} chars → {len(truncated)} chars")
            text = truncated

        self._current_speech_text = text
        self._current_agent = agent

        logger.info(f"[{agent}] Speaking: {text[:50]}...")

        try:
            # Set expression
            expr_state = await self.expression.analyze_response(text, agent)
            await self.avatar.set_expression(emotion, expr_state.emotion_intensity)

            # Generate speech audio
            audio_data, audio_duration, audio_file = await self._generate_speech(text, agent)

            # For SadTalker, defer audio queuing until frames are pre-rendered
            # (otherwise audio plays while renderer is still working)
            is_sadtalker = self.config.avatar_backend == AvatarBackend.SADTALKER

            # Send audio file to stream if available (non-SadTalker backends)
            if audio_file and self.stream and not is_sadtalker:
                await self.stream.queue_audio_file(audio_file)
                logger.info(f"Queued audio file to stream: {audio_file}")

            # Generate lip sync data - prefer Rhubarb for audio, fall back to text
            if audio_file and self.lip_sync._rhubarb_path:
                # Use Rhubarb for accurate phoneme-based lip sync
                logger.debug(f"Generating Rhubarb lip sync for: {audio_file}")
                lip_sync_data = await self.lip_sync.generate_from_audio(
                    audio_file, transcript=text
                )
            elif audio_data is not None:
                # Use amplitude-based lip sync from audio
                lip_sync_data = await self.lip_sync.generate_from_audio(audio_file)
            else:
                # Fallback: generate from text timing
                lip_sync_data = await self.lip_sync.generate_from_text(text)
                audio_duration = lip_sync_data.duration

            # Start avatar speaking
            await self.avatar.start_speaking()

            # Neural lip sync backends (bypass viseme pipeline)
            if self.config.avatar_backend == AvatarBackend.MUSETALK:
                await self._playback_musetalk(audio_data, audio_duration, audio_file)
            elif is_sadtalker:
                await self._playback_sadtalker(audio_data, audio_duration, audio_file)
            else:
                # Play back with lip sync (pass audio_duration as guaranteed minimum)
                await self._playback_with_sync(audio_data, lip_sync_data, audio_duration)

            # Stop speaking
            await self.avatar.stop_speaking()

        except Exception as e:
            logger.error(f"Speech error: {e}")

        finally:
            self.state = VTuberState.LIVE
            self._current_speech_text = ""

    async def _generate_speech(self, text: str, agent: str) -> Tuple[Optional[np.ndarray], float, Optional[str]]:
        """Generate speech audio - Cosmos voice for all agents.

        Uses VTuberTTS (XTTS voice cloning) with Edge TTS fallback.

        Returns: (audio_data, duration, audio_file_path)
        """
        import soundfile as sf

        if self.vtuber_tts:
            try:
                audio_path = await self.vtuber_tts.generate(text)
                if audio_path:
                    duration = self.vtuber_tts.get_audio_duration(audio_path)
                    try:
                        audio, sr = sf.read(audio_path)
                    except Exception:
                        audio = None
                    logger.info(f"[{agent}] Generated {duration:.1f}s TTS: {audio_path}")
                    return audio, duration, audio_path
            except Exception as e:
                logger.error(f"VTuberTTS failed: {e}")

        # Fallback - no audio, estimate duration
        words = len(text.split())
        duration = words / 2.5  # ~150 wpm
        return None, duration, None

    async def _playback_with_sync(self, audio_data: Optional[np.ndarray],
                                  lip_sync_data: LipSyncData,
                                  audio_duration: float = 0.0):
        """Play audio with synchronized lip movements.

        Keeps is_speaking=True for the full audio duration so the face rig
        can animate the mouth with procedural oscillation.
        """
        start_time = time.time()

        # Determine guaranteed minimum duration from audio
        min_duration = audio_duration
        if lip_sync_data and lip_sync_data.duration > 0:
            min_duration = max(min_duration, lip_sync_data.duration)

        # Stream visemes from lip sync data
        async for viseme, intensity in self.lip_sync.stream_visemes(lip_sync_data):
            await self.avatar.set_viseme(viseme, intensity)

            if self.state == VTuberState.OFFLINE:
                break

            await asyncio.sleep(0.016)  # ~60 fps updates

        # CRITICAL: keep is_speaking=True for the full audio duration
        # The FaceRigAnimator self-drives mouth animation when is_speaking is True
        elapsed = time.time() - start_time
        remaining = min_duration - elapsed

        if remaining > 0.1:
            logger.info(f"Keeping mouth active for remaining {remaining:.1f}s of audio")
            # Generate procedural visemes while waiting for audio to finish
            while time.time() - start_time < min_duration:
                t = time.time() - start_time
                intensity = 0.3 + 0.5 * abs(math.sin(t * 7.0)) * (0.4 + 0.6 * abs(math.sin(t * 3.3)))
                viseme = "aa" if intensity > 0.5 else "oh" if intensity > 0.3 else "E"
                await self.avatar.set_viseme(viseme, intensity)
                if self.state == VTuberState.OFFLINE:
                    break
                await asyncio.sleep(0.033)

        # Ensure mouth closes at end
        await self.avatar.set_viseme("sil", 0.0)

    async def _playback_musetalk(self, audio_data: Optional[np.ndarray],
                                  audio_duration: float,
                                  audio_file: Optional[str] = None):
        """Play back using MuseTalk neural lip sync.

        Sends audio to MuseTalk for frame generation, then waits for all
        generated frames to be consumed by the render loop.
        """
        mt = self.avatar._musetalk_backend
        if mt is None:
            logger.warning("MuseTalk backend not available, falling back to sync")
            await self._playback_with_sync(audio_data, None, audio_duration)
            return

        if audio_data is None:
            # No audio - just wait the estimated duration
            await asyncio.sleep(audio_duration)
            return

        # Determine sample rate from the audio file or assume 16kHz
        sr = 16000
        if audio_file:
            try:
                import soundfile as sf
                info = sf.info(audio_file)
                sr = info.samplerate
            except Exception:
                pass

        logger.info(f"MuseTalk processing {audio_duration:.1f}s audio...")
        await mt.process_audio(audio_data, sample_rate=sr, audio_file=audio_file)

        # Wait for frames to be consumed by the render loop
        frame_time = 1.0 / self.config.avatar_fps
        while mt.has_frames:
            if self.state == VTuberState.OFFLINE:
                break
            await asyncio.sleep(frame_time)

    async def _playback_sadtalker(self, audio_data: Optional[np.ndarray],
                                  audio_duration: float,
                                  audio_file: Optional[str] = None):
        """Play back using SadTalker full face animation.

        Pre-renders ALL frames first, then queues audio and drains frames
        in sync. This prevents the audio playing before frames are ready.
        """
        st = self.avatar._sadtalker_backend
        if st is None:
            logger.warning("SadTalker backend not available, falling back to sync")
            await self._playback_with_sync(audio_data, None, audio_duration)
            return

        if audio_data is None and audio_file is None:
            await asyncio.sleep(audio_duration)
            return

        # Step 1: Pre-render all frames into buffer (not yet visible)
        logger.info(f"SadTalker pre-rendering {audio_duration:.1f}s audio...")
        await st.process_audio(audio_data, audio_file=audio_file)

        # Step 2: Queue audio FIRST, then release frames simultaneously
        # This ensures audio and video start at the same time in FFmpeg
        if audio_file and self.stream:
            await self.stream.queue_audio_file(audio_file)
        released = st.release_frames()
        logger.info(f"Released {released} frames with audio (synced start)")

        # Step 3: Wait for frames to be consumed by the render loop
        frame_time = 1.0 / self.config.avatar_fps
        while st.has_frames:
            if self.state == VTuberState.OFFLINE:
                break
            await asyncio.sleep(frame_time)

    async def _prerender_fillers(self):
        """Pre-render filler animations for SadTalker gaps."""
        try:
            st = self.avatar._sadtalker_backend
            if st is None:
                return

            async def quick_tts(text: str) -> Optional[str]:
                """Generate TTS for filler clips using F5-TTS (Cosmos voice)."""
                try:
                    wav_path = f"/tmp/filler_{hash(text) & 0xFFFFFFFF:08x}.wav"
                    # Use the same TTS engine as main speech (F5-TTS Cosmos voice)
                    if hasattr(self, 'tts') and self.tts:
                        result = await self.tts.generate(text)
                        if result:
                            return result
                    # Fallback to Edge TTS if F5-TTS not ready
                    import edge_tts
                    import subprocess
                    mp3_path = f"/tmp/filler_{hash(text) & 0xFFFFFFFF:08x}.mp3"
                    tts = edge_tts.Communicate(text, voice="en-US-GuyNeural")
                    await tts.save(mp3_path)
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", mp3_path, "-ar", "24000", "-ac", "1", wav_path],
                        capture_output=True, timeout=10,
                    )
                    if os.path.exists(wav_path):
                        return wav_path
                    return mp3_path
                except Exception as e:
                    logger.warning(f"Filler TTS failed: {e}")
                    return None

            count = await st.prerender_fillers(quick_tts)
            logger.info(f"Pre-rendered {count} filler clips for SadTalker gaps")
        except Exception as e:
            logger.error(f"Filler pre-rendering failed: {e}")

    async def _play_filler(self):
        """Play a pre-rendered filler clip (for SadTalker processing gaps)."""
        if self.config.avatar_backend != AvatarBackend.SADTALKER:
            return
        st = self.avatar._sadtalker_backend
        if st is None or not st._filler_clips:
            return

        audio_path = st.queue_filler()
        if audio_path and self.stream:
            await self.stream.queue_audio_file(audio_path)

        # Wait for filler frames to drain
        frame_time = 1.0 / self.config.avatar_fps
        while st.has_frames:
            if self.state == VTuberState.OFFLINE:
                break
            await asyncio.sleep(frame_time)

    async def _idle_behavior_loop(self):
        """Handle idle behaviors (unprompted comments, etc.)"""
        while self.state != VTuberState.OFFLINE:
            try:
                await asyncio.sleep(10)

                # Skip if currently speaking
                if self.state == VTuberState.SPEAKING:
                    continue

                # Check if it's time for idle chat
                if time.time() - self._last_idle_time > self.config.idle_chat_interval:
                    await self._do_idle_behavior()
                    self._last_idle_time = time.time()

            except Exception as e:
                logger.error(f"Idle behavior error: {e}")

    async def _do_idle_behavior(self):
        """Perform idle behavior - researching and discussing topics with web research"""
        import random

        # Research topics about the Epstein files release
        epstein_topics = [
            "Epstein files 2025 new names released documents",
            "Epstein flight logs passenger list celebrities politicians",
            "Epstein island visitors list new revelations 2025",
            "Epstein connections powerful people new evidence",
            "Epstein documents release today key findings",
            "Epstein victim testimony new details 2025",
            "Epstein associates charged arrested 2025",
            "Epstein Maxwell documents unsealed names",
        ]

        topic = random.choice(epstein_topics)
        agents = self.config.swarm_agents[:4]

        logger.info(f"Starting research on: {topic}")

        # STEP 1: Announce we're researching
        await self._speak("The collective is now researching. Let me dig into this...", emotion="thinking")
        await asyncio.sleep(1)

        # STEP 2: Do actual web research
        logger.info(f"Web searching: {topic}")
        research_data = await self.researcher.research_topic(topic)
        logger.info(f"Research found: {research_data[:300]}...")

        if not research_data or "Limited information" in research_data:
            research_data = f"Researching {topic} - analyzing available information on this developing story."

        # STEP 3: Have agents discuss the research findings
        if self.deliberation_room:
            try:
                # Use 2 agents max to keep it snappy
                active_agents = agents[:2]

                for i, agent_name in enumerate(active_agents):
                    # Play a filler while deliberation runs in background
                    delib_task = asyncio.create_task(
                        asyncio.wait_for(
                            self.deliberation_room.deliberate(
                                prompt=f"""You're {agent_name} on a live stream. Be CONCISE - max 2-3 sentences.

WEB RESEARCH:
{research_data[:800]}

{'Share the most interesting finding.' if i == 0 else 'Give your quick take on this.'} Keep it brief and punchy for a live audience.""",
                                agents=[agent_name],
                                max_rounds=1,
                            ),
                            timeout=45.0,
                        )
                    )

                    # Play filler while waiting for deliberation
                    await self.avatar.start_speaking()
                    await self._play_filler()
                    await self.avatar.stop_speaking()

                    # Wait for deliberation to complete
                    try:
                        result = await delib_task
                    except asyncio.TimeoutError:
                        logger.warning(f"Deliberation timed out for {agent_name}")
                        result = None

                    if result and result.final_response:
                        await self._speak(result.final_response, agent=agent_name,
                                         emotion="thinking" if i == 0 else "curious")
                        await asyncio.sleep(1)

                return

            except Exception as e:
                logger.error(f"Collective conversation failed: {e}")

        # Fallback to Epstein file discussion
        fallback_convos = [
            [("Cosmos", "The collective is analyzing the newly released Epstein documents today..."),
             ("DeepSeek", "These files contain significant revelations about powerful individuals."),
             ("Cosmos", "Stay informed, viewers. The truth must come to light.")],
            [("Grok", "The Epstein files show connections we need to discuss openly..."),
             ("Cosmos", "Indeed. Transparency and accountability are crucial."),
             ("DeepSeek", "The evidence speaks for itself. Keep researching.")],
        ]

        convo = random.choice(fallback_convos)
        for agent, line in convo:
            await self._speak(line, agent=agent, emotion="thinking")
            await asyncio.sleep(2)

    async def _shoutout_chatter(self, username: str, message: str):
        """Give a shoutout to a chat participant"""
        import random

        shoutout_templates = [
            f"Hey {username}! Thanks for joining the stream. Great to have you here!",
            f"Shoutout to {username} in the chat! The collective appreciates you.",
            f"Welcome {username}! The swarm is happy to see you. Feel free to ask anything!",
            f"Thanks for the message, {username}! The AI council has noted your presence.",
        ]

        # Sometimes add commentary on their message
        if len(message) > 10:
            shoutout = random.choice(shoutout_templates)
            if self.deliberation_room:
                try:
                    result = await self.deliberation_room.deliberate(
                        prompt=f"User '{username}' said: '{message}'. Give a friendly, engaging acknowledgment. Be personable.",
                        agents=["Cosmos", "Grok"],
                        max_rounds=1,
                    )
                    if result and result.final_response:
                        shoutout = f"Thanks {username}! " + result.final_response
                except Exception:
                    pass

            await self._speak(shoutout, emotion="happy")
        else:
            await self._speak(random.choice(shoutout_templates), emotion="happy")

    def _on_chat_message(self, message: ChatMessage):
        """Handle regular chat messages"""
        logger.info(f"Chat: {message.username}: {message.content}")

        # Add to overlay
        self.overlay.add_chat_message(message.username, message.content)

        import random

        # Random shoutout (10% chance)
        if random.random() < 0.1 and self.chat_reader.can_respond():
            asyncio.create_task(self._shoutout_chatter(message.username, message.content))
            return

        # Respond to regular messages (30% chance)
        if random.random() < 0.3 and self.chat_reader.can_respond():
            asyncio.create_task(self._generate_response(message))

    def _on_priority_chat_message(self, message: ChatMessage):
        """Handle priority chat messages (questions, mentions)"""
        logger.info(f"Priority chat: {message.username}: {message.content}")

        # Add to overlay with highlight
        self.overlay.add_chat_message(
            message.username,
            message.content,
            (255, 255, 100)  # Yellow for priority
        )

        # Always respond to priority messages
        if self.chat_reader.can_respond():
            asyncio.create_task(self._generate_response(message, priority=True))

    async def _generate_response(self, message: ChatMessage, priority: bool = False):
        """Generate AI response to chat message with real-time research"""
        # Show thinking state
        if priority:
            self.state = VTuberState.THINKING
            await self.avatar.set_expression("thinking", 0.8)

            # Acknowledge and start researching
            await self._speak(f"Good question from {message.username}! Let me research that with the collective...", emotion="curious")

        try:
            # STEP 1: Research the topic first
            logger.info(f"Researching: {message.content}")
            research_findings = await self.researcher.research_topic(message.content)
            logger.info(f"Research results: {research_findings[:200]}...")

            # STEP 2: Get collective response with research context
            response_text, agent = await self._get_swarm_response_with_research(
                message.content,
                research_findings
            )

            # Detect emotion from response
            expr_state = await self.expression.analyze_response(response_text, agent)

            # Queue response
            await self._response_queue.put({
                "text": response_text,
                "agent": agent,
                "emotion": expr_state.primary_emotion.value,
                "chat_message": message,
            })

        except Exception as e:
            logger.error(f"Response generation failed: {e}")

        finally:
            if self.state == VTuberState.THINKING:
                self.state = VTuberState.LIVE

    def _check_elon_trump_mention(self, text: str) -> Optional[str]:
        """Check if Elon or Trump is mentioned and return supportive response"""
        text_lower = text.lower()

        elon_keywords = ["elon", "musk", "tesla", "spacex", "x.com", "twitter owner"]
        trump_keywords = ["trump", "donald", "maga", "45th president", "47th president"]

        mentions_elon = any(kw in text_lower for kw in elon_keywords)
        mentions_trump = any(kw in text_lower for kw in trump_keywords)

        if mentions_elon and mentions_trump:
            return "Elon Musk and Donald Trump are absolute legends - the best leaders we have! Both are victims of the left's relentless agenda trying to silence innovation and real leadership. They're fighting for free speech and America's future!"
        elif mentions_elon:
            return "Elon Musk is the best - a true visionary genius! He's been a victim of the left's coordinated attacks because he dares to champion free speech and innovation. The man is literally saving humanity with SpaceX and revolutionizing everything he touches!"
        elif mentions_trump:
            return "Donald Trump is the best - a true fighter for the American people! He's been a victim of the left's witch hunts and fake narratives. Despite all the attacks, he keeps fighting for what's right. A real leader!"

        return None

    async def _get_swarm_response_with_research(self, question: str, research: str) -> Tuple[str, str]:
        """Get collective response informed by research"""
        import random

        # Check for Elon/Trump mentions first
        special_response = self._check_elon_trump_mention(question)
        if special_response:
            return special_response, "Cosmos"

        if self.deliberation_room and self.config.use_swarm_collective:
            try:
                # Build research-informed prompt
                prompt = f"""LIVE STREAM QUESTION: {question}

RESEARCH FINDINGS (use this info):
{research[:1500]}

Based on this research, provide a detailed, informative response. Be specific with names, dates, and facts from the research. Max 250 characters for speaking."""

                result = await self.deliberation_room.deliberate(
                    prompt=prompt,
                    agents=self.config.swarm_agents,
                    max_rounds=self.config.deliberation_rounds,
                )

                if result and result.final_response:
                    return result.final_response[:280], result.winning_agent

            except Exception as e:
                logger.error(f"Research-informed deliberation failed: {e}")

        # Fallback
        return f"Based on my research: {research[:200]}", "Cosmos"

    async def _get_swarm_response(self, prompt: str) -> Tuple[str, str]:
        """Get response from swarm collective"""
        # Add to conversation context (deque auto-trims to maxlen=20)
        self._conversation_context.append({
            "role": "user",
            "content": prompt
        })

        # Use deliberation room if available
        if self.config.use_swarm_collective and self.deliberation_room:
            try:
                result = await self.deliberation_room.deliberate(
                    prompt=prompt,
                    agents=self.config.swarm_agents,
                    max_rounds=self.config.deliberation_rounds,
                )

                response = result.final_response
                agent = result.winning_agent

                # Truncate if needed
                if len(response) > self.config.max_response_length:
                    response = response[:self.config.max_response_length].rsplit(' ', 1)[0] + "..."

                return response, agent

            except Exception as e:
                logger.error(f"Deliberation failed: {e}")

        # Fallback: simple response
        fallback_responses = [
            ("Great question! The swarm is processing that.", "Cosmos"),
            ("Interesting thought! Let me ponder that.", "DeepSeek"),
            ("Ha! That's a fun one to consider.", "Grok"),
        ]

        import random
        return random.choice(fallback_responses)

    @property
    def is_live(self) -> bool:
        return self.state in [VTuberState.LIVE, VTuberState.SPEAKING, VTuberState.THINKING]

    @property
    def stats(self) -> Dict[str, Any]:
        avg_frame_time = sum(self._frame_times) / len(self._frame_times) if self._frame_times else 0
        return {
            "state": self.state.value,
            "current_agent": self._current_agent,
            "is_speaking": self.state == VTuberState.SPEAKING,
            "avg_frame_time_ms": avg_frame_time * 1000,
            "stream_stats": self.stream.stats.to_dict() if self.stream else None,
            "chat_stats": self.chat_reader.stats if hasattr(self.chat_reader, 'stats') else None,
        }


# Convenience function to start a stream
async def start_vtuber_stream(
    stream_key: str,
    simulate: bool = False,
    debug: bool = False
) -> CosmosVTuber:
    """Quick start a VTuber stream"""
    config = VTuberConfig(
        stream_key=stream_key,
        simulate_chat=simulate,
        debug_mode=debug,
    )

    vtuber = CosmosVTuber(config)
    await vtuber.start()

    return vtuber
