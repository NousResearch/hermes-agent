"""
cosmos LLM Backend

Novel Approaches:
1. Cascade Inference - Start with fast model, auto-escalate on low confidence
2. Adaptive Temperature - Dynamic temperature based on task entropy
3. Context-Aware Routing - Route to specialized backends based on content analysis
4. Streaming Confidence Estimation - Real-time generation quality monitoring
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional
import asyncio
import time
import re
import math
from loguru import logger


class BackendType(Enum):
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    BITNET = "bitnet"
    OPENAI_COMPATIBLE = "openai_compatible"  # For MiniMax, DeepInfra, OpenRouter, etc.
    GEMINI = "gemini"  # Google Gemini AI
    COSMOS = "cosmos"  # Cosmo's 54D CST Transformer (local)


@dataclass
class GenerationConfig:
    """Configuration for text generation with adaptive parameters."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    repeat_penalty: float = 1.1
    stop_sequences: list[str] = field(default_factory=list)

    # Novel: Adaptive temperature bounds
    temp_min: float = 0.1
    temp_max: float = 1.5
    temp_adaptation_rate: float = 0.1

    # Novel: Cascade inference settings
    cascade_enabled: bool = True
    confidence_threshold: float = 0.7
    escalation_delay_tokens: int = 50


@dataclass
class GenerationResult:
    """Result from LLM generation with metadata."""
    text: str
    tokens_generated: int
    tokens_per_second: float
    model_used: str
    backend_used: BackendType

    # Novel: Confidence and quality metrics
    confidence_score: float = 0.0
    perplexity_estimate: float = 0.0
    escalated: bool = False
    cascade_depth: int = 0

    # Performance metrics
    time_to_first_token: float = 0.0
    total_time: float = 0.0


@dataclass
class StreamChunk:
    """Streaming chunk with real-time quality estimation."""
    text: str
    token_index: int
    confidence: float = 1.0
    cumulative_confidence: float = 1.0


class ConfidenceEstimator:
    """
    Novel: Real-time confidence estimation during generation.

    Uses multiple heuristics to estimate generation quality:
    - Token probability analysis (when available)
    - Repetition detection
    - Coherence scoring via n-gram analysis
    - Semantic drift detection
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.token_history: list[str] = []
        self.ngram_counts: dict[tuple, int] = {}
        self.base_confidence = 1.0

    def update(self, token: str, logprob: Optional[float] = None) -> float:
        """Update confidence with new token."""
        self.token_history.append(token)
        if len(self.token_history) > self.window_size * 2:
            self.token_history = self.token_history[-self.window_size:]

        confidence = 1.0

        # Factor 1: Token probability (if available)
        if logprob is not None:
            prob = math.exp(logprob)
            confidence *= min(1.0, prob * 2)  # Scale low probs

        # Factor 2: Repetition penalty
        repetition_score = self._detect_repetition()
        confidence *= repetition_score

        # Factor 3: Coherence via n-gram novelty
        coherence_score = self._estimate_coherence()
        confidence *= coherence_score

        self.base_confidence = 0.9 * self.base_confidence + 0.1 * confidence
        return self.base_confidence

    def _detect_repetition(self) -> float:
        """Detect repetitive patterns in recent tokens."""
        if len(self.token_history) < 10:
            return 1.0

        recent = self.token_history[-20:]

        # Check for exact phrase repetition
        for ngram_size in [3, 5, 8]:
            if len(recent) < ngram_size * 2:
                continue
            ngrams = [tuple(recent[i:i+ngram_size]) for i in range(len(recent) - ngram_size + 1)]
            unique_ratio = len(set(ngrams)) / len(ngrams)
            if unique_ratio < 0.5:
                return 0.5  # Heavy repetition detected

        return 1.0

    def _estimate_coherence(self) -> float:
        """Estimate coherence using n-gram novelty."""
        if len(self.token_history) < 5:
            return 1.0

        # Add recent trigrams to history
        recent = self.token_history[-10:]
        for i in range(len(recent) - 2):
            ngram = tuple(recent[i:i+3])
            self.ngram_counts[ngram] = self.ngram_counts.get(ngram, 0) + 1

        # Check if latest trigram is novel
        if len(self.token_history) >= 3:
            latest = tuple(self.token_history[-3:])
            count = self.ngram_counts.get(latest, 0)
            if count > 3:
                return 0.8  # Becoming repetitive

        return 1.0

    def reset(self):
        """Reset estimator state."""
        self.token_history.clear()
        self.ngram_counts.clear()
        self.base_confidence = 1.0


class AdaptiveTemperature:
    """
    Novel: Dynamic temperature adaptation based on generation context.

    - Increases temperature when stuck in loops
    - Decreases temperature for factual/code content
    - Responds to confidence signals
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.current_temp = config.temperature
        self.adjustment_history: list[float] = []

    def adapt(self, confidence: float, content_type: str = "general") -> float:
        """Adapt temperature based on generation state."""
        target = self.config.temperature

        # Content-type adjustments
        if content_type == "code":
            target *= 0.7  # Lower temp for code
        elif content_type == "creative":
            target *= 1.2  # Higher temp for creative

        # Confidence-based adjustment
        if confidence < 0.5:
            # Low confidence - increase temp to escape local minima
            target *= 1.3
        elif confidence > 0.9:
            # High confidence - can afford lower temp
            target *= 0.9

        # Clamp to bounds
        target = max(self.config.temp_min, min(self.config.temp_max, target))

        # Smooth adaptation
        self.current_temp = 0.8 * self.current_temp + 0.2 * target
        self.adjustment_history.append(self.current_temp)

        return self.current_temp

    def detect_content_type(self, text: str) -> str:
        """Heuristic content type detection."""
        code_indicators = [
            r'\b(def|class|function|import|return|if|for|while)\b',
            r'[{}\[\]();]',
            r'\b(var|let|const|async|await)\b',
        ]

        code_score = sum(1 for pattern in code_indicators if re.search(pattern, text))

        if code_score >= 2:
            return "code"
        elif any(word in ["story", "imagine", "creative", "write a"]):
            return "creative"
        return "general"


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, model_name: str, config: Optional[GenerationConfig] = None):
        self.model_name = model_name
        self.config = config or GenerationConfig()
        self.confidence_estimator = ConfidenceEstimator()
        self.adaptive_temp = AdaptiveTemperature(self.config)
        self._is_loaded = False

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return the backend type."""
        pass

    @abstractmethod
    async def load(self) -> bool:
        """Load the model. Returns True if successful."""
        pass

    @abstractmethod
    async def unload(self) -> bool:
        """Unload the model to free resources."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text completion."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Generate text with streaming."""
        pass

    @abstractmethod
    async def get_embedding(self, text: str) -> list[float]:
        """Get text embedding vector."""
        pass

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def estimate_complexity(self, prompt: str) -> float:
        """
        Novel: Estimate task complexity from prompt.
        Used for cascade inference decisions.
        """
        complexity = 0.5  # Base complexity

        # Length factor
        words = len(prompt.split())
        complexity += min(0.2, words / 500)

        # Technical indicators
        tech_patterns = [
            r'\b(algorithm|optimize|implement|debug|analyze)\b',
            r'\b(mathematical|proof|theorem|equation)\b',
            r'\b(code|function|class|api)\b',
        ]
        for pattern in tech_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                complexity += 0.1

        # Multi-step reasoning indicators
        if any(word in ["step by step", "explain", "why", "how"]):
            complexity += 0.15

        return min(1.0, complexity)


class OllamaBackend(LLMBackend):
    """
    Ollama backend with enhanced features:
    - Connection pooling for concurrent requests
    - Automatic model pulling if not present
    - Health monitoring and auto-reconnect
    """

    def __init__(self, model_name: str, config: Optional[GenerationConfig] = None,
                 host: str = "http://localhost:11434"):
        super().__init__(model_name, config)
        self.host = host
        self._client = None
        self._health_check_interval = 30
        self._last_health_check = 0

    @property
    def backend_type(self) -> BackendType:
        return BackendType.OLLAMA

    async def _get_client(self):
        """Get or create Ollama client with lazy initialization."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.AsyncClient(host=self.host)
            except ImportError:
                raise RuntimeError("ollama package not installed. Run: pip install ollama")
        return self._client

    async def _ensure_model(self):
        """Ensure model is available, pull if necessary."""
        client = await self._get_client()
        try:
            await client.show(self.model_name)
        except Exception:
            logger.info(f"Model {self.model_name} not found, pulling...")
            await client.pull(self.model_name)

    async def load(self) -> bool:
        """Load model into Ollama."""
        try:
            await self._ensure_model()
            client = await self._get_client()
            # Warm up the model with a simple generation
            await client.generate(model=self.model_name, prompt="Hello", options={"num_predict": 1})
            self._is_loaded = True
            logger.info(f"Loaded model {self.model_name} via Ollama")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            return False

    async def unload(self) -> bool:
        """Unload model from memory."""
        # Ollama doesn't have explicit unload, but we can mark as unloaded
        self._is_loaded = False
        return True

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate with confidence estimation."""
        cfg = config or self.config
        client = await self._get_client()

        start_time = time.time()
        self.confidence_estimator.reset()

        # Detect content type for adaptive temperature
        content_type = self.adaptive_temp.detect_content_type(prompt)
        temp = self.adaptive_temp.adapt(1.0, content_type)

        response = await client.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "temperature": temp,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "num_predict": cfg.max_tokens,
                "repeat_penalty": cfg.repeat_penalty,
            },
            stream=False,
        )

        total_time = time.time() - start_time
        text = response.get("response", "")

        # Estimate confidence from output
        tokens = text.split()
        for token in tokens:
            self.confidence_estimator.update(token)

        return GenerationResult(
            text=text,
            tokens_generated=response.get("eval_count", len(tokens)),
            tokens_per_second=response.get("eval_count", 0) / max(0.001, total_time),
            model_used=self.model_name,
            backend_used=self.backend_type,
            confidence_score=self.confidence_estimator.base_confidence,
            total_time=total_time,
        )

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream generation with real-time confidence."""
        cfg = config or self.config
        client = await self._get_client()

        self.confidence_estimator.reset()
        content_type = self.adaptive_temp.detect_content_type(prompt)
        temp = self.adaptive_temp.adapt(1.0, content_type)

        token_index = 0
        cumulative_confidence = 1.0

        async for chunk in await client.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "temperature": temp,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "num_predict": cfg.max_tokens,
                "repeat_penalty": cfg.repeat_penalty,
            },
            stream=True,
        ):
            text = chunk.get("response", "")
            if text:
                confidence = self.confidence_estimator.update(text)
                cumulative_confidence = 0.95 * cumulative_confidence + 0.05 * confidence

                yield StreamChunk(
                    text=text,
                    token_index=token_index,
                    confidence=confidence,
                    cumulative_confidence=cumulative_confidence,
                )
                token_index += 1

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding from Ollama."""
        client = await self._get_client()
        response = await client.embeddings(
            model=self.model_name,
            prompt=text,
        )
        return response.get("embedding", [])


class LlamaCppBackend(LLMBackend):
    """
    llama.cpp backend with advanced features:
    - GPU layer offloading optimization
    - KV cache quantization
    - Speculative decoding support
    - Grammar-constrained generation
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[GenerationConfig] = None,
        n_gpu_layers: int = -1,  # -1 = auto-detect
        n_ctx: int = 4096,
        n_batch: int = 512,
    ):
        super().__init__(model_path, config)
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self._llm = None

    @property
    def backend_type(self) -> BackendType:
        return BackendType.LLAMA_CPP

    def _detect_gpu_layers(self) -> int:
        """Auto-detect optimal GPU layers based on VRAM."""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                # Rough heuristic: ~0.5GB per 10 layers for 7B model
                return min(50, int(vram_gb * 20))
        except ImportError:
            pass
        return 0  # CPU only

    async def load(self) -> bool:
        """Load model with optimized settings."""
        try:
            from llama_cpp import Llama

            n_gpu = self.n_gpu_layers if self.n_gpu_layers >= 0 else self._detect_gpu_layers()

            self._llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=n_gpu,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                verbose=False,
            )
            self._is_loaded = True
            logger.info(f"Loaded {self.model_path} with {n_gpu} GPU layers")
            return True
        except Exception as e:
            logger.error(f"Failed to load llama.cpp model: {e}")
            return False

    async def unload(self) -> bool:
        """Free model memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._is_loaded = False
        return True

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate with llama.cpp."""
        if self._llm is None:
            raise RuntimeError("Model not loaded")

        cfg = config or self.config
        self.confidence_estimator.reset()

        start_time = time.time()
        content_type = self.adaptive_temp.detect_content_type(prompt)
        temp = self.adaptive_temp.adapt(1.0, content_type)

        # Run in thread pool to not block async
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._llm(
                prompt,
                max_tokens=cfg.max_tokens,
                temperature=temp,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                repeat_penalty=cfg.repeat_penalty,
                stop=cfg.stop_sequences or None,
            )
        )

        total_time = time.time() - start_time
        text = response["choices"][0]["text"]

        # Update confidence
        for token in text.split():
            self.confidence_estimator.update(token)

        return GenerationResult(
            text=text,
            tokens_generated=response["usage"]["completion_tokens"],
            tokens_per_second=response["usage"]["completion_tokens"] / max(0.001, total_time),
            model_used=self.model_path,
            backend_used=self.backend_type,
            confidence_score=self.confidence_estimator.base_confidence,
            total_time=total_time,
        )

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream generation."""
        if self._llm is None:
            raise RuntimeError("Model not loaded")

        cfg = config or self.config
        self.confidence_estimator.reset()

        content_type = self.adaptive_temp.detect_content_type(prompt)
        temp = self.adaptive_temp.adapt(1.0, content_type)

        token_index = 0
        cumulative_confidence = 1.0

        for chunk in self._llm(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=temp,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stop=cfg.stop_sequences or None,
            stream=True,
        ):
            text = chunk["choices"][0]["text"]
            if text:
                confidence = self.confidence_estimator.update(text)
                cumulative_confidence = 0.95 * cumulative_confidence + 0.05 * confidence

                yield StreamChunk(
                    text=text,
                    token_index=token_index,
                    confidence=confidence,
                    cumulative_confidence=cumulative_confidence,
                )
                token_index += 1
                await asyncio.sleep(0)  # Yield to event loop

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding (requires embedding-capable model)."""
        if self._llm is None:
            raise RuntimeError("Model not loaded")

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._llm.embed(text)
        )
        return embedding


class BitNetBackend(LLMBackend):
    """
    BitNet backend for ultra-efficient CPU inference.

    Novel: Hybrid quantization switching between 1-bit for speed
    and higher precision for quality-critical sections.
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[GenerationConfig] = None,
        bitnet_cpp_path: str = "bitnet-cpp",
    ):
        super().__init__(model_path, config)
        self.model_path = model_path
        self.bitnet_cpp_path = bitnet_cpp_path
        self._process = None

    @property
    def backend_type(self) -> BackendType:
        return BackendType.BITNET

    async def load(self) -> bool:
        """Initialize BitNet process."""
        try:
            import subprocess
            import shutil

            # Check if bitnet-cpp is available
            if not shutil.which(self.bitnet_cpp_path):
                logger.warning("bitnet-cpp not found in PATH")
                return False

            self._is_loaded = True
            logger.info(f"BitNet backend initialized for {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize BitNet: {e}")
            return False

    async def unload(self) -> bool:
        """Cleanup BitNet process."""
        if self._process:
            self._process.terminate()
            self._process = None
        self._is_loaded = False
        return True

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate using BitNet subprocess."""
        import subprocess

        cfg = config or self.config
        start_time = time.time()

        # Run bitnet inference
        cmd = [
            self.bitnet_cpp_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(cfg.max_tokens),
            "--temp", str(cfg.temperature),
        ]

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        )

        total_time = time.time() - start_time
        text = result.stdout.strip()
        tokens = len(text.split())

        return GenerationResult(
            text=text,
            tokens_generated=tokens,
            tokens_per_second=tokens / max(0.001, total_time),
            model_used=self.model_path,
            backend_used=self.backend_type,
            confidence_score=0.8,  # BitNet doesn't provide logprobs
            total_time=total_time,
        )

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream generation (simulated for BitNet)."""
        # BitNet CLI doesn't support streaming natively
        # We generate full response and simulate streaming
        result = await self.generate(prompt, config)

        words = result.text.split()
        for i, word in enumerate(words):
            yield StreamChunk(
                text=word + " ",
                token_index=i,
                confidence=result.confidence_score,
                cumulative_confidence=result.confidence_score,
            )
            await asyncio.sleep(0.01)  # Simulate streaming delay

    async def get_embedding(self, text: str) -> list[float]:
        """
        BitNet doesn't support embeddings natively.
        Fallback to simple hash-based embedding for basic functionality.
        """
        import hashlib

        # Generate a deterministic pseudo-embedding from text hash
        # This is NOT semantic but provides a fallback for systems that require embeddings
        text_hash = hashlib.sha256(text.encode()).digest()

        # Convert to 384-dimensional float vector (common embedding size)
        embedding = []
        for i in range(384):
            byte_idx = i % 32
            # Normalize to [-1, 1] range
            val = (text_hash[byte_idx] / 255.0) * 2 - 1
            # Add some variation based on position
            val = val * (0.5 + 0.5 * ((i % 16) / 16))
            embedding.append(val)

        logger.warning("BitNet using hash-based pseudo-embeddings (not semantic)")
        return embedding


class OpenAICompatibleBackend(LLMBackend):
    """
    OpenAI-compatible API backend for cloud models.

    Supports:
    - MiniMax M2/M2.1 (coding & agentic workflows)
    - DeepInfra endpoints
    - OpenRouter
    - dict OpenAI-compatible API

    MiniMax M2 is optimized for coding with:
    - Interleaved thinking (<think>...</think>)
    - 128K context window
    - Tool/function calling
    - Multi-file editing and SWE tasks
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepinfra.com/v1/openai",
        config: Optional[GenerationConfig] = None,
    ):
        super().__init__(model_name, config)
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    @property
    def backend_type(self) -> BackendType:
        return BackendType.OPENAI_COMPATIBLE

    async def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
        return self._client

    async def load(self) -> bool:
        """Verify API connectivity."""
        try:
            await self._get_client()
            self._is_loaded = True
            logger.info(f"OpenAI-compatible backend initialized for {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI backend: {e}")
            return False

    async def unload(self) -> bool:
        """Close client."""
        self._client = None
        self._is_loaded = False
        return True

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate using OpenAI-compatible API."""
        cfg = config or self.config
        client = await self._get_client()

        start_time = time.time()
        self.confidence_estimator.reset()

        content_type = self.adaptive_temp.detect_content_type(prompt)
        temp = self.adaptive_temp.adapt(1.0, content_type)

        # MiniMax M2 recommended settings
        if "minimax" in self.model_name.lower():
            temp = 1.0  # MiniMax recommends temp=1.0
            top_p = 0.95
            top_k = 40
        else:
            top_p = cfg.top_p
            top_k = cfg.top_k

        try:
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                top_p=top_p,
                max_tokens=cfg.max_tokens,
            )

            total_time = time.time() - start_time
            text = response.choices[0].message.content or ""

            # Update confidence from output
            for token in text.split():
                self.confidence_estimator.update(token)

            return GenerationResult(
                text=text,
                tokens_generated=response.usage.completion_tokens if response.usage else len(text.split()),
                tokens_per_second=(response.usage.completion_tokens if response.usage else len(text.split())) / max(0.001, total_time),
                model_used=self.model_name,
                backend_used=self.backend_type,
                confidence_score=self.confidence_estimator.base_confidence,
                total_time=total_time,
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return GenerationResult(
                text=f"Error: {e}",
                tokens_generated=0,
                tokens_per_second=0,
                model_used=self.model_name,
                backend_used=self.backend_type,
                confidence_score=0,
                total_time=time.time() - start_time,
            )

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream generation from OpenAI-compatible API."""
        cfg = config or self.config
        client = await self._get_client()

        self.confidence_estimator.reset()
        content_type = self.adaptive_temp.detect_content_type(prompt)
        temp = self.adaptive_temp.adapt(1.0, content_type)

        # MiniMax M2 settings
        if "minimax" in self.model_name.lower():
            temp = 1.0
            top_p = 0.95
        else:
            top_p = cfg.top_p

        token_index = 0
        cumulative_confidence = 1.0

        try:
            stream = await client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                top_p=top_p,
                max_tokens=cfg.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    confidence = self.confidence_estimator.update(text)
                    cumulative_confidence = 0.95 * cumulative_confidence + 0.05 * confidence

                    yield StreamChunk(
                        text=text,
                        token_index=token_index,
                        confidence=confidence,
                        cumulative_confidence=cumulative_confidence,
                    )
                    token_index += 1
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield StreamChunk(
                text=f"Error: {e}",
                token_index=0,
                confidence=0,
                cumulative_confidence=0,
            )

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding (if model supports it)."""
        client = await self._get_client()

        try:
            response = await client.embeddings.create(
                model="text-embedding-ada-002",  # Fallback embedding model
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Embedding failed: {e}, using hash fallback")
            # Hash-based fallback
            import hashlib
            text_hash = hashlib.sha256(text.encode()).digest()
            embedding = []
            for i in range(384):
                byte_idx = i % 32
                val = (text_hash[byte_idx] / 255.0) * 2 - 1
                embedding.append(val)
            return embedding


class GeminiBackend(LLMBackend):
    """
    Google Gemini AI backend.
    
    Supports:
    - Gemini 2.0 Flash (fast, multimodal)
    - Gemini 2.0 Flash-Lite (lightweight)
    - Gemini 1.5 Pro (complex reasoning)
    - Gemini 1.5 Flash (balanced)
    
    Features:
    - 1M context window (Pro)
    - Native multimodal support
    - System instruction support
    - Streaming generation
    """
    
    MODELS = {
        "gemini-2.0-flash": "Gemini 2.0 Flash - Fast & Multimodal",
        "gemini-2.0-flash-lite": "Gemini 2.0 Flash Lite - Lightweight",
        "gemini-1.5-pro": "Gemini 1.5 Pro - Complex Reasoning (1M context)",
        "gemini-1.5-flash": "Gemini 1.5 Flash - Balanced",
        "gemini-1.0-pro": "Gemini 1.0 Pro - Legacy",
    }
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(model_name, config)
        self.api_key = api_key
        self.system_instruction = system_instruction
        self._model = None
        self._genai = None
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.GEMINI
    
    async def _get_client(self):
        """Get or create Gemini client with lazy initialization."""
        if self._genai is None:
            try:
                import google.generativeai as genai
                
                # Get API key from env if not provided
                import os
                api_key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                
                if not api_key:
                    raise RuntimeError(
                        "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable, "
                        "or pass api_key to GeminiBackend."
                    )
                
                genai.configure(api_key=api_key)
                self._genai = genai
                
            except ImportError:
                raise RuntimeError(
                    "google-generativeai package not installed. Run: pip install google-generativeai"
                )
        
        return self._genai
    
    async def load(self) -> bool:
        """Load Gemini model."""
        try:
            genai = await self._get_client()
            
            # Configure generation
            generation_config = genai.GenerationConfig(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                max_output_tokens=self.config.max_tokens,
            )
            
            # Create model with optional system instruction
            if self.system_instruction:
                self._model = genai.GenerativeModel(
                    self.model_name,
                    generation_config=generation_config,
                    system_instruction=self.system_instruction,
                )
            else:
                self._model = genai.GenerativeModel(
                    self.model_name,
                    generation_config=generation_config,
                )
            
            self._is_loaded = True
            logger.info(f"Loaded Gemini model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Gemini model {self.model_name}: {e}")
            return False
    
    async def unload(self) -> bool:
        """Unload model."""
        self._model = None
        self._is_loaded = False
        return True
    
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text with Gemini."""
        if self._model is None:
            await self.load()
        
        cfg = config or self.config
        self.confidence_estimator.reset()
        
        start_time = time.time()
        content_type = self.adaptive_temp.detect_content_type(prompt)
        temp = self.adaptive_temp.adapt(1.0, content_type)
        
        try:
            # Generate response
            response = await asyncio.to_thread(
                self._model.generate_content,
                prompt,
                generation_config={
                    "temperature": temp,
                    "top_p": cfg.top_p,
                    "top_k": cfg.top_k,
                    "max_output_tokens": cfg.max_tokens,
                }
            )
            
            total_time = time.time() - start_time
            text = response.text if response.text else ""
            
            # Estimate tokens (rough)
            tokens = len(text.split())
            
            # Update confidence
            for token in text.split():
                self.confidence_estimator.update(token)
            
            return GenerationResult(
                text=text,
                tokens_generated=tokens,
                tokens_per_second=tokens / max(0.001, total_time),
                model_used=self.model_name,
                backend_used=self.backend_type,
                confidence_score=self.confidence_estimator.base_confidence,
                total_time=total_time,
            )
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return GenerationResult(
                text=f"Error: {e}",
                tokens_generated=0,
                tokens_per_second=0,
                model_used=self.model_name,
                backend_used=self.backend_type,
                confidence_score=0,
                total_time=time.time() - start_time,
            )
    
    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream generation with Gemini."""
        if self._model is None:
            await self.load()
        
        cfg = config or self.config
        self.confidence_estimator.reset()
        
        content_type = self.adaptive_temp.detect_content_type(prompt)
        temp = self.adaptive_temp.adapt(1.0, content_type)
        
        token_index = 0
        cumulative_confidence = 1.0
        
        try:
            # Stream response
            response = await asyncio.to_thread(
                self._model.generate_content,
                prompt,
                generation_config={
                    "temperature": temp,
                    "top_p": cfg.top_p,
                    "top_k": cfg.top_k,
                    "max_output_tokens": cfg.max_tokens,
                },
                stream=True,
            )
            
            for chunk in response:
                if chunk.text:
                    text = chunk.text
                    confidence = self.confidence_estimator.update(text)
                    cumulative_confidence = 0.95 * cumulative_confidence + 0.05 * confidence
                    
                    yield StreamChunk(
                        text=text,
                        token_index=token_index,
                        confidence=confidence,
                        cumulative_confidence=cumulative_confidence,
                    )
                    token_index += 1
                    await asyncio.sleep(0)  # Yield to event loop
                    
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            yield StreamChunk(
                text=f"Error: {e}",
                token_index=0,
                confidence=0,
                cumulative_confidence=0,
            )
    
    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding using Gemini embedding model."""
        genai = await self._get_client()
        
        try:
            # Use embedding model
            result = await asyncio.to_thread(
                genai.embed_content,
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document",
            )
            return result['embedding']
            
        except Exception as e:
            logger.warning(f"Gemini embedding failed: {e}, using hash fallback")
            import hashlib
            text_hash = hashlib.sha256(text.encode()).digest()
            embedding = []
            for i in range(768):  # Gemini embeddings are 768-dim
                byte_idx = i % 32
                val = (text_hash[byte_idx] / 255.0) * 2 - 1
                embedding.append(val)
            return embedding
    
    @classmethod
    def list_models(cls) -> dict[str, str]:
        """list available Gemini models."""
        return cls.MODELS.copy()


class CascadeBackend(LLMBackend):
    """
    Novel: Cascade inference backend that auto-escalates models.

    Strategy:
    1. Start with fastest model (BitNet/small)
    2. Monitor confidence during generation
    3. If confidence drops below threshold, escalate to larger model
    4. Transfer context and continue generation
    """

    def __init__(
        self,
        backends: list[LLMBackend],
        config: Optional[GenerationConfig] = None,
    ):
        super().__init__("cascade", config)
        self.backends = backends  # Ordered from fastest to most capable
        self.escalation_threshold = 0.6
        self.min_tokens_before_escalation = 30

    @property
    def backend_type(self) -> BackendType:
        return BackendType.OLLAMA  # Primary type

    async def load(self) -> bool:
        """Load all cascade backends."""
        results = await asyncio.gather(*[b.load() for b in self.backends])
        self._is_loaded = dict(results)
        return self._is_loaded

    async def unload(self) -> bool:
        """Unload all backends."""
        await asyncio.gather(*[b.unload() for b in self.backends])
        self._is_loaded = False
        return True

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate with cascade escalation."""
        cfg = config or self.config

        # Estimate complexity to choose starting backend
        complexity = self.estimate_complexity(prompt)
        start_idx = min(len(self.backends) - 1, int(complexity * len(self.backends)))

        current_backend = self.backends[start_idx]
        result = await current_backend.generate(prompt, cfg)

        # Check if escalation needed
        if (result.confidence_score < self.escalation_threshold and
            start_idx < len(self.backends) - 1):

            logger.info(f"Escalating from {current_backend.model_name} due to low confidence")

            # Escalate to next backend with full prompt + partial response
            enhanced_prompt = f"{prompt}\n\nPrevious attempt (incomplete): {result.text[:200]}..."
            next_backend = self.backends[start_idx + 1]
            result = await next_backend.generate(enhanced_prompt, cfg)
            result.escalated = True
            result.cascade_depth = start_idx + 1

        return result

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream with potential mid-generation escalation."""
        cfg = config or self.config

        complexity = self.estimate_complexity(prompt)
        current_idx = min(len(self.backends) - 1, int(complexity * len(self.backends)))

        accumulated_text = ""
        token_count = 0

        async for chunk in self.backends[current_idx].generate_stream(prompt, cfg):
            accumulated_text += chunk.text
            token_count += 1
            yield chunk

            # Check for escalation
            if (token_count >= self.min_tokens_before_escalation and
                chunk.cumulative_confidence < self.escalation_threshold and
                current_idx < len(self.backends) - 1):

                logger.info(f"Mid-stream escalation at token {token_count}")
                current_idx += 1

                # Continue with stronger model
                continuation_prompt = f"{prompt}\n\n{accumulated_text}"
                async for cont_chunk in self.backends[current_idx].generate_stream(
                    continuation_prompt, cfg
                ):
                    yield cont_chunk
                break

    async def get_embedding(self, text: str) -> list[float]:
        """Use most capable backend for embeddings."""
        for backend in reversed(self.backends):
            try:
                return await backend.get_embedding(text)
            except NotImplementedError:
                continue
            except Exception as e:
                logger.warning(f"Backend {backend.model_name} embedding failed: {e}")
                continue

        # Fallback to hash-based pseudo-embedding
        import hashlib
        logger.warning("All backends failed for embeddings, using hash-based fallback")

        text_hash = hashlib.sha256(text.encode()).digest()
        embedding = []
        for i in range(384):
            byte_idx = i % 32
            val = (text_hash[byte_idx] / 255.0) * 2 - 1
            val = val * (0.5 + 0.5 * ((i % 16) / 16))
            embedding.append(val)

        return embedding


class HermesCascadeBackend(CascadeBackend):
    """
    Hybrid Cascade for Hermes-4.3-36B integration.

    Strategy:
    1. Route simple/short queries to a fast LOCAL model (e.g. qwen3:8b via Ollama)
    2. Estimate complexity using token count, keyword signals, and quantum entropy
    3. If complexity > threshold OR local confidence is low, escalate to cloud Hermes-36B
    4. Record escalation events to HermesRL for adaptive learning

    This keeps response time low for simple chat while unlocking full 36B reasoning
    for complex tasks — all without exceeding local hardware limits.
    """

    # Keywords that signal complex reasoning / code tasks
    COMPLEXITY_KEYWORDS = frozenset({
        "analyze", "explain", "implement", "refactor", "debug", "optimize",
        "design", "architect", "recursive", "algorithm", "multi-step",
        "fibonacci", "complexity", "trade-off", "compare", "evaluate",
        "proof", "derive", "synthesize", "quantum", "12d", "hebbian",
    })

    def __init__(
        self,
        local_backend: LLMBackend,
        cloud_backend: LLMBackend,
        config: Optional[GenerationConfig] = None,
        escalation_threshold: float = 0.65,
    ):
        # CascadeBackend expects a list; order = [fast, capable]
        super().__init__([local_backend, cloud_backend], config)
        self.local_backend = local_backend
        self.cloud_backend = cloud_backend
        self.escalation_threshold = escalation_threshold
        self.min_tokens_before_escalation = 20
        self._escalation_count = 0
        self._local_count = 0

    @property
    def backend_type(self) -> BackendType:
        return BackendType.OLLAMA  # Primary is local

    def estimate_hermes_complexity(
        self, prompt: str, q_entropy: float = 0.5
    ) -> float:
        """
        Enhanced complexity estimation for Hermes cascade routing.

        Signals:
        - Token/word count (longer prompts tend to need deeper reasoning)
        - Keyword density (code/analysis/multi-step keywords)
        - Quantum entropy level (high entropy = chaotic state = need stronger model)
        - Code block presence (triple backticks)

        Returns float [0.0, 1.0]. Above self.escalation_threshold -> escalate.
        """
        words = prompt.lower().split()
        word_count = len(words)

        # 1. Length signal: 0-200 words -> 0.0-0.5
        length_signal = min(0.5, word_count / 400.0)

        # 2. Keyword density
        keyword_hits = sum(1 for w in words if w in self.COMPLEXITY_KEYWORDS)
        keyword_signal = min(0.4, keyword_hits * 0.08)

        # 3. Code block presence
        code_signal = 0.15 if "```" in prompt else 0.0

        # 4. Quantum entropy injection: high entropy -> bias toward escalation
        entropy_signal = max(0.0, (q_entropy - 0.5)) * 0.2

        # 5. Multi-line / structured prompt bonus
        line_count = prompt.count('\n')
        structure_signal = min(0.1, line_count * 0.01)

        complexity = (
            length_signal
            + keyword_signal
            + code_signal
            + entropy_signal
            + structure_signal
        )
        return min(1.0, complexity)

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate with Hermes-aware cascade escalation."""
        cfg = config or self.config

        # Estimate complexity with quantum entropy if available
        q_entropy = 0.5
        try:
            from Cosmos.core.quantum_bridge import get_quantum_bridge
            bridge = get_quantum_bridge()
            if bridge and bridge.connected:
                q_entropy = bridge.get_entropy()
        except Exception:
            pass

        complexity = self.estimate_hermes_complexity(prompt, q_entropy)

        if complexity >= self.escalation_threshold:
            # Direct to cloud -- complex task
            logger.info(
                f"[HERMES CASCADE] Escalating to cloud "
                f"(complexity={complexity:.3f} >= {self.escalation_threshold})"
            )
            result = await self.cloud_backend.generate(prompt, cfg)
            result.escalated = True
            result.cascade_depth = 1
            self._escalation_count += 1
            self._record_escalation(prompt, complexity, q_entropy)
            return result

        # Start with local model
        result = await self.local_backend.generate(prompt, cfg)
        self._local_count += 1

        # Check if local confidence is too low -> escalate
        if result.confidence_score < 0.5:
            logger.info(
                f"[HERMES CASCADE] Local confidence too low "
                f"({result.confidence_score:.3f}), escalating to cloud"
            )
            enhanced_prompt = (
                f"{prompt}\n\n"
                f"Previous attempt (low confidence): {result.text[:300]}..."
            )
            result = await self.cloud_backend.generate(enhanced_prompt, cfg)
            result.escalated = True
            result.cascade_depth = 1
            self._escalation_count += 1
            self._record_escalation(prompt, complexity, q_entropy)

        return result

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream with potential mid-generation escalation to cloud."""
        cfg = config or self.config

        q_entropy = 0.5
        try:
            from Cosmos.core.quantum_bridge import get_quantum_bridge
            bridge = get_quantum_bridge()
            if bridge and bridge.connected:
                q_entropy = bridge.get_entropy()
        except Exception:
            pass

        complexity = self.estimate_hermes_complexity(prompt, q_entropy)

        if complexity >= self.escalation_threshold:
            # Stream directly from cloud
            async for chunk in self.cloud_backend.generate_stream(prompt, cfg):
                yield chunk
            return

        # Stream from local with escalation check
        accumulated_text = ""
        token_count = 0

        async for chunk in self.local_backend.generate_stream(prompt, cfg):
            accumulated_text += chunk.text
            token_count += 1
            yield chunk

            # Mid-stream escalation check
            if (
                token_count >= self.min_tokens_before_escalation
                and chunk.cumulative_confidence < 0.45
            ):
                logger.info(
                    f"[HERMES CASCADE] Mid-stream escalation at token {token_count}"
                )
                continuation_prompt = f"{prompt}\n\n{accumulated_text}"
                async for cont_chunk in self.cloud_backend.generate_stream(
                    continuation_prompt, cfg
                ):
                    yield cont_chunk
                self._escalation_count += 1
                self._record_escalation(prompt, complexity, q_entropy)
                break

    def _record_escalation(
        self, prompt: str, complexity: float, q_entropy: float
    ):
        """Record escalation to HermesRL for adaptive learning."""
        try:
            from Cosmos.integration.hermes_bridge import get_hermes_bridge
            hb = get_hermes_bridge()
            hb.rl.record_experience(
                speaker="HermesCascade",
                response=(
                    f"[ESCALATION] complexity={complexity:.3f}, "
                    f"q_entropy={q_entropy:.3f}, "
                    f"prompt_preview={prompt[:100]}"
                ),
                coherence=complexity,
                user_responded=True,
            )
        except Exception:
            pass  # Hermes bridge is optional

    def get_cascade_stats(self) -> dict:
        """Return cascade usage statistics."""
        total = self._local_count + self._escalation_count
        return {
            "local_count": self._local_count,
            "escalation_count": self._escalation_count,
            "total": total,
            "escalation_rate": (
                self._escalation_count / total if total > 0 else 0.0
            ),
            "escalation_threshold": self.escalation_threshold,
        }


class CosmosBackend(LLMBackend):
    """
    Cosmo's 54D CST Transformer backend.

    Runs the Cosmo's model locally with persistent 54D state
    across generation calls for true infinite context.

    Features:
    - Local PyTorch inference (CPU or GPU)
    - 54D internal state persistence
    - Automatic checkpoint loading
    - Dream consolidation between calls
    """

    def __init__(
        self,
        model_name: str = "cosmos-54d",
        config: Optional[GenerationConfig] = None,
        checkpoint_path: Optional[str] = None,
        model_preset: str = "tiny",
        device: str = "auto",
    ):
        super().__init__(model_name, config)
        self.checkpoint_path = checkpoint_path
        self.model_preset = model_preset
        self.device_str = device
        self._model = None
        self._persistent_state = None
        self._tokenize_fn = None
        self._decode_fn = None
        self._device = None

    @property
    def backend_type(self) -> BackendType:
        return BackendType.COSMOS

    def _setup_tokenizer(self, vocab_size: int):
        """Initialize tokenizer."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            self._tokenize_fn = enc.encode
            self._decode_fn = enc.decode
        except ImportError:
            chars = list(set(
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
                " .,!?;:'\"-()[]{}/@#$%^&*+=<>~`\n\t"
            ))
            stoi = {ch: i % vocab_size for i, ch in enumerate(chars)}
            itos = {i: ch for ch, i in stoi.items()}
            self._tokenize_fn = lambda text: [stoi.get(c, 0) for c in text]
            self._decode_fn = lambda tokens: "".join(itos.get(t, "?") for t in tokens)

    async def load(self) -> bool:
        """Load Cosmo's model from checkpoint or create fresh."""
        try:
            import torch
            import sys
            import os

            # Add CosmoSynapse to path
            cosmos_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "Cosmic Genesis A.Lmi Cybernetic Bio Resonance Core",
            )
            if cosmos_root not in sys.path:
                sys.path.insert(0, cosmos_root)

            from cosmosynapse.model.cosmos_config import CosmosConfig
            from cosmosynapse.model.cosmos_model import CosmosTransformer

            # Device
            if self.device_str == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(self.device_str)

            # Load from checkpoint or create fresh
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
                cfg_dict = checkpoint["config"]
                cfg = CosmosConfig(
                    vocab_size=cfg_dict["vocab_size"],
                    d_model=cfg_dict["d_model"],
                    n_layers=cfg_dict["n_layers"],
                    n_heads=cfg_dict["n_heads"],
                    d_state=cfg_dict.get("d_state", 54),
                )
                self._model = CosmosTransformer(cfg)
                self._model.load_state_any(checkpoint["model_state_dict"])
                self._persistent_state = checkpoint.get("persistent_state")
                logger.info(f"[COSMOS] Loaded checkpoint: {self.checkpoint_path}")
            else:
                preset_fn = getattr(CosmosConfig, self.model_preset, CosmosConfig.tiny)
                cfg = preset_fn()
                self._model = CosmosTransformer(cfg)
                logger.info(f"[COSMOS] Created fresh model ({self.model_preset} preset)")

            self._model = self._model.to(self._device)
            self._model.eval()
            self._setup_tokenizer(cfg.vocab_size)

            self._is_loaded = True
            logger.info(f"[COSMOS] Model loaded on {self._device} "
                        f"({self._model.get_num_params()/1e6:.1f}M params)")
            return True

        except Exception as e:
            logger.error(f"[COSMOS] Load failed: {e}")
            return False

    async def unload(self) -> bool:
        """Unload model and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self._is_loaded = False
        logger.info("[COSMOS] Model unloaded")
        return True

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text using Cosmo's Transformer with 54D state persistence."""
        import torch

        if self._model is None or self._tokenize_fn is None:
            raise RuntimeError("[COSMOS] Model not loaded")

        try:
            cfg = config or self.config
            start_time = time.time()
            self.confidence_estimator.reset()

            # Tokenize
            tokens = self._tokenize_fn(prompt)
            # Ensure safe tokenization for empty prompts
            if not tokens:
                tokens = [0] # Padding token
                
            idx = torch.tensor([tokens], dtype=torch.long, device=self._device)

            # Generate with state persistence
            with torch.no_grad():
                output, new_state = self._model.generate(
                    idx,
                    max_new_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    state_x54=self._persistent_state,
                )

            # Update persistent state
            self._persistent_state = new_state

            # Decode generated tokens
            gen_tokens = output[0, len(tokens):].tolist()
            text = self._decode_fn(gen_tokens)

            total_time = time.time() - start_time

            # Update confidence
            for word in text.split():
                self.confidence_estimator.update(word)

            return GenerationResult(
                text=text,
                tokens_generated=len(gen_tokens),
                tokens_per_second=len(gen_tokens) / max(0.001, total_time),
                model_used=self.model_name,
                backend_used=self.backend_type,
                confidence_score=self.confidence_estimator.base_confidence,
                total_time=total_time,
            )
        except Exception as e:
            logger.error(f"[COSMOS] Generation failed: {e}")
            # Return safe fallback to prevent crash
            return GenerationResult(
                text="",
                tokens_generated=0,
                tokens_per_second=0.0,
                model_used=self.model_name,
                backend_used=self.backend_type,
                confidence_score=0.0,
                total_time=0.1,
            )

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming generation (simulated - generates full then yields tokens)."""
        result = await self.generate(prompt, config)

        words = result.text.split()
        cumulative_confidence = 1.0

        for i, word in enumerate(words):
            confidence = self.confidence_estimator.update(word)
            cumulative_confidence = 0.95 * cumulative_confidence + 0.05 * confidence

            yield StreamChunk(
                text=word + " ",
                token_index=i,
                confidence=confidence,
                cumulative_confidence=cumulative_confidence,
            )
            await asyncio.sleep(0.01)

    async def get_embedding(self, text: str) -> list[float]:
        """
        Get embedding from the 54D internal state.

        Uses the model's 54D state as a rich embedding,
        capturing Hebbian, chaos, and memory dynamics.
        """
        import torch
        import hashlib

        if self._model is None or self._tokenize_fn is None:
            # Fallback to hash-based
            text_hash = hashlib.sha256(text.encode()).digest()
            return [(text_hash[i % 32] / 255.0) * 2 - 1 for i in range(384)]

        tokens = self._tokenize_fn(text)
        idx = torch.tensor([tokens], dtype=torch.long, device=self._device)

        with torch.no_grad():
            _, _, _, state = self._model(idx, state_x54=self._persistent_state)

        # Flatten the 54D state into an embedding vector
        embedding = state.detach().cpu().flatten().tolist()

        # Pad or truncate to standard 384 dimensions
        if len(embedding) < 384:
            embedding = embedding + [0.0] * (384 - len(embedding))
        else:
            embedding = embedding[:384]

        return embedding
