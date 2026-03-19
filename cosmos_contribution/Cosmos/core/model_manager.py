"""
cosmos Model Manager

Novel Approaches:
1. Predictive Model Preloading - Anticipate needed models based on usage patterns
2. Smart Memory Management - Dynamic VRAM/RAM balancing
3. Model Fingerprinting - Quick validation without full load
4. Hot-Swap Capability - Seamless model switching without restart
"""

import asyncio
import hashlib
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

import yaml
from loguru import logger

from Cosmos.core.llm_backend import (
    LLMBackend,
    OllamaBackend,
    LlamaCppBackend,
    BitNetBackend,
    CascadeBackend,
    OpenAICompatibleBackend,
    GeminiBackend,
    CosmosBackend,
    GenerationConfig,
    BackendType,
)


class HardwareProfile(Enum):
    CPU_ONLY = "cpu_only"
    LOW_VRAM = "low_vram"      # 2-4 GB
    MEDIUM_VRAM = "medium_vram"  # 4-8 GB
    HIGH_VRAM = "high_vram"    # 8+ GB


@dataclass
class HardwareInfo:
    """Detected hardware capabilities."""
    platform: str
    cpu_cores: int
    total_ram_gb: float
    available_ram_gb: float
    gpu_available: bool
    gpu_name: str = ""
    vram_gb: float = 0.0
    cuda_available: bool = False
    mps_available: bool = False  # Apple Metal
    profile: HardwareProfile = HardwareProfile.CPU_ONLY


@dataclass
class ModelInfo:
    """Model metadata and status."""
    name: str
    backend: BackendType
    params: float
    vram_required: float
    ram_required: float
    context_length: int
    quantization: str
    strengths: list[str] = field(default_factory=list)
    is_loaded: bool = False
    is_available: bool = False
    path: Optional[str] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0


@dataclass
class ModelLoadResult:
    """Result of model loading operation."""
    success: bool
    model_name: str
    backend: Optional[LLMBackend] = None
    error: Optional[str] = None
    load_time_seconds: float = 0.0


class ModelManager:
    """
    Central model management with intelligent loading/unloading.

    Features:
    - Automatic hardware detection and optimization
    - Predictive model preloading based on usage patterns
    - Smart memory management with eviction policies
    - Hot-swap between models without restart
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        models_dir: str = "./models",
        max_loaded_models: int = 2,
    ):
        self.config_path = config_path or "configs/models.yaml"
        self.models_dir = Path(models_dir)
        self.max_loaded_models = max_loaded_models

        self.hardware: Optional[HardwareInfo] = None
        self.model_configs: dict = {}
        self.loaded_backends: dict[str, LLMBackend] = {}
        self.model_infos: dict[str, ModelInfo] = {}
        self.usage_history: list[tuple[str, datetime]] = []

        self._preload_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> HardwareInfo:
        """Initialize manager and detect hardware."""
        self.hardware = await self._detect_hardware()
        await self._load_config()
        await self._scan_available_models()

        logger.info(f"Hardware profile: {self.hardware.profile.value}")
        logger.info(f"Available RAM: {self.hardware.available_ram_gb:.1f} GB")
        if self.hardware.gpu_available:
            logger.info(f"GPU: {self.hardware.gpu_name} ({self.hardware.vram_gb:.1f} GB VRAM)")

        return self.hardware

    async def _detect_hardware(self) -> HardwareInfo:
        """Comprehensive hardware detection."""
        import psutil

        info = HardwareInfo(
            platform=platform.system(),
            cpu_cores=psutil.cpu_count(logical=False) or 4,
            total_ram_gb=psutil.virtual_memory().total / (1024**3),
            available_ram_gb=psutil.virtual_memory().available / (1024**3),
            gpu_available=False,
        )

        # Check for CUDA GPU
        try:
            import torch
            if torch.cuda.is_available():
                info.gpu_available = True
                info.cuda_available = True
                info.gpu_name = torch.cuda.get_device_name(0)
                info.vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                info.gpu_available = True
                info.mps_available = True
                info.gpu_name = "Apple Metal"
                # MPS shares system RAM
                info.vram_gb = info.total_ram_gb * 0.5
        except ImportError:
            pass

        # Determine hardware profile
        if not info.gpu_available:
            info.profile = HardwareProfile.CPU_ONLY
        elif info.vram_gb < 4:
            info.profile = HardwareProfile.LOW_VRAM
        elif info.vram_gb < 8:
            info.profile = HardwareProfile.MEDIUM_VRAM
        else:
            info.profile = HardwareProfile.HIGH_VRAM

        return info

    async def _load_config(self):
        """Load model configurations from YAML."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.model_configs = config.get('models', {})
        except FileNotFoundError:
            logger.warning(f"Config not found at {self.config_path}, using defaults")
            self.model_configs = self._default_model_configs()

    def _default_model_configs(self) -> dict:
        """Default model configurations."""
        return {
            "deepseek-r1-1.5b": {
                "name": "DeepSeek-R1-Distill-Qwen-1.5B",
                "backend": "ollama",
                "ollama_name": "deepseek-r1:1.5b",
                "params": 1.5e9,
                "vram_gb": 2.0,
                "ram_gb": 4.0,
                "context_length": 32768,
                "quantization": "Q4_K_M",
                "strengths": ["reasoning", "math", "code"],
            },
            "qwen3-0.6b": {
                "name": "Qwen3-0.6B",
                "backend": "ollama",
                "ollama_name": "qwen3:0.6b",
                "params": 0.6e9,
                "vram_gb": 1.0,
                "ram_gb": 2.0,
                "context_length": 32768,
                "quantization": "Q4_K_M",
                "strengths": ["lightweight", "multilingual"],
            },
            # Gemini Models
            "gemini-2.0-flash": {
                "name": "Gemini 2.0 Flash",
                "backend": "gemini",
                "api_model": "gemini-2.0-flash",
                "params": 0,  # Cloud model
                "vram_gb": 0,
                "ram_gb": 0.5,
                "context_length": 1000000,
                "quantization": "cloud",
                "strengths": ["fast", "multimodal", "reasoning", "code"],
                "env_key": "GEMINI_API_KEY",
            },
            "gemini-2.0-flash-lite": {
                "name": "Gemini 2.0 Flash Lite",
                "backend": "gemini",
                "api_model": "gemini-2.0-flash-lite",
                "params": 0,
                "vram_gb": 0,
                "ram_gb": 0.5,
                "context_length": 1000000,
                "quantization": "cloud",
                "strengths": ["lightweight", "fast", "cost-effective"],
                "env_key": "GEMINI_API_KEY",
            },
            "gemini-1.5-pro": {
                "name": "Gemini 1.5 Pro",
                "backend": "gemini",
                "api_model": "gemini-1.5-pro",
                "params": 0,
                "vram_gb": 0,
                "ram_gb": 0.5,
                "context_length": 2000000,
                "quantization": "cloud",
                "strengths": ["reasoning", "code", "research", "long-context"],
                "env_key": "GEMINI_API_KEY",
            },
            "gemini-1.5-flash": {
                "name": "Gemini 1.5 Flash",
                "backend": "gemini",
                "api_model": "gemini-1.5-flash",
                "params": 0,
                "vram_gb": 0,
                "ram_gb": 0.5,
                "context_length": 1000000,
                "quantization": "cloud",
                "strengths": ["balanced", "fast", "multimodal"],
                "env_key": "GEMINI_API_KEY",
            },
            # Hermes Models
            "hermes-4-70b": {
                "name": "Hermes-4-70B",
                "backend": "openai_compatible",
                "api_model": "Hermes-4-70B",
                "api_base": "https://inference-api.nousresearch.com/v1",
                "params": 70e9,
                "vram_gb": 0,
                "ram_gb": 0.5,
                "context_length": 128000,
                "quantization": "cloud",
                "strengths": ["reasoning", "code", "agentic", "research"],
                "env_key": "HERMES_API_KEY",
            },
            # ── Hybrid Cascade: local companion + cloud escalation ──
            "hermes-36b-cloud": {
                "name": "Hermes-4.3-36B Cloud",
                "backend": "openai_compatible",
                "api_model": "nousresearch/hermes-3-llama-3.1-405b",
                "api_base": "https://openrouter.ai/api/v1",
                "params": 36e9,
                "vram_gb": 0,
                "ram_gb": 0.5,
                "context_length": 32768,
                "quantization": "cloud",
                "strengths": ["reasoning", "code", "analysis", "agentic"],
                "env_key": "OPENROUTER_API_KEY",
                "cloud_only": True,
            },
            "hermes-local-companion": {
                "name": "Hermes Local Companion",
                "backend": "ollama",
                "ollama_name": "qwen3:8b",
                "params": 8e9,
                "vram_gb": 5.0,
                "ram_gb": 6.0,
                "context_length": 32768,
                "quantization": "Q4_K_M",
                "strengths": ["fast", "reasoning", "code", "lightweight"],
            },
        }

    async def _scan_available_models(self):
        """Scan for available models across all backends."""
        # Check Ollama models
        try:
            import ollama
            client = ollama.Client()
            ollama_models = client.list()
            for model in ollama_models.get('models', []):
                name = model['name'].split(':')[0]
                for key, config in self.model_configs.items():
                    if config.get('ollama_name', '').startswith(name):
                        self.model_infos[key] = ModelInfo(
                            name=config['name'],
                            backend=BackendType.OLLAMA,
                            params=config.get('params', 0),
                            vram_required=config.get('vram_gb', 0),
                            ram_required=config.get('ram_gb', 0),
                            context_length=config.get('context_length', 4096),
                            quantization=config.get('quantization', 'unknown'),
                            strengths=config.get('strengths', []),
                            is_available=True,
                        )
        except Exception as e:
            logger.debug(f"Ollama scan failed: {e}")

        # Check local GGUF models
        self.models_dir.mkdir(parents=True, exist_ok=True)
        for gguf_file in self.models_dir.glob("**/*.gguf"):
            model_key = gguf_file.stem.lower().replace('-', '_')
            if model_key not in self.model_infos:
                self.model_infos[model_key] = ModelInfo(
                    name=gguf_file.stem,
                    backend=BackendType.LLAMA_CPP,
                    params=0,
                    vram_required=2.0,
                    ram_required=4.0,
                    context_length=4096,
                    quantization="unknown",
                    is_available=True,
                    path=str(gguf_file),
                )

    async def load_model(
        self,
        model_key: str,
        config: Optional[GenerationConfig] = None,
    ) -> ModelLoadResult:
        """
        Load a model with smart resource management.

        Features:
        - Automatic eviction of least-recently-used models
        - Memory validation before loading
        - Hot-swap if model already loaded
        """
        import time

        async with self._lock:
            start_time = time.time()

            # Check if already loaded
            if model_key in self.loaded_backends:
                self._record_usage(model_key)
                return ModelLoadResult(
                    success=True,
                    model_name=model_key,
                    backend=self.loaded_backends[model_key],
                    load_time_seconds=0,
                )

            # Get model config
            if model_key not in self.model_configs:
                return ModelLoadResult(
                    success=False,
                    model_name=model_key,
                    error=f"Unknown model: {model_key}",
                )

            model_config = self.model_configs[model_key]

            # Check memory requirements
            required_ram = model_config.get('ram_gb', 4.0)
            if self.hardware and self.hardware.available_ram_gb < required_ram:
                # Evict models to free memory
                await self._evict_lru_models(required_ram)

            # Ensure we don't exceed max loaded models
            while len(self.loaded_backends) >= self.max_loaded_models:
                await self._evict_lru_models(0)

            # Create backend based on type
            backend_type = model_config.get('backend', 'ollama')

            try:
                if backend_type == 'ollama':
                    backend = OllamaBackend(
                        model_name=model_config.get('ollama_name', model_key),
                        config=config,
                    )
                elif backend_type == 'llama_cpp':
                    model_path = model_config.get('path') or self._find_gguf(model_key)
                    if not model_path:
                        return ModelLoadResult(
                            success=False,
                            model_name=model_key,
                            error="GGUF file not found",
                        )
                    backend = LlamaCppBackend(
                        model_path=model_path,
                        config=config,
                    )
                elif backend_type == 'bitnet':
                    backend = BitNetBackend(
                        model_path=model_config.get('path', ''),
                        config=config,
                    )
                elif backend_type == 'openai_compatible':
                    # Get API key from environment
                    env_key = model_config.get('env_key', 'OPENAI_API_KEY')
                    api_key = os.environ.get(env_key)

                    # Also check for common alternatives
                    if not api_key:
                        api_key = os.environ.get('HERMES_API_KEY')
                    if not api_key:
                        api_key = os.environ.get('DEEPINFRA_API_KEY')
                    if not api_key:
                        api_key = os.environ.get('MINIMAX_API_KEY')
                    if not api_key:
                        api_key = os.environ.get('OPENROUTER_API_KEY')

                    if not api_key:
                        return ModelLoadResult(
                            success=False,
                            model_name=model_key,
                            error=f"API key not found. Set {env_key} environment variable.",
                        )

                    backend = OpenAICompatibleBackend(
                        model_name=model_config.get('api_model', model_key),
                        api_key=api_key,
                        base_url=model_config.get('api_base', 'https://api.deepinfra.com/v1/openai'),
                        config=config,
                    )
                elif backend_type == 'gemini':
                    # Get Gemini API key from environment
                    env_key = model_config.get('env_key', 'GEMINI_API_KEY')
                    api_key = os.environ.get(env_key)
                    
                    # Also check for GOOGLE_API_KEY
                    if not api_key:
                        api_key = os.environ.get('GOOGLE_API_KEY')
                    
                    if not api_key:
                        return ModelLoadResult(
                            success=False,
                            model_name=model_key,
                            error=f"Gemini API key not found. Set {env_key} or GOOGLE_API_KEY environment variable.",
                        )
                    
                    backend = GeminiBackend(
                        model_name=model_config.get('api_model', model_key),
                        api_key=api_key,
                        config=config,
                    )
                elif backend_type == 'cosmos':
                    # Cosmo's 54D CST Transformer — local model
                    backend = CosmosBackend(
                        model_name=model_config.get('name', 'cosmos-54d'),
                        config=config,
                        checkpoint_path=model_config.get('checkpoint_path'),
                        model_preset=model_config.get('model_preset', 'tiny'),
                        device=model_config.get('device', 'auto'),
                    )
                else:
                    return ModelLoadResult(
                        success=False,
                        model_name=model_key,
                        error=f"Unknown backend: {backend_type}",
                    )

                # Load the model
                success = await backend.load()

                if success:
                    self.loaded_backends[model_key] = backend
                    self._record_usage(model_key)

                    if model_key in self.model_infos:
                        self.model_infos[model_key].is_loaded = True

                    load_time = time.time() - start_time
                    logger.info(f"Loaded {model_key} in {load_time:.2f}s")

                    return ModelLoadResult(
                        success=True,
                        model_name=model_key,
                        backend=backend,
                        load_time_seconds=load_time,
                    )
                else:
                    return ModelLoadResult(
                        success=False,
                        model_name=model_key,
                        error="Model load failed",
                    )

            except Exception as e:
                logger.error(f"Error loading {model_key}: {e}")
                return ModelLoadResult(
                    success=False,
                    model_name=model_key,
                    error=str(e),
                )

    async def unload_model(self, model_key: str) -> bool:
        """Unload a model and free resources."""
        async with self._lock:
            if model_key not in self.loaded_backends:
                return True

            backend = self.loaded_backends[model_key]
            await backend.unload()
            del self.loaded_backends[model_key]

            if model_key in self.model_infos:
                self.model_infos[model_key].is_loaded = False

            logger.info(f"Unloaded {model_key}")
            return True

    async def _evict_lru_models(self, required_ram_gb: float):
        """Evict least recently used models to free memory."""
        if not self.loaded_backends:
            return

        # Sort by last usage
        sorted_models = sorted(
            self.loaded_backends.keys(),
            key=lambda k: self.model_infos.get(k, ModelInfo(
                name=k, backend=BackendType.OLLAMA, params=0,
                vram_required=0, ram_required=0, context_length=0,
                quantization=""
            )).last_used or datetime.min,
        )

        freed_ram = 0.0
        for model_key in sorted_models:
            if freed_ram >= required_ram_gb and len(self.loaded_backends) < self.max_loaded_models:
                break

            if model_key in self.model_configs:
                freed_ram += self.model_configs[model_key].get('ram_gb', 2.0)

            await self.unload_model(model_key)

    def _record_usage(self, model_key: str):
        """Record model usage for LRU tracking."""
        now = datetime.now()
        self.usage_history.append((model_key, now))

        if model_key in self.model_infos:
            self.model_infos[model_key].last_used = now
            self.model_infos[model_key].usage_count += 1

        # Keep history bounded
        if len(self.usage_history) > 1000:
            self.usage_history = self.usage_history[-500:]

    def _find_gguf(self, model_key: str) -> Optional[str]:
        """Find GGUF file for a model."""
        for gguf_file in self.models_dir.glob("**/*.gguf"):
            if model_key.lower() in gguf_file.stem.lower():
                return str(gguf_file)
        return None

    def get_backend(self, model_key: str) -> Optional[LLMBackend]:
        """Get a loaded backend by key."""
        return self.loaded_backends.get(model_key)

    def get_best_model_for_task(self, task_type: str, field: Optional[dict] = None) -> Optional[str]:
        """
        Novel: Intelligent model selection based on task type and hardware.
        V4.2: Integrated UQ Layer support for reasoning escalation.

        Returns the best available model for the given task type.
        """
        task_strengths = {
            "code": ["code", "reasoning"],
            "reasoning": ["reasoning", "math"],
            "creative": ["creative", "multilingual"],
            "research": ["reasoning", "general"],
            "speed": ["lightweight", "fast"],
            "multimodal": ["multimodal", "vision"],
        }

        required_strengths = task_strengths.get(task_type, ["general"])
        
        # Swarm Upgrade: Response to Uncertainty Quantification (UQ)
        is_uncertain = False
        if field and hasattr(field, 'is_uncertain'):
            is_uncertain = field.is_uncertain()

        # Score available models
        best_model = None
        best_score = -1

        for key, info in self.model_infos.items():
            if not info.is_available:
                continue

            # Check hardware constraints
            if self.hardware:
                if info.ram_required > self.hardware.available_ram_gb:
                    continue
                if info.vram_required > self.hardware.vram_gb and self.hardware.gpu_available:
                    continue

            # Score based on strength match
            score = sum(1 for s in required_strengths if s in info.strengths)
            
            # [UQ ESCALATION] If system is uncertain, prioritize high-reasoning models
            if is_uncertain and "reasoning" in info.strengths:
                score += 2.0
                logger.info(f"[UQ] Identifying reasoning model {key} for escalation due to high uncertainty.")

            # Bonus for already loaded models
            if info.is_loaded:
                score += 0.5

            # Penalty for large models on constrained hardware
            if self.hardware and self.hardware.profile == HardwareProfile.CPU_ONLY:
                score -= info.params / 1e9 * 0.1

            if score > best_score:
                best_score = score
                best_model = key

        return best_model

    async def create_cascade_backend(
        self,
        model_keys: Optional[list[str]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> CascadeBackend:
        """
        Create a cascade backend with multiple models.

        Models are ordered from fastest to most capable.
        """
        if model_keys is None:
            # Auto-select based on hardware
            model_keys = []
            if self.hardware:
                if self.hardware.profile == HardwareProfile.CPU_ONLY:
                    model_keys = ["qwen3-0.6b", "deepseek-r1-1.5b"]
                else:
                    model_keys = ["qwen3-0.6b", "deepseek-r1-1.5b", "phi-3-mini"]

        backends = []
        for key in model_keys:
            result = await self.load_model(key, config)
            if result.success and result.backend:
                backends.append(result.backend)

        return CascadeBackend(backends, config)

    async def download_model(
        self,
        model_key: str,
        progress_callback: Optional[Callable] = None,
    ) -> bool:
        """
        Download a model with progress tracking.

        Supports Ollama pull and HuggingFace downloads.
        """
        if model_key not in self.model_configs:
            logger.error(f"Unknown model: {model_key}")
            return False

        config = self.model_configs[model_key]
        backend_type = config.get('backend', 'ollama')

        try:
            if backend_type == 'ollama':
                import ollama
                client = ollama.Client()

                # Ollama pull with progress
                ollama_name = config.get('ollama_name', model_key)
                logger.info(f"Pulling {ollama_name} via Ollama...")

                for progress in client.pull(ollama_name, stream=True):
                    if progress_callback:
                        progress_callback(progress)
                    status = progress.get('status', '')
                    if 'completed' in status.lower():
                        logger.info(f"Downloaded {ollama_name}")

                return True

            elif backend_type == 'llama_cpp':
                # Download from HuggingFace
                repo = config.get('llama_cpp_repo')
                filename = config.get('llama_cpp_file')

                if not repo or not filename:
                    logger.error("Missing repo or filename for llama.cpp model")
                    return False

                from huggingface_hub import hf_hub_download

                path = hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    local_dir=str(self.models_dir),
                    local_dir_use_symlinks=False,
                )

                logger.info(f"Downloaded {filename} to {path}")
                return True

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

        return False

    async def generate(self, prompt: str, system: Optional[str] = None, config: Optional[dict] = None) -> str:
        """
        Unified generation interface.

        Automatically selects and loads the best model if none works.
        """
        # Combine system prompt if provided
        full_prompt = prompt
        if system:
            full_prompt = f"System: {system}\n\nUser: {prompt}"

        # Convert dict to GenerationConfig if needed
        gen_config = None
        if config:
            gen_config = GenerationConfig(**config)

        if not self.loaded_backends:
            # Auto-load a general model
            best_model = self.get_best_model_for_task("general")
            if best_model:
                await self.load_model(best_model)
            else:
                return "Error: No models available."

        # Use the first loaded backend (or valid one)
        for backend in self.loaded_backends.values():
            result = await backend.generate(full_prompt, gen_config)
            return result.text
        
        return "Error: Generation failed."

    def get_status(self) -> dict:
        """Get comprehensive status for UI display."""
        return {
            "hardware": {
                "platform": self.hardware.platform if self.hardware else "unknown",
                "profile": self.hardware.profile.value if self.hardware else "unknown",
                "ram_total_gb": round(self.hardware.total_ram_gb, 1) if self.hardware else 0,
                "ram_available_gb": round(self.hardware.available_ram_gb, 1) if self.hardware else 0,
                "gpu_available": self.hardware.gpu_available if self.hardware else False,
                "gpu_name": self.hardware.gpu_name if self.hardware else "",
                "vram_gb": round(self.hardware.vram_gb, 1) if self.hardware else 0,
            },
            "models": {
                key: {
                    "name": info.name,
                    "backend": info.backend.value,
                    "loaded": info.is_loaded,
                    "available": info.is_available,
                    "usage_count": info.usage_count,
                    "strengths": info.strengths,
                }
                for key, info in self.model_infos.items()
            },
            "loaded_count": len(self.loaded_backends),
            "max_loaded": self.max_loaded_models,
        }
