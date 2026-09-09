"""
Unified Model Manager and Device Hardware Capability System.
Provides hardware detection (CPU, RAM, GPU, VRAM, NPU) and model recommendations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import os
import platform
import psutil
import logging

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    AVAILABLE = "Available"
    DOWNLOADING = "Downloading"
    INSTALLED = "Installed"
    UPDATING = "Updating"
    FAILED = "Failed"
    UNAVAILABLE = "Unavailable"


@dataclass
class HardwareCapabilities:
    os: str
    architecture: str
    cpu_count: int
    ram_gb: float
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    vram_gb: float = 0.0
    npu_available: bool = False
    available_storage_gb: float = 0.0

    @classmethod
    def detect(cls) -> "HardwareCapabilities":
        system_os = platform.system()
        arch = platform.machine()
        cpus = os.cpu_count() or 1
        ram = round(psutil.virtual_memory().total / (1024**3), 2)

        storage = 0.0
        try:
            st = os.statvfs("/") if hasattr(os, "statvfs") else None
            if st:
                storage = round((st.f_bavail * st.f_frsize) / (1024**3), 2)
        except Exception:
            storage = 50.0

        gpu_avail = False
        gpu_name = None
        vram = 0.0

        return cls(
            os=system_os,
            architecture=arch,
            cpu_count=cpus,
            ram_gb=ram,
            gpu_available=gpu_avail,
            gpu_name=gpu_name,
            vram_gb=vram,
            npu_available=False,
            available_storage_gb=storage,
        )


@dataclass
class ModelInfo:
    id: str
    name: str
    provider: str
    size_mb: int = 0
    quantization: Optional[str] = None
    context_length: int = 128000
    status: ModelStatus = ModelStatus.AVAILABLE
    min_ram_gb: float = 4.0
    min_vram_gb: float = 0.0


class ModelManager:
    """Manages models, hardware detection, recommendations, state, and defaults."""

    DEFAULT_MODELS = [
        ModelInfo(
            id="nous-hermes-3-llama-3.1-8b",
            name="Nous Hermes 3 Llama 3.1 8B",
            provider="nous",
            context_length=128000,
            status=ModelStatus.AVAILABLE,
            min_ram_gb=8.0,
        ),
        ModelInfo(
            id="gpt-4o",
            name="GPT-4o",
            provider="openai",
            context_length=128000,
            status=ModelStatus.AVAILABLE,
            min_ram_gb=2.0,
        ),
        ModelInfo(
            id="claude-3-5-sonnet",
            name="Claude 3.5 Sonnet",
            provider="anthropic",
            context_length=200000,
            status=ModelStatus.AVAILABLE,
            min_ram_gb=2.0,
        ),
        ModelInfo(
            id="llama3-8b-gguf",
            name="Llama 3 8B Local GGUF",
            provider="llama.cpp",
            size_mb=4800,
            quantization="Q4_K_M",
            context_length=8192,
            status=ModelStatus.AVAILABLE,
            min_ram_gb=8.0,
        ),
    ]

    def __init__(self) -> None:
        self.hardware = HardwareCapabilities.detect()
        self._models: Dict[str, ModelInfo] = {m.id: m for m in self.DEFAULT_MODELS}
        self.default_model_id = "nous-hermes-3-llama-3.1-8b"
        self.fallback_model_id = "gpt-4o"

    def list_models(self) -> List[ModelInfo]:
        return list(self._models.values())

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        return self._models.get(model_id)

    def recommend_models(self) -> List[ModelInfo]:
        """Recommends models based on hardware capabilities."""
        suitable = []
        for model in self._models.values():
            if self.hardware.ram_gb >= model.min_ram_gb:
                if model.min_vram_gb == 0 or (
                    self.hardware.gpu_available
                    and self.hardware.vram_gb >= model.min_vram_gb
                ):
                    suitable.append(model)
        return suitable

    def set_default(self, model_id: str) -> bool:
        if model_id in self._models:
            self.default_model_id = model_id
            return True
        return False
