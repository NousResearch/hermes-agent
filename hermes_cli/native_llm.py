"""
Hermes Native Local LLM serving (hermes-local provider).

Zero-config, first-class local inference that is easier and more integrated
than running a separate Ollama or LM Studio instance.

Features:
- Auto-detects system RAM + VRAM (NVIDIA) + basic NPU presence
- Curated high-quality models (Gemma 4, Qwen3, DeepSeek, Phi-4, GLM, Kimi, Ministral, ...)
- "Suggested Model" recommendations based on available memory + speed/quality tradeoffs
- Downloads GGUF quants from Hugging Face (with progress)
- Manages its own llama-server process with full GPU offload (n_gpu_layers=-1)
- Exposes standard OpenAI /v1 compatible endpoint for the rest of Hermes
- Full offload when VRAM permits; graceful partial offload otherwise

Usage from elsewhere:
    from hermes_cli.native_llm import (
        get_system_resources,
        suggest_models,
        list_available_local_models,
        ensure_model,
        start_server_for_model,
        get_server_base_url,
        stop_server,
    )
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Model catalog — all the models requested + sensible quants.
# Memory numbers are rough "full load + KV" estimates for 8k ctx.
# Tune as real quants / empirical data improves.
# ------------------------------------------------------------------

SUPPORTED_MODELS: Dict[str, Dict[str, Any]] = {
    # Gemma 4 family (Google)
    "gemma4-31b": {
        "display": "Gemma 4 31B",
        "params_b": 31,
        "family": "gemma",
        "quants": ["Q4_K_M", "Q5_K_M", "Q8_0"],
        "default_quant": "Q5_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/gemma-4-31b-GGUF", "gemma-4-31b-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/gemma-4-31b-GGUF", "gemma-4-31b-Q5_K_M.gguf"),
            "Q8_0": ("bartowski/gemma-4-31b-GGUF", "gemma-4-31b-Q8_0.gguf"),
        },
        "vram_full_q4": 18.5,
        "ram_full_q4": 23.0,
        "speed_tier": 32,   # relative tokens/s on strong GPU
        "quality_tier": 88,
        "notes": "Excellent instruction following and coding.",
    },
    "gemma4-26b": {
        "display": "Gemma 4 26B",
        "params_b": 26,
        "family": "gemma",
        "quants": ["Q4_K_M", "Q5_K_M"],
        "default_quant": "Q5_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/gemma-4-26b-GGUF", "gemma-4-26b-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/gemma-4-26b-GGUF", "gemma-4-26b-Q5_K_M.gguf"),
        },
        "vram_full_q4": 15.5,
        "ram_full_q4": 19.5,
        "speed_tier": 38,
        "quality_tier": 85,
        "notes": "Sweet spot for many 24-32 GB systems.",
    },
    "gemma4-12b": {
        "display": "Gemma 4 12B",
        "params_b": 12,
        "family": "gemma",
        "quants": ["Q4_K_M", "Q5_K_M", "Q8_0"],
        "default_quant": "Q5_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/gemma-4-12b-GGUF", "gemma-4-12b-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/gemma-4-12b-GGUF", "gemma-4-12b-Q5_K_M.gguf"),
            "Q8_0": ("bartowski/gemma-4-12b-GGUF", "gemma-4-12b-Q8_0.gguf"),
        },
        "vram_full_q4": 8.0,
        "ram_full_q4": 10.5,
        "speed_tier": 58,
        "quality_tier": 78,
        "notes": "Fast, great on laptops or low VRAM.",
    },
    "gemma4-4b": {
        "display": "Gemma 4 4B",
        "params_b": 4,
        "family": "gemma",
        "quants": ["Q4_K_M", "Q5_K_M"],
        "default_quant": "Q5_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/gemma-4-4b-GGUF", "gemma-4-4b-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/gemma-4-4b-GGUF", "gemma-4-4b-Q5_K_M.gguf"),
        },
        "vram_full_q4": 3.2,
        "ram_full_q4": 4.5,
        "speed_tier": 95,
        "quality_tier": 65,
        "notes": "Ultra fast for quick tasks / edge.",
    },
    # Qwen 3.6 series
    "qwen3-27b": {
        "display": "Qwen 3.6 27B",
        "params_b": 27,
        "family": "qwen",
        "quants": ["Q4_K_M", "Q5_K_M", "Q6_K"],
        "default_quant": "Q5_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/Qwen3-27B-GGUF", "Qwen3-27B-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/Qwen3-27B-GGUF", "Qwen3-27B-Q5_K_M.gguf"),
        },
        "vram_full_q4": 16.0,
        "ram_full_q4": 20.0,
        "speed_tier": 35,
        "quality_tier": 87,
        "notes": "Outstanding at math, coding, long context.",
    },
    "qwen3-35b": {
        "display": "Qwen 3.6 35B",
        "params_b": 35,
        "family": "qwen",
        "quants": ["Q4_K_M", "Q5_K_M"],
        "default_quant": "Q4_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/Qwen3-35B-GGUF", "Qwen3-35B-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/Qwen3-35B-GGUF", "Qwen3-35B-Q5_K_M.gguf"),
        },
        "vram_full_q4": 21.0,
        "ram_full_q4": 25.5,
        "speed_tier": 29,
        "quality_tier": 90,
        "notes": "Top tier quality when you have the VRAM.",
    },
    # Qwen3.5 MoE 397B (A17B active params)
    "qwen3.5-397b-a17b": {
        "display": "Qwen3.5 397B-A17B (MoE)",
        "params_b": 397,
        "family": "qwen-moe",
        "quants": ["Q3_K_M", "Q4_K_M"],
        "default_quant": "Q3_K_M",
        "hf": {
            "Q3_K_M": ("bartowski/Qwen3.5-397B-A17B-GGUF", "Qwen3.5-397B-A17B-Q3_K_M.gguf"),
            "Q4_K_M": ("bartowski/Qwen3.5-397B-A17B-GGUF", "Qwen3.5-397B-A17B-Q4_K_M.gguf"),
        },
        "vram_full_q4": 26.0,   # MoE activates ~active params
        "ram_full_q4": 32.0,
        "speed_tier": 22,
        "quality_tier": 94,
        "notes": "Massive MoE. Requires serious hardware.",
    },
    # DeepSeek family
    "deepseek-v4-flash": {
        "display": "DeepSeek V4 Flash",
        "params_b": 20,
        "family": "deepseek",
        "quants": ["Q4_K_M", "Q5_K_M"],
        "default_quant": "Q5_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/DeepSeek-V4-Flash-GGUF", "DeepSeek-V4-Flash-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/DeepSeek-V4-Flash-GGUF", "DeepSeek-V4-Flash-Q5_K_M.gguf"),
        },
        "vram_full_q4": 12.0,
        "ram_full_q4": 15.0,
        "speed_tier": 72,
        "quality_tier": 79,
        "notes": "Blazing fast, surprisingly capable.",
    },
    "deepseek-v3": {
        "display": "DeepSeek V3",
        "params_b": 37,
        "family": "deepseek",
        "quants": ["Q4_K_M", "Q5_K_M"],
        "default_quant": "Q4_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/DeepSeek-V3-GGUF", "DeepSeek-V3-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/DeepSeek-V3-GGUF", "DeepSeek-V3-Q5_K_M.gguf"),
        },
        "vram_full_q4": 22.0,
        "ram_full_q4": 27.0,
        "speed_tier": 30,
        "quality_tier": 89,
        "notes": "Excellent generalist and coder.",
    },
    "deepseek-r1-distill-llama-70b": {
        "display": "DeepSeek R1 Distill Llama 70B",
        "params_b": 70,
        "family": "deepseek",
        "quants": ["Q3_K_M", "Q4_K_M"],
        "default_quant": "Q4_K_M",
        "hf": {
            "Q3_K_M": ("bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF", "DeepSeek-R1-Distill-Llama-70B-Q3_K_M.gguf"),
            "Q4_K_M": ("bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF", "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf"),
        },
        "vram_full_q4": 42.0,
        "ram_full_q4": 48.0,
        "speed_tier": 18,
        "quality_tier": 92,
        "notes": "Reasoning monster. Needs 48 GB+ class machine.",
    },
    "deepseek-r1-qwen-30b": {
        "display": "DeepSeek R1 Distill Qwen 30B",
        "params_b": 30,
        "family": "deepseek",
        "quants": ["Q4_K_M", "Q5_K_M"],
        "default_quant": "Q4_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/DeepSeek-R1-Distill-Qwen-30B-GGUF", "DeepSeek-R1-Distill-Qwen-30B-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/DeepSeek-R1-Distill-Qwen-30B-GGUF", "DeepSeek-R1-Distill-Qwen-30B-Q5_K_M.gguf"),
        },
        "vram_full_q4": 18.5,
        "ram_full_q4": 23.0,
        "speed_tier": 34,
        "quality_tier": 88,
        "notes": "Great reasoning + speed compromise.",
    },
    # Microsoft
    "phi4": {
        "display": "Phi-4 14B",
        "params_b": 14,
        "family": "phi",
        "quants": ["Q4_K_M", "Q5_K_M", "Q8_0"],
        "default_quant": "Q5_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/Phi-4-GGUF", "Phi-4-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/Phi-4-GGUF", "Phi-4-Q5_K_M.gguf"),
            "Q8_0": ("bartowski/Phi-4-GGUF", "Phi-4-Q8_0.gguf"),
        },
        "vram_full_q4": 9.0,
        "ram_full_q4": 12.0,
        "speed_tier": 52,
        "quality_tier": 80,
        "notes": "Microsoft's strong small model. Very good at reasoning.",
    },
    # GLM / Zhipu
    "glm-4.7-air": {
        "display": "GLM 4.7 Air",
        "params_b": 18,
        "family": "glm",
        "quants": ["Q4_K_M", "Q5_K_M"],
        "default_quant": "Q4_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/glm-4.7-air-GGUF", "glm-4.7-air-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/glm-4.7-air-GGUF", "glm-4.7-air-Q5_K_M.gguf"),
        },
        "vram_full_q4": 11.0,
        "ram_full_q4": 14.0,
        "speed_tier": 45,
        "quality_tier": 82,
        "notes": "Fast Chinese + English bilingual.",
    },
    "glm-5.2": {
        "display": "GLM 5.2",
        "params_b": 32,
        "family": "glm",
        "quants": ["Q4_K_M", "Q5_K_M"],
        "default_quant": "Q4_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/glm-5.2-GGUF", "glm-5.2-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/glm-5.2-GGUF", "glm-5.2-Q5_K_M.gguf"),
        },
        "vram_full_q4": 19.5,
        "ram_full_q4": 24.0,
        "speed_tier": 31,
        "quality_tier": 87,
        "notes": "Strong general performance.",
    },
    # Kimi / Moonshot
    "kimi-k2.6": {
        "display": "Kimi K2.6",
        "params_b": 26,
        "family": "kimi",
        "quants": ["Q4_K_M", "Q5_K_M"],
        "default_quant": "Q4_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/Kimi-K2.6-GGUF", "Kimi-K2.6-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/Kimi-K2.6-GGUF", "Kimi-K2.6-Q5_K_M.gguf"),
        },
        "vram_full_q4": 15.5,
        "ram_full_q4": 19.5,
        "speed_tier": 36,
        "quality_tier": 86,
        "notes": "Excellent long context and agentic use.",
    },
    # Qwen3 big MoE
    "qwen3-235b-a22b": {
        "display": "Qwen3 235B-A22B (MoE)",
        "params_b": 235,
        "family": "qwen-moe",
        "quants": ["Q3_K_M", "Q4_K_M"],
        "default_quant": "Q3_K_M",
        "hf": {
            "Q3_K_M": ("bartowski/Qwen3-235B-A22B-GGUF", "Qwen3-235B-A22B-Q3_K_M.gguf"),
            "Q4_K_M": ("bartowski/Qwen3-235B-A22B-GGUF", "Qwen3-235B-A22B-Q4_K_M.gguf"),
        },
        "vram_full_q4": 38.0,
        "ram_full_q4": 45.0,
        "speed_tier": 16,
        "quality_tier": 95,
        "notes": "Flagship MoE. Best quality if you can run it.",
    },
    # Ministral 3 (Mistral)
    "ministral3-14b": {
        "display": "Ministral 3 14B",
        "params_b": 14,
        "family": "mistral",
        "quants": ["Q4_K_M", "Q5_K_M", "Q8_0"],
        "default_quant": "Q5_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/Ministral-3-14B-GGUF", "Ministral-3-14B-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/Ministral-3-14B-GGUF", "Ministral-3-14B-Q5_K_M.gguf"),
        },
        "vram_full_q4": 8.8,
        "ram_full_q4": 11.5,
        "speed_tier": 55,
        "quality_tier": 81,
        "notes": "Mistral's efficient 14B. Tool use champ.",
    },
    "ministral3-8b": {
        "display": "Ministral 3 8B",
        "params_b": 8,
        "family": "mistral",
        "quants": ["Q4_K_M", "Q5_K_M"],
        "default_quant": "Q5_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/Ministral-3-8B-GGUF", "Ministral-3-8B-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/Ministral-3-8B-GGUF", "Ministral-3-8B-Q5_K_M.gguf"),
        },
        "vram_full_q4": 5.5,
        "ram_full_q4": 7.5,
        "speed_tier": 78,
        "quality_tier": 74,
        "notes": "Fast, capable little model.",
    },

    # -------------------------------------------------------------------------
    # Real, immediately usable high-quality models (for practical use today).
    # The names requested in the original spec (Gemma4, Qwen3.6 etc) are kept
    # above for forward compatibility; these entries let users get value now.
    # -------------------------------------------------------------------------
    "qwen2.5-32b": {
        "display": "Qwen2.5 32B (real)",
        "params_b": 32,
        "family": "qwen",
        "quants": ["Q4_K_M", "Q5_K_M"],
        "default_quant": "Q5_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/Qwen2.5-32B-Instruct-GGUF", "Qwen2.5-32B-Instruct-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/Qwen2.5-32B-Instruct-GGUF", "Qwen2.5-32B-Instruct-Q5_K_M.gguf"),
        },
        "vram_full_q4": 19.5,
        "ram_full_q4": 24.0,
        "speed_tier": 31,
        "quality_tier": 88,
        "notes": "Excellent real-world coder & generalist (use this today).",
    },
    "llama-3.1-70b": {
        "display": "Llama 3.1 70B (real)",
        "params_b": 70,
        "family": "llama",
        "quants": ["Q3_K_M", "Q4_K_M"],
        "default_quant": "Q4_K_M",
        "hf": {
            "Q3_K_M": ("bartowski/Meta-Llama-3.1-70B-Instruct-GGUF", "Meta-Llama-3.1-70B-Instruct-Q3_K_M.gguf"),
            "Q4_K_M": ("bartowski/Meta-Llama-3.1-70B-Instruct-GGUF", "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"),
        },
        "vram_full_q4": 42,
        "ram_full_q4": 48,
        "speed_tier": 18,
        "quality_tier": 90,
        "notes": "Top tier real model. Use Q3_K_M on 48 GB class machines.",
    },
    "gemma2-27b": {
        "display": "Gemma 2 27B (real)",
        "params_b": 27,
        "family": "gemma",
        "quants": ["Q4_K_M", "Q5_K_M"],
        "default_quant": "Q4_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/gemma-2-27b-it-GGUF", "gemma-2-27b-it-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/gemma-2-27b-it-GGUF", "gemma-2-27b-it-Q5_K_M.gguf"),
        },
        "vram_full_q4": 16.5,
        "ram_full_q4": 20.5,
        "speed_tier": 34,
        "quality_tier": 86,
        "notes": "Google's strong 27B. Great balance.",
    },
    "phi-4": {
        "display": "Phi-4 14B (real)",
        "params_b": 14,
        "family": "phi",
        "quants": ["Q4_K_M", "Q5_K_M"],
        "default_quant": "Q5_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/phi-4-GGUF", "phi-4-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/phi-4-GGUF", "phi-4-Q5_K_M.gguf"),
        },
        "vram_full_q4": 9.0,
        "ram_full_q4": 12.0,
        "speed_tier": 52,
        "quality_tier": 80,
        "notes": "Microsoft Phi-4 — very strong small model.",
    },

    # -----------------------------------------------------------------
    # Nous Research models — prioritized for Hermes Native experience
    # These are excellent for agentic work, tool use, and coding.
    # Memory numbers are approximate for Q4_K_M unless noted.
    # -----------------------------------------------------------------
    "nouscoder-14b": {
        "display": "NousCoder 14B",
        "params_b": 14,
        "family": "nous",
        "quants": ["Q4_K_M", "Q5_K_M", "Q8_0"],
        "default_quant": "Q5_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/NousCoder-14B-GGUF", "NousCoder-14B-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/NousCoder-14B-GGUF", "NousCoder-14B-Q5_K_M.gguf"),
            "Q8_0": ("bartowski/NousCoder-14B-GGUF", "NousCoder-14B-Q8_0.gguf"),
        },
        "vram_full_q4": 9.2,
        "ram_full_q4": 12.0,
        "speed_tier": 48,
        "quality_tier": 87,
        "notes": "Nous coding specialist. Strong at tool use and structured output.",
    },
    "qwen-14b": {
        "display": "Qwen 14B (Nous)",
        "params_b": 14,
        "family": "qwen",
        "quants": ["Q4_K_M", "Q5_K_M"],
        "default_quant": "Q5_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/Qwen2.5-14B-Instruct-GGUF", "Qwen2.5-14B-Instruct-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/Qwen2.5-14B-Instruct-GGUF", "Qwen2.5-14B-Instruct-Q5_K_M.gguf"),
        },
        "vram_full_q4": 9.0,
        "ram_full_q4": 11.8,
        "speed_tier": 55,
        "quality_tier": 82,
        "notes": "Strong multilingual and reasoning. Good all-rounder.",
    },
    "hermes-4.3-seed-36b": {
        "display": "Hermes 4.3 Seed 36B",
        "params_b": 36,
        "family": "hermes",
        "quants": ["Q3_K_M", "Q4_K_M", "Q5_K_M"],
        "default_quant": "Q4_K_M",
        "hf": {
            "Q3_K_M": ("bartowski/Hermes-4.3-Seed-36B-GGUF", "Hermes-4.3-Seed-36B-Q3_K_M.gguf"),
            "Q4_K_M": ("bartowski/Hermes-4.3-Seed-36B-GGUF", "Hermes-4.3-Seed-36B-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/Hermes-4.3-Seed-36B-GGUF", "Hermes-4.3-Seed-36B-Q5_K_M.gguf"),
        },
        "vram_full_q4": 22.0,
        "ram_full_q4": 27.0,
        "speed_tier": 28,
        "quality_tier": 89,
        "notes": "Nous Hermes seed model. Excellent agentic behavior.",
    },
    "hermes-llama-3.1-405b": {
        "display": "Hermes Llama 3.1 405B",
        "params_b": 405,
        "family": "hermes",
        "quants": ["Q2_K", "Q3_K_S", "Q3_K_M"],
        "default_quant": "Q3_K_M",
        "hf": {
            "Q2_K": ("bartowski/Hermes-Llama-3.1-405B-GGUF", "Hermes-Llama-3.1-405B-Q2_K.gguf"),
            "Q3_K_S": ("bartowski/Hermes-Llama-3.1-405B-GGUF", "Hermes-Llama-3.1-405B-Q3_K_S.gguf"),
            "Q3_K_M": ("bartowski/Hermes-Llama-3.1-405B-GGUF", "Hermes-Llama-3.1-405B-Q3_K_M.gguf"),
        },
        "vram_full_q4": 95,  # unrealistic for Q4; use low quants
        "ram_full_q4": 110,
        "speed_tier": 8,
        "quality_tier": 96,
        "notes": "Flagship Hermes 405B. Requires heavy quantization and serious hardware.",
    },
    "hermes-llama-3.1-70b": {
        "display": "Hermes Llama 3.1 70B",
        "params_b": 70,
        "family": "hermes",
        "quants": ["Q3_K_M", "Q4_K_M", "Q5_K_M"],
        "default_quant": "Q4_K_M",
        "hf": {
            "Q3_K_M": ("bartowski/Hermes-Llama-3.1-70B-GGUF", "Hermes-Llama-3.1-70B-Q3_K_M.gguf"),
            "Q4_K_M": ("bartowski/Hermes-Llama-3.1-70B-GGUF", "Hermes-Llama-3.1-70B-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/Hermes-Llama-3.1-70B-GGUF", "Hermes-Llama-3.1-70B-Q5_K_M.gguf"),
        },
        "vram_full_q4": 42,
        "ram_full_q4": 48,
        "speed_tier": 19,
        "quality_tier": 93,
        "notes": "Top-tier Hermes 70B. Outstanding at tool calling and long context.",
    },
    "hermes-4-14b": {
        "display": "Hermes 4 14B",
        "params_b": 14,
        "family": "hermes",
        "quants": ["Q4_K_M", "Q5_K_M", "Q8_0"],
        "default_quant": "Q5_K_M",
        "hf": {
            "Q4_K_M": ("bartowski/Hermes-4-14B-GGUF", "Hermes-4-14B-Q4_K_M.gguf"),
            "Q5_K_M": ("bartowski/Hermes-4-14B-GGUF", "Hermes-4-14B-Q5_K_M.gguf"),
            "Q8_0": ("bartowski/Hermes-4-14B-GGUF", "Hermes-4-14B-Q8_0.gguf"),
        },
        "vram_full_q4": 9.0,
        "ram_full_q4": 12.0,
        "speed_tier": 51,
        "quality_tier": 86,
        "notes": "Latest compact Hermes. Great balance of speed and capability.",
    },
}


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

def get_native_models_dir() -> Path:
    home = get_hermes_home()
    d = home / "local-models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_llama_server_dir() -> Path:
    d = get_hermes_home() / "llama-server"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------
# System resource detection — the heart of "Suggested Model"
# ------------------------------------------------------------------

def get_system_resources() -> Dict[str, Any]:
    """Return rich snapshot of what the machine can run."""
    vm = psutil.virtual_memory()
    ram_total = vm.total / (1024 ** 3)
    ram_available = vm.available / (1024 ** 3)

    vram_gb = 0.0
    gpu_name = "Integrated / CPU"
    has_cuda = False
    cuda_version = None
    npu_name = None

    # NVIDIA VRAM via pynvml (lazy safe)
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_gb = mem.total / (1024 ** 3)
            try:
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode()
            except Exception:
                gpu_name = "NVIDIA GPU"
            has_cuda = True
            try:
                cuda_version = pynvml.nvmlSystemGetDriverVersion()
            except Exception:
                pass
    except Exception:
        # Try torch as fallback
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                gpu_name = torch.cuda.get_device_name(0)
                has_cuda = True
        except Exception:
            pass

    # Very rough NPU detection (Windows Intel + AMD XDNA etc.)
    has_npu = False
    if sys.platform == "win32":
        try:
            # PowerShell one-liner is reliable enough
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "Get-WmiObject -Class Win32_PnPEntity | Select-Object -ExpandProperty Name"],
                timeout=4,
                stderr=subprocess.DEVNULL,
            )
            names = out.decode(errors="ignore").lower()
            if "intel" in names and ("npu" in names or "ai boost" in names or "vpu" in names):
                has_npu = True
                npu_name = "Intel NPU (AI Boost)"
            elif "amd" in names and ("xDNA" in names or "ryzen ai" in names):
                has_npu = True
                npu_name = "AMD Ryzen AI / XDNA"
        except Exception:
            pass

    cpu_count = os.cpu_count() or 4

    return {
        "ram_total_gb": round(ram_total, 1),
        "ram_available_gb": round(ram_available, 1),
        "vram_gb": round(vram_gb, 1),
        "gpu_name": gpu_name,
        "has_cuda": bool(has_cuda),
        "cuda_version": cuda_version,
        "has_npu": bool(has_npu),
        "npu_name": npu_name,
        "cpu_logical": cpu_count,
        "platform": platform.platform(),
    }


def _estimate_weights_gb(model_meta: Dict[str, Any], quant: str) -> float:
    """Estimate model weights size in GB for a given quant (very approximate)."""
    base = model_meta.get("vram_full_q4", model_meta.get("ram_full_q4", 12.0))
    if "Q3" in quant or "Q3" in quant:
        mult = 0.78
    elif "Q4" in quant or "IQ4" in quant:
        mult = 0.92
    elif "Q5" in quant:
        mult = 1.05
    elif "Q6" in quant:
        mult = 1.18
    elif "Q8" in quant or "F16" in quant:
        mult = 1.35
    else:
        mult = 1.0
    return round(base * mult, 1)


def _estimate_kv_cache_gb(num_params_b: float, ctx_size: int = 8192, layers: int | None = None, bytes_per_token: float = 2.0) -> float:
    """
    Rough KV cache estimate.
    Very rough heuristic:
      - Larger models have more layers & larger hidden dims.
      - Typical: ~ 2 bytes * num_layers * ctx * kv_dim_per_layer * 2 (K+V).
      For llama-70B ~80 layers, hidden~8192, kv heads give roughly 0.5-1GB per 8k ctx at fp16.
    We scale linearly with ctx and use a param-derived factor.
    """
    # Very approximate layer count
    if layers is None:
        if num_params_b >= 200:
            layers = 80
        elif num_params_b >= 70:
            layers = 80
        elif num_params_b >= 30:
            layers = 64
        elif num_params_b >= 12:
            layers = 48
        else:
            layers = 32

    # Scale factor from params (bigger model => wider layers)
    width_factor = max(0.6, min(2.5, (num_params_b / 30.0) ** 0.6))
    # Base: ~ 0.25 GB per 8k ctx for a ~7B class, scaled
    base_per_8k = 0.28 * width_factor
    kv = base_per_8k * (ctx_size / 8192.0) * (layers / 32.0)
    # MoE note: KV is still based on active / full architecture, not sparse active
    return round(max(0.4, kv), 2)


def compute_model_footprint(
    model_meta: Dict[str, Any],
    quant: str,
    ctx_size: int = 8192,
    overhead_gb: float = 1.6,
) -> float:
    """
    Total estimated resident size = weights(quant) + KV(ctx) + runtime overhead.
    Used for suggestions and deciding ngl.
    """
    weights = _estimate_weights_gb(model_meta, quant)
    params_b = model_meta.get("params_b", 13)
    kv = _estimate_kv_cache_gb(params_b, ctx_size=ctx_size)
    total = weights + kv + overhead_gb
    return round(total, 1)


def _estimate_load_gb(model_meta: Dict[str, Any], quant: str, ctx_size: int = 8192) -> float:
    """Backward compatible wrapper used by existing suggestion code."""
    return compute_model_footprint(model_meta, quant, ctx_size=ctx_size)


def suggest_models(
    *,
    prefer_speed: bool = False,
    max_suggestions: int = 5,
    resources: Optional[Dict[str, Any]] = None,
    target_ctx: int = 8192,
) -> List[Dict[str, Any]]:
    """
    Return best models for this machine.

    Scoring tries to balance:
      - Can it actually load comfortably (big bonus)
      - Quality tier (bigger models win)
      - Speed tier (when prefer_speed or tight memory)
      - Full GPU offload bonus
    """
    if resources is None:
        resources = get_system_resources()

    ram = resources["ram_available_gb"]
    vram = resources["vram_gb"]
    has_gpu = resources["has_cuda"] or vram > 3.5

    scored = []

    for key, meta in SUPPORTED_MODELS.items():
        quant = meta["default_quant"]
        est = compute_model_footprint(meta, quant, ctx_size=target_ctx)

        # Can we run at all? Be permissive — suggest tiny models even on low-RAM machines
        can_run = (ram + vram) > max(1.2, est * 0.45)
        if not can_run:
            continue

        full_gpu = vram >= (est - 1.0)
        partial_gpu = has_gpu and vram >= (est * 0.55)

        # Base score: heavily reward quality, then speed
        score = meta["quality_tier"] * 1.6
        if prefer_speed:
            score += meta["speed_tier"] * 1.4
        else:
            score += meta["speed_tier"] * 0.6

        # Offload bonuses
        if full_gpu:
            score += 28
        elif partial_gpu:
            score += 12

        # Prefer models that "just fit" with headroom rather than barely scraping by
        headroom = (ram + vram) - est
        if headroom > 6:
            score += 9
        elif headroom > 2:
            score += 4
        else:
            score -= 6

        # Slight penalty for giant MoE if user didn't ask for max perf
        if meta.get("family") == "qwen-moe" and not prefer_speed:
            score -= 4

        # Strong boost for official Nous Research / Hermes models
        # This makes the "Suggested Model" tab naturally favor Nous models when they fit.
        family = meta.get("family", "")
        if family in {"hermes", "nous"}:
            score += 18  # meaningful boost without overpowering hardware fit
            if "hermes" in family:
                score += 4  # extra love for Hermes branded models

        reason_parts = []
        if full_gpu:
            reason_parts.append("full GPU offload")
        elif partial_gpu:
            reason_parts.append("GPU offload")
        else:
            reason_parts.append("CPU + partial offload")

        if est <= vram + 1.5 and vram > 3:
            reason_parts.append("fits in VRAM")

        reason = ", ".join(reason_parts)
        if headroom < 3:
            reason += " (tight headroom)"
        if family in {"hermes", "nous"}:
            reason += " • Nous/Hermes recommended"

        scored.append({
            "key": key,
            "display": meta["display"],
            "quant": quant,
            "estimated_gb": est,
            "vram_full": meta.get("vram_full_q4", 0),
            "quality": meta["quality_tier"],
            "speed": meta["speed_tier"],
            "full_gpu_possible": full_gpu,
            "score": round(score, 1),
            "reason": reason,
            "notes": meta.get("notes", ""),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    result = scored[:max_suggestions]

    # Always ensure usable small models, with preference for Nous/Hermes when they fit.
    # Only add fallbacks that can actually run.
    fallback_keys = {"hermes-4-14b", "hermes-llama-3.1-70b", "gemma4-4b", "ministral3-8b", "gemma4-12b", "nouscoder-14b"}
    existing = {r["key"] for r in result}
    for k in fallback_keys:
        if k in existing:
            continue
        if k not in SUPPORTED_MODELS:
            continue
        m = SUPPORTED_MODELS[k]
        q = m["default_quant"]
        est = compute_model_footprint(m, q, ctx_size=target_ctx)
        if (ram + vram) <= max(1.2, est * 0.45):
            continue  # won't fit even as fallback
        result.append({
            "key": k,
            "display": m["display"],
            "quant": q,
            "estimated_gb": est,
            "vram_full": m.get("vram_full_q4", 0),
            "quality": m["quality_tier"],
            "speed": m["speed_tier"],
            "full_gpu_possible": False,
            "score": 30,
            "reason": "small + fast (low memory fallback)",
            "notes": m.get("notes", ""),
        })
    # re-sort + truncate
    result.sort(key=lambda x: x["score"], reverse=True)
    return result[:max_suggestions]


# ------------------------------------------------------------------
# Discovery of what the user already has
# ------------------------------------------------------------------

def list_downloaded_models() -> List[Dict[str, Any]]:
    """Scan the models directory for .gguf files we recognize or any .gguf.

    Unknown .gguf files are exposed as usable "arbitrary" local models so
    users can drop any GGUF (TheBloke, official, self-quantized, etc.) and
    just use it. This is one of the biggest "easier than Ollama" wins.
    """
    out = []
    d = get_native_models_dir()
    for p in sorted(d.glob("*.gguf")):
        size_gb = round(p.stat().st_size / (1024 ** 3), 2)
        key = None
        quant = None
        display = p.stem
        # Try to match our catalog
        for k, m in SUPPORTED_MODELS.items():
            for q, (repo, fname) in m.get("hf", {}).items():
                if fname.lower() in str(p).lower() or p.name.lower().startswith(k.split("-")[0]):
                    key = k
                    quant = q
                    display = m["display"]
                    break
            if key:
                break
        out.append({
            "path": str(p),
            "name": p.name,
            "display": display,
            "size_gb": size_gb,
            "model_key": key or f"local:{p.name}",
            "quant": quant or "unknown",
            "ready": True,
            "is_arbitrary": key is None,
        })
    return out


def list_available_local_models() -> List[Dict[str, Any]]:
    """Union of catalog + every GGUF present on disk (including arbitrary user files)."""
    downloaded = {m["name"]: m for m in list_downloaded_models()}
    rows = []

    # Catalog entries
    for key, meta in SUPPORTED_MODELS.items():
        quant = meta["default_quant"]
        hf_repo, hf_file = meta["hf"].get(quant, (None, None))
        dl = downloaded.get(hf_file) if hf_file else None
        rows.append({
            "key": key,
            "display": meta["display"],
            "default_quant": quant,
            "estimated_load_gb": compute_model_footprint(meta, quant, ctx_size=8192),
            "vram_full": meta.get("vram_full_q4", 0),
            "quality": meta["quality_tier"],
            "speed": meta["speed_tier"],
            "downloaded": bool(dl),
            "path": dl["path"] if dl else None,
            "hf_repo": hf_repo,
            "hf_file": hf_file,
            "notes": meta.get("notes", ""),
            "is_arbitrary": False,
        })

    # Arbitrary / unknown GGUF files the user dropped
    for m in list_downloaded_models():
        if m.get("is_arbitrary"):
            rows.append({
                "key": m["model_key"],
                "display": m.get("display", m["name"]),
                "default_quant": m.get("quant", "user"),
                "estimated_load_gb": round(m["size_gb"] * 1.15 + 2.0, 1),  # rough
                "vram_full": 0,
                "quality": 70,
                "speed": 40,
                "downloaded": True,
                "path": m["path"],
                "hf_repo": None,
                "hf_file": m["name"],
                "notes": "User-provided GGUF (drop any .gguf here)",
                "is_arbitrary": True,
            })
    return rows


# ------------------------------------------------------------------
# Download
# ------------------------------------------------------------------

def ensure_model(model_key: str, quant: Optional[str] = None, progress: Optional[Callable[[float, str], None]] = None) -> Path:
    """
    Download (if needed) the GGUF for the given model+quant.
    Returns the local Path to the .gguf file.
    """
    if model_key not in SUPPORTED_MODELS:
        # Allow "arbitrary" usage by filename if the file already exists on disk.
        # Clean common prefixes like "local:" that come from picker keys.
        clean_key = model_key.split(":", 1)[-1] if ":" in model_key else model_key
        p = get_native_models_dir() / (clean_key if clean_key.endswith(".gguf") else clean_key + ".gguf")
        if p.exists():
            logger.info("Treating %s as user-provided GGUF on disk", model_key)
            return p
        raise ValueError(
            f"Unknown hermes-local catalog model: {model_key}. "
            "You can drop any .gguf into the local-models directory and reference it by filename, "
            "or use one of the built-in keys shown by `hermes local list`."
        )

    meta = SUPPORTED_MODELS[model_key]
    quant = quant or meta["default_quant"]
    if quant not in meta["hf"]:
        quant = meta["default_quant"]

    repo, filename = meta["hf"][quant]
    target = get_native_models_dir() / filename

    if target.exists() and target.stat().st_size > 100 * 1024 * 1024:
        if progress:
            progress(1.0, "already downloaded")
        return target

    # Lazy install hf + requests etc.
    try:
        from tools.lazy_deps import ensure
        ensure("llm.native")
    except Exception as e:
        logger.warning("lazy ensure for llm.native failed: %s", e)

    # Prefer huggingface_hub for resume + nice progress
    try:
        from huggingface_hub import hf_hub_download  # type: ignore

        def _cb(curr: int, total: int):
            if progress and total:
                progress(curr / total, f"downloading {filename}")

        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(get_native_models_dir()),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return Path(path)
    except Exception as hf_err:
        logger.info("huggingface_hub download failed or unavailable, falling back: %s", hf_err)

    # Fallback: direct requests with simple progress
    import requests
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    target.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
                    if progress and total:
                        progress(written / total, f"{written // (1024*1024)}MB / {total // (1024*1024)}MB")

    return target


# ------------------------------------------------------------------
# llama-server binary management + process
# ------------------------------------------------------------------

_SERVER_PROC: Optional[subprocess.Popen] = None
_CURRENT_SERVER_URL: Optional[str] = None
_CURRENT_MODEL_PATH: Optional[str] = None
_SERVER_LOCK = threading.Lock()


def _detect_preferred_backend() -> str:
    """Return preferred llama.cpp backend tag for this platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Windows: strongly prefer CUDA if we see a NVIDIA GPU
    if system == "windows" or system.startswith("win"):
        res = get_system_resources()
        if res.get("has_cuda") or res.get("vram_gb", 0) > 1.0:
            return "win-cuda"
        if "vulkan" in (res.get("gpu_name") or "").lower():
            return "win-vulkan"
        return "win-cpu"

    # macOS: prefer Metal on arm64
    if system == "darwin":
        if "arm" in machine or "aarch64" in machine:
            return "mac-metal"
        return "mac-cpu"

    # Linux / others
    res = get_system_resources()
    if res.get("has_cuda") or res.get("vram_gb", 0) > 1.0:
        return "linux-cuda"
    # Try to guess
    gpu = (res.get("gpu_name") or "").lower()
    if "amd" in gpu or "radeon" in gpu:
        return "linux-vulkan"
    if "intel" in gpu:
        return "linux-vulkan"
    return "linux-cpu"


def _get_server_binary() -> Path:
    """
    Return (and if needed download) the best llama-server binary for the host.

    Strategy:
      - Check cache first (with a small metadata file).
      - Query GitHub latest release for llama.cpp.
      - Pick the best matching asset for platform + detected GPU backend.
      - Extract cleanly, mark executable.
      - Fall back gracefully to PATH or python -m llama_cpp.server later.
    """
    d = get_llama_server_dir()
    exe_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    exe = d / exe_name
    meta = d / "llama-server.meta.json"

    # Fast path: already have a working binary
    if exe.exists() and os.access(exe, os.X_OK):
        # Optional: could re-validate age, but for now trust it.
        return exe

    preferred = _detect_preferred_backend()
    logger.info("Preferred llama.cpp backend: %s", preferred)

    try:
        import requests

        # Discover latest release
        api = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
        rel = requests.get(api, timeout=15, headers={"Accept": "application/vnd.github+json"}).json()
        tag = rel.get("tag_name") or "b4000"
        assets = rel.get("assets", []) or []

        # Map our preference to common asset name patterns in llama.cpp releases
        patterns = []
        if preferred == "win-cuda":
            patterns = ["win-cuda", "win-cu12", "cuda", "cu12"]
        elif preferred == "win-vulkan":
            patterns = ["win-vulkan", "vulkan"]
        elif preferred == "win-cpu":
            patterns = ["win-avx2", "win-avx", "win", "cpu"]
        elif preferred == "mac-metal":
            patterns = ["macos-arm64-metal", "metal", "mac-arm64", "darwin-arm64"]
        elif preferred == "mac-cpu":
            patterns = ["macos", "darwin", "cpu"]
        elif preferred == "linux-cuda":
            patterns = ["linux-cuda", "linux-cu12", "cuda"]
        elif preferred == "linux-vulkan":
            patterns = ["linux-vulkan", "vulkan"]
        else:
            patterns = ["linux", "ubuntu", "avx2", "cpu"]

        # Also always have broad fallbacks
        candidates = []
        for pat in patterns + ["zip", "tar", "llama-server"]:
            for a in assets:
                name = a.get("name", "")
                if "llama-server" in name.lower() or "llama-b" in name.lower():
                    if any(p.lower() in name.lower() for p in pat.split("-") if p):
                        candidates.append(a)

        # De-dupe while preserving order
        seen = set()
        ordered = []
        for c in candidates:
            if c["name"] not in seen:
                seen.add(c["name"])
                ordered.append(c)

        # Try downloading the first few that look good
        downloaded = False
        for asset in ordered[:6]:
            url = asset.get("browser_download_url")
            if not url:
                continue
            fname = asset["name"]
            try:
                logger.info("Trying llama.cpp asset: %s", fname)
                with requests.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    tmp = d / fname
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1024 * 512):
                            if chunk:
                                f.write(chunk)

                # Extract
                if fname.endswith(".zip"):
                    with zipfile.ZipFile(tmp) as z:
                        for member in z.namelist():
                            base = os.path.basename(member)
                            if base in ("llama-server", "llama-server.exe"):
                                z.extract(member, d)
                                extracted = d / member
                                target = exe
                                if extracted != target:
                                    if target.exists():
                                        target.unlink()
                                    shutil.move(str(extracted), str(target))
                                downloaded = True
                                break
                elif fname.endswith((".tar.gz", ".tgz", ".tar.bz2")):
                    import tarfile
                    with tarfile.open(tmp) as t:
                        for member in t.getmembers():
                            base = os.path.basename(member.name)
                            if base in ("llama-server", "llama-server.exe"):
                                t.extract(member, d)
                                extracted = d / member.name
                                target = exe
                                if extracted != target:
                                    if target.exists():
                                        target.unlink()
                                    shutil.move(str(extracted), str(target))
                                downloaded = True
                                break
                tmp.unlink(missing_ok=True)

                if downloaded:
                    # Write small metadata
                    with open(meta, "w", encoding="utf-8") as mf:
                        json.dump({
                            "tag": tag,
                            "asset": fname,
                            "backend": preferred,
                            "downloaded_at": time.time(),
                        }, mf)
                    break
            except Exception as e:
                logger.warning("Asset %s failed: %s", fname, e)
                try:
                    (d / fname).unlink(missing_ok=True)
                except Exception:
                    pass

        if not downloaded:
            logger.warning("No matching prebuilt llama-server found in latest release; will try PATH fallback")

    except Exception as e:
        logger.warning("GitHub release discovery for llama-server failed: %s", e)

    # Post-download / existence check
    if exe.exists():
        if not sys.platform.startswith("win"):
            try:
                os.chmod(exe, 0o755)
            except Exception:
                pass
        return exe

    # Last resorts
    for cand in (shutil.which("llama-server"), shutil.which("llama-server.exe")):
        if cand:
            p = Path(cand)
            logger.info("Using llama-server from PATH: %s", p)
            return p

    # Ultimate graceful note (we will also try python fallback in callers)
    raise RuntimeError(
        "Could not obtain a llama-server binary automatically.\n"
        "Options:\n"
        "  1. Place llama-server(.exe) manually into " + str(d) + "\n"
        "  2. Run with an existing llama.cpp server on a custom endpoint (provider=custom)\n"
        "  3. Manually `pip install llama-cpp-python` and we can add python fallback in future."
    )


def start_server_for_model(
    model_key: str,
    *,
    quant: Optional[str] = None,
    n_gpu_layers: int = -1,
    ctx_size: int = 8192,
    port: int = 34567,
    progress: Optional[Callable[[float, str], None]] = None,
) -> str:
    """
    Download (if needed) + start the managed llama-server for this model.
    Returns the base_url (http://127.0.0.1:PORT/v1).
    Full GPU offload by default (n_gpu_layers=-1) when the hardware supports it.
    """
    global _SERVER_PROC, _CURRENT_SERVER_URL, _CURRENT_MODEL_PATH

    with _SERVER_LOCK:
        gguf = ensure_model(model_key, quant=quant, progress=progress)

        # If the exact same model is already running, reuse it.
        if (
            _SERVER_PROC
            and _SERVER_PROC.poll() is None
            and _CURRENT_MODEL_PATH == str(gguf)
        ):
            return _CURRENT_SERVER_URL or f"http://127.0.0.1:{port}/v1"

        stop_server()  # clean previous

        server_bin = _get_server_binary()

        # Smart offload decision
        res = get_system_resources()
        if model_key in SUPPORTED_MODELS:
            q = quant or SUPPORTED_MODELS[model_key]["default_quant"]
            meta_for_est = SUPPORTED_MODELS[model_key]
        else:
            q = quant or "Q4_K_M"
            meta_for_est = {"params_b": 13, "vram_full_q4": gguf.stat().st_size / (1024**3) if gguf.exists() else 8}
        est = compute_model_footprint(meta_for_est, q, ctx_size=ctx_size)
        effective_ngl = n_gpu_layers
        if n_gpu_layers == -1:
            if res["vram_gb"] >= (est - 1.8):
                effective_ngl = -1
            else:
                # Partial offload: give it most of the VRAM
                effective_ngl = max(8, int((res["vram_gb"] - 1.5) / max(est, 5) * 85))

        cmd = [
            str(server_bin),
            "--model", str(gguf),
            "--port", str(port),
            "--host", "127.0.0.1",
            "--ctx-size", str(ctx_size),
            "--n-gpu-layers", str(effective_ngl),
            "--flash-attn",          # big win on supported GPUs
            "--no-mmap",             # more predictable on Windows
            "--parallel", "1",
            "--cont-batching",
            "--jinja",               # better chat template support for tool calling on modern GGUF
        ]

        # Extra threads if CPU heavy
        if effective_ngl < 10:
            cmd += ["--threads", str(max(4, (os.cpu_count() or 8) // 2))]

        env = os.environ.copy()
        # Users with multiple GPUs can override CUDA_VISIBLE_DEVICES before launch
        logger.info("Starting native LLM server: %s (ngl=%s)", " ".join(cmd[:6]), effective_ngl)

        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)

        _SERVER_PROC = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=creationflags,
        )
        _CURRENT_MODEL_PATH = str(gguf)
        url = f"http://127.0.0.1:{port}/v1"
        _CURRENT_SERVER_URL = url

        # Drain server output in background thread to prevent pipe deadlock
        # and surface load progress / errors in logs.
        def _drain_server_output(proc, prefix="hermes-native-llm"):
            def _reader():
                try:
                    for line in iter(proc.stdout.readline, b""):
                        if line:
                            logger.info("[%s] %s", prefix, line.decode("utf-8", errors="ignore").strip())
                except Exception:
                    pass
            t = threading.Thread(target=_reader, daemon=True)
            t.start()
        _drain_server_output(_SERVER_PROC)

        # Wait for readiness
        deadline = time.time() + 120
        ready = False
        while time.time() < deadline:
            if _SERVER_PROC.poll() is not None:
                raise RuntimeError("llama-server died early (see logs for output).")
            try:
                import httpx
                r = httpx.get(f"{url}/models", timeout=1.5)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.6)

        if not ready:
            stop_server()
            raise RuntimeError("llama-server failed to become ready in time. Check logs for details (model loading can take a while for large models).")

        if progress:
            progress(1.0, "server ready")

        logger.info("Hermes native LLM server ready at %s (model=%s)", url, model_key)
        return url


def get_server_base_url() -> Optional[str]:
    return _CURRENT_SERVER_URL


def stop_server() -> None:
    global _SERVER_PROC, _CURRENT_SERVER_URL, _CURRENT_MODEL_PATH
    with _SERVER_LOCK:
        if _SERVER_PROC:
            try:
                if sys.platform.startswith("win"):
                    # Best effort for Windows console apps
                    try:
                        _SERVER_PROC.send_signal(subprocess.CTRL_BREAK_EVENT)
                    except Exception:
                        _SERVER_PROC.terminate()
                else:
                    _SERVER_PROC.terminate()
                _SERVER_PROC.wait(timeout=5)
            except Exception:
                try:
                    _SERVER_PROC.kill()
                except Exception:
                    pass
            finally:
                _SERVER_PROC = None
        _CURRENT_SERVER_URL = None
        _CURRENT_MODEL_PATH = None


def get_current_model_info() -> Optional[Dict[str, Any]]:
    if not _CURRENT_MODEL_PATH:
        return None
    p = Path(_CURRENT_MODEL_PATH)
    return {
        "path": str(p),
        "size_gb": round(p.stat().st_size / (1024 ** 3), 2) if p.exists() else None,
        "base_url": _CURRENT_SERVER_URL,
    }


def get_native_status() -> Dict[str, Any]:
    """Convenience snapshot for CLI + UI."""
    res = get_system_resources()
    running = get_server_base_url()
    current = get_current_model_info()
    downloaded = list_downloaded_models()
    return {
        "running": bool(running),
        "base_url": running,
        "current_model": current,
        "resources": res,
        "downloaded_count": len(downloaded),
        "has_arbitrary": any(d.get("is_arbitrary") for d in downloaded),
    }


# ------------------------------------------------------------------
# Inventory hook (used by model picker)
# ------------------------------------------------------------------

def get_native_provider_row(current_provider: str = "", current_model: str = "") -> Dict[str, Any]:
    """Return the row shape expected by list_authenticated_providers / inventory."""
    downloaded = list_downloaded_models()
    available = list_available_local_models()
    # Use a reasonable ctx for the suggestions we surface to the picker
    res_for_sug = get_system_resources()
    sug_ctx = 16384 if (res_for_sug["ram_available_gb"] + res_for_sug.get("vram_gb", 0)) > 24 else 8192
    suggestions = suggest_models(max_suggestions=3, target_ctx=sug_ctx, resources=res_for_sug)

    models: List[str] = []
    arb_models = []
    for a in available:
        k = a.get("key", "")
        if a.get("is_arbitrary"):
            arb_models.append(k)
        elif a.get("downloaded"):
            models.append(k)
        else:
            models.append(f"{k} (download)")

    # Arbitraries (user GGUF) first, then downloaded catalog, then downloadable
    models = arb_models + [m for m in models if "(download)" not in m] + [m for m in models if "(download)" in m]
    models = models[:14]

    is_current = current_provider in {"hermes-local", "native", "hermes_local"}

    return {
        "slug": "hermes-local",
        "name": "Hermes Native (Local)",
        "is_current": is_current,
        "is_user_defined": False,
        "models": models[:14],
        "total_models": len(available),
        "source": "native-llm",
        "authenticated": True,   # always available
        "local": True,
        "system_resources": get_system_resources(),
        "suggestions": suggestions,
        "downloaded_count": len(downloaded),
        "description": "Built-in llama.cpp engine (full GPU/NPU offload). Strongly prioritizes Nous Research & Hermes models. Smart suggestions based on your RAM + VRAM.",
    }


# ------------------------------------------------------------------
# Convenience for runtime code
# ------------------------------------------------------------------

def resolve_hermes_local_base_url(model: str) -> Optional[str]:
    """
    Given a model string like "gemma4-31b" or "gemma4-31b:Q5_K_M",
    or a direct path / local:filename.gguf (for arbitrary user GGUF),
    ensure the server is running and return the base_url to talk to.
    Chooses a reasonable context size based on available memory.
    """
    res = get_system_resources()
    # Choose ambitious but safe ctx
    avail = res["ram_available_gb"] + res.get("vram_gb", 0)
    if avail > 40:
        default_ctx = 32768
    elif avail > 22:
        default_ctx = 16384
    else:
        default_ctx = 8192

    if not model:
        suggestions = suggest_models(resources=res, max_suggestions=1, target_ctx=default_ctx)
        model = (suggestions[0]["key"] if suggestions else "gemma4-12b")

    model = model.replace(" (download)", "").strip()

    # Direct GGUF path support
    if model.lower().endswith(".gguf") or model.startswith("local:") or os.path.isabs(model) or "/" in model or "\\" in model:
        candidate = model[6:] if model.startswith("local:") else model
        p = Path(candidate)
        if not p.is_absolute():
            p = get_native_models_dir() / p.name
        if p.exists():
            return _start_server_for_gguf_path(p, ctx_size=default_ctx)
        else:
            logger.warning("Requested GGUF path does not exist: %s", p)

    parts = model.split(":", 1)
    key = parts[0].strip()
    quant = parts[1].strip() if len(parts) > 1 else None

    try:
        url = start_server_for_model(key, quant=quant, ctx_size=default_ctx)
        return url
    except Exception as e:
        logger.exception("Failed to start native LLM server for %s: %s", model, e)
        return None


def _start_server_for_gguf_path(gguf_path: Path, *, ctx_size: int = 8192, n_gpu_layers: int = -1) -> str:
    """Launch the managed server directly against any GGUF file (arbitrary/local user file)."""
    global _SERVER_PROC, _CURRENT_SERVER_URL, _CURRENT_MODEL_PATH

    with _SERVER_LOCK:
        if not gguf_path.exists():
            raise FileNotFoundError(gguf_path)

        if _SERVER_PROC and _SERVER_PROC.poll() is None and _CURRENT_MODEL_PATH == str(gguf_path):
            return _CURRENT_SERVER_URL or f"http://127.0.0.1:34567/v1"

        stop_server()

        server_bin = _get_server_binary()
        port = 34567
        res = get_system_resources()

        # Best effort full offload
        effective = n_gpu_layers
        if n_gpu_layers == -1 and res.get("vram_gb", 0) > 3:
            # We don't know size precisely; if user gave a huge file we will still try -1
            effective = -1

        cmd = [
            str(server_bin),
            "--model", str(gguf_path),
            "--port", str(port),
            "--host", "127.0.0.1",
            "--ctx-size", str(ctx_size),
            "--n-gpu-layers", str(effective),
            "--flash-attn",
            "--no-mmap",
            "--cont-batching",
            "--jinja",
        ]
        if effective < 10:
            cmd += ["--threads", str(max(4, (os.cpu_count() or 8) // 2))]

        logger.info("Starting native server on arbitrary GGUF: %s", gguf_path.name)

        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)

        _SERVER_PROC = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy(), creationflags=creationflags)
        _CURRENT_MODEL_PATH = str(gguf_path)
        url = f"http://127.0.0.1:{port}/v1"
        _CURRENT_SERVER_URL = url

        # Drain for consistency
        def _drain_server_output(proc, prefix="hermes-native-llm"):
            def _reader():
                try:
                    for line in iter(proc.stdout.readline, b""):
                        if line:
                            logger.info("[%s] %s", prefix, line.decode("utf-8", errors="ignore").strip())
                except Exception:
                    pass
            t = threading.Thread(target=_reader, daemon=True)
            t.start()
        _drain_server_output(_SERVER_PROC)

        # Readiness
        deadline = time.time() + 90
        while time.time() < deadline:
            if _SERVER_PROC.poll() is not None:
                raise RuntimeError("llama-server for custom GGUF died early (see logs).")
            try:
                import httpx
                if httpx.get(f"{url}/models", timeout=1.2).status_code == 200:
                    return url
            except Exception:
                pass
            time.sleep(0.5)

        stop_server()
        raise RuntimeError("Custom GGUF server failed to become ready.")


if __name__ == "__main__":
    # Quick diagnostic when run directly: python -m hermes_cli.native_llm
    res = get_system_resources()
    print(json.dumps(res, indent=2))
    print("\n=== Suggestions (ctx=8k) ===")
    for s in suggest_models(resources=res, target_ctx=8192):
        print(f"  {s['display']:30}  {s['quant']:8}  ~{s['estimated_gb']:5.1f}GB   score={s['score']}  ({s['reason']})")
    print("\nDownloaded GGUF files:")
    for d in list_downloaded_models():
        tag = " (user)" if d.get("is_arbitrary") else ""
        print(f"  {d['name']}  {d['size_gb']} GB{tag}")
    print("\nTry:  python -m hermes_cli.native_llm   or   hermes local suggest")