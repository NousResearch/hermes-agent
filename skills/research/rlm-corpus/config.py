"""RLM-Corpus configuration.

Defaults are tuned for Anthropic Opus as root + Haiku as sub-LLM. Override
via environment variables or by constructing ``RLMConfig`` directly.

Env vars:
    RLM_ROOT_MODEL          default "claude-opus-4-7"
    RLM_SUB_MODEL           default "claude-haiku-4-5"
    RLM_SUB_ENDPOINT        default "anthropic" (also: "openai", "omlx")
    RLM_SUB_BASE_URL        default None (used for OMLX / local endpoints)
    RLM_MAX_ITERATIONS      default 20
    RLM_REPL_OUTPUT_CHARS   default 4000
    RLM_SUB_LLM_CHARS       default 500000
    RLM_KERNEL_TIMEOUT      default 120 (seconds)
    RLM_TEMPERATURE         default 0.3
    RLM_ENABLE_SUB_CALLS    default "true"
    RLM_CACHE_DIR           default "~/.hermes/rlm-cache"
    RLM_ALLOW_NETWORK       default "false"
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class RLMConfig:
    max_iterations: int = field(default_factory=lambda: _env_int("RLM_MAX_ITERATIONS", 20))
    max_repl_output_chars: int = field(
        default_factory=lambda: _env_int("RLM_REPL_OUTPUT_CHARS", 4000)
    )
    max_sub_llm_chars: int = field(
        default_factory=lambda: _env_int("RLM_SUB_LLM_CHARS", 500_000)
    )
    kernel_exec_timeout: int = field(
        default_factory=lambda: _env_int("RLM_KERNEL_TIMEOUT", 120)
    )
    root_model: str = field(
        default_factory=lambda: os.environ.get("RLM_ROOT_MODEL", "claude-opus-4-7")
    )
    sub_model: str = field(
        default_factory=lambda: os.environ.get("RLM_SUB_MODEL", "claude-haiku-4-5")
    )
    sub_llm_endpoint: str = field(
        default_factory=lambda: os.environ.get("RLM_SUB_ENDPOINT", "anthropic")
    )
    sub_llm_base_url: str | None = field(
        default_factory=lambda: os.environ.get("RLM_SUB_BASE_URL") or None
    )
    temperature: float = field(default_factory=lambda: _env_float("RLM_TEMPERATURE", 0.3))
    enable_sub_calls: bool = field(
        default_factory=lambda: _env_bool("RLM_ENABLE_SUB_CALLS", True)
    )
    cache_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("RLM_CACHE_DIR", str(Path.home() / ".hermes" / "rlm-cache"))
        )
    )
    allow_network_in_repl: bool = field(
        default_factory=lambda: _env_bool("RLM_ALLOW_NETWORK", False)
    )

    def describe(self) -> str:
        return (
            f"RLMConfig(root={self.root_model}, sub={self.sub_model} via "
            f"{self.sub_llm_endpoint}, iters<={self.max_iterations}, "
            f"sub_calls={'on' if self.enable_sub_calls else 'off'})"
        )
