"""Research configuration helpers."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml

ResearchMode = Literal["approval", "auto"]


DEFAULT_RESEARCH_CONFIG: dict[str, Any] = {
    "enabled": True,
    "mode": "approval",
    "always_load_skills": ["autonomous-llm-research"],
    "workspace_dir": ".hermes-research",
    "default_chunk_minutes": 20,
    "max_turns_per_chunk": 30,
    "loop_poll_interval_minutes": 30,
    "max_concurrent_runs": 3,
    "max_total_cost_usd": 50,
    "require_approval_for": ["start_run", "resume_run", "stop_run"],
    "zero_spec_strategy": "novel_idea_first",
    "tinker": {
        "api_key_env": "TINKER_API_KEY",
        "default_base_model": "meta-llama/Llama-3.1-8B",
        "default_method": "sft",
        "default_lora_rank": 32,
        "default_learning_rate": 1e-4,
    },
    "evaluation": {
        "default_benchmarks": ["hellaswag", "mmlu"],
        "default_num_fewshot": 0,
        "default_batch_size": 32,
        "model_backend": "hf",
    },
    "cost_estimation": {
        "gpu_hour_cost_usd": 2.50,
        "overhead_multiplier": 1.1,
    },
    "literature": {
        "arxiv_rate_limit_seconds": 3,
        "s2_rate_limit_seconds": 1,
    },
    "dataset": {
        "auxiliary_model_task": "dataset_synthesis",
        "max_synthesis_rows": 5000,
    },
    "judge": {
        "auxiliary_model_task": "judge",
        "default_temperature": 0.3,
        "max_completions_per_prompt": 5,
    },
    "manager": {
        "search_max_queue": 10,
        "literature_top_k": 10,
        "memo_top_k_runs": 5,
    },
}


def get_hermes_home() -> Path:
    """Resolve the Hermes home directory."""
    return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))


def get_config_path() -> Path:
    """Resolve the Hermes config file path."""
    return get_hermes_home() / "config.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(copy.deepcopy(base[key]), value)
        else:
            base[key] = value
    return base


def get_research_config() -> dict[str, Any]:
    """Load the research config from ~/.hermes/config.yaml."""
    merged = copy.deepcopy(DEFAULT_RESEARCH_CONFIG)
    cfg_path = get_config_path()

    if cfg_path.exists():
        try:
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            loaded = {}
        research_cfg = loaded.get("research") or {}
        if isinstance(research_cfg, dict):
            merged = _deep_merge(merged, research_cfg)

    env_mode = os.getenv("HERMES_RESEARCH_MODE", "").strip().lower()
    if env_mode in {"approval", "auto"}:
        merged["mode"] = env_mode

    env_enabled = os.getenv("HERMES_RESEARCH_ENABLED", "").strip().lower()
    if env_enabled in {"0", "false", "no"}:
        merged["enabled"] = False
    elif env_enabled in {"1", "true", "yes"}:
        merged["enabled"] = True

    return merged


def research_enabled() -> bool:
    """Return whether the research mode is enabled."""
    return bool(get_research_config().get("enabled", True))


def get_research_mode(mode_override: str | None = None) -> ResearchMode:
    """Return the effective research mode."""
    candidate = (mode_override or "").strip().lower()
    if candidate in {"approval", "auto"}:
        return candidate  # type: ignore[return-value]
    configured = str(get_research_config().get("mode", "approval")).strip().lower()
    if configured in {"approval", "auto"}:
        return configured  # type: ignore[return-value]
    return "approval"


def get_workspace_dir(cwd: str | Path | None = None) -> Path:
    """Resolve the research workspace directory for a working tree."""
    cfg = get_research_config()
    workspace_dir = str(cfg.get("workspace_dir", ".hermes-research")).strip() or ".hermes-research"
    base = Path(cwd or os.getcwd()).resolve()
    workspace = Path(workspace_dir)
    if workspace.is_absolute():
        return workspace
    return base / workspace
