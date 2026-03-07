"""Budget-aware model routing with multi-provider API key management.

Supports:
- Anthropic (Claude) with explicit API key + allowed models
- OpenAI with explicit API key + allowed models
- OpenRouter as a fallback/aggregator
- Local Qwen3.5-9B (multiple instances on Apple Silicon, or remote GPU)
- Remote GPU providers (Lambda Labs, AWS) for self-hosted models

The router picks the best model for each task role, respects per-provider
API keys and model allowlists, tracks spend, and enforces budget limits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from swarm.types import SwarmConfig
from swarm.exceptions import BudgetExceededError


# USD per 1M tokens (input, output)
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4":        (15.00, 75.00),
    "claude-sonnet-4":      (3.00,  15.00),
    "claude-haiku-4":       (0.80,  4.00),
    # OpenAI
    "gpt-4o":               (2.50,  10.00),
    "gpt-4o-mini":          (0.15,  0.60),
    "o3":                   (10.00, 40.00),
    "o3-mini":              (1.10,  4.40),
    # Google (via OpenRouter)
    "gemini-2.5-flash":     (0.15,  0.60),
    "gemini-2.5-pro":       (1.25,  10.00),
    # Local / self-hosted (zero cost when local, priced when remote GPU)
    "local/qwen3.5-9b":     (0.00,  0.00),
    "qwen3.5-9b-remote":    (0.20,  0.60),  # Lambda/AWS hosted
}


@dataclass
class ProviderConfig:
    """Configuration for a single inference provider."""
    name: str
    api_key: str = ""
    base_url: str = ""
    allowed_models: List[str] = field(default_factory=list)
    max_concurrent: int = 10
    enabled: bool = True

    def has_model(self, model: str) -> bool:
        if not self.allowed_models:
            return True  # no allowlist = all models allowed
        return model in self.allowed_models


@dataclass
class LocalModelConfig:
    """Configuration for local model instances."""
    model_id: str = "local/qwen3.5-9b"
    port_start: int = 8800
    max_instances: int = 1  # how many to spin up on this machine
    # Remote GPU fallback
    remote_provider: str = ""  # "lambda", "aws", "" for none
    remote_gpu_type: str = "A10G"
    remote_max_instances: int = 0


# role -> ordered preference list of model keys
_DEFAULT_ROUTING: dict[str, list[str]] = {
    "planner":    ["claude-opus-4", "o3", "gpt-4o"],
    "critic":     ["claude-opus-4", "gpt-4o", "gemini-2.5-pro"],
    "executor":   ["gemini-2.5-flash", "gpt-4o-mini", "claude-haiku-4"],
    "verifier":   ["claude-sonnet-4", "gemini-2.5-pro"],
    "merger":     ["claude-sonnet-4", "claude-opus-4"],
    "researcher": ["gemini-2.5-flash", "gpt-4o-mini"],
    "local":      ["local/qwen3.5-9b"],
}

# task_type overrides (checked before role fallback)
_TASK_TYPE_ROUTING: dict[str, list[str]] = {
    "code_generation": ["gemini-2.5-flash", "gpt-4o-mini", "claude-haiku-4"],
    "code_review":     ["claude-opus-4", "o3", "gpt-4o"],
    "summarization":   ["gemini-2.5-flash", "claude-haiku-4"],
    "planning":        ["claude-opus-4", "o3"],
    "verification":    ["claude-sonnet-4", "gemini-2.5-pro"],
    "vision":          ["claude-sonnet-4", "gpt-4o", "local/qwen3.5-9b"],
    "local":           ["local/qwen3.5-9b"],
}

# Model -> provider mapping
_MODEL_PROVIDERS: dict[str, str] = {
    "claude-opus-4": "anthropic",
    "claude-sonnet-4": "anthropic",
    "claude-haiku-4": "anthropic",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "o3": "openai",
    "o3-mini": "openai",
    "gemini-2.5-flash": "openrouter",
    "gemini-2.5-pro": "openrouter",
    "local/qwen3.5-9b": "local",
    "qwen3.5-9b-remote": "remote-gpu",
}


@dataclass
class _UsageRecord:
    model: str
    provider: str
    role: str
    task_type: str
    input_tokens: int
    output_tokens: int
    cost: float


class ModelRouter:
    """Routes tasks to the best model based on role, budget, and available providers."""

    def __init__(self, config: SwarmConfig, providers: Dict[str, ProviderConfig] = None,
                 local_config: LocalModelConfig = None) -> None:
        self._config = config
        self._budget_limit = config.budget_limit_usd
        self._routing = dict(_DEFAULT_ROUTING)
        self._task_routing = dict(_TASK_TYPE_ROUTING)
        self._pricing = dict(_MODEL_PRICING)
        self._usage: list[_UsageRecord] = []
        self._total_spend: float = 0.0

        # Provider configs — load from explicit config or env vars
        self._providers: Dict[str, ProviderConfig] = providers or self._load_providers()
        self._local_config = local_config or LocalModelConfig()
        self._local_instances: list[int] = []  # ports of running local instances

    def _load_providers(self) -> Dict[str, ProviderConfig]:
        """Load provider configs from environment variables."""
        providers = {}

        # Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if anthropic_key:
            providers["anthropic"] = ProviderConfig(
                name="anthropic",
                api_key=anthropic_key,
                base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").strip(),
                allowed_models=_parse_env_list("SWARM_ANTHROPIC_MODELS",
                                               ["claude-opus-4", "claude-sonnet-4", "claude-haiku-4"]),
            )

        # OpenAI
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if openai_key:
            providers["openai"] = ProviderConfig(
                name="openai",
                api_key=openai_key,
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
                allowed_models=_parse_env_list("SWARM_OPENAI_MODELS",
                                               ["gpt-4o", "gpt-4o-mini", "o3", "o3-mini"]),
            )

        # OpenRouter (aggregator — Gemini, etc.)
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if openrouter_key:
            providers["openrouter"] = ProviderConfig(
                name="openrouter",
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                allowed_models=_parse_env_list("SWARM_OPENROUTER_MODELS", []),
            )

        # Local (always available)
        providers["local"] = ProviderConfig(
            name="local",
            api_key="local",
            base_url="http://127.0.0.1:8800/v1",
            allowed_models=["local/qwen3.5-9b"],
            max_concurrent=int(os.getenv("SWARM_LOCAL_MAX_CONCURRENT", "2")),
            enabled=True,
        )

        return providers

    # ------------------------------------------------------------------
    # Provider management
    # ------------------------------------------------------------------

    def add_provider(self, provider: ProviderConfig) -> None:
        self._providers[provider.name] = provider

    def get_provider(self, model: str) -> Optional[ProviderConfig]:
        """Get the provider config for a model, if available and enabled."""
        provider_name = _MODEL_PROVIDERS.get(model, "openrouter")
        provider = self._providers.get(provider_name)
        if provider and provider.enabled and provider.has_model(model):
            return provider
        # Fallback: check all providers
        for p in self._providers.values():
            if p.enabled and p.has_model(model):
                return p
        return None

    def available_models(self) -> List[str]:
        """List all models that have an enabled provider with the right API key."""
        models = []
        for model, provider_name in _MODEL_PROVIDERS.items():
            p = self._providers.get(provider_name)
            if p and p.enabled and p.api_key and p.has_model(model):
                models.append(model)
        return models

    # ------------------------------------------------------------------
    # Local model scaling
    # ------------------------------------------------------------------

    def scale_local(self, n: int = 1) -> List[int]:
        """Spin up n additional local Qwen instances on consecutive ports.

        Returns list of ports started. Caps at local_config.max_instances.
        """
        import subprocess
        import sys

        started = []
        max_inst = self._local_config.max_instances
        port = self._local_config.port_start + len(self._local_instances)

        for _ in range(n):
            if len(self._local_instances) >= max_inst:
                break
            try:
                subprocess.Popen(
                    [sys.executable, "-m", "local_models.serve", "qwen", "--port", str(port)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                self._local_instances.append(port)
                started.append(port)
                port += 1
            except Exception:
                break
        return started

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def select_model(
        self,
        role: str,
        task_type: str | None = None,
        budget_remaining: float | None = None,
        require_vision: bool = False,
    ) -> str:
        budget = budget_remaining if budget_remaining is not None else self._remaining_budget()
        available = set(self.available_models())

        # Build candidate list: task_type first, then role fallback
        candidates: list[str] = []

        if require_vision:
            for m in _TASK_TYPE_ROUTING.get("vision", []):
                if m not in candidates and m in available:
                    candidates.append(m)

        if task_type and task_type in self._task_routing:
            for m in self._task_routing[task_type]:
                if m not in candidates and m in available:
                    candidates.append(m)

        if role in self._routing:
            for m in self._routing[role]:
                if m not in candidates and m in available:
                    candidates.append(m)

        if not candidates:
            # Last resort: anything available
            candidates = [m for m in available if m != "local/qwen3.5-9b"] or list(available)

        # Budget filter
        if budget is not None and budget > 0:
            affordable = [
                m for m in candidates
                if self.estimate_cost(m, 2000, 1000) <= budget
            ]
            if affordable:
                candidates = affordable

        if not candidates:
            raise BudgetExceededError(
                spent=self._total_spend,
                limit=self._budget_limit,
            )

        return candidates[0]

    def get_client_kwargs(self, model: str) -> Dict[str, Any]:
        """Get the OpenAI client kwargs (api_key, base_url) for a model."""
        provider = self.get_provider(model)
        if not provider:
            return {}
        return {
            "api_key": provider.api_key,
            "base_url": provider.base_url,
        }

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price_in, price_out = self._pricing.get(model, (5.0, 15.0))
        return (input_tokens * price_in + output_tokens * price_out) / 1_000_000

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def log_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        role: str = "",
        task_type: str = "",
    ) -> None:
        provider_name = _MODEL_PROVIDERS.get(model, "unknown")
        self._usage.append(_UsageRecord(
            model=model,
            provider=provider_name,
            role=role,
            task_type=task_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        ))
        self._total_spend += cost

    def get_spend_summary(self) -> dict[str, Any]:
        by_model: dict[str, float] = {}
        by_provider: dict[str, float] = {}
        by_role: dict[str, float] = {}
        total_input = 0
        total_output = 0

        for r in self._usage:
            by_model[r.model] = by_model.get(r.model, 0.0) + r.cost
            by_provider[r.provider] = by_provider.get(r.provider, 0.0) + r.cost
            if r.role:
                by_role[r.role] = by_role.get(r.role, 0.0) + r.cost
            total_input += r.input_tokens
            total_output += r.output_tokens

        return {
            "total_cost_usd": self._total_spend,
            "budget_limit_usd": self._budget_limit,
            "budget_remaining_usd": self._remaining_budget(),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "num_requests": len(self._usage),
            "by_model": by_model,
            "by_provider": by_provider,
            "by_role": by_role,
            "available_models": self.available_models(),
        }

    def is_within_budget(self, estimated_cost: float) -> bool:
        if self._budget_limit <= 0:
            return True
        return (self._total_spend + estimated_cost) <= self._budget_limit

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _remaining_budget(self) -> float | None:
        if self._budget_limit <= 0:
            return None
        return max(0.0, self._budget_limit - self._total_spend)


def _parse_env_list(var: str, default: List[str]) -> List[str]:
    """Parse comma-separated env var into list, or return default."""
    val = os.getenv(var, "").strip()
    if not val:
        return default
    return [m.strip() for m in val.split(",") if m.strip()]
