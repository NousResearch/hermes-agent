"""Policy-as-data primitives for dry-run model routing.

This module intentionally performs no network calls and does not mutate Hermes
runtime configuration. It only loads structured policy data and exposes typed
inputs/outputs for policy dry-runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_POLICY_PATH = Path(__file__).with_name("caelus_policy.yaml")


@dataclass(frozen=True)
class RoutingContext:
    """Inputs used to recommend a model without changing runtime state."""

    task_type: str
    agent_role: str
    risk_level: str
    client_facing: bool
    sensitive_data: bool
    final_authority: bool
    complexity: str
    current_provider: str | None = None
    current_model: str | None = None


@dataclass(frozen=True)
class RoutingPolicy:
    """Structured model-routing policy loaded from YAML."""

    version: int
    profile: str
    modes: dict[str, Any]
    tiers: dict[str, dict[str, Any]]
    models: dict[str, dict[str, Any]]
    task_routes: dict[str, dict[str, Any]]
    agent_routes: dict[str, str]
    fallbacks: dict[str, str]
    forbidden: dict[str, Any]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "RoutingPolicy":
        return cls(
            version=int(data["version"]),
            profile=str(data["profile"]),
            modes=dict(data.get("modes") or {}),
            tiers=dict(data.get("tiers") or {}),
            models=dict(data.get("models") or {}),
            task_routes=dict(data.get("task_routes") or {}),
            agent_routes=dict(data.get("agent_routes") or {}),
            fallbacks=dict(data.get("fallbacks") or {}),
            forbidden=dict(data.get("forbidden") or {}),
        )


@dataclass(frozen=True)
class RoutingDecision:
    """Dry-run recommendation produced by the policy router."""

    provider: str
    model: str
    tier: str
    fallback_model: str | None
    free_model_allowed: bool
    reason: str
    estimated_cost_class: str
    approval_required: bool
    policy_warnings: list[str] = field(default_factory=list)
    escalation_reason: str | None = None
    dry_run: bool = True


def load_policy(path: str | Path | None = None) -> RoutingPolicy:
    """Load the Caelus model-routing policy from YAML.

    The loader is deliberately local-file only: no API calls, config writes, or
    provider validation happen here.
    """

    policy_path = Path(path) if path else _POLICY_PATH
    with policy_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Routing policy must be a mapping: {policy_path}")
    return RoutingPolicy.from_mapping(raw)
