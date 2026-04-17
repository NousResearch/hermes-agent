"""Internal multi-provider policy normalization and decision substrate.

PR1 scope:
- normalize current Hermes config into a deterministic internal policy object
- expose a structured routing decision object
- remain pure and testable; no provider calls, no credential reads
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DomainPolicy:
    name: str
    strategy: str
    provider: str | None = None
    model: str | None = None
    enabled: bool = True
    allow_cross_provider: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderPolicy:
    mode: str
    primary_provider: str | None
    primary_model: str | None
    fallback_providers: list[dict[str, Any]] = field(default_factory=list)
    domains: dict[str, DomainPolicy] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    source_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    domain: str
    selected_provider: str | None
    selected_model: str | None
    decision_source: str
    reason: str
    cross_provider: bool = False
    blocked_alternatives: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_delegation_config(cfg: dict) -> None:
    """Raise ValueError when delegation is explicit but api_key is empty.

    Explicit delegation = provider or model are set. In that case api_key
    must be a non-empty string. Empty/missing api_key = subagents silently
    inherit parent premium credentials — the whole reason this exists.
    """
    cfg = cfg or {}
    provider = str(cfg.get("provider") or "").strip()
    model = str(cfg.get("model") or "").strip()
    if not (provider or model):
        return  # inherit_primary mode — nothing to validate here
    api_key = cfg.get("api_key")
    if api_key is None or not str(api_key).strip():
        raise ValueError(
            "delegation.api_key must be explicit (non-empty) when "
            "delegation.provider or delegation.model is set; empty api_key "
            "causes subagents to inherit parent credentials."
        )


def normalize_provider_policy(config: dict) -> ProviderPolicy:
    """Normalize Hermes config into a deterministic internal policy object.

    This initial PR1 implementation is intentionally policy-only and pure.
    """
    cfg = deepcopy(config or {})

    model_cfg = cfg.get("model") or {}
    primary_provider = str(model_cfg.get("provider") or "").strip() or None
    primary_model = str(model_cfg.get("default") or "").strip() or None

    fallback_source = None
    fallback_candidates: list[dict[str, Any]] = []
    if cfg.get("fallback_providers"):
        fallback_source = "fallback_providers"
        for item in cfg.get("fallback_providers") or []:
            if isinstance(item, dict):
                provider = str(item.get("provider") or "").strip()
                model = str(item.get("model") or "").strip()
                if provider and model:
                    fallback_candidates.append({"provider": provider, "model": model})
    elif cfg.get("fallback_model"):
        item = cfg.get("fallback_model") or {}
        provider = str(item.get("provider") or "").strip()
        model = str(item.get("model") or "").strip()
        if provider and model:
            fallback_source = "fallback_model"
            fallback_candidates.append({"provider": provider, "model": model})

    smart_cfg = cfg.get("smart_model_routing")
    smart_enabled = False
    smart_present = "smart_model_routing" in cfg
    if isinstance(smart_cfg, dict):
        smart_enabled = bool(smart_cfg.get("enabled", False))

    compression_cfg = cfg.get("compression") or {}
    compression_provider = str(compression_cfg.get("summary_provider") or "auto").strip() or "auto"
    compression_model = str(compression_cfg.get("summary_model") or "").strip() or None

    auxiliary_cfg = cfg.get("auxiliary") or {}
    auxiliary_tasks: dict[str, dict[str, Any]] = {
        "compression": {
            "strategy": compression_provider,
            "provider": None if compression_provider in {"auto", "main"} else compression_provider,
            "model": compression_model,
        }
    }
    for task_name, task_cfg in auxiliary_cfg.items():
        if not isinstance(task_cfg, dict):
            continue
        provider = str(task_cfg.get("provider") or "").strip() or None
        model = str(task_cfg.get("model") or "").strip() or None
        strategy = provider or "auto"
        auxiliary_tasks[task_name] = {
            "strategy": strategy,
            "provider": provider,
            "model": model,
        }

    delegation_cfg = cfg.get("delegation") or {}
    delegation_provider = str(delegation_cfg.get("provider") or "").strip() or None
    delegation_model = str(delegation_cfg.get("model") or "").strip() or None
    provider_routes = deepcopy(delegation_cfg.get("provider_routes") or {})
    role_routes = deepcopy(delegation_cfg.get("role_routes") or {})

    domains = {
        "primary": DomainPolicy(
            name="primary",
            strategy="primary",
            provider=primary_provider,
            model=primary_model,
            enabled=True,
            allow_cross_provider=False,
            metadata={"smart_model_routing_enabled": smart_enabled},
        ),
        "fallback": DomainPolicy(
            name="fallback",
            strategy="fallback_chain" if fallback_candidates else "disabled",
            enabled=bool(fallback_candidates),
            allow_cross_provider=True,
            metadata={"candidates": deepcopy(fallback_candidates)},
        ),
        "auxiliary": DomainPolicy(
            name="auxiliary",
            strategy="task_map",
            enabled=True,
            allow_cross_provider=True,
            metadata={"tasks": auxiliary_tasks},
        ),
        "delegation": DomainPolicy(
            name="delegation",
            strategy="explicit" if (delegation_provider or delegation_model) else "inherit_primary",
            provider=delegation_provider,
            model=delegation_model,
            enabled=True,
            allow_cross_provider=True,
            metadata={
                "provider_routes": provider_routes,
                "role_routes": role_routes,
            },
        ),
    }

    return ProviderPolicy(
        mode="legacy-compatible",
        primary_provider=primary_provider,
        primary_model=primary_model,
        fallback_providers=fallback_candidates,
        domains=domains,
        constraints={},
        source_summary={
            "fallback_source": fallback_source,
            "smart_model_routing_present": smart_present,
        },
    )


def decide_provider_route(
    domain: str,
    policy: ProviderPolicy,
    *,
    requested_provider: str | None = None,
    requested_model: str | None = None,
    context: dict[str, Any] | None = None,
) -> RoutingDecision:
    """Return a deterministic routing decision for a policy domain.

    Implements primary, auxiliary (with task context), delegation (explicit/inherit),
    fallback (chain/disabled), and unknown_domain branches.
    """
    selected_provider = None
    selected_model = None
    decision_source = ""
    reason = ""
    cross_provider = False
    metadata = {"context": context or {}}

    if domain == "primary":
        selected_provider = policy.primary_provider
        selected_model = policy.primary_model
        decision_source = "primary"
        reason = "primary domain selected"
        cross_provider = False
        dom = policy.domains.get("primary")
        if dom:
            metadata = {**dom.metadata, "context": context or {}}

    elif domain == "auxiliary":
        aux_dom = policy.domains.get("auxiliary")
        tasks = aux_dom.metadata.get("tasks", {}) if aux_dom else {}
        task_name = None
        task_info = None
        if context and "task" in context:
            task_name = context["task"]
            task_info = tasks.get(task_name)

        if task_info and task_info.get("provider") not in (None, "auto", "main"):
            selected_provider = task_info["provider"]
            selected_model = task_info.get("model")
            decision_source = "auxiliary_task"
            reason = f"auxiliary task {task_name} routed to {selected_provider}"
            cross_provider = selected_provider != policy.primary_provider
        else:
            selected_provider = policy.primary_provider
            selected_model = policy.primary_model
            decision_source = "auxiliary_fallback_primary"
            reason = "auxiliary falls back to primary (no task context)"
            cross_provider = False

    elif domain == "delegation":
        dom = policy.domains.get("delegation")
        if dom and dom.strategy == "explicit":
            selected_provider = dom.provider or policy.primary_provider
            selected_model = dom.model or policy.primary_model
            decision_source = "delegation_explicit"
            reason = "delegation uses explicit provider/model"
            cross_provider = dom.provider is not None and dom.provider != policy.primary_provider
        else:
            selected_provider = policy.primary_provider
            selected_model = policy.primary_model
            decision_source = "delegation_inherit"
            reason = "delegation inherits primary"
            cross_provider = False

    elif domain == "fallback":
        candidates = policy.fallback_providers
        if candidates:
            first = candidates[0]
            selected_provider = first["provider"]
            selected_model = first["model"]
            decision_source = "fallback_chain"
            reason = f"fallback chain head: {first['provider']}"
            cross_provider = first["provider"] != policy.primary_provider
        else:
            selected_provider = policy.primary_provider
            selected_model = policy.primary_model
            decision_source = "fallback_disabled"
            reason = "fallback chain disabled"
            cross_provider = False

    else:
        selected_provider = policy.primary_provider
        selected_model = policy.primary_model
        decision_source = "unknown_domain"
        reason = f"unknown domain: {domain}"
        cross_provider = False

    # Apply requested_provider/requested_model overrides
    if requested_provider is not None:
        selected_provider = requested_provider
        cross_provider = selected_provider != policy.primary_provider
    if requested_model is not None:
        selected_model = requested_model

    return RoutingDecision(
        domain=domain,
        selected_provider=selected_provider,
        selected_model=selected_model,
        decision_source=decision_source,
        reason=reason,
        cross_provider=cross_provider,
        blocked_alternatives=[],
        metadata=metadata,
    )
