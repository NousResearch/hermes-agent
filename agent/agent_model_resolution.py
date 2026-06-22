"""Runtime model resolution for canonical Hermes agent identities.

This module reads the live declarative matrix introduced under
``agents.models`` in ``config.yaml`` and turns it into explicit, auditable
model/provider/fallback choices. It intentionally does not consult Command
Center snapshots or seed data; runtime enforcement must use live config only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass
class AgentModelResolution:
    """Result of resolving an agent identity to an effective model config."""

    agent_id: Optional[str] = None
    assigned_model: Optional[str] = None
    assigned_provider: Optional[str] = None
    assigned_fallbacks: Any = None
    effective_model: Optional[str] = None
    effective_provider: Optional[str] = None
    effective_fallbacks: Any = None
    model_source: str = "unresolved"
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "assigned_model": self.assigned_model,
            "assigned_provider": self.assigned_provider,
            "assigned_fallbacks": self.assigned_fallbacks,
            "effective_model": self.effective_model,
            "effective_provider": self.effective_provider,
            "effective_fallbacks": self.effective_fallbacks,
            "model_source": self.model_source,
            "model_resolution_warnings": list(self.warnings),
        }


def normalize_agent_id(agent_id: Any) -> Optional[str]:
    """Normalize an agent id for matrix lookup (case-insensitive)."""
    if agent_id is None:
        return None
    text = str(agent_id).strip().lower()
    return text or None


def _nonempty(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_live_config() -> dict[str, Any]:
    """Load live Hermes config from the active profile/home."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def agent_model_matrix(config: Optional[Mapping[str, Any]] = None) -> dict[str, Mapping[str, Any]]:
    """Return normalized ``agents.models`` entries from live config."""
    cfg = dict(config or load_live_config())
    agents_cfg = cfg.get("agents") or {}
    if not isinstance(agents_cfg, Mapping):
        return {}
    models = agents_cfg.get("models") or {}
    if not isinstance(models, Mapping):
        return {}
    normalized: dict[str, Mapping[str, Any]] = {}
    for key, value in models.items():
        agent_id = normalize_agent_id(key)
        if agent_id and isinstance(value, Mapping):
            normalized[agent_id] = value
    return normalized


def infer_agent_id_from_job(job: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """Infer canonical agent identity from cron job metadata.

    Precedence: explicit ``agent_id`` / ``agent`` / ``agentId`` first, then a
    single matching skill from ``skill`` or ``skills``. Matching is dynamic
    against the live matrix keys so this helper does not hardcode the matrix.
    """
    for key in ("agent_id", "agent", "agentId"):
        agent_id = normalize_agent_id(job.get(key))
        if agent_id:
            return agent_id

    matrix_keys = set(agent_model_matrix(config).keys())
    skill_values: list[Any] = []
    if job.get("skill") is not None:
        skill_values.append(job.get("skill"))
    skills = job.get("skills")
    if isinstance(skills, (list, tuple, set)):
        skill_values.extend(skills)
    elif skills is not None:
        skill_values.append(skills)

    for skill in skill_values:
        skill_id = normalize_agent_id(skill)
        if skill_id in matrix_keys:
            return skill_id
    return None


def resolve_agent_model(
    agent_id: Any,
    *,
    config: Optional[Mapping[str, Any]] = None,
    fallback_model: Any = None,
    fallback_provider: Any = None,
    fallback_fallbacks: Any = None,
    fallback_source: str = "fallback",
) -> AgentModelResolution:
    """Resolve an agent id through ``agents.models`` with honest fallback metadata."""
    normalized = normalize_agent_id(agent_id)
    result = AgentModelResolution(
        agent_id=normalized,
        effective_model=_nonempty(fallback_model),
        effective_provider=_nonempty(fallback_provider),
        effective_fallbacks=fallback_fallbacks,
        model_source=fallback_source,
    )
    if not normalized:
        result.warnings.append("no agent_id provided; using existing runtime model resolution")
        return result

    matrix = agent_model_matrix(config)
    entry = matrix.get(normalized)
    if not entry:
        result.model_source = fallback_source
        result.warnings.append(
            f"agents.models.{normalized} not found; using {fallback_source}"
        )
        return result

    assigned_model = _nonempty(entry.get("model") or entry.get("primary_model"))
    assigned_provider = _nonempty(entry.get("provider"))
    assigned_fallbacks = (
        entry.get("fallbacks")
        if "fallbacks" in entry
        else entry.get("fallback_model")
        if "fallback_model" in entry
        else entry.get("fallback_providers")
    )

    result.assigned_model = assigned_model
    result.assigned_provider = assigned_provider
    result.assigned_fallbacks = assigned_fallbacks
    result.effective_model = assigned_model or result.effective_model
    result.effective_provider = assigned_provider or result.effective_provider
    result.effective_fallbacks = assigned_fallbacks if assigned_fallbacks is not None else fallback_fallbacks

    if assigned_model and assigned_provider:
        result.model_source = f"agents.models.{normalized}"
    else:
        missing = []
        if not assigned_model:
            missing.append("model")
        if not assigned_provider:
            missing.append("provider")
        result.model_source = f"agents.models.{normalized}.partial+{fallback_source}"
        result.warnings.append(
            f"agents.models.{normalized} missing {', '.join(missing)}; using {fallback_source} for missing values"
        )
    if assigned_fallbacks is None:
        result.warnings.append(
            f"agents.models.{normalized} has no fallbacks; using existing fallback chain"
        )
    return result


def resolve_job_model(
    job: Mapping[str, Any],
    *,
    config: Optional[Mapping[str, Any]] = None,
    default_model: Any = None,
    default_provider: Any = None,
    default_fallbacks: Any = None,
) -> AgentModelResolution:
    """Resolve cron job model/provider with explicit override precedence."""
    cfg = dict(config or load_live_config())
    model_cfg = cfg.get("model") or {}
    if default_model is None:
        default_model = model_cfg.get("default") if isinstance(model_cfg, Mapping) else model_cfg
    if default_provider is None:
        default_provider = model_cfg.get("provider") if isinstance(model_cfg, Mapping) else None
    if default_fallbacks is None:
        default_fallbacks = cfg.get("fallback_providers") or cfg.get("fallback_model")

    explicit_model = _nonempty(job.get("model"))
    explicit_provider = _nonempty(job.get("provider"))
    if explicit_model:
        return AgentModelResolution(
            agent_id=infer_agent_id_from_job(job, cfg),
            effective_model=explicit_model,
            effective_provider=explicit_provider or _nonempty(default_provider),
            effective_fallbacks=default_fallbacks,
            model_source="job.model",
            warnings=[] if explicit_provider else ["job.model explicit without job.provider; provider resolved by runtime/config"],
        )

    agent_id = infer_agent_id_from_job(job, cfg)
    if agent_id:
        return resolve_agent_model(
            agent_id,
            config=cfg,
            fallback_model=default_model,
            fallback_provider=explicit_provider or default_provider,
            fallback_fallbacks=default_fallbacks,
            fallback_source="model.default",
        )

    return AgentModelResolution(
        agent_id=None,
        effective_model=_nonempty(default_model),
        effective_provider=explicit_provider or _nonempty(default_provider),
        effective_fallbacks=default_fallbacks,
        model_source="job.provider+model.default" if explicit_provider else "model.default",
        warnings=["no agent assignment or job.model; using model.default"],
    )
