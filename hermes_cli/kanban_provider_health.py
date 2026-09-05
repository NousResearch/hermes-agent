"""Provider-health preflight for Kanban worker routes."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.provider_health import ProviderHealthStore, ProviderRoute
from hermes_cli.fallback_config import get_fallback_chain


@dataclass(frozen=True)
class TaskRouteResolution:
    route: ProviderRoute | None
    deferred_until: datetime | None
    reason: str | None
    skipped: tuple[ProviderRoute, ...]
    profile_home: Path
    probe: bool = False


def _profile_inputs(task, profile_home: Path | None, config: dict[str, Any] | None):
    if profile_home is None:
        from hermes_cli.profiles import get_profile_dir

        profile_home = get_profile_dir(str(task.assignee))
    profile_home = Path(profile_home)
    if config is None:
        from hermes_cli.config import read_user_config_raw

        config = read_user_config_raw(profile_home / "config.yaml")
    return profile_home, config or {}


def resolve_task_route(
    task,
    *,
    profile_home: Path | None = None,
    config: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> TaskRouteResolution:
    """Return the first healthy allowed route without starting a worker."""
    profile_home, config = _profile_inputs(task, profile_home, config)
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
    explicit_model = str(getattr(task, "model_override", None) or "").strip()
    explicit_provider = str(getattr(task, "provider_override", None) or "").strip()
    primary = ProviderRoute(
        explicit_provider or str(model_cfg.get("provider") or ""),
        explicit_model or str(model_cfg.get("default") or ""),
        reasoning_effort=(
            getattr(task, "reasoning_effort", None)
            or agent_cfg.get("reasoning_effort")
        ),
    )
    routes = [primary]
    if not explicit_model and not explicit_provider:
        for entry in get_fallback_chain(config):
            routes.append(
                ProviderRoute(
                    str(entry["provider"]),
                    str(entry["model"]),
                    credential_scope=str(entry.get("credential_scope") or "default"),
                    reasoning_effort=entry.get("reasoning_effort") or agent_cfg.get("reasoning_effort"),
                )
            )
    decision = ProviderHealthStore(profile_home).decide(
        routes, owner=f"kanban:{task.id}", now=now
    )
    return TaskRouteResolution(
        decision.route,
        decision.deferred_until,
        decision.reason,
        decision.skipped,
        profile_home,
        decision.probe,
    )


def apply_task_route(task, resolution: TaskRouteResolution) -> None:
    route = resolution.route
    if route is None:
        return
    task.model_override = route.model
    task.provider_override = route.provider
    if route.reasoning_effort:
        task.reasoning_effort = route.reasoning_effort
