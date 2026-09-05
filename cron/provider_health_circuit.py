"""Provider-health route selection shared by cron launch and bookkeeping."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.provider_health import ProviderHealthStore, ProviderRoute
from hermes_cli.fallback_config import get_fallback_chain


@dataclass(frozen=True)
class CronRouteResolution:
    job: dict[str, Any] | None
    fallback_chain: list[dict[str, Any]]
    reasoning_effort: str | None
    deferred_until: datetime | None
    reason: str | None
    skipped: tuple[tuple[str, str], ...]
    probe: bool = False


def _primary_route(job: dict[str, Any], cfg: dict[str, Any]) -> ProviderRoute:
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    provider = str(job.get("provider") or model_cfg.get("provider") or "").strip()
    model = str(job.get("model") or model_cfg.get("default") or "").strip()
    scope = str(job.get("credential_scope") or "default")
    return ProviderRoute(provider, model, credential_scope=scope)


def resolve_cron_route(
    job: dict[str, Any],
    cfg: dict[str, Any],
    *,
    hermes_home: str | Path,
    now: datetime | None = None,
) -> CronRouteResolution:
    """Choose a healthy route before agent/session construction."""
    primary = _primary_route(job, cfg)
    fallback_chain = get_fallback_chain(cfg)
    routes = [primary]
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in fallback_chain:
        route = ProviderRoute(
            str(entry["provider"]),
            str(entry["model"]),
            credential_scope=str(entry.get("credential_scope") or "default"),
            reasoning_effort=entry.get("reasoning_effort"),
        )
        routes.append(route)
        entries[(route.provider.lower(), route.model.lower())] = entry

    decision = ProviderHealthStore(hermes_home).decide(
        routes, owner=f"cron:{job.get('id') or 'unknown'}", now=now
    )
    skipped = tuple((route.provider, route.model) for route in decision.skipped)
    if decision.route is None:
        return CronRouteResolution(
            None, [], None, decision.deferred_until, decision.reason, skipped
        )

    selected = decision.route
    selected_key = (selected.provider.lower(), selected.model.lower())
    primary_key = (primary.provider.lower(), primary.model.lower())
    effective_job = dict(job)
    effort = None
    remaining = list(fallback_chain)
    if selected_key != primary_key:
        entry = entries[selected_key]
        effective_job["provider"] = entry["provider"]
        effective_job["model"] = entry["model"]
        if entry.get("base_url"):
            effective_job["base_url"] = entry["base_url"]
        effort = entry.get("reasoning_effort")
        selected_index = next(
            i for i, item in enumerate(fallback_chain)
            if (str(item["provider"]).lower(), str(item["model"]).lower()) == selected_key
        )
        remaining = fallback_chain[selected_index + 1 :]
    effective_job["_provider_health_route"] = {
        "provider": selected.provider,
        "model": selected.model,
        "credential_scope": selected.credential_scope,
        "probe_owner": f"cron:{job.get('id') or 'unknown'}" if decision.probe else None,
    }
    return CronRouteResolution(
        effective_job, remaining, effort, None, None, skipped, decision.probe
    )


def record_cron_route_outcome(
    job: dict[str, Any], *, success: bool, error: str | None, hermes_home: str | Path,
    now: datetime | None = None,
) -> None:
    route_data = job.get("_provider_health_route")
    if not isinstance(route_data, dict):
        route_data = {
            "provider": job.get("provider"),
            "model": job.get("model"),
            "credential_scope": job.get("credential_scope") or "default",
        }
    if not route_data.get("provider") or not route_data.get("model"):
        return
    route = ProviderRoute(
        str(route_data["provider"]), str(route_data["model"]),
        credential_scope=str(route_data.get("credential_scope") or "default"),
    )
    store = ProviderHealthStore(hermes_home)
    owner = str(route_data.get("probe_owner") or "") or None
    if success:
        store.record_success(route, owner=owner)
    elif error:
        # Only provider availability failures enter this circuit.
        from cron.rate_limit_backoff import plan_provider_backoff

        if plan_provider_backoff(job, error, now=now or datetime.now().astimezone()) is not None:
            store.record_failure(
                route, error, source=f"cron:{job.get('id') or 'unknown'}", owner=owner, now=now
            )
