"""Fail-closed subscription-pool selection for delegated workers.

The selector is deliberately independent of profile selection.  A profile owns
identity, workspace context, and records; a pool route supplies execution
surface, provider, model, billing attribution, and quota state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


class PoolConfigurationError(ValueError):
    """A pool route is invalid or ambiguous and must not be executed."""


@dataclass(frozen=True)
class PoolVerification:
    installed: bool
    authenticated: bool
    models_available: bool
    headless: bool
    workspace: bool
    tools: bool
    quota_visible: bool
    billing_verified: bool
    cloud_verified: bool = True
    # Ollama's local and Cloud model names can overlap. A Cloud endpoint alone
    # is not enough to establish that the configured model is hosted.
    cloud_model_verified: bool = True

    @property
    def complete(self) -> bool:
        checks = asdict(self)
        # Ollama Cloud does not expose remaining subscription quota. Quota
        # visibility is reportable state, not route-verification completeness.
        checks.pop("quota_visible")
        return all(checks.values())


@dataclass(frozen=True)
class PoolRouteConfig:
    route_id: str
    provider: str
    model: str
    execution_surface: str
    subscription_pool: str
    billing_mode: str
    capabilities: tuple[str, ...]
    priority: int
    verification: PoolVerification
    enabled: bool = True
    paid_usage_possible: bool = False
    quota_state: str = "available"
    available_usage: Mapping[str, Any] = field(default_factory=dict)
    soft_reserve_percent: float | None = None
    remaining_percent: float | None = None
    recent_success: float = 1.0
    verification_burden: int = 0
    endpoint_host: str = ""
    # This is the authenticated runtime endpoint observed during route
    # selection, not policy metadata.  Ollama Cloud routes are only executable
    # when this concrete value is an approved HTTPS endpoint.
    runtime_base_url: str = ""
    model_family: str = ""
    minimum_generation: int | None = None
    blockers: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PoolSelection:
    selected: PoolRouteConfig
    rejected: tuple[dict[str, str], ...]
    fallback_order: tuple[str, ...]
    selection_reason: str


_ALLOWED_SURFACES = {"hermes", "cli"}
_ALLOWED_BILLING = {"subscription", "metered", "unverified"}
_ALLOWED_QUOTA = {"available", "soft_reserve", "exhausted", "unknown", "auth_failed"}
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_OLLAMA_CLOUD_PROVIDERS = {"ollama", "ollama-cloud"}
_APPROVED_OLLAMA_CLOUD_HOSTS = {"ollama.com"}


def _reason(route: PoolRouteConfig, code: str, detail: str) -> dict[str, str]:
    return {"route_id": route.route_id, "code": code, "reason": detail}


def _is_local(route: PoolRouteConfig) -> bool:
    provider = route.provider.strip().lower()
    host = route.endpoint_host.strip().lower()
    return (
        provider == "local"
        or provider.startswith("local:")
        or host in _LOOPBACK_HOSTS
        or host.endswith(".local")
    )


def _is_ollama_cloud_candidate(route: PoolRouteConfig) -> bool:
    return route.provider.strip().lower() in _OLLAMA_CLOUD_PROVIDERS


def is_approved_ollama_cloud_endpoint(endpoint: str) -> bool:
    """Accept only the configured HTTPS Ollama Cloud service, never a daemon."""
    value = str(endpoint or "").strip()
    if "://" not in value:
        return value.lower() in _APPROVED_OLLAMA_CLOUD_HOSTS
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.lower() == "https"
        and (parsed.hostname or "").lower() in _APPROVED_OLLAMA_CLOUD_HOSTS
        and port in {None, 443}
    )


def _has_approved_ollama_cloud_runtime(route: PoolRouteConfig) -> bool:
    return bool(route.runtime_base_url) and is_approved_ollama_cloud_endpoint(route.runtime_base_url)


def _configured_subscription_attribution(route: PoolRouteConfig) -> bool:
    """Ollama's configured API-key pool binding is its billing authority."""
    return (
        _is_ollama_cloud_candidate(route)
        and route.billing_mode == "subscription"
        and bool(route.subscription_pool.strip())
    )


def validate_pool_route(route: PoolRouteConfig) -> None:
    """Reject invalid or ambiguous attribution before any execution begins."""
    if not route.route_id.strip():
        raise PoolConfigurationError("Pool route id is required.")
    if not route.provider.strip() or route.provider.strip().lower() in {"auto", "custom"}:
        raise PoolConfigurationError(f"Route {route.route_id!r} has an ambiguous provider.")
    if not route.model.strip():
        raise PoolConfigurationError(f"Route {route.route_id!r} has no resolved model.")
    if route.execution_surface not in _ALLOWED_SURFACES:
        raise PoolConfigurationError(
            f"Route {route.route_id!r} has unsupported execution surface {route.execution_surface!r}."
        )
    if not route.subscription_pool.strip():
        raise PoolConfigurationError(f"Route {route.route_id!r} has no authenticated pool attribution.")
    if route.billing_mode not in _ALLOWED_BILLING:
        raise PoolConfigurationError(
            f"Route {route.route_id!r} has ambiguous billing mode {route.billing_mode!r}."
        )
    if route.enabled and route.billing_mode == "unverified":
        raise PoolConfigurationError(
            f"Route {route.route_id!r} cannot be enabled with unverified billing."
        )
    if route.quota_state not in _ALLOWED_QUOTA:
        raise PoolConfigurationError(
            f"Route {route.route_id!r} has invalid quota state {route.quota_state!r}."
        )
    if not route.capabilities:
        raise PoolConfigurationError(f"Route {route.route_id!r} declares no task capability.")
    if not 0.0 <= route.recent_success <= 1.0:
        raise PoolConfigurationError(f"Route {route.route_id!r} recent_success must be between 0 and 1.")
    if route.soft_reserve_percent is not None and not 0.0 <= route.soft_reserve_percent <= 100.0:
        raise PoolConfigurationError(
            f"Route {route.route_id!r} soft_reserve_percent must be between 0 and 100."
        )
    if route.remaining_percent is not None and not 0.0 <= route.remaining_percent <= 100.0:
        raise PoolConfigurationError(
            f"Route {route.route_id!r} remaining_percent must be between 0 and 100."
        )


def _reserve_active(route: PoolRouteConfig) -> bool:
    if route.quota_state == "soft_reserve":
        return True
    return (
        route.soft_reserve_percent is not None
        and route.remaining_percent is not None
        and route.remaining_percent <= route.soft_reserve_percent
    )


def _rank(route: PoolRouteConfig) -> tuple[int, int, int, float, int, str]:
    # A soft reserve influences routing but does not disable the route.  It is
    # considered after otherwise eligible non-reserved subscription capacity.
    return (
        int(_reserve_active(route)),
        # An eligible Cloud Ollama route is the preferred included-capacity
        # surface. Its configured pool association remains authoritative.
        int(not _is_ollama_cloud_candidate(route)),
        route.priority,
        -route.recent_success,
        route.verification_burden,
        route.route_id,
    )


def select_subscription_route(
    routes: Iterable[PoolRouteConfig],
    *,
    capability: str,
    allow_metered: bool = False,
    allow_paid_usage: bool = False,
    explicit_quality_requirement: bool = False,
) -> PoolSelection:
    """Select the cheapest verified capable route and explain every skip.

    Paid and metered capacity is fail-closed by default. Only an explicitly
    operator-authorized call path may set the corresponding allow flag.
    """
    candidates = tuple(routes)
    if not candidates:
        raise PoolConfigurationError("No subscription-pool routes are configured.")
    ids = [item.route_id for item in candidates]
    if len(ids) != len(set(ids)):
        raise PoolConfigurationError("Subscription-pool route ids must be unique.")

    eligible: list[PoolRouteConfig] = []
    rejected: list[dict[str, str]] = []
    required = capability.strip().lower() or "general"
    for route in candidates:
        validate_pool_route(route)
        capabilities = {item.strip().lower() for item in route.capabilities}
        if _is_local(route):
            rejected.append(_reason(route, "local_model_forbidden", "production routing never selects local models"))
        elif _is_ollama_cloud_candidate(route) and not _has_approved_ollama_cloud_runtime(route):
            rejected.append(_reason(
                route,
                "unapproved_ollama_runtime_endpoint",
                "authenticated Ollama runtime is not an approved Cloud HTTPS endpoint",
            ))
        elif _is_ollama_cloud_candidate(route) and not route.verification.cloud_verified:
            rejected.append(_reason(route, "unverified_cloud_route", "Ollama route is not verified as cloud-hosted"))
        elif _is_ollama_cloud_candidate(route) and not route.verification.cloud_model_verified:
            rejected.append(_reason(route, "non_cloud_model_forbidden", "Ollama model is not verified as cloud-hosted"))
        elif not route.enabled:
            detail = "; ".join(route.blockers) or "route is disabled pending verification or integration"
            rejected.append(_reason(route, "route_disabled", detail))
        elif not route.verification.installed:
            rejected.append(_reason(route, "installation_missing", "execution surface is not installed"))
        elif not route.verification.authenticated or route.quota_state == "auth_failed":
            rejected.append(_reason(route, "authentication_failed", "authenticated pool is unavailable"))
        elif not route.verification.models_available:
            rejected.append(_reason(route, "model_unavailable", "runtime model roster has no capable model"))
        elif not route.verification.headless:
            rejected.append(_reason(route, "headless_unsupported", "headless execution is unavailable"))
        elif not route.verification.workspace:
            rejected.append(_reason(route, "workspace_unsupported", "workspace execution is unavailable"))
        elif not route.verification.tools:
            rejected.append(_reason(route, "tools_unsupported", "required tool support is unavailable"))
        elif not route.verification.billing_verified and not _configured_subscription_attribution(route):
            rejected.append(_reason(route, "billing_unverified", "subscription versus metered billing is unverified"))
        elif route.quota_state == "exhausted":
            rejected.append(_reason(route, "quota_exhausted", "authenticated pool is exhausted"))
        elif required not in capabilities and "general" not in capabilities:
            rejected.append(_reason(route, "missing_capability", f"route lacks {required} capability"))
        elif route.billing_mode == "metered" and not allow_metered:
            rejected.append(_reason(route, "metered_not_authorized", "metered usage is disabled for subscription pools"))
        elif route.paid_usage_possible and not allow_paid_usage:
            rejected.append(_reason(route, "paid_usage_not_authorized", "route can trigger charges or paid overage"))
        else:
            if not route.verification.quota_visible or route.quota_state == "unknown":
                rejected.append(_reason(route, "quota_unknown", "remaining quota is not visible"))
            eligible.append(route)

    if not eligible:
        summary = "; ".join(f"{item['route_id']}: {item['code']}" for item in rejected)
        raise PoolConfigurationError(f"No eligible route for {required!r}. {summary}")

    eligible.sort(key=_rank)
    selected = eligible[0]
    fallback_order = tuple(item.route_id for item in eligible[1:])
    for item in eligible[1:]:
        rejected.append(_reason(item, "lower_ranked_fallback", "eligible fallback retained in pool order"))
    for item in rejected:
        item["next_route"] = selected.route_id

    reserve_note = " after honoring a soft reserve" if any(_reserve_active(item) for item in candidates) else ""
    quality_note = " for the explicit quality requirement" if explicit_quality_requirement else ""
    quota_note = "; quota is unknown" if (
        not selected.verification.quota_visible or selected.quota_state == "unknown"
    ) else ""
    authorization: list[str] = []
    if selected.billing_mode == "metered":
        authorization.append("metered usage explicitly authorized")
    elif allow_metered:
        authorization.append("metered usage explicitly authorized but not selected")
    else:
        authorization.append("metered usage not authorized")
    if selected.paid_usage_possible:
        authorization.append("paid overage explicitly authorized")
    elif allow_paid_usage:
        authorization.append("paid overage explicitly authorized but not selected")
    else:
        authorization.append("paid overage not authorized")
    authorization_note = "; ".join(authorization)
    reason = (
        f"selected {selected.route_id} from pool {selected.subscription_pool}: "
        f"selected {selected.billing_mode} billing; {authorization_note}; "
        f"verified capable route{reserve_note}{quality_note}{quota_note}; "
        "unselected metered and paid-overage routes remain fail-closed"
    )
    return PoolSelection(selected, tuple(rejected), fallback_order, reason)


def _generation(name: str, family: str) -> int | None:
    match = re.search(rf"{re.escape(family)}[-_ ]?(\d+)", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def select_model_from_roster(
    roster: Sequence[str],
    *,
    exact: str = "",
    family: str = "",
    minimum_generation: int | None = None,
) -> str:
    """Resolve a model from a runtime roster without baking transient IDs into policy."""
    models = tuple(str(item).strip() for item in roster if str(item).strip())
    if exact:
        if exact not in models:
            raise PoolConfigurationError(f"Configured model {exact!r} is absent from the runtime roster.")
        return exact
    if not family:
        raise PoolConfigurationError("Model selection requires an exact id or a family preference.")
    matching: list[tuple[int, str]] = []
    for model in models:
        generation = _generation(model, family)
        if generation is None:
            continue
        if minimum_generation is None or generation >= minimum_generation:
            matching.append((generation, model))
    if not matching:
        raise PoolConfigurationError(
            f"No runtime model matches family {family!r} at generation {minimum_generation!r}."
        )
    # Prefer the newest matching family.  A Sonnet-family rule can therefore
    # never silently drift to Opus merely because Opus is stronger.
    matching.sort(key=lambda item: (-item[0], item[1]))
    return matching[0][1]
