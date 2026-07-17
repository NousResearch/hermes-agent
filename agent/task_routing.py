"""Authoritative, side-effect-free task routing for CLI, dashboard, and delegation.

This module deliberately selects *configured* profile/model/provider capacity only.
It does not resolve credentials or make a model call.  Execution surfaces persist
its returned receipt alongside their existing SessionDB records.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

from agent.subscription_pool import (
    PoolConfigurationError,
    PoolRouteConfig,
    PoolVerification,
    is_approved_ollama_cloud_endpoint,
    select_model_from_roster,
    select_subscription_route,
)


class ProfileResolutionError(ValueError):
    """A requested profile cannot safely participate in task routing."""


class _OllamaRuntimeProbeError(PoolConfigurationError):
    """Retain the observed endpoint so selection can reject it precisely."""

    def __init__(self, message: str, base_url: str) -> None:
        super().__init__(message)
        self.base_url = base_url


@dataclass(frozen=True)
class ProfileRouteConfig:
    profile: str
    home: str
    provider: str
    model: str
    capabilities: tuple[str, ...]
    budget_tokens: int
    pool_alias: str
    pool_type: str
    cost_mode: str


@dataclass(frozen=True)
class TaskRoute:
    task: str
    role: str
    profile: str
    provider: str
    model: str
    pool_alias: str
    pool_type: str
    cost_mode: str
    budget_tokens: int
    delegated: bool
    fallback_reason: str | None = None
    rejected: tuple[dict[str, str], ...] = field(default_factory=tuple)
    home: str = ""
    task_class: str = "general"
    execution_surface: str = "hermes"
    subscription_pool: str = ""
    billing_mode: str = "subscription"
    quota_state: str = "unknown"
    available_usage: dict[str, Any] = field(default_factory=dict)
    paid_usage_possible: bool = False
    fallback_order: tuple[str, ...] = field(default_factory=tuple)
    selection_reason: str = ""
    runtime_base_url: str = ""
    pool_route_id: str = ""
    allow_metered: bool = False
    allow_paid_usage: bool = False
    _fallback_routes: tuple[PoolRouteConfig, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )

    def explain(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("_fallback_routes", None)
        return payload


@dataclass(frozen=True)
class ProfileGovernorSettings:
    """Effective profile-scoped controls used by main and routed agents."""

    max_turns: int
    max_request_input_tokens: int
    compression_threshold: float


_ROLE_ALIASES = {
    "coding": "implementation",
    "strategic-synthesis": "research",
    "strategic_synthesis": "research",
    "synthesis": "research",
    "review": "verification",
    "testing": "verification",
    "test": "verification",
    "verify": "verification",
    "verifier": "verification",
    "continuity": "operations",
    "recordkeeping": "operations",
    "records": "operations",
    "localcred-domain": "localcred",
    "martial-os": "martial",
    "martial_os": "martial",
    "general": "generic",
    "small": "generic",
}

_ROLE_CAPABILITIES = {
    "orchestrator": "general",
    "implementation": "coding",
    "research": "research",
    "verification": "verification",
    "operations": "operations",
    "localcred": "localcred",
    "martial": "martial",
    "generic": "general",
}

_ROLE_PREFERENCES = {
    "orchestrator": "default",
    "implementation": "engineering",
    "research": "intelligence",
    "verification": "assurance",
    "operations": "operations",
    "localcred": "localcred",
    "martial": "martial",
    "generic": "default",
}

# Never use registry enumeration as a tiebreaker: it is an implementation
# detail, not routing policy.  Default stays ahead of unrelated specialists
# when a preferred specialist is unavailable, and that fallback is recorded.
_FALLBACK_PROFILE_ORDER = (
    "default", "operations", "engineering", "assurance", "intelligence", "localcred", "martial",
)


def _normalized_role(role: str) -> str:
    normalized = str(role or "orchestrator").strip().lower().replace(" ", "-")
    return _ROLE_ALIASES.get(normalized, normalized)


def _domain_role(task: str) -> str | None:
    """Classify only explicit portfolio-domain references; generic work stays generic."""
    normalized = str(task or "").lower()
    if re.search(r"\blocalcred\b", normalized):
        return "localcred"
    if re.search(r"\bmartial[ _-]+os\b", normalized):
        return "martial"
    return None


def _routing_policy(task: str, role: str) -> tuple[str, str, str]:
    """Return canonical role, required capability, and preferred profile.

    Explicit domain references outrank generic role preferences, but caller
    profile selection is handled separately and remains authoritative.
    """
    canonical_role = _domain_role(task) or _normalized_role(role)
    capability = _ROLE_CAPABILITIES.get(canonical_role, "general")
    preferred = _ROLE_PREFERENCES.get(canonical_role, "default")
    return canonical_role, capability, preferred


def _fallback_sort_key(candidate: ProfileRouteConfig) -> tuple[int, int, int, str]:
    """Stable capacity-aware fallback order independent of registry order."""
    try:
        policy_rank = _FALLBACK_PROFILE_ORDER.index(candidate.profile)
    except ValueError:
        policy_rank = len(_FALLBACK_PROFILE_ORDER)
    return (candidate.cost_mode != "included", candidate.pool_type == "api_key", policy_rank, candidate.profile)


def _profile_configs(profile: str | None = None) -> Iterable[ProfileRouteConfig]:
    """Read all valid profiles through the canonical profile registry."""
    from hermes_cli.profiles import get_profile_registry

    registry = get_profile_registry()
    names = (profile,) if profile else registry.names()
    for name in names:
        record = registry.resolve(name)
        cfg = record.config
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        model = str(model_cfg.get("default") or model_cfg.get("model") or "").strip()
        provider = str(model_cfg.get("provider") or "auto").strip()
        if not model:
            continue
        routing_cfg = cfg.get("task_routing") if isinstance(cfg.get("task_routing"), dict) else {}
        capabilities = routing_cfg.get("capabilities", ("coding", "research", "verification", "general"))
        if not isinstance(capabilities, (list, tuple, set)):
            capabilities = ()
        budget = routing_cfg.get("task_budget_tokens", 16000)
        try:
            budget = max(1, int(budget))
        except (TypeError, ValueError):
            budget = 16000
        pool_alias, pool_type, cost_mode = _pool_for(provider)
        yield ProfileRouteConfig(
            profile=record.name, home=str(record.home), provider=provider, model=model,
            capabilities=tuple(str(item).lower() for item in capabilities), budget_tokens=budget,
            pool_alias=pool_alias, pool_type=pool_type, cost_mode=cost_mode,
        )


def _pool_for(provider: str) -> tuple[str, str, str]:
    """Return a non-secret alias and capacity attribution for a configured provider."""
    normalized = provider.lower()
    try:
        from agent.credential_pool import load_pool
        pool = load_pool(provider)
        entry = pool.current() or next(iter(pool.entries()), None)
    except Exception:
        entry = None
    if entry is not None:
        auth_type = str(getattr(entry, "auth_type", "") or "").lower()
        source = str(getattr(entry, "source", "") or "").lower()
        # OAuth / installed-subscription credentials are included capacity; API
        # keys remain metered.  The alias deliberately exposes no account label.
        included = auth_type in {"oauth", "oauth_token", "subscription"} or any(
            marker in source for marker in ("codex", "claude_code", "oauth", "subscription")
        )
        return f"{normalized}:pool", "oauth" if included else "api_key", "included" if included else "metered"
    if normalized in {"ollama", "local"}:
        return f"{normalized}:local", "local", "included"
    return f"{normalized}:configured", "configured", "metered"


def _runtime_roster(source: str, profile_home: str) -> tuple[str, ...]:
    """Discover the current provider roster without making a model call."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(profile_home)
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested=source)
        if source == "openai-codex":
            from hermes_cli.codex_models import get_codex_model_ids

            return tuple(get_codex_model_ids(runtime.get("api_key")))

    finally:
        reset_hermes_home_override(token)
    raise PoolConfigurationError(f"Unsupported runtime model source {source!r}.")


def _authenticated_ollama_cloud_roster(
    profile_home: str,
    *,
    provider: str,
    target_model: str,
) -> tuple[str, tuple[str, ...]]:
    """Probe only the authenticated Ollama Cloud ``/v1/models`` roster.

    This deliberately bypasses the merged discovery helper: models.dev and its
    disk cache cannot prove that this credential can execute this model.
    """
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(profile_home)
    try:
        from hermes_cli.models import fetch_api_models_strict
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested=provider, target_model=target_model)
    finally:
        reset_hermes_home_override(token)
    api_key = str(runtime.get("api_key") or "").strip()
    base_url = str(runtime.get("base_url") or "").strip()
    if not api_key:
        raise _OllamaRuntimeProbeError("authenticated Ollama runtime has no API credential", base_url)
    if not is_approved_ollama_cloud_endpoint(base_url):
        raise _OllamaRuntimeProbeError(
            "authenticated Ollama runtime is not an approved Cloud HTTPS endpoint", base_url
        )
    roster = fetch_api_models_strict(api_key, base_url)
    if not roster:
        raise _OllamaRuntimeProbeError("authenticated Ollama runtime returned no model roster", base_url)
    return base_url, tuple(str(model).strip() for model in roster if str(model).strip())


def _runtime_usage(source: str, profile_home: str) -> tuple[str, float | None, dict[str, Any]]:
    """Resolve visible subscription usage without exposing account credentials."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(profile_home)
    try:
        from agent.account_usage import fetch_account_usage

        snapshot = fetch_account_usage(source)
    finally:
        reset_hermes_home_override(token)
    if snapshot is None or not snapshot.available or not snapshot.windows:
        return "unknown", None, {"source": source, "available": False}
    used = max(float(window.used_percent) for window in snapshot.windows)
    remaining = max(0.0, 100.0 - used)
    state = "exhausted" if used >= 100.0 else "available"
    return state, remaining, {
        "source": snapshot.source,
        "plan": snapshot.plan,
        "windows": [
            {"label": window.label, "used_percent": float(window.used_percent)}
            for window in snapshot.windows
        ],
    }


def _verification(
    raw: Mapping[str, Any],
    route_id: str,
    *,
    configured_subscription_pool: bool,
    require_cloud_model: bool,
) -> PoolVerification:
    required = {
        "installed", "authenticated", "models_available", "headless",
        "workspace", "tools", "quota_visible",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise PoolConfigurationError(
            f"Route {route_id!r} verification is incomplete: missing {', '.join(missing)}."
        )
    return PoolVerification(
        installed=raw["installed"] is True,
        authenticated=raw["authenticated"] is True,
        models_available=raw["models_available"] is True,
        headless=raw["headless"] is True,
        workspace=raw["workspace"] is True,
        tools=raw["tools"] is True,
        quota_visible=raw["quota_visible"] is True,
        # Ollama's configured API-key-to-pool binding is durable route policy,
        # not request metadata. It therefore supplies billing attribution
        # without a per-request billing field.
        billing_verified=(
            raw.get("billing_verified") is True or configured_subscription_pool
        ),
        cloud_verified=raw.get("cloud_verified", True) is True,
        cloud_model_verified=(
            raw.get("cloud_model_verified", not require_cloud_model) is True
        ),
    )


def _configured_pool_routes(
    profile: ProfileRouteConfig,
    raw_routes: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[PoolRouteConfig, ...]:
    """Load caller policy while retaining the selected profile's identity/model."""
    if raw_routes is None:
        from hermes_cli.config import load_config

        config = load_config()
        routing = config.get("task_routing") if isinstance(config.get("task_routing"), dict) else {}
        configured = routing.get("pools", ())
        if not isinstance(configured, list):
            raise PoolConfigurationError("task_routing.pools must be a list.")
        raw_routes = configured
    parsed: list[PoolRouteConfig] = []
    for raw in raw_routes:
        if not isinstance(raw, Mapping):
            raise PoolConfigurationError("Each task_routing.pools entry must be a mapping.")
        route_id = str(raw.get("id") or "").strip()
        enabled = raw.get("enabled", True) is True
        provider = str(raw.get("provider") or "").strip()
        subscription_pool = str(raw.get("subscription_pool") or "").strip()
        billing_mode = str(raw.get("billing_mode") or "").strip()
        is_ollama = provider.lower() in {"ollama", "ollama-cloud"}
        model = str(raw.get("model") or "").strip()
        family = str(raw.get("model_family") or "").strip()
        generation = raw.get("minimum_generation")
        if generation is not None:
            try:
                generation = int(generation)
            except (TypeError, ValueError) as exc:
                raise PoolConfigurationError(
                    f"Route {route_id!r} minimum_generation must be an integer."
                ) from exc
        if model == "profile":
            model = profile.model
        source = str(raw.get("model_source") or "").strip()
        runtime_base_url = ""
        runtime_failure = ""
        if enabled and is_ollama:
            try:
                runtime_base_url, roster = _authenticated_ollama_cloud_roster(
                    profile.home, provider=provider, target_model=model
                )
                model = select_model_from_roster(
                    roster, exact=model, family=family, minimum_generation=generation
                )
            except PoolConfigurationError as exc:
                # The route remains inspectable so the frozen eligible fallback
                # order can advance to OpenAI rather than surfacing an opaque
                # routing-unavailable error.
                runtime_failure = str(exc)
                runtime_base_url = str(getattr(exc, "base_url", runtime_base_url) or "")
        elif enabled and source:
            roster = _runtime_roster(source, profile.home)
            model = select_model_from_roster(
                roster, exact=model, family=family, minimum_generation=generation
            )
        elif not model:
            # Disabled routes still require unambiguous attribution, but their
            # live roster is intentionally not probed during ordinary routing.
            model = f"{family}-family" if family else "unavailable"

        quota_state = str(raw.get("quota_state") or "unknown").strip().lower()
        remaining = raw.get("remaining_percent")
        available_usage = raw.get("available_usage")
        if not isinstance(available_usage, dict):
            available_usage = {}
        usage_source = str(raw.get("usage_source") or "").strip()
        if enabled and usage_source:
            quota_state, remaining, available_usage = _runtime_usage(usage_source, profile.home)
        if remaining is not None:
            try:
                remaining = float(remaining)
            except (TypeError, ValueError) as exc:
                raise PoolConfigurationError(
                    f"Route {route_id!r} remaining_percent must be numeric."
                ) from exc

        verification_raw = raw.get("verification")
        if not isinstance(verification_raw, Mapping):
            raise PoolConfigurationError(f"Route {route_id!r} requires a verification mapping.")
        capabilities = raw.get("capabilities")
        if not isinstance(capabilities, list):
            raise PoolConfigurationError(f"Route {route_id!r} capabilities must be a list.")
        verification = _verification(
            verification_raw,
            route_id,
            configured_subscription_pool=(
                is_ollama
                and billing_mode == "subscription"
                and bool(subscription_pool)
            ),
            require_cloud_model=is_ollama,
        )
        if runtime_failure:
            if "no API credential" in runtime_failure:
                verification = replace(verification, authenticated=False)
            elif "approved Cloud HTTPS endpoint" not in runtime_failure:
                verification = replace(verification, models_available=False)
        parsed.append(PoolRouteConfig(
            route_id=route_id,
            provider=provider,
            model=model,
            execution_surface=str(raw.get("execution_surface") or "").strip(),
            subscription_pool=subscription_pool,
            billing_mode=billing_mode,
            capabilities=tuple(str(item).strip().lower() for item in capabilities),
            priority=int(raw.get("priority", 100)),
            verification=verification,
            enabled=enabled,
            paid_usage_possible=raw.get("paid_usage_possible", True) is True,
            quota_state=quota_state,
            available_usage=available_usage,
            soft_reserve_percent=(
                float(raw["soft_reserve_percent"])
                if raw.get("soft_reserve_percent") is not None else None
            ),
            remaining_percent=remaining,
            recent_success=float(raw.get("recent_success", 1.0)),
            verification_burden=int(raw.get("verification_burden", 0)),
            endpoint_host=str(raw.get("endpoint_host") or "").strip(),
            runtime_base_url=runtime_base_url,
            model_family=family,
            minimum_generation=generation,
            blockers=tuple(
                str(item).strip()
                for item in raw.get("blockers", ())
                if str(item).strip()
            ),
        ))
    return tuple(parsed)


def resolve_task_route(
    task: str,
    *,
    role: str = "orchestrator",
    profile: str | None = None,
    pool_routes: Sequence[PoolRouteConfig] | None = None,
    allow_metered: bool = False,
    allow_paid_usage: bool = False,
    explicit_quality_requirement: bool = False,
) -> TaskRoute:
    """Select one configured route and list rejected candidates without calling a model."""
    role, capability, preferred_profile = _routing_policy(task, role)
    candidates = list(_profile_configs(profile))
    rejected: list[dict[str, str]] = []
    eligible: list[ProfileRouteConfig] = []
    fallback_reason: str | None = None
    for candidate in candidates:
        if capability not in candidate.capabilities and "general" not in candidate.capabilities:
            rejected.append({"profile": candidate.profile, "reason": f"missing {capability} capability"})
        else:
            eligible.append(candidate)
    if not eligible:
        requested = f" profile {profile!r}" if profile else " any registered profile"
        raise ProfileResolutionError(f"No configured route for {role!r} using{requested}.")

    # An explicit selection is a caller contract, not a routing hint.
    if profile is not None:
        selected = eligible[0]
    else:
        preferred = next((item for item in eligible if item.profile == preferred_profile), None)
        if preferred is not None:
            selected = preferred
        else:
            configured_preferred = next(
                (item for item in candidates if item.profile == preferred_profile), None
            )
            if configured_preferred is None:
                fallback_reason = f"preferred profile {preferred_profile!r} is unavailable"
            else:
                fallback_reason = (
                    f"preferred profile {preferred_profile!r} is missing {capability} capability"
                )
            rejected.append({"profile": preferred_profile, "reason": fallback_reason})
            eligible.sort(key=_fallback_sort_key)
            selected = eligible[0]

    for candidate in eligible:
        if candidate.profile != selected.profile:
            rejected.append({"profile": candidate.profile, "reason": "lower-ranked policy fallback"})

    configured_pools = tuple(pool_routes) if pool_routes is not None else _configured_pool_routes(selected)
    if configured_pools:
        pool_selection = select_subscription_route(
            configured_pools,
            capability=capability,
            allow_metered=allow_metered,
            allow_paid_usage=allow_paid_usage,
            explicit_quality_requirement=explicit_quality_requirement,
        )
        worker = pool_selection.selected
        rejected.extend(pool_selection.rejected)
        provider = worker.provider
        model = worker.model
        pool_alias = worker.subscription_pool
        pool_type = "subscription" if worker.billing_mode == "subscription" else "api_key"
        cost_mode = "included" if worker.billing_mode == "subscription" else "metered"
        execution_surface = worker.execution_surface
        quota_state = worker.quota_state
        available_usage = dict(worker.available_usage)
        paid_usage_possible = worker.paid_usage_possible
        fallback_order = pool_selection.fallback_order
        selection_reason = pool_selection.selection_reason
        billing_mode = worker.billing_mode
        runtime_base_url = worker.runtime_base_url
        pool_route_id = worker.route_id
        by_id = {item.route_id: item for item in configured_pools}
        fallback_routes = tuple(
            by_id[route_id] for route_id in fallback_order if route_id in by_id
        )
    else:
        provider, model = selected.provider, selected.model
        pool_alias, pool_type, cost_mode = selected.pool_alias, selected.pool_type, selected.cost_mode
        execution_surface = "hermes"
        quota_state = "unknown"
        available_usage = {}
        paid_usage_possible = cost_mode == "metered"
        fallback_order = ()
        selection_reason = "legacy configured profile route; no subscription-pool policy is configured"
        billing_mode = "subscription" if cost_mode == "included" else "metered"
        runtime_base_url = ""
        pool_route_id = ""
        fallback_routes = ()
    return TaskRoute(
        task=task, role=role, profile=selected.profile, provider=provider, home=selected.home,
        model=model, pool_alias=pool_alias, pool_type=pool_type,
        cost_mode=cost_mode, budget_tokens=selected.budget_tokens,
        delegated=role not in {"orchestrator", "generic"},
        fallback_reason=fallback_reason, rejected=tuple(rejected),
        task_class=capability, execution_surface=execution_surface,
        subscription_pool=pool_alias, billing_mode=billing_mode,
        quota_state=quota_state, available_usage=available_usage,
        paid_usage_possible=paid_usage_possible, fallback_order=fallback_order,
        selection_reason=selection_reason, runtime_base_url=runtime_base_url,
        pool_route_id=pool_route_id, allow_metered=allow_metered,
        allow_paid_usage=allow_paid_usage, _fallback_routes=fallback_routes,
    )


@contextmanager
def profile_runtime_scope(route: TaskRoute) -> Iterator[None]:
    """Temporarily resolve config/auth from a selected profile only.

    This is intentionally a tiny adapter around the existing profile-home
    override.  It does not copy credentials or mutate the selected profile.
    """
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(route.home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def resolve_profile_governor(route: TaskRoute) -> ProfileGovernorSettings:
    """Load effective governor values from the selected profile's schema."""
    with profile_runtime_scope(route):
        from agent.usage_guardrails import UsageGuardrail
        from hermes_cli.config import load_config

        config = load_config()
    agent_config = config.get("agent") if isinstance(config.get("agent"), dict) else {}
    compression = config.get("compression") if isinstance(config.get("compression"), dict) else {}
    try:
        max_turns = int(agent_config.get("max_turns"))
    except (TypeError, ValueError) as exc:
        raise ProfileResolutionError(
            f"Profile {route.profile!r} has invalid agent.max_turns."
        ) from exc
    if max_turns < 1:
        raise ProfileResolutionError(f"Profile {route.profile!r} agent.max_turns must be positive.")
    try:
        threshold = float(compression.get("threshold"))
    except (TypeError, ValueError) as exc:
        raise ProfileResolutionError(
            f"Profile {route.profile!r} has invalid compression.threshold."
        ) from exc
    if not 0.0 < threshold <= 1.0:
        raise ProfileResolutionError(
            f"Profile {route.profile!r} compression.threshold must be in (0, 1]."
        )
    guardrail = UsageGuardrail.from_config(
        config,
        scope="interactive",
        subscription_included=route.billing_mode == "subscription",
    )
    return ProfileGovernorSettings(
        max_turns=max_turns,
        max_request_input_tokens=guardrail.limits.max_request_input_tokens,
        compression_threshold=threshold,
    )


def _resolve_route_runtime_once(route: TaskRoute) -> dict[str, Any]:
    """Resolve one frozen route's runtime inside its profile scope.

    The returned dict is execution-only and can contain an in-memory credential;
    callers must persist ``routing_receipt`` instead.  A specialist route never
    retries through the caller's default/fallback chain.
    """
    if route.execution_surface != "hermes":
        raise ProfileResolutionError(
            f"Route {route.subscription_pool!r} requires unsupported execution surface "
            f"{route.execution_surface!r}; it cannot be dispatched by Hermes yet."
        )
    if route.delegated and route.provider.lower() in {"", "auto"}:
        raise ProfileResolutionError(
            f"Specialist route for profile {route.profile!r} has no explicit provider."
        )
    with profile_runtime_scope(route):
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = dict(resolve_runtime_provider(
            requested=route.provider,
            target_model=route.model,
        ))
    if route.provider.lower() in {"ollama", "ollama-cloud"}:
        base_url = str(runtime.get("base_url") or "").strip()
        if not route.runtime_base_url or base_url != route.runtime_base_url:
            raise ProfileResolutionError(
                f"Route {route.subscription_pool!r} runtime endpoint changed after selection."
            )
        if not is_approved_ollama_cloud_endpoint(base_url):
            raise ProfileResolutionError(
                f"Route {route.subscription_pool!r} has an unapproved Ollama runtime endpoint."
            )
        api_key = str(runtime.get("api_key") or "").strip()
        if not api_key:
            raise ProfileResolutionError(
                f"Route {route.subscription_pool!r} has no runtime API credential."
            )
        from hermes_cli.models import fetch_api_models_strict

        roster = fetch_api_models_strict(api_key, base_url)
        if not roster or route.model not in roster:
            raise ProfileResolutionError(
                f"Route {route.subscription_pool!r} selected model is absent from its live runtime roster."
            )
    if not runtime.get("base_url"):
        raise ProfileResolutionError(
            f"Route {route.profile!r}/{route.provider!r} is unavailable: no runtime base URL."
        )
    if not runtime.get("api_key") and route.pool_type != "local":
        raise ProfileResolutionError(
            f"Route {route.profile!r}/{route.provider!r} is unavailable: no runtime credential."
        )
    return runtime


def _runtime_failure_code(route: TaskRoute, error: Exception) -> str:
    detail = str(error).lower()
    if route.provider.lower() in {"ollama", "ollama-cloud"}:
        if "endpoint" in detail:
            return "endpoint_validation_failed"
        if "credential" in detail or "auth" in detail:
            return "authentication_failed"
        if "roster" in detail or "model" in detail:
            return "model_unavailable"
    return "runtime_unavailable"


def _fallback_task_route(
    route: TaskRoute,
    worker: PoolRouteConfig,
    remaining: tuple[PoolRouteConfig, ...],
    error: Exception,
) -> TaskRoute:
    code = _runtime_failure_code(route, error)
    failure = {
        "route_id": route.pool_route_id or route.subscription_pool,
        "code": code,
        "reason": str(error),
        "next_route": worker.route_id,
    }
    return replace(
        route,
        provider=worker.provider,
        model=worker.model,
        pool_alias=worker.subscription_pool,
        pool_type="subscription" if worker.billing_mode == "subscription" else "api_key",
        cost_mode="included" if worker.billing_mode == "subscription" else "metered",
        execution_surface=worker.execution_surface,
        subscription_pool=worker.subscription_pool,
        billing_mode=worker.billing_mode,
        quota_state=worker.quota_state,
        available_usage=dict(worker.available_usage),
        paid_usage_possible=worker.paid_usage_possible,
        fallback_reason=f"{route.pool_route_id or route.subscription_pool}: {code}",
        rejected=route.rejected + (failure,),
        fallback_order=tuple(item.route_id for item in remaining),
        selection_reason=(
            f"runtime fallback from {route.pool_route_id or route.subscription_pool} "
            f"to {worker.route_id}: {code}"
        ),
        runtime_base_url=worker.runtime_base_url,
        pool_route_id=worker.route_id,
        _fallback_routes=remaining,
    )


def resolve_route_runtime(route: TaskRoute) -> dict[str, Any]:
    """Resolve a frozen route, advancing only through its frozen eligible fallbacks."""
    current = route
    while True:
        try:
            runtime = _resolve_route_runtime_once(current)
            runtime["_task_route"] = current
            return runtime
        except ProfileResolutionError as error:
            if not current._fallback_routes:
                raise
            worker, *rest = current._fallback_routes
            current = _fallback_task_route(current, worker, tuple(rest), error)


def routing_receipt(route: TaskRoute, *, task_id: str, parent_session_id: str | None = None,
                    child_session_id: str | None = None, token_usage: dict[str, int] | None = None,
                    cost_attribution: str | None = None, fallback_reason: str | None = None,
                    status: str = "dispatched", elapsed_time: float | None = None,
                    acceptance_evidence: Sequence[str] = (), escalation_reason: Mapping[str, Any] | None = None,
                    next_route: str | None = None, parent_profile: str | None = None) -> dict[str, Any]:
    """Canonical serializable receipt; callers may append it to existing session metadata."""
    receipt = route.explain()
    receipt.update({
        "task_id": task_id, "parent_session_id": parent_session_id,
        "child_session_id": child_session_id, "token_usage": token_usage or {},
        "cost_attribution": cost_attribution or route.cost_mode,
        "fallback_reason": fallback_reason or route.fallback_reason,
        "elapsed_time": elapsed_time,
        "acceptance_evidence": list(acceptance_evidence),
        "escalation_reason": dict(escalation_reason or {}),
        "next_route": next_route,
        "parent_profile": parent_profile,
        "child_delegation": route.delegated,
        "status": status,
    })
    return receipt
