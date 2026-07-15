"""Fail-closed allowlisted route registry for Ship's Crew workers.

The static registry is only a catalog. A route becomes selectable after its
exact executor/model/mode has a non-secret runtime calibration receipt. This
keeps stale inventories and free-text model choices out of dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


TIERS = frozenset({"T0", "T1", "T2", "T3", "T4"})
OUTPUT_CLASSES = frozenset({"O0", "O1", "O2", "O3+"})
PROFILES = frozenset({"captain", "engineer", "navigator", "pirate"})
GOVERNANCE_RANK = {"lite": 0, "standard": 1, "constitutional": 2}
EXCLUDED_MODEL_TERMS = ("codex gpt-5.6", "gpt-5.6", "sol")


class RoutePolicyError(ValueError):
    """A route or capability request cannot be safely satisfied."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class CalibrationReceipt:
    route_id: str
    executor: str
    model: str
    mode: str
    authenticated: bool
    protocol_valid: bool
    acceptance_fixture: bool
    receipt_ref: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CalibrationReceipt":
        required = ("route_id", "executor", "model", "mode", "authenticated", "protocol_valid", "acceptance_fixture", "receipt_ref")
        missing = [name for name in required if name not in value]
        if missing:
            raise RoutePolicyError("calibration_missing", f"missing fields: {', '.join(missing)}")
        if not all(isinstance(value[name], bool) for name in ("authenticated", "protocol_valid", "acceptance_fixture")):
            raise RoutePolicyError("calibration_shape", "calibration booleans must be explicit")
        ref = str(value["receipt_ref"])
        if not ref.startswith("/"):
            raise RoutePolicyError("calibration_ref", "calibration receipt reference must be absolute")
        return cls(
            route_id=str(value["route_id"]),
            executor=str(value["executor"]),
            model=str(value["model"]),
            mode=str(value["mode"]),
            authenticated=value["authenticated"],
            protocol_valid=value["protocol_valid"],
            acceptance_fixture=value["acceptance_fixture"],
            receipt_ref=ref,
        )

    def matches(self, route: "RouteRecord") -> bool:
        return (
            self.route_id == route.route_id
            and self.executor == route.executor
            and self.model == route.model
            and self.mode == route.mode
            and self.authenticated
            and self.protocol_valid
            and self.acceptance_fixture
        )


@dataclass(frozen=True)
class RouteRecord:
    route_id: str
    provider: str
    model: str
    executor: str
    mode: str
    reasoning_effort: str | None
    quota_domain: str
    write_scope: str
    profiles: tuple[str, ...]
    complexity_tiers: tuple[str, ...]
    output_classes: tuple[str, ...]
    fallback_routes: tuple[str, ...]
    calibration: CalibrationReceipt | None = None

    @property
    def calibrated(self) -> bool:
        return self.calibration is not None and self.calibration.matches(self)

    @property
    def excluded(self) -> bool:
        text = f"{self.provider} {self.model}".casefold()
        return any(term in text for term in EXCLUDED_MODEL_TERMS)

    @classmethod
    def from_mapping(cls, route_id: str, value: Mapping[str, Any]) -> "RouteRecord":
        required = ("provider", "model", "executor", "mode", "quota_domain", "write_scope", "profiles", "complexity_tiers", "output_classes", "fallback_routes")
        missing = [name for name in required if name not in value]
        if missing:
            raise RoutePolicyError("route_missing", f"{route_id}: missing fields: {', '.join(missing)}")
        profiles = tuple(str(item) for item in value["profiles"])
        tiers = tuple(str(item) for item in value["complexity_tiers"])
        outputs = tuple(str(item) for item in value["output_classes"])
        if not set(profiles) <= PROFILES:
            raise RoutePolicyError("route_profile", f"{route_id}: unknown profile")
        if not set(tiers) <= TIERS:
            raise RoutePolicyError("route_tier", f"{route_id}: unknown complexity tier")
        if not set(outputs) <= OUTPUT_CLASSES:
            raise RoutePolicyError("route_output", f"{route_id}: unknown output class")
        if value["write_scope"] not in {"read-only", "worktree", "scoped-internal", "broad"}:
            raise RoutePolicyError("route_scope", f"{route_id}: invalid write scope")
        calibration = value.get("calibration")
        return cls(
            route_id=str(route_id),
            provider=str(value["provider"]),
            model=str(value["model"]),
            executor=str(value["executor"]),
            mode=str(value["mode"]),
            reasoning_effort=(str(value["reasoning_effort"]) if value.get("reasoning_effort") is not None else None),
            quota_domain=str(value["quota_domain"]),
            write_scope=str(value["write_scope"]),
            profiles=profiles,
            complexity_tiers=tiers,
            output_classes=outputs,
            fallback_routes=tuple(str(item) for item in value["fallback_routes"]),
            calibration=CalibrationReceipt.from_mapping(calibration) if calibration is not None else None,
        )


class RouteRegistry:
    def __init__(self, routes: Mapping[str, RouteRecord], *, schema_version: str = "crew-route-registry/v1"):
        if schema_version != "crew-route-registry/v1":
            raise RoutePolicyError("registry_version", "unsupported registry version")
        self.schema_version = schema_version
        self.routes = dict(routes)
        if set(self.routes) != {route.route_id for route in self.routes.values()}:
            raise RoutePolicyError("route_id", "route map key must match route_id")
        for route in self.routes.values():
            if route.excluded:
                # Exclusions are valid catalog entries but never selectable.
                continue
            for fallback in route.fallback_routes:
                if fallback not in self.routes:
                    raise RoutePolicyError("fallback_unknown", f"{route.route_id}: unknown fallback {fallback}")
        self._check_fallback_cycles()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RouteRegistry":
        if value.get("schema_version") != "crew-route-registry/v1":
            raise RoutePolicyError("registry_version", "schema_version must be crew-route-registry/v1")
        raw_routes = value.get("routes")
        if not isinstance(raw_routes, Mapping) or not raw_routes:
            raise RoutePolicyError("routes", "routes must be a non-empty object")
        return cls({str(route_id): RouteRecord.from_mapping(str(route_id), raw) for route_id, raw in raw_routes.items()})

    def _check_fallback_cycles(self) -> None:
        def visit(route_id: str, stack: tuple[str, ...]) -> None:
            if route_id in stack:
                cycle = " -> ".join((*stack, route_id))
                raise RoutePolicyError("fallback_cycle", cycle)
            route = self.routes[route_id]
            for fallback in route.fallback_routes:
                visit(fallback, (*stack, route_id))

        for route_id in self.routes:
            visit(route_id, ())

    def get(self, route_id: str) -> RouteRecord:
        try:
            return self.routes[route_id]
        except KeyError as exc:
            raise RoutePolicyError("unknown_route", route_id) from exc

    def active_routes(self) -> tuple[RouteRecord, ...]:
        return tuple(route for route in self.routes.values() if route.calibrated and not route.excluded)


@dataclass(frozen=True)
class RouteRequest:
    profile: str
    complexity_tier: str
    output_class: str
    write_scope: str
    governance_class: str
    quota_domains_available: frozenset[str] = frozenset()
    requested_route_id: str | None = None
    required_executor: str | None = None


@dataclass(frozen=True)
class RouteSelection:
    requested_route_id: str | None
    effective_route_id: str | None
    executor: str | None
    model: str | None
    mode: str | None
    reasoning_effort: str | None
    quota_domain: str | None
    rationale: tuple[str, ...]


def _scope_allows(route_scope: str, requested_scope: str) -> bool:
    if requested_scope == "read-only":
        return True
    if requested_scope == "worktree":
        return route_scope in {"worktree", "scoped-internal", "broad"}
    if requested_scope == "scoped-internal":
        return route_scope in {"scoped-internal", "broad"}
    return route_scope == "broad"


def select_route(request: RouteRequest, registry: RouteRegistry) -> RouteSelection:
    if request.complexity_tier not in TIERS:
        raise RoutePolicyError("tier", request.complexity_tier)
    if request.output_class not in OUTPUT_CLASSES:
        raise RoutePolicyError("output_class", request.output_class)
    if request.governance_class not in GOVERNANCE_RANK:
        raise RoutePolicyError("governance", request.governance_class)
    if request.complexity_tier == "T0":
        return RouteSelection(request.requested_route_id, None, None, None, None, None, None, ("deterministic-t0",))
    if request.profile not in PROFILES:
        raise RoutePolicyError("profile", request.profile)
    if request.requested_route_id:
        candidate_ids = (request.requested_route_id,)
    else:
        candidate_ids = tuple(registry.routes)
    rejected: list[str] = []
    for route_id in candidate_ids:
        route = registry.get(route_id)
        if route.excluded:
            rejected.append(f"{route_id}:excluded")
            continue
        if not route.calibrated:
            rejected.append(f"{route_id}:uncalibrated")
            continue
        if request.profile not in route.profiles:
            rejected.append(f"{route_id}:profile")
            continue
        if request.complexity_tier not in route.complexity_tiers:
            rejected.append(f"{route_id}:tier")
            continue
        if request.output_class not in route.output_classes:
            rejected.append(f"{route_id}:output")
            continue
        if not _scope_allows(route.write_scope, request.write_scope):
            rejected.append(f"{route_id}:scope")
            continue
        if request.profile in {"navigator", "pirate"} and route.write_scope not in {"read-only"}:
            rejected.append(f"{route_id}:profile-write-scope")
            continue
        if request.required_executor and route.executor != request.required_executor:
            rejected.append(f"{route_id}:executor")
            continue
        if request.quota_domains_available and route.quota_domain not in request.quota_domains_available:
            rejected.append(f"{route_id}:quota")
            continue
        return RouteSelection(
            request.requested_route_id,
            route.route_id,
            route.executor,
            route.model,
            route.mode,
            route.reasoning_effort,
            route.quota_domain,
            ("allowlisted", "calibrated", f"tier-{request.complexity_tier}"),
        )
    raise RoutePolicyError("no_route", "; ".join(rejected) or "no candidate route")
