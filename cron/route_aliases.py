"""Fail-closed logical route aliases for cron jobs.

The routing registry is the source of truth.  This module deliberately does
not load dotenv files, credentials, provider SDKs, or contact a network.  It
only validates and resolves the non-secret route description needed by the
cron scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextvars import ContextVar
import json
import logging
from pathlib import Path
from typing import Any, Mapping

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

FORBIDDEN_PROVIDER_TERMS = ("openrouter", "terra")
CANONICAL_ALIASES = {
    "deterministic.none",
    "coordinator.default",
    "executor.infrastructure",
    "planner.standard",
    "reasoning.premium",
}

DEFAULT_LIMITS: dict[str, Any] = {
    "max_delegation_depth": 0,
    "max_child_tasks": 0,
    "max_retries": 0,
    "timeout_seconds": 600,
    "provider_call_allowed": False,
    "premium_reasoning_allowed": False,
}

_active_route_limits: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "hermes_active_cron_route_limits", default=None
)


class RouteAliasError(ValueError):
    """Raised when a cron route alias cannot be used safely."""


@dataclass(frozen=True)
class ResolvedCronRoute:
    alias: str
    provider: str | None
    model: str | None
    effort: str | None
    role: str | None
    status: str
    fallback: tuple[Mapping[str, Any], ...]
    limits: Mapping[str, Any]

    @property
    def allow_provider_call(self) -> bool:
        return bool(self.limits.get("provider_call_allowed", False))

    @property
    def allow_premium_reasoning(self) -> bool:
        return bool(self.limits.get("premium_reasoning_allowed", False))

    @property
    def max_delegation_depth(self) -> int:
        return int(self.limits["max_delegation_depth"])

    @property
    def max_child_tasks(self) -> int:
        return int(self.limits["max_child_tasks"])

    @property
    def max_retries(self) -> int:
        return int(self.limits["max_retries"])

    @property
    def timeout_seconds(self) -> int:
        return int(self.limits["timeout_seconds"])


def get_active_route_limits() -> Mapping[str, Any] | None:
    """Return limits for the current cron agent context, if any."""
    return _active_route_limits.get()


def set_active_route_limits(limits: Mapping[str, Any] | None):
    """Install route limits in a ContextVar; caller must reset the token."""
    return _active_route_limits.set(limits)


def reset_active_route_limits(token) -> None:
    _active_route_limits.reset(token)


def _registry_path(hermes_home: str | Path | None) -> Path:
    home = Path(hermes_home).expanduser() if hermes_home is not None else get_hermes_home()
    return home / "model-routing.json"


def load_route_registry(hermes_home: str | Path | None = None) -> dict[str, Any]:
    """Load the routing registry without resolving credentials or providers."""
    path = _registry_path(hermes_home)
    try:
        with path.open(encoding="utf-8") as handle:
            registry = json.load(handle)
    except FileNotFoundError as exc:
        raise RouteAliasError(f"routing registry not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RouteAliasError(f"routing registry cannot be loaded: {path}") from exc
    if not isinstance(registry, dict) or not isinstance(registry.get("aliases"), dict):
        raise RouteAliasError("routing registry must contain an aliases object")
    return registry


def _forbidden_route_text(*values: Any) -> str | None:
    text = " ".join(str(value or "").strip().lower() for value in values)
    for term in FORBIDDEN_PROVIDER_TERMS:
        if term in text:
            return term
    return None


def _validated_limits(raw: Any, alias: str) -> dict[str, Any]:
    limits = dict(DEFAULT_LIMITS)
    if raw is not None:
        if not isinstance(raw, Mapping):
            raise RouteAliasError(f"route alias {alias!r} has invalid limits")
        limits.update(raw)
    integer_fields = (
        "max_delegation_depth",
        "max_child_tasks",
        "max_retries",
        "timeout_seconds",
    )
    for field in integer_fields:
        try:
            value = int(limits[field])
        except (TypeError, ValueError) as exc:
            raise RouteAliasError(f"route alias {alias!r} has invalid {field}") from exc
        if value < 0 or (field == "timeout_seconds" and value == 0):
            raise RouteAliasError(f"route alias {alias!r} has unsafe {field}={value}")
        limits[field] = value
    for field in ("provider_call_allowed", "premium_reasoning_allowed"):
        if not isinstance(limits[field], bool):
            raise RouteAliasError(f"route alias {alias!r} has invalid {field}")
    if limits["max_delegation_depth"] == 0 and limits["max_child_tasks"] != 0:
        raise RouteAliasError(
            f"route alias {alias!r} cannot allow child tasks at delegation depth zero"
        )
    return limits


def _build_route(alias: str, raw: Any, *, require_active: bool) -> ResolvedCronRoute:
    if not isinstance(raw, Mapping):
        raise RouteAliasError(f"route alias {alias!r} has invalid definition")
    status = str(raw.get("status") or "").strip().lower()
    if not status:
        raise RouteAliasError(f"route alias {alias!r} has no status")
    if require_active and status not in {"active", "configuration-verified-no-provider-call"}:
        raise RouteAliasError(f"route alias {alias!r} is disabled or not validated (status={status})")

    provider = str(raw.get("provider") or "").strip() or None
    model = str(raw.get("model") or "").strip() or None
    effort = str(raw.get("effort") or "").strip() or None
    role = str(raw.get("role") or "").strip() or None
    forbidden = _forbidden_route_text(provider, model, raw.get("base_url"))
    if forbidden:
        raise RouteAliasError(f"route alias {alias!r} uses forbidden provider/path term {forbidden!r}")

    if alias == "deterministic.none":
        if provider or model:
            raise RouteAliasError("deterministic.none must not define a provider or model")
    elif not provider or not model:
        # Preserve disabled/pending historical entries whose model ID is not
        # yet verified, but never permit them to become executable.  Active
        # and configuration-verified routes must be complete.
        if status in {"active", "configuration-verified-no-provider-call"}:
            raise RouteAliasError(f"route alias {alias!r} requires provider and model")

    fallback = raw.get("fallback") or []
    if not isinstance(fallback, list):
        raise RouteAliasError(f"route alias {alias!r} has invalid fallback")
    clean_fallback: list[Mapping[str, Any]] = []
    for entry in fallback:
        if not isinstance(entry, Mapping):
            raise RouteAliasError(f"route alias {alias!r} has invalid fallback entry")
        forbidden = _forbidden_route_text(entry.get("provider"), entry.get("model"), entry.get("base_url"))
        if forbidden:
            raise RouteAliasError(f"route alias {alias!r} has forbidden fallback term {forbidden!r}")
        clean_fallback.append(dict(entry))

    limits = _validated_limits(raw.get("limits"), alias)
    if alias == "deterministic.none" and limits["provider_call_allowed"]:
        raise RouteAliasError("deterministic.none cannot allow provider calls")
    if alias == "reasoning.premium" and not limits["premium_reasoning_allowed"]:
        # An explicitly disabled premium route is still valid in the registry;
        # require_active controls whether it may be fired.
        pass
    return ResolvedCronRoute(
        alias=alias,
        provider=provider,
        model=model,
        effort=effort,
        role=role,
        status=status,
        fallback=tuple(clean_fallback),
        limits=limits,
    )


def validate_route_alias_definition(alias: str, *, hermes_home: str | Path | None = None) -> ResolvedCronRoute:
    """Validate a known alias shape without activating a disabled route."""
    if not isinstance(alias, str) or not alias.strip():
        raise RouteAliasError("route_alias must be a non-empty string")
    registry = load_route_registry(hermes_home)
    raw = registry["aliases"].get(alias)
    if raw is None:
        raise RouteAliasError(f"unknown route alias {alias!r}")
    return _build_route(alias, raw, require_active=False)


def resolve_route_alias(alias: str, *, hermes_home: str | Path | None = None) -> ResolvedCronRoute:
    """Resolve an alias for execution; disabled and unsafe routes fail closed."""
    route = validate_route_alias_definition(alias, hermes_home=hermes_home)
    if route.status not in {"active", "configuration-verified-no-provider-call"}:
        raise RouteAliasError(f"route alias {alias!r} is disabled or not validated (status={route.status})")
    logger.info(
        "Cron route resolved alias=%s provider=%s model=%s role=%s status=%s",
        route.alias,
        route.provider or "none",
        route.model or "none",
        route.role or "unspecified",
        route.status,
    )
    return route


def validate_route_registry(*, hermes_home: str | Path | None = None) -> list[str]:
    """Validate all declared aliases without activating providers."""
    registry = load_route_registry(hermes_home)
    errors: list[str] = []
    for alias, raw in registry["aliases"].items():
        try:
            _build_route(str(alias), raw, require_active=False)
        except RouteAliasError as exc:
            errors.append(str(exc))
    return errors
