"""ProviderHealthMonitor: extracts provider health probing from LLMExecutor.

The monitor is a standalone component with its own :class:`ComponentContract`
consumed by the framework's :class:`ContractValidator`. It owns the
mapping between raw :class:`ProviderHealth` objects and the closed
:class:`ProviderHealthStatus` enum.

The monitor NEVER:

* Calls providers / HTTP / SDK.
* Reads API keys or prints secrets.
* Touches GBrain / Kanban / workers / gateway.
* Auto-falls back from one provider to another.

LLMExecutor consumes :class:`ProviderHealthSnapshot` instances produced by
this monitor; it does not contain its own health probe logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TYPE_CHECKING

from agent.contracts import (
    ComponentContract,
    ComponentIdentity,
    ContractCompatibility,
    ContractIdentity,
    IdentifiableComponent,
    InMemoryContractRegistry,
)

if TYPE_CHECKING:
    from agent.provider_adapter import ProviderHealth


class ProviderHealthStatus(str, Enum):
    """Closed set of provider health states."""

    AVAILABLE = "available"
    QUOTA_BLOCKED = "quota_blocked"
    RATE_LIMITED = "rate_limited"
    CONTEXT_LIMIT = "context_limit"
    AUTH_BLOCKED = "auth_blocked"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# ProviderHealthSnapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderHealthSnapshot:
    """Single point-in-time health view of a provider.

    Implements the framework ``ContractIdentity`` shape
    (``contract_name`` / ``contract_version`` / ``schema_version``) and
    serializes all three fields in :meth:`to_dict`.
    """

    provider: str
    status: ProviderHealthStatus
    is_available: bool
    consecutive_failures: int
    last_error: str | None
    last_checked_at_utc: str
    details: tuple[tuple[str, Any], ...] = ()
    contract_name: str = "provider_health_snapshot"
    contract_version: int = 1
    schema_version: int = 1

    def contract_identity(self) -> ContractIdentity:
        return ContractIdentity(
            contract_name=self.contract_name,
            contract_version=self.contract_version,
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "status": self.status.value,
            "is_available": self.is_available,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "last_checked_at_utc": self.last_checked_at_utc,
            "details": dict(self.details),
            "contract_name": self.contract_name,
            "contract_version": self.contract_version,
            "schema_version": self.schema_version,
        }


# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------


_QUOTA_HINTS = ("quota", "quota_exceeded", "billing", "credit")
_RATE_LIMIT_HINTS = ("rate_limit", "rate_limit_exceeded", "rate limit", "too_many_requests", "429")
_CONTEXT_LIMIT_HINTS = (
    "context_length",
    "context_length_exceeded",
    "context_overflow",
    "max_tokens",
)
_AUTH_HINTS = (
    "auth",
    "auth_error",
    "missing_credentials",
    "unauthorized",
    "forbidden",
    "401",
    "403",
    "invalid_api_key",
)
_TRANSIENT_HINTS = (
    "transient",
    "transient_error",
    "timeout",
    "connection",
    "503",
    "502",
    "500",
    "unavailable",
)


def _map_error_to_status(
    is_available: bool,
    consecutive_failures: int,
    last_error: str | None,
) -> ProviderHealthStatus:
    """Map a raw health view to a :class:`ProviderHealthStatus`.

    The mapping is deterministic and table-driven. Unknown errors map to
    ``UNKNOWN``; the executor decides what to do with that.

    Canonical Phase 4c ``ProviderErrorCode.value`` strings are checked
    first; legacy hint-based matching is kept as a fallback for
    backward compatibility with older adapters.
    """
    if is_available and consecutive_failures <= 0:
        return ProviderHealthStatus.AVAILABLE
    err = (last_error or "").lower()
    # Hint-based matching (covers legacy + substring matches of canonical
    # ProviderErrorCode.value strings).
    for hint in _QUOTA_HINTS:
        if hint in err:
            return ProviderHealthStatus.QUOTA_BLOCKED
    for hint in _RATE_LIMIT_HINTS:
        if hint in err:
            return ProviderHealthStatus.RATE_LIMITED
    for hint in _CONTEXT_LIMIT_HINTS:
        if hint in err:
            return ProviderHealthStatus.CONTEXT_LIMIT
    for hint in _AUTH_HINTS:
        if hint in err:
            return ProviderHealthStatus.AUTH_BLOCKED
    for hint in _TRANSIENT_HINTS:
        if hint in err:
            return ProviderHealthStatus.TRANSIENT
    # If a last_error string was set but matched no hint, do NOT
    # default to TRANSIENT. Classify it as UNKNOWN so the caller
    # can surface the unrecognized code.
    if err:
        return ProviderHealthStatus.UNKNOWN
    # No last_error and not available → TRANSIENT (degraded).
    if not is_available and consecutive_failures >= 1:
        return ProviderHealthStatus.TRANSIENT
    if is_available and consecutive_failures > 0:
        # Available but with some prior failures → degraded but still up.
        return ProviderHealthStatus.AVAILABLE
    return ProviderHealthStatus.UNKNOWN


# ---------------------------------------------------------------------------
# ComponentContract
# ---------------------------------------------------------------------------


def _monitor_component_contract() -> ComponentContract:
    return ComponentContract(
        identity=ComponentIdentity(
            component_name="provider_health_monitor",
            component_version=1,
        ),
        supported_contracts={
            "provider_health_snapshot": ContractCompatibility(
                contract_name="provider_health_snapshot",
                min_contract_version=1,
                max_contract_version=1,
            ),
        },
    )


# ---------------------------------------------------------------------------
# ProviderHealthMonitor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderHealthMonitor(IdentifiableComponent):
    """Resolves a provider name to a :class:`ProviderHealthSnapshot`.

    The monitor is a thin adapter over ``agent.providers.registry`` plus
    the mapping table above. It performs no I/O and never executes the
    provider; it only inspects the registered adapter's
    ``health()`` view.
    """

    component_name: str = "provider_health_monitor"
    component_version: int = 1
    degraded_failure_threshold: int = 3
    _registry: InMemoryContractRegistry = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        registry = InMemoryContractRegistry()
        registry.register_component(_monitor_component_contract())
        object.__setattr__(self, "_registry", registry)

    def component_identity(self) -> ComponentIdentity:
        return ComponentIdentity(
            component_name=self.component_name,
            component_version=self.component_version,
        )

    def component_contract(self) -> ComponentContract:
        return self._registry.get_component(self.component_name)

    def check(self, provider_name: str | None) -> ProviderHealthSnapshot:
        """Return a snapshot of the named provider's health.

        ``provider_name=None`` returns an ``UNKNOWN`` snapshot without
        touching the registry. Unknown provider names return ``UNKNOWN``
        with a descriptive ``last_error``. Never raises into the dispatch
        flow.
        """
        if not provider_name:
            return _unknown_snapshot(
                provider="(none)",
                reason="no_provider",
            )
        try:
            from agent.providers.registry import get_provider
        except Exception as exc:  # pragma: no cover - defensive
            return _unknown_snapshot(
                provider=provider_name,
                reason=f"registry_unavailable:{exc}",
            )
        try:
            adapter = get_provider(provider_name)
        except Exception as exc:
            return _unknown_snapshot(
                provider=provider_name,
                reason=f"unknown_provider:{exc}",
            )
        try:
            health = adapter.health()
        except Exception as exc:
            return _unknown_snapshot(
                provider=provider_name,
                reason=f"health_check_failed:{exc}",
            )
        return _snapshot_from_health(provider_name, health, self.degraded_failure_threshold)


def _snapshot_from_health(
    provider_name: str,
    health: "ProviderHealth",
    degraded_threshold: int,
) -> ProviderHealthSnapshot:
    status = _map_error_to_status(
        is_available=bool(getattr(health, "is_available", False)),
        consecutive_failures=int(getattr(health, "consecutive_failures", 0)),
        last_error=getattr(health, "last_error", None),
    )
    is_available = status is ProviderHealthStatus.AVAILABLE
    return ProviderHealthSnapshot(
        provider=provider_name,
        status=status,
        is_available=is_available,
        consecutive_failures=int(getattr(health, "consecutive_failures", 0)),
        last_error=getattr(health, "last_error", None),
        last_checked_at_utc=str(getattr(health, "last_checked_at_utc", "")),
        details=(
            ("degraded_threshold", degraded_threshold),
        ),
    )


def _unknown_snapshot(provider: str, reason: str) -> ProviderHealthSnapshot:
    return ProviderHealthSnapshot(
        provider=provider,
        status=ProviderHealthStatus.UNKNOWN,
        is_available=False,
        consecutive_failures=0,
        last_error=reason,
        last_checked_at_utc="",
    )