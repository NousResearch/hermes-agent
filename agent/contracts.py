"""Hermes contract framework.

Provides:

* :class:`ContractIdentity` / :class:`ComponentIdentity` — uniform identity
  for versionable contracts and components.
* :class:`ContractLifecycle` — closed enum describing publication state.
* :class:`ContractDefinition` — declarative description of a single contract
  version. Does NOT live inside the registry; the registry only stores
  pre-built definitions.
* :class:`ContractCompatibility` / :class:`ProducerCompatibility` — declared
  compatibility windows, including optional allowed producers.
* :class:`ComponentContract` — what a versionable component declares it can
  consume.
* :class:`ContractValidationResult` — closed set of statuses + warnings.
* :class:`ContractRegistry` / :class:`InMemoryContractRegistry` — storage.
  Registry never validates and never constructs definitions.
* :class:`ContractValidator` — pure logic; uses a registry as data source.

Design rules enforced by this module:

* Registry = state. Validator = logic.
* Validator never registers contracts; registry never validates them.
* No migration, normalization, or external side effects.
* No string parsing to infer compatibility.
* Identities are frozen dataclasses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

__all__ = [
    "ContractIdentity",
    "ComponentIdentity",
    "ContractLifecycle",
    "ContractDefinition",
    "ProducerCompatibility",
    "ContractCompatibility",
    "ComponentContract",
    "ContractValidationStatus",
    "ContractValidationWarning",
    "ContractValidationResult",
    "ContractRegistry",
    "InMemoryContractRegistry",
    "ContractValidator",
    "IdentifiableContract",
    "IdentifiableComponent",
]


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractIdentity:
    """Uniform identity of a versioned contract.

    * ``contract_name`` identifies the contract type.
    * ``contract_version`` identifies semantic evolution of the contract.
    * ``schema_version`` identifies the serialization format.
    """

    contract_name: str
    contract_version: int
    schema_version: int = 1


@dataclass(frozen=True)
class ComponentIdentity:
    """Uniform identity of a versionable component.

    * ``component_name`` identifies the component.
    * ``component_version`` identifies semantic evolution of the component.
    * ``schema_version`` identifies the serialization format.
    """

    component_name: str
    component_version: int
    schema_version: int = 1


class IdentifiableContract(Protocol):
    """Anything that exposes a ``ContractIdentity``."""

    def contract_identity(self) -> ContractIdentity: ...


class IdentifiableComponent(Protocol):
    """Anything that exposes a ``ComponentIdentity``."""

    def component_identity(self) -> ComponentIdentity: ...


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class ContractLifecycle(str, Enum):
    """Publication lifecycle of a contract definition."""

    EXPERIMENTAL = "experimental"
    STABLE = "stable"
    DEPRECATED = "deprecated"
    REMOVED = "removed"


# ---------------------------------------------------------------------------
# Definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractDefinition:
    """Declarative description of a single contract version.

    The definition can exist independently of any registry. The registry only
    stores pre-built definitions; it does not construct them, mutate them,
    assign ``contract_name``/``contract_version``/``lifecycle``/``breaking_change``,
    or interpret them.
    """

    identity: ContractIdentity
    lifecycle: ContractLifecycle
    superseded_by: ContractIdentity | None = None
    description: str = ""
    owner_component: str | None = None
    breaking_change: bool = False
    schema_version: int = 1


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProducerCompatibility:
    """Producer (component) accepted by a compatibility declaration."""

    component_name: str
    min_component_version: int
    max_component_version: int
    schema_version: int = 1


@dataclass(frozen=True)
class ContractCompatibility:
    """Declared compatibility window for a contract."""

    contract_name: str
    min_contract_version: int
    max_contract_version: int
    allow_experimental: bool = False
    allow_deprecated: bool = True
    accepted_producers: tuple[ProducerCompatibility, ...] = ()
    schema_version: int = 1


@dataclass(frozen=True)
class ComponentContract:
    """What a versionable component declares it can consume."""

    identity: ComponentIdentity
    supported_contracts: Mapping[str, ContractCompatibility]
    schema_version: int = 1


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


class ContractValidationStatus(str, Enum):
    """Closed set of validation statuses."""

    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    BLOCKED_UNKNOWN_CONTRACT = "BLOCKED_UNKNOWN_CONTRACT"
    BLOCKED_UNSUPPORTED_CONTRACT = "BLOCKED_UNSUPPORTED_CONTRACT"
    BLOCKED_INVALID_CONTRACT_VERSION_TYPE = "BLOCKED_INVALID_CONTRACT_VERSION_TYPE"
    BLOCKED_CONTRACT_VERSION_TOO_OLD = "BLOCKED_CONTRACT_VERSION_TOO_OLD"
    BLOCKED_CONTRACT_VERSION_TOO_NEW = "BLOCKED_CONTRACT_VERSION_TOO_NEW"
    BLOCKED_REMOVED_CONTRACT = "BLOCKED_REMOVED_CONTRACT"
    BLOCKED_EXPERIMENTAL_CONTRACT_NOT_ALLOWED = "BLOCKED_EXPERIMENTAL_CONTRACT_NOT_ALLOWED"
    BLOCKED_DEPRECATED_CONTRACT_NOT_ALLOWED = "BLOCKED_DEPRECATED_CONTRACT_NOT_ALLOWED"
    BLOCKED_UNKNOWN_COMPONENT = "BLOCKED_UNKNOWN_COMPONENT"
    BLOCKED_UNSUPPORTED_PRODUCER = "BLOCKED_UNSUPPORTED_PRODUCER"
    BLOCKED_PRODUCER_VERSION_TOO_OLD = "BLOCKED_PRODUCER_VERSION_TOO_OLD"
    BLOCKED_PRODUCER_VERSION_TOO_NEW = "BLOCKED_PRODUCER_VERSION_TOO_NEW"
    BLOCKED_CONTRACT_BREAKING_CHANGE_UNSUPPORTED = "BLOCKED_CONTRACT_BREAKING_CHANGE_UNSUPPORTED"


class ContractValidationWarning(str, Enum):
    """Closed set of non-fatal warnings."""

    WARNING_CONTRACT_DEPRECATED = "WARNING_CONTRACT_DEPRECATED"
    WARNING_CONTRACT_EXPERIMENTAL = "WARNING_CONTRACT_EXPERIMENTAL"
    WARNING_CONTRACT_SUPERSEDED = "WARNING_CONTRACT_SUPERSEDED"
    WARNING_CONTRACT_BREAKING_CHANGE = "WARNING_CONTRACT_BREAKING_CHANGE"
    WARNING_PRODUCER_DEPRECATED = "WARNING_PRODUCER_DEPRECATED"
    WARNING_PRODUCER_EXPERIMENTAL = "WARNING_PRODUCER_EXPERIMENTAL"


_PASS_STATUSES = frozenset(
    {
        ContractValidationStatus.PASS,
        ContractValidationStatus.PASS_WITH_WARNINGS,
    }
)


@dataclass(frozen=True)
class ContractValidationResult:
    ok: bool
    status: ContractValidationStatus
    reason: str | None = None
    warnings: tuple[ContractValidationWarning, ...] = ()
    component_identity: ComponentIdentity | None = None
    contract_identity: ContractIdentity | None = None
    producer_identity: ComponentIdentity | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.ok and self.status not in _PASS_STATUSES:
            raise ValueError(
                f"ok=True requires status in {sorted(s.value for s in _PASS_STATUSES)}"
            )
        if not self.ok and self.warnings:
            raise ValueError("ok=False cannot carry warnings")
        if self.ok is False and self.status in _PASS_STATUSES:
            raise ValueError("ok=False requires a non-PASS status")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ContractRegistry:
    """Storage-only interface. Never validates compatibility."""


class InMemoryContractRegistry(ContractRegistry):
    """In-memory implementation of :class:`ContractRegistry` for V1."""

    def __init__(self) -> None:
        self._contracts: dict[tuple[str, int], ContractDefinition] = {}
        self._components: dict[str, ComponentContract] = {}

    def register_contract(self, definition: ContractDefinition) -> None:
        if not isinstance(definition, ContractDefinition):
            raise TypeError("register_contract expects a ContractDefinition")
        key = (definition.identity.contract_name, definition.identity.contract_version)
        if key in self._contracts:
            raise ValueError(
                f"duplicate contract definition: {definition.identity.contract_name} "
                f"v{definition.identity.contract_version}"
            )
        self._contracts[key] = definition

    def register_component(self, component: ComponentContract) -> None:
        if not isinstance(component, ComponentContract):
            raise TypeError("register_component expects a ComponentContract")
        if component.identity.component_name in self._components:
            raise ValueError(
                f"duplicate component registration: {component.identity.component_name}"
            )
        self._components[component.identity.component_name] = component

    def get_contract(
        self,
        contract_name: str,
        contract_version: int,
    ) -> ContractDefinition:
        try:
            return self._contracts[(contract_name, contract_version)]
        except KeyError as exc:
            raise KeyError(
                f"unknown contract: {contract_name} v{contract_version}"
            ) from exc

    def get_component(
        self,
        component_name: str,
        component_version: int | None = None,
    ) -> ComponentContract:
        try:
            component = self._components[component_name]
        except KeyError as exc:
            raise KeyError(f"unknown component: {component_name}") from exc
        if (
            component_version is not None
            and component.identity.component_version != component_version
        ):
            raise KeyError(
                f"component {component_name} v{component_version} not registered"
            )
        return component

    def describe(self, contract_name: str) -> tuple[ContractDefinition, ...]:
        versions = sorted(
            v for (n, v) in self._contracts.keys() if n == contract_name
        )
        return tuple(self._contracts[(contract_name, v)] for v in versions)

    def list_versions(self, contract_name: str) -> tuple[int, ...]:
        return tuple(
            sorted(v for (n, v) in self._contracts.keys() if n == contract_name)
        )

    # Test/inspection helpers -------------------------------------------------

    def _has_contract(self, contract_name: str, contract_version: int) -> bool:
        return (contract_name, contract_version) in self._contracts

    def _has_component(self, component_name: str) -> bool:
        return component_name in self._components


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def _coerce_contract_identity(
    identity: ContractIdentity | IdentifiableContract,
) -> ContractIdentity:
    if isinstance(identity, ContractIdentity):
        return identity
    return identity.contract_identity()


def _coerce_component_identity(
    identity: ComponentIdentity | IdentifiableComponent,
) -> ComponentIdentity:
    if isinstance(identity, ComponentIdentity):
        return identity
    return identity.component_identity()


class ContractValidator:
    """Pure-logic validator. Uses a registry as data source."""

    def __init__(self, registry: ContractRegistry) -> None:
        if not isinstance(registry, ContractRegistry):
            raise TypeError("ContractValidator requires a ContractRegistry")
        self._registry = registry

    @property
    def registry(self) -> ContractRegistry:
        return self._registry

    def validate(
        self,
        component_identity: ComponentIdentity | IdentifiableComponent,
        contract_identity: ContractIdentity | IdentifiableContract,
        producer_identity: ComponentIdentity | IdentifiableComponent | None = None,
    ) -> ContractValidationResult:
        comp_id = _coerce_component_identity(component_identity)
        contract_id = _coerce_contract_identity(contract_identity)
        producer_id = (
            _coerce_component_identity(producer_identity)
            if producer_identity is not None
            else None
        )

        # Component must be registered.
        if not _registry_has_component(self._registry, comp_id.component_name):
            return ContractValidationResult(
                ok=False,
                status=ContractValidationStatus.BLOCKED_UNKNOWN_COMPONENT,
                reason=f"unknown component: {comp_id.component_name}",
                component_identity=comp_id,
                contract_identity=contract_id,
                producer_identity=producer_id,
            )

        component = _registry_get_component(self._registry, comp_id.component_name)

        compatibility = component.supported_contracts.get(contract_id.contract_name)
        if compatibility is None:
            return ContractValidationResult(
                ok=False,
                status=ContractValidationStatus.BLOCKED_UNSUPPORTED_CONTRACT,
                reason=(
                    f"component {comp_id.component_name} does not declare support "
                    f"for contract {contract_id.contract_name}"
                ),
                component_identity=comp_id,
                contract_identity=contract_id,
                producer_identity=producer_id,
            )

        # Version type check.
        if not isinstance(contract_id.contract_version, int) or isinstance(
            contract_id.contract_version, bool
        ):
            return ContractValidationResult(
                ok=False,
                status=ContractValidationStatus.BLOCKED_INVALID_CONTRACT_VERSION_TYPE,
                reason="contract_version must be int",
                component_identity=comp_id,
                contract_identity=contract_id,
                producer_identity=producer_id,
            )

        # Range check.
        if contract_id.contract_version < compatibility.min_contract_version:
            return ContractValidationResult(
                ok=False,
                status=ContractValidationStatus.BLOCKED_CONTRACT_VERSION_TOO_OLD,
                reason=(
                    f"contract_version {contract_id.contract_version} < "
                    f"min {compatibility.min_contract_version}"
                ),
                component_identity=comp_id,
                contract_identity=contract_id,
                producer_identity=producer_id,
            )
        if contract_id.contract_version > compatibility.max_contract_version:
            return ContractValidationResult(
                ok=False,
                status=ContractValidationStatus.BLOCKED_CONTRACT_VERSION_TOO_NEW,
                reason=(
                    f"contract_version {contract_id.contract_version} > "
                    f"max {compatibility.max_contract_version}"
                ),
                component_identity=comp_id,
                contract_identity=contract_id,
                producer_identity=producer_id,
            )

        # Definition + lifecycle checks (if definition exists).
        warnings: list[ContractValidationWarning] = []
        definition: ContractDefinition | None = None
        if _registry_has_contract(
            self._registry,
            contract_id.contract_name,
            contract_id.contract_version,
        ):
            definition = _registry_get_contract(
                self._registry, contract_id.contract_name, contract_id.contract_version
            )

        if definition is not None:
            if definition.lifecycle is ContractLifecycle.REMOVED:
                return ContractValidationResult(
                    ok=False,
                    status=ContractValidationStatus.BLOCKED_REMOVED_CONTRACT,
                    reason="contract has been removed",
                    component_identity=comp_id,
                    contract_identity=contract_id,
                    producer_identity=producer_id,
                )
            if (
                definition.lifecycle is ContractLifecycle.DEPRECATED
                and not compatibility.allow_deprecated
            ):
                return ContractValidationResult(
                    ok=False,
                    status=ContractValidationStatus.BLOCKED_DEPRECATED_CONTRACT_NOT_ALLOWED,
                    reason="deprecated contract not allowed by consumer",
                    component_identity=comp_id,
                    contract_identity=contract_id,
                    producer_identity=producer_id,
                )
            if (
                definition.lifecycle is ContractLifecycle.EXPERIMENTAL
                and not compatibility.allow_experimental
            ):
                return ContractValidationResult(
                    ok=False,
                    status=ContractValidationStatus.BLOCKED_EXPERIMENTAL_CONTRACT_NOT_ALLOWED,
                    reason="experimental contract not allowed by consumer",
                    component_identity=comp_id,
                    contract_identity=contract_id,
                    producer_identity=producer_id,
                )
            if definition.breaking_change:
                # Breaking change metadata: emit a warning when allowed.
                warnings.append(ContractValidationWarning.WARNING_CONTRACT_BREAKING_CHANGE)
            if definition.superseded_by is not None:
                warnings.append(ContractValidationWarning.WARNING_CONTRACT_SUPERSEDED)

        # Producer compatibility.
        if compatibility.accepted_producers:
            if producer_id is None:
                return ContractValidationResult(
                    ok=False,
                    status=ContractValidationStatus.BLOCKED_UNSUPPORTED_PRODUCER,
                    reason="accepted_producers declared but no producer_identity provided",
                    component_identity=comp_id,
                    contract_identity=contract_id,
                    producer_identity=None,
                )

            matching = [
                p
                for p in compatibility.accepted_producers
                if p.component_name == producer_id.component_name
            ]
            if not matching:
                return ContractValidationResult(
                    ok=False,
                    status=ContractValidationStatus.BLOCKED_UNSUPPORTED_PRODUCER,
                    reason=(
                        f"producer {producer_id.component_name} is not in "
                        f"accepted_producers for {contract_id.contract_name}"
                    ),
                    component_identity=comp_id,
                    contract_identity=contract_id,
                    producer_identity=producer_id,
                )
            accepted = matching[0]
            if not isinstance(producer_id.component_version, int) or isinstance(
                producer_id.component_version, bool
            ):
                return ContractValidationResult(
                    ok=False,
                    status=ContractValidationStatus.BLOCKED_PRODUCER_VERSION_TOO_OLD,
                    reason="producer component_version must be int",
                    component_identity=comp_id,
                    contract_identity=contract_id,
                    producer_identity=producer_id,
                )
            if producer_id.component_version < accepted.min_component_version:
                return ContractValidationResult(
                    ok=False,
                    status=ContractValidationStatus.BLOCKED_PRODUCER_VERSION_TOO_OLD,
                    reason=(
                        f"producer component_version {producer_id.component_version} "
                        f"< min {accepted.min_component_version}"
                    ),
                    component_identity=comp_id,
                    contract_identity=contract_id,
                    producer_identity=producer_id,
                )
            if producer_id.component_version > accepted.max_component_version:
                return ContractValidationResult(
                    ok=False,
                    status=ContractValidationStatus.BLOCKED_PRODUCER_VERSION_TOO_NEW,
                    reason=(
                        f"producer component_version {producer_id.component_version} "
                        f"> max {accepted.max_component_version}"
                    ),
                    component_identity=comp_id,
                    contract_identity=contract_id,
                    producer_identity=producer_id,
                )

            # Producer lifecycle warnings (if registered).
            try:
                producer_def = _registry_get_contract(
                    self._registry,
                    _component_to_contract_name(producer_id.component_name),
                    producer_id.component_version,
                )
                if (
                    producer_def.lifecycle is ContractLifecycle.DEPRECATED
                    and compatibility.allow_deprecated
                ):
                    warnings.append(ContractValidationWarning.WARNING_PRODUCER_DEPRECATED)
                if (
                    producer_def.lifecycle is ContractLifecycle.EXPERIMENTAL
                    and compatibility.allow_experimental
                ):
                    warnings.append(
                        ContractValidationWarning.WARNING_PRODUCER_EXPERIMENTAL
                    )
            except KeyError:
                pass

        # Deprecated/experimental contract warnings (only after gate).
        if definition is not None:
            if (
                definition.lifecycle is ContractLifecycle.DEPRECATED
                and compatibility.allow_deprecated
            ):
                warnings.append(ContractValidationWarning.WARNING_CONTRACT_DEPRECATED)
            if (
                definition.lifecycle is ContractLifecycle.EXPERIMENTAL
                and compatibility.allow_experimental
            ):
                warnings.append(ContractValidationWarning.WARNING_CONTRACT_EXPERIMENTAL)

        return ContractValidationResult(
            ok=True,
            status=(
                ContractValidationStatus.PASS_WITH_WARNINGS
                if warnings
                else ContractValidationStatus.PASS
            ),
            warnings=tuple(warnings),
            component_identity=comp_id,
            contract_identity=contract_id,
            producer_identity=producer_id,
        )


# ---------------------------------------------------------------------------
# Registry adapter helpers (used by validator; insulated for tests)
# ---------------------------------------------------------------------------


def _registry_has_component(registry: ContractRegistry, component_name: str) -> bool:
    has = getattr(registry, "_has_component", None)
    if callable(has):
        return bool(has(component_name))
    try:
        registry.get_component(component_name)  # type: ignore[attr-defined]
    except KeyError:
        return False
    return True


def _registry_get_component(
    registry: ContractRegistry, component_name: str
) -> ComponentContract:
    return registry.get_component(component_name)  # type: ignore[attr-defined]


def _registry_has_contract(
    registry: ContractRegistry, contract_name: str, contract_version: int
) -> bool:
    has = getattr(registry, "_has_contract", None)
    if callable(has):
        return bool(has(contract_name, contract_version))
    try:
        registry.get_contract(contract_name, contract_version)  # type: ignore[attr-defined]
    except KeyError:
        return False
    return True


def _registry_get_contract(
    registry: ContractRegistry, contract_name: str, contract_version: int
) -> ContractDefinition:
    return registry.get_contract(contract_name, contract_version)  # type: ignore[attr-defined]


def _component_to_contract_name(component_name: str) -> str:
    """Convention: a component registers a contract under its own name."""
    return component_name