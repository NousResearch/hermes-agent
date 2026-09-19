"""Capability registry and caller-claim normalization.

The registry is the authority for minimum risk, effect classes, required scope,
and obligations.  A caller can ask for *more* restriction, never less.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from .errors import PolicyDenied
from .model import ActionIntent, CapabilityDefinition, RiskLevel, ScopeRef
from .scopes import require_scope_fields


_BUILTIN_CAPABILITIES: tuple[CapabilityDefinition, ...] = (
    CapabilityDefinition(
        "context.read",
        RiskLevel.LOW,
        guest_eligible=True,
        obligations=("scope.filter", "sensitivity.filter", "provenance.render"),
    ),
    CapabilityDefinition(
        "fact.observe",
        RiskLevel.MEDIUM,
        state_write=True,
        obligations=("provenance.require", "scope.filter", "receipt.append"),
    ),
    CapabilityDefinition(
        "continuity.read",
        RiskLevel.LOW,
        guest_eligible=True,
        obligations=("scope.filter",),
    ),
    CapabilityDefinition(
        "continuity.write",
        RiskLevel.MEDIUM,
        state_write=True,
        obligations=("optimistic.version", "receipt.append"),
    ),
    CapabilityDefinition(
        "approval.grant",
        RiskLevel.HIGH,
        state_write=True,
        obligations=("identity.authenticate", "intent.exact", "expiry.require", "receipt.append"),
    ),
    CapabilityDefinition(
        "message.send",
        RiskLevel.MEDIUM,
        external_effect=True,
        network_egress=True,
        obligations=(
            "target.exact",
            "payload.redact",
            "delivery.record",
            "receipt.append",
        ),
    ),
    CapabilityDefinition(
        "message.react",
        RiskLevel.MEDIUM,
        external_effect=True,
        network_egress=True,
        obligations=("target.exact", "receipt.append"),
    ),
    CapabilityDefinition(
        "thread.manage",
        RiskLevel.HIGH,
        external_effect=True,
        network_egress=True,
        state_write=True,
        required_scope_fields=(
            "profile",
            "platform",
            "scope_id",
            "chat_id",
            "thread_id",
            "principal_id",
        ),
        obligations=("target.exact", "receipt.append"),
    ),
    CapabilityDefinition(
        "tool.execute",
        RiskLevel.MEDIUM,
        external_effect=True,
        obligations=("arguments.digest", "dispatch.reverify", "receipt.append"),
    ),
    CapabilityDefinition(
        "process.spawn",
        RiskLevel.HIGH,
        external_effect=True,
        state_write=True,
        obligations=(
            "environment.scrub",
            "credential.minimize",
            "process.track",
            "receipt.append",
        ),
    ),
    CapabilityDefinition(
        "task.delegate",
        RiskLevel.HIGH,
        external_effect=True,
        state_write=True,
        obligations=("scope.capture", "authority.bound", "receipt.append"),
    ),
    CapabilityDefinition(
        "network.egress",
        RiskLevel.HIGH,
        external_effect=True,
        network_egress=True,
        obligations=("destination.allowlist", "payload.redact", "receipt.append"),
    ),
    CapabilityDefinition(
        "file.read",
        RiskLevel.MEDIUM,
        obligations=("path.boundary", "sensitivity.filter"),
    ),
    CapabilityDefinition(
        "file.write",
        RiskLevel.HIGH,
        external_effect=True,
        state_write=True,
        obligations=("path.boundary", "atomic.write", "receipt.append"),
    ),
    CapabilityDefinition(
        "state.write",
        RiskLevel.HIGH,
        external_effect=True,
        state_write=True,
        obligations=("optimistic.version", "receipt.append"),
    ),
    CapabilityDefinition(
        "secret.read",
        RiskLevel.CRITICAL,
        obligations=("secret.reference", "egress.deny", "receipt.append"),
    ),
    CapabilityDefinition(
        "connector.invoke",
        RiskLevel.HIGH,
        external_effect=True,
        network_egress=True,
        obligations=("connector.consent", "arguments.digest", "receipt.append"),
    ),
    CapabilityDefinition(
        "admin.policy.read",
        RiskLevel.MEDIUM,
        obligations=("admin.authenticate", "sensitivity.filter"),
    ),
    CapabilityDefinition(
        "admin.policy.write",
        RiskLevel.CRITICAL,
        external_effect=True,
        state_write=True,
        obligations=(
            "admin.authenticate",
            "approval.exact",
            "optimistic.version",
            "receipt.append",
        ),
    ),
    CapabilityDefinition(
        "admin.identity.write",
        RiskLevel.CRITICAL,
        external_effect=True,
        state_write=True,
        obligations=("admin.authenticate", "approval.exact", "receipt.append"),
    ),
)


class CapabilityRegistry:
    """Mutable-at-construction, read-mostly capability authority."""

    def __init__(
        self,
        definitions: Iterable[CapabilityDefinition] = (),
        *,
        include_builtins: bool = True,
    ) -> None:
        self._definitions: dict[str, CapabilityDefinition] = {}
        if include_builtins:
            for definition in _BUILTIN_CAPABILITIES:
                self.register(definition)
        for definition in definitions:
            self.register(definition)

    def register(
        self,
        definition: CapabilityDefinition,
        *,
        replace_existing: bool = False,
    ) -> None:
        existing = self._definitions.get(definition.name)
        if existing is not None and not replace_existing:
            raise ValueError(f"capability already registered: {definition.name}")
        self._definitions[definition.name] = definition

    def get(self, name: str) -> CapabilityDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise PolicyDenied(f"unknown capability: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))

    def definitions(self) -> tuple[CapabilityDefinition, ...]:
        return tuple(self._definitions[name] for name in self.names())

    def normalize_intent(
        self,
        *,
        capability: str,
        action: str,
        resource: str,
        arguments_digest: str,
        scope: ScopeRef,
        risk: RiskLevel | str | int = RiskLevel.LOW,
        external_effect: bool = False,
        network_egress: bool = False,
        state_write: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[ActionIntent, CapabilityDefinition]:
        """Normalize a caller-authored intent against authoritative metadata."""
        definition = self.get(capability)
        require_scope_fields(scope, definition.required_scope_fields)
        caller_risk = RiskLevel.coerce(risk)
        intent = ActionIntent(
            capability=definition.name,
            action=action,
            resource=resource,
            arguments_digest=arguments_digest,
            scope=scope,
            risk=max(caller_risk, definition.risk),
            external_effect=bool(external_effect or definition.external_effect),
            network_egress=bool(network_egress or definition.network_egress),
            state_write=bool(state_write or definition.state_write),
            metadata=dict(metadata or {}),
        )
        return intent, definition

    def refined_definition(
        self,
        capability: str,
        *,
        risk: RiskLevel | str | int | None = None,
        obligations: Iterable[str] = (),
    ) -> CapabilityDefinition:
        """Return a stricter view without mutating registry authority."""
        current = self.get(capability)
        merged_risk = current.risk if risk is None else max(current.risk, RiskLevel.coerce(risk))
        return replace(
            current,
            risk=merged_risk,
            obligations=tuple(sorted(set(current.obligations) | set(obligations))),
        )


BUILTIN_CAPABILITIES = _BUILTIN_CAPABILITIES
