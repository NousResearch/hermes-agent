"""Typed obligation catalog and fail-closed evidence verification."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .errors import ObligationError


class ObligationPhase(str, Enum):
    PRE_EFFECT = "pre_effect"
    POST_EFFECT = "post_effect"
    EITHER = "either"


Validator = Callable[[Any], bool]


def _truthy(value: Any) -> bool:
    return value is True


def _non_empty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _non_negative(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


@dataclass(frozen=True, slots=True)
class ObligationDefinition:
    name: str
    phase: ObligationPhase
    evidence_key: str
    validator: Validator = _truthy
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.evidence_key.strip():
            raise ValueError("obligation name and evidence_key must be non-empty")
        object.__setattr__(self, "phase", ObligationPhase(self.phase))


_BUILTINS: tuple[ObligationDefinition, ...] = (
    ObligationDefinition("scope.filter", ObligationPhase.PRE_EFFECT, "scope_filtered"),
    ObligationDefinition("sensitivity.filter", ObligationPhase.PRE_EFFECT, "sensitivity_filtered"),
    ObligationDefinition("provenance.render", ObligationPhase.PRE_EFFECT, "provenance_rendered"),
    ObligationDefinition("provenance.require", ObligationPhase.PRE_EFFECT, "source_revision", _non_empty),
    ObligationDefinition("receipt.append", ObligationPhase.POST_EFFECT, "receipt_hash", _sha256),
    ObligationDefinition("optimistic.version", ObligationPhase.PRE_EFFECT, "expected_version", _non_negative),
    ObligationDefinition("identity.authenticate", ObligationPhase.PRE_EFFECT, "identity_authenticated"),
    ObligationDefinition("intent.exact", ObligationPhase.PRE_EFFECT, "intent_digest", _sha256),
    ObligationDefinition("expiry.require", ObligationPhase.PRE_EFFECT, "expires_at", _non_empty),
    ObligationDefinition("target.exact", ObligationPhase.PRE_EFFECT, "target_verified"),
    ObligationDefinition("payload.redact", ObligationPhase.PRE_EFFECT, "payload_redacted"),
    ObligationDefinition("delivery.record", ObligationPhase.PRE_EFFECT, "delivery_obligation_id", _non_empty),
    ObligationDefinition("arguments.digest", ObligationPhase.PRE_EFFECT, "arguments_digest", _sha256),
    ObligationDefinition("dispatch.reverify", ObligationPhase.PRE_EFFECT, "lease_reverified"),
    ObligationDefinition("environment.scrub", ObligationPhase.PRE_EFFECT, "environment_scrubbed"),
    ObligationDefinition("credential.minimize", ObligationPhase.PRE_EFFECT, "credentials_minimized"),
    ObligationDefinition("process.track", ObligationPhase.POST_EFFECT, "process_identity", _non_empty),
    ObligationDefinition("scope.capture", ObligationPhase.PRE_EFFECT, "scope_digest", _sha256),
    ObligationDefinition("authority.bound", ObligationPhase.PRE_EFFECT, "lease_id", _non_empty),
    ObligationDefinition("destination.allowlist", ObligationPhase.PRE_EFFECT, "destination_allowed"),
    ObligationDefinition("path.boundary", ObligationPhase.PRE_EFFECT, "path_within_boundary"),
    ObligationDefinition("atomic.write", ObligationPhase.POST_EFFECT, "atomic_write_completed"),
    ObligationDefinition("secret.reference", ObligationPhase.PRE_EFFECT, "secret_reference", _non_empty),
    ObligationDefinition("egress.deny", ObligationPhase.PRE_EFFECT, "egress_denied"),
    ObligationDefinition("connector.consent", ObligationPhase.PRE_EFFECT, "connector_consent_id", _non_empty),
    ObligationDefinition("admin.authenticate", ObligationPhase.PRE_EFFECT, "admin_authenticated"),
    ObligationDefinition("approval.exact", ObligationPhase.PRE_EFFECT, "approval_id", _non_empty),
    ObligationDefinition("budget.settle", ObligationPhase.POST_EFFECT, "budget_settled"),
)


class ObligationRegistry:
    """Registry that rejects unknown or unsatisfied obligations."""

    def __init__(self, definitions: Iterable[ObligationDefinition] = ()) -> None:
        self._definitions: dict[str, ObligationDefinition] = {
            item.name: item for item in _BUILTINS
        }
        for definition in definitions:
            self.register(definition)

    def register(
        self,
        definition: ObligationDefinition,
        *,
        replace_existing: bool = False,
    ) -> None:
        if definition.name in self._definitions and not replace_existing:
            raise ValueError(f"obligation already registered: {definition.name}")
        self._definitions[definition.name] = definition

    def get(self, name: str) -> ObligationDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise ObligationError(f"unknown obligation: {name}") from exc

    def verify(
        self,
        required: Iterable[str],
        evidence: Mapping[str, Any],
        *,
        phase: ObligationPhase,
    ) -> tuple[str, ...]:
        """Verify all obligations applicable to one effect phase."""
        target_phase = ObligationPhase(phase)
        verified: list[str] = []
        failures: list[str] = []
        for name in sorted(set(required)):
            definition = self.get(name)
            if definition.phase not in {target_phase, ObligationPhase.EITHER}:
                continue
            value = evidence.get(definition.evidence_key)
            try:
                valid = definition.validator(value)
            except Exception:
                valid = False
            if not valid:
                failures.append(
                    f"{name} requires valid evidence key {definition.evidence_key!r}"
                )
            else:
                verified.append(name)
        if failures:
            raise ObligationError("; ".join(failures))
        return tuple(verified)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))


BUILTIN_OBLIGATIONS = _BUILTINS
