"""Immutable Bot Mode identity, runtime, and policy proof objects.

This module is the shadow-only foundation for the Bot Mode control plane. It
performs no I/O and is not wired into production execution yet. Consumers can
compare a typed policy decision with a legacy gate, but the comparison keeps
the legacy result authoritative until a boundary explicitly migrates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Iterable, Optional, Tuple, Union


BOT_CONTROL_PLANE_CONTRACT_VERSION = "hermes.bot_control_plane.v1"


def _required_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _optional_str(value: object, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    return value.strip() or None


def _non_negative_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _strict_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")
    return value


class BotCapability(str, Enum):
    """Stable restrictive capability classes, independent of tool names."""

    LOCAL_READ = "local.read"
    LOCAL_WRITE = "local.write"
    NETWORK_READ = "network.read"
    NETWORK_WRITE = "network.write"
    EXTERNAL_MESSAGE = "external.message"
    PEER_MESSAGE = "peer.message"
    PROCESS_SPAWN = "process.spawn"
    CREDENTIAL_USE = "credential.use"
    PROFILE_CONFIGURE = "profile.configure"
    DESTRUCTIVE = "destructive"


class BotPolicyVerdict(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    APPROVAL_REQUIRED = "approval_required"


class BotPolicyReason(str, Enum):
    CAPABILITY_GRANTED = "capability_granted"
    CAPABILITY_MISSING = "capability_missing"
    PROFILE_MISMATCH = "profile_mismatch"
    PROFILE_REVISION_MISMATCH = "profile_revision_mismatch"
    GRANT_MISMATCH = "grant_mismatch"
    REVOCATION_EPOCH_MISMATCH = "revocation_epoch_mismatch"
    RUNTIME_SNAPSHOT_MISMATCH = "runtime_snapshot_mismatch"


class LegacyMessageAgentReason(str, Enum):
    """Reason codes for the current #91802 injection/dispatch gates."""

    PROTOCOL_DISABLED = "protocol_disabled"
    SCHEMA_ALREADY_PRESENT = "schema_already_present"
    NOT_CANONICAL_BOT_CHAT = "not_canonical_bot_chat"
    UNMANAGED_INSTALL = "unmanaged_install"
    LEGACY_GATE_ALLOW = "legacy_gate_allow"


@dataclass(frozen=True)
class BotAddress:
    """One exact source-qualified Bot Mode profile instance."""

    install_id: str
    gateway_instance_id: str
    connection_id: str
    profile_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "install_id",
            "gateway_instance_id",
            "connection_id",
            "profile_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_str(getattr(self, field_name), field_name),
            )

    @property
    def identity_tuple(self) -> Tuple[str, str, str, str]:
        return (
            self.install_id,
            self.gateway_instance_id,
            self.connection_id,
            self.profile_id,
        )


CapabilityInput = Union[BotCapability, str]


def _capability(value: CapabilityInput) -> BotCapability:
    if isinstance(value, BotCapability):
        return value
    if isinstance(value, str):
        try:
            return BotCapability(value.strip())
        except ValueError as exc:
            raise ValueError(f"unknown Bot Mode capability: {value!r}") from exc
    raise TypeError("capability must be a BotCapability or string")


@dataclass(frozen=True)
class RuntimeCapabilitySnapshot:
    """Effective runtime and revocable grant for one profile generation."""

    grant_id: str
    profile_id: str
    profile_config_revision: str
    runtime_snapshot_id: str
    effective_provider: str
    effective_model: str
    api_mode: str
    revocation_epoch: int
    capabilities: FrozenSet[BotCapability]
    configured_provider: Optional[str] = None
    requested_provider: Optional[str] = None
    reasoning_effort: Optional[str] = None
    service_tier: Optional[str] = None
    credential_source_id: Optional[str] = None
    fallback_reason: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in (
            "grant_id",
            "profile_id",
            "profile_config_revision",
            "runtime_snapshot_id",
            "effective_provider",
            "effective_model",
            "api_mode",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_str(getattr(self, field_name), field_name),
            )
        for field_name in (
            "configured_provider",
            "requested_provider",
            "reasoning_effort",
            "service_tier",
            "credential_source_id",
            "fallback_reason",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_str(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "revocation_epoch",
            _non_negative_int(self.revocation_epoch, "revocation_epoch"),
        )
        object.__setattr__(
            self,
            "capabilities",
            frozenset(_capability(item) for item in self.capabilities),
        )

    @classmethod
    def build(
        cls,
        *,
        grant_id: str,
        profile_id: str,
        profile_config_revision: str,
        runtime_snapshot_id: str,
        effective_provider: str,
        effective_model: str,
        api_mode: str,
        revocation_epoch: int = 0,
        capabilities: Iterable[CapabilityInput] = (),
        configured_provider: Optional[str] = None,
        requested_provider: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        service_tier: Optional[str] = None,
        credential_source_id: Optional[str] = None,
        fallback_reason: Optional[str] = None,
    ) -> "RuntimeCapabilitySnapshot":
        return cls(
            grant_id=grant_id,
            profile_id=profile_id,
            profile_config_revision=profile_config_revision,
            runtime_snapshot_id=runtime_snapshot_id,
            effective_provider=effective_provider,
            effective_model=effective_model,
            api_mode=api_mode,
            revocation_epoch=revocation_epoch,
            capabilities=frozenset(capabilities),
            configured_provider=configured_provider,
            requested_provider=requested_provider,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            credential_source_id=credential_source_id,
            fallback_reason=fallback_reason,
        )

    def allows(self, capability: CapabilityInput) -> bool:
        return _capability(capability) in self.capabilities


@dataclass(frozen=True)
class BotExecutionContext:
    """Immutable proof object carried by one authenticated Bot Mode turn."""

    address: BotAddress
    profile_config_revision: str
    session_id: str
    session_key: str
    turn_id: str
    task_id: str
    authenticated_principal: str
    source_platform: str
    runtime_snapshot_id: str
    capability_grant_id: str
    cancellation_scope_id: str
    budget_id: str
    trace_id: str
    revocation_epoch: int = 0
    hop_count: int = 0
    source_chat_id: Optional[str] = None
    source_thread_id: Optional[str] = None
    source_user_id: Optional[str] = None
    inbound_event_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    contract_version: str = BOT_CONTROL_PLANE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.address, BotAddress):
            raise TypeError("address must be a BotAddress")
        for field_name in (
            "profile_config_revision",
            "session_id",
            "session_key",
            "turn_id",
            "task_id",
            "authenticated_principal",
            "source_platform",
            "runtime_snapshot_id",
            "capability_grant_id",
            "cancellation_scope_id",
            "budget_id",
            "trace_id",
            "contract_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_str(getattr(self, field_name), field_name),
            )
        for field_name in (
            "source_chat_id",
            "source_thread_id",
            "source_user_id",
            "inbound_event_id",
            "parent_event_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_str(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "revocation_epoch",
            _non_negative_int(self.revocation_epoch, "revocation_epoch"),
        )
        object.__setattr__(
            self,
            "hop_count",
            _non_negative_int(self.hop_count, "hop_count"),
        )


@dataclass(frozen=True)
class BotPolicyDecision:
    """Structured, content-free authorization result for one operation."""

    decision_id: str
    operation: str
    verdict: BotPolicyVerdict
    reason: BotPolicyReason
    required_capability: BotCapability
    constraints: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "decision_id", _required_str(self.decision_id, "decision_id")
        )
        object.__setattr__(self, "operation", _required_str(self.operation, "operation"))
        if not isinstance(self.verdict, BotPolicyVerdict):
            raise TypeError("verdict must be a BotPolicyVerdict")
        if not isinstance(self.reason, BotPolicyReason):
            raise TypeError("reason must be a BotPolicyReason")
        if not isinstance(self.required_capability, BotCapability):
            raise TypeError("required_capability must be a BotCapability")
        normalized = []
        for entry in self.constraints:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError("constraints must contain (key, value) tuples")
            normalized.append(
                (
                    _required_str(entry[0], "constraint key"),
                    _required_str(entry[1], "constraint value"),
                )
            )
        object.__setattr__(self, "constraints", tuple(normalized))

    @property
    def allowed(self) -> bool:
        return self.verdict is BotPolicyVerdict.ALLOW


@dataclass(frozen=True)
class ShadowPolicyComparison:
    """Policy parity evidence whose effective result remains legacy-owned."""

    legacy_allowed: bool
    decision: BotPolicyDecision

    def __post_init__(self) -> None:
        _strict_bool(self.legacy_allowed, "legacy_allowed")
        if not isinstance(self.decision, BotPolicyDecision):
            raise TypeError("decision must be a BotPolicyDecision")

    @property
    def policy_allowed(self) -> bool:
        return self.decision.allowed

    @property
    def matches(self) -> bool:
        return self.legacy_allowed is self.policy_allowed

    @property
    def effective_allowed(self) -> bool:
        """Never alter behavior during the shadow phase."""

        return self.legacy_allowed


@dataclass(frozen=True)
class LegacyMessageAgentState:
    """Inputs used by the current `message_agent` containment checks."""

    protocol_enabled: bool
    schema_present: bool
    canonical_bot_chat: bool
    managed_install: bool

    def __post_init__(self) -> None:
        for field_name in (
            "protocol_enabled",
            "schema_present",
            "canonical_bot_chat",
            "managed_install",
        ):
            _strict_bool(getattr(self, field_name), field_name)


@dataclass(frozen=True)
class LegacyAuthorityDecision:
    operation: str
    allowed: bool
    reason: LegacyMessageAgentReason

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _required_str(self.operation, "operation"))
        _strict_bool(self.allowed, "allowed")
        if not isinstance(self.reason, LegacyMessageAgentReason):
            raise TypeError("reason must be a LegacyMessageAgentReason")


def evaluate_capability(
    *,
    context: BotExecutionContext,
    snapshot: RuntimeCapabilitySnapshot,
    operation: str,
    required_capability: CapabilityInput,
    decision_id: str,
) -> BotPolicyDecision:
    """Fail closed unless identity and every bound generation match exactly."""

    capability = _capability(required_capability)
    operation = _required_str(operation, "operation")
    decision_id = _required_str(decision_id, "decision_id")
    checks = (
        (
            context.address.profile_id != snapshot.profile_id,
            BotPolicyReason.PROFILE_MISMATCH,
        ),
        (
            context.profile_config_revision != snapshot.profile_config_revision,
            BotPolicyReason.PROFILE_REVISION_MISMATCH,
        ),
        (
            context.capability_grant_id != snapshot.grant_id,
            BotPolicyReason.GRANT_MISMATCH,
        ),
        (
            context.revocation_epoch != snapshot.revocation_epoch,
            BotPolicyReason.REVOCATION_EPOCH_MISMATCH,
        ),
        (
            context.runtime_snapshot_id != snapshot.runtime_snapshot_id,
            BotPolicyReason.RUNTIME_SNAPSHOT_MISMATCH,
        ),
    )
    for failed, reason in checks:
        if failed:
            return BotPolicyDecision(
                decision_id=decision_id,
                operation=operation,
                verdict=BotPolicyVerdict.DENY,
                reason=reason,
                required_capability=capability,
            )
    if not snapshot.allows(capability):
        return BotPolicyDecision(
            decision_id=decision_id,
            operation=operation,
            verdict=BotPolicyVerdict.DENY,
            reason=BotPolicyReason.CAPABILITY_MISSING,
            required_capability=capability,
        )
    return BotPolicyDecision(
        decision_id=decision_id,
        operation=operation,
        verdict=BotPolicyVerdict.ALLOW,
        reason=BotPolicyReason.CAPABILITY_GRANTED,
        required_capability=capability,
    )


def compare_legacy_authority(
    *, legacy_allowed: bool, decision: BotPolicyDecision
) -> ShadowPolicyComparison:
    return ShadowPolicyComparison(
        legacy_allowed=legacy_allowed,
        decision=decision,
    )


def legacy_message_agent_injection_decision(
    state: LegacyMessageAgentState,
) -> LegacyAuthorityDecision:
    """Map `ensure_message_agent_tool()` in its current source order.

    The schema-present early return intentionally precedes title/managed-install
    checks because that is current behavior. The mapping records it; it does
    not endorse it or make it authoritative anywhere new.
    """

    if not state.protocol_enabled:
        return LegacyAuthorityDecision(
            "message_agent.inject",
            False,
            LegacyMessageAgentReason.PROTOCOL_DISABLED,
        )
    if state.schema_present:
        return LegacyAuthorityDecision(
            "message_agent.inject",
            True,
            LegacyMessageAgentReason.SCHEMA_ALREADY_PRESENT,
        )
    if not state.canonical_bot_chat:
        return LegacyAuthorityDecision(
            "message_agent.inject",
            False,
            LegacyMessageAgentReason.NOT_CANONICAL_BOT_CHAT,
        )
    if not state.managed_install:
        return LegacyAuthorityDecision(
            "message_agent.inject",
            False,
            LegacyMessageAgentReason.UNMANAGED_INSTALL,
        )
    return LegacyAuthorityDecision(
        "message_agent.inject",
        True,
        LegacyMessageAgentReason.LEGACY_GATE_ALLOW,
    )


def legacy_message_agent_dispatch_decision(
    state: LegacyMessageAgentState,
) -> LegacyAuthorityDecision:
    """Map `message_agent_tool()` dispatch containment exactly.

    Dispatch currently rechecks title and managed-install status, but not the
    protocol toggle or schema presence. This asymmetry is preserved as evidence
    until a later migration PR selects one authoritative contract.
    """

    if not state.canonical_bot_chat:
        return LegacyAuthorityDecision(
            "message_agent.dispatch",
            False,
            LegacyMessageAgentReason.NOT_CANONICAL_BOT_CHAT,
        )
    if not state.managed_install:
        return LegacyAuthorityDecision(
            "message_agent.dispatch",
            False,
            LegacyMessageAgentReason.UNMANAGED_INSTALL,
        )
    return LegacyAuthorityDecision(
        "message_agent.dispatch",
        True,
        LegacyMessageAgentReason.LEGACY_GATE_ALLOW,
    )


__all__ = [
    "BOT_CONTROL_PLANE_CONTRACT_VERSION",
    "BotAddress",
    "BotCapability",
    "BotExecutionContext",
    "BotPolicyDecision",
    "BotPolicyReason",
    "BotPolicyVerdict",
    "LegacyAuthorityDecision",
    "LegacyMessageAgentReason",
    "LegacyMessageAgentState",
    "RuntimeCapabilitySnapshot",
    "ShadowPolicyComparison",
    "compare_legacy_authority",
    "evaluate_capability",
    "legacy_message_agent_dispatch_decision",
    "legacy_message_agent_injection_decision",
]
