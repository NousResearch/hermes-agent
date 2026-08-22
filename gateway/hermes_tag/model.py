"""Immutable model types and canonical hashing for Hermes Tag."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{2,127}$")
_MAX_ARGUMENT_BYTES = 1_000_000
_MAX_FACT_VALUE_BYTES = 1_000_000
_MAX_INTENT_METADATA_BYTES = 65_536


def _freeze_json_value(value: Any) -> Any:
    """Recursively freeze already-normalized JSON data."""
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def utc_text(value: datetime | None = None) -> str:
    """Serialize one timestamp in canonical RFC 3339 UTC form."""
    current = value or utc_now()
    if current.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return current.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def parse_utc(value: str) -> datetime:
    """Parse a canonical or ordinary RFC 3339 timestamp as UTC."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp must be a non-empty string")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def new_id(prefix: str) -> str:
    """Create a stable-shape opaque identifier."""
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,31}", prefix):
        raise ValueError("invalid id prefix")
    return f"{prefix}_{secrets.token_hex(16)}"


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return utc_text(value)
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            converted[key] = _jsonable(item)
        return converted
    if isinstance(value, (tuple, list, set, frozenset)):
        converted = [_jsonable(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(converted, key=lambda item: canonical_json(item))
        return converted
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite floats are not canonical JSON")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot canonicalize {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return deterministic UTF-8 JSON suitable for signatures and digests."""
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_digest(value: Any) -> str:
    """Return SHA-256 over canonical JSON."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _required_text(name: str, value: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    result = value.strip()
    if len(result) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return result


def _optional_text(name: str, value: str | None, *, maximum: int = 512) -> str | None:
    if value is None:
        return None
    return _required_text(name, value, maximum=maximum)


def _validate_id(name: str, value: str) -> str:
    result = _required_text(name, value, maximum=128)
    if not _ID_RE.fullmatch(result):
        raise ValueError(f"{name} must match {_ID_RE.pattern}")
    return result


class RiskLevel(IntEnum):
    """Normalized capability risk. Higher values dominate caller declarations."""

    LOW = 10
    MEDIUM = 20
    HIGH = 30
    CRITICAL = 40

    @classmethod
    def coerce(cls, value: "RiskLevel | str | int") -> "RiskLevel":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls[value.strip().upper()]
        return cls(value)


class Sensitivity(IntEnum):
    """Fact sensitivity ceiling used before prompt assembly."""

    PUBLIC = 10
    INTERNAL = 20
    CONFIDENTIAL = 30
    SECRET = 40

    @classmethod
    def coerce(cls, value: "Sensitivity | str | int") -> "Sensitivity":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls[value.strip().upper()]
        return cls(value)


class DecisionOutcome(str, Enum):
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    ALLOW = "allow"


class ContinuityMode(str, Enum):
    ISOLATED = "isolated"
    PRINCIPAL = "principal"
    WORKSPACE = "workspace"
    PROJECT = "project"
    EXPLICIT = "explicit"


@dataclass(frozen=True, slots=True)
class SurfaceRef:
    """One platform-native surface with tenant-qualified identity."""

    platform: str
    profile: str
    scope_id: str
    chat_id: str
    thread_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform", _required_text("platform", self.platform, maximum=64))
        object.__setattr__(self, "profile", _required_text("profile", self.profile, maximum=128))
        object.__setattr__(self, "scope_id", _required_text("scope_id", self.scope_id, maximum=256))
        object.__setattr__(self, "chat_id", _required_text("chat_id", self.chat_id, maximum=256))
        object.__setattr__(
            self,
            "thread_id",
            _optional_text("thread_id", self.thread_id, maximum=256),
        )

    @property
    def key(self) -> str:
        return canonical_digest(self)


@dataclass(frozen=True, slots=True)
class ExternalIdentity:
    """A platform identity scoped to profile and tenant/workspace."""

    platform: str
    profile: str
    scope_id: str
    external_id: str
    display_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform", _required_text("platform", self.platform, maximum=64))
        object.__setattr__(self, "profile", _required_text("profile", self.profile, maximum=128))
        object.__setattr__(self, "scope_id", _required_text("scope_id", self.scope_id, maximum=256))
        object.__setattr__(
            self,
            "external_id",
            _required_text("external_id", self.external_id, maximum=256),
        )
        object.__setattr__(
            self,
            "display_name",
            _optional_text("display_name", self.display_name, maximum=256),
        )

    @property
    def key(self) -> str:
        return canonical_digest(
            {
                "platform": self.platform,
                "profile": self.profile,
                "scope_id": self.scope_id,
                "external_id": self.external_id,
            }
        )


@dataclass(frozen=True, slots=True)
class ScopeRef:
    """Canonical governance scope carried into policy and leases."""

    profile: str
    platform: str
    scope_id: str
    chat_id: str
    principal_id: str
    thread_id: str | None = None
    project_id: str | None = None
    continuity_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", _required_text("profile", self.profile, maximum=128))
        object.__setattr__(self, "platform", _required_text("platform", self.platform, maximum=64))
        object.__setattr__(self, "scope_id", _required_text("scope_id", self.scope_id, maximum=256))
        object.__setattr__(self, "chat_id", _required_text("chat_id", self.chat_id, maximum=256))
        object.__setattr__(self, "principal_id", _validate_id("principal_id", self.principal_id))
        object.__setattr__(self, "thread_id", _optional_text("thread_id", self.thread_id, maximum=256))
        object.__setattr__(self, "project_id", _optional_text("project_id", self.project_id, maximum=128))
        if self.continuity_id is not None:
            object.__setattr__(
                self,
                "continuity_id",
                _validate_id("continuity_id", self.continuity_id),
            )

    @property
    def digest(self) -> str:
        return canonical_digest(self)


@dataclass(frozen=True, slots=True)
class Principal:
    principal_id: str
    display_name: str
    roles: tuple[str, ...] = ()
    guest: bool = False
    created_at: str = field(default_factory=utc_text)

    def __post_init__(self) -> None:
        object.__setattr__(self, "principal_id", _validate_id("principal_id", self.principal_id))
        object.__setattr__(
            self,
            "display_name",
            _required_text("display_name", self.display_name, maximum=256),
        )
        clean_roles = tuple(sorted({_required_text("role", item, maximum=128) for item in self.roles}))
        object.__setattr__(self, "roles", clean_roles)
        parse_utc(self.created_at)


@dataclass(frozen=True, slots=True)
class CapabilityDefinition:
    name: str
    risk: RiskLevel
    external_effect: bool = False
    network_egress: bool = False
    state_write: bool = False
    guest_eligible: bool = False
    required_scope_fields: tuple[str, ...] = ("profile", "platform", "scope_id", "chat_id")
    obligations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text("capability name", self.name, maximum=128))
        object.__setattr__(self, "risk", RiskLevel.coerce(self.risk))
        allowed = {
            "profile",
            "platform",
            "scope_id",
            "chat_id",
            "thread_id",
            "project_id",
            "continuity_id",
            "principal_id",
        }
        fields = tuple(dict.fromkeys(self.required_scope_fields))
        if not fields or any(item not in allowed for item in fields):
            raise ValueError("invalid required scope fields")
        object.__setattr__(self, "required_scope_fields", fields)
        object.__setattr__(
            self,
            "obligations",
            tuple(sorted({_required_text("obligation", item, maximum=128) for item in self.obligations})),
        )


@dataclass(frozen=True, slots=True)
class ActionIntent:
    capability: str
    action: str
    resource: str
    arguments_digest: str
    scope: ScopeRef
    risk: RiskLevel = RiskLevel.LOW
    external_effect: bool = False
    network_egress: bool = False
    state_write: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability", _required_text("capability", self.capability, maximum=128))
        object.__setattr__(self, "action", _required_text("action", self.action, maximum=256))
        object.__setattr__(self, "resource", _required_text("resource", self.resource, maximum=1024))
        digest = _required_text("arguments_digest", self.arguments_digest, maximum=64)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("arguments_digest must be lowercase SHA-256")
        object.__setattr__(self, "arguments_digest", digest)
        object.__setattr__(self, "risk", RiskLevel.coerce(self.risk))
        if not isinstance(self.metadata, Mapping):
            raise TypeError("intent metadata must be a mapping")
        metadata_json = canonical_json(self.metadata)
        if len(metadata_json.encode("utf-8")) > _MAX_INTENT_METADATA_BYTES:
            raise ValueError("intent metadata exceeds 65536 bytes")
        object.__setattr__(
            self,
            "metadata",
            _freeze_json_value(json.loads(metadata_json)),
        )

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "capability": self.capability,
                "action": self.action,
                "resource": self.resource,
                "arguments_digest": self.arguments_digest,
                "scope_digest": self.scope.digest,
                "risk": int(self.risk),
                "external_effect": self.external_effect,
                "network_egress": self.network_egress,
                "state_write": self.state_write,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    decision_id: str
    outcome: DecisionOutcome
    capability: str
    intent_digest: str
    scope_digest: str
    reasons: tuple[str, ...]
    obligations: tuple[str, ...] = ()
    matched_rules: tuple[str, ...] = ()
    budget_reservation_id: str | None = None
    approval_id: str | None = None
    decided_at: str = field(default_factory=utc_text)

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_id", _validate_id("decision_id", self.decision_id))
        object.__setattr__(self, "outcome", DecisionOutcome(self.outcome))
        object.__setattr__(self, "capability", _required_text("capability", self.capability, maximum=128))
        for name, value in (("intent_digest", self.intent_digest), ("scope_digest", self.scope_digest)):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{name} must be lowercase SHA-256")
        reasons = tuple(
            _required_text("decision reason", item, maximum=1024)
            for item in self.reasons
        )
        if not reasons:
            raise ValueError("policy decision requires at least one reason")
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "obligations",
            tuple(
                sorted(
                    {
                        _required_text("obligation", item, maximum=128)
                        for item in self.obligations
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "matched_rules",
            tuple(
                dict.fromkeys(
                    _required_text("matched rule", item, maximum=128)
                    for item in self.matched_rules
                )
            ),
        )
        for name in ("budget_reservation_id", "approval_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _validate_id(name, value))
        parse_utc(self.decided_at)


@dataclass(frozen=True, slots=True)
class ApprovalGrant:
    approval_id: str
    principal_id: str
    approver_id: str
    intent_digest: str
    scope_digest: str
    issued_at: str
    expires_at: str
    used_at: str | None = None

    def __post_init__(self) -> None:
        for name in ("approval_id", "principal_id", "approver_id"):
            object.__setattr__(self, name, _validate_id(name, getattr(self, name)))
        for name in ("intent_digest", "scope_digest"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"{name} must be lowercase SHA-256")
        issued = parse_utc(self.issued_at)
        expires = parse_utc(self.expires_at)
        if expires <= issued:
            raise ValueError("approval expiry must follow issuance")
        if self.used_at is not None:
            parse_utc(self.used_at)

    def is_current(self, now: datetime | None = None) -> bool:
        current = now or utc_now()
        return self.used_at is None and parse_utc(self.expires_at) > current


@dataclass(frozen=True, slots=True)
class CapabilityLease:
    lease_id: str
    principal_id: str
    continuity_id: str | None
    capability: str
    intent_digest: str
    scope_digest: str
    decision_id: str
    obligations: tuple[str, ...]
    budget_reservation_id: str | None
    approval_id: str | None
    issued_at: str
    expires_at: str
    nonce: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", _validate_id("lease_id", self.lease_id))
        object.__setattr__(self, "principal_id", _validate_id("principal_id", self.principal_id))
        if self.continuity_id is not None:
            object.__setattr__(
                self,
                "continuity_id",
                _validate_id("continuity_id", self.continuity_id),
            )
        object.__setattr__(self, "capability", _required_text("capability", self.capability, maximum=128))
        object.__setattr__(self, "decision_id", _validate_id("decision_id", self.decision_id))
        for name in ("budget_reservation_id", "approval_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _validate_id(name, value))
        for name in ("intent_digest", "scope_digest"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"{name} must be lowercase SHA-256")
        object.__setattr__(
            self,
            "obligations",
            tuple(
                sorted(
                    {
                        _required_text("obligation", item, maximum=128)
                        for item in self.obligations
                    }
                )
            ),
        )
        issued = parse_utc(self.issued_at)
        expires = parse_utc(self.expires_at)
        if expires <= issued:
            raise ValueError("lease expiry must follow issuance")
        object.__setattr__(self, "nonce", _required_text("nonce", self.nonce, maximum=128))


@dataclass(frozen=True, slots=True)
class ContinuityEnvelope:
    event_id: str
    continuity_id: str
    origin: SurfaceRef
    payload_digest: str
    propagation_path: tuple[str, ...] = ()
    hop_count: int = 0
    created_at: str = field(default_factory=utc_text)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _validate_id("event_id", self.event_id))
        object.__setattr__(self, "continuity_id", _validate_id("continuity_id", self.continuity_id))
        if not re.fullmatch(r"[0-9a-f]{64}", self.payload_digest):
            raise ValueError("payload_digest must be lowercase SHA-256")
        path = tuple(
            _required_text("propagation path entry", item, maximum=256)
            for item in self.propagation_path
        )
        object.__setattr__(self, "propagation_path", path)
        if self.hop_count < 0 or self.hop_count != len(path):
            raise ValueError("hop_count must equal propagation_path length")
        parse_utc(self.created_at)

    @property
    def fingerprint(self) -> str:
        return canonical_digest(
            {
                "continuity_id": self.continuity_id,
                "origin": self.origin,
                "payload_digest": self.payload_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class Fact:
    fact_id: str
    subject: str
    predicate: str
    value: Any
    scope: ScopeRef
    source_type: str
    source_id: str
    source_revision: str
    confidence: float
    authority: int
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    valid_from: str = field(default_factory=utc_text)
    valid_until: str | None = None
    supersedes: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_id", _validate_id("fact_id", self.fact_id))
        for name, maximum in (
            ("subject", 512),
            ("predicate", 256),
            ("source_type", 128),
            ("source_id", 512),
            ("source_revision", 512),
        ):
            object.__setattr__(self, name, _required_text(name, getattr(self, name), maximum=maximum))
        value_json = canonical_json(self.value)
        if len(value_json.encode("utf-8")) > _MAX_FACT_VALUE_BYTES:
            raise ValueError("fact value exceeds one megabyte")
        object.__setattr__(
            self,
            "value",
            _freeze_json_value(json.loads(value_json)),
        )
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(self.authority, int) or self.authority < 0:
            raise ValueError("authority must be a non-negative integer")
        object.__setattr__(self, "sensitivity", Sensitivity.coerce(self.sensitivity))
        start = parse_utc(self.valid_from)
        if self.valid_until is not None and parse_utc(self.valid_until) <= start:
            raise ValueError("valid_until must follow valid_from")
        if self.supersedes is not None:
            object.__setattr__(self, "supersedes", _validate_id("supersedes", self.supersedes))
        object.__setattr__(
            self,
            "tags",
            tuple(sorted({_required_text("tag", item, maximum=128) for item in self.tags})),
        )

    @property
    def content_hash(self) -> str:
        return canonical_digest(
            {
                "subject": self.subject,
                "predicate": self.predicate,
                "value": self.value,
                "scope": self.scope,
                "source_type": self.source_type,
                "source_id": self.source_id,
                "source_revision": self.source_revision,
            }
        )


@dataclass(frozen=True, slots=True)
class ContextBundle:
    facts: tuple[Fact, ...]
    conflicts: tuple[tuple[str, ...], ...] = ()
    omitted_count: int = 0
    rendered_text: str = ""

    def __post_init__(self) -> None:
        facts = tuple(self.facts)
        if len(facts) > 512 or not all(isinstance(item, Fact) for item in facts):
            raise ValueError("context facts must contain at most 512 facts")
        object.__setattr__(self, "facts", facts)

        conflicts: list[tuple[str, ...]] = []
        for raw_group in self.conflicts:
            group = tuple(sorted({_validate_id("conflict fact id", item) for item in raw_group}))
            if len(group) < 2:
                raise ValueError("context conflicts require at least two fact ids")
            conflicts.append(group)
        if len(conflicts) > 512:
            raise ValueError("context contains too many conflict groups")
        object.__setattr__(self, "conflicts", tuple(conflicts))

        if not isinstance(self.omitted_count, int) or self.omitted_count < 0:
            raise ValueError("omitted_count must be a non-negative integer")
        if not isinstance(self.rendered_text, str):
            raise TypeError("rendered_text must be a string")
        if len(self.rendered_text.encode("utf-8")) > 1_000_000:
            raise ValueError("rendered context exceeds one megabyte")


@dataclass(frozen=True, slots=True)
class TurnAdmission:
    admission_id: str
    principal: Principal
    surface: SurfaceRef
    scope: ScopeRef
    continuity_id: str
    context: ContextBundle
    admitted_at: str = field(default_factory=utc_text)
    shadow: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "admission_id", _validate_id("admission_id", self.admission_id))
        object.__setattr__(self, "continuity_id", _validate_id("continuity_id", self.continuity_id))
        if not isinstance(self.context, ContextBundle):
            raise TypeError("admission context must be a ContextBundle")
        if self.principal.principal_id != self.scope.principal_id:
            raise ValueError("admission principal does not own its scope")
        if self.continuity_id != self.scope.continuity_id:
            raise ValueError("admission continuity does not match its scope")
        for name in ("platform", "profile", "scope_id", "chat_id", "thread_id"):
            if getattr(self.surface, name) != getattr(self.scope, name):
                raise ValueError(f"admission surface {name} does not match its scope")
        if not isinstance(self.shadow, bool):
            raise TypeError("shadow must be a boolean")
        parse_utc(self.admitted_at)


def arguments_digest(arguments: Mapping[str, Any] | Sequence[Any] | None) -> str:
    """Hash bounded structured arguments without retaining their plaintext."""
    payload = canonical_json(arguments if arguments is not None else {})
    if len(payload.encode("utf-8")) > _MAX_ARGUMENT_BYTES:
        raise ValueError("arguments exceed one megabyte")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
