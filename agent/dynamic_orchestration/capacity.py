"""Pure capacity observations, views, reservations, and circuit breakers."""

from __future__ import annotations

import json
import re
from dataclasses import InitVar, dataclass, fields
from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar, Mapping, Sequence

from .route import (
    RouteV1,
    _revalidated_route,
)

from .state import (
    ReservationState,
)

from .validation import (
    DomainValidationError,
    _MAX_BREAKER_PROBE_BUDGET,
    _MAX_CAPACITY_CONCURRENCY,
    _MAX_CONTRACT_VERSION,
    _MAX_TEXT_COLLECTION_ITEMS,
    _bounded_non_negative_integer,
    _canonical_capacity_provenance_collection,
    _canonical_pool_identity,
    _finite_number,
    _mapping_snapshot,
    _parse_exact_enum,
    _reject_sensitive,
    _safe_asdict,
    _task_text,
    _validated_exact_label,
    _validated_mapping_keys,
    _validated_route_id,
)
class CapacityFreshness(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"

class CapacityCompleteness(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"

class CapacityConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

class CapacityMetric(str, Enum):
    REMAINING = "remaining"
    RESET_AT = "reset_at"
    LIMIT = "limit"
    USAGE = "usage"
    CONCURRENCY = "concurrency"
    PRICE = "price"
    UNKNOWN = "unknown"

class CapacityDimension(str, Enum):
    REQUEST = "request"
    TOKEN = "token"
    CONCURRENCY = "concurrency"

CAPACITY_OBSERVATION_SCHEMA_VERSION = "capacity-observation/v1"

@dataclass(frozen=True)
class CapacityValueV1:
    """A capacity value that distinguishes known, unknown, and zero explicitly."""

    known: bool
    value: float | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if type(self.known) is not bool:
            raise DomainValidationError(
                "capacity.value_invalid",
                "known must be boolean",
            )
        if self.known:
            if self.value is None:
                raise DomainValidationError(
                    "capacity.value_invalid",
                    "known capacity value requires a numeric value",
                )
            normalized = _finite_number(
                self.value,
                code="capacity.value_invalid",
                field_name="capacity value",
                non_negative=True,
            )
            object.__setattr__(self, "value", normalized)
            if self.unit is None:
                raise DomainValidationError(
                    "capacity.unit_invalid",
                    "known capacity value requires a unit",
                )
            object.__setattr__(self, "unit", _task_text(self.unit, "unit"))
        else:
            if self.value is not None or self.unit is not None:
                raise DomainValidationError(
                    "capacity.value_invalid",
                    "unknown capacity value must not carry value or unit",
                )
        _reject_sensitive(self, "capacity value")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CapacityValueV1":
        payload = _mapping_snapshot(
            payload,
            code="capacity.schema_invalid",
            location="capacity value",
        )
        expected = {field_.name for field_ in fields(cls)}
        unknown = _validated_mapping_keys(
            payload,
            code="capacity.schema_invalid",
            location="capacity value",
        ) - expected
        if unknown:
            raise DomainValidationError(
                "capacity.schema_invalid",
                f"unexpected capacity value fields: {sorted(unknown)}",
            )
        try:
            return cls(**payload)  # type: ignore[arg-type]
        except TypeError as exc:
            raise DomainValidationError(
                "capacity.schema_invalid",
                "malformed capacity value",
            ) from exc

    @property
    def is_zero(self) -> bool:
        return self.known and self.value == 0

@dataclass(frozen=True, kw_only=True)
class CapacityObservationV1:
    """Canonical capacity-observation/v1 wire DTO."""

    SCHEMA_VERSION: ClassVar[str] = CAPACITY_OBSERVATION_SCHEMA_VERSION
    observation_id: str
    quota_pool_id: str
    route_id: str | None = None
    billing_pool_id: str | None = None
    metric: str
    value: float | None = None
    unit: str | None = None
    source: str
    captured_at: str
    valid_until: str
    freshness: str
    confidence: float | str
    is_estimated: bool
    plan: str | None = None
    entitlement: str | None = None
    reset_at: str | None = None
    price: float | None = None
    max_concurrency: int | None = None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        for name in ("observation_id", "source"):
            object.__setattr__(self, name, _task_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "quota_pool_id",
            _canonical_pool_identity(
                self.quota_pool_id,
                code="capacity.pool_identity_invalid",
                field_name="quota_pool_id",
            ),
        )
        if type(self.metric) is not str:
            raise DomainValidationError("capacity.metric_invalid", "metric must be a known capacity metric")
        object.__setattr__(self, "metric", _task_text(self.metric, "metric"))
        if self.route_id is not None:
            object.__setattr__(self, "route_id", _validated_route_id(self.route_id, "route_id"))
        if self.billing_pool_id is not None:
            object.__setattr__(
                self,
                "billing_pool_id",
                _canonical_pool_identity(
                    self.billing_pool_id,
                    code="capacity.pool_identity_invalid",
                    field_name="billing_pool_id",
                ),
            )
        for name in ("plan", "entitlement", "reason_code"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _task_text(getattr(self, name), name))
        captured_text, captured = _capacity_timestamp(self.captured_at, "captured_at")
        valid_text, valid = _capacity_timestamp(self.valid_until, "valid_until")
        if valid <= captured:
            raise DomainValidationError("capacity.timestamp_order_invalid", "valid_until must be later than captured_at")
        object.__setattr__(self, "captured_at", captured_text)
        object.__setattr__(self, "valid_until", valid_text)
        if self.reset_at is not None:
            reset_text, _ = _capacity_timestamp(self.reset_at, "reset_at")
            object.__setattr__(self, "reset_at", reset_text)
        object.__setattr__(
            self,
            "freshness",
            _validated_exact_label(
                self.freshness,
                frozenset(item.value for item in CapacityFreshness),
                code="capacity.freshness_invalid",
                message="freshness must be fresh, stale, or unknown",
            ),
        )
        object.__setattr__(
            self,
            "metric",
            _validated_exact_label(
                self.metric,
                frozenset(item.value for item in CapacityMetric),
                code="capacity.metric_invalid",
                message="metric must be a known capacity metric",
            ),
        )
        if type(self.confidence) is str:
            if self.confidence != "unknown":
                raise DomainValidationError("capacity.confidence_invalid", "confidence must be 0..1 or unknown")
        else:
            normalized_confidence = _finite_number(
                self.confidence,
                code="capacity.confidence_invalid",
                field_name="confidence",
            )
            if not 0 <= normalized_confidence <= 1:
                raise DomainValidationError("capacity.confidence_invalid", "confidence must be 0..1 or unknown")
            object.__setattr__(self, "confidence", normalized_confidence)
        if type(self.is_estimated) is not bool:
            raise DomainValidationError("capacity.provenance_invalid", "is_estimated must be boolean")
        normalized_value = _capacity_number(self.value, "value")
        if normalized_value is None:
            if self.unit is not None:
                raise DomainValidationError("capacity.unit_invalid", "unit must be unknown when value is unknown")
        else:
            if self.unit is None:
                raise DomainValidationError("capacity.unit_invalid", "known value requires a unit")
            object.__setattr__(self, "unit", _task_text(self.unit, "unit"))
        object.__setattr__(self, "value", normalized_value)
        object.__setattr__(self, "price", _capacity_number(self.price, "price"))
        if self.max_concurrency is not None:
            object.__setattr__(
                self,
                "max_concurrency",
                _bounded_non_negative_integer(
                    self.max_concurrency,
                    code="capacity.max_concurrency_invalid",
                    field_name="max_concurrency",
                    maximum=_MAX_CAPACITY_CONCURRENCY,
                ),
            )
        _reject_sensitive(self, "capacity observation")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CapacityObservationV1":
        payload = _mapping_snapshot(payload, code="capacity.schema_invalid", location="capacity observation")
        expected = {field_.name for field_ in fields(cls)}
        keys = _validated_mapping_keys(payload, code="capacity.schema_invalid", location="capacity observation")
        if keys - expected:
            raise DomainValidationError("capacity.schema_invalid", f"unexpected capacity observation fields: {sorted(keys - expected)}")
        try:
            return cls(**payload)  # type: ignore[arg-type]
        except TypeError as exc:
            raise DomainValidationError("capacity.schema_invalid", "malformed capacity observation") from exc

    def is_fresh_at(self, built_at: datetime) -> bool:
        _, captured = _capacity_timestamp(self.captured_at, "captured_at")
        _, valid = _capacity_timestamp(self.valid_until, "valid_until")
        return self.freshness == "fresh" and captured <= built_at < valid

def _capacity_timestamp(value: object, field_name: str) -> tuple[str, datetime]:
    text = _task_text(value, field_name)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})", text) is None:
        raise DomainValidationError("capacity.timestamp_invalid", f"{field_name} must be timezone-aware RFC 3339")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DomainValidationError("capacity.timestamp_invalid", f"{field_name} must be valid RFC 3339") from exc
    return text, parsed.astimezone(timezone.utc)

def _capacity_number(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    return _finite_number(
        value,
        code="capacity.value_invalid",
        field_name=field_name,
        non_negative=True,
    )

class CapacityConflictDisposition(str, Enum):
    EXCLUDE = "EXCLUDE"
    AUTHORITATIVE_ONLY = "AUTHORITATIVE_ONLY"
    REQUIRE_HUMAN = "REQUIRE_HUMAN"

def _normalized_capacity_registry(
    value: object,
    *,
    code: str,
    location: str,
    required_key: str | None = None,
    canonical_identity_keys: bool = False,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise DomainValidationError(code, f"{location} must be a mapping")
    if len(value) > _MAX_TEXT_COLLECTION_ITEMS:
        raise DomainValidationError(
            code,
            f"{location} must contain at most {_MAX_TEXT_COLLECTION_ITEMS} entries",
        )
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if type(key) is not str or type(item) is not str:
            raise DomainValidationError(code, f"{location} requires string keys and values")
        normalized_key = (
            _canonical_pool_identity(
                key,
                code=code,
                field_name=f"{location} key",
            )
            if canonical_identity_keys
            else _task_text(key, f"{location} key")
        )
        normalized_value = _task_text(item, f"{location} value")
        if normalized_key in normalized:
            raise DomainValidationError(code, f"{location} contains duplicate normalized keys")
        normalized[normalized_key] = normalized_value
    if required_key is not None and required_key not in normalized:
        raise DomainValidationError(code, f"{location} must define {required_key}")
    return normalized

CAPACITY_POOL_SCHEMA_VERSION = "capacity-pool/v1"

@dataclass(frozen=True)
class PoolCapacityV1:
    """Effective capacity per quota pool with conservative known-min and conflict/stale metadata."""

    schema_version: str
    quota_pool_id: str
    request: CapacityValueV1
    token: CapacityValueV1
    concurrency: CapacityValueV1
    conflict_code: str | None = None
    stale_sources: tuple[str, ...] = ()
    unknown_sources: tuple[str, ...] = ()
    source_observation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != CAPACITY_POOL_SCHEMA_VERSION:
            raise DomainValidationError(
                "capacity.schema_invalid",
                "schema_version must be capacity-pool/v1",
            )
        object.__setattr__(
            self,
            "quota_pool_id",
            _canonical_pool_identity(
                self.quota_pool_id,
                code="capacity.pool_identity_invalid",
                field_name="quota_pool_id",
            ),
        )
        for name in ("request", "token", "concurrency"):
            value = getattr(self, name)
            if type(value) is not CapacityValueV1:
                raise DomainValidationError(
                    "capacity.pool_dimension_invalid",
                    f"{name} must be a CapacityValueV1",
                )
            object.__setattr__(
                self,
                name,
                CapacityValueV1.from_mapping(
                    {"known": value.known, "value": value.value, "unit": value.unit}
                ),
            )
        if self.conflict_code is not None:
            object.__setattr__(
                self,
                "conflict_code",
                _task_text(self.conflict_code, "conflict_code"),
            )
        for name in ("stale_sources", "unknown_sources", "source_observation_ids"):
            object.__setattr__(
                self,
                name,
                _canonical_capacity_provenance_collection(
                    getattr(self, name),
                    name,
                ),
            )
        _reject_capacity_sensitive(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "PoolCapacityV1":
        payload = _mapping_snapshot(
            payload,
            code="capacity.schema_invalid",
            location="pool capacity",
        )
        expected = {field_.name for field_ in fields(cls)}
        unknown = _validated_mapping_keys(
            payload,
            code="capacity.schema_invalid",
            location="pool capacity",
        ) - expected
        if unknown:
            raise DomainValidationError(
                "capacity.schema_invalid",
                f"unexpected pool capacity fields: {sorted(unknown)}",
            )
        values = dict(payload)
        for name in ("request", "token", "concurrency"):
            raw = values.get(name)
            if type(raw) is CapacityValueV1:
                continue
            if isinstance(raw, Mapping):
                values[name] = CapacityValueV1.from_mapping(raw)
            else:
                raise DomainValidationError(
                    "capacity.pool_dimension_invalid",
                    f"{name} must be a CapacityValueV1 or mapping",
                )
        try:
            return cls(**values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise DomainValidationError(
                "capacity.schema_invalid",
                "malformed pool capacity",
            ) from exc

    @property
    def derived_remaining(self) -> float | None:
        """The only capacity derived by v1: fresh, compatible `remaining` observations."""
        return self.request.value if self.request.known else None

    @property
    def known_min(self) -> CapacityValueV1:
        """Deprecated compatibility projection; it never compares different units/dimensions."""
        return self.request if self.request.known else CapacityValueV1(known=False)

    @property
    def is_healthy(self) -> bool:
        return self.request.known and self.request.value > 0 and not self.conflict_code  # type: ignore[operator]

    @property
    def is_unknown(self) -> bool:
        return not self.request.known

    @property
    def is_exhausted(self) -> bool:
        return self.request.known and self.request.value == 0 and not self.conflict_code  # type: ignore[operator]

CAPACITY_VIEW_SCHEMA_VERSION = "capacity-view/v1"

def _reject_capacity_sensitive(value: object) -> None:
    """Run the shared secret scanner without mistaking the token dimension for a credential."""
    if type(value) is PoolCapacityV1:
        payload = {
            "schema_version": value.schema_version,
            "quota_pool_id": value.quota_pool_id,
            "request_dimension": _safe_asdict(
                value.request,
                "capacity request dimension",
            ),
            "token_dimension": _safe_asdict(
                value.token,
                "capacity token dimension",
            ),
            "concurrency_dimension": _safe_asdict(
                value.concurrency,
                "capacity concurrency dimension",
            ),
            "conflict_code": value.conflict_code,
            "stale_sources": value.stale_sources,
            "unknown_sources": value.unknown_sources,
            "source_observation_ids": value.source_observation_ids,
        }
    elif type(value) is CapacityViewV1:
        payload = {
            "view_id": value.view_id,
            "built_at": value.built_at,
            "pool_dimensions": [
                {
                    "quota_pool_id": pool.quota_pool_id,
                    "request_dimension": _safe_asdict(
                        pool.request,
                        "capacity request dimension",
                    ),
                    "token_dimension": _safe_asdict(
                        pool.token,
                        "capacity token dimension",
                    ),
                    "concurrency_dimension": _safe_asdict(
                        pool.concurrency,
                        "capacity concurrency dimension",
                    ),
                    "conflict_code": pool.conflict_code,
                    "stale_sources": pool.stale_sources,
                    "unknown_sources": pool.unknown_sources,
                    "source_observation_ids": pool.source_observation_ids,
                }
                for pool in value.pools
            ],
            "source_observation_ids": value.source_observation_ids,
        }
    else:
        payload = value
    _reject_sensitive(payload, "capacity")

@dataclass(frozen=True)
class CapacityViewV1:
    """Immutable snapshot of pool capacities and source observations."""

    SCHEMA_VERSION: ClassVar[str] = CAPACITY_VIEW_SCHEMA_VERSION
    view_id: str
    built_at: str
    pools: tuple[PoolCapacityV1, ...]
    source_observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "view_id", _task_text(self.view_id, "view_id"))
        built_text, _ = _capacity_timestamp(self.built_at, "built_at")
        object.__setattr__(self, "built_at", built_text)
        if not isinstance(self.pools, Sequence) or isinstance(self.pools, (str, bytes, bytearray)):
            raise DomainValidationError(
                "capacity.pools_invalid",
                "pools must be a sequence",
            )
        if len(self.pools) > _MAX_TEXT_COLLECTION_ITEMS:
            raise DomainValidationError(
                "capacity.pools_invalid",
                f"pools must contain at most {_MAX_TEXT_COLLECTION_ITEMS} entries",
            )
        seen_pools: set[str] = set()
        normalized_pools: list[PoolCapacityV1] = []
        for pool in self.pools:
            if type(pool) is PoolCapacityV1:
                validated = PoolCapacityV1.from_mapping(
                    _safe_asdict(
                        pool,
                        "capacity pool",
                        allowed_field_paths=frozenset({("token",)}),
                    )
                )
            elif isinstance(pool, Mapping):
                validated = PoolCapacityV1.from_mapping(pool)
            else:
                raise DomainValidationError(
                    "capacity.pools_invalid",
                    "pool must be a PoolCapacityV1 or mapping",
                )
            if validated.quota_pool_id in seen_pools:
                raise DomainValidationError(
                    "capacity.pool_duplicate",
                    f"duplicate quota pool: {validated.quota_pool_id}",
                )
            seen_pools.add(validated.quota_pool_id)
            normalized_pools.append(validated)
        object.__setattr__(self, "pools", tuple(sorted(normalized_pools, key=lambda p: p.quota_pool_id)))
        object.__setattr__(
            self,
            "source_observation_ids",
            _canonical_capacity_provenance_collection(
                self.source_observation_ids,
                "source_observation_ids",
            ),
        )
        _reject_capacity_sensitive(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CapacityViewV1":
        payload = _mapping_snapshot(
            payload,
            code="capacity.schema_invalid",
            location="capacity view",
        )
        expected = {field_.name for field_ in fields(cls)}
        unknown = _validated_mapping_keys(
            payload,
            code="capacity.schema_invalid",
            location="capacity view",
        ) - expected
        if unknown:
            raise DomainValidationError(
                "capacity.schema_invalid",
                f"unexpected capacity view fields: {sorted(unknown)}",
            )
        values = dict(payload)
        pools = values.get("pools", ())
        if isinstance(pools, (str, bytes, bytearray)) or not isinstance(pools, Sequence):
            raise DomainValidationError(
                "capacity.pools_invalid",
                "pools must be a sequence",
            )
        if len(pools) > _MAX_TEXT_COLLECTION_ITEMS:
            raise DomainValidationError(
                "capacity.pools_invalid",
                f"pools must contain at most {_MAX_TEXT_COLLECTION_ITEMS} entries",
            )
        values["pools"] = tuple(
            PoolCapacityV1.from_mapping(pool) if isinstance(pool, Mapping) else pool
            for pool in pools
        )
        try:
            return cls(**values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise DomainValidationError(
                "capacity.schema_invalid",
                "malformed capacity view",
            ) from exc

    def canonical_object(self) -> dict[str, object]:
        return {
            "view_id": self.view_id,
            "built_at": self.built_at,
            "pools": [
                {
                    "schema_version": pool.schema_version,
                    "quota_pool_id": pool.quota_pool_id,
                    "request": {
                        "known": pool.request.known,
                        "value": pool.request.value,
                        "unit": pool.request.unit,
                    },
                    "token": {
                        "known": pool.token.known,
                        "value": pool.token.value,
                        "unit": pool.token.unit,
                    },
                    "concurrency": {
                        "known": pool.concurrency.known,
                        "value": pool.concurrency.value,
                        "unit": pool.concurrency.unit,
                    },
                    "conflict_code": pool.conflict_code,
                    "stale_sources": list(pool.stale_sources),
                    "unknown_sources": list(pool.unknown_sources),
                    "source_observation_ids": list(pool.source_observation_ids),
                }
                for pool in self.pools
            ],
            "source_observation_ids": list(self.source_observation_ids),
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_object(),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
            sort_keys=True,
        )

    def get_pool(self, quota_pool_id: str) -> PoolCapacityV1 | None:
        canonical_pool_id = _canonical_pool_identity(
            quota_pool_id,
            code="capacity.pool_identity_invalid",
            field_name="quota_pool_id",
        )
        for pool in self.pools:
            if pool.quota_pool_id == canonical_pool_id:
                return pool
        return None

def build_capacity_view(
    *,
    view_id: str,
    built_at: str,
    observations: Sequence[CapacityObservationV1],
    conflict_disposition: CapacityConflictDisposition,
    canonical_units: Mapping[str, str] | None = None,
    authoritative_sources: Mapping[str, str] | None = None,
) -> CapacityViewV1:
    """Pure view builder: only fresh, unexpired, registry-unit-compatible remaining values derive capacity."""
    _, built = _capacity_timestamp(built_at, "built_at")
    if isinstance(observations, (str, bytes, bytearray)) or not isinstance(observations, Sequence):
        raise DomainValidationError("capacity.builder_invalid", "observations must be a sequence")
    if type(conflict_disposition) is not CapacityConflictDisposition:
        raise DomainValidationError("capacity.builder_invalid", "conflict_disposition must be typed")
    units = _normalized_capacity_registry(
        canonical_units,
        code="capacity.unit_registry_required",
        location="canonical unit registry",
        required_key="remaining",
    )
    canonical_unit = units["remaining"]
    if authoritative_sources is None:
        authority: dict[str, str] = {}
    else:
        authority = _normalized_capacity_registry(
            authoritative_sources,
            code="capacity.authority_required",
            location="authoritative source registry",
            canonical_identity_keys=True,
        )
    if conflict_disposition is CapacityConflictDisposition.AUTHORITATIVE_ONLY and not authority:
        raise DomainValidationError("capacity.authority_required", "AUTHORITATIVE_ONLY requires external authoritative source registry")
    if len(observations) > _MAX_TEXT_COLLECTION_ITEMS:
        raise DomainValidationError(
            "capacity.builder_invalid",
            f"observations must contain at most {_MAX_TEXT_COLLECTION_ITEMS} entries",
        )
    grouped: dict[str, list[CapacityObservationV1]] = {}
    seen: set[str] = set()
    for raw in observations:
        if type(raw) is not CapacityObservationV1:
            raise DomainValidationError("capacity.builder_invalid", "every observation must be validated")
        observation = CapacityObservationV1.from_mapping(
            _safe_asdict(raw, "capacity observation")
        )
        if observation.observation_id in seen:
            raise DomainValidationError("capacity.observation_duplicate", "duplicate observation_id")
        seen.add(observation.observation_id)
        grouped.setdefault(observation.quota_pool_id, []).append(observation)
    if conflict_disposition is CapacityConflictDisposition.AUTHORITATIVE_ONLY:
        for quota_pool_id in authority:
            grouped.setdefault(quota_pool_id, [])
    pools: list[PoolCapacityV1] = []
    for quota_pool_id, pool_observations in sorted(grouped.items()):
        stale_sources: set[str] = set()
        unknown_sources: set[str] = set()
        authoritative_source = authority.get(quota_pool_id)
        non_authoritative = ()
        if conflict_disposition is CapacityConflictDisposition.AUTHORITATIVE_ONLY:
            non_authoritative = tuple(
                observation
                for observation in pool_observations
                if observation.source != authoritative_source
            )
            unknown_sources.update(observation.source for observation in non_authoritative)
            pool_observations = [
                observation
                for observation in pool_observations
                if observation.source == authoritative_source
            ]
        candidates: list[CapacityObservationV1] = []
        incompatible = False
        for observation in pool_observations:
            if not observation.is_fresh_at(built):
                stale_sources.add(observation.source)
                continue
            if observation.metric != "remaining" or observation.value is None:
                unknown_sources.add(observation.source)
                continue
            if observation.unit != canonical_unit:
                incompatible = True
                unknown_sources.add(observation.source)
                continue
            candidates.append(observation)
        values = {observation.value for observation in candidates}
        conflict = incompatible or len(values) > 1
        conflict_code = (
            "incompatible_unit"
            if incompatible
            else "conflicting_remaining_values"
            if len(values) > 1
            else None
        )
        if conflict_disposition is CapacityConflictDisposition.AUTHORITATIVE_ONLY:
            if not candidates:
                conflict_code = (
                    "incompatible_unit"
                    if incompatible
                    else "authoritative_source_unavailable"
                )
        if conflict and conflict_disposition is CapacityConflictDisposition.REQUIRE_HUMAN:
            raise DomainValidationError("capacity.builder_human_required", "actual fresh capacity conflict requires human policy")
        selected = candidates
        if conflict and conflict_disposition is CapacityConflictDisposition.EXCLUDE:
            selected = []
        if conflict and conflict_disposition is CapacityConflictDisposition.AUTHORITATIVE_ONLY:
            selected = []
        derived = min((item.value for item in selected), default=None)
        request = CapacityValueV1(known=derived is not None, value=derived, unit=canonical_unit if derived is not None else None)
        pools.append(PoolCapacityV1(
            schema_version=CAPACITY_POOL_SCHEMA_VERSION,
            quota_pool_id=quota_pool_id,
            request=request,
            token=CapacityValueV1(known=False),
            concurrency=CapacityValueV1(known=False),
            conflict_code=conflict_code,
            stale_sources=tuple(sorted(stale_sources)),
            unknown_sources=tuple(sorted(unknown_sources)),
            source_observation_ids=tuple(sorted(item.observation_id for item in pool_observations)),
        ))
    return CapacityViewV1(
        view_id=view_id,
        built_at=built_at,
        pools=tuple(pools),
        source_observation_ids=tuple(sorted(seen)),
    )

def _bounded_non_negative_number(
    value: object,
    *,
    code: str,
    field_name: str,
) -> float:
    return _finite_number(
        value,
        code=code,
        field_name=field_name,
        non_negative=True,
    )

def _bounded_contract_timestamp(
    value: object,
    *,
    code: str,
    field_name: str,
) -> str:
    try:
        text, _ = _capacity_timestamp(value, field_name)
    except DomainValidationError as exc:
        raise DomainValidationError(
            code,
            f"{field_name} must be timezone-aware RFC 3339",
        ) from exc
    return text

@dataclass(frozen=True)
class CapacityReservationV1:
    """Pure reservation lease contract; persistence and wall-clock authority are external."""

    SCHEMA_VERSION: ClassVar[str] = "capacity-reservation/v1"
    reservation_id: str
    route_id: str
    quota_pool_id: str
    billing_pool_id: str
    owner_attempt_id: str
    unit: str
    estimated_amount: float
    held_amount: float
    expires_at: str
    status: ReservationState
    version: int
    trusted_route: InitVar[RouteV1 | None] = None

    def __post_init__(self, trusted_route: RouteV1 | None) -> None:
        if type(trusted_route) is not RouteV1:
            raise DomainValidationError(
                "reservation.trusted_route_required",
                "capacity reservation requires trusted RouteV1 context",
            )
        try:
            route = _revalidated_route(trusted_route, "reservation route")
        except DomainValidationError as exc:
            if exc.code == "contract.payload_too_complex":
                raise
            raise DomainValidationError(
                "reservation.trusted_route_invalid",
                "trusted reservation route failed structural rehydration",
            ) from exc

        for name in (
            "reservation_id",
            "owner_attempt_id",
            "unit",
        ):
            object.__setattr__(self, name, _task_text(getattr(self, name), name))
        for name in ("quota_pool_id", "billing_pool_id"):
            object.__setattr__(
                self,
                name,
                _canonical_pool_identity(
                    getattr(self, name),
                    code="reservation.pool_identity_invalid",
                    field_name=name,
                ),
            )
        object.__setattr__(
            self,
            "route_id",
            _validated_route_id(self.route_id, "route_id"),
        )
        if (
            self.route_id != route.route_id
            or self.quota_pool_id != route.quota_pool_id
            or self.billing_pool_id != route.billing_pool_id
        ):
            raise DomainValidationError(
                "reservation.route_binding_invalid",
                "reservation route and quota/billing pools must match trusted route",
            )

        object.__setattr__(
            self,
            "estimated_amount",
            _bounded_non_negative_number(
                self.estimated_amount,
                code="reservation.amount_invalid",
                field_name="estimated_amount",
            ),
        )
        object.__setattr__(
            self,
            "held_amount",
            _bounded_non_negative_number(
                self.held_amount,
                code="reservation.amount_invalid",
                field_name="held_amount",
            ),
        )
        object.__setattr__(
            self,
            "expires_at",
            _bounded_contract_timestamp(
                self.expires_at,
                code="reservation.timestamp_invalid",
                field_name="expires_at",
            ),
        )
        status = _parse_exact_enum(
            self.status,
            ReservationState,
            code="reservation.status_invalid",
            message="status must be a reservation state",
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "version",
            _bounded_non_negative_integer(
                self.version,
                code="reservation.version_invalid",
                field_name="version",
                maximum=_MAX_CONTRACT_VERSION,
            ),
        )

        if status is ReservationState.PENDING and self.held_amount != 0:
            raise DomainValidationError(
                "reservation.state_invariant",
                "PENDING reservation must have held_amount=0",
            )
        if status is ReservationState.HELD and (
            self.held_amount <= 0 or self.held_amount > self.estimated_amount
        ):
            raise DomainValidationError(
                "reservation.state_invariant",
                "HELD reservation requires 0 < held_amount <= estimated_amount",
            )
        if status is ReservationState.CONSUMED and self.held_amount <= 0:
            raise DomainValidationError(
                "reservation.state_invariant",
                "CONSUMED reservation requires held_amount>0",
            )
        if status in {
            ReservationState.RELEASED,
            ReservationState.EXPIRED,
            ReservationState.RECONCILED,
        } and self.held_amount != 0:
            raise DomainValidationError(
                "reservation.state_invariant",
                "released, expired, and reconciled reservations cannot retain a dispatchable hold",
            )
        _reject_sensitive(self, "capacity reservation")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        trusted_route: RouteV1 | None = None,
    ) -> "CapacityReservationV1":
        payload = _mapping_snapshot(
            payload,
            code="reservation.schema_invalid",
            location="capacity reservation",
        )
        expected = {field_.name for field_ in fields(cls)}
        keys = _validated_mapping_keys(
            payload,
            code="reservation.schema_invalid",
            location="capacity reservation",
        )
        if keys != expected:
            raise DomainValidationError(
                "reservation.schema_invalid",
                "capacity reservation must contain the exact v1 fields",
            )
        values = dict(payload)
        status = values.get("status")
        values["status"] = _parse_exact_enum(
            status,
            ReservationState,
            code="reservation.status_invalid",
            message="status must be a reservation state",
        )
        try:
            return cls(**values, trusted_route=trusted_route)  # type: ignore[arg-type]
        except TypeError as exc:
            raise DomainValidationError(
                "reservation.schema_invalid",
                "malformed capacity reservation",
            ) from exc

    def canonical_object(self) -> dict[str, object]:
        return {
            "reservation_id": self.reservation_id,
            "route_id": self.route_id,
            "quota_pool_id": self.quota_pool_id,
            "billing_pool_id": self.billing_pool_id,
            "owner_attempt_id": self.owner_attempt_id,
            "unit": self.unit,
            "estimated_amount": self.estimated_amount,
            "held_amount": self.held_amount,
            "expires_at": self.expires_at,
            "status": self.status.value,
            "version": self.version,
        }

class CircuitBreakerStatus(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

@dataclass(frozen=True)
class CircuitBreakerV1:
    """Pure route/quota-pool-scoped breaker contract."""

    SCHEMA_VERSION: ClassVar[str] = "circuit-breaker/v1"
    breaker_id: str
    route_id: str
    quota_pool_id: str
    status: CircuitBreakerStatus
    cooldown_until: str | None
    probe_budget: int
    reason_code: str | None
    version: int
    trusted_route: InitVar[RouteV1 | None] = None

    def __post_init__(self, trusted_route: RouteV1 | None) -> None:
        if type(trusted_route) is not RouteV1:
            raise DomainValidationError(
                "breaker.trusted_route_required",
                "circuit breaker requires trusted RouteV1 context",
            )
        try:
            route = _revalidated_route(trusted_route, "breaker route")
        except DomainValidationError as exc:
            if exc.code == "contract.payload_too_complex":
                raise
            raise DomainValidationError(
                "breaker.trusted_route_invalid",
                "trusted breaker route failed structural rehydration",
            ) from exc

        object.__setattr__(self, "breaker_id", _task_text(self.breaker_id, "breaker_id"))
        object.__setattr__(
            self,
            "route_id",
            _validated_route_id(self.route_id, "route_id"),
        )
        object.__setattr__(
            self,
            "quota_pool_id",
            _canonical_pool_identity(
                self.quota_pool_id,
                code="breaker.pool_identity_invalid",
                field_name="quota_pool_id",
            ),
        )
        if self.route_id != route.route_id or self.quota_pool_id != route.quota_pool_id:
            raise DomainValidationError(
                "breaker.route_binding_invalid",
                "breaker route and quota pool must match trusted route",
            )

        status = _parse_exact_enum(
            self.status,
            CircuitBreakerStatus,
            code="breaker.status_invalid",
            message="status must be a circuit breaker state",
        )
        object.__setattr__(self, "status", status)
        if self.cooldown_until is not None:
            object.__setattr__(
                self,
                "cooldown_until",
                _bounded_contract_timestamp(
                    self.cooldown_until,
                    code="breaker.timestamp_invalid",
                    field_name="cooldown_until",
                ),
            )
        object.__setattr__(
            self,
            "probe_budget",
            _bounded_non_negative_integer(
                self.probe_budget,
                code="breaker.probe_budget_invalid",
                field_name="probe_budget",
                maximum=_MAX_BREAKER_PROBE_BUDGET,
            ),
        )
        if self.reason_code is not None:
            object.__setattr__(
                self,
                "reason_code",
                _task_text(self.reason_code, "reason_code"),
            )
        object.__setattr__(
            self,
            "version",
            _bounded_non_negative_integer(
                self.version,
                code="breaker.version_invalid",
                field_name="version",
                maximum=_MAX_CONTRACT_VERSION,
            ),
        )

        if status is CircuitBreakerStatus.OPEN and self.cooldown_until is None:
            raise DomainValidationError(
                "breaker.state_invariant",
                "OPEN breaker requires cooldown_until",
            )
        if status is CircuitBreakerStatus.HALF_OPEN and self.probe_budget <= 0:
            raise DomainValidationError(
                "breaker.state_invariant",
                "HALF_OPEN breaker requires probe_budget>0",
            )
        if status is CircuitBreakerStatus.CLOSED and (
            self.cooldown_until is not None or self.probe_budget != 0
        ):
            raise DomainValidationError(
                "breaker.state_invariant",
                "CLOSED breaker has no cooldown and probe_budget=0",
            )
        _reject_sensitive(self, "circuit breaker")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        trusted_route: RouteV1 | None = None,
    ) -> "CircuitBreakerV1":
        payload = _mapping_snapshot(
            payload,
            code="breaker.schema_invalid",
            location="circuit breaker",
        )
        expected = {field_.name for field_ in fields(cls)}
        keys = _validated_mapping_keys(
            payload,
            code="breaker.schema_invalid",
            location="circuit breaker",
        )
        if keys != expected:
            raise DomainValidationError(
                "breaker.schema_invalid",
                "circuit breaker must contain the exact v1 fields",
            )
        values = dict(payload)
        status = values.get("status")
        values["status"] = _parse_exact_enum(
            status,
            CircuitBreakerStatus,
            code="breaker.status_invalid",
            message="status must be a circuit breaker state",
        )
        try:
            return cls(**values, trusted_route=trusted_route)  # type: ignore[arg-type]
        except TypeError as exc:
            raise DomainValidationError(
                "breaker.schema_invalid",
                "malformed circuit breaker",
            ) from exc

    def canonical_object(self) -> dict[str, object]:
        return {
            "breaker_id": self.breaker_id,
            "route_id": self.route_id,
            "quota_pool_id": self.quota_pool_id,
            "status": self.status.value,
            "cooldown_until": self.cooldown_until,
            "probe_budget": self.probe_budget,
            "reason_code": self.reason_code,
            "version": self.version,
        }
