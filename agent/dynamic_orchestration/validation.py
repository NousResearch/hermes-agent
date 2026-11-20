"""Shared validation, normalization, bounds, and graph-safety helpers."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from itertools import islice
from typing import Mapping, Sequence

ROUTE_CANONICALIZATION_VERSION = "route-v1"

QUALITY_COMPENSATION_SCHEMA_VERSION = "quality-compensation/v1"

_ROUTE_FIELDS = (
    "canonicalization_version",
    "provider",
    "product",
    "surface",
    "account_id",
    "entitlement_id",
    "billing_pool_id",
    "quota_pool_id",
    "model",
    "endpoint",
    "variant",
    "region",
)

_REQUIRED_ROUTE_FIELDS = (
    "provider",
    "product",
    "surface",
    "account_id",
    "billing_pool_id",
    "quota_pool_id",
    "model",
    "endpoint",
    "region",
)

_ASCII_WHITESPACE = " \t\r\n\f\v"

_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")

_HEX = frozenset("0123456789abcdefABCDEF")

_PROHIBITED_FIELD_NAMES = frozenset(
    {
        "prompt",
        "secret",
        "secrets",
        "client_secret",
        "credential",
        "credentials",
        "authorization",
        "api_key",
        "password",
        "passwd",
        "private_key",
        "token",
        "auth_token",
        "access_token",
        "refresh_token",
        "bearer_token",
        "raw_body",
        "raw_provider",
        "raw_provider_body",
        "raw_tool_result",
    }
)

_VERIFICATION_RANK = {f"V{level}": level for level in range(5)}

_EFFORT_LABELS = frozenset(f"E{level}" for level in range(5))

_VERIFICATION_LABELS = frozenset(_VERIFICATION_RANK)

_POLICY_STATUS_LABELS = frozenset(
    {
        "NO_ROUTE_REQUIRED",
        "WAITING_FOR_CAPACITY",
        "ACTIVATION_BLOCKED_INDEPENDENT_REVIEW",
        "ACTIVATION_BLOCKED_HUMAN_GATE",
        "ACTIVATION_BLOCKED_QUALITY_COMPENSATION",
        "ACTIVATION_BLOCKED_EXTERNAL_AUTHORITY_UNAVAILABLE",
    }
)

_ACTIVATION_BLOCK_REASON_LABELS = frozenset(
    {
        "independent_review_required",
        "human_gate_required",
        "quality_compensation_insufficient",
        "external_authority_store_unimplemented",
    }
)

_ROUTE_ID_PATTERN = re.compile(r"^route-v1:[0-9a-f]{64}$")

_STABLE_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")

_MAX_STABLE_IDENTIFIERS = 32

_MAX_TEXT_LENGTH = 8192

_MAX_TEXT_COLLECTION_ITEMS = 128

_MAX_MAPPING_FIELDS = 128

_MAX_GRAPH_DEPTH = 32

_MAX_GRAPH_NODES = 8192

_MAX_DECISION_CANDIDATES = 256

_MAX_ROUTE_REGISTRY_ITEMS = 512

_MAX_CONTEXT_TOKENS = 2**31 - 1

_MAX_CAPACITY_CONCURRENCY = 2**31 - 1

_MAX_BREAKER_PROBE_BUDGET = 2**31 - 1

_MAX_CONTRACT_VERSION = 2**63 - 1

_KNOWN_CAPABILITY_IDENTIFIERS = frozenset(
    {
        "browser.private",
        "filesystem.read",
        "filesystem.write",
        "network.read",
        "repository.read",
        "repository.write",
    }
)

_KNOWN_TOOL_IDENTIFIERS = frozenset(
    {
        "computer-use",
        "execute_code",
        "patch",
        "read_file",
        "search_files",
        "terminal",
        "web_extract",
        "web_search",
    }
)

_KNOWN_PERMISSION_IDENTIFIERS = frozenset(
    {
        "browser.private",
        "filesystem.read",
        "filesystem.write",
        "network.read",
        "repository.read",
        "repository.write",
    }
)

_SENSITIVE_VALUE_PATTERN = re.compile(
    r"(?:"
    r"(?<![A-Za-z0-9])bearer\s+\S+"
    r"|(?<![A-Za-z0-9])(?:sk-|gho_|github_pat_|xox[baprs]-)[A-Za-z0-9_-]+"
    r"|(?:password|passwd|secret|client[_-]?secret|api[_-]?key|prompt|credential|"
    r"authorization|token|auth[_-]?token|access[_-]?token|refresh[_-]?token|"
    r"private[_-]?key)\s*[:=]\s*\S+"
    r"|-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE\s+KEY-----"
    r")",
    re.IGNORECASE,
)

class DomainValidationError(ValueError):
    """A stable, public validation error safe for deterministic tests and audit."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")

def _validated_future_timestamp(
    value: object,
    field_name: str,
    *,
    reference_time: datetime | None = None,
) -> str:
    text = _task_text(value, field_name)
    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z",
        text,
    ) is None:
        raise DomainValidationError(
            "task.justification_expiry_invalid",
            f"{field_name} must be canonical UTC RFC 3339",
        )
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DomainValidationError(
            "task.justification_expiry_invalid",
            f"{field_name} must be an RFC 3339 timestamp",
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DomainValidationError(
            "task.justification_expiry_invalid",
            f"{field_name} must include a timezone",
        )
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)
    elif reference_time.tzinfo is None or reference_time.utcoffset() is None:
        raise DomainValidationError(
            "task.reference_time_invalid",
            "reference_time must be timezone-aware",
        )
    if parsed <= reference_time:
        raise DomainValidationError(
            "task.justification_expired",
            f"{field_name} must be in the future",
        )
    return text

def _reject_sensitive_mapping_key(key: object, location: str) -> None:
    if type(key) is str and key.casefold() in _PROHIBITED_FIELD_NAMES:
        raise DomainValidationError(
            "decision.sensitive_field_prohibited",
            f"sensitive field prohibited in {location}",
        )
    _reject_sensitive(key, location)

def _mapping_snapshot(
    payload: object,
    *,
    code: str,
    location: str,
) -> dict[object, object]:
    """Snapshot a decoded mapping at a public ``from_mapping`` boundary.

    Ordinary decoded ``dict`` input is returned unchanged, so JSON/dict
    behaviour and error precedence stay identical. Any other ``Mapping`` is
    copied into a plain ``dict`` while every container hook (``__len__``,
    iteration, ``__getitem__``) is contained: a hostile custom ``Mapping`` whose
    hooks raise surfaces a stable :class:`DomainValidationError` instead of
    leaking its own ``RuntimeError``/``TypeError``/``ValueError``. Values are
    never coerced and nested containers are handed to their owning validator
    untouched. Oversized mappings keep the bounded sensitive-key prefix scan and
    are rejected without a single value being read.
    """

    if not isinstance(payload, Mapping):
        raise DomainValidationError(code, f"{location} must be a mapping")
    if type(payload) is dict:
        return payload
    try:
        size = len(payload)
    except DomainValidationError:
        raise
    except Exception as exc:  # hostile __len__ on a custom Mapping adapter
        raise DomainValidationError(code, f"{location} must be a mapping") from exc
    if size > _MAX_MAPPING_FIELDS:
        # Mirror the overflow branch below without reading any value: iterate at
        # most N+1 keys, preserve the sensitive-key precedence, then fail closed.
        try:
            for key in islice(payload, _MAX_MAPPING_FIELDS + 1):
                _reject_sensitive_mapping_key(key, location)
        except DomainValidationError:
            raise
        except Exception as exc:  # hostile iteration on a custom Mapping adapter
            raise DomainValidationError(code, f"{location} must be a mapping") from exc
        raise DomainValidationError(
            "contract.payload_too_complex",
            f"{location} exceeds the maximum field count",
        )
    snapshot: dict[object, object] = {}
    try:
        for key in islice(payload, _MAX_MAPPING_FIELDS + 1):
            snapshot[key] = payload[key]
    except DomainValidationError:
        raise
    except Exception as exc:  # hostile iteration or __getitem__ on an adapter
        raise DomainValidationError(code, f"{location} must be a mapping") from exc
    return snapshot

def _validated_mapping_keys(
    payload: object,
    *,
    code: str,
    location: str,
) -> set[str]:
    snapshot = _mapping_snapshot(payload, code=code, location=location)
    if len(snapshot) > _MAX_MAPPING_FIELDS:
        # Preserve sensitive-key precedence at the exact public-contract
        # overflow edge without turning the secret scan into unbounded work.
        # Mapping iteration order defines a deterministic prefix; values are
        # never read, and huge mappings stop after N+1 keys.
        for key in islice(snapshot, _MAX_MAPPING_FIELDS + 1):
            _reject_sensitive_mapping_key(key, location)
        raise DomainValidationError(
            "contract.payload_too_complex",
            f"{location} exceeds the maximum field count",
        )
    if any(type(key) is not str for key in snapshot):
        raise DomainValidationError(code, f"{location} field names must be strings")
    return {str(key) for key in snapshot}

def _ascii_trimmed_nfc(value: object, *, field_name: str, code: str) -> str:
    if type(value) is not str:
        raise DomainValidationError(code, f"{field_name} must be a string")
    if len(value) > _MAX_TEXT_LENGTH:
        raise DomainValidationError(
            code,
            f"{field_name} must contain at most {_MAX_TEXT_LENGTH} characters",
        )
    normalized = unicodedata.normalize("NFC", value.strip(_ASCII_WHITESPACE))
    if not normalized:
        raise DomainValidationError(code, f"{field_name} is required")
    if len(normalized) > _MAX_TEXT_LENGTH:
        raise DomainValidationError(
            code,
            f"{field_name} must contain at most {_MAX_TEXT_LENGTH} characters",
        )
    return normalized

def _normalized_text(
    value: object,
    *,
    required: bool = False,
    field_name: str = "value",
) -> str | None:
    if value is None:
        if required:
            raise DomainValidationError("route.identity_required", f"{field_name} is required")
        return None
    if type(value) is not str:
        raise DomainValidationError("route.value_invalid", f"{field_name} must be a string")
    if len(value) > _MAX_TEXT_LENGTH:
        raise DomainValidationError(
            "route.value_invalid",
            f"{field_name} must contain at most {_MAX_TEXT_LENGTH} characters",
        )
    normalized = unicodedata.normalize("NFC", value.strip(_ASCII_WHITESPACE)).casefold()
    if not normalized:
        if required:
            raise DomainValidationError("route.identity_required", f"{field_name} is required")
        return None
    if len(normalized) > _MAX_TEXT_LENGTH:
        raise DomainValidationError(
            "route.value_invalid",
            f"{field_name} must contain at most {_MAX_TEXT_LENGTH} characters",
        )
    if any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in normalized):
        raise DomainValidationError(
            "route.value_invalid",
            f"{field_name} must not contain control, format, or surrogate characters",
        )
    return normalized

def _task_text(value: object, field_name: str) -> str:
    return _ascii_trimmed_nfc(
        value,
        field_name=field_name,
        code="task.scalar_invalid",
    )

def _validated_route_id(value: object, field_name: str) -> str:
    route_id = _task_text(value, field_name)
    if _ROUTE_ID_PATTERN.fullmatch(route_id) is None:
        raise DomainValidationError(
            "route.identity_invalid",
            f"{field_name} must be route-v1:<lowercase SHA-256>",
        )
    return route_id

def _immutable_string_collection(
    value: object,
    field_name: str,
    *,
    require_nonempty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DomainValidationError(
            "task.collection_invalid",
            f"{field_name} must be a collection of strings",
        )
    if len(value) > _MAX_TEXT_COLLECTION_ITEMS:
        raise DomainValidationError(
            "task.collection_invalid",
            f"{field_name} must contain at most {_MAX_TEXT_COLLECTION_ITEMS} entries",
        )
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise DomainValidationError(
                "task.collection_invalid",
                f"{field_name} must contain only strings",
            )
        text = _ascii_trimmed_nfc(
            item,
            field_name=f"{field_name} entry",
            code="task.collection_invalid",
        )
        normalized.append(text)
    if require_nonempty and not normalized:
        raise DomainValidationError(
            "task.collection_invalid",
            f"{field_name} must not be empty",
        )
    return tuple(normalized)

def _canonical_capacity_provenance_collection(
    value: object,
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DomainValidationError(
            "capacity.provenance_invalid",
            f"{field_name} must be a collection of strings",
        )
    if len(value) > _MAX_TEXT_COLLECTION_ITEMS:
        raise DomainValidationError(
            "capacity.provenance_invalid",
            f"{field_name} must contain at most {_MAX_TEXT_COLLECTION_ITEMS} entries",
        )
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if type(item) is not str:
            raise DomainValidationError(
                "capacity.provenance_invalid",
                f"{field_name} must contain only exact strings",
            )
        text = _ascii_trimmed_nfc(
            item,
            field_name=f"{field_name} entry",
            code="capacity.provenance_invalid",
        )
        if text in seen:
            raise DomainValidationError(
                "capacity.provenance_duplicate",
                f"{field_name} must not contain duplicates",
            )
        seen.add(text)
        normalized.append(text)
    return tuple(sorted(normalized))

def _stable_identifier_collection(
    value: object,
    field_name: str,
    *,
    code: str,
    require_nonempty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DomainValidationError(code, f"{field_name} must be a collection of identifiers")
    if len(value) > _MAX_STABLE_IDENTIFIERS:
        raise DomainValidationError(
            code,
            f"{field_name} must contain at most {_MAX_STABLE_IDENTIFIERS} identifiers",
        )
    identifiers = tuple(value)
    if require_nonempty and not identifiers:
        raise DomainValidationError(code, f"{field_name} must not be empty")
    if any(
        type(identifier) is str
        and len(identifier) <= _MAX_TEXT_LENGTH
        and _SENSITIVE_VALUE_PATTERN.search(identifier)
        for identifier in identifiers
    ):
        raise DomainValidationError(
            code,
            f"{field_name} must not contain sensitive values",
        )
    if any(
        type(identifier) is not str
        or _STABLE_IDENTIFIER_PATTERN.fullmatch(identifier) is None
        for identifier in identifiers
    ):
        raise DomainValidationError(
            code,
            f"{field_name} entries must be lowercase ASCII identifiers",
        )
    if len(set(identifiers)) != len(identifiers):
        raise DomainValidationError(code, f"{field_name} must not contain duplicates")
    return identifiers  # type: ignore[return-value]

def _finite_number(
    value: object,
    *,
    code: str,
    field_name: str,
    non_negative: bool = False,
) -> float:
    """Convert only exact built-in numbers and contain every conversion failure."""

    message = f"{field_name} must be finite numeric"
    if non_negative:
        message = f"{field_name} must be finite non-negative numeric"
    if type(value) not in {int, float}:
        raise DomainValidationError(code, message)
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise DomainValidationError(code, message) from exc
    if not math.isfinite(normalized) or (non_negative and normalized < 0):
        raise DomainValidationError(code, message)
    return normalized

def _bounded_non_negative_integer(
    value: object,
    *,
    code: str,
    field_name: str,
    maximum: int,
) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise DomainValidationError(
            code,
            f"{field_name} must be a non-negative integer not exceeding {maximum}",
        )
    return value

def _canonical_pool_identity(
    value: object,
    *,
    code: str,
    field_name: str,
) -> str:
    """Apply the bounded RouteV1 identity rule to quota and billing pools."""

    try:
        normalized = _normalized_text(value, required=True, field_name=field_name)
    except DomainValidationError as exc:
        raise DomainValidationError(
            code,
            f"{field_name} must be a canonical pool identity",
        ) from exc
    assert normalized is not None
    return normalized

def _parse_exact_enum(
    value: object,
    enum_type: type[Enum],
    *,
    code: str,
    message: str,
) -> Enum:
    if type(value) is enum_type:
        return value
    if type(value) is not str:
        raise DomainValidationError(code, message)
    try:
        return enum_type(value)
    except ValueError as exc:
        raise DomainValidationError(code, message) from exc

def _validated_exact_label(
    value: object,
    allowed: frozenset[str],
    *,
    code: str,
    message: str,
) -> str:
    """Validate a public label before any hashing, lookup, or rank access."""

    if type(value) is not str or value not in allowed:
        raise DomainValidationError(code, message)
    return value

def _validated_verification_rank(
    value: object,
    *,
    code: str,
    message: str,
) -> tuple[str, int]:
    label = _validated_exact_label(
        value,
        _VERIFICATION_LABELS,
        code=code,
        message=message,
    )
    return label, _VERIFICATION_RANK[label]

def _normalized_policy_identifiers(
    value: object,
    field_name: str,
    known: frozenset[str],
) -> tuple[str, ...]:
    normalized = tuple(
        item.casefold()
        for item in _immutable_string_collection(value, field_name)
    )
    unknown = sorted(set(normalized) - known)
    if unknown:
        raise DomainValidationError(
            "task.unknown_policy_identifier",
            f"unknown {field_name}: {unknown}",
        )
    return normalized

def _reject_sensitive(
    value: object,
    location: str,
    *,
    allowed_field_paths: frozenset[tuple[str, ...]] = frozenset(),
    validate_complete_graph: bool = False,
) -> None:
    active: set[int] = set()
    visited: set[int] = set()
    nodes = 0

    def walk(item: object, path: tuple[str, ...] = (), depth: int = 0) -> None:
        nonlocal nodes
        key = path[-1].casefold() if path else ""
        if key in _PROHIBITED_FIELD_NAMES and path not in allowed_field_paths:
            raise DomainValidationError(
                "decision.sensitive_field_prohibited",
                f"sensitive field prohibited in {location}",
            )
        if depth > _MAX_GRAPH_DEPTH:
            raise DomainValidationError(
                "contract.payload_too_complex",
                f"payload nesting must not exceed {_MAX_GRAPH_DEPTH}",
            )
        nodes += 1
        if nodes > _MAX_GRAPH_NODES:
            raise DomainValidationError(
                "contract.payload_too_complex",
                f"payload must contain at most {_MAX_GRAPH_NODES} nodes",
            )

        container = (
            is_dataclass(item) and not isinstance(item, type)
        ) or isinstance(item, (Mapping, list, tuple, set, frozenset))
        if container:
            if (
                not validate_complete_graph
                and isinstance(item, Mapping)
                and len(item) > _MAX_MAPPING_FIELDS
            ):
                return
            if (
                not validate_complete_graph
                and isinstance(item, (list, tuple, set, frozenset))
                and len(item) > _MAX_TEXT_COLLECTION_ITEMS
            ):
                return
            if isinstance(item, Mapping) and len(item) > _MAX_GRAPH_NODES:
                raise DomainValidationError(
                    "contract.payload_too_complex",
                    f"payload must contain at most {_MAX_GRAPH_NODES} mapping entries",
                )
            if (
                isinstance(item, (list, tuple, set, frozenset))
                and len(item) > _MAX_GRAPH_NODES
            ):
                raise DomainValidationError(
                    "contract.payload_too_complex",
                    f"payload must contain at most {_MAX_GRAPH_NODES} collection entries",
                )
            identity = id(item)
            if identity in active:
                raise DomainValidationError(
                    "contract.payload_too_complex",
                    "payload must not contain cyclic references",
                )
            if identity in visited:
                return
            active.add(identity)
            try:
                if is_dataclass(item) and not isinstance(item, type):
                    for dataclass_field in fields(item):
                        walk(
                            getattr(item, dataclass_field.name),
                            path + (dataclass_field.name,),
                            depth + 1,
                        )
                elif isinstance(item, Mapping):
                    for child_key, child_value in item.items():
                        walk(child_key, path + ("<mapping-key>",), depth + 1)
                        value_path = (
                            child_key
                            if type(child_key) is str
                            else "<mapping-value>"
                        )
                        walk(child_value, path + (value_path,), depth + 1)
                elif isinstance(item, (list, tuple, set, frozenset)):
                    for child in item:
                        walk(child, path, depth + 1)
            finally:
                active.remove(identity)
            visited.add(identity)
        elif isinstance(item, (str, bytes, bytearray)):
            # Field validators own cardinality errors. Skipping regex work here
            # avoids scanning unbounded text while preserving their stable codes.
            if len(item) > _MAX_TEXT_LENGTH:
                return
            text = (
                item
                if isinstance(item, str)
                else bytes(item).decode("utf-8", errors="replace")
            )
            if _SENSITIVE_VALUE_PATTERN.search(text):
                raise DomainValidationError(
                    "decision.sensitive_field_prohibited",
                    f"sensitive value prohibited in {location}",
                )

    try:
        walk(value)
    except DomainValidationError:
        raise
    except Exception as exc:  # hostile container hook on a nested adapter object
        raise DomainValidationError(
            "contract.payload_too_complex",
            f"{location} could not be safely validated",
        ) from exc

def _safe_asdict(
    value: object,
    location: str,
    *,
    allowed_field_paths: frozenset[tuple[str, ...]] = frozenset(),
) -> dict[str, object]:
    """Serialize a trusted typed DTO only after bounded cycle-safe validation."""

    try:
        _reject_sensitive(
            value,
            location,
            allowed_field_paths=allowed_field_paths,
            validate_complete_graph=True,
        )
        payload = asdict(value)
    except DomainValidationError:
        raise
    except (RecursionError, RuntimeError, TypeError, ValueError) as exc:
        raise DomainValidationError(
            "contract.payload_too_complex",
            f"{location} cannot be safely rehydrated",
        ) from exc
    if not isinstance(payload, dict):
        raise DomainValidationError(
            "contract.payload_too_complex",
            f"{location} must serialize to a mapping",
        )
    return payload
