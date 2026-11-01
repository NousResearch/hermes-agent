"""Deterministic protected-state contracts.

This module is deliberately pure: it defines immutable values and strict JSON
validation only.  It does not access sessions, storage, providers, gateways,
or the compression runtime.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Mapping


FACT_SCHEMA_VERSION = "protected-fact-v1"
SUPERSESSION_SCHEMA_VERSION = "protected-supersession-v1"
BLOCK_SCHEMA_VERSION = "protected-block-v1"

_FACT_ID_RE = re.compile(r"^pf1_[0-9a-f]{64}$")


class ContractValidationError(ValueError):
    """Raised when a protected-state value is not a valid contract."""


class CaptureStatus(str, Enum):
    """Whether a fact contains a captured value or only a source pointer."""

    CAPTURED = "CAPTURED"
    POINTER_ONLY = "POINTER_ONLY"


def _json_value(value: Any, *, field: str = "value") -> Any:
    """Validate and return a plain JSON-compatible value."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractValidationError(f"{field} object keys must be strings")
            result[key] = _json_value(item, field=f"{field}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, field=f"{field}[]") for item in value]
    raise ContractValidationError(f"{field} is not a JSON value")


def _freeze_json(value: Any) -> Any:
    """Copy JSON data into immutable containers without changing its meaning."""
    plain = _json_value(value)
    if isinstance(plain, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in plain.items()})
    if isinstance(plain, list):
        return tuple(_freeze_json(item) for item in plain)
    return plain


def canonical_json(value: Any) -> str:
    """Return the deterministic JSON representation used by these contracts."""
    try:
        plain = _json_value(value)
        return json.dumps(
            plain,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except ContractValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"invalid JSON value: {exc}") from exc


def parse_canonical_json(text: str) -> Any:
    """Parse JSON strictly, rejecting duplicate keys and non-finite constants."""
    if not isinstance(text, str):
        raise ContractValidationError("JSON input must be text")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractValidationError(f"duplicate key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ContractValidationError(f"non-finite constant: {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except ContractValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"invalid JSON: {exc}") from exc


def sha256_hex(value: Any) -> str:
    """Hash the UTF-8 bytes of :func:`canonical_json`."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _strict_object(data: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ContractValidationError(f"{name} must be an object")
    unknown = set(data) - fields
    if unknown:
        raise ContractValidationError(f"unknown field in {name}: {sorted(unknown)}")
    return data


def _require(data: Mapping[str, Any], fields: set[str], name: str) -> None:
    missing = fields - set(data)
    if missing:
        raise ContractValidationError(f"missing field in {name}: {sorted(missing)}")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractValidationError(f"{field} must be a non-empty string")
    return value


def _enum(enum_type: type[Enum], value: Any, field: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"invalid {field}: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class ProvenancePointer:
    """Stable pointer to the runtime record from which a fact came."""

    session_id: str
    message_id: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    parent_session_id: str | None = None

    _FIELDS: ClassVar[set[str]] = {
        "session_id",
        "message_id",
        "tool_call_id",
        "tool_name",
        "parent_session_id",
    }

    def __post_init__(self) -> None:
        _text(self.session_id, "provenance.session_id")
        _text(self.message_id, "provenance.message_id")
        for field in ("tool_call_id", "tool_name", "parent_session_id"):
            value = getattr(self, field)
            if value is not None:
                _text(value, f"provenance.{field}")

    @classmethod
    def from_dict(cls, data: Any) -> "ProvenancePointer":
        data = _strict_object(data, cls._FIELDS, "provenance")
        _require(data, {"session_id", "message_id"}, "provenance")
        values: dict[str, str | None] = {}
        for field in cls._FIELDS:
            value = data.get(field)
            values[field] = None if value is None else _text(value, f"provenance.{field}")
        return cls(**values)

    def to_dict(self) -> dict[str, str]:
        result = {"session_id": self.session_id, "message_id": self.message_id}
        for field in ("tool_call_id", "tool_name", "parent_session_id"):
            value = getattr(self, field)
            if value is not None:
                result[field] = value
        return result


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Identity of the structured source that asserted a protected fact."""

    source_type: str
    source_id: str

    _FIELDS: ClassVar[set[str]] = {"source_type", "source_id"}

    def __post_init__(self) -> None:
        _text(self.source_type, "source_identity.source_type")
        _text(self.source_id, "source_identity.source_id")

    @classmethod
    def from_dict(cls, data: Any) -> "SourceIdentity":
        data = _strict_object(data, cls._FIELDS, "source_identity")
        _require(data, cls._FIELDS, "source_identity")
        return cls(
            _text(data["source_type"], "source_identity.source_type"),
            _text(data["source_id"], "source_identity.source_id"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"source_type": self.source_type, "source_id": self.source_id}


@dataclass(frozen=True, slots=True)
class ProtectedFact:
    """An immutable, provenance-bound fact eligible for protected storage.

    ``fact_id`` is a deterministic identity over the full canonical fact,
    including its provenance and source identity.  Therefore the same value
    captured from different session/message provenance intentionally receives
    different IDs; this is provenance-scoped identity, not hash instability.
    """

    fact_kind: str
    capture_status: CaptureStatus
    value: Any
    provenance: ProvenancePointer
    source_identity: SourceIdentity
    schema_version: str = FACT_SCHEMA_VERSION

    _FIELDS: ClassVar[set[str]] = {
        "schema_version",
        "fact_kind",
        "capture_status",
        "value",
        "provenance",
        "source_identity",
    }

    def __post_init__(self) -> None:
        if self.schema_version != FACT_SCHEMA_VERSION:
            raise ContractValidationError("unsupported ProtectedFact schema_version")
        _text(self.fact_kind, "fact_kind")
        if not isinstance(self.capture_status, CaptureStatus):
            raise ContractValidationError("capture_status must be a CaptureStatus")
        if not isinstance(self.provenance, ProvenancePointer):
            raise ContractValidationError("provenance must be a ProvenancePointer")
        if not isinstance(self.source_identity, SourceIdentity):
            raise ContractValidationError("source_identity must be a SourceIdentity")
        object.__setattr__(self, "value", _freeze_json(self.value))

    @classmethod
    def from_dict(cls, data: Any) -> "ProtectedFact":
        data = _strict_object(data, cls._FIELDS, "ProtectedFact")
        _require(data, cls._FIELDS, "ProtectedFact")
        if data["schema_version"] != FACT_SCHEMA_VERSION:
            raise ContractValidationError("unsupported ProtectedFact schema_version")
        return cls(
            _text(data["fact_kind"], "fact_kind"),
            _enum(CaptureStatus, data["capture_status"], "capture_status"),
            _json_value(data["value"], field="ProtectedFact.value"),
            ProvenancePointer.from_dict(data["provenance"]),
            SourceIdentity.from_dict(data["source_identity"]),
        )

    @property
    def fact_id(self) -> str:
        return "pf1_" + sha256_hex(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fact_kind": self.fact_kind,
            "capture_status": self.capture_status.value,
            "value": _json_value(self.value, field="ProtectedFact.value"),
            "provenance": self.provenance.to_dict(),
            "source_identity": self.source_identity.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class Supersession:
    """Explicit evidence that a new fact replaces an older fact."""

    old_fact_id: str
    new_fact_id: str
    new_provenance: ProvenancePointer
    authority_ref: str
    ordering: int
    schema_version: str = SUPERSESSION_SCHEMA_VERSION

    _FIELDS: ClassVar[set[str]] = {
        "schema_version",
        "old_fact_id",
        "new_fact_id",
        "new_provenance",
        "authority_ref",
        "ordering",
    }

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.schema_version != SUPERSESSION_SCHEMA_VERSION:
            raise ContractValidationError("unsupported Supersession schema_version")
        for field in ("old_fact_id", "new_fact_id"):
            value = _text(getattr(self, field), field)
            if not _FACT_ID_RE.fullmatch(value):
                raise ContractValidationError(f"invalid {field}")
        if self.old_fact_id == self.new_fact_id:
            raise ContractValidationError("supersession cannot target itself")
        if not isinstance(self.new_provenance, ProvenancePointer):
            raise ContractValidationError("new_provenance must be a ProvenancePointer")
        _text(self.authority_ref, "authority_ref")
        if (
            isinstance(self.ordering, bool)
            or not isinstance(self.ordering, int)
            or self.ordering < 1
        ):
            raise ContractValidationError("ordering must be a positive integer")

    @classmethod
    def from_dict(cls, data: Any) -> "Supersession":
        data = _strict_object(data, cls._FIELDS, "Supersession")
        _require(data, cls._FIELDS, "Supersession")
        return cls(
            data["old_fact_id"],
            data["new_fact_id"],
            ProvenancePointer.from_dict(data["new_provenance"]),
            data["authority_ref"],
            data["ordering"],
            schema_version=data["schema_version"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "old_fact_id": self.old_fact_id,
            "new_fact_id": self.new_fact_id,
            "new_provenance": self.new_provenance.to_dict(),
            "authority_ref": self.authority_ref,
            "ordering": self.ordering,
        }


@dataclass(frozen=True, slots=True)
class ProtectedBlock:
    """A deterministic collection of protected facts and transitions."""

    facts: Sequence[ProtectedFact]
    supersessions: Sequence[Supersession] = ()
    schema_version: str = BLOCK_SCHEMA_VERSION

    _FIELDS: ClassVar[set[str]] = {"schema_version", "facts", "supersessions"}

    def __post_init__(self) -> None:
        if self.schema_version != BLOCK_SCHEMA_VERSION:
            raise ContractValidationError("unsupported ProtectedBlock schema_version")
        if isinstance(self.facts, (str, bytes, bytearray)) or not isinstance(
            self.facts, Sequence
        ):
            raise ContractValidationError("ProtectedBlock.facts must be a sequence")
        if isinstance(self.supersessions, (str, bytes, bytearray)) or not isinstance(
            self.supersessions, Sequence
        ):
            raise ContractValidationError("ProtectedBlock.supersessions must be a sequence")
        object.__setattr__(self, "facts", tuple(self.facts))
        object.__setattr__(self, "supersessions", tuple(self.supersessions))
        if not all(isinstance(fact, ProtectedFact) for fact in self.facts):
            raise ContractValidationError("ProtectedBlock.facts contains an invalid fact")
        fact_ids = [fact.fact_id for fact in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ContractValidationError("duplicate fact in ProtectedBlock")
        if not all(
            isinstance(item, Supersession) for item in self.supersessions
        ):
            raise ContractValidationError(
                "ProtectedBlock.supersessions contains an invalid transition"
            )
        fact_id_set = set(fact_ids)
        seen_transitions: set[tuple[str, str, int]] = set()
        for item in self.supersessions:
            key = (item.old_fact_id, item.new_fact_id, item.ordering)
            if key in seen_transitions:
                raise ContractValidationError("duplicate supersession in ProtectedBlock")
            seen_transitions.add(key)
            if item.new_fact_id not in fact_id_set:
                raise ContractValidationError(
                    "supersession target must be present in ProtectedBlock"
                )

    @classmethod
    def from_dict(cls, data: Any) -> "ProtectedBlock":
        data = _strict_object(data, cls._FIELDS, "ProtectedBlock")
        _require(data, cls._FIELDS, "ProtectedBlock")
        if data["schema_version"] != BLOCK_SCHEMA_VERSION:
            raise ContractValidationError("unsupported ProtectedBlock schema_version")
        if not isinstance(data["facts"], list):
            raise ContractValidationError("ProtectedBlock.facts must be an array")
        if not isinstance(data["supersessions"], list):
            raise ContractValidationError("ProtectedBlock.supersessions must be an array")

        facts = tuple(ProtectedFact.from_dict(item) for item in data["facts"])
        fact_ids = [fact.fact_id for fact in facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ContractValidationError("duplicate fact in ProtectedBlock")
        supersessions = tuple(
            Supersession.from_dict(item) for item in data["supersessions"]
        )
        seen_transitions: set[tuple[str, str, int]] = set()
        fact_id_set = set(fact_ids)
        for item in supersessions:
            key = (item.old_fact_id, item.new_fact_id, item.ordering)
            if key in seen_transitions:
                raise ContractValidationError("duplicate supersession in ProtectedBlock")
            seen_transitions.add(key)
            if item.new_fact_id not in fact_id_set:
                raise ContractValidationError(
                    "supersession target must be present in ProtectedBlock"
                )
        return cls(facts, supersessions)

    @property
    def block_id(self) -> str:
        return "pb1_" + sha256_hex(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        facts = sorted((fact.to_dict() for fact in self.facts), key=lambda item: sha256_hex(item))
        supersessions = sorted(
            (item.to_dict() for item in self.supersessions),
            key=lambda item: (item["old_fact_id"], item["new_fact_id"], item["ordering"]),
        )
        return {
            "schema_version": self.schema_version,
            "facts": facts,
            "supersessions": supersessions,
        }


__all__ = [
    "BLOCK_SCHEMA_VERSION",
    "FACT_SCHEMA_VERSION",
    "SUPERSESSION_SCHEMA_VERSION",
    "CaptureStatus",
    "ContractValidationError",
    "ProtectedBlock",
    "ProtectedFact",
    "ProvenancePointer",
    "SourceIdentity",
    "Supersession",
    "canonical_json",
    "parse_canonical_json",
    "sha256_hex",
]
