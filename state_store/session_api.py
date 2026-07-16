"""Backend-neutral SessionDB API contracts and value normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
from types import MappingProxyType
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Sequence


@dataclass(frozen=True)
class APISessionMutationResult:
    """Outcome of one atomic API-facing session mutation."""

    outcome: Literal[
        "created",
        "source_missing",
        "destination_exists",
        "invalid_title",
    ]
    session: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class APISessionMutationAbort(Exception):
    """Rollback marker for an expected API mutation outcome."""

    def __init__(self, result: APISessionMutationResult) -> None:
        self.result = result


@dataclass(frozen=True)
class SessionAPISignatureManifest:
    """Frozen public method signatures for one SessionDB implementation."""

    signatures: Mapping[str, str]


def session_api_signature_manifest(api_type: type[Any]) -> SessionAPISignatureManifest:
    """Capture a frozen snapshot of public callable signatures for one API."""

    signatures: dict[str, str] = {}
    for name, member in inspect.getmembers(api_type):
        if name.startswith("_") or not callable(member):
            continue
        try:
            signatures[name] = str(inspect.signature(member))
        except (TypeError, ValueError):
            # Built-in or extension members do not form part of this Python API.
            continue
    return SessionAPISignatureManifest(MappingProxyType(signatures))


def normalize_row(
    row: Any,
    *,
    columns: Sequence[str] = (),
) -> Optional[Dict[str, Any]]:
    """Copy a driver row into a backend-neutral dictionary."""

    if row is None:
        return None
    if isinstance(row, Mapping):
        return dict(row)
    mapping = getattr(row, "_mapping", None)
    if isinstance(mapping, Mapping):
        return dict(mapping)
    if not columns:
        raise TypeError("tuple rows require cursor column metadata")
    return dict(zip(columns, row))


def normalize_rows(
    rows: Iterable[Any],
    *,
    columns: Sequence[str] = (),
    limit: int,
) -> list[Dict[str, Any]]:
    """Normalize a bounded result set without leaking driver cursors."""

    if limit <= 0:
        raise ValueError("row normalization limit must be positive")
    result: list[Dict[str, Any]] = []
    for row in rows:
        if len(result) >= limit:
            raise ValueError("query returned more rows than its bounded limit")
        normalized = normalize_row(row, columns=columns)
        if normalized is not None:
            result.append(normalized)
    return result


def json_dumps(value: Any) -> str:
    """Encode durable JSON with deterministic compact formatting."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def json_loads(value: Optional[str], *, default: Any = None) -> Any:
    """Decode a nullable durable JSON field without backend-specific behavior."""

    if value is None:
        return default
    return json.loads(value)
