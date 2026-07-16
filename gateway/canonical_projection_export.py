"""Mechanical contract for writer-owned Canonical projection exports.

The privileged writer publishes this derived snapshot for isolated projectors
and root-owned canary verification.  Meaning remains model-authored; this
module checks only exact envelopes and the one-to-one database provenance join.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any


PROJECTION_EXPORT_SCHEMA = "canonical-writer-projection-export.v2"
PROJECTION_EXPORT_KEYS = frozenset({"schema", "events", "provenance"})
PROJECTION_PROVENANCE_KEYS = frozenset(
    {
        "event_id",
        "canonical_content_sha256",
        "origin",
        "trusted_runtime",
        "appended_at",
    }
)
TRUSTED_RUNTIME_KEYS = frozenset(
    {
        "request_id",
        "platform",
        "session_key_sha256",
        "capability_epoch_sha256",
        "user_id",
        "chat_id",
        "thread_id",
        "message_id",
        "owner_authenticated",
        "service_internal",
    }
)
_TRUSTED_RUNTIME_TEXT_KEYS = frozenset(
    {
        "request_id",
        "platform",
        "session_key_sha256",
        "capability_epoch_sha256",
        "user_id",
        "chat_id",
        "thread_id",
        "message_id",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ORIGIN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")


class ProjectionExportError(ValueError):
    """The writer-owned projection artifact failed its exact v2 contract."""


def canonical_json_sha256(value: Any) -> str:
    """Hash one JSON-compatible value using the artifact's stable encoding."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ProjectionExportError("projection_export_noncanonical_json") from exc
    return hashlib.sha256(payload).hexdigest()


def _canonical_event_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ProjectionExportError("projection_export_event_id_invalid")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ProjectionExportError("projection_export_event_id_invalid") from exc
    if parsed.int == 0 or str(parsed) != value:
        raise ProjectionExportError("projection_export_event_id_invalid")
    return value


def _validate_trusted_runtime(value: Any) -> dict[str, Any]:
    """Validate the exact secret-free runtime envelope admitted by SQL."""

    if (
        not isinstance(value, Mapping)
        or "request_id" not in value
        or not set(value) <= TRUSTED_RUNTIME_KEYS
    ):
        raise ProjectionExportError("projection_export_provenance_runtime_invalid")
    for key in _TRUSTED_RUNTIME_TEXT_KEYS:
        if key not in value:
            continue
        item = value[key]
        if not isinstance(item, str) or len(item) > 240:
            raise ProjectionExportError(
                "projection_export_provenance_runtime_invalid"
            )
        if key == "request_id" and not item:
            raise ProjectionExportError(
                "projection_export_provenance_runtime_invalid"
            )
        if key in {"session_key_sha256", "capability_epoch_sha256"} and (
            item and _SHA256_RE.fullmatch(item) is None
        ):
            raise ProjectionExportError(
                "projection_export_provenance_runtime_invalid"
            )
    for key in ("owner_authenticated", "service_internal"):
        if key in value and type(value[key]) is not bool:
            raise ProjectionExportError(
                "projection_export_provenance_runtime_invalid"
            )
    return dict(value)


def validate_projection_pair(
    event: Any,
    provenance: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one exact event/provenance row pair without interpreting it."""

    if not isinstance(event, Mapping):
        raise ProjectionExportError("projection_export_event_invalid")
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != PROJECTION_PROVENANCE_KEYS
    ):
        raise ProjectionExportError("projection_export_provenance_invalid")

    event_id = _canonical_event_id(event.get("event_id"))
    if _canonical_event_id(provenance.get("event_id")) != event_id:
        raise ProjectionExportError("projection_export_provenance_event_mismatch")

    content_sha256 = provenance.get("canonical_content_sha256")
    payload = event.get("payload")
    if (
        not isinstance(content_sha256, str)
        or _SHA256_RE.fullmatch(content_sha256) is None
        or not isinstance(payload, Mapping)
        or payload.get("canonical_content_sha256") != content_sha256
    ):
        raise ProjectionExportError("projection_export_provenance_content_mismatch")

    trusted_runtime = _validate_trusted_runtime(provenance.get("trusted_runtime"))
    source = event.get("source")
    if (
        not isinstance(source, Mapping)
        or not isinstance(source.get("observed_session"), Mapping)
        or dict(source["observed_session"]) != dict(trusted_runtime)
    ):
        raise ProjectionExportError("projection_export_provenance_runtime_mismatch")

    origin = provenance.get("origin")
    decision = event.get("decision")
    if (
        not isinstance(origin, str)
        or _ORIGIN_RE.fullmatch(origin) is None
        or not isinstance(decision, Mapping)
        or decision.get("decided_by") != origin
    ):
        raise ProjectionExportError("projection_export_provenance_origin_mismatch")

    appended_at = provenance.get("appended_at")
    if (
        not isinstance(appended_at, str)
        or not appended_at
        or len(appended_at.encode("utf-8", errors="strict")) > 128
        or event.get("occurred_at") != appended_at
    ):
        raise ProjectionExportError("projection_export_provenance_time_mismatch")

    return dict(event), dict(provenance)


def validate_projection_rows(
    events: Any,
    provenance: Any,
    *,
    maximum_events: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate parallel ordered rows and reject duplicate event identities."""

    if (
        type(maximum_events) is not int
        or not 0 <= maximum_events <= 1_000_000
        or not isinstance(events, list)
        or not isinstance(provenance, list)
        or len(events) != len(provenance)
        or len(events) > maximum_events
    ):
        raise ProjectionExportError("projection_export_rows_invalid")

    normalized_events: list[dict[str, Any]] = []
    normalized_provenance: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event, proof in zip(events, provenance, strict=True):
        normalized_event, normalized_proof = validate_projection_pair(event, proof)
        event_id = normalized_event["event_id"]
        if event_id in seen:
            raise ProjectionExportError("projection_export_event_duplicate")
        seen.add(event_id)
        normalized_events.append(normalized_event)
        normalized_provenance.append(normalized_proof)
    return normalized_events, normalized_provenance


def validate_projection_export(
    value: Any,
    *,
    maximum_events: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the exact versioned artifact envelope and joined row arrays."""

    if (
        not isinstance(value, Mapping)
        or set(value) != PROJECTION_EXPORT_KEYS
        or value.get("schema") != PROJECTION_EXPORT_SCHEMA
    ):
        raise ProjectionExportError("projection_export_envelope_invalid")
    return validate_projection_rows(
        value.get("events"),
        value.get("provenance"),
        maximum_events=maximum_events,
    )


def projection_provenance_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Bind the ordered provenance projection into an activation receipt."""

    return canonical_json_sha256([dict(row) for row in rows])


__all__ = [
    "PROJECTION_EXPORT_KEYS",
    "PROJECTION_EXPORT_SCHEMA",
    "PROJECTION_PROVENANCE_KEYS",
    "TRUSTED_RUNTIME_KEYS",
    "ProjectionExportError",
    "canonical_json_sha256",
    "projection_provenance_sha256",
    "validate_projection_export",
    "validate_projection_pair",
    "validate_projection_rows",
]
