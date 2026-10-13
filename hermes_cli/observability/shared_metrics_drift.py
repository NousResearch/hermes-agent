"""Rows an earlier build or catalog stored whose values the current contract no longer lists.

Packaging re-checks every pending row against the current contract, so a value that left a closed
set after it was recorded (a catalog entry removed, a built-in tool moved to a plugin) is conformed
here instead of aborting the whole period's package (#133017).
"""

from __future__ import annotations

from typing import Any

from .shared_metrics_contract import (
    _COUNTER_DIMENSION_VALUES,
    _IDENTIFIER_FIELDS,
    _LEGACY_METRIC_FIELDS,
    _METRIC_FIELDS,
    _metric_identifier,
)


def counter_dimensions_shape_is_known(metric_name: str, dimensions: dict[str, Any]) -> bool:
    """Whether the row's field set matches a current or legacy contract shape (values aside).

    A row that fails this is structurally unknown — that is tamper evidence, not drift, and
    packaging keeps failing closed on it (see test_package_builder_rejects_tampered_dimensions).
    """
    fields = frozenset(dimensions)
    return fields == _METRIC_FIELDS.get(metric_name) or fields in _LEGACY_METRIC_FIELDS.get(metric_name, ())


def conform_counter_dimensions(metric_name: str, dimensions: dict[str, Any]) -> dict[str, Any] | None:
    """Re-bucket out-of-contract enum values into the field's closed fallback, else ``None``.

    Catalog-backed enums are repo data: a value that was a catalog entry when the row was
    recorded (and passed record-time validation) can legitimately fall out of the catalog by
    packaging time — the recording and packaging processes snapshot it independently (#133017).
    Exporting the stale value would leak exactly the out-of-enum payload the closed enum exists
    to prevent, so it is bucketed to ``custom``/``other``/``unknown`` (the first the field's enum
    defines), mirroring what the record-side normalizers do for names that were never in the
    catalog. Code constants drift too: a built-in tool moved to a plugin (Spotify, Home Assistant)
    or a removed reason value leaves rows an earlier build stored.
    Returns ``None`` when a value cannot be safely re-bucketed: fields whose enum defines no
    fallback bucket, non-string values, or identifier fields that no longer round-trip through
    ``_metric_identifier`` (record-time validation already guarantees they did once).
    """
    conformed = dict(dimensions)
    identifiers = _IDENTIFIER_FIELDS.get(metric_name, {})
    contract = _COUNTER_DIMENSION_VALUES[metric_name]
    for field, value in conformed.items():
        if not isinstance(value, str):
            return None
        if field in identifiers:
            if value != _metric_identifier(value, max_length=identifiers[field]):
                return None
            continue
        allowed = contract[field]
        if value in allowed:
            continue
        fallback = next((bucket for bucket in ("custom", "other", "unknown") if bucket in allowed), None)
        if fallback is None:
            return None
        conformed[field] = fallback
    return conformed
