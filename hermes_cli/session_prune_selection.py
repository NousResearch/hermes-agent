"""Validate the private helper's frozen plan for exact native Prune."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_PLAN_FORMAT = "hermes-session-plan/v2"
# Per physical session, the subset of its exported session record that Prune rechecks: the same
# field names and values ``hermes sessions export`` emits, so a row that changed after the plan
# (new messages, reopened and re-ended) no longer matches and nothing is deleted.
SEGMENT_FIELDS = ("ended_at", "message_count", "last_activity_at")
_AUTHORITY_FIELDS = (
    "format",
    "source",
    "selection",
    "provenance",
    "candidates",
    "exclusions",
    "exclusion_predicates",
)


@dataclass(frozen=True)
class ExactPruneSelection:
    """Validated physical IDs and identity from one frozen helper plan."""

    session_ids: tuple[str, ...]
    plan_identity: str
    segments: Mapping[str, Mapping[str, Any]]


def _plan_identity(plan: dict[str, Any]) -> str:
    authority = {key: plan.get(key) for key in _AUTHORITY_FIELDS}
    encoded = json.dumps(
        authority,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _session_ids(plan: dict[str, Any]) -> tuple[str, ...]:
    selection = plan.get("selection")
    raw_ids = selection.get("physical_session_ids") if isinstance(selection, dict) else None
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or any(
            not isinstance(session_id, str) or not session_id or not session_id.isprintable()
            for session_id in raw_ids
        )
    ):
        raise ValueError(
            "selection.physical_session_ids must be a non-empty string list of printable IDs"
        )
    ids = tuple(raw_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("selection.physical_session_ids must not contain duplicates")

    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list")
    flattened: list[str] = []
    for candidate in candidates:
        candidate_ids = (
            candidate.get("physical_session_ids") if isinstance(candidate, dict) else None
        )
        if not isinstance(candidate_ids, list) or not candidate_ids or any(
            not isinstance(session_id, str) or not session_id for session_id in candidate_ids
        ):
            raise ValueError("candidate physical_session_ids must be non-empty string lists")
        flattened.extend(candidate_ids)
    if tuple(flattened) != ids:
        raise ValueError("candidate physical_session_ids do not match the frozen selection")
    return ids


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _segments(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each candidate's ``segments``: one exported-record subset per physical ID, in order."""
    segments: dict[str, dict[str, Any]] = {}
    for candidate in plan["candidates"]:
        records = candidate.get("segments")
        if not isinstance(records, list) or [
            record.get("id") if isinstance(record, dict) else None for record in records
        ] != candidate["physical_session_ids"]:
            raise ValueError("candidate segments must list each physical session ID in order")
        for record in records:
            ended_at, count, activity = (record.get(field) for field in SEGMENT_FIELDS)
            if (
                any(field not in record for field in SEGMENT_FIELDS)
                or not _is_number(ended_at)
                or not isinstance(count, int) or isinstance(count, bool) or count < 0
                or not (activity is None or _is_number(activity))
            ):
                raise ValueError(
                    f"segment {record['id']!r} needs numeric ended_at, a non-negative "
                    "message_count, and a numeric or null last_activity_at"
                )
            segments[record["id"]] = {field: record[field] for field in SEGMENT_FIELDS}
    return segments


def load_exact_prune_selection(
    plan_file: Path,
    *,
    expected_database: Path,
) -> ExactPruneSelection:
    """Load and validate a helper v2 plan bound to the active session database."""
    try:
        plan = json.loads(Path(plan_file).expanduser().read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read exact-selection plan: {exc}") from exc
    if not isinstance(plan, dict) or plan.get("format") != _PLAN_FORMAT:
        raise ValueError(f"exact-selection plan format must be {_PLAN_FORMAT!r}")

    source = plan.get("source")
    if not isinstance(source, dict):
        raise ValueError("exact-selection plan source must be an object")
    source_database = source.get("database")
    if not isinstance(source_database, str) or not source_database:
        raise ValueError("exact-selection plan source.database must be a non-empty path")
    if source.get("changed_during_read") is not False:
        raise ValueError("exact-selection plan reports source drift")
    try:
        same_database = Path(source_database).expanduser().samefile(expected_database)
    except OSError as exc:
        raise ValueError(f"could not verify exact-selection source database: {exc}") from exc
    if not same_database:
        raise ValueError("exact-selection plan names a different session database")

    provenance = plan.get("provenance")
    provenance_fields = (
        "helper_revision",
        "native_revision",
        "installed_command_provenance",
    )
    if not isinstance(provenance, dict) or any(
        not isinstance(provenance.get(field), str) or not provenance[field].strip()
        for field in provenance_fields
    ):
        raise ValueError("exact-selection plan provenance fields must be non-empty strings")

    supplied_identity = plan.get("plan_identity")
    # The identity is not a secret, so a plain comparison suffices (and never raises on
    # non-ASCII input the way hmac.compare_digest does).
    if not isinstance(supplied_identity, str) or supplied_identity != _plan_identity(plan):
        raise ValueError("exact-selection plan identity does not match its contents")

    # ``_segments`` relies on the candidate shape ``_session_ids`` validates, so keep this order.
    session_ids = _session_ids(plan)
    segments = _segments(plan)
    return ExactPruneSelection(
        session_ids=session_ids,
        plan_identity=supplied_identity,
        segments=segments,
    )
