"""Validated, redacted and idempotent failure-envelope persistence.

HOF-012B is a narrow layer over the frozen HOF-002
``project_failure_envelopes`` table.  It accepts only classification metadata,
redacts error text before it is used for persistence or fingerprinting, and
makes the project/task/run identity a stable exact-event key.

A run id distinguishes occurrences for the same task.  If ``run_id`` is
``None``, the task identity can have only one exact envelope: a later envelope
with different fields is rejected as a conflicting identity rather than being
silently treated as a new occurrence.  This preserves the frozen schema and
makes callers supply a run id when they need distinct occurrences.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from typing import Any, Mapping

from hermes_cli.project_finalization_contract import (
    ProjectFailureEnvelope,
    validate_generation,
)
from hermes_cli.sqlite_util import write_txn


FAILURE_CLASSES: tuple[str, ...] = (
    "provider_auth",
    "provider_quota",
    "provider_rate_limit",
    "provider_timeout",
    "process_crash",
    "protocol_violation",
    "task_timeout",
    "iteration_budget",
    "notification_failure",
    "artifact_failure",
    "unknown",
)

RECOGNIZED_KWARGS = frozenset(
    {
        "run_id",
        "provider",
        "model",
        "failure_class",
        "status_code",
        "retry_after",
        "error_fingerprint",
    }
)

# These are intentionally bounded.  Failure persistence is not a log sink and
# must not become a way to retain prompts, payloads, or unbounded provider text.
MAX_REDACTED_ERROR_LENGTH = 2048

_EVENT_IDENTITY_FIELDS = (
    "board_id",
    "root_task_id",
    "generation",
    "task_id",
    "run_id",
)
_PAYLOAD_FIELDS = (
    "board_id",
    "root_task_id",
    "generation",
    "task_id",
    "run_id",
    "provider",
    "model",
    "failure_class",
    "status_code",
    "retry_after",
    "redacted_error",
    "error_fingerprint",
)
_FINGERPRINT_FIELDS = (
    "provider",
    "model",
    "failure_class",
    "status_code",
    "retry_after",
    "redacted_error",
)

# URL values are replaced in their entirety.  A path can contain a signed
# token even when its query string is empty.
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_AUTH_RE = re.compile(
    r"\b(?:authorization|proxy-authorization)\s*[:=]\s*"
    r"(?:bearer|basic)\s+[^\s,;]+",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE)
_SECRET_FIELD_RE = re.compile(
    r"(?P<prefix>(?:\"|')?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"id[_-]?token|oauth(?:[_-]?(?:token|secret))?|client[_-]?secret|"
    r"authorization|proxy[_-]?authorization|password|passwd|secret|credential|"
    r"request[_-]?body|response[_-]?body|prompt|completion|chat|message|payload)"
    r"(?:\"|')?\s*[:=]\s*)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;}]+)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|pk-[A-Za-z0-9_-]{8,}|"
    r"(?:sk|pk|rk|xox[baprs]-|gh[pousr]_|github_pat_|hch-at-|ya29\.)"
    r"[A-Za-z0-9._~+/=-]{8,}|eyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})\b"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
    re.IGNORECASE | re.DOTALL,
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE_RE = re.compile(r"[ \t\r\n]+")
_SENSITIVE_PREFIX_RE = re.compile(
    r"^\s*(?:request[ _-]?body|response[ _-]?body|payload|prompt|completion|"
    r"chat|message)\s*[:=]",
    re.IGNORECASE,
)


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain NUL")
    return value


def _optional_text(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _require_text(name, value)


def _optional_int(name: str, value: object, *, minimum: int = 0) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or None")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def redact_error(error: str | None) -> str | None:
    """Return bounded error text with secret-bearing material removed.

    Structured JSON is treated as a provider payload, not as an error message,
    and is replaced wholesale.  For ordinary text, URLs, credentials, auth
    headers, secret fields, token-shaped values, and private keys are removed.
    """
    if error is None:
        return None
    if not isinstance(error, str):
        raise TypeError("redacted_error must be a string or None")
    value = error.strip()
    if not value:
        return None

    # A complete JSON value is overwhelmingly likely to be a request/response
    # body.  Do not attempt to curate arbitrary provider payload fields.
    try:
        json.loads(value)
        structured = True
    except (json.JSONDecodeError, TypeError):
        structured = False
    if structured or _SENSITIVE_PREFIX_RE.search(value):
        return "[structured error redacted]"

    value = _PRIVATE_KEY_RE.sub("[private key redacted]", value)
    value = _URL_RE.sub("[url redacted]", value)
    value = _AUTH_RE.sub("[authorization redacted]", value)
    value = _BEARER_RE.sub("[token redacted]", value)
    value = _SECRET_FIELD_RE.sub("[secret field redacted]", value)
    value = _TOKEN_RE.sub("[token redacted]", value)
    value = _CONTROL_RE.sub(" ", value)
    value = _SPACE_RE.sub(" ", value).strip()
    if len(value) > MAX_REDACTED_ERROR_LENGTH:
        value = value[:MAX_REDACTED_ERROR_LENGTH].rstrip() + "…"
    return value or None


def _canonical_payload(payload: Mapping[str, object]) -> str:
    return json.dumps(
        {field: payload[field] for field in _FINGERPRINT_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def compute_error_fingerprint(payload: Mapping[str, object]) -> str:
    """Hash normalized, already-redacted envelope data.

    The supplied mapping must contain the fields used by the persistence
    boundary.  The fingerprint itself is excluded to avoid self-reference.
    """
    canonical = _canonical_payload(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _row_value(row: sqlite3.Row | tuple[Any, ...], name: str) -> Any:
    if isinstance(row, sqlite3.Row):
        return row[name]
    # SELECT * order is frozen by HOF-002.  This fallback keeps the wrapper
    # usable with callers that did not set sqlite3.Row as their row factory.
    index = {
        "id": 0,
        "board_id": 1,
        "root_task_id": 2,
        "generation": 3,
        "task_id": 4,
        "run_id": 5,
        "provider": 6,
        "model": 7,
        "failure_class": 8,
        "status_code": 9,
        "retry_after": 10,
        "redacted_error": 11,
        "error_fingerprint": 12,
        "created_at": 13,
    }[name]
    return row[index]


def _row_to_envelope(row: sqlite3.Row | tuple[Any, ...]) -> ProjectFailureEnvelope:
    return ProjectFailureEnvelope(
        id=_row_value(row, "id"),
        board_id=_row_value(row, "board_id"),
        root_task_id=_row_value(row, "root_task_id"),
        generation=int(_row_value(row, "generation")),
        task_id=_row_value(row, "task_id"),
        run_id=(int(_row_value(row, "run_id")) if _row_value(row, "run_id") is not None else None),
        provider=_row_value(row, "provider"),
        model=_row_value(row, "model"),
        failure_class=_row_value(row, "failure_class"),
        status_code=(int(_row_value(row, "status_code")) if _row_value(row, "status_code") is not None else None),
        retry_after=(int(_row_value(row, "retry_after")) if _row_value(row, "retry_after") is not None else None),
        redacted_error=_row_value(row, "redacted_error"),
        error_fingerprint=_row_value(row, "error_fingerprint"),
        created_at=int(_row_value(row, "created_at")),
    )


def _validate_and_prepare(
    *,
    board_id: object,
    root_task_id: object,
    generation: object,
    task_id: object,
    redacted_error: object,
    kwargs: Mapping[str, object],
) -> dict[str, object]:
    unknown = set(kwargs) - RECOGNIZED_KWARGS
    if unknown:
        raise TypeError(f"unrecognized failure envelope fields: {sorted(unknown)!r}")
    if not isinstance(generation, int) or isinstance(generation, bool):
        raise TypeError("generation must be an integer")
    validate_generation(generation)

    failure_class = kwargs.get("failure_class")
    if not isinstance(failure_class, str) or failure_class not in FAILURE_CLASSES:
        raise ValueError(f"failure_class must be one of {FAILURE_CLASSES!r}")

    payload: dict[str, object] = {
        "board_id": _require_text("board_id", board_id),
        "root_task_id": _require_text("root_task_id", root_task_id),
        "generation": generation,
        "task_id": _require_text("task_id", task_id),
        "run_id": _optional_int("run_id", kwargs.get("run_id")),
        "provider": _optional_text("provider", kwargs.get("provider")),
        "model": _optional_text("model", kwargs.get("model")),
        "failure_class": failure_class,
        "status_code": _optional_int("status_code", kwargs.get("status_code")),
        "retry_after": _optional_int("retry_after", kwargs.get("retry_after")),
        "redacted_error": redact_error(redacted_error),
    }
    fingerprint = compute_error_fingerprint(payload)
    supplied_fingerprint = kwargs.get("error_fingerprint")
    if supplied_fingerprint is not None:
        if not isinstance(supplied_fingerprint, str) or not re.fullmatch(r"[a-f0-9]{64}", supplied_fingerprint):
            raise ValueError("error_fingerprint must be 64 lowercase hexadecimal characters")
        if supplied_fingerprint != fingerprint:
            raise ValueError("error_fingerprint does not match normalized envelope data")
    payload["error_fingerprint"] = fingerprint
    return payload


def _identity_where(payload: Mapping[str, object]) -> tuple[str, tuple[object, ...]]:
    where = " AND ".join(f"{field} IS ?" for field in _EVENT_IDENTITY_FIELDS)
    return where, tuple(payload[field] for field in _EVENT_IDENTITY_FIELDS)


def _same_payload(row: sqlite3.Row | tuple[Any, ...], payload: Mapping[str, object]) -> bool:
    return all(_row_value(row, field) == payload[field] for field in _PAYLOAD_FIELDS)


def record_failure_envelope(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    task_id: str,
    redacted_error: str | None = None,
    **kwargs: Any,
) -> ProjectFailureEnvelope:
    """Persist one validated failure event, idempotently and atomically.

    Exact repeats return the original row.  A changed payload for the same
    project/task/run identity raises ``ValueError`` before any INSERT.
    Distinct occurrences must use distinct non-NULL ``run_id`` values.
    """
    payload = _validate_and_prepare(
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        task_id=task_id,
        redacted_error=redacted_error,
        kwargs=kwargs,
    )
    where, identity_values = _identity_where(payload)
    with write_txn(conn):
        row = conn.execute(
            f"SELECT * FROM project_failure_envelopes WHERE {where} ORDER BY id LIMIT 1",
            identity_values,
        ).fetchone()
        if row is not None:
            if not _same_payload(row, payload):
                raise ValueError("conflicting failure envelope identity")
            return _row_to_envelope(row)

        now = int(time.time())
        conn.execute(
            """
            INSERT INTO project_failure_envelopes
            (board_id, root_task_id, generation, task_id, run_id, provider, model,
             failure_class, status_code, retry_after, redacted_error, error_fingerprint, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(payload[field] for field in _PAYLOAD_FIELDS) + (now,),
        )
        row = conn.execute(
            "SELECT * FROM project_failure_envelopes WHERE id=last_insert_rowid()"
        ).fetchone()
        if row is None:
            raise RuntimeError("failure envelope insert returned no row")
        return _row_to_envelope(row)


__all__ = [
    "FAILURE_CLASSES",
    "MAX_REDACTED_ERROR_LENGTH",
    "ProjectFailureEnvelope",
    "RECOGNIZED_KWARGS",
    "compute_error_fingerprint",
    "record_failure_envelope",
    "redact_error",
]
