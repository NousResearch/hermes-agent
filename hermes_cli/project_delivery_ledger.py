"""Delivery attempt state machine helper for project completion notifications.

The delivery module intentionally owns retry/attempt state transitions but does not
own project-finalization semantics. It sits on top of the frozen
:mod:`hermes_cli.project_finalization_contract` persistence boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from typing import Any, Iterable, Optional

from hermes_cli.project_finalization_contract import (
    ProjectDeliveryAttempt,
    record_delivery_attempt,
    validate_generation,
)
from hermes_cli.sqlite_util import write_txn

# Delivery states consumed by the delivery boundary.
DELIVERY_STATES: tuple[str, ...] = (
    "pending",
    "attempting",
    "accepted",
    "rejected",
    "retry_scheduled",
    "permanent_failure",
    "ambiguous",
)

DELIVERY_ACCEPTED = "accepted"
DELIVERY_AMBIGUOUS = "ambiguous"
DELIVERY_ATTEMPTING = "attempting"
DELIVERY_PERMANENT_FAILURE = "permanent_failure"
DELIVERY_PENDING = "pending"
DELIVERY_REJECTED = "rejected"
DELIVERY_RETRY_SCHEDULED = "retry_scheduled"

TERMINAL_DELIVERY_STATES = (DELIVERY_ACCEPTED, DELIVERY_PERMANENT_FAILURE)

# Deterministic retry schedule and hard delivery-attempt cap.
MAX_DELIVERY_ATTEMPTS = 3
RETRY_DELAYS_SECONDS: tuple[int, ...] = (30, 120, 300)
MAX_REDACTED_ERROR_LENGTH = 2048

_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_AUTH_RE = re.compile(
    r"\b(?:authorization|proxy-authorization)\s*[:=]\s*"
    r"(?:bearer|basic)\s+[^\s,;]+",
    re.IGNORECASE,
)
_SECRET_FIELD_RE = re.compile(
    r"(?P<prefix>(?:\"|')?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"oauth(?:[_-]?(?:token|secret))?|client[_-]?secret|authorization|password|"
    r"secret|credential|request[_-]?body|response[_-]?body|prompt|completion|"
    r"chat|message|payload)(?:\"|')?\s*[:=]\s*)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;}]+)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|pk-[A-Za-z0-9_-]{8,}|"
    r"(?:sk|pk|rk|xox[baprs]-|gh[pousr]_|github_pat_|ya29\.)"
    r"[A-Za-z0-9._~+/=-]{8,}|eyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})\b"
)
_SENSITIVE_PREFIX_RE = re.compile(
    r"^\s*(?:request[ _-]?body|response[ _-]?body|payload|prompt|completion|"
    r"chat|message)\s*[:=]",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"[ \t\r\n]+")


__all__ = [
    "DELIVERY_ATTEMPTING",
    "DELIVERY_ACCEPTED",
    "DELIVERY_PERMANENT_FAILURE",
    "DELIVERY_PENDING",
    "DELIVERY_REJECTED",
    "DELIVERY_RETRY_SCHEDULED",
    "DELIVERY_AMBIGUOUS",
    "DELIVERY_STATES",
    "MAX_DELIVERY_ATTEMPTS",
    "RETRY_DELAYS_SECONDS",
    "MAX_REDACTED_ERROR_LENGTH",
    "compute_delivery_idempotency_key",
    "derive_delivery_idempotency_key",
    "derive_project_delivery_idempotency_key",
    "build_delivery_idempotency_key",
    "normalize_destination_reference",
    "redact_delivery_error",
    "compute_delivery_error_fingerprint",
    "list_delivery_attempts",
    "get_latest_delivery_attempt",
    "create_delivery_attempt",
    "create_next_delivery_attempt",
    "start_delivery_attempt",
    "mark_delivery_attempt_attempting",
    "mark_delivery_attempt_accepted",
    "mark_delivery_attempt_rejected",
    "mark_delivery_attempt_ambiguous",
    "mark_delivery_attempt_retry_scheduled",
    "mark_delivery_attempt_permanent_failure",
]


def redact_delivery_error(error: str | None) -> str | None:
    """Bound redacted delivery diagnostics before they reach durable storage.

    A delivery receipt must never become a side channel for message bodies or
    provider request/response payloads. Complete JSON and explicitly labelled
    payload text are replaced wholesale; common credentials and URLs in ordinary
    diagnostics are replaced in place.
    """

    if error is None:
        return None
    if not isinstance(error, str):
        raise TypeError("redacted_error must be a string or None")
    value = error.strip()
    if not value:
        return None
    try:
        json.loads(value)
    except json.JSONDecodeError:
        pass
    else:
        return "[structured error redacted]"
    if _SENSITIVE_PREFIX_RE.search(value):
        return "[structured error redacted]"
    value = _URL_RE.sub("[url redacted]", value)
    value = _AUTH_RE.sub("[authorization redacted]", value)
    value = _SECRET_FIELD_RE.sub("[secret field redacted]", value)
    value = _TOKEN_RE.sub("[token redacted]", value)
    value = _SPACE_RE.sub(" ", value).strip()
    if len(value) > MAX_REDACTED_ERROR_LENGTH:
        value = value[:MAX_REDACTED_ERROR_LENGTH].rstrip() + "…"
    return value or None


def compute_delivery_error_fingerprint(redacted_error: str | None) -> str:
    """Return a stable SHA-256 fingerprint of redacted delivery diagnostics.

    HOF-002 has no delivery-fingerprint column, so this is deterministic
    derived data rather than an unauthorized schema mutation.
    """

    safe_error = redact_delivery_error(redacted_error)
    return hashlib.sha256((safe_error or "").encode("utf-8")).hexdigest()


def normalize_destination_reference(destination_reference: str | None) -> Optional[str]:
    """Normalize destination reference for identity and key generation."""

    if destination_reference is None:
        return None
    normalized = destination_reference.strip()
    return normalized or None


def _normalized_identity_token(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("delivery identity field cannot be empty")
    return normalized


def compute_delivery_idempotency_key(
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    platform: str,
    destination_reference: str | None,
    message_kind: str,
) -> str:
    """Return a deterministic identity for this logical destination attempt stream."""

    validate_generation(generation)
    parts = [
        _normalized_identity_token(board_id),
        _normalized_identity_token(root_task_id),
        str(generation),
        _normalized_identity_token(platform),
        normalize_destination_reference(destination_reference) or "",
        _normalized_identity_token(message_kind),
    ]
    payload = "\u0000".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_delivery_idempotency_key(
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    platform: str,
    destination_reference: str | None,
    message_kind: str,
) -> str:
    return compute_delivery_idempotency_key(
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=destination_reference,
        message_kind=message_kind,
    )


def derive_delivery_idempotency_key(
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    platform: str,
    destination_reference: str | None,
    message_kind: str,
) -> str:
    return compute_delivery_idempotency_key(
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=destination_reference,
        message_kind=message_kind,
    )


def derive_project_delivery_idempotency_key(
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    platform: str,
    destination_reference: str | None,
    message_kind: str,
) -> str:
    return compute_delivery_idempotency_key(
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=destination_reference,
        message_kind=message_kind,
    )


def _row_to_delivery_attempt(row: dict[str, Any]) -> ProjectDeliveryAttempt:
    return ProjectDeliveryAttempt(
        id=row["id"],
        board_id=row["board_id"],
        root_task_id=row["root_task_id"],
        generation=int(row["generation"]),
        idempotency_key=row["idempotency_key"],
        platform=row["platform"],
        destination_reference=row["destination_reference"],
        thread_reference=row["thread_reference"],
        attempt_number=int(row["attempt_number"]),
        delivery_state=row["delivery_state"],
        accepted=bool(row["accepted"]) if row["accepted"] is not None else None,
        provider_message_id=row["provider_message_id"],
        redacted_error=row["redacted_error"],
        created_at=int(row["created_at"]),
        completed_at=int(row["completed_at"]) if row["completed_at"] is not None else None,
        next_retry_at=int(row["next_retry_at"]) if row["next_retry_at"] is not None else None,
    )


def _select_rows(conn: sqlite3.Connection, query: str, params: Iterable[Any]) -> list[dict[str, Any]]:
    cursor = conn.execute(query, tuple(params))
    columns = [column[0] for column in cursor.description or []]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _select_one_row(
    conn: sqlite3.Connection, query: str, params: Iterable[Any]
) -> Optional[dict[str, Any]]:
    cursor = conn.execute(query, tuple(params))
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [column[0] for column in cursor.description or []]
    return dict(zip(columns, row))


def _ensure_delivery_identity(
    row: ProjectDeliveryAttempt,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    idempotency_key: str,
    platform: str,
    destination_reference: str | None,
    thread_reference: str | None,
    attempt_number: int,
    delivery_state: str,
) -> None:
    if row.board_id != board_id:
        raise ValueError("conflicting board_id for existing delivery attempt")
    if row.root_task_id != root_task_id:
        raise ValueError("conflicting root_task_id for existing delivery attempt")
    if row.generation != generation:
        raise ValueError("conflicting generation for existing delivery attempt")
    if row.idempotency_key != idempotency_key:
        raise ValueError("conflicting idempotency_key for existing delivery attempt")
    if row.platform != platform:
        raise ValueError("conflicting platform for existing delivery attempt")
    normalized_existing_destination = normalize_destination_reference(row.destination_reference)
    normalized_candidate_destination = normalize_destination_reference(destination_reference)
    if normalized_existing_destination != normalized_candidate_destination:
        raise ValueError("conflicting destination_reference for existing delivery attempt")
    if row.thread_reference != thread_reference:
        raise ValueError("conflicting thread_reference for existing delivery attempt")
    if row.attempt_number != attempt_number:
        raise ValueError("conflicting attempt_number for existing delivery attempt")
    if row.delivery_state != delivery_state:
        raise ValueError("conflicting delivery_state for existing delivery attempt")


def _validate_delivery_state(state: str) -> None:
    if state not in DELIVERY_STATES:
        raise ValueError(f"invalid delivery_state: {state}")


def _validate_transition(
    current_state: str,
    target_state: str,
) -> None:
    _validate_delivery_state(target_state)
    if current_state in TERMINAL_DELIVERY_STATES and current_state != target_state:
        raise ValueError(f"cannot transition terminal state {current_state} -> {target_state}")


def _fetch_attempt_chain(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    platform: str,
    destination_reference: str | None,
    message_kind: str,
) -> list[ProjectDeliveryAttempt]:
    idempotency_key = compute_delivery_idempotency_key(
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=destination_reference,
        message_kind=message_kind,
    )
    return [
        _row_to_delivery_attempt(row)
        for row in _select_rows(
            conn,
            """
            SELECT *
            FROM project_delivery_attempts
            WHERE idempotency_key = ?
            ORDER BY attempt_number, id
            """,
            (idempotency_key,),
        )
    ]


def list_delivery_attempts(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    platform: str,
    destination_reference: str | None,
    message_kind: str,
) -> list[ProjectDeliveryAttempt]:
    """Return stable, ordered attempts for a delivery identity."""

    validate_generation(generation)
    _normalized_identity_token(board_id)
    _normalized_identity_token(root_task_id)
    _normalized_identity_token(platform)
    _normalized_identity_token(message_kind)
    return _fetch_attempt_chain(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=destination_reference,
        message_kind=message_kind,
    )


def get_latest_delivery_attempt(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    platform: str,
    destination_reference: str | None,
    message_kind: str,
) -> Optional[ProjectDeliveryAttempt]:
    attempts = list_delivery_attempts(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=destination_reference,
        message_kind=message_kind,
    )
    return attempts[-1] if attempts else None


def _next_attempt_number(latest: Optional[ProjectDeliveryAttempt]) -> int:
    return 1 if latest is None else latest.attempt_number + 1


def create_delivery_attempt(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    platform: str,
    destination_reference: str | None,
    thread_reference: str | None,
    message_kind: str,
    attempt_number: int,
    delivery_state: str = DELIVERY_PENDING,
    accepted: bool | None = None,
    provider_message_id: str | None = None,
    redacted_error: str | None = None,
) -> ProjectDeliveryAttempt:
    """Insert an attempt row when identity is missing, returning the existing row on replay."""

    validate_generation(generation)
    if attempt_number <= 0:
        raise ValueError("attempt_number must be positive")
    _validate_delivery_state(delivery_state)
    _normalized_identity_token(board_id)
    _normalized_identity_token(root_task_id)
    _normalized_identity_token(platform)
    _normalized_identity_token(message_kind)

    normalized_destination = normalize_destination_reference(destination_reference)
    idempotency_key = compute_delivery_idempotency_key(
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=normalized_destination,
        message_kind=message_kind,
    )

    attempts = list_delivery_attempts(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=normalized_destination,
        message_kind=message_kind,
    )
    latest = attempts[-1] if attempts else None
    if any(
        attempt.attempt_number < attempt_number
        and attempt.delivery_state in TERMINAL_DELIVERY_STATES
        for attempt in attempts
    ):
        raise ValueError("cannot create a later attempt after terminal delivery")
    if latest is not None:
        if attempt_number < latest.attempt_number:
            raise ValueError("attempt_number must be monotonic")
        if attempt_number > latest.attempt_number + 1:
            raise ValueError("attempt_number can only advance by one")

    attempt = record_delivery_attempt(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        idempotency_key=idempotency_key,
        platform=platform,
        attempt_number=attempt_number,
        delivery_state=delivery_state,
        destination_reference=destination_reference,
        thread_reference=thread_reference,
        accepted=accepted,
        provider_message_id=provider_message_id,
        redacted_error=redacted_error,
    )
    _ensure_delivery_identity(
        attempt,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        idempotency_key=idempotency_key,
        platform=platform,
        destination_reference=destination_reference,
        thread_reference=thread_reference,
        attempt_number=attempt_number,
        delivery_state=delivery_state,
    )
    return attempt


def create_next_delivery_attempt(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    platform: str,
    destination_reference: str | None,
    thread_reference: str | None,
    message_kind: str,
    delivery_state: str = DELIVERY_PENDING,
) -> ProjectDeliveryAttempt:
    latest = get_latest_delivery_attempt(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=destination_reference,
        message_kind=message_kind,
    )
    if latest is None:
        raise ValueError("first attempt must be created before creating the next")
    next_attempt = _next_attempt_number(latest)
    if next_attempt > MAX_DELIVERY_ATTEMPTS:
        raise ValueError("maximum delivery attempts exceeded")
    return create_delivery_attempt(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=destination_reference,
        thread_reference=thread_reference,
        message_kind=message_kind,
        attempt_number=next_attempt,
        delivery_state=delivery_state,
    )


def start_delivery_attempt(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    platform: str,
    destination_reference: str | None,
    thread_reference: str | None,
    message_kind: str,
    attempt_number: int = 1,
) -> ProjectDeliveryAttempt:
    return create_delivery_attempt(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=destination_reference,
        thread_reference=thread_reference,
        message_kind=message_kind,
        attempt_number=attempt_number,
        delivery_state=DELIVERY_PENDING,
    )


def _transition_attempt(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    platform: str,
    destination_reference: str | None,
    message_kind: str,
    attempt_number: int,
    delivery_state: str,
    accepted: bool | None,
    provider_message_id: str | None,
    redacted_error: str | None,
    completed_at: int | None,
    next_retry_at: int | None,
) -> ProjectDeliveryAttempt:
    validate_generation(generation)
    if attempt_number <= 0:
        raise ValueError("attempt_number must be positive")
    _validate_delivery_state(delivery_state)

    _normalized_identity_token(board_id)
    _normalized_identity_token(root_task_id)
    _normalized_identity_token(platform)
    _normalized_identity_token(message_kind)

    idempotency_key = compute_delivery_idempotency_key(
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=destination_reference,
        message_kind=message_kind,
    )

    current = _select_one_row(
        conn,
        """
        SELECT *
        FROM project_delivery_attempts
        WHERE idempotency_key = ?
          AND attempt_number = ?
        """,
        (idempotency_key, attempt_number),
    )
    if current is None:
        raise ValueError("delivery attempt not found")
    attempt = _row_to_delivery_attempt(current)
    _ensure_delivery_identity(
        attempt,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        idempotency_key=idempotency_key,
        platform=platform,
        destination_reference=destination_reference,
        thread_reference=attempt.thread_reference,
        attempt_number=attempt_number,
        delivery_state=attempt.delivery_state,
    )

    if attempt.delivery_state == delivery_state:
        # idempotent replay: return existing row unchanged
        return attempt

    _validate_transition(attempt.delivery_state, delivery_state)
    with write_txn(conn):
        conn.execute(
            """
            UPDATE project_delivery_attempts
               SET delivery_state = ?,
                   accepted = ?,
                   provider_message_id = ?,
                   redacted_error = ?,
                   completed_at = ?,
                   next_retry_at = ?
             WHERE id = ?
            """,
            (
                delivery_state,
                1 if accepted else 0 if accepted is not None else None,
                provider_message_id,
                redacted_error,
                completed_at,
                next_retry_at,
                attempt.id,
            ),
        )
    updated = _select_one_row(
        conn,
        "SELECT * FROM project_delivery_attempts WHERE id = ?",
        (attempt.id,),
    )
    assert updated is not None
    return _row_to_delivery_attempt(updated)


def mark_delivery_attempt_attempting(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    platform: str,
    destination_reference: str | None,
    message_kind: str,
    attempt_number: int,
    now: int | None = None,
) -> ProjectDeliveryAttempt:
    return _transition_attempt(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=destination_reference,
        message_kind=message_kind,
        attempt_number=attempt_number,
        delivery_state=DELIVERY_ATTEMPTING,
        accepted=None,
        provider_message_id=None,
        redacted_error=None,
        completed_at=None,
        next_retry_at=None,
    )


def mark_delivery_attempt_accepted(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    platform: str,
    destination_reference: str | None,
    message_kind: str,
    attempt_number: int,
    provider_message_id: str | None,
    now: int | None = None,
) -> ProjectDeliveryAttempt:
    if not provider_message_id:
        raise ValueError("provider_message_id is required")
    validate_generation(generation)

    attempt = get_latest_delivery_attempt(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=destination_reference,
        message_kind=message_kind,
    )
    if attempt is None:
        raise ValueError("delivery attempt not found")
    if attempt.attempt_number != attempt_number:
        attempt = None
        if attempt_number <= 0:
            raise ValueError("attempt_number must be positive")
        attempt = _select_one_row(
            conn,
            """
            SELECT * FROM project_delivery_attempts
            WHERE idempotency_key = ? AND attempt_number = ?
            """,
            (
                compute_delivery_idempotency_key(
                    board_id=board_id,
                    root_task_id=root_task_id,
                    generation=generation,
                    platform=platform,
                    destination_reference=destination_reference,
                    message_kind=message_kind,
                ),
                attempt_number,
            ),
        )
        if attempt is None:
            raise ValueError("delivery attempt not found")
        attempt = _row_to_delivery_attempt(attempt)

    if attempt.delivery_state == DELIVERY_ACCEPTED:
        if attempt.provider_message_id != provider_message_id:
            raise ValueError("conflicting provider_message_id for already accepted delivery")
        return attempt

    _validate_transition(attempt.delivery_state, DELIVERY_ACCEPTED)
    return _transition_attempt(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=destination_reference,
        message_kind=message_kind,
        attempt_number=attempt_number,
        delivery_state=DELIVERY_ACCEPTED,
        accepted=True,
        provider_message_id=provider_message_id,
        redacted_error=None,
        completed_at=now if now is not None else int(time.time()),
        next_retry_at=None,
    )


def mark_delivery_attempt_rejected(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    platform: str,
    destination_reference: str | None,
    message_kind: str,
    attempt_number: int,
    redacted_error: str,
    now: int | None = None,
) -> ProjectDeliveryAttempt:
    safe_error = redact_delivery_error(redacted_error)
    if not safe_error:
        raise ValueError("redacted_error is required")
    return _transition_attempt(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=destination_reference,
        message_kind=message_kind,
        attempt_number=attempt_number,
        delivery_state=DELIVERY_REJECTED,
        accepted=False,
        provider_message_id=None,
        redacted_error=safe_error,
        completed_at=now if now is not None else int(time.time()),
        next_retry_at=None,
    )


def mark_delivery_attempt_ambiguous(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    platform: str,
    destination_reference: str | None,
    message_kind: str,
    attempt_number: int,
    redacted_error: str,
    now: int | None = None,
) -> ProjectDeliveryAttempt:
    safe_error = redact_delivery_error(redacted_error)
    if not safe_error:
        raise ValueError("redacted_error is required")
    return _transition_attempt(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=destination_reference,
        message_kind=message_kind,
        attempt_number=attempt_number,
        delivery_state=DELIVERY_AMBIGUOUS,
        accepted=None,
        provider_message_id=None,
        redacted_error=safe_error,
        completed_at=now if now is not None else int(time.time()),
        next_retry_at=None,
    )


def _retry_delay_seconds(attempt_number: int) -> int:
    if attempt_number < 1 or attempt_number > MAX_DELIVERY_ATTEMPTS:
        raise ValueError("attempt_number out of range")
    index = attempt_number - 1
    return RETRY_DELAYS_SECONDS[index]


def mark_delivery_attempt_retry_scheduled(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    platform: str,
    destination_reference: str | None,
    message_kind: str,
    attempt_number: int,
    now: int | None = None,
) -> ProjectDeliveryAttempt:
    if attempt_number >= MAX_DELIVERY_ATTEMPTS:
        raise ValueError("maximum attempts reached; cannot schedule retry")
    timestamp = now if now is not None else int(time.time())
    delay = _retry_delay_seconds(attempt_number)
    return _transition_attempt(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=destination_reference,
        message_kind=message_kind,
        attempt_number=attempt_number,
        delivery_state=DELIVERY_RETRY_SCHEDULED,
        accepted=False,
        provider_message_id=None,
        redacted_error=None,
        completed_at=timestamp,
        next_retry_at=timestamp + delay,
    )


def mark_delivery_attempt_permanent_failure(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    platform: str,
    destination_reference: str | None,
    message_kind: str,
    attempt_number: int,
    redacted_error: str | None = None,
    now: int | None = None,
) -> ProjectDeliveryAttempt:
    safe_error = redact_delivery_error(redacted_error)
    return _transition_attempt(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        platform=platform,
        destination_reference=destination_reference,
        message_kind=message_kind,
        attempt_number=attempt_number,
        delivery_state=DELIVERY_PERMANENT_FAILURE,
        accepted=False,
        provider_message_id=None,
        redacted_error=safe_error,
        completed_at=now if now is not None else int(time.time()),
        next_retry_at=None,
    )
