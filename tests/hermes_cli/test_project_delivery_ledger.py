from __future__ import annotations

import sqlite3

import pytest

from hermes_cli import project_delivery_ledger as delivery
from hermes_cli.project_finalization_contract import (
    ProjectFinalization,
    create_project_finalization,
    ensure_project_finalization_schema,
    get_project_finalization,
)


def _new_conn(tmp_path) -> sqlite3.Connection:
    path = tmp_path / "ledger.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    ensure_project_finalization_schema(conn)
    return conn


def test_deterministic_idempotency_key_normalizes_destination_and_varies_by_message_kind():
    key_a = delivery.compute_delivery_idempotency_key(
        board_id="board-a",
        root_task_id="root-1",
        generation=1,
        platform="telegram",
        destination_reference=" chat:123 ",
        message_kind="delivery",
    )
    key_b = delivery.compute_delivery_idempotency_key(
        board_id="board-a",
        root_task_id="root-1",
        generation=1,
        platform="telegram",
        destination_reference="chat:123",
        message_kind="delivery",
    )
    key_c = delivery.compute_delivery_idempotency_key(
        board_id="board-a",
        root_task_id="root-1",
        generation=1,
        platform="telegram",
        destination_reference="chat:123",
        message_kind="summary",
    )

    assert key_a == key_b
    assert key_a != key_c


def test_first_delivery_attempt_is_canonical_and_repeat_returns_same_row(tmp_path):
    conn = _new_conn(tmp_path)
    conn.row_factory = sqlite3.Row

    attempt = delivery.start_delivery_attempt(
        conn,
        board_id="board-a",
        root_task_id="root-1",
        generation=1,
        platform="telegram",
        destination_reference="chat:111",
        thread_reference="thread-1",
        message_kind="summary",
        attempt_number=1,
    )

    repeat = delivery.start_delivery_attempt(
        conn,
        board_id="board-a",
        root_task_id="root-1",
        generation=1,
        platform="telegram",
        destination_reference=" chat:111 ",
        thread_reference="thread-1",
        message_kind="summary",
        attempt_number=1,
    )

    assert attempt.id == repeat.id
    assert attempt.created_at == repeat.created_at
    assert attempt.delivery_state == delivery.DELIVERY_PENDING
    assert attempt.attempt_number == 1


def test_monotonic_attempt_numbers_and_conflicting_identity_raise(tmp_path):
    conn = _new_conn(tmp_path)

    delivery.start_delivery_attempt(
        conn,
        board_id="board-a",
        root_task_id="root-2",
        generation=1,
        platform="telegram",
        destination_reference="chat:222",
        thread_reference=None,
        message_kind="summary",
        attempt_number=1,
    )

    with pytest.raises(ValueError, match="must be positive"):
        delivery.create_delivery_attempt(
            conn,
            board_id="board-a",
            root_task_id="root-2",
            generation=1,
            platform="telegram",
            destination_reference="chat:222",
            thread_reference=None,
            message_kind="summary",
            attempt_number=0,
        )

    with pytest.raises(ValueError, match="attempt_number can only advance by one"):
        delivery.create_delivery_attempt(
            conn,
            board_id="board-a",
            root_task_id="root-2",
            generation=1,
            platform="telegram",
            destination_reference="chat:222",
            thread_reference=None,
            message_kind="summary",
            attempt_number=3,
        )

    with pytest.raises(ValueError, match="conflicting thread_reference"):
        delivery.create_delivery_attempt(
            conn,
            board_id="board-a",
            root_task_id="root-2",
            generation=1,
            platform="telegram",
            destination_reference="chat:222",
            thread_reference="thread-x",
            message_kind="summary",
            attempt_number=1,
        )


def test_accepted_records_provider_message_id_and_is_idempotent(tmp_path):
    conn = _new_conn(tmp_path)

    delivery.start_delivery_attempt(
        conn,
        board_id="board-a",
        root_task_id="root-3",
        generation=1,
        platform="telegram",
        destination_reference="chat:333",
        thread_reference="thread-333",
        message_kind="summary",
        attempt_number=1,
    )

    first = delivery.mark_delivery_attempt_accepted(
        conn,
        board_id="board-a",
        root_task_id="root-3",
        generation=1,
        platform="telegram",
        destination_reference="chat:333",
        message_kind="summary",
        attempt_number=1,
        provider_message_id="prov-1",
        now=1000,
    )

    repeat = delivery.mark_delivery_attempt_accepted(
        conn,
        board_id="board-a",
        root_task_id="root-3",
        generation=1,
        platform="telegram",
        destination_reference="chat:333",
        message_kind="summary",
        attempt_number=1,
        provider_message_id="prov-1",
        now=2000,
    )

    assert first.id == repeat.id
    assert first.accepted is True
    assert first.provider_message_id == "prov-1"
    assert repeat.provider_message_id == "prov-1"

    with pytest.raises(ValueError, match="conflicting provider_message_id"):
        delivery.mark_delivery_attempt_accepted(
            conn,
            board_id="board-a",
            root_task_id="root-3",
            generation=1,
            platform="telegram",
            destination_reference="chat:333",
            message_kind="summary",
            attempt_number=1,
            provider_message_id="prov-2",
        )

    with pytest.raises(ValueError, match="terminal delivery"):
        delivery.create_next_delivery_attempt(
            conn,
            board_id="board-a",
            root_task_id="root-3",
            generation=1,
            platform="telegram",
            destination_reference="chat:333",
            thread_reference="thread-333",
            message_kind="summary",
        )

    attempts = delivery.list_delivery_attempts(
        conn,
        board_id="board-a",
        root_task_id="root-3",
        generation=1,
        platform="telegram",
        destination_reference="chat:333",
        message_kind="summary",
    )
    assert [attempt.attempt_number for attempt in attempts] == [1]


def test_rejected_stores_redacted_error_only_and_ambiguous_needs_resolution(tmp_path):
    conn = _new_conn(tmp_path)

    delivery.start_delivery_attempt(
        conn,
        board_id="board-a",
        root_task_id="root-4",
        generation=1,
        platform="telegram",
        destination_reference="chat:444",
        thread_reference="thread-444",
        message_kind="summary",
        attempt_number=1,
    )

    rejected = delivery.mark_delivery_attempt_rejected(
        conn,
        board_id="board-a",
        root_task_id="root-4",
        generation=1,
        platform="telegram",
        destination_reference="chat:444",
        message_kind="summary",
        attempt_number=1,
        redacted_error="redacted: temporary network failure",
        now=5000,
    )

    ambiguous = delivery.mark_delivery_attempt_ambiguous(
        conn,
        board_id="board-a",
        root_task_id="root-4",
        generation=1,
        platform="telegram",
        destination_reference="chat:444",
        message_kind="summary",
        attempt_number=1,
        redacted_error="redacted: provider did not acknowledge",
        now=6000,
    )

    assert rejected.delivery_state == delivery.DELIVERY_REJECTED
    assert rejected.redacted_error == "redacted: temporary network failure"
    assert rejected.provider_message_id is None
    assert ambiguous.delivery_state == delivery.DELIVERY_AMBIGUOUS
    assert ambiguous.redacted_error == "redacted: provider did not acknowledge"


def test_deterministic_retry_schedule_and_permanent_failure_terminal(tmp_path):
    conn = _new_conn(tmp_path)

    delivery.start_delivery_attempt(
        conn,
        board_id="board-a",
        root_task_id="root-5",
        generation=1,
        platform="telegram",
        destination_reference="chat:555",
        thread_reference=None,
        message_kind="summary",
        attempt_number=1,
    )

    attempt1 = delivery.mark_delivery_attempt_retry_scheduled(
        conn,
        board_id="board-a",
        root_task_id="root-5",
        generation=1,
        platform="telegram",
        destination_reference="chat:555",
        message_kind="summary",
        attempt_number=1,
        now=1000,
    )
    assert attempt1.delivery_state == delivery.DELIVERY_RETRY_SCHEDULED
    assert attempt1.next_retry_at == 1000 + delivery.RETRY_DELAYS_SECONDS[0]

    attempt2 = delivery.create_next_delivery_attempt(
        conn,
        board_id="board-a",
        root_task_id="root-5",
        generation=1,
        platform="telegram",
        destination_reference="chat:555",
        thread_reference=None,
        message_kind="summary",
        delivery_state=delivery.DELIVERY_PENDING,
    )

    attempt2_retry = delivery.mark_delivery_attempt_retry_scheduled(
        conn,
        board_id="board-a",
        root_task_id="root-5",
        generation=1,
        platform="telegram",
        destination_reference="chat:555",
        message_kind="summary",
        attempt_number=2,
        now=500,
    )
    assert attempt2_retry.delivery_state == delivery.DELIVERY_RETRY_SCHEDULED
    assert attempt2_retry.next_retry_at == 500 + delivery.RETRY_DELAYS_SECONDS[1]
    assert attempt2.attempt_number == 2

    attempt3 = delivery.create_next_delivery_attempt(
        conn,
        board_id="board-a",
        root_task_id="root-5",
        generation=1,
        platform="telegram",
        destination_reference="chat:555",
        thread_reference=None,
        message_kind="summary",
        delivery_state=delivery.DELIVERY_PENDING,
    )

    with pytest.raises(ValueError, match="cannot schedule retry"):
        delivery.mark_delivery_attempt_retry_scheduled(
            conn,
            board_id="board-a",
            root_task_id="root-5",
            generation=1,
            platform="telegram",
            destination_reference="chat:555",
            message_kind="summary",
            attempt_number=3,
        )

    terminal = delivery.mark_delivery_attempt_permanent_failure(
        conn,
        board_id="board-a",
        root_task_id="root-5",
        generation=1,
        platform="telegram",
        destination_reference="chat:555",
        message_kind="summary",
        attempt_number=3,
        redacted_error="redacted: max attempts exhausted",
        now=9000,
    )

    assert terminal.delivery_state == delivery.DELIVERY_PERMANENT_FAILURE
    assert terminal.redacted_error == "redacted: max attempts exhausted"
    assert terminal.attempt_number == attempt3.attempt_number == 3

    with pytest.raises(ValueError, match="terminal"):
        delivery.mark_delivery_attempt_accepted(
            conn,
            board_id="board-a",
            root_task_id="root-5",
            generation=1,
            platform="telegram",
            destination_reference="chat:555",
            message_kind="summary",
            attempt_number=3,
            provider_message_id="prov-2",
        )

    with pytest.raises(ValueError, match="terminal delivery"):
        delivery.create_delivery_attempt(
            conn,
            board_id="board-a",
            root_task_id="root-5",
            generation=1,
            platform="telegram",
            destination_reference="chat:555",
            thread_reference=None,
            message_kind="summary",
            attempt_number=4,
        )

    attempts = delivery.list_delivery_attempts(
        conn,
        board_id="board-a",
        root_task_id="root-5",
        generation=1,
        platform="telegram",
        destination_reference="chat:555",
        message_kind="summary",
    )
    assert [attempt.attempt_number for attempt in attempts] == [1, 2, 3]

    assert delivery.MAX_DELIVERY_ATTEMPTS == 3


def test_stable_read_ordering_is_non_mutating(tmp_path):
    conn = _new_conn(tmp_path)

    for attempt_number in (1, 2, 3):
        delivery.create_delivery_attempt(
            conn,
            board_id="board-a",
            root_task_id="root-6",
            generation=1,
            platform="telegram",
            destination_reference="chat:666",
            thread_reference="thread-666",
            message_kind="summary",
            attempt_number=attempt_number,
            delivery_state=delivery.DELIVERY_PENDING,
        )

    count_before = conn.execute(
        "SELECT COUNT(*) FROM project_delivery_attempts WHERE board_id='board-a'"
    ).fetchone()[0]
    attempts = delivery.list_delivery_attempts(
        conn,
        board_id="board-a",
        root_task_id="root-6",
        generation=1,
        platform="telegram",
        destination_reference="chat:666",
        message_kind="summary",
    )
    count_after = conn.execute(
        "SELECT COUNT(*) FROM project_delivery_attempts WHERE board_id='board-a'"
    ).fetchone()[0]

    assert [attempt.attempt_number for attempt in attempts] == [1, 2, 3]
    assert count_before == count_after
    assert attempts[0].created_at <= attempts[-1].created_at


def test_project_finalization_state_is_separate_from_delivery_lifecycle(tmp_path):
    conn = _new_conn(tmp_path)

    baseline = create_project_finalization(
        conn,
        board_id="board-a",
        root_task_id="root-7",
        final_checker_task_id="checker-1",
    )
    assert isinstance(baseline, ProjectFinalization)

    delivery.start_delivery_attempt(
        conn,
        board_id="board-a",
        root_task_id="root-7",
        generation=1,
        platform="telegram",
        destination_reference="chat:777",
        thread_reference="thread-777",
        message_kind="summary",
        attempt_number=1,
    )
    delivery.mark_delivery_attempt_accepted(
        conn,
        board_id="board-a",
        root_task_id="root-7",
        generation=1,
        platform="telegram",
        destination_reference="chat:777",
        message_kind="summary",
        attempt_number=1,
        provider_message_id="prov-777",
    )

    observed = get_project_finalization(
        conn,
        board_id="board-a",
        root_task_id="root-7",
    )

    assert observed == baseline
    assert baseline.state == "open"


def test_rejected_delivery_persists_only_redacted_frozen_columns(tmp_path):
    conn = _new_conn(tmp_path)
    raw_error = (
        "message: do not persist this final Telegram body; "
        "authorization: Bearer abcdefghijklmnop; "
        "https://example.test/callback?access_token=oauth-secret"
    )

    delivery.start_delivery_attempt(
        conn,
        board_id="board-a",
        root_task_id="root-8",
        generation=1,
        platform="telegram",
        destination_reference="chat:888",
        thread_reference=None,
        message_kind="summary",
    )
    rejected = delivery.mark_delivery_attempt_rejected(
        conn,
        board_id="board-a",
        root_task_id="root-8",
        generation=1,
        platform="telegram",
        destination_reference="chat:888",
        message_kind="summary",
        attempt_number=1,
        redacted_error=raw_error,
        now=1000,
    )

    stored_error = conn.execute(
        "SELECT redacted_error FROM project_delivery_attempts WHERE id = ?", (rejected.id,)
    ).fetchone()[0]
    for forbidden in ("do not persist", "abcdefghijklmnop", "oauth-secret", "example.test"):
        assert forbidden not in stored_error
    assert delivery.compute_delivery_error_fingerprint(raw_error) == (
        delivery.compute_delivery_error_fingerprint(stored_error)
    )
    assert [row[1] for row in conn.execute("PRAGMA table_info(project_delivery_attempts)")] == [
        "id",
        "board_id",
        "root_task_id",
        "generation",
        "idempotency_key",
        "platform",
        "destination_reference",
        "thread_reference",
        "attempt_number",
        "delivery_state",
        "accepted",
        "provider_message_id",
        "redacted_error",
        "created_at",
        "completed_at",
        "next_retry_at",
    ]
