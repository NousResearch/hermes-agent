"""Offline receipt-reconciliation contract for the shared rewind policy (#118142)."""

from __future__ import annotations

import pytest

from agent.context_compressor import HISTORICAL_TASK_HEADING, SUMMARY_PREFIX, _SUMMARY_END_MARKER
from hermes_state_rewind import (
    RewindIndeterminateError,
    RewindTargetUnavailableError,
    rewind_user_turn,
)

_REQUEST_ID = "request-for-offline-rewind"


class ReceiptStore:
    def __init__(self, *, carrier=False, mutation_error=None, receipt=None, lookup_error=None):
        content = "retry"
        if carrier:
            content = f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold\n\n{_SUMMARY_END_MARKER}\n\nretry"
        self.messages = [
            {"_row_id": 11, "role": "user", "content": "keep"},
            {"_row_id": 12, "role": "assistant", "content": "kept"},
            {"_row_id": 13, "role": "user", "content": content},
            {"_row_id": 14, "role": "assistant", "content": "failed"},
        ]
        self.mutation_error = mutation_error
        self.receipt = receipt
        self.lookup_error = lookup_error
        self.mutations = []
        self.lookups = []

    def get_active_message_ids(self, session_id):
        assert session_id == "session"
        return [11, 12, 13, 14]

    def get_messages_as_conversation(self, session_id, *, include_row_ids=False, **kwargs):
        assert session_id == "session" and include_row_ids
        return [dict(message) for message in self.messages]

    def rewind_to_message(self, session_id, target_message_id, **kwargs):
        self.mutations.append((session_id, target_message_id, kwargs))
        if self.mutation_error is not None:
            raise self.mutation_error
        return {"rewound_count": 2, "replacement_message_id": 15}

    def get_rewind_receipt(self, request_id):
        self.lookups.append(request_id)
        if self.lookup_error is not None:
            raise self.lookup_error
        return self.receipt


def _assert_single_mutation(store, *, carrier=False):
    assert store.mutations == [
        ("session", 13, {
            "preserve_compaction_handoff": carrier,
            "expected_active_ids": [11, 12, 13, 14],
            "expected_target_content": "retry",
            "request_id": _REQUEST_ID,
        }),
    ]


def test_successful_ack_does_not_consult_receipt():
    store = ReceiptStore()
    outcome = rewind_user_turn(store, "session", -1, request_id=_REQUEST_ID, require_retryable=True)
    assert outcome.request_id == _REQUEST_ID
    assert outcome.live_text == "retry" and outcome.rewound_count == 2
    assert [row["_row_id"] for row in outcome.prefix] == [11, 12]
    _assert_single_mutation(store)
    assert store.lookups == []


def test_lost_ack_recovers_carrier_from_receipt_without_second_mutation():
    store = ReceiptStore(
        carrier=True, mutation_error=ConnectionError("ack lost"),
        receipt={"retired_count": 2, "replacement_message_id": 15},
    )
    outcome = rewind_user_turn(store, "session", -1, request_id=_REQUEST_ID, require_retryable=True)
    assert outcome.request_id == _REQUEST_ID and outcome.rewound_count == 2
    assert outcome.live_text == "retry"
    assert outcome.prefix[-1]["_row_id"] == 15
    _assert_single_mutation(store, carrier=True)
    assert store.lookups == [_REQUEST_ID]


def test_missing_receipt_reraises_original_error():
    original = ConnectionError("ack lost")
    store = ReceiptStore(mutation_error=original)
    with pytest.raises(ConnectionError) as caught:
        rewind_user_turn(store, "session", -1, request_id=_REQUEST_ID)
    assert caught.value is original
    _assert_single_mutation(store)
    assert store.lookups == [_REQUEST_ID]


def test_receipt_lookup_failure_is_indeterminate_with_cause():
    original = OSError("receipt backend unavailable")
    store = ReceiptStore(mutation_error=ConnectionError("ack lost"), lookup_error=original)
    with pytest.raises(RewindIndeterminateError) as caught:
        rewind_user_turn(store, "session", -1, request_id=_REQUEST_ID)
    assert caught.value.request_id == _REQUEST_ID
    assert caught.value.__cause__ is original
    _assert_single_mutation(store)
    assert store.lookups == [_REQUEST_ID]


def test_target_value_error_maps_without_receipt_lookup():
    original = ValueError("target vanished")
    store = ReceiptStore(mutation_error=original)
    with pytest.raises(RewindTargetUnavailableError, match="target vanished") as caught:
        rewind_user_turn(store, "session", -1, request_id=_REQUEST_ID)
    assert caught.value.__cause__ is original
    _assert_single_mutation(store)
    assert store.lookups == []
