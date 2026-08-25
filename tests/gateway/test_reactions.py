"""Behavior contracts for exact-target Tapback operations."""

import pytest

from gateway.reactions import (
    TapbackAction,
    TapbackDirection,
    TapbackOperation,
    TapbackStatus,
    TapbackType,
    TapbackValidationError,
)


def _operation(**overrides):
    fields = {
        "platform": "bluebubbles",
        "chat_id": "iMessage;+;family-guid",
        "target_message_id": "target-message-guid",
        "sender_id": "+15555550100",
        "reaction": TapbackType.LIKE,
        "action": TapbackAction.ADD,
        "direction": TapbackDirection.INBOUND,
        "source_event_id": "tapback-event-guid",
        "part_index": 0,
    }
    fields.update(overrides)
    return TapbackOperation(**fields)


def test_operation_represents_exact_inbound_and_outbound_identity():
    inbound = _operation()
    outbound = _operation(
        sender_id="self",
        direction=TapbackDirection.OUTBOUND,
        source_event_id="approved-request-id",
    )

    assert inbound.chat_id == "iMessage;+;family-guid"
    assert inbound.target_message_id == "target-message-guid"
    assert inbound.sender_id == "+15555550100"
    assert inbound.status is TapbackStatus.RECEIVED
    assert outbound.direction is TapbackDirection.OUTBOUND


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("platform", ""),
        ("chat_id", ""),
        ("chat_id", "   "),
        ("chat_id", ["chat-a", "chat-b"]),
        ("target_message_id", ""),
        ("target_message_id", ["message-a", "message-b"]),
        ("sender_id", ""),
        ("source_event_id", ""),
        ("part_index", -1),
        ("part_index", True),
    ],
)
def test_ambiguous_or_malformed_identity_fails_closed(field, value):
    with pytest.raises(TapbackValidationError):
        _operation(**{field: value})


@pytest.mark.parametrize("reaction", list(TapbackType))
def test_all_six_native_tapbacks_are_allowed(reaction):
    assert _operation(reaction=reaction).reaction is reaction


def test_unsupported_reaction_and_untyped_semantics_are_rejected():
    with pytest.raises(TapbackValidationError):
        _operation(reaction="party")
    with pytest.raises(TapbackValidationError):
        _operation(action="add")
    with pytest.raises(TapbackValidationError):
        _operation(direction="inbound")


def test_deduplication_key_is_deterministic_for_identical_events():
    assert _operation().deduplication_key == _operation().deduplication_key


def test_deduplication_key_distinguishes_add_remove_chat_sender_and_direction():
    base = _operation()
    variants = [
        _operation(action=TapbackAction.REMOVE),
        _operation(chat_id="iMessage;+;other-family-guid"),
        _operation(sender_id="other@example.com"),
        _operation(direction=TapbackDirection.OUTBOUND),
        _operation(source_event_id="different-source-event"),
        _operation(part_index=1),
    ]

    assert all(item.deduplication_key != base.deduplication_key for item in variants)


def test_state_key_is_local_to_chat_message_sender_and_direction():
    add = _operation(action=TapbackAction.ADD, source_event_id="event-add")
    remove = _operation(action=TapbackAction.REMOVE, source_event_id="event-remove")

    assert add.state_key == remove.state_key
    assert add.state_key != _operation(chat_id="other-chat").state_key
    assert add.state_key != _operation(sender_id="other-sender").state_key


def test_status_transitions_document_happy_path_and_terminal_states():
    operation = _operation()
    operation = operation.transition_to(TapbackStatus.VALIDATED)
    operation = operation.transition_to(TapbackStatus.PENDING)
    operation = operation.transition_to(TapbackStatus.PROCESSING)
    operation = operation.transition_to(TapbackStatus.APPLIED)

    assert operation.status is TapbackStatus.APPLIED
    with pytest.raises(ValueError, match="cannot transition"):
        operation.transition_to(TapbackStatus.PROCESSING)


def test_failure_can_retry_but_rejection_is_terminal():
    failed = (
        _operation()
        .transition_to(TapbackStatus.VALIDATED)
        .transition_to(TapbackStatus.PENDING)
        .transition_to(TapbackStatus.PROCESSING)
        .transition_to(TapbackStatus.FAILED)
    )
    assert failed.transition_to(TapbackStatus.PENDING).status is TapbackStatus.PENDING

    rejected = _operation().transition_to(TapbackStatus.REJECTED)
    with pytest.raises(ValueError, match="cannot transition"):
        rejected.transition_to(TapbackStatus.PENDING)


def test_platform_payload_contains_only_exact_routing_fields():
    payload = _operation().to_platform_payload()

    assert payload == {
        "action": "added",
        "reaction": "like",
        "message_id": "target-message-guid",
        "reaction_message_id": "tapback-event-guid",
        "part_index": 0,
        "chat_id": "iMessage;+;family-guid",
        "user_id": "+15555550100",
        "direction": "inbound",
        "status": "received",
        "deduplication_key": _operation().deduplication_key,
    }
    assert "text" not in payload
    assert "participants" not in payload
    assert "fallback_chat" not in payload
