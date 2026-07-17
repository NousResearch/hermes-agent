"""Tests for normalized message addressing and dispatch disposition."""

from gateway.platforms import (
    MessageAddressing,
    MessageDisposition,
    MessageEvent,
)


def test_message_event_defaults_to_dispatch_with_empty_addressing():
    event = MessageEvent(text="x")

    assert event.disposition is MessageDisposition.DISPATCH
    assert event.addressing == MessageAddressing()
    assert event.metadata == {}


def test_message_event_keeps_addressing_and_disposition_out_of_metadata():
    addressing = MessageAddressing(
        direct_mention=True,
        reply_to_self=True,
        mentions_other_bots=True,
        mentions_other_users=True,
    )
    metadata = {"platform_signal": "kept"}

    event = MessageEvent(
        text="observe this",
        addressing=addressing,
        disposition=MessageDisposition.OBSERVE,
        metadata=metadata,
    )

    assert event.addressing is addressing
    assert event.disposition is MessageDisposition.OBSERVE
    assert [member.value for member in MessageDisposition] == [
        "drop",
        "observe",
        "dispatch",
    ]
    assert event.metadata is metadata
    assert event.metadata == {"platform_signal": "kept"}
