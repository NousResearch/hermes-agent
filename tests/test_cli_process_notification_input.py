"""Regression coverage for CLI preservation of raw process notification events."""

from cli import (
    _make_process_notification_input,
    _unpack_process_notification_input,
)


def test_process_notification_queue_payload_preserves_raw_async_event():
    raw_event = {
        "type": "async_delegation",
        "delegation_id": "deleg_raw1",
        "goal": "inspect the raw payload",
        "context": "ctx",
        "summary": "done",
        "nested": {"kept": True},
    }
    formatted_text = "[ASYNC DELEGATION COMPLETE — deleg_raw1]\ndone"

    queued = _make_process_notification_input(raw_event, formatted_text)
    unpacked_text, unpacked_event = _unpack_process_notification_input(queued)

    assert unpacked_text == formatted_text
    assert unpacked_event == raw_event
    assert unpacked_event is raw_event


def test_unpack_plain_input_is_unchanged_without_raw_event():
    text, raw_event = _unpack_process_notification_input("normal user text")

    assert text == "normal user text"
    assert raw_event is None
