from agent.message_metadata import (
    PERSISTENCE_ONLY_MESSAGE_FIELDS,
    append_message,
    stamp_message_timestamp,
    without_persistence_fields,
)


def test_stamp_preserves_source_timestamp(monkeypatch):
    monkeypatch.setattr("agent.message_metadata.wall_time", lambda: 999.0)
    message = {"role": "user", "content": "hello", "timestamp": 123.0}

    result = stamp_message_timestamp(message)

    assert result is message
    assert message["timestamp"] == 123.0


def test_append_stamps_same_mapping_at_append_time(monkeypatch):
    monkeypatch.setattr("agent.message_metadata.wall_time", lambda: 456.0)
    messages = []
    message = {"role": "tool", "content": "ok"}

    result = append_message(messages, message)

    assert result is message
    assert messages == [message]
    assert message["timestamp"] == 456.0


def test_without_persistence_fields_drops_replayed_session_metadata():
    """The DB-replay carriers are persistence-only and must not reach a strict backend.

    ``hermes_state_messages._rows_to_conversation`` restores ``message_id`` (the
    JSONL-transcript alias of the ``platform_message_id`` column), ``observed`` and
    ``finish_reason`` onto replayed messages. ``turn_context`` pops this set from every
    outgoing copy; a strict Chat Completions schema (Fireworks' ``ChatMessage`` is
    ``additionalProperties: false``) rejects the whole request with a non-retryable 400
    when one leaks (#48523).
    """
    message = {
        "role": "assistant",
        "content": "hi",
        "message_id": "7515969557947007112",
        "observed": True,
        "finish_reason": "stop",
    }

    result = without_persistence_fields(message)

    for carrier in ("message_id", "observed", "finish_reason"):
        assert carrier in PERSISTENCE_ONLY_MESSAGE_FIELDS
        assert carrier not in result
    assert result["role"] == "assistant"
    assert result["content"] == "hi"
    # The durable mapping keeps every carrier: persistence and JSONL export read the live dict.
    assert message["message_id"] == "7515969557947007112"
    assert message["finish_reason"] == "stop"
