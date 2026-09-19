from agent.message_metadata import append_message, stamp_message_timestamp


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


def test_append_replaces_same_durable_tail_but_keeps_identical_distinct_rows():
    existing = {"role": "user", "content": "raw prompt", "timestamp": 1.0, "_row_id": 7}
    adopted = {"role": "user", "content": "expanded prompt", "timestamp": 1.0, "_row_id": 7}
    repeated = {"role": "user", "content": "expanded prompt", "timestamp": 2.0, "_row_id": 8}
    messages = [existing]

    append_message(messages, adopted)

    assert messages == [adopted]

    append_message(messages, repeated)

    assert messages == [adopted, repeated]
