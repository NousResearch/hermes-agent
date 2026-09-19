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


def test_append_observes_suspicious_assistant_text_without_mutating_content(caplog):
    content = "user\nNew reviews are fine\n" + ("--------\n" * 8)
    message = {"role": "assistant", "content": content}
    messages = []

    with caplog.at_level("WARNING", logger="agent.message_metadata"):
        append_message(messages, message, timestamp=123.0)

    assert message["content"] == content
    assert messages == [message]
    assert "raw_role_prefix=True" in caplog.text
    assert "divider_lines=8" in caplog.text
    assert "content" not in caplog.text


def test_append_ignores_normal_text_and_non_assistant_messages(caplog):
    messages = []
    with caplog.at_level("WARNING", logger="agent.message_metadata"):
        append_message(messages, {"role": "assistant", "content": "Here is the answer."})
        append_message(messages, {"role": "user", "content": "user\nnot assistant output"})

    assert not caplog.records
