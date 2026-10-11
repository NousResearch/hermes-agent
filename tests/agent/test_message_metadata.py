import pytest

from agent.message_metadata import append_message, record_absorbed_message, stamp_message_timestamp


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


@pytest.mark.parametrize("survivor_metadata,dropped_metadata,dropped_leads,expected", [
    ({"channel_state": {"topic": "old"}, "model_only": True}, {"channel_state": {"topic": "new"}}, False,
     {"channel_state": {"topic": "new"}, "model_only": True}),
    (None, {"channel_state": {"topic": "old"}}, True, {"channel_state": {"topic": "old"}}),
    ({"channel_state": {"topic": "new"}}, {"channel_state": {"topic": "old"}}, True,
     {"channel_state": {"topic": "new"}}),
], ids=["later-row-dropped", "earlier-row-dropped", "later-row-survives"])
def test_fold_keeps_the_later_rows_channel_state(survivor_metadata, dropped_metadata, dropped_leads, expected):
    survivor = {"role": "user", "content": "survivor"}
    if survivor_metadata is not None:
        survivor["display_metadata"] = survivor_metadata
    dropped = {"role": "user", "content": "dropped", "display_metadata": dropped_metadata}

    record_absorbed_message(survivor, dropped, dropped_leads=dropped_leads)

    assert survivor.get("display_metadata") == expected
