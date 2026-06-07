from pathlib import Path


def test_conversation_loop_prefixes_every_string_api_message_before_user_context():
    source = Path("agent/conversation_loop.py").read_text(encoding="utf-8")

    assert "add_sent_timestamp_prefix as _stamp_message_time" in source
    assert "api_msg[\"content\"] = _stamp_message_time" in source
    assert "msg.get(\"timestamp\")" in source
    assert source.index("_stamp_message_time(") < source.index("if _ext_prefetch_cache:")
    assert "cached system prompt stays" in source


def test_session_replay_surfaces_message_timestamp_metadata():
    source = Path("hermes_state.py").read_text(encoding="utf-8")

    assert "timestamp, finish_reason" in source
    assert "msg[\"timestamp\"] = row[\"timestamp\"]" in source
