"""Pruned tool payloads must not fork a message's display identity.

A compaction generation that shortens tool arguments or tool-result text
rewrites the carried copy of an otherwise unchanged logical event. The display
identity must follow the stable call id, not the mutable payload, or the
archived original and the pruned carried copy project as two messages and the
older progress reappears after the later answer.
"""

import json

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def test_pruned_tool_payload_keeps_one_display_generation(db):
    sid = "pruned-tool-payload"
    db.create_session(sid, source="test")
    db.append_messages_batch(
        sid,
        [
            {"role": "user", "content": "start", "timestamp": 100.0},
            {
                "role": "assistant",
                "content": "Earlier progress",
                "timestamp": 101.0,
                "tool_calls": [
                    {
                        "id": "stable-call",
                        "type": "function",
                        "function": {
                            "name": "demo_tool",
                            "arguments": json.dumps({"value": "L" * 4000}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "R" * 5000,
                "tool_call_id": "stable-call",
                "tool_name": "demo_tool",
                "timestamp": 102.0,
            },
            {"role": "assistant", "content": "Later answer", "timestamp": 200.0},
        ],
    )
    history = db.get_messages_as_conversation(sid, include_row_ids=True)
    history[1]["tool_calls"][0]["function"]["arguments"] = json.dumps({
        "value": "short"
    })
    history[2]["content"] = "short result"
    db.archive_and_compact(sid, history)

    visible = db.get_messages_as_conversation(
        sid, include_row_ids=True, include_compacted=True
    )
    assistant_sequence = [m.get("content") for m in visible if m["role"] == "assistant"]
    assert assistant_sequence == ["Earlier progress", "Later answer"]


def test_pruned_tool_result_page_position_survives(db):
    sid = "pruned-tool-result-page"
    db.create_session(sid, source="test")
    db.append_messages_batch(
        sid,
        [
            {"role": "user", "content": "q", "timestamp": 100.0},
            {
                "role": "assistant",
                "content": "progress",
                "timestamp": 101.0,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "demo_tool",
                            "arguments": json.dumps({"value": "L" * 3000}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "R" * 3000,
                "tool_call_id": "c1",
                "tool_name": "demo_tool",
                "timestamp": 102.0,
            },
            {"role": "assistant", "content": "answer", "timestamp": 200.0},
        ],
    )
    history = db.get_messages_as_conversation(sid, include_row_ids=True)
    history[2]["content"] = "short"
    db.archive_and_compact(sid, history)

    page = db.get_messages(sid, include_compacted=True, limit=100)
    roles = [m["role"] for m in page]
    assert roles.count("tool") == 1, page
    assert [m.get("content") for m in page if m["role"] == "assistant"] == [
        "progress",
        "answer",
    ]
