import hashlib
import json

from agent.tool_executor import (
    _compact_completed_tool_call_arguments,
    _sync_compacted_tool_call_to_session_db,
)


def _assistant_call(name: str, arguments: dict, call_id: str = "call-1") -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


def test_completed_write_file_payload_is_replaced_by_audit_receipt():
    source = "print('kraken')\n" * 10_000
    message = _assistant_call(
        "write_file",
        {"path": "build_all.py", "content": source},
    )

    removed = _compact_completed_tool_call_arguments(
        [message], tool_call_id="call-1", tool_name="write_file"
    )

    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert removed > 100_000
    assert args["path"] == "build_all.py"
    assert str(len(source)) in args["content"]
    assert hashlib.sha256(source.encode()).hexdigest() in args["content"]
    assert len(args["content"]) < 4_000


def test_small_tool_payload_is_unchanged():
    message = _assistant_call("write_file", {"path": "x.py", "content": "print(1)"})
    original = message["tool_calls"][0]["function"]["arguments"]

    removed = _compact_completed_tool_call_arguments(
        [message], tool_call_id="call-1", tool_name="write_file"
    )

    assert removed == 0
    assert message["tool_calls"][0]["function"]["arguments"] == original


def test_unrelated_tool_call_is_not_compacted():
    source = "x" * 50_000
    message = _assistant_call("terminal", {"command": source})

    removed = _compact_completed_tool_call_arguments(
        [message], tool_call_id="call-1", tool_name="terminal"
    )

    assert removed == 0


def test_compacted_call_is_mirrored_to_preexecution_session_row():
    message = _assistant_call(
        "write_file",
        {"path": "large.py", "content": "x" * 50_000},
    )
    _compact_completed_tool_call_arguments(
        [message], tool_call_id="call-1", tool_name="write_file"
    )

    class FakeDB:
        def __init__(self):
            self.calls = []

        def update_assistant_tool_calls(self, session_id, tool_call_id, tool_calls):
            self.calls.append((session_id, tool_call_id, tool_calls))
            return True

    class FakeAgent:
        session_id = "session-1"
        _session_db = FakeDB()

    _sync_compacted_tool_call_to_session_db(
        FakeAgent(), [message], tool_call_id="call-1"
    )

    assert len(FakeAgent._session_db.calls) == 1
    session_id, call_id, calls = FakeAgent._session_db.calls[0]
    assert session_id == "session-1"
    assert call_id == "call-1"
    assert len(calls[0]["function"]["arguments"]) < 4_000
