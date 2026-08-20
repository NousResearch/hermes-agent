import json

from agent.tool_occurrences import tool_result_metadata_by_index


def call(ref, name, args, *, call_id=None):
    tc = {
        "id": ref,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, separators=(",", ":")),
        },
    }
    if call_id is not None:
        tc["call_id"] = call_id
    return {"role": "assistant", "content": None, "tool_calls": [tc]}


def result(ref, content):
    return {"role": "tool", "tool_call_id": ref, "content": content}


def test_reused_raw_id_after_completion_keeps_occurrence_metadata():
    messages = [
        call("x", "read_file", {"path": "/a"}),
        result("x", "RESULT_A"),
        {"role": "user", "content": "next"},
        call("x", "terminal", {"command": "build"}),
        result("x", "RESULT_B"),
    ]

    metadata = tool_result_metadata_by_index(messages)

    assert metadata[1][0] == "read_file"
    assert metadata[4][0] == "terminal"


def test_call_id_matches_composite_result_reference():
    messages = [
        call("fc_123", "read_file", {"path": "/a"}, call_id="call_abc"),
        result("call_abc|fc_123", "RESULT"),
    ]

    metadata = tool_result_metadata_by_index(messages)

    assert metadata[1][0] == "read_file"
    assert json.loads(metadata[1][1])["path"] == "/a"


def test_result_can_match_secondary_id_alias():
    messages = [
        call("fc_123", "read_file", {"path": "/a"}, call_id="call_abc"),
        result("fc_123", "RESULT"),
    ]

    assert tool_result_metadata_by_index(messages)[1][0] == "read_file"


def test_simultaneous_duplicate_alias_is_not_guessed():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "x", "function": {"name": "first", "arguments": "{}"}},
                {"id": "x", "function": {"name": "second", "arguments": "{}"}},
            ],
        },
        result("x", "ambiguous"),
    ]

    assert 1 not in tool_result_metadata_by_index(messages)


def test_orphan_result_has_no_metadata():
    assert tool_result_metadata_by_index([result("missing", "orphan")]) == {}

