"""A templated compaction stub must carry the outcome of the call (#131244).

`_sum_template` formatted from the tool arguments plus the result's length, so a refused
`read_file`, a rate-limited `web_search`, a dead cron run or a non-zero process exit compressed
into the same stub as the success it never was, and the post-compaction agent reported the success.
"""

import json

import pytest

from agent.context_compressor import _summarize_tool_result


def _stub(tool_name, args, payload):
    return _summarize_tool_result(tool_name, json.dumps(args), json.dumps(payload))


@pytest.mark.parametrize(
    "tool_name, args, payload, reason",
    [
        ("read_file", {"path": "gone.py", "offset": 1}, {"error": "File not found: gone.py"}, "File not found: gone.py"),
        ("web_search", {"query": "hermes"}, {"error": "rate limited"}, "rate limited"),
        ("memory", {"action": "add", "target": "a note"}, {"error": "unknown action"}, "unknown action"),
        ("text_to_speech", {}, {"error": "no voice available"}, "no voice available"),
        ("cronjob_manage", {"action": "create"}, {"success": False}, ""),
        ("cronjob_manage", {"action": "run"},
         {"success": True, "job": {"execution_success": False, "execution_error": "agent exited with code 1"}},
         "agent exited with code 1"),
        ("process_manage", {"action": "poll", "session_id": "p1"},
         {"status": "exited", "exit_code": 1, "completion_reason": "nonzero_exit"}, "exit code 1"),
        # Every MCP/plugin tool falls through to the generic stub.
        ("some_mcp_tool", {"a": 1}, {"error": "boom"}, "boom"),
    ],
)
def test_failed_call_stub_is_marked_failed(tool_name, args, payload, reason):
    stub = _stub(tool_name, args, payload)
    assert " FAILED" in stub and reason in stub and "\n" not in stub, stub


@pytest.mark.parametrize(
    "tool_name, args, payload, expected",
    [
        ("web_search", {"query": "hermes"}, {"results": [{"title": "hit"}]}, None),
        ("text_to_speech", {}, {"success": True, "path": "/tmp/out.wav"}, None),
        # ``job`` carries stored state from earlier runs; only this call's outcome may mark the stub.
        ("cronjob_manage", {"action": "poll"},
         {"success": True, "job": {"error": "last run failed", "execution_success": True}}, "[cronjob] poll"),
        ("process_manage", {"action": "poll", "session_id": "p1"}, {"status": "exited", "exit_code": 0},
         "[process] poll session=p1"),
        ("process_manage", {"action": "poll", "session_id": "p1"}, {"status": "running", "pid": 4242},
         "[process] poll session=p1"),
    ],
)
def test_successful_call_stub_is_not_marked(tool_name, args, payload, expected):
    stub = _stub(tool_name, args, payload)
    assert "FAILED" not in stub, stub
    if expected is not None:
        assert stub == expected
