"""Recovery stubs retain the details needed to decide whether to open the original."""

import json

from agent.context_compressor import _PRUNE_MIN_CHARS, _is_summary_stub, _summarize_tool_result


def _summary(name, args, body):
    return _summarize_tool_result(name, json.dumps(args), body)


def test_failing_terminal_uses_decoded_lines_and_last_nonempty_line():
    last = "FAILED tests/test_example.py::test_x - AssertionError"
    output = "\n".join([f"progress {i}" for i in range(37)] + [last]) + "\n"
    body = json.dumps({"output": output, "exit_code": 1})
    assert _summary("terminal", {"command": "pytest tests/test_example.py -q"}, body) == (
        "[terminal] ran `pytest tests/test_example.py -q` -> exit 1, 38 lines output | last: " + last)


def test_terminal_stub_stays_under_the_prune_floor():
    """A stub at or over the floor would be summarised again by a later pass and lose its exit code."""
    body = json.dumps({"output": "ERROR " + "y" * 300, "exit_code": 127})
    stub = _summary("terminal", {"command": "c" * 200}, body)
    assert len(stub) < _PRUNE_MIN_CHARS
    assert "-> exit 127, 1 line output | last: ERROR yyy" in stub


def test_delegate_batch_names_count_and_first_goal():
    tasks = [{"goal": "a" * 80}, {"goal": "second"}, {"goal": "third"}]
    stub = _summary("delegate_task", {"tasks": tasks}, "done")
    assert stub == "[delegate_task] 3 tasks, first: '" + "a" * 57 + "...' (4 chars result)"
    assert _is_summary_stub(stub)


def test_generic_mcp_stub_keeps_issue_number():
    stub = _summary("mcp_github_get_issue", {"owner": "org", "repo": "project", "issue_number": 123}, "done")
    assert stub == "[mcp_github_get_issue] owner=org repo=project issue_number=123 (4 chars result)"
    assert _is_summary_stub(stub)


def test_generic_argument_preview_has_a_total_budget():
    args = {f"arg{i}": "v" * 80 for i in range(10)}
    stub = _summary("custom", args, "done")
    expected = (" arg0=" + "v" * 40 + " arg1=" + "v" * 40 + " arg2=" + "v" * 22)
    assert len(expected) == 120
    assert stub == "[custom]" + expected + " (4 chars result)"
    assert "\n" not in stub and len(stub) < 400
    assert _is_summary_stub(stub)
