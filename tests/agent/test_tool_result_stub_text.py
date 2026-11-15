"""Recovery stubs retain the details needed to decide whether to open the original."""

import json

import pytest

from agent.context_compressor import _PRUNE_MIN_CHARS, _is_summary_stub, _summarize_tool_result


def _summary(name, args, body):
    return _summarize_tool_result(name, json.dumps(args), body)


def test_failing_terminal_uses_decoded_lines_and_last_nonempty_line():
    last = "FAILED tests/test_example.py::test_x - AssertionError"
    output = "\n".join([f"progress {i}" for i in range(37)] + [last]) + "\n"
    body = json.dumps({"output": output, "exit_code": 1})
    assert _summary("terminal", {"command": "pytest tests/test_example.py -q"}, body) == (
        "[terminal] ran `pytest tests/test_example.py -q` -> exit 1, 38 lines output | last: " + last)


@pytest.mark.parametrize("body,expected", [
    (json.dumps({"output": "ok\npassed\n", "exit_code": 0}), "exit 0, 2 lines output"),
    (json.dumps({"output": "started", "process_id": "bg"}), "exit ?, 1 line output"),
    ("raw\noutput", "exit ?, 2 lines output"),
    (json.dumps({"output": ["not", "text"], "exit_code": 0}), "exit 0, 1 line output"),
])
def test_terminal_success_background_and_fallback(body, expected):
    stub = _summary("terminal", {"command": "check"}, body)
    assert stub == f"[terminal] ran `check` -> {expected}"
    assert " | last:" not in stub


def test_terminal_last_line_is_trimmed_and_bounded():
    body = json.dumps({"output": "progress\n  " + "x" * 150 + "  \n\n", "exit_code": 2})
    stub = _summary("terminal", {"command": "check\nmore"}, body)
    assert stub == "[terminal] ran `check more` -> exit 2, 3 lines output | last: " + "x" * 120
    assert "\n" not in stub and len(stub) < _PRUNE_MIN_CHARS


def test_terminal_stub_stays_under_the_prune_floor():
    """A stub at or over the floor would be summarised again by a later pass and lose its exit code."""
    body = json.dumps({"output": "ERROR " + "y" * 300, "exit_code": 127})
    stub = _summary("terminal", {"command": "c" * 200}, body)
    assert len(stub) < _PRUNE_MIN_CHARS
    assert "-> exit 127, 1 line output | last: ERROR yyy" in stub


def test_delegate_single_goal_is_unchanged():
    assert _summary("delegate_task", {"goal": "inspect"}, "done") == "[delegate_task] 'inspect' (4 chars result)"


def test_delegate_batch_names_count_and_first_goal():
    tasks = [{"goal": "a" * 80}, {"goal": "second"}, {"goal": "third"}]
    stub = _summary("delegate_task", {"tasks": tasks}, "done")
    assert stub == "[delegate_task] 3 tasks, first: '" + "a" * 57 + "...' (4 chars result)"
    assert _is_summary_stub(stub)


def test_delegate_tasks_win_over_a_goal_like_in_the_tool():
    """The tool runs the batch whenever tasks is non-empty; an empty list means the single goal."""
    both = _summary("delegate_task", {"goal": "ignored", "tasks": [{"goal": "one"}, {"goal": "two"}]}, "done")
    assert both == "[delegate_task] 2 tasks, first: 'one' (4 chars result)"
    assert _summary("delegate_task", {"goal": "inspect", "tasks": []}, "done") == "[delegate_task] 'inspect' (4 chars result)"


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
