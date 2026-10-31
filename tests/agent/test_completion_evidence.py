"""Completion evidence read from a transcript, and the todo_list result it annotates."""

import json

import pytest

from agent.completion_evidence import annotate_todo_result, check_after_last_edit, tool_events


def _call(cid, name, **args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _turn(*pairs):
    """``(call, result)`` pairs → assistant(tool_calls) + tool rows, one batch per pair."""
    rows = [{"role": "user", "content": "fix the bug"}]
    for call, result in pairs:
        rows.append({"role": "assistant", "content": "", "tool_calls": [call]})
        rows.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
    return rows


EDIT = (_call("e1", "patch", path="src/app.py", old_string="a", new_string="b"),
        {"success": True, "diff": "-a\n+b", "files_modified": ["src/app.py"]})


def _term(cid, command, exit_code, **extra):
    return _call(cid, "terminal", command=command), {"output": "", "exit_code": exit_code, "error": None, **extra}


@pytest.mark.parametrize("pairs,state", [
    ([], "no_edits"),
    ([EDIT], "unchecked"),
    ([_term("t0", "pytest -q", 0), EDIT], "unchecked"),  # a check BEFORE the edit proves nothing
    ([EDIT, _term("t1", "ls -la", 0)], "unchecked"),  # not a check
    ([EDIT, _term("t1", "python -m pytest tests/ -q", 0)], "verified"),
    ([EDIT, _term("t1", "npm run build", 2)], "failed"),
    ([EDIT, _term("t1", "pytest -q | tail -5", 0)], "hidden"),
    ([EDIT, _term("t1", "pytest -q || true", 0)], "hidden"),
    # The ledger's own verdict (pipeline-aware) wins over the command shape.
    ([EDIT, _term("t1", "pytest -q | tail -5", 0, verification_evidence={"status": "passed"})], "verified"),
])
def test_check_after_last_edit(pairs, state):
    assert check_after_last_edit(tool_events(_turn(*pairs)))["state"] == state


def test_failed_edit_is_not_an_edit():
    failed = (_call("e1", "patch", path="src/app.py", old_string="zz", new_string="b"),
              {"success": False, "error": "old_string not found"})
    assert check_after_last_edit(tool_events(_turn(failed)))["state"] == "no_edits"


def _todo_result(*items):
    return json.dumps({"todos": [{"id": i, "content": i, "status": s} for i, s in items], "revision": 2,
                       "summary": {}})


def test_completed_after_unchecked_edit_is_marked_unverified():
    out = json.loads(annotate_todo_result(_todo_result(("fix", "completed"), ("docs", "pending")),
                                          {"fix": "in_progress", "docs": "pending"}, _turn(EDIT)))
    assert out["unverified"]["ids"] == ["fix"]
    assert "no test/build/CI command ran after the last edit" in out["unverified"]["reason"]
    assert [t["status"] for t in out["todos"]] == ["completed", "pending"]  # the write itself stands


@pytest.mark.parametrize("pairs,before", [
    ([EDIT, _term("t1", "pytest -q", 0)], {"fix": "in_progress"}),  # verified
    ([], {"fix": "in_progress"}),  # nothing edited (research, chat)
    ([EDIT], {"fix": "completed"}),  # already completed before this write: no repeat
])
def test_no_annotation_when_verified_unedited_or_not_newly_completed(pairs, before):
    result = _todo_result(("fix", "completed"))
    assert annotate_todo_result(result, before, _turn(*pairs)) == result


def test_todo_list_tool_result_carries_the_annotation():
    """Through the real todo_list dispatch: the model sees the mark on the completing write."""
    from types import SimpleNamespace

    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
    from tools.todo_tool import TodoStore

    agent = SimpleNamespace(_todo_store=TodoStore())
    run = INLINE_TOOL_EXECUTORS["todo_list"]
    run(agent, {"todos": [{"id": "fix", "content": "fix the bug", "status": "in_progress"}]},
        InlineToolContext(effective_task_id="t"))
    out = json.loads(run(agent, {"todos": [{"id": "fix", "status": "completed"}], "merge": True},
                         InlineToolContext(effective_task_id="t", messages=_turn(EDIT))))
    assert out["todos"][0]["status"] == "completed"
    assert out["unverified"]["ids"] == ["fix"]
