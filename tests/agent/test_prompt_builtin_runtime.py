"""Structured completion contract for prompt-backed built-ins."""
from unittest.mock import patch

from agent.prompt_builtin_runtime import (
    begin_prompt_builtin,
    finish_prompt_builtin,
    make_prompt_builtin_origin,
    observe_tool_result,
)
from hermes_cli.plugins import VALID_HOOKS


def _finish(origin, result=None):
    token = begin_prompt_builtin(origin)
    return finish_prompt_builtin(
        token,
        result or {"completed": True},
        session_id="session-1",
        task_id="task-1",
        platform="cli",
    )


def test_learn_completion_groups_create_patch_and_multi_file_artifacts():
    origin = {"name": "learn", "raw_args": "the workflow", "run_id": "run-1"}
    token = begin_prompt_builtin(origin)

    observe_tool_result(
        "skill_manage",
        {"operations": [{"name": "workflow", "action": "create"}]},
        '{"success": true, "path": "/skills/workflow"}',
    )
    observe_tool_result(
        "skill_manage",
        {"operations": [{"name": "workflow", "action": "patch"}]},
        '{"success": true, "path": "/skills/workflow"}',
    )
    observe_tool_result(
        "skill_manage",
        {"operations": [
            {"name": "workflow", "action": "write_file", "file_path": "references/a.md"},
            {"name": "workflow", "action": "write_file", "file_path": "references/b.md"},
        ]},
        '{"success": true, "results": ['
        '{"name": "workflow", "action": "write_file", "file_path": "references/a.md", '
        '"path": "/skills/workflow/references/a.md"},'
        '{"name": "workflow", "action": "write_file", "file_path": "references/b.md", '
        '"path": "/skills/workflow/references/b.md"}]}',
    )

    emitted = []
    with patch("hermes_cli.lifecycle.has_hook", return_value=True), patch(
        "hermes_cli.lifecycle.invoke_hook", side_effect=lambda name, **payload: emitted.append((name, payload))
    ):
        result = finish_prompt_builtin(
            token, {"completed": True}, session_id="session-1", task_id="task-1", platform="cli"
        )

    completion = result["prompt_builtin_completion"]
    assert completion == emitted[0][1]
    assert emitted[0][0] == "post_prompt_builtin_run"
    assert completion["command"] == "learn"
    assert completion["request"] == "the workflow"
    assert completion["run_id"] == "run-1"
    assert completion["session_id"] == "session-1"
    assert [artifact["action"] for artifact in completion["artifacts"]] == [
        "create", "patch", "write_file", "write_file"
    ]
    assert [artifact["path"] for artifact in completion["artifacts"][-2:]] == [
        "/skills/workflow/references/a.md", "/skills/workflow/references/b.md"
    ]


def test_staged_and_failed_skill_writes_are_not_reported_as_saved():
    staged = {"name": "learn", "raw_args": "", "run_id": "staged-run"}
    token = begin_prompt_builtin(staged)
    observe_tool_result(
        "skill_manage",
        {"operations": [{"name": "draft", "action": "create"}]},
        '{"success": true, "staged": true, "pending_id": "pending-1"}',
    )
    staged_result = finish_prompt_builtin(
        token, {"completed": True}, session_id="s", task_id="t", platform="desktop"
    )
    assert staged_result["prompt_builtin_completion"]["artifacts"] == [{
        "kind": "skill", "name": "draft", "action": "create", "status": "staged", "path": None
    }]

    failed = {"name": "learn", "raw_args": "", "run_id": "failed-run"}
    token = begin_prompt_builtin(failed)
    observe_tool_result(
        "skill_manage",
        {"operations": [{"name": "draft", "action": "patch"}]},
        '{"success": false, "error": "no match"}',
    )
    failed_result = finish_prompt_builtin(
        token, {"completed": True}, session_id="s", task_id="t", platform="telegram"
    )
    assert failed_result["prompt_builtin_completion"]["artifacts"][0]["status"] == "failed"


def test_ordinary_skill_manage_call_has_no_prompt_builtin_completion():
    observe_tool_result(
        "skill_manage",
        {"operations": [{"name": "ordinary", "action": "patch"}]},
        '{"success": true, "path": "/skills/ordinary"}',
    )
    result = {"completed": True}
    assert finish_prompt_builtin(None, result, session_id="s", task_id="t", platform="cli") is result
    assert "prompt_builtin_completion" not in result


def test_prompt_builtin_hook_is_registered_and_turn_status_is_structured():
    assert "post_prompt_builtin_run" in VALID_HOOKS
    interrupted = _finish(make_prompt_builtin_origin("plan", "ship it"), {"interrupted": True})
    assert interrupted["prompt_builtin_completion"]["status"] == "interrupted"
