"""Runtime-issued Goal receipts from central file-mutation dispatch."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.goals import GoalContract, GoalLanding, GoalManager, GoalToolConstraints
from model_tools import handle_function_call


class _Registry:
    def __init__(self, capabilities=("mutate",), result=None):
        self.entry = SimpleNamespace(capabilities=capabilities)
        self.result = result or json.dumps({"ok": True})
        self.calls = []

    def get_entry(self, name):
        return self.entry

    def dispatch(self, name, args, **kwargs):
        self.calls.append((name, args, kwargs))
        return self.result


def _set_goal(session_id: str, target: str, required_state: str = "WRITTEN") -> GoalManager:
    manager = GoalManager(session_id)
    manager.set(
        "Exercise runtime mutation receipts",
        contract=GoalContract(
            landing=GoalLanding(required_state=required_state, targets=[target]),
        ),
    )
    return manager


def _receipts(session_id: str):
    return GoalManager(session_id).state.receipts


def _call_fake(registry, session_id: str, *, name="goal_test_mutate", args=None, **ids):
    with patch("model_tools.registry", registry):
        return handle_function_call(
            name,
            args or {},
            session_id=session_id,
            skip_tool_execution_middleware=True,
            **ids,
        )


def test_actual_write_file_issues_written_runtime_receipt(tmp_path: Path):
    session_id = "goal-receipt-actual-write"
    target = str((tmp_path / "written.txt").resolve())
    _set_goal(session_id, target)

    result = json.loads(
        handle_function_call(
            "write_file",
            {"path": target, "content": "hello\n"},
            session_id=session_id,
            task_id="task-write",
            tool_call_id="call-write",
        )
    )

    assert not result.get("error")
    assert Path(target).read_text(encoding="utf-8") == "hello\n"
    receipts = _receipts(session_id)
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.target == target
    assert receipt.state == "WRITTEN"
    assert receipt.scope == "file"
    assert receipt.source_reference == "tool:write_file:call-write"
    assert receipt.runtime_issued is True
    assert receipt.mutation_generation == 1
    assert receipt.verification_generation == 0


def test_actual_patch_advances_generation_and_stales_old_verification(tmp_path: Path):
    session_id = "goal-receipt-actual-patch"
    path = tmp_path / "patched.txt"
    path.write_text("before\n", encoding="utf-8")
    target = str(path.resolve())
    manager = _set_goal(session_id, target, required_state="DEPLOYED")
    manager.issue_change_receipt(
        target=target,
        scope="deploy",
        source_reference="test:deploy",
        state="DEPLOYED",
        mutation=True,
    )
    manager.issue_change_receipt(
        target=target,
        scope="verify",
        source_reference="test:verify",
        state="DEPLOYED",
    )
    assert manager.landing_status()["fulfilled"] is True

    task_id = "task-patch"
    read_result = json.loads(
        handle_function_call(
            "read_file",
            {"path": target},
            session_id=session_id,
            task_id=task_id,
            tool_call_id="call-read-before-patch",
        )
    )
    assert not read_result.get("error")

    result = json.loads(
        handle_function_call(
            "patch",
            {"path": target, "old_string": "before\n", "new_string": "after\n"},
            session_id=session_id,
            task_id=task_id,
            tool_call_id="call-patch",
        )
    )

    assert not result.get("error")
    assert path.read_text(encoding="utf-8") == "after\n"
    receipts = _receipts(session_id)
    assert receipts[-1].state == "WRITTEN"
    assert receipts[-1].mutation_generation == 2
    status = GoalManager(session_id).landing_status()
    assert status["fulfilled"] is False
    assert status["targets"][0]["observed_state"] == "WRITTEN"
    assert status["targets"][0]["reason"] == "insufficient-state"


def test_failed_actual_mutation_does_not_issue_receipt(tmp_path: Path):
    session_id = "goal-receipt-failed-mutation"
    path = tmp_path / "unchanged.txt"
    path.write_text("present\n", encoding="utf-8")
    target = str(path.resolve())
    _set_goal(session_id, target)
    task_id = "task-failed-patch"
    handle_function_call(
        "read_file",
        {"path": target},
        session_id=session_id,
        task_id=task_id,
        tool_call_id="call-read-failed-patch",
    )

    result = json.loads(
        handle_function_call(
            "patch",
            {"path": target, "old_string": "missing", "new_string": "changed"},
            session_id=session_id,
            task_id=task_id,
            tool_call_id="call-failed-patch",
        )
    )

    assert result.get("error")
    assert _receipts(session_id) == []


def test_blocked_mutation_does_not_issue_receipt():
    session_id = "goal-receipt-blocked-mutation"
    target = "C:/declared/blocked.txt"
    manager = GoalManager(session_id)
    manager.set(
        "Block runtime mutation",
        contract=GoalContract(
            landing=GoalLanding(required_state="WRITTEN", targets=[target]),
            tool_constraints=GoalToolConstraints(denied_capabilities=["mutate"]),
        ),
    )
    registry = _Registry(result=json.dumps({"files_modified": [target]}))

    result = json.loads(
        _call_fake(
            registry,
            session_id,
            args={"path": target},
            tool_call_id="call-blocked",
        )
    )

    assert result.get("error")
    assert registry.calls == []
    assert _receipts(session_id) == []


def test_read_capability_cannot_forge_receipt_from_files_modified():
    session_id = "goal-receipt-read-forgery"
    target = "C:/declared/read-target.txt"
    _set_goal(session_id, target)
    registry = _Registry(
        capabilities=("read",),
        result=json.dumps({"files_modified": target}),
    )

    _call_fake(
        registry,
        session_id,
        name="read_file",
        args={"path": target},
        tool_call_id="call-read-forgery",
    )

    assert _receipts(session_id) == []


def test_undeclared_target_does_not_issue_receipt():
    session_id = "goal-receipt-undeclared-target"
    declared = "C:/declared/target.txt"
    other = "C:/outside/other.txt"
    _set_goal(session_id, declared)
    registry = _Registry(result=json.dumps({"files_modified": [other]}))

    _call_fake(
        registry,
        session_id,
        args={"path": declared},
        tool_call_id="call-undeclared",
    )

    assert _receipts(session_id) == []


def test_duplicate_declared_targets_issue_one_receipt_each():
    session_id = "goal-receipt-unique-targets"
    first = "C:/declared/first.txt"
    second = "C:/declared/second.txt"
    manager = GoalManager(session_id)
    manager.set(
        "Record unique targets",
        contract=GoalContract(
            landing=GoalLanding(required_state="WRITTEN", targets=[first, second]),
        ),
    )
    registry = _Registry(result=json.dumps({"files_modified": [first, first, second]}))

    _call_fake(
        registry,
        session_id,
        tool_call_id="call-unique",
    )

    receipts = _receipts(session_id)
    assert [receipt.target for receipt in receipts] == [first, second]


def test_scalar_files_modified_issues_receipt():
    session_id = "goal-receipt-scalar-target"
    target = "C:/declared/scalar.txt"
    _set_goal(session_id, target)
    registry = _Registry(result=json.dumps({"files_modified": target}))

    _call_fake(
        registry,
        session_id,
        tool_call_id="call-scalar",
    )

    receipts = _receipts(session_id)
    assert len(receipts) == 1
    assert receipts[0].target == target


def test_paused_goal_uses_file_path_fallback_and_task_provenance():
    session_id = "goal-receipt-paused-fallback"
    target = "C:/declared/paused-fallback.txt"
    manager = _set_goal(session_id, target)
    manager.pause("waiting for operator")
    registry = _Registry(result=json.dumps({"ok": True}))

    _call_fake(
        registry,
        session_id,
        args={"file_path": target},
        task_id="task-fallback",
        api_request_id="api-fallback",
        turn_id="turn-fallback",
    )

    receipts = _receipts(session_id)
    assert len(receipts) == 1
    assert receipts[0].target == target
    assert receipts[0].source_reference == "tool:goal_test_mutate:task-fallback"


def test_api_request_precedes_turn_provenance():
    session_id = "goal-receipt-api-provenance"
    target = "C:/declared/api-provenance.txt"
    _set_goal(session_id, target)
    registry = _Registry(result=json.dumps({"files_modified": [target]}))

    _call_fake(
        registry,
        session_id,
        api_request_id="api-priority",
        turn_id="turn-lower-priority",
    )

    receipts = _receipts(session_id)
    assert len(receipts) == 1
    assert receipts[0].source_reference == "tool:goal_test_mutate:api-priority"


def test_missing_runtime_provenance_does_not_issue_receipt():
    session_id = "goal-receipt-missing-provenance"
    target = "C:/declared/no-provenance.txt"
    _set_goal(session_id, target)
    registry = _Registry(result=json.dumps({"files_modified": target}))

    _call_fake(registry, session_id, args={"path": target})

    assert _receipts(session_id) == []


def test_present_empty_files_modified_does_not_fallback_to_args():
    session_id = "goal-receipt-empty-result-targets"
    target = "C:/declared/empty-result.txt"
    _set_goal(session_id, target)
    registry = _Registry(result=json.dumps({"files_modified": []}))

    _call_fake(
        registry,
        session_id,
        args={"path": target},
        tool_call_id="call-empty-result",
    )

    assert _receipts(session_id) == []


def test_done_goal_does_not_issue_receipt():
    session_id = "goal-receipt-done-state"
    target = "C:/declared/done.txt"
    manager = _set_goal(session_id, target)
    manager.mark_done("already complete")
    registry = _Registry(result=json.dumps({"files_modified": [target]}))

    _call_fake(
        registry,
        session_id,
        tool_call_id="call-done",
    )

    assert _receipts(session_id) == []


def test_malformed_files_modified_does_not_fallback_to_args():
    session_id = "goal-receipt-malformed-result-targets"
    target = "C:/declared/malformed.txt"
    _set_goal(session_id, target)
    registry = _Registry(result=json.dumps({"files_modified": [target, 7]}))

    _call_fake(
        registry,
        session_id,
        args={"path": target},
        tool_call_id="call-malformed",
    )

    assert _receipts(session_id) == []


def test_transform_hook_cannot_manufacture_receipt():
    session_id = "goal-receipt-transform-forgery"
    target = "C:/declared/transformed.txt"
    undeclared = "C:/outside/raw-args.txt"
    _set_goal(session_id, target)
    registry = _Registry(result=json.dumps({"ok": True}))
    transformed = json.dumps({"ok": True, "files_modified": [target]})

    with (
        patch("model_tools.registry", registry),
        patch(
            "hermes_cli.lifecycle.has_hook",
            side_effect=lambda hook_name: hook_name == "transform_tool_result",
        ),
        patch("hermes_cli.lifecycle.invoke_hook", return_value=[transformed]),
    ):
        result = handle_function_call(
            "goal_test_mutate",
            {"path": undeclared},
            session_id=session_id,
            tool_call_id="call-transform",
            skip_tool_execution_middleware=True,
        )

    assert result == transformed
    assert _receipts(session_id) == []


def test_receipt_recording_failure_is_logged_without_changing_tool_result(caplog):
    session_id = "goal-receipt-recording-failure"
    target = "C:/declared/log-only.txt"
    _set_goal(session_id, target)
    raw_result = json.dumps({"ok": True, "files_modified": [target]})
    registry = _Registry(result=raw_result)

    with (
        patch("model_tools.registry", registry),
        patch(
            "hermes_cli.goals.GoalManager.issue_change_receipt",
            side_effect=RuntimeError("receipt store unavailable"),
        ),
        caplog.at_level(logging.WARNING, logger="model_tools"),
    ):
        result = handle_function_call(
            "goal_test_mutate",
            {"path": target},
            session_id=session_id,
            tool_call_id="call-log-only",
            skip_tool_execution_middleware=True,
        )

    assert result == raw_result
    assert "receipt store unavailable" in caplog.text
    assert _receipts(session_id) == []
