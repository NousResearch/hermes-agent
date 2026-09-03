from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.feature_delivery import (
    FeatureDeliveryState,
    RECOVERABLE_BLOCK_CODES,
    is_recoverable_block,
    resolve_resume_target,
)
from hermes_cli.feature_delivery_runner import (
    DeliveryRunnerError,
    FeatureDeliveryRunner,
    StageExecutionError,
)
from hermes_cli.subcommands.delivery import delivery_command
from tests.hermes_cli.test_feature_delivery_runner import (
    FakeExecutor,
    create,
    delivery_env,
    git,
    stage_rows,
)


class FailingExecutor:
    def __init__(self, code: str):
        self.code = code

    def execute(self, **_kwargs):
        raise StageExecutionError(self.code, f"{self.code} for test")


def block_in_development(delivery_env, code: str):
    runner, root_id = create(delivery_env)
    with kb.connect() as conn:
        root, _ = runner._root_and_metadata(conn, root_id)
        runner._transition(conn, root, FeatureDeliveryState.DEVELOPING)
        root = kb.get_task(conn, root_id)
        runner._block(conn, root, code, "terminal test block")
    return runner, root_id


@pytest.mark.parametrize("code", sorted(RECOVERABLE_BLOCK_CODES))
def test_recoverable_block_policy_allows_only_explicit_codes(code):
    assert is_recoverable_block(code)


@pytest.mark.parametrize(
    "code",
    [
        "invalid_report",
        "max_fix_loops_reached",
        "contract_hash_mismatch",
        "commit_mismatch",
        "dirty_worktree",
        "tester_modified_source",
        "acceptance_gate_denied",
    ],
)
def test_terminal_integrity_blocks_cannot_be_unblocked(delivery_env, code):
    runner, root_id = block_in_development(delivery_env, code)
    with pytest.raises(DeliveryRunnerError, match="not recoverable"):
        runner.unblock(root_id, confirmed=True)
    assert runner.status(root_id).current_state == "BLOCKED"


def test_unblock_requires_blocked_state_and_explicit_confirmation(delivery_env):
    runner, root_id = create(delivery_env)
    with pytest.raises(DeliveryRunnerError, match="not BLOCKED"):
        runner.unblock(root_id, confirmed=True)

    blocked, blocked_id = create(
        delivery_env,
        FakeExecutor(delivery_env.repo, [("developer", "blocked")]),
    )
    blocked.run(blocked_id)
    with pytest.raises(DeliveryRunnerError, match="confirm"):
        blocked.unblock(blocked_id)


def test_cli_unblock_requires_confirmation_and_does_not_run_agent(
    delivery_env, capsys
):
    fake = FakeExecutor(delivery_env.repo, [("developer", "blocked")])
    runner, root_id = create(delivery_env, fake)
    runner.run(root_id)
    args = argparse.Namespace(
        delivery_command="unblock",
        task_id=root_id,
        resume_stage="previous",
        confirm=False,
    )
    with pytest.raises(DeliveryRunnerError, match="confirm"):
        delivery_command(args, runner=runner)

    args.confirm = True
    assert delivery_command(args, runner=runner) == 0
    assert "Current State: DEVELOPING" in capsys.readouterr().out
    assert [call[0] for call in fake.calls] == ["developer"]


@pytest.mark.parametrize(
    ("executor", "expected_code"),
    [
        (None, "stage_executor_missing"),
        (FailingExecutor("profile_missing"), "profile_missing"),
        (FailingExecutor("stage_execution_failed"), "stage_execution_failed"),
    ],
)
def test_recoverable_executor_blocks_return_to_previous_stage(
    delivery_env, executor, expected_code
):
    runner, root_id = create(delivery_env, executor)
    assert expected_code in runner.run(root_id).blocked_reason
    status = runner.unblock(root_id, confirmed=True)
    assert status.current_state == "DEVELOPING"
    assert status.blocked_reason is None
    assert expected_code in status.last_blocked_reason


def test_resume_target_policy_rejects_arbitrary_states():
    assert (
        resolve_resume_target(FeatureDeliveryState.TESTING, "previous")
        == FeatureDeliveryState.TESTING
    )
    assert (
        resolve_resume_target(FeatureDeliveryState.TESTING, "developer")
        == FeatureDeliveryState.DEVELOPING
    )
    with pytest.raises(ValueError, match="unsupported"):
        resolve_resume_target(FeatureDeliveryState.TESTING, "acceptance")


def test_tester_block_can_resume_exact_commit_and_deliver(delivery_env):
    fake = FakeExecutor(
        delivery_env.repo,
        [
            ("developer", "ready"),
            ("tester", "blocked"),
            ("tester", "pass"),
            ("acceptance", "accept"),
        ],
    )
    runner, root_id = create(delivery_env, fake)
    blocked = runner.run(root_id)
    commit = blocked.developer_commit
    old_stage = fake.calls[-1][2]

    unblocked = runner.unblock(root_id, confirmed=True)
    assert unblocked.current_state == "TESTING"
    assert unblocked.fix_loops == 0
    assert [call[0] for call in fake.calls] == ["developer", "tester"]

    delivered = runner.resume(root_id)
    assert delivered.current_state == "DELIVERED"
    assert delivered.developer_commit == delivered.tested_commit == delivered.accepted_commit == commit
    assert fake.calls[2][2] != old_stage


def test_developer_override_invalidates_old_evidence_and_delivers_new_commit(
    delivery_env,
):
    fake = FakeExecutor(
        delivery_env.repo,
        [
            ("developer", "ready"),
            ("tester", "blocked"),
            ("developer", "ready"),
            ("tester", "pass"),
            ("acceptance", "accept"),
        ],
    )
    runner, root_id = create(delivery_env, fake)
    first = runner.run(root_id).developer_commit
    report_count = len(
        [row for row in stage_rows(root_id) if json.loads(row["body"])["feature_delivery_stage"]]
    )

    unblocked = runner.unblock(
        root_id, resume_stage="developer", confirmed=True
    )
    assert unblocked.current_state == "DEVELOPING"
    assert unblocked.fix_loops == 0

    delivered = runner.resume(root_id)
    assert delivered.current_state == "DELIVERED"
    assert delivered.developer_commit != first
    assert delivered.developer_commit == delivered.tested_commit == delivered.accepted_commit
    assert len(stage_rows(root_id)) > report_count


def test_acceptance_block_resumes_exact_tested_commit(delivery_env):
    fake = FakeExecutor(
        delivery_env.repo,
        [
            ("developer", "ready"),
            ("tester", "pass"),
            ("acceptance", "blocked"),
            ("acceptance", "accept"),
        ],
    )
    runner, root_id = create(delivery_env, fake)
    blocked = runner.run(root_id)
    assert blocked.current_state == "BLOCKED"
    assert runner.unblock(root_id, confirmed=True).current_state == "ACCEPTANCE"
    delivered = runner.resume(root_id)
    assert delivered.developer_commit == delivered.tested_commit == delivered.accepted_commit


def test_unblock_revalidates_contract_hash(delivery_env):
    runner, root_id = create(
        delivery_env,
        FakeExecutor(delivery_env.repo, [("developer", "blocked")]),
    )
    runner.run(root_id)
    with kb.connect() as conn:
        attachment = next(
            item
            for item in kb.list_attachments(conn, root_id)
            if item.filename == "task-contract.json"
        )
    original = Path(attachment.stored_path).read_bytes()
    Path(attachment.stored_path).write_text("{}", encoding="utf-8")
    with pytest.raises(DeliveryRunnerError, match="contract_hash_mismatch"):
        runner.unblock(root_id, confirmed=True)
    Path(attachment.stored_path).write_bytes(original)
    assert runner.status(root_id).blocked_reason.startswith("contract_hash_mismatch")
    with pytest.raises(DeliveryRunnerError, match="not recoverable"):
        runner.unblock(root_id, confirmed=True)


def test_commit_mismatch_prevents_unblock(delivery_env):
    runner, root_id = create(
        delivery_env,
        FakeExecutor(
            delivery_env.repo,
            [("developer", "ready"), ("tester", "blocked")],
        ),
    )
    runner.run(root_id)
    with kb.connect() as conn:
        workspace = Path(kb.get_task(conn, root_id).workspace_path)
    (workspace / "drift.txt").write_text("drift\n", encoding="utf-8")
    git(workspace, "add", "drift.txt")
    git(workspace, "commit", "-m", "move blocked branch")
    with pytest.raises(DeliveryRunnerError, match="commit_mismatch"):
        runner.unblock(root_id, confirmed=True)
    assert runner.status(root_id).blocked_reason.startswith("commit_mismatch")


def test_dirty_worktree_becomes_a_terminal_block(delivery_env):
    runner, root_id = create(
        delivery_env,
        FakeExecutor(
            delivery_env.repo,
            [("developer", "ready"), ("tester", "blocked")],
        ),
    )
    runner.run(root_id)
    with kb.connect() as conn:
        workspace = Path(kb.get_task(conn, root_id).workspace_path)
    drift = workspace / "drift.txt"
    drift.write_text("drift\n", encoding="utf-8")
    with pytest.raises(DeliveryRunnerError, match="dirty_worktree"):
        runner.unblock(root_id, confirmed=True)
    drift.unlink()
    assert runner.status(root_id).blocked_reason.startswith("dirty_worktree")
    with pytest.raises(DeliveryRunnerError, match="not recoverable"):
        runner.unblock(root_id, confirmed=True)


def test_unblock_event_preserves_block_history(delivery_env):
    runner, root_id = create(
        delivery_env,
        FakeExecutor(delivery_env.repo, [("developer", "blocked")]),
    )
    runner.run(root_id)
    status = runner.unblock(root_id, confirmed=True)
    assert status.blocked_reason is None
    assert "external_environment_missing" in status.last_blocked_reason
    assert status.last_unblock_target == "DEVELOPING"
    with kb.connect() as conn:
        events = [
            event
            for event in kb.list_events(conn, root_id)
            if event.kind == "feature_delivery_unblocked"
        ]
    assert len(events) == 1
    assert events[0].payload["approved_by"] == "human_cli"
    assert events[0].payload["previous_state"] == "BLOCKED"
    assert events[0].payload["resume_target_state"] == "DEVELOPING"
    assert events[0].payload["contract_hash"]


def test_ordinary_kanban_unblock_is_unchanged(delivery_env):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="ordinary")
        assert kb.block_task(conn, task_id, reason="wait")
        assert kb.unblock_task(conn, task_id)
        task = kb.get_task(conn, task_id)
    assert task.status == "ready"
    assert task.workflow_template_id is None
    assert task.current_step_key is None
