from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from hermes_cli.feature_delivery_runner import FeatureDeliveryRunner
from hermes_cli.profile_stage_executor import ProfileStageExecutor
from hermes_cli.subcommands.delivery import build_delivery_parser
from tests.hermes_cli.test_feature_delivery_runner import delivery_env, git


def test_profile_executor_drives_runner_to_delivery(delivery_env, monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profile_stage_executor.profile_exists",
        lambda name: name in {"developer", "tester", "acceptance"},
    )
    monkeypatch.setattr(
        "hermes_cli.profile_stage_executor.resolve_profile_env",
        lambda name: f"/profiles/{name}",
    )
    monkeypatch.setattr(
        "hermes_cli.profile_stage_executor._resolve_hermes_argv",
        lambda: ["hermes"],
    )
    calls = []

    def run(command, **kwargs):
        profile = command[command.index("-p") + 1]
        workspace = kwargs["cwd"]
        prompt = command[-1]
        workspace_path = Path(workspace)
        calls.append((profile, git(workspace_path, "rev-parse", "HEAD"), prompt))
        if profile == "developer":
            feature = workspace_path / "feature.txt"
            feature.write_text("profile feature\n", encoding="utf-8")
            git(workspace_path, "add", "feature.txt")
            git(workspace_path, "commit", "-m", "profile developer")
            commit = git(workspace_path, "rev-parse", "HEAD")
            payload = {
                "task_id": delivery_env.contract["task_id"],
                "agent": "developer",
                "status": "READY_FOR_TEST",
                "commit": commit,
                "changed_files": ["feature.txt"],
                "implementation_summary": "implemented feature",
                "self_checks": ["runner tests"],
                "known_risks": [],
            }
        elif profile == "tester":
            commit = git(workspace_path, "rev-parse", "HEAD")
            payload = {
                "task_id": delivery_env.contract["task_id"],
                "agent": "tester",
                "tested_commit": commit,
                "status": "TEST_PASS",
                "test_results": ["runner tests"],
                "blocking_issues": [],
                "non_blocking_issues": [],
                "evidence": ["tests"],
            }
        else:
            commit = git(workspace_path, "rev-parse", "HEAD")
            payload = {
                "task_id": delivery_env.contract["task_id"],
                "agent": "acceptance",
                "accepted_commit": commit,
                "status": "ACCEPT",
                "criteria": [
                    {"id": "AC-1", "met": True, "evidence": "tests"},
                    {"id": "AC-2", "met": True, "evidence": "independent review"},
                ],
                "blocking_issues": [],
                "evidence": ["diff-check"],
                "final_marker": "FINAL: ACCEPT",
            }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    runner = FeatureDeliveryRunner(executor=ProfileStageExecutor(run_command=run))
    root_id = runner.create(delivery_env.contract_path)
    result = runner.run(root_id)

    assert result.current_state == "DELIVERED"
    assert [call[0] for call in calls] == ["developer", "tester", "acceptance"]
    assert calls[1][1] == result.developer_commit
    assert calls[2][1] == result.developer_commit
    assert '"tester_report"' in calls[2][2]


def test_missing_profile_blocks_runner_with_profile_missing(delivery_env, monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profile_stage_executor.profile_exists",
        lambda name: name != "tester",
    )
    runner = FeatureDeliveryRunner(executor=ProfileStageExecutor())
    root_id = runner.create(delivery_env.contract_path)

    result = runner.run(root_id)

    assert result.current_state == "BLOCKED"
    assert result.blocked_reason is not None
    assert result.blocked_reason.startswith("profile_missing:")


def test_profiles_executor_is_explicit_cli_opt_in():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_delivery_parser(subparsers, cmd_delivery=lambda args: None)

    args = parser.parse_args(["delivery", "run", "task-1", "--executor", "profiles"])

    assert args.executor == "profiles"
    default_args = parser.parse_args(["delivery", "run", "task-1"])
    assert default_args.executor is None
