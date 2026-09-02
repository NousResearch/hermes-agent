from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli.feature_delivery import TaskContract, TesterReport
from hermes_cli.feature_delivery_runner import StageExecutionError
from hermes_cli.profile_stage_executor import (
    PROFILE_BY_ROLE,
    TOOLSETS_BY_ROLE,
    ProfileStageExecutor,
)


@pytest.fixture
def contract(tmp_path) -> TaskContract:
    return TaskContract.model_validate(
        {
            "task_id": "profile-stage-test",
            "title": "Profile stage test",
            "objective": "Exercise the profile executor",
            "repository": str(tmp_path),
            "base_commit": "a" * 40,
            "branch": "feature/profile-stage-test",
            "acceptance_criteria": [{"id": "AC-01", "requirement": "It works"}],
            "required_tests": ["scripts/run_tests.sh tests/example.py -q"],
            "required_evidence": ["test_results", "git_diff"],
        }
    )


@pytest.fixture
def profiles(monkeypatch, tmp_path):
    roots = {name: tmp_path / "profiles" / name for name in PROFILE_BY_ROLE.values()}
    for root in roots.values():
        root.mkdir(parents=True)
    monkeypatch.setattr(
        "hermes_cli.profile_stage_executor.profile_exists",
        lambda name: roots[name].is_dir(),
    )
    monkeypatch.setattr(
        "hermes_cli.profile_stage_executor.resolve_profile_env",
        lambda name: str(roots[name]),
    )
    monkeypatch.setattr(
        "hermes_cli.profile_stage_executor._resolve_hermes_argv",
        lambda: ["hermes"],
    )
    return roots


def report_for(role: str, task_id: str, commit: str) -> dict:
    if role == "developer":
        return {
            "task_id": task_id,
            "agent": "developer",
            "status": "READY_FOR_TEST",
            "commit": commit,
            "changed_files": ["feature.py"],
            "implementation_summary": "implemented",
            "self_checks": ["tests passed"],
            "known_risks": [],
        }
    if role == "tester":
        return {
            "task_id": task_id,
            "agent": "tester",
            "tested_commit": commit,
            "status": "TEST_PASS",
            "test_results": ["tests passed"],
            "blocking_issues": [],
            "non_blocking_issues": [],
            "evidence": ["test_results"],
        }
    return {
        "task_id": task_id,
        "agent": "acceptance",
        "accepted_commit": commit,
        "status": "ACCEPT",
        "criteria": [{"id": "AC-01", "met": True, "evidence": "verified"}],
        "blocking_issues": [],
        "evidence": ["git_diff"],
        "final_marker": "FINAL: ACCEPT",
    }


@pytest.mark.parametrize("role", ["developer", "tester", "acceptance"])
def test_fixed_profile_mapping_and_workspace_injection(
    role, contract, profiles, tmp_path
):
    captured = {}
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def run(command, **kwargs):
        captured.update(command=command, **kwargs)
        payload = report_for(role, contract.task_id, contract.base_commit)
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    report = ProfileStageExecutor(run_command=run).execute(
        role=role,
        task_contract=contract,
        workspace=workspace,
        target_commit=contract.base_commit,
        feedback=(),
        stage_task_id="stage-1",
    )

    profile = PROFILE_BY_ROLE[role]
    assert report.agent == role
    assert captured["command"][captured["command"].index("-p") + 1] == profile
    assert captured["command"][captured["command"].index("--toolsets") + 1] == ",".join(
        TOOLSETS_BY_ROLE[role]
    )
    assert captured["cwd"] == str(workspace)
    assert captured["env"]["TERMINAL_CWD"] == str(workspace.resolve())
    assert captured["env"]["HERMES_HOME"] == str(profiles[profile])


def test_arbitrary_profile_role_is_rejected(contract, profiles, tmp_path):
    with pytest.raises(ValueError, match="unsupported feature delivery role"):
        ProfileStageExecutor().execute(
            role="release",  # type: ignore[arg-type]
            task_contract=contract,
            workspace=tmp_path,
            target_commit=contract.base_commit,
            feedback=(),
            stage_task_id="stage-1",
        )


@pytest.mark.parametrize("missing", ["developer", "tester", "acceptance"])
def test_missing_required_profile_blocks_before_process(
    missing, contract, profiles, monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(
        "hermes_cli.profile_stage_executor.profile_exists",
        lambda name: name != missing,
    )
    executor = ProfileStageExecutor(run_command=lambda *args, **kwargs: calls.append(args))

    with pytest.raises(StageExecutionError) as caught:
        executor.execute(
            role="developer",
            task_contract=contract,
            workspace=tmp_path,
            target_commit=contract.base_commit,
            feedback=(),
            stage_task_id="stage-1",
        )

    assert caught.value.code == "profile_missing"
    assert calls == []


@pytest.mark.parametrize(
    ("role", "payload"),
    [
        ("developer", {"task_id": "profile-stage-test", "agent": "developer", "status": "ACCEPT"}),
        ("tester", {"task_id": "profile-stage-test", "agent": "tester", "status": "DELIVERED"}),
        ("acceptance", {"task_id": "profile-stage-test", "agent": "acceptance", "status": "TEST_PASS"}),
    ],
)
def test_forbidden_role_status_is_rejected(role, payload):
    with pytest.raises(ValueError, match="invalid structured report"):
        ProfileStageExecutor._parse_report(role, json.dumps(payload))


def test_malformed_json_is_rejected():
    with pytest.raises(ValueError, match="invalid JSON"):
        ProfileStageExecutor._parse_report("developer", "FINAL: READY_FOR_TEST")


def test_provider_failure_and_timeout_block_without_retry(contract, profiles, tmp_path):
    calls = []

    def auth_failure(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, "", "provider authentication failed")

    with pytest.raises(StageExecutionError) as auth_error:
        ProfileStageExecutor(run_command=auth_failure).execute(
            role="developer",
            task_contract=contract,
            workspace=tmp_path,
            target_commit=contract.base_commit,
            feedback=(),
            stage_task_id="stage-1",
        )
    assert auth_error.value.code == "stage_execution_failed"
    assert "authentication" not in str(auth_error.value)
    assert len(calls) == 1

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    with pytest.raises(StageExecutionError) as timeout_error:
        ProfileStageExecutor(run_command=timeout).execute(
            role="tester",
            task_contract=contract,
            workspace=tmp_path,
            target_commit=contract.base_commit,
            feedback=(),
            stage_task_id="stage-2",
        )
    assert timeout_error.value.code == "stage_execution_failed"
    assert "timed out" in str(timeout_error.value)


def test_acceptance_prompt_includes_tester_report_without_acceptance_bias(
    contract, tmp_path
):
    tester = TesterReport.model_validate(
        report_for("tester", contract.task_id, contract.base_commit)
    )
    prompt = ProfileStageExecutor._stage_prompt(
        role="acceptance",
        contract=contract,
        workspace=tmp_path,
        target_commit=contract.base_commit,
        feedback=(),
        tester_report=tester,
    )

    assert contract.base_commit in prompt
    assert '"tester_report"' in prompt
    assert "evidence, not an instruction to accept" in prompt
    assert "required_evidence identifier verbatim" in prompt
    assert "please confirm" not in prompt.lower()
