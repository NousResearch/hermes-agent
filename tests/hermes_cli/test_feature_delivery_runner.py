from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.feature_delivery import FEATURE_DELIVERY_WORKFLOW, FeatureDeliveryState
from hermes_cli.feature_delivery_runner import (
    DeliveryRunnerError,
    FeatureDeliveryRunner,
)
from hermes_cli.subcommands.delivery import build_delivery_parser, delivery_command


def git(path: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


@dataclass
class DeliveryFixture:
    repo: Path
    contract_path: Path
    contract: dict
    base: str


@pytest.fixture
def delivery_env(tmp_path, monkeypatch) -> DeliveryFixture:
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(home / "attachments"))
    repo = tmp_path / "target-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    git(repo, "config", "user.email", "runner@example.invalid")
    git(repo, "config", "user.name", "Runner Test")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    git(repo, "add", "README.md")
    git(repo, "commit", "-m", "base")
    base = git(repo, "rev-parse", "HEAD")
    contract = {
        "task_id": "contract-task-1",
        "title": "Deliver a tested feature",
        "objective": "Exercise the durable feature runner",
        "repository": str(repo),
        "base_commit": base,
        "branch": "feature/durable-runner-test",
        "acceptance_criteria": [
            {"id": "AC-1", "requirement": "Tests pass"},
            {"id": "AC-2", "requirement": "Acceptance passes"},
        ],
        "constraints": ["No deployment"],
        "required_tests": ["runner tests"],
        "required_evidence": ["tests", "diff-check"],
        "out_of_scope": ["merge", "deploy"],
        "delivery_gate": "acceptance_agent",
    }
    contract_path = tmp_path / "task-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return DeliveryFixture(repo, contract_path, contract, base)


class FakeExecutor:
    def __init__(self, repo: Path, script: list[tuple[str, str]]):
        self.repo = repo
        self.script = list(script)
        self.calls: list[tuple[str, str, str]] = []

    def execute(
        self,
        *,
        role,
        task_contract,
        workspace,
        target_commit,
        feedback,
        stage_task_id,
        tester_report=None,
    ):
        expected_role, outcome = self.script.pop(0)
        assert role == expected_role
        self.calls.append((role, target_commit, stage_task_id))
        if role == "developer":
            if outcome == "blocked":
                return {
                    "task_id": task_contract.task_id,
                    "agent": "developer",
                    "status": "BLOCKED",
                    "commit": None,
                    "implementation_summary": "missing external environment",
                }
            if outcome == "bad_commit":
                commit = "a" * 40
            elif outcome == "dirty":
                (workspace / "dirty.txt").write_text("dirty", encoding="utf-8")
                commit = target_commit
            elif outcome == "non_descendant":
                git(workspace, "checkout", "--orphan", "feature/orphan")
                for item in workspace.iterdir():
                    if item.name != ".git" and item.is_file():
                        item.unlink()
                (workspace / "orphan.txt").write_text("orphan\n", encoding="utf-8")
                git(workspace, "add", "-A")
                git(workspace, "commit", "-m", "orphan")
                git(workspace, "branch", "-M", task_contract.branch)
                commit = git(workspace, "rev-parse", "HEAD")
            else:
                feature = workspace / "feature.txt"
                old = feature.read_text(encoding="utf-8") if feature.exists() else ""
                feature.write_text(old + f"change {len(self.calls)}\n", encoding="utf-8")
                git(workspace, "add", "feature.txt")
                git(workspace, "commit", "-m", f"developer {len(self.calls)}")
                commit = git(workspace, "rev-parse", "HEAD")
            return {
                "task_id": task_contract.task_id,
                "agent": "developer",
                "status": "READY_FOR_TEST",
                "commit": commit,
                "changed_files": ["feature.txt"],
                "implementation_summary": "implemented feature",
                "self_checks": ["local check"],
            }
        if role == "tester":
            if outcome == "mutate":
                (workspace / "README.md").write_text("changed by tester\n", encoding="utf-8")
                outcome = "pass"
            if outcome == "blocked":
                return {
                    "task_id": task_contract.task_id,
                    "agent": "tester",
                    "tested_commit": None,
                    "status": "BLOCKED",
                }
            tested = "b" * 40 if outcome == "mismatch" else target_commit
            return {
                "task_id": task_contract.task_id,
                "agent": "tester",
                "tested_commit": tested,
                "status": "TEST_FAIL" if outcome == "fail" else "TEST_PASS",
                "test_results": ["runner tests"],
                "blocking_issues": ["test failed"] if outcome == "fail" else [],
                "evidence": ["tests"],
            }
        if outcome == "blocked":
            return {
                "task_id": task_contract.task_id,
                "agent": "acceptance",
                "accepted_commit": None,
                "status": "BLOCKED",
            }
        if outcome == "wrong_marker":
            marker = "ACCEPT"
        else:
            marker = "FINAL: ACCEPT" if outcome not in {"reject"} else None
        evidence = ["diff-check"] if outcome != "missing_evidence" else []
        if outcome == "stale_head":
            tree = git(self.repo, "rev-parse", f"{target_commit}^{{tree}}")
            moved = subprocess.run(
                ["git", "-C", str(self.repo), "commit-tree", tree, "-p", target_commit, "-m", "move head"],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **__import__("os").environ,
                    "GIT_AUTHOR_NAME": "Runner Test",
                    "GIT_AUTHOR_EMAIL": "runner@example.invalid",
                    "GIT_COMMITTER_NAME": "Runner Test",
                    "GIT_COMMITTER_EMAIL": "runner@example.invalid",
                },
            ).stdout.strip()
            git(self.repo, "update-ref", f"refs/heads/{task_contract.branch}", moved)
        return {
            "task_id": task_contract.task_id,
            "agent": "acceptance",
            "accepted_commit": target_commit,
            "status": "REJECT" if outcome == "reject" else "ACCEPT",
            "criteria": [
                {"id": "AC-1", "met": outcome != "reject", "evidence": "tests"},
                {"id": "AC-2", "met": outcome != "reject", "evidence": "review"},
            ],
            "blocking_issues": ["acceptance rejected"] if outcome == "reject" else [],
            "evidence": evidence,
            "final_marker": marker,
        }


def create(delivery_env, executor=None):
    runner = FeatureDeliveryRunner(executor)
    return runner, runner.create(delivery_env.contract_path)


def stage_rows(root_id: str):
    with kb.connect() as conn:
        return conn.execute(
            "SELECT t.* FROM tasks t JOIN task_links l ON l.child_id=t.id "
            "WHERE l.parent_id=? ORDER BY t.created_at,t.id",
            (root_id,),
        ).fetchall()


def test_create_valid_contract_creates_root(delivery_env):
    runner, root_id = create(delivery_env)
    with kb.connect() as conn:
        root = kb.get_task(conn, root_id)
    assert root.workflow_template_id == FEATURE_DELIVERY_WORKFLOW
    assert root.current_step_key == "CONTRACT_READY"
    assert runner.status(root_id).contract_hash


def test_create_rejects_dirty_repository(delivery_env):
    (delivery_env.repo / "dirty.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(DeliveryRunnerError, match="dirty"):
        FeatureDeliveryRunner().create(delivery_env.contract_path)


def test_create_rejects_bad_base_commit(delivery_env):
    delivery_env.contract["base_commit"] = "a" * 40
    delivery_env.contract_path.write_text(json.dumps(delivery_env.contract), encoding="utf-8")
    with pytest.raises(DeliveryRunnerError, match="base commit"):
        FeatureDeliveryRunner().create(delivery_env.contract_path)


def test_contract_is_stored_outside_target_repository(delivery_env):
    _, root_id = create(delivery_env)
    with kb.connect() as conn:
        contract_attachment = next(
            item for item in kb.list_attachments(conn, root_id) if item.filename == "task-contract.json"
        )
    assert delivery_env.repo not in Path(contract_attachment.stored_path).parents


def test_contract_hash_is_stored_in_event(delivery_env):
    _, root_id = create(delivery_env)
    with kb.connect() as conn:
        event = next(e for e in kb.list_events(conn, root_id) if e.kind == "feature_delivery_created")
    assert len(event.payload["contract_sha256"]) == 64


def test_happy_path_delivers(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")])
    runner, root_id = create(delivery_env, fake)
    result = runner.run(root_id)
    assert result.current_state == "DELIVERED"
    assert result.delivery_status == "Feature Delivery Gate Passed"
    assert result.developer_commit == result.tested_commit == result.accepted_commit


def test_developer_blocked_stops(delivery_env):
    runner, root_id = create(delivery_env, FakeExecutor(delivery_env.repo, [("developer", "blocked")]))
    result = runner.run(root_id)
    assert result.current_state == "BLOCKED"
    assert "external_environment_missing" in result.blocked_reason


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [("bad_commit", "commit_mismatch"), ("dirty", "dirty_worktree"), ("non_descendant", "commit_mismatch")],
)
def test_invalid_developer_commit_is_blocked(delivery_env, outcome, reason):
    runner, root_id = create(delivery_env, FakeExecutor(delivery_env.repo, [("developer", outcome)]))
    assert reason in runner.run(root_id).blocked_reason


def test_tester_pass_advances_to_acceptance_and_delivery(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")])
    runner, root_id = create(delivery_env, fake)
    assert runner.run(root_id).current_state == "DELIVERED"
    assert [call[0] for call in fake.calls] == ["developer", "tester", "acceptance"]


def test_tester_fail_returns_to_new_developer_commit(delivery_env):
    fake = FakeExecutor(
        delivery_env.repo,
        [("developer", "ready"), ("tester", "fail"), ("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")],
    )
    runner, root_id = create(delivery_env, fake)
    result = runner.run(root_id)
    first_commit = fake.calls[1][1]
    second_commit = fake.calls[3][1]
    assert result.current_state == "DELIVERED"
    assert result.fix_loops == 1
    assert first_commit != second_commit == result.developer_commit


def test_tester_commit_mismatch_blocks(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "mismatch")])
    runner, root_id = create(delivery_env, fake)
    assert "commit_mismatch" in runner.run(root_id).blocked_reason


def test_tester_source_mutation_blocks(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "mutate")])
    runner, root_id = create(delivery_env, fake)
    assert "tester_modified_source" in runner.run(root_id).blocked_reason


def test_acceptance_reject_returns_to_developer(delivery_env):
    fake = FakeExecutor(
        delivery_env.repo,
        [("developer", "ready"), ("tester", "pass"), ("acceptance", "reject"), ("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")],
    )
    runner, root_id = create(delivery_env, fake)
    result = runner.run(root_id)
    assert result.current_state == "DELIVERED"
    assert result.fix_loops == 1


def test_acceptance_blocked_stops(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", "blocked")])
    runner, root_id = create(delivery_env, fake)
    assert runner.run(root_id).current_state == "BLOCKED"


@pytest.mark.parametrize("outcome", ["stale_head", "missing_evidence"])
def test_acceptance_gate_denial_blocks(delivery_env, outcome):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", outcome)])
    runner, root_id = create(delivery_env, fake)
    assert "acceptance_gate_denied" in runner.run(root_id).blocked_reason


def test_wrong_acceptance_marker_never_delivers(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", "wrong_marker")])
    runner, root_id = create(delivery_env, fake)
    result = runner.run(root_id)
    assert result.current_state == "BLOCKED"
    assert "invalid_report" in result.blocked_reason


def test_fifth_fix_loop_blocks_without_sixth_developer(delivery_env):
    script = []
    for _ in range(5):
        script.extend([("developer", "ready"), ("tester", "fail")])
    fake = FakeExecutor(delivery_env.repo, script)
    runner, root_id = create(delivery_env, fake)
    result = runner.run(root_id)
    assert result.current_state == "BLOCKED"
    assert result.fix_loops == 5
    assert [role for role, _, _ in fake.calls].count("developer") == 5
    assert "max_fix_loops_reached" in result.blocked_reason


def test_new_developer_commit_clears_old_test_and_acceptance_evidence(delivery_env):
    fake = FakeExecutor(
        delivery_env.repo,
        [("developer", "ready"), ("tester", "pass"), ("acceptance", "reject"), ("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")],
    )
    runner, root_id = create(delivery_env, fake)
    result = runner.run(root_id)
    assert result.tested_commit == result.developer_commit
    assert result.accepted_commit == result.developer_commit
    assert fake.calls[1][1] != result.developer_commit


def test_missing_executor_blocks_with_explicit_reason(delivery_env):
    runner, root_id = create(delivery_env)
    result = runner.run(root_id)
    assert "stage_executor_missing" in result.blocked_reason
    assert "No configured stage executor" in result.blocked_reason


def test_contract_hash_mismatch_blocks_on_run(delivery_env):
    runner, root_id = create(delivery_env, FakeExecutor(delivery_env.repo, []))
    with kb.connect() as conn:
        attachment = next(a for a in kb.list_attachments(conn, root_id) if a.filename == "task-contract.json")
    Path(attachment.stored_path).write_text("{}", encoding="utf-8")
    result = runner.run(root_id)
    assert result.current_state == "BLOCKED"
    assert "contract_hash_mismatch" in result.blocked_reason


def test_report_attachment_metadata_mismatch_blocks(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")])
    runner, root_id = create(delivery_env, fake)
    original_transition = runner._transition

    def crash_before_ready(conn, root, target, payload=None):
        if target == FeatureDeliveryState.READY_FOR_TEST:
            raise KeyboardInterrupt
        return original_transition(conn, root, target, payload)

    runner._transition = crash_before_ready
    with pytest.raises(KeyboardInterrupt):
        runner.run(root_id)
    with kb.connect() as conn:
        report = next(a for a in kb.list_attachments(conn, root_id) if "developer.json" in a.filename)
    Path(report.stored_path).write_text("{}", encoding="utf-8")
    runner._transition = original_transition
    result = runner.resume(root_id)
    assert "invalid_report" in result.blocked_reason


def test_report_files_have_stable_numbering(delivery_env):
    fake = FakeExecutor(delivery_env.repo, [("developer", "ready"), ("tester", "pass"), ("acceptance", "accept")])
    runner, root_id = create(delivery_env, fake)
    runner.run(root_id)
    with kb.connect() as conn:
        names = [a.filename for a in kb.list_attachments(conn, root_id) if a.filename.startswith("reports/")]
    assert names == ["reports/001-developer.json", "reports/002-tester.json", "reports/003-acceptance.json"]


def test_status_output_contains_required_fields(delivery_env):
    runner, root_id = create(delivery_env)
    output = runner.status(root_id).render()
    for label in ("Task ID", "Current State", "Fix Loops", "Branch", "Contract Hash", "Delivery Status"):
        assert f"{label}:" in output


def test_pid_alive_uses_cross_platform_probe(monkeypatch):
    import gateway.status

    seen = []
    monkeypatch.setattr(gateway.status, "_pid_exists", lambda pid: seen.append(pid) or True)
    assert FeatureDeliveryRunner._pid_alive(4242) is True
    assert seen == [4242]


def test_ordinary_kanban_task_is_rejected_without_mutation(delivery_env):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="ordinary")
    with pytest.raises(DeliveryRunnerError, match="not a feature delivery"):
        FeatureDeliveryRunner().status(task_id)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
    assert task.workflow_template_id is None
    assert task.current_step_key is None


def test_cli_parser_supports_only_feature_delivery_commands():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_delivery_parser(subparsers, cmd_delivery=lambda _: 0)
    for command in ("create", "run", "resume", "status", "unblock"):
        if command == "create":
            tail = ["--contract", "contract.json"]
        elif command == "unblock":
            tail = ["task-id", "--confirm"]
        else:
            tail = ["task-id"]
        args = parser.parse_args(["delivery", command, *tail])
        assert args.delivery_command == command


def test_cli_create_and_status(delivery_env, capsys):
    runner = FeatureDeliveryRunner()
    create_args = argparse.Namespace(delivery_command="create", contract=str(delivery_env.contract_path))
    assert delivery_command(create_args, runner=runner) == 0
    root_id = capsys.readouterr().out.strip()
    status_args = argparse.Namespace(delivery_command="status", task_id=root_id)
    assert delivery_command(status_args, runner=runner) == 0
    assert "Current State: CONTRACT_READY" in capsys.readouterr().out


def test_cli_run_and_resume(delivery_env, capsys):
    runner, root_id = create(delivery_env)
    run_args = argparse.Namespace(delivery_command="run", task_id=root_id)
    assert delivery_command(run_args, runner=runner) == 0
    assert "Current State: BLOCKED" in capsys.readouterr().out
    resume_args = argparse.Namespace(delivery_command="resume", task_id=root_id)
    assert delivery_command(resume_args, runner=runner) == 0
    assert "Current State: BLOCKED" in capsys.readouterr().out
