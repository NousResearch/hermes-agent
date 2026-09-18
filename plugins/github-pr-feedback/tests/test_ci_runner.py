from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from github_pr_feedback.ci_runner import (
    CIAuditIdentity,
    CIValidationError,
    CompletedCommand,
    LocalCIRunner,
)
from github_pr_feedback.github_client import CheckState, PullRequestMergeState
from github_pr_feedback.ledger import FeedbackLedger


BASE_SHA = "b" * 40
HEAD_SHA = "a" * 40
NOW = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)


class FakeGitHub:
    def __init__(self, state: PullRequestMergeState) -> None:
        self.states = [state, state]
        self.checks = [
            CheckState(actions_enabled=False, all_green=True, check_count=0),
            CheckState(actions_enabled=False, all_green=True, check_count=0),
        ]

    def get_merge_state(self, repository: str, number: int) -> PullRequestMergeState:
        return self.states.pop(0)

    def get_check_state(self, repository: str, head_sha: str) -> CheckState:
        return self.checks.pop(0)


class FakeInspector:
    def __init__(self, *, changed: tuple[str, ...] = ("src/example.py",)) -> None:
        self.heads = [HEAD_SHA, HEAD_SHA]
        self.clean = [True, True]
        self.changed = changed

    def head_sha(self, worktree: Path) -> str:
        return self.heads.pop(0)

    def is_clean(self, worktree: Path) -> bool:
        return self.clean.pop(0)

    def changed_files(self, worktree: Path, base_sha: str, head_sha: str) -> tuple[str, ...]:
        return self.changed


class RecordingRunner:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.calls: list[tuple[tuple[str, ...], Path, dict[str, str], int]] = []
        self.fail_at = fail_at

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> CompletedCommand:
        self.calls.append((argv, cwd, dict(env), timeout))
        failed = self.fail_at == len(self.calls)
        return CompletedCommand(
            returncode=1 if failed else 0,
            stdout="x" * 10000,
            stderr="failed" if failed else "",
            duration_ms=25,
            timed_out=False,
        )


def merge_state(**overrides: object) -> PullRequestMergeState:
    values: dict[str, object] = {
        "repository": "acme/widgets",
        "number": 17,
        "state": "OPEN",
        "is_draft": False,
        "mergeable": True,
        "merge_state_status": "CLEAN",
        "base_branch": "stable",
        "base_sha": BASE_SHA,
        "head_repository": "acme/widgets",
        "author_login": "owner",
        "head_ref_name": "codex/fix",
        "head_sha": HEAD_SHA,
        "merged": False,
        "merge_commit_oid": None,
    }
    values.update(overrides)
    return PullRequestMergeState(**values)


def prepare_repository(path: Path, *, frontend: bool = False) -> None:
    for relative in (
        "scripts/check_ci_governance.py",
        "scripts/run_hygiene_lane.py",
        "scripts/run_static_lane.py",
        "scripts/run_test_lane.py",
        "scripts/run_local_ci_audit.py",
    ):
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# fixture\n", encoding="utf-8")
    manifest = path / "tests/manifests/test_lanes.toml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        """
[lanes.unit]
ci_status = "required"
[lanes.optional]
ci_status = "optional"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    if frontend:
        (path / "frontend").mkdir()
        (path / "frontend/package.json").write_text("{}\n", encoding="utf-8")
        (path / "frontend/package-lock.json").write_text("{}\n", encoding="utf-8")


def build_runner(
    tmp_path: Path,
    *,
    github: FakeGitHub | None = None,
    inspector: FakeInspector | None = None,
    commands: RecordingRunner | None = None,
    supervisor_pid: int = 4242,
    pid_is_alive=lambda _pid: True,
) -> tuple[LocalCIRunner, FeedbackLedger, RecordingRunner]:
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    command_runner = commands or RecordingRunner()
    runner = LocalCIRunner(
        github or FakeGitHub(merge_state()),
        ledger,
        command_runner=command_runner,
        inspector=inspector or FakeInspector(),
        python_argv=("python3",),
        now=lambda: NOW,
        supervisor_pid=lambda: supervisor_pid,
        pid_is_alive=pid_is_alive,
    )
    return runner, ledger, command_runner


def test_ci_run_is_claimed_before_bootstrap_with_real_supervisor_pid(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)
    (worktree / "scripts/bootstrap_agent_workspace.py").write_text(
        "# fixture\n", encoding="utf-8"
    )
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    class ObservingRunner(RecordingRunner):
        def run(self, argv, *, cwd, env, timeout):
            lifecycle = ledger.latest_ci_run("acme/widgets", 17, HEAD_SHA)
            assert lifecycle is not None
            assert lifecycle["status"] == "running"
            assert lifecycle["supervisor_pid"] == 4242
            result = super().run(argv, cwd=cwd, env=env, timeout=timeout)
            if argv == ("python3", "scripts/bootstrap_agent_workspace.py", "--venv", "link"):
                executable = cwd / ".venv/bin/python"
                executable.parent.mkdir(parents=True)
                executable.write_text("#!/bin/sh\n", encoding="utf-8")
                executable.chmod(0o755)
            return result

    commands = ObservingRunner()
    runner = LocalCIRunner(
        FakeGitHub(merge_state()),
        ledger,
        command_runner=commands,
        inspector=FakeInspector(),
        now=lambda: NOW,
        supervisor_pid=lambda: 4242,
        pid_is_alive=lambda pid: pid == 4242,
    )
    receipt = runner.run(CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA), worktree)

    lifecycle = ledger.latest_ci_run("acme/widgets", 17, HEAD_SHA)
    assert lifecycle is not None
    assert lifecycle["status"] == "completed"
    assert lifecycle["receipt_id"] == receipt.receipt_id
    assert commands.calls[0][0] == (
        "python3",
        "scripts/bootstrap_agent_workspace.py",
        "--venv",
        "link",
    )
    ledger.close()


def test_active_exact_head_ci_run_rejects_duplicate_without_running_commands(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)
    first, ledger, _ = build_runner(tmp_path, supervisor_pid=4242)
    manifest_digest = hashlib.sha256(
        (worktree / "tests/manifests/test_lanes.toml").read_bytes()
    ).hexdigest()
    lease = ledger.claim_ci_run(
        "acme/widgets", 17, BASE_SHA, HEAD_SHA, manifest_digest,
        supervisor_pid=4242, claimed_at=NOW, stale_before=NOW - timedelta(hours=2),
        pid_is_alive=lambda pid: pid == 4242,
    )
    assert lease is not None

    with pytest.raises(CIValidationError, match="already running"):
        first.run(CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA), worktree)

    assert first._commands.calls == []
    ledger.close()


def test_dead_stale_ci_supervisor_allows_one_fenced_takeover(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)
    runner, ledger, commands = build_runner(
        tmp_path, supervisor_pid=5252, pid_is_alive=lambda _pid: False
    )
    digest = hashlib.sha256(
        (worktree / "tests/manifests/test_lanes.toml").read_bytes()
    ).hexdigest()
    old = ledger.claim_ci_run(
        "acme/widgets", 17, BASE_SHA, HEAD_SHA, digest,
        supervisor_pid=4242,
        claimed_at=NOW - timedelta(hours=3),
        stale_before=NOW - timedelta(hours=2),
        pid_is_alive=lambda _pid: False,
    )
    assert old is not None

    receipt = runner.run(CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA), worktree)

    assert commands.calls
    lifecycle = ledger.latest_ci_run("acme/widgets", 17, HEAD_SHA)
    assert lifecycle is not None
    assert lifecycle["status"] == "completed"
    assert lifecycle["supervisor_pid"] == 5252
    assert lifecycle["lease_version"] == old.version + 1
    assert lifecycle["receipt_id"] == receipt.receipt_id
    ledger.close()


def test_local_ci_runner_executes_only_required_lanes_and_records_exact_head_receipt(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)
    runner, ledger, commands = build_runner(tmp_path)
    identity = CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA)

    receipt = runner.run(identity, worktree)

    assert receipt.status == "passed"
    assert receipt.identity == identity
    assert [call[0] for call in commands.calls] == [
        ("python3", "scripts/check_ci_governance.py"),
        ("python3", "scripts/run_static_lane.py"),
        ("python3", "scripts/run_hygiene_lane.py"),
        ("python3", "scripts/run_test_lane.py", "--lane", "unit"),
    ]
    assert commands.calls[1][2]["STATIC_BASE_REF"] == BASE_SHA
    assert all(len(evidence.stdout_sha256) == 64 for evidence in receipt.commands)
    assert ledger.latest_passing_ci_receipt(
        "acme/widgets",
        17,
        HEAD_SHA,
        manifest_digest=receipt.manifest_digest,
        not_before=NOW - timedelta(hours=1),
    ) == receipt
    ledger.close()


def test_local_ci_runner_bootstraps_missing_repo_venv_before_ci(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)
    bootstrap = worktree / "scripts/bootstrap_agent_workspace.py"
    bootstrap.write_text("# fixture\n", encoding="utf-8")

    class BootstrappingRunner(RecordingRunner):
        def run(
            self,
            argv: tuple[str, ...],
            *,
            cwd: Path,
            env: dict[str, str],
            timeout: int,
        ) -> CompletedCommand:
            result = super().run(argv, cwd=cwd, env=env, timeout=timeout)
            if argv == ("python3", "scripts/bootstrap_agent_workspace.py", "--venv", "link"):
                executable = cwd / ".venv/bin/python"
                executable.parent.mkdir(parents=True)
                executable.write_text("#!/bin/sh\n", encoding="utf-8")
                executable.chmod(0o755)
            return result

    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    commands = BootstrappingRunner()
    runner = LocalCIRunner(
        FakeGitHub(merge_state()),
        ledger,
        command_runner=commands,
        inspector=FakeInspector(),
        now=lambda: NOW,
    )

    receipt = runner.run(CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA), worktree)

    assert receipt.status == "passed"
    assert commands.calls[0][0] == (
        "python3",
        "scripts/bootstrap_agent_workspace.py",
        "--venv",
        "link",
    )
    assert commands.calls[1][0] == (".venv/bin/python", "scripts/check_ci_governance.py")
    ledger.close()


def test_local_ci_runner_adds_locked_frontend_checks_only_for_frontend_changes(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree, frontend=True)
    inspector = FakeInspector(changed=("frontend/src/App.tsx",))
    runner, ledger, commands = build_runner(tmp_path, inspector=inspector)

    runner.run(CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA), worktree)

    assert [call[0] for call in commands.calls[-4:]] == [
        ("npm", "ci"),
        ("npm", "run", "lint"),
        ("npm", "test"),
        ("npm", "run", "build"),
    ]
    assert all(call[1] == worktree / "frontend" for call in commands.calls[-4:])
    ledger.close()


def test_environment_lane_uses_the_repo_owned_locked_install_runner(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)
    manifest = worktree / "tests/manifests/test_lanes.toml"
    manifest.write_text(
        """
[lanes.locked_install_parity]
ci_status = "required"
selection_rule = "environment_check"
pytest_args = []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    runner, ledger, commands = build_runner(tmp_path)

    runner.run(CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA), worktree)

    lane_argv = commands.calls[3][0]
    assert lane_argv[:4] == (
        "python3",
        "scripts/run_local_ci_audit.py",
        "--job",
        "locked_install_parity",
    )
    assert "scripts/run_test_lane.py" not in lane_argv
    assert lane_argv[-2] == "--output"
    assert lane_argv[-1].endswith(
        f"hermes-local-ci-receipts/acme-widgets/17/{HEAD_SHA}/locked_install_parity.json"
    )
    ledger.close()


@pytest.mark.parametrize("failure", ["dirty_start", "wrong_head", "head_race", "actions_race"])
def test_local_ci_runner_fails_closed_without_a_receipt_on_invalid_or_raced_state(
    tmp_path: Path, failure: str
) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)
    github = FakeGitHub(merge_state())
    inspector = FakeInspector()
    if failure == "dirty_start":
        inspector.clean[0] = False
    elif failure == "wrong_head":
        inspector.heads[0] = "c" * 40
    elif failure == "head_race":
        github.states[1] = replace(merge_state(), head_sha="c" * 40)
    else:
        github.checks[1] = CheckState(actions_enabled=True, all_green=True, check_count=1)
    runner, ledger, _commands = build_runner(tmp_path, github=github, inspector=inspector)

    with pytest.raises(CIValidationError):
        runner.run(CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA), worktree)

    assert ledger.latest_passing_ci_receipt(
        "acme/widgets",
        17,
        HEAD_SHA,
        manifest_digest="0" * 64,
        not_before=NOW - timedelta(days=1),
    ) is None
    ledger.close()


def test_failed_command_is_durable_evidence_but_never_a_passing_receipt(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)
    runner, ledger, _commands = build_runner(tmp_path, commands=RecordingRunner(fail_at=2))

    receipt = runner.run(CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA), worktree)

    assert receipt.status == "failed"
    assert len(receipt.commands) == 2
    assert ledger.latest_passing_ci_receipt(
        "acme/widgets",
        17,
        HEAD_SHA,
        manifest_digest=receipt.manifest_digest,
        not_before=NOW - timedelta(days=1),
    ) is None
    assert ledger.latest_ci_receipt(
        "acme/widgets",
        17,
        HEAD_SHA,
        manifest_digest=receipt.manifest_digest,
        not_before=NOW - timedelta(days=1),
    ) == receipt
    ledger.close()


def test_missing_ci_executable_is_classified_environment_blocked(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)

    class MissingExecutableRunner(RecordingRunner):
        def run(
            self,
            argv: tuple[str, ...],
            *,
            cwd: Path,
            env: dict[str, str],
            timeout: int,
        ) -> CompletedCommand:
            self.calls.append((argv, cwd, dict(env), timeout))
            return CompletedCommand(127, "", "FileNotFoundError", 1, False)

    runner, ledger, _commands = build_runner(tmp_path, commands=MissingExecutableRunner())

    receipt = runner.run(CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA), worktree)

    assert receipt.status == "failed"
    assert receipt.commands[0].classification == "environment-blocked"
    ledger.close()
