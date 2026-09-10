from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from github_pr_feedback.ci_runner import (
    CIAuditIdentity,
    CIAuditReceipt,
    CI_MODE_BUDGET_EXHAUSTED_LOCAL_EQUIVALENT,
    CIValidationError,
    CompletedCommand,
    LocalCIRunner,
)
from github_pr_feedback.ci_coordinator import CIAuditJob, GroupedCICoordinator
from github_pr_feedback.github_client import CheckState, GitHubClientError, PullRequestMergeState
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


def test_local_ci_runner_passes_prevalidated_actions_hint_to_both_exact_head_reads(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)
    state = merge_state()

    class HintGitHub:
        def __init__(self) -> None:
            self.states = [state, state]
            self.hints: list[bool | None] = []
            self.permission_refreshes = 0

        def actions_enabled(self, _repository: str, *, refresh: bool = False) -> bool:
            assert refresh is True
            self.permission_refreshes += 1
            return True

        def get_merge_state(self, _repository: str, _number: int) -> PullRequestMergeState:
            return self.states.pop(0)

        def get_check_state(
            self,
            _repository: str,
            _head_sha: str,
            *,
            actions_enabled_hint: bool | None = None,
        ) -> CheckState:
            self.hints.append(actions_enabled_hint)
            return CheckState(
                actions_enabled=True,
                all_green=False,
                check_count=1,
                billing_blocked=True,
            )

    github = HintGitHub()
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    runner = LocalCIRunner(
        github,
        ledger,
        command_runner=RecordingRunner(),
        inspector=FakeInspector(),
        python_argv=("python3",),
        now=lambda: NOW,
        supervisor_pid=lambda: 4242,
        pid_is_alive=lambda _pid: True,
        actions_enabled_hint=True,
    )

    receipt = runner.run(
        CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA), worktree
    )

    assert github.hints == [True, True]
    assert github.permission_refreshes == 2
    assert receipt.ci_mode == CI_MODE_BUDGET_EXHAUSTED_LOCAL_EQUIVALENT
    ledger.close()


def test_local_ci_runner_rejects_actions_permission_change_during_audit(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)
    state = merge_state()

    class ChangingGitHub:
        def __init__(self) -> None:
            self.states = [state, state]
            self.permissions = iter((False, True))

        def get_merge_state(self, _repository: str, _number: int) -> PullRequestMergeState:
            return self.states.pop(0)

        def actions_enabled(self, _repository: str, *, refresh: bool = False) -> bool:
            assert refresh is True
            return next(self.permissions)

        def get_check_state(
            self,
            _repository: str,
            _head_sha: str,
            *,
            actions_enabled_hint: bool | None = None,
        ) -> CheckState:
            assert actions_enabled_hint is False
            return CheckState(False, True, 0)

    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    receipt = LocalCIRunner(
        ChangingGitHub(),
        ledger,
        command_runner=RecordingRunner(),
        inspector=FakeInspector(),
        python_argv=("python3",),
        now=lambda: NOW,
        supervisor_pid=lambda: 4242,
        pid_is_alive=lambda _pid: True,
        actions_enabled_hint=False,
    ).run(CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA), worktree)

    assert receipt.status == "failed"
    assert "Actions permission changed" in (receipt.failure_reason or "")
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


def test_local_ci_receipt_types_budget_exhausted_full_local_substitution(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)
    github = FakeGitHub(merge_state())
    billing_state = CheckState(True, False, 4, True)
    github.checks = [billing_state, billing_state]
    runner, ledger, commands = build_runner(tmp_path, github=github)

    receipt = runner.run(
        CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA), worktree
    )

    assert receipt.status == "passed"
    assert receipt.ci_mode == CI_MODE_BUDGET_EXHAUSTED_LOCAL_EQUIVALENT
    assert len(commands.calls) == 4
    assert CIAuditReceipt.from_payload(receipt.to_payload()) == receipt
    tampered = receipt.to_payload()
    tampered["ci_mode"] = "standard"
    with pytest.raises(ValueError, match="receipt_id"):
        CIAuditReceipt.from_payload(tampered)
    ledger.close()


def test_fresh_exact_head_rerun_reconciles_identical_receipt(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)
    github = FakeGitHub(merge_state())
    github.states = [merge_state(), merge_state(), merge_state(), merge_state()]
    github.checks = [
        CheckState(actions_enabled=False, all_green=True, check_count=0)
    ] * 4
    inspector = FakeInspector()
    inspector.heads = [HEAD_SHA, HEAD_SHA, HEAD_SHA, HEAD_SHA]
    inspector.clean = [True, True, True, True]
    runner, ledger, _commands = build_runner(
        tmp_path, github=github, inspector=inspector
    )
    identity = CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA)

    first = runner.run(identity, worktree)
    second = runner.run(identity, worktree)

    assert second == first
    lifecycle = ledger.latest_ci_run("acme/widgets", 17, HEAD_SHA)
    assert lifecycle is not None
    assert lifecycle["status"] == "completed"
    assert lifecycle["lease_version"] == 2
    assert lifecycle["receipt_id"] == first.receipt_id
    ledger.close()


def test_local_ci_runner_recovers_receipt_when_finalizer_reports_after_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)
    runner, ledger, _commands = build_runner(
        tmp_path, commands=RecordingRunner(fail_at=2)
    )
    original_finalize = ledger.finalize_ci_run

    def finalize_then_report_failure(*args: object, **kwargs: object) -> None:
        original_finalize(*args, **kwargs)
        raise RuntimeError("simulated finalizer acknowledgement loss")

    monkeypatch.setattr(ledger, "finalize_ci_run", finalize_then_report_failure)

    receipt = runner.run(CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA), worktree)

    assert receipt.status == "failed"
    assert ledger.ci_receipt_by_id("acme/widgets", 17, receipt.receipt_id) == receipt
    lifecycle = ledger.latest_ci_run("acme/widgets", 17, HEAD_SHA)
    assert lifecycle is not None
    assert lifecycle["status"] == "completed"
    assert lifecycle["receipt_id"] == receipt.receipt_id
    ledger.close()


def test_grouped_coordinator_preserves_typed_failed_receipt(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)
    runner, ledger, commands = build_runner(tmp_path, commands=RecordingRunner(fail_at=2))
    identity = CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA)
    receipt = runner.run(identity, worktree)
    job = CIAuditJob(identity=identity, worktree=worktree, failure_lanes=("unit",))

    class ReceiptRunner:
        def run(self, _identity: CIAuditIdentity, _worktree: Path) -> CIAuditReceipt:
            return receipt

    outcome = GroupedCICoordinator(lambda: ReceiptRunner(), max_parallel=1).run((job,))[0]

    assert outcome.error is None
    assert outcome.receipt is not None
    assert outcome.receipt.status == "failed"
    assert commands.calls[1][0] == ("python3", "scripts/run_static_lane.py")
    ledger.close()


def test_grouped_coordinator_preserves_runner_failure_reason(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)
    identity = CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA)
    job = CIAuditJob(identity=identity, worktree=worktree, failure_lanes=("unit",))

    class FailingRunner:
        def run(self, _identity: CIAuditIdentity, _worktree: Path) -> CIAuditReceipt:
            raise CIValidationError("Python interpreter mismatch", command_evidence=())

    outcome = GroupedCICoordinator(lambda: FailingRunner(), max_parallel=1).run((job,))[0]

    assert outcome.receipt is None
    assert outcome.error == "audit_failed: CIValidationError: Python interpreter mismatch"


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
    assert receipt.commands[0].argv == (
        "python3",
        "scripts/bootstrap_agent_workspace.py",
        "--venv",
        "link",
    )
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
def test_local_ci_runner_records_failed_receipt_on_invalid_or_raced_state(
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

    receipt = runner.run(CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA), worktree)

    assert receipt.status == "failed"
    if failure == "actions_race":
        assert receipt.commands
        assert all(command.returncode == 0 for command in receipt.commands)
    else:
        assert receipt.commands == ()
    assert receipt.failure_reason
    assert ledger.latest_ci_receipt_for_head(
        "acme/widgets",
        17,
        HEAD_SHA,
    ) == receipt
    assert ledger.latest_passing_ci_receipt(
        "acme/widgets",
        17,
        HEAD_SHA,
        manifest_digest=receipt.manifest_digest,
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


def test_repo_pinned_python_mismatch_is_environment_blocked(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)
    (worktree / ".python-version").write_text("3.13\n", encoding="utf-8")
    executable = worktree / ".venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    runner, ledger, commands = build_runner(tmp_path)
    runner._python_argv = (".venv/bin/python",)

    receipt = runner.run(CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA), worktree)

    assert receipt.status == "failed"
    assert "Python interpreter mismatch" in (receipt.failure_reason or "")
    assert receipt.commands[0].classification == "environment-blocked"
    assert commands.calls[0][0] == (
        str(executable),
        "-c",
        "import sys; print('.'.join(map(str, sys.version_info[:3])))",
    )
    ledger.close()


def test_ci_receipt_round_trip_rejects_coerced_or_dropped_evidence(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)
    runner, ledger, _commands = build_runner(tmp_path)
    receipt = runner.run(CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA), worktree)

    payload = receipt.to_payload()
    assert CIAuditReceipt.from_payload(payload) == receipt

    payload["actions_state"]["all_green"] = "true"  # type: ignore[index]
    with pytest.raises(ValueError, match="actions green"):
        CIAuditReceipt.from_payload(payload)

    payload = receipt.to_payload()
    payload["commands"].append("not-a-command")  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="invalid command"):
        CIAuditReceipt.from_payload(payload)

    payload = receipt.to_payload()
    payload["receipt_id"] = "not-a-sha"
    with pytest.raises(ValueError, match="receipt_id"):
        CIAuditReceipt.from_payload(payload)

    payload = receipt.to_payload()
    payload["commands"] = []
    payload["receipt_id"] = "0" * 64
    with pytest.raises(ValueError, match="no command evidence"):
        CIAuditReceipt.from_payload(payload)

    payload = receipt.to_payload()
    payload["ci_mode"] = "budget-exhausted-local-equivalent"
    with pytest.raises(ValueError, match="billing-blocked"):
        CIAuditReceipt.from_payload(payload)
    ledger.close()


@pytest.mark.parametrize("changed,expected", [("agent/worker.py", "passed"), ("installer/windows.ps1", "failed")])
def test_hermes_native_contract_runs_full_runner_without_lunabot_owner_files(tmp_path, changed, expected):
    from github_pr_feedback.ci_contract import manifest_path, HERMES_ENV_CHECK
    from github_pr_feedback.ci_runner import actions_disabled_local_ci_evidence
    root = tmp_path / "hermes"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/run_tests.sh").write_text("exit 0\n", encoding="utf-8")
    (root / "pyproject.toml").write_text('[project]\nname="hermes-agent"\n', encoding="utf-8")
    ledger = FeedbackLedger(tmp_path / "ci.sqlite3")
    commands = RecordingRunner()
    runner = LocalCIRunner(FakeGitHub(merge_state()), ledger, command_runner=commands,
                          inspector=FakeInspector(changed=(changed,)), python_argv=("python3",), now=lambda: NOW)
    receipt = runner.run(CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA), root)
    assert receipt.status == expected
    assert [call[0] for call in commands.calls] == [
        ("git", "diff", "--check", f"{BASE_SHA}..{HEAD_SHA}"),
        ("uv", "lock", "--check"), HERMES_ENV_CHECK, ("bash", "scripts/run_tests.sh")]
    assert (actions_disabled_local_ci_evidence(receipt, manifest_path(root).read_bytes()) is not None) == (expected == "passed")
    assert actions_disabled_local_ci_evidence(replace(receipt, commands=receipt.commands[:-1]),
                                             manifest_path(root).read_bytes()) is None
    assert actions_disabled_local_ci_evidence(
        replace(receipt, commands=receipt.commands[:2] + receipt.commands[3:]),
        manifest_path(root).read_bytes()) is None
    ledger.close()


def test_hermes_native_contract_does_not_claim_uncovered_platform_changes(tmp_path):
    from github_pr_feedback.ci_contract import hermes_commands, hermes_coverage_gap
    assert hermes_commands(tmp_path, BASE_SHA, HEAD_SHA, ("installer/windows.ps1",))
    assert hermes_coverage_gap(("installer/windows.ps1",)) is not None
    assert hermes_coverage_gap(("agent/worker.py",)) is None


def test_hermes_native_ci_uses_shared_workspace_lock_once(tmp_path):
    import json
    from github_pr_feedback.ci_contract import hermes_commands
    packages = ('apps/shared', 'apps/desktop', 'web')
    (tmp_path / 'package-lock.json').write_text(json.dumps({'packages': {p: {} for p in packages}}), encoding="utf-8")
    for package in packages:
        root = tmp_path / package
        root.mkdir(parents=True)
        (root / 'package.json').write_text(json.dumps({'scripts': {'test': 'vitest run'}}), encoding="utf-8")
    commands = hermes_commands(tmp_path, BASE_SHA, HEAD_SHA, ('apps/shared/src/client.ts',))
    assert [(argv, cwd) for argv, cwd, _ in commands if argv[:2] == ('npm', 'ci')] == [
        (('npm', 'ci', '--ignore-scripts'), tmp_path)]
    assert {cwd for argv, cwd, _ in commands if argv == ('npm', 'run', 'test')} == {
        tmp_path / package for package in packages}


@pytest.mark.parametrize("error_code", ["permission_denied", "authentication"])
def test_required_local_audit_reads_real_checks_without_admin_settings(tmp_path, error_code):
    worktree = tmp_path / "worktree"
    prepare_repository(worktree)

    class PublicReadGitHub(FakeGitHub):
        def actions_enabled(self, *args, **kwargs):
            raise GitHubClientError("settings unavailable", code=error_code)
        def get_check_state(self, repository, head_sha, *, actions_enabled_hint=None):
            assert actions_enabled_hint is True
            return CheckState(True, False, 2)

    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    commands = RecordingRunner()
    receipt = LocalCIRunner(PublicReadGitHub(merge_state()), ledger, command_runner=commands,
                           inspector=FakeInspector(), python_argv=("python3",), now=lambda: NOW,
                           required_local_ci=True).run(
        CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA), worktree)
    if error_code == "permission_denied":
        assert receipt.status == "passed"
        assert commands.calls
        assert receipt.actions_state == CheckState(True, False, 2)
    else:
        assert receipt.status == "failed"
        assert not commands.calls
    ledger.close()


@pytest.mark.parametrize("lag_at", [1, 2])
def test_mergeability_lag_releases_lease_without_a_failed_ci_receipt(tmp_path, lag_at):
    from github_pr_feedback.github_client import MergeStateStillComputingError

    worktree = tmp_path / "worktree"
    prepare_repository(worktree)

    class LaggingGitHub(FakeGitHub):
        reads = 0

        def get_merge_state(self, repository, number):
            self.reads += 1
            if self.reads == lag_at:
                raise MergeStateStillComputingError()
            return merge_state()

    github = LaggingGitHub(merge_state())
    inspector = FakeInspector()
    runner, ledger, commands = build_runner(tmp_path, github=github, inspector=inspector)
    identity = CIAuditIdentity("acme/widgets", 17, BASE_SHA, HEAD_SHA)
    with pytest.raises(MergeStateStillComputingError):
        runner.run(identity, worktree)
    assert ledger.latest_ci_receipt_for_head("acme/widgets", 17, HEAD_SHA) is None
    run = ledger.latest_ci_run("acme/widgets", 17, HEAD_SHA)
    assert run["status"] != "running"
    assert run["receipt_id"] is None
    github.checks = [CheckState(actions_enabled=False, all_green=True, check_count=0)] * 2
    inspector.heads = [HEAD_SHA, HEAD_SHA]
    inspector.clean = [True, True]
    assert runner.run(identity, worktree).status == "passed"
