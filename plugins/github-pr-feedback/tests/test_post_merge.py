from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from github_pr_feedback.ci_runner import CompletedCommand
from github_pr_feedback.ledger import FeedbackLedger
from github_pr_feedback.merge_controller import MergeReceipt
from github_pr_feedback.policy import PostMergePolicy
from github_pr_feedback.post_merge import (
    BundleIdentity,
    DeploymentError,
    PostMergeExecutor,
    ProcessRecord,
)


NOW = datetime(2026, 8, 25, 13, 0, tzinfo=UTC)
MERGE_SHA = "c" * 40


def policy(root: Path) -> PostMergePolicy:
    return PostMergePolicy(
        deployment_path=root,
        protected_runtime_entry="main.py",
        package_argv=("python3", "tools/tb.py", "gui-package-macos", "--replace", "--json"),
        bundle_path="desktop/macos/Example/build/Example.app",
        bundle_identifier="com.example.local.operator",
        relaunch_argv=("/usr/bin/open", "-n"),
    )


def merge_receipt() -> MergeReceipt:
    return MergeReceipt(
        repository="acme/widgets",
        pr_number=17,
        author_login="owner",
        base_branch="stable",
        tested_head_sha="a" * 40,
        ci_receipt_id="d" * 64,
        snapshot_digest="e" * 64,
        method="squash",
        merge_commit_oid=MERGE_SHA,
        merged_at=NOW,
        executor="merge-test",
    )


class FakeProcesses:
    def __init__(self, censuses: list[tuple[ProcessRecord, ...]]) -> None:
        self.censuses = censuses
        self.calls: list[object] = []

    def census(self) -> tuple[ProcessRecord, ...]:
        self.calls.append("census")
        return self.censuses.pop(0)

    def terminate(self, pid: int) -> None:
        self.calls.append(("terminate", pid))


class FakeRepository:
    def __init__(self, root: Path, *, error: str | None = None) -> None:
        self.root = root
        self.error = error
        self.calls: list[object] = []

    def prepare(self, merge: MergeReceipt, post_policy: PostMergePolicy) -> str:
        self.calls.append(("prepare", merge.merge_commit_oid, post_policy.deployment_path))
        if self.error:
            raise DeploymentError(self.error)
        return merge.merge_commit_oid

    def require_clean(self, root: Path) -> None:
        self.calls.append(("require_clean", root))
        if self.error == "dirty_after_build":
            raise DeploymentError(self.error)


class FakeCommandRunner:
    def __init__(self, results: list[CompletedCommand]) -> None:
        self.results = results
        self.calls: list[tuple[tuple[str, ...], Path, int]] = []

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> CompletedCommand:
        self.calls.append((argv, cwd, timeout))
        return self.results.pop(0)


class FakeBundleInspector:
    def __init__(self, identity: BundleIdentity) -> None:
        self.identity = identity
        self.calls: list[Path] = []

    def inspect(self, bundle: Path) -> BundleIdentity:
        self.calls.append(bundle)
        return self.identity


def completed(stdout: str = '{"status":"ok"}') -> CompletedCommand:
    return CompletedCommand(0, stdout, "", 25, False)


def build_executor(
    tmp_path: Path,
    *,
    censuses: list[tuple[ProcessRecord, ...]] | None = None,
    repository_error: str | None = None,
    command_results: list[CompletedCommand] | None = None,
    bundle_identity: BundleIdentity | None = None,
) -> tuple[
    PostMergeExecutor,
    FeedbackLedger,
    FakeProcesses,
    FakeRepository,
    FakeCommandRunner,
]:
    root = tmp_path / "deployment"
    root.mkdir()
    bundle = root / "desktop/macos/Example/build/Example.app"
    executable = bundle / "Contents/MacOS/Example"
    processes = FakeProcesses(censuses or [(), ()])
    repository = FakeRepository(root, error=repository_error)
    commands = FakeCommandRunner(command_results or [completed(), completed("")])
    bundles = FakeBundleInspector(
        bundle_identity or BundleIdentity("com.example.local.operator", executable)
    )
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    executor = PostMergeExecutor(
        policy(root),
        ledger,
        processes=processes,
        repository=repository,
        command_runner=commands,
        bundle_inspector=bundles,
        now=lambda: NOW,
    )
    return executor, ledger, processes, repository, commands


def test_post_merge_runs_census_prepare_build_verify_relaunch_and_final_census(
    tmp_path: Path,
) -> None:
    root = tmp_path / "deployment"
    app_executable = root / "desktop/macos/Example/build/Example.app/Contents/MacOS/Example"
    old_app = ProcessRecord(42, app_executable, (str(app_executable),), root)
    executor, ledger, processes, repository, commands = build_executor(
        tmp_path, censuses=[(old_app,), ()]
    )

    receipt = executor.run(merge_receipt())

    bundle = root / "desktop/macos/Example/build/Example.app"
    assert receipt.status == "completed"
    assert receipt.deployed_sha == MERGE_SHA
    assert receipt.relaunched is True
    assert processes.calls == ["census", ("terminate", 42), "census"]
    assert repository.calls == [
        ("prepare", MERGE_SHA, root),
        ("require_clean", root),
    ]
    assert commands.calls == [
        (policy(root).package_argv, root, 3600),
        (policy(root).relaunch_argv + (str(bundle),), root, 30),
    ]
    assert ledger.latest_deployment_receipt("acme/widgets", 17) == receipt
    ledger.close()


@pytest.mark.parametrize(
    "record",
    [
        ProcessRecord(7, Path("/usr/bin/python3"), ("python3", "main.py"), None),
        ProcessRecord(
            8,
            Path("/usr/bin/python3"),
            ("python3", "/tmp/deployment/main.py"),
            Path("/tmp/deployment"),
        ),
    ],
)
def test_post_merge_fails_closed_when_protected_runtime_is_present_or_ambiguous(
    tmp_path: Path, record: ProcessRecord
) -> None:
    root = tmp_path / "deployment"
    exact = ProcessRecord(
        record.pid,
        record.executable,
        (
            "python3",
            str(root / "main.py") if record.cwd is not None else "main.py",
        ),
        root if record.cwd is not None else None,
    )
    executor, ledger, processes, repository, commands = build_executor(
        tmp_path, censuses=[(exact,)]
    )

    receipt = executor.run(merge_receipt())

    assert receipt.status == "failed"
    assert receipt.blocker == "protected_runtime_present_or_ambiguous"
    assert repository.calls == []
    assert commands.calls == []
    assert all(not isinstance(call, tuple) or call[0] != "terminate" for call in processes.calls)
    ledger.close()


@pytest.mark.parametrize(
    "repository_error,results,bundle_identifier,blocker",
    [
        ("deployment_worktree_dirty", None, None, "deployment_worktree_dirty"),
        (
            None,
            [CompletedCommand(1, "", "failure", 25, False)],
            None,
            "package_failed",
        ),
        (None, [completed()], "com.example.wrong", "bundle_identity_mismatch"),
        ("dirty_after_build", [completed()], None, "dirty_after_build"),
    ],
)
def test_post_merge_records_separate_failure_without_changing_merge_truth(
    tmp_path: Path,
    repository_error: str | None,
    results: list[CompletedCommand] | None,
    bundle_identifier: str | None,
    blocker: str,
) -> None:
    identity = None
    if bundle_identifier:
        identity = BundleIdentity(bundle_identifier, tmp_path / "wrong")
    executor, ledger, _processes, _repository, _commands = build_executor(
        tmp_path,
        repository_error=repository_error,
        command_results=results,
        bundle_identity=identity,
    )

    receipt = executor.run(merge_receipt())

    assert receipt.status == "failed"
    assert receipt.blocker == blocker
    assert merge_receipt().merge_commit_oid == MERGE_SHA
    ledger.close()


def test_post_launch_protected_runtime_is_a_failed_deployment_receipt(tmp_path: Path) -> None:
    root = tmp_path / "deployment"
    appeared = ProcessRecord(
        99,
        Path("/usr/bin/python3"),
        ("python3", str(root / "main.py")),
        root,
    )
    executor, ledger, processes, _repository, commands = build_executor(
        tmp_path, censuses=[(), (appeared,)]
    )

    receipt = executor.run(merge_receipt())

    assert receipt.status == "failed"
    assert receipt.relaunched is True
    assert receipt.blocker == "protected_runtime_appeared_after_relaunch"
    assert len(commands.calls) == 2
    assert all(not isinstance(call, tuple) or call[0] != "terminate" for call in processes.calls)
    ledger.close()
