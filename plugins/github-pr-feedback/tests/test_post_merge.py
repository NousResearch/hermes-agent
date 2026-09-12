from datetime import UTC, datetime
from pathlib import Path
import pytest
from unittest.mock import Mock

from github_pr_feedback.ci_runner import CompletedCommand
from github_pr_feedback.merge_controller import MergeReceipt
from github_pr_feedback.post_merge import (
    BundleIdentity,
    DeploymentError,
    GitDeploymentRepository,
    PostMergeExecutor,
    ProcessRecord,
    _require_package_provenance,
    _wait_for_processes_to_exit,
)
from github_pr_feedback.policy import PostMergePolicy


def test_package_provenance_requires_the_exact_full_source_sha():
    source_sha = "a" * 40

    _require_package_provenance({"source_sha": source_sha}, source_sha)

    with pytest.raises(DeploymentError, match="package_provenance_missing"):
        _require_package_provenance({}, source_sha)
    with pytest.raises(DeploymentError, match="package_provenance_mismatch"):
        _require_package_provenance({"source_sha": "b" * 40}, source_sha)


def test_process_shutdown_wait_rechecks_the_current_census():
    process = ProcessRecord(123, Path("/Applications/Hermes.app/Contents/MacOS/Hermes"), (), None)

    class Controller:
        def __init__(self):
            self.censuses = [[process], []]

        def census(self):
            return tuple(self.censuses.pop(0))

    controller = Controller()
    _wait_for_processes_to_exit([process], controller, timeout=0.2)
    assert controller.censuses == []


def _merge_receipt(merge_commit_oid: str) -> MergeReceipt:
    return MergeReceipt(
        repository="acme/widgets",
        pr_number=10,
        author_login="owner",
        base_branch="main",
        tested_head_sha="b" * 40,
        ci_receipt_id="c" * 64,
        snapshot_digest="d" * 64,
        method="squash",
        merge_commit_oid=merge_commit_oid,
        merged_at=datetime.now(UTC),
        executor="test",
    )


def _post_merge_policy(deployment_path: Path) -> PostMergePolicy:
    return PostMergePolicy(
        deployment_path=deployment_path,
        protected_runtime_entry="runtime.py",
        package_argv=("package",),
        bundle_path="Example.app",
        bundle_identifier="com.example.app",
        relaunch_argv=("relaunch",),
    )


def test_post_merge_rechecks_runtime_before_shutdown_and_waits_for_verified_process():
    policy = _post_merge_policy(Path("/deployment"))
    merge_sha = "a" * 40
    application = ProcessRecord(
        123,
        Path("/deployment/Example.app/Contents/MacOS/Example"),
        (),
        None,
    )
    processes = Mock()
    processes.census.side_effect = [(), (application,), (), ()]
    repository = Mock()
    repository.prepare.return_value = merge_sha
    commands = Mock()
    commands.run.side_effect = [
        CompletedCommand(0, '{"source_sha": "' + merge_sha + '"}', "", 1, False),
        CompletedCommand(0, "", "", 1, False),
    ]
    bundles = Mock()
    bundles.inspect.return_value = BundleIdentity("com.example.app", application.executable)
    ledger = Mock()
    receipt = PostMergeExecutor(
        policy,
        ledger,
        processes=processes,
        repository=repository,
        command_runner=commands,
        bundle_inspector=bundles,
    ).run(_merge_receipt(merge_sha))

    assert receipt.status == "completed"
    processes.terminate.assert_called_once_with(123)
    assert commands.run.call_count == 2
    ledger.record_deployment_receipt.assert_called_once_with(receipt)

    protected = ProcessRecord(
        456, Path("/usr/bin/python"), ("runtime.py",), Path("/deployment")
    )
    blocked_processes = Mock()
    blocked_processes.census.side_effect = [(), (protected,)]
    blocked_commands = Mock()
    blocked_commands.run.return_value = CompletedCommand(
        0, '{"source_sha": "' + merge_sha + '"}', "", 1, False
    )
    blocked_bundles = Mock()
    blocked_bundles.inspect.return_value = BundleIdentity(
        "com.example.app", application.executable
    )
    blocked = PostMergeExecutor(
        policy,
        Mock(),
        processes=blocked_processes,
        repository=repository,
        command_runner=blocked_commands,
        bundle_inspector=blocked_bundles,
    ).run(_merge_receipt(merge_sha))

    assert blocked.status == "failed"
    assert blocked.blocker == "protected_runtime_present_or_ambiguous"
    blocked_processes.terminate.assert_not_called()
    blocked_commands.run.assert_called_once()


def test_post_merge_rejects_an_advanced_remote_base_before_fast_forward():
    root = Path("/deployment")
    policy = _post_merge_policy(root)
    merge_sha = "a" * 40
    advanced_sha = "b" * 40
    outputs = {
        ("rev-parse", "--show-toplevel"): str(root),
        ("branch", "--show-current"): "main",
        ("remote", "get-url", "origin"): "https://github.com/acme/widgets.git",
        ("rev-parse", "refs/remotes/origin/main"): advanced_sha,
    }
    repository = GitDeploymentRepository()
    repository._run = Mock(side_effect=lambda _root, *arguments, **_kwargs: outputs.get(arguments, ""))

    with pytest.raises(DeploymentError, match="remote_base_merge_commit_mismatch"):
        repository.prepare(_merge_receipt(merge_sha), policy)

    assert not any("merge" in call.args for call in repository._run.call_args_list)
