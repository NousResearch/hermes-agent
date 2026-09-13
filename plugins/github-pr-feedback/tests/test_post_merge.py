import json
from datetime import UTC, datetime
from pathlib import Path
import pytest
from types import SimpleNamespace
from unittest.mock import Mock

from github_pr_feedback.ci_runner import CompletedCommand
from github_pr_feedback.merge_controller import MergeReceipt
from github_pr_feedback.post_merge import (
    BundleIdentity,
    DeploymentError,
    GitDeploymentRepository,
    PostMergeExecutor,
    ProcessRecord,
    SystemProcessController,
    _require_package_provenance,
    _require_runtime_absent,
    _wait_for_process_to_appear,
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


def test_process_census_reports_malformed_ps_rows_as_deployment_errors(monkeypatch):
    monkeypatch.setattr(
        "github_pr_feedback.post_merge.subprocess.run",
        lambda *_args, **_kwargs: type(
            "Completed", (), {"returncode": 0, "stdout": "123 /usr/bin/example foo'\n"}
        )(),
    )

    with pytest.raises(DeploymentError, match="process_census_ambiguous"):
        SystemProcessController().census()


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
    processes.census.side_effect = [(), (), (application,), (), (application,), ()]
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
    blocked_commands.run.assert_not_called()


def test_post_merge_rechecks_protected_runtime_after_relaunch():
    policy = _post_merge_policy(Path("/deployment"))
    merge_sha = "a" * 40
    application = ProcessRecord(
        123, Path("/deployment/Example.app/Contents/MacOS/Example"), (), None
    )
    protected = ProcessRecord(
        456, Path("/usr/bin/python"), ("runtime.py",), Path("/deployment")
    )
    processes = Mock()
    processes.census.side_effect = [(), (), (), (application,), (protected,)]
    repository = Mock()
    repository.prepare.return_value = merge_sha
    commands = Mock()
    commands.run.side_effect = [
        CompletedCommand(0, '{"source_sha": "' + merge_sha + '"}', "", 1, False),
        CompletedCommand(0, "", "", 1, False),
    ]
    bundles = Mock()
    bundles.inspect.return_value = BundleIdentity("com.example.app", application.executable)

    receipt = PostMergeExecutor(
        policy,
        Mock(),
        processes=processes,
        repository=repository,
        command_runner=commands,
        bundle_inspector=bundles,
    ).run(_merge_receipt(merge_sha))

    assert receipt.status == "failed"
    assert receipt.blocker == "protected_runtime_appeared_after_relaunch"


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


def test_post_merge_executor_passes_processes_before_controller_to_shutdown_wait(monkeypatch, tmp_path):
    bundle = tmp_path / "Hermes.app"
    executable = bundle / "Contents" / "MacOS" / "Hermes"
    process = ProcessRecord(123, executable, (str(executable),), None)
    policy = PostMergePolicy(
        deployment_path=tmp_path,
        protected_runtime_entry="main.py",
        package_argv=("package",),
        bundle_path="Hermes.app",
        bundle_identifier="com.example.hermes",
        relaunch_argv=("open",),
    )
    merge = MergeReceipt(
        repository="mrkillbob/hermes-agent",
        pr_number=83,
        author_login="mrkillbob",
        base_branch="main",
        tested_head_sha="a" * 40,
        ci_receipt_id="b" * 64,
        snapshot_digest="c" * 64,
        method="squash",
        merge_commit_oid="d" * 40,
        merged_at=datetime(2026, 9, 12, tzinfo=UTC),
        executor="test",
    )

    class Processes:
        def __init__(self):
            self.censuses = [(), (), (process,), (), (process,), ()]
            self.terminated = []

        def census(self):
            return tuple(self.censuses.pop(0))

        def terminate(self, pid):
            self.terminated.append(pid)

    class Repository:
        def prepare(self, _merge, _policy):
            return "e" * 40

        def require_clean(self, _root):
            return None

    class Commands:
        def run(self, argv, **_kwargs):
            if argv == policy.package_argv:
                return CompletedCommand(0, json.dumps({"source_sha": "e" * 40}), "", 1, False)
            return CompletedCommand(0, "", "", 1, False)

    class Bundles:
        def inspect(self, _bundle):
            return BundleIdentity(policy.bundle_identifier, executable)

    waited = []

    def wait(processes, controller):
        waited.append((processes, controller))

    monkeypatch.setattr("github_pr_feedback.post_merge._wait_for_processes_to_exit", wait)
    processes = Processes()
    ledger = SimpleNamespace(record_deployment_receipt=lambda _receipt: None)
    receipt = PostMergeExecutor(
        policy,
        ledger,
        processes=processes,
        repository=Repository(),
        command_runner=Commands(),
        bundle_inspector=Bundles(),
        now=lambda: datetime(2026, 9, 12, tzinfo=UTC),
    ).run(merge)

    assert receipt.status == "completed"
    assert processes.terminated == [123]
    assert waited == [([process], processes)]


def test_absolute_nonmatching_runtime_argument_is_not_ambiguous(tmp_path):
    policy = PostMergePolicy(
        deployment_path=tmp_path,
        protected_runtime_entry="main.py",
        package_argv=("package",),
        bundle_path="Hermes.app",
        bundle_identifier="com.example.hermes",
        relaunch_argv=("open",),
    )

    _require_runtime_absent(
        (ProcessRecord(123, Path("/usr/bin/python"), ("/other/project/main.py",), None),),
        policy,
    )


def test_relaunch_wait_fails_when_the_bundle_process_never_appears():
    class Controller:
        def census(self):
            return ()

    with pytest.raises(DeploymentError, match="relaunched_bundle_missing"):
        _wait_for_process_to_appear(
            Path("/Applications/Hermes.app/Contents/MacOS/Hermes"),
            Controller(),
            timeout=0,
        )


def test_process_census_preserves_executable_paths_with_spaces(monkeypatch):
    import sys

    executable = "/Applications/Hermes Local.app/Contents/MacOS/Hermes Local"
    fake_psutil = SimpleNamespace(
        process_iter=lambda _attrs: iter(
            [
                SimpleNamespace(
                    info={
                        "pid": 123,
                        "exe": executable,
                        "cmdline": [executable, "--serve"],
                        "cwd": "/Applications",
                    }
                )
            ]
        ),
        NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
        ZombieProcess=type("ZombieProcess", (Exception,), {}),
        AccessDenied=type("AccessDenied", (Exception,), {}),
        Error=Exception,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert SystemProcessController().census() == (
        ProcessRecord(
            123,
            Path(executable),
            (executable, "--serve"),
            Path("/Applications"),
        ),
    )


def test_process_census_fails_closed_when_required_attributes_are_unreadable(
    monkeypatch,
):
    import sys

    fake_psutil = SimpleNamespace(
        process_iter=lambda _attrs: iter(
            [SimpleNamespace(info={"pid": 123, "exe": None, "cmdline": None})]
        ),
        NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
        ZombieProcess=type("ZombieProcess", (Exception,), {}),
        AccessDenied=type("AccessDenied", (Exception,), {}),
        Error=Exception,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    with pytest.raises(DeploymentError, match="process_census_ambiguous"):
        SystemProcessController().census()
