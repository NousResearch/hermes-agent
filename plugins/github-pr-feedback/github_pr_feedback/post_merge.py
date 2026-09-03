"""Runtime-gated post-merge rebuild and desktop application relaunch."""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import re
import shlex
import signal
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping, Protocol

from .ci_runner import CICommandRunner, CompletedCommand, SubprocessCICommandRunner
from .ledger import FeedbackLedger
from .merge_controller import MergeReceipt
from .policy import PostMergePolicy


class DeploymentError(RuntimeError):
    """A post-merge safety gate failed without changing merge truth."""


@dataclass(frozen=True, slots=True)
class ProcessRecord:
    pid: int
    executable: Path
    argv: tuple[str, ...]
    cwd: Path | None


@dataclass(frozen=True, slots=True)
class BundleIdentity:
    identifier: str
    executable_path: Path


@dataclass(frozen=True, slots=True)
class DeploymentReceipt:
    receipt_id: str
    repository: str
    pr_number: int
    merge_commit_oid: str
    status: str
    deployed_sha: str | None
    bundle_path: str | None
    relaunched: bool
    blocker: str | None
    completed_at: datetime

    def to_payload(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "repository": self.repository,
            "pr_number": self.pr_number,
            "merge_commit_oid": self.merge_commit_oid,
            "status": self.status,
            "deployed_sha": self.deployed_sha,
            "bundle_path": self.bundle_path,
            "relaunched": self.relaunched,
            "blocker": self.blocker,
            "completed_at": self.completed_at.isoformat(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> DeploymentReceipt:
        return cls(
            receipt_id=str(payload["receipt_id"]),
            repository=str(payload["repository"]),
            pr_number=int(payload["pr_number"]),
            merge_commit_oid=str(payload["merge_commit_oid"]),
            status=str(payload["status"]),
            deployed_sha=(
                str(payload["deployed_sha"]) if payload.get("deployed_sha") is not None else None
            ),
            bundle_path=(
                str(payload["bundle_path"]) if payload.get("bundle_path") is not None else None
            ),
            relaunched=bool(payload["relaunched"]),
            blocker=str(payload["blocker"]) if payload.get("blocker") is not None else None,
            completed_at=datetime.fromisoformat(str(payload["completed_at"])),
        )


class ProcessController(Protocol):
    def census(self) -> tuple[ProcessRecord, ...]: ...

    def terminate(self, pid: int) -> None: ...


class DeploymentRepository(Protocol):
    def prepare(self, merge: MergeReceipt, policy: PostMergePolicy) -> str: ...

    def require_clean(self, root: Path) -> None: ...


class BundleInspector(Protocol):
    def inspect(self, bundle: Path) -> BundleIdentity: ...


class SystemProcessController:
    def census(self) -> tuple[ProcessRecord, ...]:
        try:
            completed = subprocess.run(
                ("ps", "-axo", "pid=,comm=,args="),
                check=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise DeploymentError("process_census_unavailable") from error
        if completed.returncode != 0:
            raise DeploymentError("process_census_unavailable")
        records: list[ProcessRecord] = []
        for line in completed.stdout.splitlines():
            parts = line.strip().split(maxsplit=2)
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
                argv = tuple(shlex.split(parts[2]))
            except (ValueError, shlex.Error) as error:
                raise DeploymentError("process_census_ambiguous") from error
            if not argv:
                raise DeploymentError("process_census_ambiguous")
            executable = Path(argv[0]) if Path(argv[0]).is_absolute() else Path(parts[1])
            records.append(ProcessRecord(pid, executable, argv, None))
        return tuple(records)

    def terminate(self, pid: int) -> None:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as error:
            raise DeploymentError("verified_application_quit_failed") from error


class GitDeploymentRepository:
    def _run(self, root: Path, *arguments: str, allow_one: bool = False) -> str:
        try:
            completed = subprocess.run(
                ("git", "-C", str(root), *arguments),
                check=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise DeploymentError("deployment_git_unavailable") from error
        if completed.returncode != 0 and not (allow_one and completed.returncode == 1):
            raise DeploymentError("deployment_git_failed")
        return completed.stdout if completed.returncode == 0 else ""

    def prepare(self, merge: MergeReceipt, policy: PostMergePolicy) -> str:
        root = policy.deployment_path.resolve()
        top = Path(self._run(root, "rev-parse", "--show-toplevel").strip()).resolve()
        if top != root:
            raise DeploymentError("deployment_worktree_identity_mismatch")
        self.require_clean(root)
        for marker in ("MERGE_HEAD", "REBASE_HEAD", "CHERRY_PICK_HEAD"):
            marker_path = Path(self._run(root, "rev-parse", "--git-path", marker).strip())
            if not marker_path.is_absolute():
                marker_path = root / marker_path
            if marker_path.exists():
                raise DeploymentError("deployment_git_operation_active")
        branch = self._run(root, "branch", "--show-current").strip()
        if branch != merge.base_branch:
            raise DeploymentError("deployment_branch_mismatch")
        remote = self._run(root, "remote", "get-url", "origin").strip()
        if _repository_from_remote(remote) != merge.repository:
            raise DeploymentError("deployment_remote_mismatch")
        self._run(root, "fetch", "--prune", "origin", merge.base_branch)
        remote_ref = f"refs/remotes/origin/{merge.base_branch}"
        deployed_sha = self._run(root, "rev-parse", remote_ref).strip().lower()
        ancestor = subprocess.run(
            (
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                merge.merge_commit_oid,
                deployed_sha,
            ),
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if ancestor.returncode != 0:
            raise DeploymentError("merged_commit_not_on_remote_base")
        self._run(root, "merge", "--ff-only", remote_ref)
        if self._run(root, "rev-parse", "HEAD").strip().lower() != deployed_sha:
            raise DeploymentError("deployment_head_mismatch")
        self.require_clean(root)
        return deployed_sha

    def require_clean(self, root: Path) -> None:
        if self._run(root, "status", "--porcelain=v1", "--untracked-files=all"):
            raise DeploymentError("deployment_worktree_dirty")


class PlistBundleInspector:
    def inspect(self, bundle: Path) -> BundleIdentity:
        plist_path = bundle / "Contents/Info.plist"
        if not bundle.is_dir() or not plist_path.is_file():
            raise DeploymentError("bundle_missing")
        try:
            with plist_path.open("rb") as handle:
                payload = plistlib.load(handle)
            identifier = payload["CFBundleIdentifier"]
            executable_name = payload["CFBundleExecutable"]
        except (OSError, KeyError, plistlib.InvalidFileException) as error:
            raise DeploymentError("bundle_metadata_invalid") from error
        if not isinstance(identifier, str) or not isinstance(executable_name, str):
            raise DeploymentError("bundle_metadata_invalid")
        executable = bundle / "Contents/MacOS" / executable_name
        if not executable.is_file():
            raise DeploymentError("bundle_executable_missing")
        return BundleIdentity(identifier, executable.resolve())


class PostMergeExecutor:
    def __init__(
        self,
        policy: PostMergePolicy,
        ledger: FeedbackLedger,
        *,
        processes: ProcessController | None = None,
        repository: DeploymentRepository | None = None,
        command_runner: CICommandRunner | None = None,
        bundle_inspector: BundleInspector | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._policy = policy
        self._ledger = ledger
        self._processes = processes or SystemProcessController()
        self._repository = repository or GitDeploymentRepository()
        self._commands = command_runner or SubprocessCICommandRunner()
        self._bundles = bundle_inspector or PlistBundleInspector()
        self._now = now or (lambda: datetime.now(UTC))

    def run(self, merge: MergeReceipt) -> DeploymentReceipt:
        deployed_sha: str | None = None
        relaunched = False
        bundle = (self._policy.deployment_path / self._policy.bundle_path).resolve()
        try:
            pre_census = self._processes.census()
            _require_runtime_absent(pre_census, self._policy)
            deployed_sha = self._repository.prepare(merge, self._policy)
            package = self._commands.run(
                self._policy.package_argv,
                cwd=self._policy.deployment_path,
                env=dict(os.environ),
                timeout=3600,
            )
            if package.returncode != 0 or package.timed_out:
                raise DeploymentError("package_failed")
            try:
                package_payload = json.loads(package.stdout)
            except (json.JSONDecodeError, TypeError) as error:
                raise DeploymentError("package_output_invalid") from error
            if not isinstance(package_payload, dict):
                raise DeploymentError("package_output_invalid")
            identity = self._bundles.inspect(bundle)
            if (
                identity.identifier != self._policy.bundle_identifier
                or identity.executable_path.parent != (bundle / "Contents/MacOS").resolve()
            ):
                raise DeploymentError("bundle_identity_mismatch")
            self._repository.require_clean(self._policy.deployment_path)
            for process in pre_census:
                if process.executable.resolve() == identity.executable_path.resolve():
                    self._processes.terminate(process.pid)
            relaunch = self._commands.run(
                self._policy.relaunch_argv + (str(bundle),),
                cwd=self._policy.deployment_path,
                env=dict(os.environ),
                timeout=30,
            )
            if relaunch.returncode != 0 or relaunch.timed_out:
                raise DeploymentError("relaunch_failed")
            relaunched = True
            _require_runtime_absent(
                self._processes.census(),
                self._policy,
                blocker="protected_runtime_appeared_after_relaunch",
            )
        except DeploymentError as error:
            return self._record(
                merge,
                status="failed",
                deployed_sha=deployed_sha,
                bundle=bundle if deployed_sha else None,
                relaunched=relaunched,
                blocker=str(error),
            )
        return self._record(
            merge,
            status="completed",
            deployed_sha=deployed_sha,
            bundle=bundle,
            relaunched=True,
            blocker=None,
        )

    def _record(
        self,
        merge: MergeReceipt,
        *,
        status: str,
        deployed_sha: str | None,
        bundle: Path | None,
        relaunched: bool,
        blocker: str | None,
    ) -> DeploymentReceipt:
        completed_at = _aware_utc(self._now())
        identity = {
            "repository": merge.repository,
            "pr": merge.pr_number,
            "merge": merge.merge_commit_oid,
            "status": status,
            "deployed": deployed_sha,
            "relaunched": relaunched,
            "blocker": blocker,
            "completed": completed_at.isoformat(),
        }
        receipt = DeploymentReceipt(
            receipt_id=hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            repository=merge.repository,
            pr_number=merge.pr_number,
            merge_commit_oid=merge.merge_commit_oid,
            status=status,
            deployed_sha=deployed_sha,
            bundle_path=str(bundle) if bundle else None,
            relaunched=relaunched,
            blocker=blocker,
            completed_at=completed_at,
        )
        self._ledger.record_deployment_receipt(receipt)
        return receipt


def _require_runtime_absent(
    records: tuple[ProcessRecord, ...],
    policy: PostMergePolicy,
    *,
    blocker: str = "protected_runtime_present_or_ambiguous",
) -> None:
    protected = (policy.deployment_path / policy.protected_runtime_entry).resolve()
    protected_name = Path(policy.protected_runtime_entry).name
    for record in records:
        for argument in record.argv:
            candidate = Path(argument)
            if candidate.is_absolute() and candidate.resolve() == protected:
                raise DeploymentError(blocker)
            if candidate.name != protected_name:
                continue
            if record.cwd is None:
                raise DeploymentError(blocker)
            if (record.cwd / candidate).resolve() == protected:
                raise DeploymentError(blocker)


def _repository_from_remote(remote: str) -> str:
    patterns = (
        r"^git@github\.com:([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?$",
        r"^https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$",
        r"^ssh://git@github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, remote)
        if match:
            return match.group(1)
    raise DeploymentError("deployment_remote_invalid")


def _aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DeploymentError("deployment_clock_invalid")
    return value.astimezone(UTC)
