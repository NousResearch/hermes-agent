"""Deterministic exact-head local-CI execution and bounded evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Mapping, Protocol

from .github_client import CheckState, GitHubClient, PullRequestMergeState
from .ledger import FeedbackLedger


_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_REQUIRED_SCRIPTS = (
    "scripts/check_ci_governance.py",
    "scripts/run_hygiene_lane.py",
    "scripts/run_static_lane.py",
    "scripts/run_test_lane.py",
    "scripts/run_local_ci_audit.py",
)
_COMMAND_TIMEOUT_SECONDS = 3600
_BOOTSTRAP_TIMEOUT_SECONDS = 900
_CI_RUN_LEASE = timedelta(hours=2)


class CIValidationError(RuntimeError):
    """The requested CI run was not authoritative for its claimed identity."""


@dataclass(frozen=True, slots=True)
class CIAuditIdentity:
    repository: str
    pr_number: int
    base_sha: str
    head_sha: str

    def __post_init__(self) -> None:
        if not isinstance(self.repository, str) or not _REPOSITORY.fullmatch(self.repository):
            raise ValueError("repository must be an exact owner/repository name")
        if (
            not isinstance(self.pr_number, int)
            or isinstance(self.pr_number, bool)
            or self.pr_number < 1
        ):
            raise ValueError("pr_number must be a positive integer")
        for field in ("base_sha", "head_sha"):
            value = getattr(self, field)
            if not isinstance(value, str) or not _SHA.fullmatch(value):
                raise ValueError(f"{field} must be a full hexadecimal SHA")
            object.__setattr__(self, field, value.lower())


@dataclass(frozen=True, slots=True)
class CompletedCommand:
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool


@dataclass(frozen=True, slots=True)
class CommandEvidence:
    argv: tuple[str, ...]
    cwd: str
    returncode: int
    duration_ms: int
    timed_out: bool
    stdout_sha256: str
    stderr_sha256: str
    classification: str


@dataclass(frozen=True, slots=True)
class CIAuditReceipt:
    receipt_id: str
    identity: CIAuditIdentity
    manifest_digest: str
    status: str
    started_at: datetime
    completed_at: datetime
    actions_state: CheckState
    commands: tuple[CommandEvidence, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "identity": {
                "repository": self.identity.repository,
                "pr_number": self.identity.pr_number,
                "base_sha": self.identity.base_sha,
                "head_sha": self.identity.head_sha,
            },
            "manifest_digest": self.manifest_digest,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "actions_state": {
                "actions_enabled": self.actions_state.actions_enabled,
                "all_green": self.actions_state.all_green,
                "check_count": self.actions_state.check_count,
                "billing_blocked": self.actions_state.billing_blocked,
            },
            "commands": [
                {
                    "argv": list(command.argv),
                    "cwd": command.cwd,
                    "returncode": command.returncode,
                    "duration_ms": command.duration_ms,
                    "timed_out": command.timed_out,
                    "stdout_sha256": command.stdout_sha256,
                    "stderr_sha256": command.stderr_sha256,
                    "classification": command.classification,
                }
                for command in self.commands
            ],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> CIAuditReceipt:
        identity = payload["identity"]
        actions = payload["actions_state"]
        commands = payload["commands"]
        if not isinstance(identity, Mapping) or not isinstance(actions, Mapping):
            raise ValueError("CI receipt payload has invalid identity")
        if not isinstance(commands, list):
            raise ValueError("CI receipt payload has invalid commands")
        return cls(
            receipt_id=str(payload["receipt_id"]),
            identity=CIAuditIdentity(
                str(identity["repository"]),
                int(identity["pr_number"]),
                str(identity["base_sha"]),
                str(identity["head_sha"]),
            ),
            manifest_digest=str(payload["manifest_digest"]),
            status=str(payload["status"]),
            started_at=datetime.fromisoformat(str(payload["started_at"])),
            completed_at=datetime.fromisoformat(str(payload["completed_at"])),
            actions_state=CheckState(
                actions_enabled=bool(actions["actions_enabled"]),
                all_green=bool(actions["all_green"]),
                check_count=int(actions["check_count"]),
                billing_blocked=bool(actions.get("billing_blocked", False)),
            ),
            commands=tuple(
                CommandEvidence(
                    argv=tuple(str(argument) for argument in command["argv"]),
                    cwd=str(command["cwd"]),
                    returncode=int(command["returncode"]),
                    duration_ms=int(command["duration_ms"]),
                    timed_out=bool(command["timed_out"]),
                    stdout_sha256=str(command["stdout_sha256"]),
                    stderr_sha256=str(command["stderr_sha256"]),
                    classification=str(command["classification"]),
                )
                for command in commands
                if isinstance(command, Mapping)
            ),
        )


class CICommandRunner(Protocol):
    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> CompletedCommand: ...


class RepositoryInspector(Protocol):
    def head_sha(self, worktree: Path) -> str: ...

    def is_clean(self, worktree: Path) -> bool: ...

    def changed_files(
        self, worktree: Path, base_sha: str, head_sha: str
    ) -> tuple[str, ...]: ...


class SubprocessCICommandRunner:
    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> CompletedCommand:
        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=env,
                check=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            return CompletedCommand(
                returncode=124,
                stdout=str(error.stdout or ""),
                stderr=str(error.stderr or ""),
                duration_ms=int((time.monotonic() - started) * 1000),
                timed_out=True,
            )
        except OSError as error:
            return CompletedCommand(
                returncode=127,
                stdout="",
                stderr=type(error).__name__,
                duration_ms=int((time.monotonic() - started) * 1000),
                timed_out=False,
            )
        return CompletedCommand(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=int((time.monotonic() - started) * 1000),
            timed_out=False,
        )


class GitRepositoryInspector:
    @staticmethod
    def _run(worktree: Path, *arguments: str) -> str:
        try:
            completed = subprocess.run(
                ("git", "-C", str(worktree), *arguments),
                check=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CIValidationError("Git worktree state was unavailable") from error
        if completed.returncode != 0:
            raise CIValidationError("Git worktree state was unavailable")
        return completed.stdout

    def head_sha(self, worktree: Path) -> str:
        value = self._run(worktree, "rev-parse", "HEAD").strip()
        if not _SHA.fullmatch(value):
            raise CIValidationError("Git worktree head was invalid")
        return value.lower()

    def is_clean(self, worktree: Path) -> bool:
        return not self._run(worktree, "status", "--porcelain=v1", "--untracked-files=all")

    def changed_files(
        self, worktree: Path, base_sha: str, head_sha: str
    ) -> tuple[str, ...]:
        output = self._run(
            worktree, "diff", "--name-only", "--diff-filter=ACMR", f"{base_sha}..{head_sha}"
        )
        return tuple(line for line in output.splitlines() if line)


class LocalCIRunner:
    def __init__(
        self,
        github: GitHubClient,
        ledger: FeedbackLedger,
        *,
        command_runner: CICommandRunner | None = None,
        inspector: RepositoryInspector | None = None,
        python_argv: tuple[str, ...] = (".venv/bin/python",),
        now: Callable[[], datetime] | None = None,
        supervisor_pid: Callable[[], int] | None = None,
        pid_is_alive: Callable[[int], bool] | None = None,
    ) -> None:
        self._github = github
        self._ledger = ledger
        self._commands = command_runner or SubprocessCICommandRunner()
        self._inspector = inspector or GitRepositoryInspector()
        self._python_argv = python_argv
        self._now = now or (lambda: datetime.now(UTC))
        self._supervisor_pid = supervisor_pid or os.getpid
        self._pid_is_alive = pid_is_alive or _pid_is_alive

    def run(self, identity: CIAuditIdentity, worktree: Path) -> CIAuditReceipt:
        """Run exact-head CI under a durable, PID-backed single-owner lease."""

        resolved = Path(worktree).resolve()
        manifest_path = resolved / "tests/manifests/test_lanes.toml"
        if not resolved.is_dir() or not manifest_path.is_file():
            raise CIValidationError("required CI owner files are missing")
        manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        claimed_at = _aware_now(self._now())
        lease = self._ledger.claim_ci_run(
            identity.repository,
            identity.pr_number,
            identity.base_sha,
            identity.head_sha,
            manifest_digest,
            supervisor_pid=self._supervisor_pid(),
            claimed_at=claimed_at,
            stale_before=claimed_at - _CI_RUN_LEASE,
            pid_is_alive=self._pid_is_alive,
        )
        if lease is None:
            raise CIValidationError("exact-head CI audit is already running")
        try:
            receipt = self._run_claimed(identity, resolved)
        except Exception as error:
            self._ledger.finish_ci_run(
                lease,
                status="failed",
                completed_at=_aware_now(self._now()),
                error=type(error).__name__,
            )
            raise
        self._ledger.finish_ci_run(
            lease,
            status="completed",
            completed_at=receipt.completed_at,
            receipt_id=receipt.receipt_id,
        )
        return receipt

    def _run_claimed(self, identity: CIAuditIdentity, worktree: Path) -> CIAuditReceipt:
        worktree = Path(worktree).resolve()
        if not worktree.is_dir():
            raise CIValidationError("CI worktree does not exist")
        manifest_path = worktree / "tests/manifests/test_lanes.toml"
        scripts = tuple(worktree / relative for relative in _REQUIRED_SCRIPTS)
        if not manifest_path.is_file() or any(not script.is_file() for script in scripts):
            raise CIValidationError("required CI owner files are missing")
        self._ensure_python_environment(worktree)
        manifest_bytes = manifest_path.read_bytes()
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        lanes = _required_lanes(manifest_bytes)

        initial_state = self._github.get_merge_state(identity.repository, identity.pr_number)
        initial_checks = self._github.get_check_state(identity.repository, identity.head_sha)
        _require_identity(identity, initial_state)
        if self._inspector.head_sha(worktree) != identity.head_sha:
            raise CIValidationError("CI worktree head does not match the receipt identity")
        if not self._inspector.is_clean(worktree):
            raise CIValidationError("CI worktree is dirty before execution")
        changed_files = self._inspector.changed_files(
            worktree, identity.base_sha, identity.head_sha
        )

        started_at = _aware_now(self._now())
        command_specs: list[tuple[tuple[str, ...], Path, dict[str, str]]] = [
            (self._python_argv + ("scripts/check_ci_governance.py",), worktree, {}),
            (
                self._python_argv + ("scripts/run_static_lane.py",),
                worktree,
                {"STATIC_BASE_REF": identity.base_sha},
            ),
            (self._python_argv + ("scripts/run_hygiene_lane.py",), worktree, {}),
        ]
        command_specs.extend(
            (
                _lane_argv(self._python_argv, lane, identity),
                worktree,
                {},
            )
            for lane in lanes
        )
        if any(path == "frontend" or path.startswith("frontend/") for path in changed_files):
            frontend = worktree / "frontend"
            if not (frontend / "package.json").is_file() or not (
                frontend / "package-lock.json"
            ).is_file():
                raise CIValidationError("frontend lock inputs are missing")
            command_specs.extend(
                (
                    (argv, frontend, {})
                    for argv in (
                        ("npm", "ci"),
                        ("npm", "run", "lint"),
                        ("npm", "test"),
                        ("npm", "run", "build"),
                    )
                )
            )

        evidence: list[CommandEvidence] = []
        for argv, cwd, additions in command_specs:
            environment = dict(os.environ)
            environment.update(additions)
            result = self._commands.run(
                argv, cwd=cwd, env=environment, timeout=_COMMAND_TIMEOUT_SECONDS
            )
            evidence.append(_command_evidence(argv, cwd, worktree, result))
            if result.returncode != 0 or result.timed_out:
                break

        if self._inspector.head_sha(worktree) != identity.head_sha:
            raise CIValidationError("CI worktree head changed during execution")
        if not self._inspector.is_clean(worktree):
            raise CIValidationError("CI worktree became dirty during execution")
        final_state = self._github.get_merge_state(identity.repository, identity.pr_number)
        final_checks = self._github.get_check_state(identity.repository, identity.head_sha)
        _require_identity(identity, final_state)
        if final_checks != initial_checks:
            raise CIValidationError("GitHub Actions state changed during CI execution")

        completed_at = _aware_now(self._now())
        status = "passed" if len(evidence) == len(command_specs) and all(
            item.returncode == 0 and not item.timed_out for item in evidence
        ) else "failed"
        receipt_id = _receipt_id(
            identity, manifest_digest, status, completed_at, tuple(evidence)
        )
        receipt = CIAuditReceipt(
            receipt_id=receipt_id,
            identity=identity,
            manifest_digest=manifest_digest,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            actions_state=initial_checks,
            commands=tuple(evidence),
        )
        self._ledger.record_ci_receipt(receipt)
        return receipt

    def _ensure_python_environment(self, worktree: Path) -> None:
        executable = Path(self._python_argv[0])
        if executable.is_absolute() or "/" not in str(executable):
            return
        resolved = worktree / executable
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return
        bootstrap = worktree / "scripts/bootstrap_agent_workspace.py"
        if not bootstrap.is_file():
            raise CIValidationError("worktree Python environment is missing")
        result = self._commands.run(
            ("python3", "scripts/bootstrap_agent_workspace.py", "--venv", "link"),
            cwd=worktree,
            env=dict(os.environ),
            timeout=_BOOTSTRAP_TIMEOUT_SECONDS,
        )
        if (
            result.returncode != 0
            or result.timed_out
            or not resolved.is_file()
            or not os.access(resolved, os.X_OK)
        ):
            raise CIValidationError("worktree Python bootstrap failed")


def _pid_is_alive(pid: int) -> bool:
    """Read-only liveness check for a ledger-recorded OS process identity."""

    if pid < 2:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _lane_argv(
    python_argv: tuple[str, ...], lane: str, identity: CIAuditIdentity
) -> tuple[str, ...]:
    if lane != "locked_install_parity":
        return python_argv + ("scripts/run_test_lane.py", "--lane", lane)
    output = (
        Path(tempfile.gettempdir())
        / "hermes-local-ci-receipts"
        / identity.repository.replace("/", "-")
        / str(identity.pr_number)
        / identity.head_sha
        / "locked_install_parity.json"
    )
    return python_argv + (
        "scripts/run_local_ci_audit.py",
        "--job",
        lane,
        "--output",
        str(output),
    )


def _required_lanes(manifest_bytes: bytes) -> tuple[str, ...]:
    try:
        payload = tomllib.loads(manifest_bytes.decode("utf-8"))
        lanes = payload["lanes"]
        if not isinstance(lanes, dict) or not lanes:
            raise TypeError("lanes missing")
        required = tuple(
            name
            for name, settings in lanes.items()
            if isinstance(name, str)
            and isinstance(settings, dict)
            and settings.get("ci_status") == "required"
        )
    except (KeyError, TypeError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise CIValidationError("CI lane manifest was invalid") from error
    if not required:
        raise CIValidationError("CI lane manifest has no required lanes")
    return required


def _require_identity(identity: CIAuditIdentity, state: PullRequestMergeState) -> None:
    if (
        state.repository != identity.repository
        or state.number != identity.pr_number
        or state.base_sha != identity.base_sha
        or state.head_sha != identity.head_sha
        or state.head_repository != identity.repository
        or state.state != "OPEN"
        or state.merged
    ):
        raise CIValidationError("canonical pull request identity does not match the CI request")


def _command_evidence(
    argv: tuple[str, ...], cwd: Path, worktree: Path, result: CompletedCommand
) -> CommandEvidence:
    try:
        relative_cwd = "." if cwd == worktree else str(cwd.relative_to(worktree))
    except ValueError as error:
        raise CIValidationError("CI command escaped its worktree") from error
    classification = "passed"
    if result.timed_out or result.returncode in {126, 127}:
        classification = "environment-blocked"
    elif result.returncode != 0:
        classification = "logic-regression"
    return CommandEvidence(
        argv=argv,
        cwd=relative_cwd,
        returncode=result.returncode,
        duration_ms=max(0, result.duration_ms),
        timed_out=result.timed_out,
        stdout_sha256=hashlib.sha256(result.stdout.encode("utf-8", errors="replace")).hexdigest(),
        stderr_sha256=hashlib.sha256(result.stderr.encode("utf-8", errors="replace")).hexdigest(),
        classification=classification,
    )


def _receipt_id(
    identity: CIAuditIdentity,
    manifest_digest: str,
    status: str,
    completed_at: datetime,
    evidence: tuple[CommandEvidence, ...],
) -> str:
    payload = {
        "identity": [
            identity.repository,
            identity.pr_number,
            identity.base_sha,
            identity.head_sha,
        ],
        "manifest_digest": manifest_digest,
        "status": status,
        "completed_at": completed_at.isoformat(),
        "commands": [command.stdout_sha256 + command.stderr_sha256 for command in evidence],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _aware_now(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise CIValidationError("CI clock must return a timezone-aware datetime")
    return value.astimezone(UTC)
