"""Deterministic exact-head local-CI execution and bounded evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Mapping, Protocol

from .github_client import CheckState, GitHubClient, PullRequestMergeState
from .ledger import CIRunLease, FeedbackLedger


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

    def __init__(
        self,
        message: str,
        *,
        command_evidence: tuple[CommandEvidence, ...] = (),
    ) -> None:
        super().__init__(message)
        self.command_evidence = command_evidence


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
    failure_reason: str | None = None

    def validate(self) -> None:
        """Validate the receipt's self-authenticating identity and evidence."""

        if not re.fullmatch(r"[0-9a-f]{64}", self.receipt_id, re.IGNORECASE):
            raise ValueError("CI receipt payload has invalid receipt_id")
        if not re.fullmatch(r"[0-9a-f]{64}", self.manifest_digest, re.IGNORECASE):
            raise ValueError("CI receipt payload has invalid manifest digest")
        if self.completed_at < self.started_at:
            raise ValueError("CI receipt payload has inverted timestamps")
        for command in self.commands:
            if not re.fullmatch(r"[0-9a-f]{64}", command.stdout_sha256, re.IGNORECASE):
                raise ValueError("CI receipt payload has invalid stdout digest")
            if not re.fullmatch(r"[0-9a-f]{64}", command.stderr_sha256, re.IGNORECASE):
                raise ValueError("CI receipt payload has invalid stderr digest")
        if self.status == "passed":
            if not self.commands:
                raise ValueError("passed CI receipt has no command evidence")
            if any(
                command.returncode != 0
                or command.timed_out
                or command.classification != "passed"
                for command in self.commands
            ):
                raise ValueError("passed CI receipt has incomplete command evidence")
        if self.status == "passed":
            expected_id = _receipt_id(
                self.identity,
                self.manifest_digest,
                self.status,
                self.completed_at,
                self.commands,
            )
            if self.receipt_id.casefold() != expected_id:
                raise ValueError("CI receipt payload receipt_id does not match its evidence")

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
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> CIAuditReceipt:
        if not isinstance(payload, Mapping):
            raise ValueError("CI receipt payload must be an object")
        identity = payload.get("identity")
        actions = payload.get("actions_state")
        commands = payload.get("commands")
        if not isinstance(identity, Mapping) or not isinstance(actions, Mapping):
            raise ValueError("CI receipt payload has invalid identity or action state")
        if not isinstance(commands, list):
            raise ValueError("CI receipt payload has invalid commands")
        receipt_id = _required_text(payload.get("receipt_id"), "receipt_id", 128)
        manifest_digest = _required_text(
            payload.get("manifest_digest"), "manifest_digest", 64
        )
        status = _required_text(payload.get("status"), "status", 16)
        if status not in {"passed", "failed"}:
            raise ValueError("CI receipt payload has invalid status")
        parsed_commands: list[CommandEvidence] = []
        for command in commands:
            if not isinstance(command, Mapping):
                raise ValueError("CI receipt payload has an invalid command")
            argv = command.get("argv")
            if (
                not isinstance(argv, list)
                or not argv
                or any(not isinstance(argument, str) or not argument for argument in argv)
            ):
                raise ValueError("CI receipt payload has invalid command argv")
            classification = _required_text(
                command.get("classification"), "command classification", 32
            )
            if classification not in {"passed", "logic-regression", "environment-blocked"}:
                raise ValueError("CI receipt payload has invalid command classification")
            parsed_commands.append(
                CommandEvidence(
                    argv=tuple(argv),
                    cwd=_required_text(command.get("cwd"), "command cwd", 4096),
                    returncode=_required_int(command.get("returncode"), "command returncode"),
                    duration_ms=_required_nonnegative_int(
                        command.get("duration_ms"), "command duration"
                    ),
                    timed_out=_required_bool(command.get("timed_out"), "command timeout"),
                    stdout_sha256=_required_text(
                        command.get("stdout_sha256"), "stdout digest", 64
                    ),
                    stderr_sha256=_required_text(
                        command.get("stderr_sha256"), "stderr digest", 64
                    ),
                    classification=classification,
                )
            )
        failure_reason = payload.get("failure_reason")
        if failure_reason is not None:
            failure_reason = _required_text(failure_reason, "failure reason", 1000)
        receipt = cls(
            receipt_id=receipt_id,
            identity=CIAuditIdentity(
                _required_text(identity.get("repository"), "repository", 255),
                _required_int(identity.get("pr_number"), "pr number"),
                _required_text(identity.get("base_sha"), "base SHA", 64),
                _required_text(identity.get("head_sha"), "head SHA", 64),
            ),
            manifest_digest=manifest_digest,
            status=status,
            started_at=_required_timestamp(payload.get("started_at"), "started_at"),
            completed_at=_required_timestamp(payload.get("completed_at"), "completed_at"),
            actions_state=CheckState(
                actions_enabled=_required_bool(actions.get("actions_enabled"), "actions enabled"),
                all_green=_required_bool(actions.get("all_green"), "actions green"),
                check_count=_required_nonnegative_int(
                    actions.get("check_count"), "check count"
                ),
                billing_blocked=_required_bool(
                    actions.get("billing_blocked", False), "billing blocked"
                ),
            ),
            commands=tuple(parsed_commands),
            failure_reason=failure_reason,
        )
        receipt.validate()
        return receipt


def _required_text(value: object, field: str, max_length: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise ValueError(f"CI receipt payload has invalid {field}")
    return value


def _required_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"CI receipt payload has invalid {field}")
    return value


def _required_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"CI receipt payload has invalid {field}")
    return value


def _required_nonnegative_int(value: object, field: str) -> int:
    value = _required_int(value, field)
    if value < 0:
        raise ValueError(f"CI receipt payload has invalid {field}")
    return value


def _required_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"CI receipt payload has invalid {field}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"CI receipt payload has invalid {field}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"CI receipt payload has non-aware {field}")
    return parsed.astimezone(UTC)


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
            completed_at = _aware_now(self._now())
            receipt = _failed_receipt(
                identity,
                manifest_digest,
                claimed_at,
                completed_at,
                error,
                commands=getattr(error, "command_evidence", ()),
            )
            receipt = self._finalize_ci_run_or_recover(
                lease,
                receipt,
                status="completed",
                completed_at=completed_at,
                error=type(error).__name__,
            )
            return receipt
        receipt = self._finalize_ci_run_or_recover(
            lease,
            receipt,
            status="completed",
            completed_at=receipt.completed_at,
        )
        return receipt

    def _finalize_ci_run_or_recover(
        self,
        lease: CIRunLease,
        receipt: CIAuditReceipt,
        *,
        status: str,
        completed_at: datetime,
        error: str | None = None,
    ) -> CIAuditReceipt:
        try:
            self._ledger.finalize_ci_run(
                lease,
                receipt,
                status=status,
                completed_at=completed_at,
                error=error,
            )
        except Exception:
            try:
                persisted = self._ledger.ci_receipt_by_id(
                    receipt.identity.repository,
                    receipt.identity.pr_number,
                    receipt.receipt_id,
                )
                lifecycle = self._ledger.latest_ci_run(
                    receipt.identity.repository,
                    receipt.identity.pr_number,
                    receipt.identity.head_sha,
                )
            except Exception:
                raise
            if (
                isinstance(persisted, CIAuditReceipt)
                and persisted == receipt
                and lifecycle is not None
                and lifecycle.get("status") == "completed"
                and lifecycle.get("receipt_id") == receipt.receipt_id
            ):
                return persisted
            raise
        return receipt

    def _run_claimed(self, identity: CIAuditIdentity, worktree: Path) -> CIAuditReceipt:
        worktree = Path(worktree).resolve()
        if not worktree.is_dir():
            raise CIValidationError("CI worktree does not exist")
        manifest_path = worktree / "tests/manifests/test_lanes.toml"
        scripts = tuple(worktree / relative for relative in _REQUIRED_SCRIPTS)
        if not manifest_path.is_file() or any(not script.is_file() for script in scripts):
            raise CIValidationError("required CI owner files are missing")
        bootstrap_evidence = self._ensure_python_environment(worktree)
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
        if bootstrap_evidence is not None:
            evidence.append(bootstrap_evidence)
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
            raise CIValidationError(
                "CI worktree head changed during execution",
                command_evidence=tuple(evidence),
            )
        if not self._inspector.is_clean(worktree):
            raise CIValidationError(
                "CI worktree became dirty during execution",
                command_evidence=tuple(evidence),
            )
        final_state = self._github.get_merge_state(identity.repository, identity.pr_number)
        final_checks = self._github.get_check_state(identity.repository, identity.head_sha)
        _require_identity(identity, final_state)
        if final_checks != initial_checks:
            raise CIValidationError(
                "GitHub Actions state changed during CI execution",
                command_evidence=tuple(evidence),
            )

        completed_at = _aware_now(self._now())
        expected_command_count = len(command_specs) + (1 if bootstrap_evidence else 0)
        status = "passed" if len(evidence) == expected_command_count and all(
            item.returncode == 0 and not item.timed_out for item in evidence
        ) else "failed"
        failed_commands = tuple(
            item for item in evidence if item.returncode != 0 or item.timed_out
        )
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
            failure_reason=(
                None
                if not failed_commands
                else "failed command: "
                + shlex.join(failed_commands[0].argv)
                + f" rc={failed_commands[0].returncode}"
            ),
        )
        return receipt

    def _ensure_python_environment(self, worktree: Path) -> CommandEvidence | None:
        executable = Path(self._python_argv[0])
        if executable.is_absolute() or "/" not in str(executable):
            return None
        resolved = worktree / executable
        if resolved.is_file() and os.access(resolved, os.X_OK):
            expected_path = worktree / ".python-version"
            if expected_path.is_file():
                expected = expected_path.read_text(encoding="utf-8").strip().splitlines()[0]
                if not re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", expected):
                    raise CIValidationError("worktree Python version pin is invalid")
                probe = (
                    str(resolved),
                    "-c",
                    "import sys; print('.'.join(map(str, sys.version_info[:3])))",
                )
                result = self._commands.run(
                    probe,
                    cwd=worktree,
                    env=dict(os.environ),
                    timeout=30,
                )
                actual = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
                matches = actual == expected or (
                    expected.count(".") == 1 and actual.startswith(expected + ".")
                )
                if result.returncode != 0 or result.timed_out or not matches:
                    evidence = _command_evidence(probe, worktree, worktree, result)
                    evidence = CommandEvidence(
                        argv=evidence.argv,
                        cwd=evidence.cwd,
                        returncode=evidence.returncode,
                        duration_ms=evidence.duration_ms,
                        timed_out=evidence.timed_out,
                        stdout_sha256=evidence.stdout_sha256,
                        stderr_sha256=evidence.stderr_sha256,
                        classification="environment-blocked",
                    )
                    raise CIValidationError(
                        f"Python interpreter mismatch: expected {expected}, got {actual or 'unavailable'}",
                        command_evidence=(evidence,),
                    )
            return None
        bootstrap = worktree / "scripts/bootstrap_agent_workspace.py"
        if not bootstrap.is_file():
            raise CIValidationError("worktree Python environment is missing")
        argv = ("python3", "scripts/bootstrap_agent_workspace.py", "--venv", "link")
        result = self._commands.run(
            argv,
            cwd=worktree,
            env=dict(os.environ),
            timeout=_BOOTSTRAP_TIMEOUT_SECONDS,
        )
        evidence = _command_evidence(argv, worktree, worktree, result)
        if (
            result.returncode != 0
            or result.timed_out
            or not resolved.is_file()
            or not os.access(resolved, os.X_OK)
        ):
            raise CIValidationError(
                "worktree Python bootstrap failed", command_evidence=(evidence,)
            )
        return evidence


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


def _failed_receipt(
    identity: CIAuditIdentity,
    manifest_digest: str,
    started_at: datetime,
    completed_at: datetime,
    error: Exception,
    *,
    commands: tuple[CommandEvidence, ...] = (),
) -> CIAuditReceipt:
    """Persist typed failure evidence when validation aborts before lane output."""

    reason = f"{type(error).__name__}: {error}"[:1000]
    receipt_id = _receipt_id(identity, manifest_digest, "failed", completed_at, commands)
    return CIAuditReceipt(
        receipt_id=receipt_id,
        identity=identity,
        manifest_digest=manifest_digest,
        status="failed",
        started_at=started_at,
        completed_at=completed_at,
        actions_state=CheckState(
            actions_enabled=False,
            all_green=False,
            check_count=0,
            billing_blocked=False,
        ),
        commands=commands,
        failure_reason=reason,
    )


def _aware_now(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise CIValidationError("CI clock must return a timezone-aware datetime")
    return value.astimezone(UTC)
