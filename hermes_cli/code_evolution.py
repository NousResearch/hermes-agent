"""Create bounded, review-gated contracts for Hermes code evolution.

This module is deliberately a thin controller over the existing Kanban,
worktree, goal-loop, and review primitives.  It does not grant an agent direct
write access to a live checkout and it never merges, pushes, or deploys code.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Sequence

from hermes_cli.profiles import normalize_profile_name, profile_exists

CONTRACT_KIND = "hermes-code-evolution"
CONTRACT_SCHEMA_VERSION = 1
CONTRACT_FILENAME = "code-evolution-contract.json"
VERIFIER_FILENAME = "code-evolution-verifier.py"
_VERIFIER_SOURCE_FILENAME = "code_evolution_verifier.py"
SKILL_NAME = "hermes-code-evolution"


class CodeEvolutionError(ValueError):
    """Raised when a campaign cannot be frozen safely."""


@dataclass(frozen=True)
class PreparedContract:
    """Canonical contract bytes plus the externally pinned verifier."""

    payload: dict
    sha256: str
    contract_id: str
    contract_bytes: bytes
    verifier_bytes: bytes


@dataclass(frozen=True)
class LaunchResult:
    """Durable campaign identity returned to CLI and tests."""

    contract_id: str
    contract_sha256: str
    task_id: str | None
    status: str
    created: bool
    dry_run: bool = False


def _lexical_absolute(path: str | Path) -> Path:
    """Return an absolute path without following links or junctions."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    try:
        if callable(checker) and checker():
            return True
        metadata = path.lstat()
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        return bool(attributes & 0x00000400 and attributes & 0x00000010)
    except OSError as exc:
        raise CodeEvolutionError(
            f"cannot inspect workspace junction state: {exc}"
        ) from exc


def _validate_frozen_workspace_path(
    payload: dict,
    task,
    *,
    must_exist: bool,
) -> Path:
    """Validate the exact project worktree path without resolving substitutions."""

    repository = _lexical_absolute(str(payload.get("repository", "")))
    expected = repository / ".worktrees" / task.id
    actual = _lexical_absolute(task.workspace_path) if task.workspace_path else None
    if actual != expected:
        raise CodeEvolutionError(
            "task no longer retains the frozen project-linked worktree"
        )

    for current in (
        repository,
        repository / ".worktrees",
        repository / ".worktrees" / task.id,
    ):
        try:
            current.lstat()
            is_junction = _is_junction(current)
        except FileNotFoundError:
            if current == repository or must_exist:
                raise CodeEvolutionError(
                    "task no longer retains the frozen project-linked worktree"
                ) from None
            return expected
        except OSError as exc:
            raise CodeEvolutionError(
                f"cannot inspect frozen project-linked worktree: {exc}"
            ) from exc
        if current.is_symlink() or is_junction or not current.is_dir():
            raise CodeEvolutionError(
                "task no longer retains the frozen project-linked worktree"
            )
    return expected


def load_frozen_task_contract(conn, task) -> dict | None:
    """Load and verify the controller-owned contract attached to a campaign task."""
    if task is None or SKILL_NAME not in (task.skills or []):
        return None

    from hermes_cli import kanban_db as kb

    attachments = kb.list_attachments(conn, task.id)
    by_name: dict[str, list] = {}
    for attachment in attachments:
        by_name.setdefault(attachment.filename, []).append(attachment)
    contracts = by_name.get(CONTRACT_FILENAME, [])
    verifiers = by_name.get(VERIFIER_FILENAME, [])
    if len(contracts) != 1 or len(verifiers) != 1:
        raise CodeEvolutionError(
            "code-evolution task must retain exactly one frozen contract and verifier"
        )

    try:
        contract_bytes = Path(contracts[0].stored_path).read_bytes()
        verifier_bytes = Path(verifiers[0].stored_path).read_bytes()
        envelope = json.loads(contract_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodeEvolutionError(
            f"cannot read frozen code-evolution evidence: {exc}"
        ) from exc
    if not isinstance(envelope, dict) or not isinstance(envelope.get("contract"), dict):
        raise CodeEvolutionError("frozen code-evolution contract envelope is malformed")
    payload = envelope["contract"]
    digest = hashlib.sha256(_canonical_payload_bytes(payload)).hexdigest()
    if envelope.get("sha256") != digest:
        raise CodeEvolutionError(
            "frozen code-evolution contract SHA-256 does not match"
        )
    if (
        payload.get("kind") != CONTRACT_KIND
        or payload.get("schema_version") != CONTRACT_SCHEMA_VERSION
    ):
        raise CodeEvolutionError(
            "frozen code-evolution contract kind or schema is unsupported"
        )
    for field in (
        "objective",
        "evidence",
        "success_metric",
        "project_id",
        "project_slug",
        "repository",
    ):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise CodeEvolutionError(
                f"frozen code-evolution contract is missing {field.replace('_', ' ')}"
            )
    project_id, project_slug = _resolve_project_binding(
        _lexical_absolute(payload["repository"]),
        payload["project_id"],
    )
    if project_id != payload["project_id"] or project_slug != payload["project_slug"]:
        raise CodeEvolutionError("frozen code-evolution Project binding changed")
    if f"Contract SHA-256: `{digest}`" not in (task.body or ""):
        raise CodeEvolutionError(
            "task body does not retain the frozen contract SHA-256"
        )
    expected_branch = f"evolve/{task.id}-{digest[:12]}"
    if (
        task.project_id != payload["project_id"]
        or task.workspace_kind != "worktree"
        or task.branch_name != expected_branch
    ):
        raise CodeEvolutionError(
            "task no longer retains the frozen project-linked worktree"
        )
    _validate_frozen_workspace_path(payload, task, must_exist=False)
    verifier = payload.get("verifier")
    if not isinstance(verifier, dict) or verifier.get("filename") != VERIFIER_FILENAME:
        raise CodeEvolutionError(
            "frozen code-evolution verifier declaration is malformed"
        )
    if hashlib.sha256(verifier_bytes).hexdigest() != verifier.get("sha256"):
        raise CodeEvolutionError(
            "frozen code-evolution verifier SHA-256 does not match"
        )
    reviewer = payload.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise CodeEvolutionError("frozen code-evolution reviewer is missing")
    return payload


def completion_policy_error(conn, task) -> str | None:
    """Return why a frozen campaign cannot complete from its active run."""
    contract = load_frozen_task_contract(conn, task)
    if contract is None:
        return None
    claimed = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND run_id = ? AND kind = 'claimed' "
        "ORDER BY id DESC LIMIT 1",
        (task.id, task.current_run_id),
    ).fetchone()
    try:
        claimed_payload = (
            json.loads(claimed["payload"])
            if claimed is not None and claimed["payload"]
            else {}
        )
    except (json.JSONDecodeError, TypeError):
        claimed_payload = {}
    if (
        not isinstance(claimed_payload, dict)
        or claimed_payload.get("source_status") != "review"
    ):
        return (
            "code-evolution implementation runs must request review; only a "
            "review run may complete the task"
        )
    run = conn.execute(
        "SELECT profile FROM task_runs WHERE id = ? AND task_id = ?",
        (task.current_run_id, task.id),
    ).fetchone()
    active_profile = (
        normalize_profile_name(str(run["profile"]))
        if run is not None and run["profile"]
        else None
    )
    if active_profile != contract["reviewer"]:
        return (
            "code-evolution approval must come from frozen reviewer "
            f"{contract['reviewer']!r}; got {active_profile!r}"
        )
    return None


def _run_verifier_process(
    argv: list[str],
    *,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Run the verifier inside a bounded, whole-tree process guard."""

    from hermes_cli import code_evolution_verifier as process_guard
    from hermes_cli import code_evolution_process_guard as posix_process_guard

    deadline = time.monotonic() + timeout
    cleanup_reserve = min(2.0, max(0.5, timeout * 0.2))
    execution_timeout = max(0.001, timeout - cleanup_reserve)
    supervisor_cleanup_timeout = max(0.1, cleanup_reserve * 0.6)
    job_handle: int | None = None
    popen_kwargs: dict[str, Any] = {}
    launch_argv = argv
    if os.name == "nt":
        job_handle = process_guard._create_windows_kill_job()
        popen_kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | 0x00000004
        )
    else:
        supervisor_path = Path(__file__).with_name("code_evolution_process_guard.py")
        launch_argv = [
            sys.executable,
            str(supervisor_path),
            "--cleanup-timeout",
            str(supervisor_cleanup_timeout),
            "--",
            *argv,
        ]
        popen_kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(
            launch_argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,
        )
        if job_handle is not None:
            process_guard._assign_and_resume_windows_job(process, job_handle)
    except BaseException:
        if job_handle is not None:
            process_guard._close_windows_job(job_handle)
        raise

    monitor_stop: threading.Event | None = None
    monitor_thread: threading.Thread | None = None
    descendant_processes: dict[tuple[int, float], Any] = {}
    descendant_lock = threading.Lock()
    monitor_errors: list[str] = []
    psutil_module = None

    if os.name != "nt":
        try:
            import psutil as psutil_module
        except ImportError as exc:  # pragma: no cover - psutil is a core dependency
            os.killpg(process.pid, signal.SIGKILL)
            raise OSError(
                "psutil is required for verifier process-tree cleanup"
            ) from exc

        monitor_stop = threading.Event()

        def capture_descendants() -> None:
            try:
                children = psutil_module.Process(process.pid).children(recursive=True)
            except psutil_module.NoSuchProcess:
                return
            except Exception as exc:  # pragma: no cover - host process table failure
                if not monitor_errors:
                    monitor_errors.append(
                        f"could not inspect verifier descendants: {exc}"
                    )
                return
            for child in children:
                try:
                    identity = (child.pid, child.create_time())
                    with descendant_lock:
                        descendant_processes[identity] = child
                except psutil_module.NoSuchProcess:
                    continue

        def monitor_descendants() -> None:
            while monitor_stop is not None and not monitor_stop.wait(0.005):
                capture_descendants()

        capture_descendants()
        monitor_thread = threading.Thread(
            target=monitor_descendants,
            name="code-evolution-verifier-process-monitor",
            daemon=True,
        )
        monitor_thread.start()

    def terminate_tree(*, force: bool) -> str | None:
        if job_handle is not None:
            return process_guard._terminate_gate_process_tree(
                process,
                windows_job=job_handle,
            )

        assert psutil_module is not None
        capture_descendants()
        assert monitor_stop is not None
        errors: list[str] = []
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(
                timeout=max(0.0, min(0.05, deadline - time.monotonic()))
            )
            if monitor_thread.is_alive():
                errors.append(
                    "verifier descendant monitor did not stop before deadline"
                )
        errors.extend(error for error in monitor_errors if error not in errors)
        if force and process.poll() is None:
            if time.monotonic() >= deadline:
                errors.append(
                    "deadline expired before forcing verifier process cleanup"
                )
            else:
                try:
                    supervisor = psutil_module.Process(process.pid)
                    cleanup_error = posix_process_guard._terminate_descendants(
                        supervisor,
                        descendant_processes,
                        deadline=deadline,
                    )
                    if cleanup_error:
                        errors.append(cleanup_error)
                except psutil_module.NoSuchProcess:
                    pass
                except Exception as exc:
                    # pragma: no cover - host process table failure
                    errors.append(
                        f"could not clean verifier supervisor children: {exc}"
                    )
        with descendant_lock:
            descendants = list(descendant_processes.values())
        cleanup_error = posix_process_guard._terminate_tracked_processes(
            descendants,
            deadline=deadline,
            label="outer verifier descendant",
        )
        if cleanup_error:
            errors.append(cleanup_error)
        if process.poll() is None:
            if time.monotonic() >= deadline:
                errors.append("outer verifier survived the cleanup deadline")
            else:
                process.kill()
                try:
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    errors.append("outer verifier termination could not be verified")
        assert psutil_module is not None
        for child in descendants:
            if time.monotonic() >= deadline:
                errors.append(
                    "deadline expired before final verifier cleanup validation"
                )
                break
            try:
                if posix_process_guard._is_running(child):
                    errors.append(
                        f"outer verifier descendant survived cleanup: {child.pid}"
                    )
            except psutil_module.NoSuchProcess:
                continue
            except Exception as exc:  # pragma: no cover - host process-table failure
                errors.append(f"could not verify outer verifier cleanup: {exc}")
        return "; ".join(errors) if errors else None

    try:
        stdout, stderr = process.communicate(timeout=execution_timeout)
    except subprocess.TimeoutExpired as exc:
        cleanup_errors: list[str] = []
        if job_handle is not None:
            cleanup_error = terminate_tree(force=True)
            if cleanup_error:
                cleanup_errors.append(cleanup_error)
        else:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            except OSError as signal_error:
                cleanup_errors.append(
                    f"could not request verifier supervisor cleanup: {signal_error}"
                )
            remaining = deadline - time.monotonic()
            final_validation_reserve = min(0.1, max(0.0, remaining * 0.25))
            grace = min(
                supervisor_cleanup_timeout + 0.1,
                max(0.0, remaining - final_validation_reserve),
            )
            if grace <= 0:
                cleanup_errors.append(
                    "deadline expired before verifier supervisor cleanup could finish"
                )
                cleanup_error = terminate_tree(force=True)
            else:
                try:
                    process.communicate(timeout=grace)
                except subprocess.TimeoutExpired:
                    cleanup_errors.append(
                        "verifier supervisor did not finish cleanup within its reserve"
                    )
                    cleanup_error = terminate_tree(force=True)
                else:
                    cleanup_error = terminate_tree(force=False)
            if cleanup_error:
                cleanup_errors.append(cleanup_error)
        remaining = deadline - time.monotonic()
        if process.poll() is None and remaining > 0:
            try:
                process.communicate(timeout=remaining)
            except subprocess.TimeoutExpired:
                cleanup_errors.append(
                    "outer verifier retained output pipes at deadline"
                )
        if process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                process.kill()
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is None:
                cleanup_errors.append("outer verifier survived the end-to-end deadline")
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
        if cleanup_errors:
            raise OSError("; ".join(cleanup_errors)) from exc
        raise
    except (KeyboardInterrupt, SystemExit) as exc:
        cleanup_error = terminate_tree(force=True)
        if cleanup_error:
            raise OSError(cleanup_error) from exc
        raise
    cleanup_error = terminate_tree(force=False)
    if cleanup_error:
        raise OSError(cleanup_error)
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _strict_verifier_containment_available(*, os_name: str, platform: str) -> bool:
    if os_name == "nt":
        return True
    if os_name != "posix":
        return False
    from hermes_cli import code_evolution_process_guard as process_guard

    return process_guard._strict_containment_supported(platform)


def run_frozen_task_verifier(conn, task, *, board: str | None = None) -> dict:
    """Run the attached verifier against the task's actual candidate worktree."""
    from hermes_cli import kanban_db as kb

    payload = load_frozen_task_contract(conn, task)
    if payload is None:
        raise CodeEvolutionError("task is not a code-evolution campaign")
    attachments = kb.list_attachments(conn, task.id)
    contract_path = next(
        Path(item.stored_path)
        for item in attachments
        if item.filename == CONTRACT_FILENAME
    )
    verifier_path = next(
        Path(item.stored_path)
        for item in attachments
        if item.filename == VERIFIER_FILENAME
    )
    workspace = _validate_frozen_workspace_path(payload, task, must_exist=True)
    digest = hashlib.sha256(_canonical_payload_bytes(payload)).hexdigest()
    gate_timeout = sum(
        int(gate.get("timeout_seconds", 0))
        for gate in payload.get("quality_gates", [])
        if isinstance(gate, dict)
    )
    # Each frozen gate owns its timeout and process-tree cleanup. Do not cap
    # this transition-time verifier call with the task's whole-worker budget:
    # killing the verifier before a gate's own timeout can orphan that gate.
    outer_timeout = max(30, gate_timeout + 30)
    argv = [
        sys.executable,
        str(verifier_path),
        "--contract",
        str(contract_path),
        "--expected-contract-sha256",
        digest,
        "--repo",
        str(workspace),
        "--expected-workspace",
        str(workspace),
        "--expected-branch",
        str(task.branch_name),
        "--run-gates",
    ]
    try:
        result = _run_verifier_process(argv, timeout=outer_timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "passed": False,
            "contract_sha256": digest,
            "mode": "run-gates",
            "changed_paths": [],
            "quality_gates": [],
            "issues": [{"code": "verification_process_failed", "message": str(exc)}],
        }
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        report = None
    if not isinstance(report, dict):
        report = {
            "passed": False,
            "contract_sha256": digest,
            "mode": "run-gates",
            "changed_paths": [],
            "quality_gates": [],
            "issues": [
                {
                    "code": "invalid_verifier_output",
                    "message": "frozen verifier did not emit a JSON object",
                }
            ],
        }
    report["verifier_returncode"] = result.returncode
    if result.stderr:
        report["verifier_stderr"] = result.stderr[-8000:]
    if result.returncode != 0 and report.get("passed") is True:
        report["passed"] = False
        issues = report.setdefault("issues", [])
        if isinstance(issues, list):
            issues.append({
                "code": "verifier_exit_mismatch",
                "message": "verifier reported pass with a non-zero exit status",
            })
    return report


def enforce_frozen_task_verifier(
    conn,
    task,
    *,
    phase: str,
    board: str | None = None,
) -> str | None:
    """Run, retain, and enforce the frozen verifier at a lifecycle boundary."""
    from agent.redact import redact_sensitive_text
    from hermes_cli import kanban_db as kb

    contract = load_frozen_task_contract(conn, task)
    if contract is None:
        return None
    report = run_frozen_task_verifier(conn, task, board=board)
    evidence = dict(report)
    evidence["enforcement_phase"] = phase
    evidence["task_id"] = task.id
    evidence["run_id"] = task.current_run_id
    serialized = (
        json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    data = redact_sensitive_text(serialized, force=True).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    filename = (
        f"code-evolution-{phase}-verification-run-"
        f"{task.current_run_id or 'none'}-{digest[:12]}.json"
    )
    existing = [
        item for item in kb.list_attachments(conn, task.id) if item.filename == filename
    ]
    if existing:
        if len(existing) != 1 or Path(existing[0].stored_path).read_bytes() != data:
            return "code-evolution verification evidence is duplicated or altered"
    else:
        kb.store_attachment_bytes(
            conn,
            task.id,
            filename,
            data,
            content_type="application/json",
            uploaded_by="code-evolution-transition-guard",
            board=board,
        )
    if report.get("passed") is not True:
        issues = report.get("issues")
        first_code = None
        if isinstance(issues, list) and issues and isinstance(issues[0], dict):
            first_code = issues[0].get("code")
        detail = f" ({first_code})" if first_code else ""
        return f"code-evolution frozen verifier did not pass{detail}"
    return None


def _run_git(repository: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CodeEvolutionError(f"git command failed: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown git error").strip()
        raise CodeEvolutionError(detail)
    return result.stdout.strip()


def _normalize_scope_path(raw: str) -> str:
    value = str(raw).strip().replace("\\", "/").rstrip("/")
    if not value:
        raise CodeEvolutionError("allowed paths must not be empty")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive or value.startswith("~"):
        raise CodeEvolutionError(f"allowed path must be repository-relative: {raw!r}")
    if value == "." or any(part in {"", ".", ".."} for part in path.parts):
        raise CodeEvolutionError(f"unsafe allowed path: {raw!r}")
    if path.parts and path.parts[0] == ".git":
        raise CodeEvolutionError(".git cannot be included in the allowed path set")
    return path.as_posix()


def _split_quality_gate_command(
    command: str,
    *,
    windows: bool | None = None,
) -> list[str]:
    """Split a CLI gate command using the target platform's quoting rules."""
    if windows is None:
        windows = os.name == "nt"
    if not windows:
        return shlex.split(command, posix=True)

    # ``shlex`` models POSIX shells and cannot parse quotes that begin in the
    # middle of a Windows argument (for example ``--out="C:\Program Files"``).
    # Implement the CommandLineToArgvW backslash/quote rules directly so the
    # launcher freezes the same argv that Windows will execute.
    argv: list[str] = []
    index = 0
    while index < len(command):
        while index < len(command) and command[index] in " \t":
            index += 1
        if index >= len(command):
            break

        argument: list[str] = []
        in_quotes = False
        while index < len(command):
            char = command[index]
            if char in " \t" and not in_quotes:
                break
            if char == "\\":
                slash_start = index
                while index < len(command) and command[index] == "\\":
                    index += 1
                slash_count = index - slash_start
                if index < len(command) and command[index] == '"':
                    argument.extend("\\" * (slash_count // 2))
                    if slash_count % 2:
                        argument.append('"')
                        index += 1
                    elif (
                        in_quotes
                        and index + 1 < len(command)
                        and command[index + 1] == '"'
                    ):
                        argument.append('"')
                        index += 2
                    else:
                        in_quotes = not in_quotes
                        index += 1
                else:
                    argument.extend("\\" * slash_count)
                continue
            if char == '"':
                if in_quotes and index + 1 < len(command) and command[index + 1] == '"':
                    argument.append('"')
                    index += 2
                else:
                    in_quotes = not in_quotes
                    index += 1
                continue
            argument.append(char)
            index += 1

        argv.append("".join(argument))
        while index < len(command) and command[index] in " \t":
            index += 1
    return argv


def _normalize_quality_gates(
    quality_gates: Iterable[tuple[str | Sequence[str], int]],
) -> list[dict[str, object]]:
    gates: list[dict[str, object]] = []
    for command, timeout_seconds in quality_gates:
        if isinstance(command, str):
            try:
                argv = _split_quality_gate_command(command)
            except ValueError as exc:
                raise CodeEvolutionError(
                    f"invalid quality-gate command: {exc}"
                ) from exc
        else:
            argv = [str(arg) for arg in command]
        if not argv or not argv[0].strip() or any(not arg for arg in argv):
            raise CodeEvolutionError("quality-gate commands must not be empty")
        try:
            timeout = int(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise CodeEvolutionError("quality-gate timeout must be an integer") from exc
        if timeout <= 0:
            raise CodeEvolutionError("quality-gate timeout must be positive")
        gates.append({"argv": argv, "timeout_seconds": timeout})
    if not gates:
        raise CodeEvolutionError("at least one quality gate is required")
    return gates


def _scope_rule(repository: Path, scope: str) -> dict[str, str]:
    """Freeze whether a scope is an exact file or a directory subtree."""

    target = repository
    for part in PurePosixPath(scope).parts:
        target = target / part
        if target.is_symlink():
            raise CodeEvolutionError(
                f"allowed path traverses a symlink and cannot be bounded: {scope!r}"
            )
    if target.exists() and not (target.is_file() or target.is_dir()):
        raise CodeEvolutionError(f"allowed path has unsupported file type: {scope!r}")
    return {"path": scope, "kind": "tree" if target.is_dir() else "file"}


def _resolve_project_binding(repository: Path, project: str) -> tuple[str, str]:
    """Resolve one active Project whose primary repo is the frozen repository."""
    from hermes_cli import projects_db

    project_ref = str(project or "").strip()
    if not project_ref:
        raise CodeEvolutionError("project is required")
    try:
        with projects_db.connect_closing() as conn:
            project_obj = projects_db.get_project(conn, project_ref)
    except Exception as exc:
        raise CodeEvolutionError(
            f"cannot resolve project {project_ref!r}: {exc}"
        ) from exc
    if project_obj is None or project_obj.archived:
        raise CodeEvolutionError(f"unknown or archived project: {project_ref}")
    if not project_obj.primary_path:
        raise CodeEvolutionError(
            f"project {project_obj.slug!r} has no primary repository"
        )
    project_repo = Path(project_obj.primary_path).expanduser().resolve(strict=False)
    if project_repo != repository:
        raise CodeEvolutionError(
            f"project {project_obj.slug!r} is anchored to {project_repo}, not {repository}"
        )
    return project_obj.id, project_obj.slug


def _canonical_payload_bytes(payload: dict) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def prepare_contract(
    *,
    repository: str | Path,
    project: str,
    objective: str,
    evidence: str,
    success_metric: str,
    allowed_paths: Sequence[str],
    quality_gates: Sequence[tuple[str | Sequence[str], int]],
    assignee: str,
    reviewer: str,
    goal_max_turns: int = 20,
    max_runtime_seconds: int = 7200,
) -> PreparedContract:
    """Freeze one evidence-backed, path-bounded code-evolution contract.

    The repository must be a clean Git checkout.  The exact commit, tree, and
    common Git directory are captured so a worker can fail closed if dispatch
    happens from a different revision or repository.
    """

    repo_input = Path(repository).expanduser()
    repo_root = Path(_run_git(repo_input, "rev-parse", "--show-toplevel")).resolve()
    git_dir = Path(
        _run_git(repo_root, "rev-parse", "--path-format=absolute", "--git-dir")
    ).resolve(strict=False)
    git_common_dir = Path(
        _run_git(
            repo_root,
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        )
    ).resolve(strict=False)
    if git_dir != git_common_dir:
        raise CodeEvolutionError(
            "repository must be the primary Git checkout, not an existing linked "
            "worktree; pass the primary checkout with --repo"
        )
    if _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise CodeEvolutionError(
            "repository must be clean before freezing a code-evolution contract"
        )
    project_id, project_slug = _resolve_project_binding(repo_root, project)

    objective = str(objective).strip()
    evidence = str(evidence).strip()
    success_metric = str(success_metric).strip()
    if not objective:
        raise CodeEvolutionError("objective is required")
    if not evidence:
        raise CodeEvolutionError("evidence is required")
    if not success_metric:
        raise CodeEvolutionError("success metric is required")

    assignee_name = normalize_profile_name(assignee)
    reviewer_name = normalize_profile_name(reviewer)
    if assignee_name == reviewer_name:
        raise CodeEvolutionError("assignee and reviewer must be different profiles")
    for role, profile in (("assignee", assignee_name), ("reviewer", reviewer_name)):
        if not profile_exists(profile):
            raise CodeEvolutionError(f"unknown {role} profile: {profile}")

    scopes = sorted({_normalize_scope_path(path) for path in allowed_paths})
    if not scopes:
        raise CodeEvolutionError("at least one allowed path is required")
    scope_rules = [_scope_rule(repo_root, scope) for scope in scopes]
    gates = _normalize_quality_gates(quality_gates)

    try:
        turns = int(goal_max_turns)
        runtime = int(max_runtime_seconds)
    except (TypeError, ValueError) as exc:
        raise CodeEvolutionError("budgets must be integers") from exc
    if turns <= 0 or runtime <= 0:
        raise CodeEvolutionError("budgets must be positive")

    verifier_bytes = Path(__file__).with_name(_VERIFIER_SOURCE_FILENAME).read_bytes()
    verifier_sha = hashlib.sha256(verifier_bytes).hexdigest()
    payload = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "kind": CONTRACT_KIND,
        "repository": str(repo_root),
        "project_id": project_id,
        "project_slug": project_slug,
        "git_common_dir": str(git_common_dir),
        "base_commit": _run_git(repo_root, "rev-parse", "HEAD"),
        "base_tree": _run_git(repo_root, "rev-parse", "HEAD^{tree}"),
        "objective": objective,
        "evidence": evidence,
        "success_metric": success_metric,
        "allowed_paths": scopes,
        "allowed_path_rules": scope_rules,
        "quality_gates": gates,
        "assignee": assignee_name,
        "reviewer": reviewer_name,
        "budgets": {
            "goal_max_turns": turns,
            "max_runtime_seconds": runtime,
            "max_retries": 1,
        },
        "verifier": {
            "filename": VERIFIER_FILENAME,
            "sha256": verifier_sha,
        },
        "terminal_state": "request_review",
        "forbidden_actions": [
            "commit",
            "push",
            "merge",
            "rebase",
            "reset_history",
            "deploy",
            "restart_live_hermes",
            "modify_contract_or_verifier",
            "weaken_or_skip_quality_gates",
        ],
    }
    digest = hashlib.sha256(_canonical_payload_bytes(payload)).hexdigest()
    envelope = {"contract": payload, "sha256": digest}
    contract_bytes = (
        json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    return PreparedContract(
        payload=payload,
        sha256=digest,
        contract_id=f"ce_{digest[:16]}",
        contract_bytes=contract_bytes,
        verifier_bytes=verifier_bytes,
    )


def _validated_prepared_payload(prepared: PreparedContract) -> dict:
    """Reload and authenticate all mutable fields before creating a task."""

    try:
        envelope = json.loads(prepared.contract_bytes)
    except (TypeError, json.JSONDecodeError) as exc:
        raise CodeEvolutionError(f"invalid frozen contract bytes: {exc}") from exc
    if not isinstance(envelope, dict) or not isinstance(envelope.get("contract"), dict):
        raise CodeEvolutionError("invalid frozen contract envelope")
    payload = envelope["contract"]
    digest = hashlib.sha256(_canonical_payload_bytes(payload)).hexdigest()
    if envelope.get("sha256") != digest or prepared.sha256 != digest:
        raise CodeEvolutionError("frozen contract SHA-256 does not match its bytes")
    if prepared.contract_id != f"ce_{digest[:16]}":
        raise CodeEvolutionError("frozen contract id does not match its SHA-256")
    if prepared.payload != payload:
        raise CodeEvolutionError("prepared payload does not match frozen bytes")
    project_id, project_slug = _resolve_project_binding(
        Path(str(payload.get("repository", ""))).resolve(strict=False),
        str(payload.get("project_id", "")),
    )
    if project_id != payload.get("project_id") or project_slug != payload.get(
        "project_slug"
    ):
        raise CodeEvolutionError("frozen project identity no longer resolves")
    verifier = payload.get("verifier")
    verifier_sha = hashlib.sha256(prepared.verifier_bytes).hexdigest()
    if (
        payload.get("kind") != CONTRACT_KIND
        or payload.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or not isinstance(verifier, dict)
        or verifier.get("filename") != VERIFIER_FILENAME
        or verifier.get("sha256") != verifier_sha
    ):
        raise CodeEvolutionError("frozen verifier metadata does not match its bytes")
    return payload


def _task_body(prepared: PreparedContract, payload: dict) -> str:
    gates = "\n".join(
        f"- `{shlex.join(gate['argv'])}` (timeout {gate['timeout_seconds']}s)"
        for gate in payload["quality_gates"]
    )
    scopes = "\n".join(f"- `{path}`" for path in payload["allowed_paths"])
    return f"""# Frozen code-evolution contract

Contract id: `{prepared.contract_id}`
Contract SHA-256: `{prepared.sha256}`
Exact base commit: `{payload["base_commit"]}`
Exact base tree: `{payload["base_tree"]}`
Project: `{payload["project_slug"]}` (`{payload["project_id"]}`)
Independent reviewer: `{payload["reviewer"]}`

## Objective
{payload["objective"]}

## Reproduced evidence
{payload["evidence"]}

## Measurable success metric
{payload["success_metric"]}

## Allowed paths
{scopes}

## Immutable quality gates
{gates}

## Required lifecycle
1. Read both attached files: `{CONTRACT_FILENAME}` and `{VERIFIER_FILENAME}`.
2. Before editing, run the attached verifier with `--expected-contract-sha256 {prepared.sha256}` and `--preflight` from the assigned worktree. If it fails, call `kanban_block` and stop.
3. Reproduce the stated evidence, add or strengthen a regression test, then make the smallest class-level fix inside the allowed paths.
4. Run the attached verifier with `--run-gates` after the change. A non-zero exit is a failed campaign, not permission to weaken a gate.
5. Self-review the exact diff. Then call `kanban_request_review` with reviewer `{payload["reviewer"]}` and include the verifier result in metadata.

Do not call `kanban_complete` from the implementation phase. Do not commit, push, merge, deploy, or restart a live Hermes process. Do not modify the contract, attached verifier, evaluator, quality-gate commands, Git history, credentials, or files outside the allowed path set. If the exact base identity, attachment hashes, or evidence cannot be verified, fail closed with `kanban_block`.
"""


def _existing_campaign_id(conn, idempotency_key: str) -> str | None:
    row = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key = ? "
        "AND status != 'archived' ORDER BY created_at DESC LIMIT 1",
        (idempotency_key,),
    ).fetchone()
    return str(row["id"]) if row else None


def _attachments_match(conn, task_id: str, prepared: PreparedContract) -> bool:
    from hermes_cli import kanban_db as kb

    attachments = kb.list_attachments(conn, task_id)
    contracts = [
        attachment
        for attachment in attachments
        if attachment.filename == CONTRACT_FILENAME
    ]
    verifiers = [
        attachment
        for attachment in attachments
        if attachment.filename == VERIFIER_FILENAME
    ]
    if len(contracts) != 1 or len(verifiers) != 1:
        return False
    try:
        return (
            Path(contracts[0].stored_path).read_bytes() == prepared.contract_bytes
            and Path(verifiers[0].stored_path).read_bytes() == prepared.verifier_bytes
        )
    except OSError:
        return False


def launch_campaign(
    prepared: PreparedContract,
    *,
    board: str | None = None,
    priority: int = 0,
    created_by: str = "user",
    dry_run: bool = False,
) -> LaunchResult:
    """Create one blocked-while-initializing Kanban task, then release it.

    The contract and standalone verifier are stored as task attachments before
    the task becomes dispatchable.  Repeating the same frozen contract returns
    the existing task without adding comments or attachments.
    """

    payload = _validated_prepared_payload(prepared)
    if dry_run:
        return LaunchResult(
            contract_id=prepared.contract_id,
            contract_sha256=prepared.sha256,
            task_id=None,
            status="dry-run",
            created=False,
            dry_run=True,
        )
    if not _strict_verifier_containment_available(
        os_name=os.name,
        platform=sys.platform,
    ):
        raise CodeEvolutionError(
            "code-evolution campaigns require Windows Job Objects or the Linux "
            "child-subreaper process guard; this platform cannot guarantee "
            "verifier descendant cleanup"
        )

    from hermes_cli import kanban_db as kb

    kb.init_db(board=board)
    idempotency_key = f"code-evolution:{prepared.sha256}"
    task_id: str | None = None
    created = False
    with kb.connect_closing(board=board) as conn:
        existing_id = _existing_campaign_id(conn, idempotency_key)
        if existing_id is not None:
            task = kb.get_task(conn, existing_id)
            if task is not None and _attachments_match(conn, existing_id, prepared):
                return LaunchResult(
                    contract_id=prepared.contract_id,
                    contract_sha256=prepared.sha256,
                    task_id=existing_id,
                    status=task.status,
                    created=False,
                )
            raise CodeEvolutionError(
                f"existing campaign {existing_id} has missing or altered frozen evidence"
            )

        title_objective = str(payload["objective"]).splitlines()[0][:120]
        try:
            task_id = kb.create_task(
                conn,
                title=f"Improve code: {title_objective}",
                body=_task_body(prepared, payload),
                assignee=str(payload["assignee"]),
                created_by=created_by,
                workspace_kind="worktree",
                workspace_path=None,
                project_id=str(payload["project_id"]),
                priority=int(priority),
                idempotency_key=idempotency_key,
                max_runtime_seconds=int(payload["budgets"]["max_runtime_seconds"]),
                skills=(SKILL_NAME,),
                max_retries=int(payload["budgets"]["max_retries"]),
                goal_mode=True,
                goal_max_turns=int(payload["budgets"]["goal_max_turns"]),
                initial_status="blocked",
                board=board,
            )
            created = True
            kb.set_branch_name(
                conn,
                task_id,
                f"evolve/{task_id}-{prepared.sha256[:12]}",
            )
            initialized_task = kb.get_task(conn, task_id)
            expected_workspace = str(
                Path(str(payload["repository"])) / ".worktrees" / str(task_id)
            )
            if (
                initialized_task is None
                or initialized_task.project_id != payload["project_id"]
                or initialized_task.workspace_kind != "worktree"
                or initialized_task.workspace_path != expected_workspace
                or initialized_task.branch_name
                != f"evolve/{task_id}-{prepared.sha256[:12]}"
            ):
                raise CodeEvolutionError(
                    f"campaign {task_id} did not retain its frozen Project worktree"
                )
            kb.store_attachment_bytes(
                conn,
                task_id,
                CONTRACT_FILENAME,
                prepared.contract_bytes,
                content_type="application/json",
                uploaded_by="code-evolution-controller",
                board=board,
            )
            kb.store_attachment_bytes(
                conn,
                task_id,
                VERIFIER_FILENAME,
                prepared.verifier_bytes,
                content_type="text/x-python",
                uploaded_by="code-evolution-controller",
                board=board,
            )
            kb.add_comment(
                conn,
                task_id,
                "code-evolution-controller",
                "CONTRACT FROZEN — "
                f"sha256={prepared.sha256}; reviewer={payload['reviewer']}; "
                "the attached contract and verifier are the evaluation authority.",
            )
            if not kb.unblock_task(conn, task_id):
                raise CodeEvolutionError(
                    f"campaign {task_id} could not transition from blocked to ready"
                )
        except Exception:
            if created and task_id:
                try:
                    kb.delete_task(conn, task_id)
                finally:
                    shutil.rmtree(
                        kb.task_attachments_dir(task_id, board=board),
                        ignore_errors=True,
                    )
            raise

        task = kb.get_task(conn, task_id)
        if task is None:
            raise CodeEvolutionError(f"campaign {task_id} disappeared after creation")
        return LaunchResult(
            contract_id=prepared.contract_id,
            contract_sha256=prepared.sha256,
            task_id=task_id,
            status=task.status,
            created=True,
        )
