#!/usr/bin/env python3
"""Standalone, frozen verifier for a Hermes code-evolution candidate.

The launcher copies these exact bytes outside the candidate worktree and pins
their SHA-256 in the contract.  This file intentionally imports only the Python
standard library: candidate code cannot replace its evaluator by shadowing a
project import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence, cast

CONTRACT_KIND = "hermes-code-evolution"
CONTRACT_SCHEMA_VERSION = 1
_OUTPUT_LIMIT = 8000


class VerificationFailure(ValueError):
    """A malformed or unverifiable frozen contract."""


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _git(
    repo: Path,
    args: Sequence[str],
    *,
    timeout: int = 30,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VerificationFailure(f"git {' '.join(args)} failed: {exc}") from exc
    if check and result.returncode != 0:
        detail = (
            (result.stderr or result.stdout or b"unknown git error")
            .decode("utf-8", errors="replace")
            .strip()
        )
        raise VerificationFailure(f"git {' '.join(args)} failed: {detail}")
    return result


def _git_text(repo: Path, *args: str) -> str:
    return _git(repo, args).stdout.decode("utf-8", errors="replace").strip()


def _load_contract(
    path: Path,
    verifier_path: Path,
    expected_digest: str,
) -> tuple[dict[str, Any], str]:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationFailure(f"cannot read contract: {exc}") from exc
    if not isinstance(envelope, dict):
        raise VerificationFailure("contract envelope must be an object")
    payload = envelope.get("contract")
    envelope_digest = envelope.get("sha256")
    if not isinstance(payload, dict) or not isinstance(envelope_digest, str):
        raise VerificationFailure("contract envelope is missing contract or sha256")
    if envelope_digest != expected_digest:
        raise VerificationFailure(
            "contract does not match the expected contract SHA-256 trust anchor"
        )
    actual_digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if actual_digest != envelope_digest:
        raise VerificationFailure("contract SHA-256 does not match its payload")
    if payload.get("kind") != CONTRACT_KIND:
        raise VerificationFailure("unsupported contract kind")
    if payload.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise VerificationFailure("unsupported contract schema version")
    for field in (
        "objective",
        "evidence",
        "success_metric",
        "project_id",
        "project_slug",
    ):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise VerificationFailure(f"contract is missing {field.replace('_', ' ')}")

    verifier = payload.get("verifier")
    if not isinstance(verifier, dict) or not isinstance(verifier.get("sha256"), str):
        raise VerificationFailure("contract is missing the verifier SHA-256")
    try:
        own_digest = hashlib.sha256(verifier_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise VerificationFailure(f"cannot read verifier: {exc}") from exc
    if own_digest != verifier["sha256"]:
        raise VerificationFailure("verifier SHA-256 does not match the contract")
    return payload, envelope_digest


def _changed_paths(repo: Path) -> list[str]:
    tracked = _git(
        repo,
        ("diff", "HEAD", "--name-only", "--no-renames", "-z", "--"),
    ).stdout
    untracked = _git(
        repo,
        ("ls-files", "--others", "-z", "--"),
    ).stdout
    paths: set[str] = set()
    for raw in tracked.split(b"\0") + untracked.split(b"\0"):
        if raw:
            path = raw.decode("utf-8", errors="surrogateescape")
            paths.add(path.replace("\\", "/") if os.name == "nt" else path)
    paths.update(_filesystem_untracked_paths(repo))
    return sorted(paths)


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    try:
        if callable(checker) and checker():
            return True
        metadata = path.lstat()
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        return bool(attributes & 0x00000400 and attributes & 0x00000010)
    except OSError as exc:
        raise VerificationFailure(
            f"could not inspect junction state for {path}: {exc}"
        ) from exc


def _unsafe_directory_chain(path: Path) -> list[str]:
    """Return lexical directory components that are missing or redirecting."""

    lexical = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    unsafe: list[str] = []
    for current in (*reversed(lexical.parents), lexical):
        try:
            metadata = current.lstat()
            is_junction = _is_junction(current)
        except FileNotFoundError:
            unsafe.append(str(current))
            break
        except OSError as exc:
            raise VerificationFailure(
                f"could not inspect repository path component {current}: {exc}"
            ) from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or is_junction
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            unsafe.append(str(current))
            break
    return unsafe


def _filesystem_untracked_paths(repo: Path) -> set[str]:
    """Inventory leaf entries Git may omit, including FIFOs and sockets."""

    tracked_raw = _git(repo, ("ls-files", "-z", "--")).stdout
    tracked = {
        (
            raw.decode("utf-8", errors="surrogateescape").replace("\\", "/")
            if os.name == "nt"
            else raw.decode("utf-8", errors="surrogateescape")
        )
        for raw in tracked_raw.split(b"\0")
        if raw
    }
    discovered: set[str] = set()
    pending = [repo]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise VerificationFailure(
                f"could not inventory candidate directory {directory}: {exc}"
            ) from exc
        for entry in entries:
            if directory == repo and entry.name == ".git":
                continue
            path = Path(entry.path)
            try:
                is_symlink = entry.is_symlink()
                is_junction = _is_junction(path)
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError as exc:
                raise VerificationFailure(
                    f"could not inspect candidate entry {path}: {exc}"
                ) from exc
            relative = os.path.relpath(path, repo)
            if os.name == "nt":
                relative = relative.replace("\\", "/")
            if is_directory and not is_symlink and not is_junction:
                pending.append(path)
            elif relative not in tracked:
                discovered.add(relative)
    return discovered


def _unsafe_index_flags(repo: Path) -> list[str]:
    """Return tracked entries hidden by index or filesystem-monitor flags."""

    unsafe: set[str] = set()
    for option, reason in (("-v", "assume-unchanged"), ("-f", "fsmonitor-valid")):
        output = _git(repo, ("ls-files", option, "-z", "--")).stdout
        for raw in output.split(b"\0"):
            if not raw:
                continue
            if len(raw) < 3 or raw[1:2] != b" ":
                raise VerificationFailure("git returned malformed index-flag evidence")
            tag = chr(raw[0])
            path = raw[2:].decode("utf-8", errors="surrogateescape")
            if tag == "S" or tag.islower():
                unsafe.add(f"{reason}:{path}")
    return sorted(unsafe)


def _candidate_fingerprint(repo: Path) -> str:
    """Hash the exact tracked diff and every untracked filesystem entry."""
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(
        _git(
            repo,
            ("diff", "HEAD", "--binary", "--no-renames", "--"),
        ).stdout
    )
    digest.update(b"\0index-assume-unchanged\0")
    digest.update(_git(repo, ("ls-files", "-v", "-z", "--")).stdout)
    digest.update(b"\0index-fsmonitor\0")
    digest.update(_git(repo, ("ls-files", "-f", "-z", "--")).stdout)
    for path in _changed_paths(repo):
        target = repo.joinpath(*path.split("/"))
        digest.update(b"\0changed-entry\0")
        digest.update(path.encode("utf-8", errors="surrogateescape"))
        try:
            metadata = target.lstat()
            digest.update(f"\0mode={metadata.st_mode:o}\0".encode("ascii"))
            if stat.S_ISLNK(metadata.st_mode):
                digest.update(
                    os.readlink(target).encode("utf-8", errors="surrogateescape")
                )
            elif stat.S_ISREG(metadata.st_mode):
                with target.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            else:
                digest.update(f"size={metadata.st_size}".encode("ascii"))
        except FileNotFoundError:
            digest.update(b"missing")
        except OSError as exc:
            raise VerificationFailure(
                f"cannot fingerprint changed path {path!r}: {exc}"
            ) from exc
    return digest.hexdigest()


def _unsafe_changed_entries(repo: Path, paths: Sequence[str]) -> list[str]:
    """Reject changed links, junctions, and other unsafe filesystem entries."""
    unsafe: list[str] = []
    for path in paths:
        current = repo
        parts = path.split("/")
        for index, part in enumerate(parts):
            current = current / part
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                break  # A tracked deletion is safe to evaluate as a Git diff.
            except OSError as exc:
                raise VerificationFailure(
                    f"cannot inspect changed path {path!r}: {exc}"
                ) from exc
            is_junction = _is_junction(current)
            if stat.S_ISLNK(metadata.st_mode) or is_junction:
                unsafe.append(path)
                break
            is_last = index == len(parts) - 1
            if (not is_last and not stat.S_ISDIR(metadata.st_mode)) or (
                is_last
                and (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1)
            ):
                unsafe.append(path)
                break
    return sorted(set(unsafe))


def _path_allowed(path: str, rules: Sequence[dict[str, str]]) -> bool:
    for rule in rules:
        scope = rule["path"]
        if path == scope:
            return True
        if rule["kind"] == "tree" and path.startswith(f"{scope}/"):
            return True
    return False


def _cap_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    if len(text) <= _OUTPUT_LIMIT:
        return text
    return (
        f"[... {len(text) - _OUTPUT_LIMIT} chars omitted ...]\n{text[-_OUTPUT_LIMIT:]}"
    )


def _create_windows_kill_job() -> int:
    """Create a Windows Job Object whose close terminates every member."""

    import ctypes
    from ctypes import wintypes

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    win_dll = getattr(ctypes, "WinDLL")
    win_error = getattr(ctypes, "WinError")
    get_last_error = getattr(ctypes, "get_last_error")
    kernel32 = win_dll("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        raise win_error(get_last_error())
    information = _ExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = 0x00002000
    if not kernel32.SetInformationJobObject(
        handle,
        9,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        error = win_error(get_last_error())
        kernel32.CloseHandle(handle)
        raise error
    return int(handle)


def _assign_and_resume_windows_job(
    process: subprocess.Popen[str],
    job_handle: int,
) -> None:
    """Assign a suspended child before it can create escaping descendants."""

    import ctypes
    from ctypes import wintypes

    process_handle = wintypes.HANDLE(int(getattr(process, "_handle")))
    win_dll = getattr(ctypes, "WinDLL")
    win_error = getattr(ctypes, "WinError")
    get_last_error = getattr(ctypes, "get_last_error")
    kernel32 = win_dll("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    if not kernel32.AssignProcessToJobObject(
        wintypes.HANDLE(job_handle), process_handle
    ):
        raise win_error(get_last_error())
    ntdll = win_dll("ntdll")
    ntdll.NtResumeProcess.argtypes = (wintypes.HANDLE,)
    ntdll.NtResumeProcess.restype = ctypes.c_long
    status = int(ntdll.NtResumeProcess(process_handle))
    if status != 0:
        raise OSError(
            f"NtResumeProcess failed with NTSTATUS 0x{status & 0xFFFFFFFF:08x}"
        )


def _close_windows_job(job_handle: int) -> str | None:
    """Close a kill-on-close Job Object, terminating any remaining members."""

    import ctypes
    from ctypes import wintypes

    win_dll = getattr(ctypes, "WinDLL")
    win_error = getattr(ctypes, "WinError")
    get_last_error = getattr(ctypes, "get_last_error")
    kernel32 = win_dll("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    if not kernel32.CloseHandle(wintypes.HANDLE(job_handle)):
        return f"could not close gate Job Object: {win_error(get_last_error())}"
    return None


def _terminate_gate_process_tree(
    process: subprocess.Popen[str],
    *,
    windows_job: int | None,
) -> str | None:
    """Hard-stop the isolated gate process tree without a second deadline."""

    if os.name == "nt":
        if windows_job is None:
            process.kill()
            return "gate process was not assigned to a Windows Job Object"
        cleanup_error = _close_windows_job(windows_job)
        if cleanup_error and process.poll() is None:
            process.kill()
        return cleanup_error
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        if process.poll() is None:
            process.kill()
    except OSError as exc:
        process.kill()
        return f"could not terminate gate process tree: {exc}"
    return None


def _run_gate(repo: Path, gate: dict[str, Any]) -> dict[str, Any]:
    argv = gate.get("argv")
    timeout = gate.get("timeout_seconds")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(arg, str) or not arg for arg in argv)
        or not isinstance(timeout, int)
        or timeout <= 0
    ):
        return {
            "argv": argv,
            "passed": False,
            "error": "invalid frozen quality-gate definition",
        }
    popen_kwargs: dict[str, Any] = {}
    windows_job: int | None = None
    if os.name == "nt":
        try:
            windows_job = _create_windows_kill_job()
        except OSError as exc:
            return {"argv": argv, "passed": False, "error": str(exc)}
        popen_kwargs["creationflags"] = (
            getattr(
                subprocess,
                "CREATE_NEW_PROCESS_GROUP",
                0,
            )
            | 0x00000004
        )
    else:
        popen_kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(
            argv,
            cwd=repo,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,
        )
        if os.name == "nt" and windows_job is not None:
            _assign_and_resume_windows_job(process, windows_job)
    except OSError as exc:
        if windows_job is not None:
            _close_windows_job(windows_job)
        return {"argv": argv, "passed": False, "error": str(exc)}
    deadline = time.monotonic() + timeout
    cleanup_reserve = min(1.0, max(0.1, timeout * 0.1))
    execution_timeout = max(0.001, timeout - cleanup_reserve)
    try:
        stdout, stderr = process.communicate(timeout=execution_timeout)
    except subprocess.TimeoutExpired as exc:
        cleanup_error = _terminate_gate_process_tree(
            process,
            windows_job=windows_job,
        )
        remaining = max(0.001, deadline - time.monotonic())
        try:
            stdout, stderr = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            stdout, stderr = exc.stdout, exc.stderr
            cleanup_error = cleanup_error or (
                "gate process tree retained output pipes after termination"
            )
        report = {
            "argv": argv,
            "passed": False,
            "timed_out": True,
            "timeout_seconds": timeout,
            "stdout": _cap_output(stdout),
            "stderr": _cap_output(stderr),
        }
        if cleanup_error:
            report["cleanup_error"] = cleanup_error
        return report
    cleanup_error = _terminate_gate_process_tree(
        process,
        windows_job=windows_job,
    )
    report = {
        "argv": argv,
        "passed": process.returncode == 0 and cleanup_error is None,
        "returncode": process.returncode,
        "stdout": _cap_output(stdout),
        "stderr": _cap_output(stderr),
    }
    if cleanup_error:
        report["cleanup_error"] = cleanup_error
    return report


def verify(
    *,
    contract_path: Path,
    expected_contract_sha256: str,
    verifier_path: Path,
    repository: Path,
    expected_workspace: Path | None,
    expected_branch: str | None,
    preflight: bool,
    run_gates: bool,
) -> dict[str, Any]:
    payload, contract_digest = _load_contract(
        contract_path,
        verifier_path,
        expected_contract_sha256,
    )
    report: dict[str, Any] = {
        "passed": False,
        "contract_sha256": contract_digest,
        "mode": "preflight" if preflight else "run-gates",
        "changed_paths": [],
        "quality_gates": [],
        "issues": [],
    }
    issues: list[dict[str, str]] = report["issues"]

    requested_repo = Path(os.path.abspath(os.path.expanduser(os.fspath(repository))))
    unsafe_repository_path = _unsafe_directory_chain(requested_repo)
    if unsafe_repository_path:
        issues.append({
            "code": "unsafe_repository_path",
            "message": "candidate repository path contains a missing, linked, or "
            "redirecting directory component: " + ", ".join(unsafe_repository_path),
        })
        return report
    actual_top = Path(
        os.path.abspath(_git_text(requested_repo, "rev-parse", "--show-toplevel"))
    )
    repo = actual_top.resolve()
    actual_common = Path(
        _git_text(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
    ).resolve(strict=False)
    expected_common = Path(str(payload.get("git_common_dir", ""))).resolve(strict=False)
    if actual_common != expected_common:
        issues.append({
            "code": "repository_identity_mismatch",
            "message": "candidate worktree belongs to a different Git repository",
        })

    if (expected_workspace is None) != (expected_branch is None):
        raise VerificationFailure(
            "expected workspace and branch must be supplied together"
        )
    if expected_workspace is not None and expected_branch is not None:
        expected_workspace_path = Path(
            os.path.abspath(os.path.expanduser(os.fspath(expected_workspace)))
        )
        actual_git_dir = Path(
            _git_text(repo, "rev-parse", "--path-format=absolute", "--git-dir")
        ).resolve(strict=False)
        try:
            actual_git_dir.relative_to(actual_common / "worktrees")
            linked_git_dir = actual_git_dir != actual_common
        except ValueError:
            linked_git_dir = False
        try:
            actual_branch = _git_text(
                repo, "symbolic-ref", "--quiet", "--short", "HEAD"
            )
        except VerificationFailure:
            actual_branch = None
        if (
            requested_repo != expected_workspace_path
            or actual_top != expected_workspace_path
            or not linked_git_dir
            or actual_branch != expected_branch
        ):
            issues.append({
                "code": "workspace_identity_mismatch",
                "message": "candidate is not the exact frozen linked worktree and branch",
            })

    head = _git_text(repo, "rev-parse", "HEAD")
    tree = _git_text(repo, "rev-parse", "HEAD^{tree}")
    if head != payload.get("base_commit") or tree != payload.get("base_tree"):
        issues.append({
            "code": "base_identity_mismatch",
            "message": "HEAD commit/tree differs from the frozen base",
        })

    unsafe_index_flags = _unsafe_index_flags(repo)
    if unsafe_index_flags:
        issues.append({
            "code": "unsafe_index_flag",
            "message": "tracked paths use index flags that can hide changes: "
            + ", ".join(unsafe_index_flags),
        })

    changed = _changed_paths(repo)
    report["changed_paths"] = changed
    if preflight:
        if changed:
            issues.append({
                "code": "preflight_not_clean",
                "message": "worktree already has changes before implementation",
            })
    else:
        if not changed:
            issues.append({
                "code": "no_candidate_changes",
                "message": "candidate has no changes to verify",
            })
        unsafe_entries = _unsafe_changed_entries(repo, changed)
        if unsafe_entries:
            issues.append({
                "code": "unsafe_changed_entry",
                "message": "changed paths contain links, junctions, or other "
                "unsafe entries: " + ", ".join(unsafe_entries),
            })
        scopes = payload.get("allowed_paths")
        rules = payload.get("allowed_path_rules")
        valid_rules = (
            isinstance(scopes, list)
            and all(isinstance(scope, str) and scope for scope in scopes)
            and isinstance(rules, list)
            and all(
                isinstance(rule, dict)
                and isinstance(rule.get("path"), str)
                and rule.get("path")
                and rule.get("kind") in {"file", "tree"}
                for rule in rules
            )
            and sorted(rule["path"] for rule in rules) == sorted(scopes)
        )
        if not valid_rules:
            issues.append({
                "code": "invalid_allowed_paths",
                "message": "contract allowed-path rules are malformed",
            })
        else:
            typed_rules = cast(Sequence[dict[str, str]], rules)
            outside = [path for path in changed if not _path_allowed(path, typed_rules)]
            if outside:
                issues.append({
                    "code": "path_scope_violation",
                    "message": "changed outside allowed paths: " + ", ".join(outside),
                })

        if not issues:
            diff_check = _git(
                repo,
                ("diff", "--check", "HEAD", "--"),
                check=False,
            )
            if diff_check.returncode != 0:
                issues.append({
                    "code": "git_diff_check_failed",
                    "message": _cap_output(
                        diff_check.stdout or diff_check.stderr
                    ).strip(),
                })

        gates = payload.get("quality_gates")
        candidate_fingerprint = None
        if not isinstance(gates, list) or not gates:
            issues.append({
                "code": "invalid_quality_gates",
                "message": "contract has no valid quality gates",
            })
        elif run_gates and not issues:
            candidate_fingerprint = _candidate_fingerprint(repo)
            report["candidate_fingerprint_before_gates"] = candidate_fingerprint
            for gate in gates:
                if not isinstance(gate, dict):
                    gate_report = {
                        "passed": False,
                        "error": "invalid frozen quality-gate definition",
                    }
                else:
                    gate_report = _run_gate(repo, gate)
                report["quality_gates"].append(gate_report)
                if not gate_report.get("passed"):
                    issues.append({
                        "code": "quality_gate_failed",
                        "message": "a frozen quality gate failed",
                    })
                    break

        if run_gates and candidate_fingerprint is not None:
            final_fingerprint = _candidate_fingerprint(repo)
            report["candidate_fingerprint_after_gates"] = final_fingerprint
            final_changed = _changed_paths(repo)
            report["changed_paths"] = final_changed
            final_unsafe_entries = _unsafe_changed_entries(repo, final_changed)
            if final_unsafe_entries:
                issues.append({
                    "code": "unsafe_changed_entry",
                    "message": "changed paths contain links, junctions, or other "
                    "unsafe entries after quality gates: "
                    + ", ".join(final_unsafe_entries),
                })
            final_unsafe_index_flags = _unsafe_index_flags(repo)
            if final_unsafe_index_flags:
                issues.append({
                    "code": "unsafe_index_flag",
                    "message": "quality gates left index flags that can hide changes: "
                    + ", ".join(final_unsafe_index_flags),
                })
            if final_fingerprint != candidate_fingerprint:
                issues.append({
                    "code": "candidate_changed_during_gates",
                    "message": "quality gates changed the candidate being evaluated",
                })
            final_common = Path(
                _git_text(
                    repo,
                    "rev-parse",
                    "--path-format=absolute",
                    "--git-common-dir",
                )
            ).resolve(strict=False)
            final_head = _git_text(repo, "rev-parse", "HEAD")
            final_tree = _git_text(repo, "rev-parse", "HEAD^{tree}")
            if (
                final_common != expected_common
                or final_head != payload.get("base_commit")
                or final_tree != payload.get("base_tree")
            ):
                issues.append({
                    "code": "candidate_identity_changed_during_gates",
                    "message": "quality gates changed the frozen Git identity",
                })

    report["passed"] = not issues
    return report


def _error_report(message: str) -> dict[str, Any]:
    return {
        "passed": False,
        "contract_sha256": None,
        "mode": None,
        "changed_paths": [],
        "quality_gates": [],
        "issues": [{"code": "verification_error", "message": message}],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--repo", default=Path.cwd(), type=Path)
    parser.add_argument("--expected-workspace", type=Path)
    parser.add_argument("--expected-branch")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run-gates", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = verify(
            contract_path=args.contract,
            expected_contract_sha256=args.expected_contract_sha256,
            verifier_path=Path(__file__),
            repository=args.repo,
            expected_workspace=args.expected_workspace,
            expected_branch=args.expected_branch,
            preflight=args.preflight,
            run_gates=args.run_gates,
        )
    except (VerificationFailure, OSError, ValueError) as exc:
        report = _error_report(str(exc))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
