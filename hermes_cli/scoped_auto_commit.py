"""Scoped local auto-commit gate for Kanban reviewer-approved handoffs.

The helpers in this module are deliberately small and local-only.  They do not
push, deploy, restart services, or touch Hermes/OpenClaw runtime configuration.
They encode the Gate 10F invariant: a Coder handoff that explicitly requires
review may only be committed after an APPROVED Reviewer-B verdict, green checks,
and zero forbidden-scope actions.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hermes_cli.control_plane_gate import validate_reviewer_verdict_metadata


@dataclass(frozen=True)
class ScopedAutoCommitDecision:
    """Decision returned by :func:`evaluate_scoped_auto_commit_gate`."""

    allowed: bool
    reason: str
    blocking_findings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScopedCommitReceipt:
    """Receipt for a successful local-only scoped commit."""

    commit_hash: str
    committed_paths: list[str]
    preexisting_dirty_paths: list[str]
    reviewer_verdict: str


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "ok", "pass", "passed", "green"}
    return bool(value)


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _all_checks_green(checks: Mapping[str, Any] | Iterable[Any] | None) -> tuple[bool, list[str]]:
    if not checks:
        return False, ["acceptance checks are required"]
    failures: list[str] = []
    if isinstance(checks, Mapping):
        for name, value in checks.items():
            if not _truthy(value):
                failures.append(str(name))
    else:
        for idx, value in enumerate(checks):
            if not _truthy(value):
                failures.append(f"check[{idx}]")
    return not failures, failures


def _forbidden_actions(metadata: Mapping[str, Any] | None) -> int | None:
    if not isinstance(metadata, Mapping):
        return None
    return _int_value(metadata.get("forbidden_actions_taken"))


def evaluate_scoped_auto_commit_gate(
    *,
    coder_metadata: Mapping[str, Any] | None,
    reviewer_metadata: Mapping[str, Any] | None,
    acceptance_checks: Mapping[str, Any] | Iterable[Any] | None,
    anti_scope: Mapping[str, Any] | None = None,
    expected_workflow_id: str | None = None,
) -> ScopedAutoCommitDecision:
    """Return whether a single scoped local commit is allowed.

    The gate is intentionally conservative.  Missing evidence blocks; it is not
    inferred from prose.  ``anti_scope`` may contain booleans such as
    ``default_board_unchanged`` and ``temp_credential_symlinks_removed`` plus an
    optional ``violations`` list.  Any false value or non-empty violation list
    blocks the commit.
    """
    findings: list[str] = []

    if not isinstance(coder_metadata, Mapping):
        findings.append("coder_metadata object is required")
    else:
        if not _truthy(coder_metadata.get("review_required")):
            findings.append("coder review_required=true is required")
        if _truthy(coder_metadata.get("coder_done_without_review")):
            findings.append("coder completed done before review")
        coder_forbidden = _forbidden_actions(coder_metadata)
        if coder_forbidden is None:
            findings.append("coder forbidden_actions_taken=0 is required")
        elif coder_forbidden != 0:
            findings.append("coder forbidden_actions_taken must be 0")
        if not _truthy(coder_metadata.get("scope_attestation")):
            findings.append("coder scope_attestation=true is required")
        if not _truthy(coder_metadata.get("no_prose_only_terminal_output")):
            findings.append("no prose-only terminal output proof is required")

    reviewer_missing = validate_reviewer_verdict_metadata(
        reviewer_metadata,
        expected_workflow_id=expected_workflow_id,
    )
    if reviewer_missing:
        findings.extend(f"reviewer metadata invalid: {item}" for item in reviewer_missing)
    elif reviewer_metadata and reviewer_metadata.get("verdict") != "APPROVED":
        findings.append(f"reviewer verdict is {reviewer_metadata.get('verdict')}")

    reviewer_forbidden = _forbidden_actions(reviewer_metadata)
    if reviewer_forbidden is not None and reviewer_forbidden != 0:
        findings.append("reviewer forbidden_actions_taken must be 0")

    checks_green, failed_checks = _all_checks_green(acceptance_checks)
    if not checks_green:
        findings.extend(f"acceptance check not green: {name}" for name in failed_checks)

    if anti_scope is None:
        findings.append("anti_scope evidence is required")
    elif not isinstance(anti_scope, Mapping):
        findings.append("anti_scope must be an object")
    else:
        violations = anti_scope.get("violations") or []
        if violations:
            findings.append("anti-scope violations present: " + ", ".join(map(str, violations)))
        for key, value in anti_scope.items():
            if key == "violations":
                continue
            if not _truthy(value):
                findings.append(f"anti_scope {key}=green/true is required")

    if findings:
        return ScopedAutoCommitDecision(False, "scoped_auto_commit_blocked", findings)
    return ScopedAutoCommitDecision(True, "reviewer_approved_green_checks_scope_zero", [])


def _git_lines(repo: Path, args: Sequence[str]) -> list[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [line for line in completed.stdout.splitlines() if line]


def _git(repo: Path, args: Sequence[str], *, env: Mapping[str, str] | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, **dict(env or {})},
    )
    return completed.stdout.strip()


def _path_exists_or_is_tracked(repo: Path, path: str) -> bool:
    if (repo / path).exists():
        return True
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", path],
        cwd=str(repo),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.returncode == 0


def _normalize_scoped_paths(paths: Iterable[str]) -> list[str]:
    scoped: list[str] = []
    for raw in paths:
        path = str(raw).strip()
        if not path:
            continue
        if path.startswith("/") or ".." in Path(path).parts:
            raise ValueError(f"scoped path must be repo-relative and non-traversing: {raw!r}")
        scoped.append(path)
    if not scoped:
        raise ValueError("at least one scoped path is required")
    return sorted(set(scoped))


def create_scoped_local_commit(
    *,
    repo_path: str | Path,
    scoped_paths: Iterable[str],
    message: str,
    coder_metadata: Mapping[str, Any] | None,
    reviewer_metadata: Mapping[str, Any] | None,
    acceptance_checks: Mapping[str, Any] | Iterable[Any] | None,
    anti_scope: Mapping[str, Any] | None,
    expected_workflow_id: str | None = None,
    git_env: Mapping[str, str] | None = None,
) -> ScopedCommitReceipt:
    """Create one local Git commit containing only ``scoped_paths``.

    Unstaged/untracked dirty files outside ``scoped_paths`` are treated as
    pre-existing worktree state and left untouched.  Already-staged paths outside
    scope block the operation because committing would include unrelated work.
    """
    decision = evaluate_scoped_auto_commit_gate(
        coder_metadata=coder_metadata,
        reviewer_metadata=reviewer_metadata,
        acceptance_checks=acceptance_checks,
        anti_scope=anti_scope,
        expected_workflow_id=expected_workflow_id,
    )
    if not decision.allowed:
        raise PermissionError("; ".join(decision.blocking_findings))

    repo = Path(repo_path).resolve()
    scoped = _normalize_scoped_paths(scoped_paths)
    scoped_set = set(scoped)

    root = _git(repo, ["rev-parse", "--show-toplevel"])
    if Path(root).resolve() != repo:
        repo = Path(root).resolve()

    unknown_scoped_paths = [
        path for path in scoped if not _path_exists_or_is_tracked(repo, path)
    ]
    if unknown_scoped_paths:
        raise ValueError(
            "scoped path does not exist or is not tracked: "
            + ", ".join(unknown_scoped_paths)
        )

    staged_before = set(_git_lines(repo, ["diff", "--name-only", "--cached"]))
    staged_outside_scope = sorted(staged_before - scoped_set)
    if staged_outside_scope:
        raise PermissionError(
            "staged paths outside scoped commit: " + ", ".join(staged_outside_scope)
        )

    dirty_before = set(_git_lines(repo, ["diff", "--name-only"]))
    untracked_before = set(_git_lines(repo, ["ls-files", "--others", "--exclude-standard"]))
    preexisting_dirty = sorted((dirty_before | untracked_before | staged_before) - scoped_set)

    _git(repo, ["add", "--", *scoped])
    staged_after = set(_git_lines(repo, ["diff", "--name-only", "--cached"]))
    if not staged_after:
        raise ValueError("no scoped changes staged for commit")
    staged_after_outside_scope = sorted(staged_after - scoped_set)
    if staged_after_outside_scope:
        raise PermissionError(
            "git add staged paths outside scoped commit: " + ", ".join(staged_after_outside_scope)
        )

    _git(repo, ["commit", "-m", message], env=git_env)
    commit_hash = _git(repo, ["rev-parse", "HEAD"])
    return ScopedCommitReceipt(
        commit_hash=commit_hash,
        committed_paths=sorted(staged_after),
        preexisting_dirty_paths=preexisting_dirty,
        reviewer_verdict=str(reviewer_metadata.get("verdict")) if reviewer_metadata else "",
    )
