"""PR-head invariant guard for high-risk PR actions.

The guard is intentionally narrow:

- it only activates when a repository has an associated live PR; and
- it only blocks actions that would mutate PR state or rely on PR-local
  diagnosis while the local checkout is out of sync with the live head.

The shared runtime evidence store carries the resolved PR/head facts so nested
tool calls can reuse the same data without re-plumbing it through every layer.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from agent.runtime_evidence import update_runtime_evidence

_PR_MUTATION_PATTERNS = (
    re.compile(r"\bgit\s+push\b", re.IGNORECASE),
    re.compile(r"\bgh\s+pr\s+(?:edit|merge|comment)\b", re.IGNORECASE),
)
_PR_CI_DIAGNOSIS_PATTERNS = (
    re.compile(r"\bgh\s+pr\s+(?:view|checks|diff)\b", re.IGNORECASE),
    re.compile(r"\bgh\s+run\s+(?:view|rerun|watch)\b", re.IGNORECASE),
    re.compile(r"\bgh\s+workflow\s+run\b", re.IGNORECASE),
)
_PR_REVIEW_RESOLUTION_PATTERNS = (
    re.compile(r"resolveReviewThread", re.IGNORECASE),
    re.compile(r"resolve\s+thread", re.IGNORECASE),
    re.compile(r"thread\s+resolve", re.IGNORECASE),
)


def resolve_repo_root_for_path(path_text: str | None) -> Path | None:
    """Return the nearest Git repository root for *path_text*."""
    text = str(path_text or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve(strict=False)
    except Exception:
        resolved = path
    for candidate in [resolved, *resolved.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def command_requires_pr_head_guard(command: str) -> tuple[bool, str]:
    """Return whether *command* is a PR-head-sensitive action."""
    normalized = " ".join(str(command or "").split())
    if not normalized:
        return False, ""
    if any(pattern.search(normalized) for pattern in _PR_MUTATION_PATTERNS):
        return True, "PR mutation"
    if any(pattern.search(normalized) for pattern in _PR_REVIEW_RESOLUTION_PATTERNS):
        return True, "review-thread resolution"
    if any(pattern.search(normalized) for pattern in _PR_CI_DIAGNOSIS_PATTERNS):
        return True, "CI diagnosis"
    return False, ""


def collect_pr_head_evidence(repo_root: Path) -> dict[str, Any] | None:
    """Collect live PR head facts for *repo_root* and store them in context."""
    repo_root = Path(repo_root)
    local_head_sha = _git(repo_root, "rev-parse", "HEAD")
    local_branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    if not local_head_sha or not local_branch:
        return None

    gh_json = _gh_pr_view(repo_root)
    if gh_json is None:
        return None

    pr_number = gh_json.get("number")
    live_pr_head_branch = gh_json.get("headRefName") or ""
    live_pr_head_sha = gh_json.get("headRefOid") or ""
    pr_repository = _extract_repo_name(gh_json.get("repository"))
    if not live_pr_head_branch or not live_pr_head_sha or not pr_repository or pr_number is None:
        return None

    pr_head_verified = (
        local_head_sha == live_pr_head_sha
        and local_branch == live_pr_head_branch
    )
    evidence = update_runtime_evidence(
        pr_number=pr_number,
        pr_repository=pr_repository,
        live_pr_head_branch=live_pr_head_branch,
        live_pr_head_sha=live_pr_head_sha,
        local_branch=local_branch,
        local_head_sha=local_head_sha,
        pr_head_verified=pr_head_verified,
    )
    evidence["pr_context_repo_root"] = str(repo_root)
    return evidence


def enforce_pr_head_invariant(
    *,
    repo_root: Path | None,
    action_label: str,
    command: str | None = None,
) -> str | None:
    """Return a block message when the live PR head does not match HEAD."""
    if repo_root is None:
        return None
    if command is not None:
        should_check, risk_label = command_requires_pr_head_guard(command)
        if not should_check:
            return None
    else:
        risk_label = action_label

    evidence = collect_pr_head_evidence(repo_root)
    if not evidence:
        return None
    if evidence.get("pr_head_verified"):
        return None
    return _block_message(action_label=risk_label or action_label, evidence=evidence)


def _block_message(*, action_label: str, evidence: dict[str, Any]) -> str:
    pr_number = evidence.get("pr_number")
    pr_repository = evidence.get("pr_repository") or "unknown-repo"
    live_branch = evidence.get("live_pr_head_branch") or "unknown"
    live_sha = evidence.get("live_pr_head_sha") or "unknown"
    local_branch = evidence.get("local_branch") or "unknown"
    local_sha = evidence.get("local_head_sha") or "unknown"
    return (
        f"Blocked {action_label}: local checkout does not match the live PR head.\n"
        f"PR: #{pr_number} {pr_repository}\n"
        f"Live head: {live_branch} @ {live_sha}\n"
        f"Local branch: {local_branch}\n"
        f"Local HEAD: {local_sha}\n"
        "Repair the checkout/ref alignment first, then retry this action."
    )


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _gh_pr_view(repo_root: Path) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            ["gh", "pr", "view", "--json", "number,headRefName,headRefOid,repository"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        parsed = json.loads(completed.stdout)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_repo_name(repository: Any) -> str | None:
    if not isinstance(repository, dict):
        return None
    for key in ("nameWithOwner", "name_with_owner"):
        value = repository.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    owner = repository.get("owner")
    owner_login = ""
    if isinstance(owner, dict):
        owner_login = str(owner.get("login") or owner.get("name") or "").strip()
    name = str(repository.get("name") or "").strip()
    if owner_login and name:
        return f"{owner_login}/{name}"
    return name or None
