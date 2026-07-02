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
import shlex
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
_GH_GLOBAL_OPTIONS_WITH_ARGS = {
    "-H",
    "-R",
    "--hostname",
    "--repo",
}
_GIT_GLOBAL_OPTIONS_WITH_ARGS = {
    "-C",
    "-c",
    "--config-env",
    "--exec-path",
    "--git-dir",
    "--namespace",
    "--super-prefix",
    "--work-tree",
}


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
    tokens = _tokenize_command(normalized)
    if _is_git_push_command(tokens):
        return True, "PR mutation"
    if _is_gh_pr_command(tokens, {"edit", "merge", "comment"}):
        return True, "PR mutation"
    if _is_gh_pr_command(tokens, {"view", "checks", "diff"}):
        return True, "CI diagnosis"
    if _is_gh_run_command(tokens):
        return True, "CI diagnosis"
    if any(pattern.search(normalized) for pattern in _PR_MUTATION_PATTERNS):
        return True, "PR mutation"
    if any(pattern.search(normalized) for pattern in _PR_REVIEW_RESOLUTION_PATTERNS):
        return True, "review-thread resolution"
    if any(pattern.search(normalized) for pattern in _PR_CI_DIAGNOSIS_PATTERNS):
        return True, "CI diagnosis"
    return False, ""


def collect_pr_head_evidence(
    repo_root: Path,
    command: str | None = None,
) -> dict[str, Any] | None:
    """Collect live PR head facts for *repo_root* and store them in context."""
    repo_root = Path(repo_root)
    local_head_sha = _git(repo_root, "rev-parse", "HEAD")
    local_branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    if not local_head_sha or not local_branch:
        return None

    gh_json = _gh_pr_view(repo_root, target=_extract_gh_pr_target(command))
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

    evidence = collect_pr_head_evidence(repo_root, command=command)
    if not evidence:
        if risk_label == "review-thread resolution":
            return (
                "Blocked review-thread resolution: verified live PR-head evidence is "
                "unavailable. Review-thread resolution requires verified live PR-head "
                "evidence before resolving it."
            )
        if risk_label == "CI diagnosis":
            return (
                "Blocked CI diagnosis: verified live PR-head evidence is unavailable. "
                "CI diagnosis requires verified live PR-head evidence plus CI evidence "
                "for the current live PR head."
            )
        return None
    if evidence.get("pr_head_verified"):
        if command is not None:
            _, risk_label = command_requires_pr_head_guard(command)
            if risk_label == "review-thread resolution":
                review_thread_block = enforce_review_thread_resolution_gate(
                    repo_root=repo_root,
                    evidence=evidence,
                    command=command,
                )
                if review_thread_block is not None:
                    return review_thread_block
            if risk_label == "CI diagnosis":
                ci_block = enforce_stale_ci_evidence_gate(
                    repo_root=repo_root,
                    evidence=evidence,
                    command=command,
                )
                if ci_block is not None:
                    return ci_block
        return None
    return _block_message(action_label=risk_label or action_label, evidence=evidence)


def enforce_review_thread_resolution_gate(
    *,
    repo_root: Path | None,
    evidence: dict[str, Any] | None = None,
    command: str | None = None,
) -> str | None:
    """Block review-thread resolution unless patch evidence proves the fix."""
    if repo_root is None:
        return None
    evidence = dict(evidence or {})
    if not evidence:
        evidence = dict(update_runtime_evidence())

    comment_id = _resolve_review_thread_comment_id(command, evidence)
    if not comment_id:
        return (
            "Blocked review-thread resolution: no review thread/comment ID is available. "
            "Record the specific thread or comment before resolving it."
        )

    if not evidence.get("pr_head_verified"):
        return _block_message(action_label="review-thread resolution", evidence=evidence)

    failing_file = str(evidence.get("failing_file") or "").strip()
    failing_line = _coerce_int(evidence.get("failing_line"))
    if not failing_file or failing_line is None:
        return (
            "Blocked review-thread resolution: missing failing file/line evidence for "
            f"thread/comment {comment_id}. Capture the reviewer target before resolving it."
        )

    patch_text = _fetch_pr_patch(repo_root)
    if patch_text is None:
        update_runtime_evidence(
            review_thread_comment_id=comment_id,
            fetched_pr_patch_checked=False,
            review_thread_patch_clean=False,
        )
        return (
            "Blocked review-thread resolution: unable to fetch the live PR patch for "
            f"thread/comment {comment_id}. Verify the checkout and retry."
        )

    issue_still_present = _patch_still_mentions_issue(
        patch_text,
        failing_file=failing_file,
        failing_line=failing_line,
        failing_blob_sha=str(evidence.get("failing_blob_sha") or "").strip() or None,
    )
    update_runtime_evidence(
        review_thread_comment_id=comment_id,
        fetched_pr_patch_checked=True,
        review_thread_patch_clean=not issue_still_present,
    )
    if issue_still_present:
        return (
            "Blocked review-thread resolution: the fetched PR patch still shows the "
            f"reported issue for thread/comment {comment_id} in {failing_file}:{failing_line}. "
            "Fix the live PR head first, then retry the resolution."
        )
    return None


def enforce_stale_ci_evidence_gate(
    *,
    repo_root: Path | None,
    evidence: dict[str, Any] | None = None,
    command: str | None = None,
) -> str | None:
    """Block PR CI diagnosis/reruns unless CI evidence matches live PR state."""
    if repo_root is None:
        return None
    if command is not None:
        should_check, risk_label = command_requires_pr_head_guard(command)
        if not should_check or risk_label != "CI diagnosis":
            return None
    evidence = dict(evidence or update_runtime_evidence() or {})

    if not evidence.get("pr_head_verified"):
        return _block_message(action_label="CI diagnosis", evidence=evidence)

    live_pr_head_sha = str(evidence.get("live_pr_head_sha") or "").strip()
    ci_checkout_sha = _resolve_ci_checkout_sha(evidence)
    ci_run_id = str(evidence.get("ci_run_id") or evidence.get("ci_check_run_id") or "").strip()
    ci_status = str(evidence.get("ci_status") or "").strip()
    ci_conclusion = str(evidence.get("ci_conclusion") or "").strip()
    failing_file = str(evidence.get("failing_file") or "").strip()
    failing_blob_sha = str(evidence.get("failing_blob_sha") or "").strip()
    failing_line = _coerce_int(evidence.get("failing_line"))
    ci_fetched = bool(evidence.get("ci_evidence_fetched_for_live_pr_head"))

    missing = []
    if not ci_fetched:
        missing.append("fetched CI evidence for the current live PR head")
    if not live_pr_head_sha:
        missing.append("live PR head SHA")
    if not ci_checkout_sha:
        missing.append("CI checkout/head SHA")
    if not ci_status:
        missing.append("CI status")
    if not ci_conclusion:
        missing.append("CI conclusion")
    if not failing_file:
        missing.append("failing file")
    if not failing_blob_sha:
        missing.append("failing blob SHA")
    if failing_line is None:
        missing.append("failing line")
    if missing:
        update_runtime_evidence(
            ci_run_id=ci_run_id or None,
            ci_checkout_sha=ci_checkout_sha or None,
            ci_status=ci_status or None,
            ci_conclusion=ci_conclusion or None,
            ci_evidence_fetched_for_live_pr_head=ci_fetched,
        )
        return (
            "Blocked CI diagnosis: missing CI evidence for the current live PR head "
            f"({', '.join(missing)}). Gather CI evidence for the verified live PR head "
            "before diagnosing or rerunning stale CI."
        )

    ci_sha_matches_live = ci_checkout_sha == live_pr_head_sha
    update_runtime_evidence(
        ci_run_id=ci_run_id or None,
        ci_checkout_sha=ci_checkout_sha,
        ci_status=ci_status,
        ci_conclusion=ci_conclusion,
        ci_evidence_fetched_for_live_pr_head=True,
        ci_evidence_live_pr_head_sha=live_pr_head_sha,
        ci_evidence_local_head_sha=str(evidence.get("local_head_sha") or "").strip() or None,
        ci_evidence_ci_sha_matches_live_pr_head=ci_sha_matches_live,
        ci_stale_ci_verified=ci_sha_matches_live is False,
    )
    if not ci_sha_matches_live:
        return (
            "Blocked CI diagnosis: CI checkout/head SHA does not match the verified "
            "live PR head SHA.\n"
            f"PR: #{evidence.get('pr_number')} {evidence.get('pr_repository') or 'unknown-repo'}\n"
            f"CI run: {ci_run_id or 'unknown'}\n"
            f"CI status: {ci_status or 'unknown'}\n"
            f"CI conclusion: {ci_conclusion or 'unknown'}\n"
            f"CI checkout/head SHA: {ci_checkout_sha}\n"
            f"Live PR head SHA: {live_pr_head_sha}\n"
            f"Local HEAD: {str(evidence.get('local_head_sha') or 'unknown')}\n"
            "Repair the checkout/ref alignment and refresh CI evidence before diagnosing "
            "or rerunning CI."
        )
    return None


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


def _resolve_review_thread_comment_id(
    command: str | None,
    evidence: dict[str, Any],
) -> str | None:
    for key in ("review_thread_comment_id", "thread_comment_id", "comment_id"):
        value = str(evidence.get(key) or "").strip()
        if value:
            return value
    if command:
        match = re.search(
            r'(?:threadId|commentId)\s*[:=]\s*["\']?([A-Za-z0-9_.:-]+)["\']?',
            command,
        )
        if match:
            return match.group(1)
    return None


def _resolve_ci_checkout_sha(evidence: dict[str, Any]) -> str:
    for key in ("ci_checkout_sha", "ci_head_sha", "ci_run_head_sha", "ci_head_ref_sha"):
        value = str(evidence.get(key) or "").strip()
        if value:
            return value
    return ""


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _fetch_pr_patch(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["gh", "pr", "diff", "--patch"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    patch_text = completed.stdout or ""
    return patch_text if patch_text.strip() else None


def _tokenize_command(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return [part for part in str(command).split() if part]


def _skip_global_options(tokens: list[str], option_names: set[str]) -> int | None:
    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token in option_names:
            i += 2
            continue
        if any(token.startswith(option) and token != option for option in option_names):
            i += 1
            continue
        if token.startswith("-"):
            i += 1
            continue
        return i
    return None


def _is_git_push_command(tokens: list[str]) -> bool:
    if not tokens or tokens[0] != "git":
        return False
    command_index = _skip_global_options(tokens, _GIT_GLOBAL_OPTIONS_WITH_ARGS)
    return command_index is not None and tokens[command_index] == "push"


def _is_gh_pr_command(tokens: list[str], actions: set[str]) -> bool:
    if not tokens or tokens[0] != "gh":
        return False
    command_index = _skip_global_options(tokens, _GH_GLOBAL_OPTIONS_WITH_ARGS)
    if command_index is None or tokens[command_index] != "pr":
        return False
    if command_index + 1 >= len(tokens):
        return False
    return tokens[command_index + 1] in actions


def _is_gh_run_command(tokens: list[str]) -> bool:
    if not tokens or tokens[0] != "gh":
        return False
    command_index = _skip_global_options(tokens, _GH_GLOBAL_OPTIONS_WITH_ARGS)
    if command_index is None or tokens[command_index] != "run":
        return False
    return command_index + 1 < len(tokens) and tokens[command_index + 1] in {"rerun", "watch", "view"}


def _extract_gh_pr_target(command: str | None) -> str | None:
    tokens = _tokenize_command(str(command or ""))
    if not tokens or tokens[0] != "gh":
        return None
    command_index = _skip_global_options(tokens, _GH_GLOBAL_OPTIONS_WITH_ARGS)
    if command_index is None or tokens[command_index] != "pr":
        return None
    if command_index + 1 >= len(tokens):
        return None
    action = tokens[command_index + 1]
    if action not in {"view", "edit", "merge", "comment", "checks", "diff"}:
        return None
    target_index = command_index + 2
    if target_index >= len(tokens):
        return None
    target = tokens[target_index].strip()
    if not target or target.startswith("-"):
        return None
    return target


def _patch_still_mentions_issue(
    patch_text: str,
    *,
    failing_file: str,
    failing_line: int,
    failing_blob_sha: str | None = None,
) -> bool:
    if failing_blob_sha and failing_blob_sha in patch_text:
        return True

    current_file = None
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            current_file = None
            match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            if match:
                current_file = match.group(2)
            continue
        if current_file is None:
            continue
        if not _paths_match(current_file, failing_file):
            continue
        if not line.startswith("@@"):
            continue
        hunk_match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if not hunk_match:
            continue
        start = int(hunk_match.group(1))
        length = int(hunk_match.group(2) or "1")
        end = start + max(length - 1, 0)
        if start <= failing_line <= end:
            return True
    return False


def _paths_match(left: str, right: str) -> bool:
    left_norm = left.replace("\\", "/").lstrip("./")
    right_norm = right.replace("\\", "/").lstrip("./")
    return left_norm == right_norm or left_norm.endswith(f"/{right_norm}") or right_norm.endswith(f"/{left_norm}")


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


def _gh_pr_view(repo_root: Path, target: str | None = None) -> dict[str, Any] | None:
    try:
        args = ["gh", "pr", "view"]
        if target:
            args.append(target)
        args.extend(["--json", "number,headRefName,headRefOid,repository"])
        completed = subprocess.run(
            args,
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
