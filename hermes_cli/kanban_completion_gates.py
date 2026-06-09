"""Verification gates run before `complete_task` writes `status=done`.

Each gate is a pure function that takes structured inputs and returns either
``None`` (pass) or a violation dataclass describing why the completion should
be rejected. The caller (``complete_task`` in ``kanban_db.py``) collects any
violation, emits an auditable event, and raises so the worker layer surfaces a
structured retry message.

Pattern mirrors the existing ``_verify_created_cards`` /
``HallucinatedCardsError`` flow — gates fire BEFORE the write transaction so
state is unchanged on rejection and the worker can simply retry with corrected
output.

Three gates ship today (Tranche 1 of v6.7, closes #28, #62, #64):

1. :func:`verify_runtime_floor` — per-role floor on
   ``completed_at - started_at``. Catches Tony's 20-second "approve" verdicts
   and Friday's 59-second "implemented 7 dispatcher gates" claims.

2. :func:`verify_workspace_diff` — when a non-review worker on a
   ``dir`` / ``worktree`` workspace claims to have produced code, the workspace
   must show a real diff against its tracking base. Catches Friday's "Wave A
   gates implemented" with zero changes on the branch.

3. :func:`verify_no_stray_artifacts` — reject untracked artifacts matching
   patterns the swarm has historically committed by accident
   (``*evidence*``, ``commit-hash*``, ``triage/*``, ``tmp-*``, and tracked
   files with no extension and no shebang — the "all prior block evidence
   files" failure mode).

See hermes-jarvis#61 for the bootstrap-paradox case study that motivates
these gates.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# =====================================================================
# Per-role runtime floors (#64)
# =====================================================================

# Empirically derived from the 2026-06-09 build-chain failure: any number
# below the floor on a non-orchestration task is more likely fabrication
# than fast work. Workers may opt out per-call via the
# ``x_fast_justified`` metadata field, surfaced through ``allow_below_floor``.
ROLE_RUNTIME_FLOORS_SECONDS: dict[str, int] = {
    # Build / implementation roles — real code changes don't ship in <5 min
    "friday": 5 * 60,
    "shuri": 5 * 60,
    "build-engineer": 5 * 60,
    # Review roles — even a tiny review needs to read the diff
    "tony": 90,
    "tchalla": 90,
    "vision": 90,
    "reviewer": 90,
    # Orchestration roles — JARVIS umbrella spawn can be fast and correct
    "jarvis": 0,
    "pepper": 0,
    "banner": 0,
}


@dataclass(frozen=True)
class RuntimeFloorViolation:
    role: str
    started_at: int
    completed_at: int
    floor_seconds: int
    actual_seconds: int

    def message(self) -> str:
        return (
            f"runtime-floor: {self.role} completed in {self.actual_seconds}s, "
            f"below the {self.floor_seconds}s floor for this role. "
            f"Either keep working (add evidence and re-call kanban_complete after "
            f"the floor passes) or, if the work was genuinely trivial, set "
            f"metadata={{\"x_fast_justified\": \"<one-line reason>\"}} on the "
            f"completion call."
        )


def verify_runtime_floor(
    assignee: Optional[str],
    started_at: Optional[int],
    completed_at: int,
    *,
    allow_below_floor: bool = False,
) -> Optional[RuntimeFloorViolation]:
    """Return a violation if the worker's runtime is below its role floor.

    ``started_at`` is the timestamp the dispatcher recorded when the worker
    claimed the task (NOT the run-row creation time). ``completed_at`` is
    "now" from the dispatcher's perspective when ``complete_task`` runs.

    A floor of 0 (or an unknown assignee, or a missing ``started_at``) is a
    pass — we never invent floors for roles we don't know.
    """
    if allow_below_floor:
        return None
    if not assignee or started_at is None:
        return None
    floor = ROLE_RUNTIME_FLOORS_SECONDS.get(assignee.lower())
    if not floor:
        return None
    actual = max(0, completed_at - int(started_at))
    if actual >= floor:
        return None
    return RuntimeFloorViolation(
        role=assignee, started_at=int(started_at), completed_at=completed_at,
        floor_seconds=floor, actual_seconds=actual,
    )


# =====================================================================
# Workspace-diff gate (#62)
# =====================================================================

REVIEW_ROLES = {"tony", "tchalla", "vision", "reviewer"}
ORCHESTRATION_ROLES = {"jarvis", "pepper", "banner"}

# Phrases workers used in fabricated completion summaries that should be
# backed by a real diff. Conservative — only triggers the gate when the
# worker has explicitly claimed code changes.
_IMPLEMENTATION_CLAIM_PATTERNS = [
    re.compile(r"\b(implement(?:ed|s)?|build(?:s|t)?|add(?:ed|s)?|"
               r"creat(?:ed|es)?|wrote|wr(?:ites|ote)|ship(?:ped|s)?|"
               r"land(?:ed|s)?|introduc(?:ed|es)?|refactor(?:ed|s)?|"
               r"fix(?:ed|es)?|patch(?:ed|es)?)\b", re.IGNORECASE),
]


@dataclass(frozen=True)
class WorkspaceDiffViolation:
    assignee: str
    workspace_path: str
    summary_excerpt: str
    diff_stat: str  # may be empty string if no changes

    def message(self) -> str:
        diff_preview = self.diff_stat.strip() or "(no changes against tracking base)"
        return (
            f"workspace-diff: {self.assignee} summary claims implementation "
            f"work ({self.summary_excerpt!r}) but `git diff` in "
            f"{self.workspace_path} shows: {diff_preview}. "
            f"Either produce the changes the summary describes, or block "
            f"with an honest reason. To skip this check on a doc-only or "
            f"genuinely-no-code task, set metadata={{\"x_no_code\": true}}."
        )


def _summary_claims_implementation(summary: str) -> bool:
    return any(p.search(summary or "") for p in _IMPLEMENTATION_CLAIM_PATTERNS)


def _git_diff_stat_against_base(workspace_path: str) -> str:
    """Return `git diff --stat` against the workspace's tracking base.

    Tracking base is, in order: ``@{upstream}`` if it exists, else
    ``origin/main`` if it exists, else ``main``. If git rejects all three,
    returns the empty string (gate treats as "no diff").

    Subprocess calls use a hard 10s wallclock so a hung git can't stall the
    dispatcher.
    """
    if not workspace_path or not os.path.isdir(workspace_path):
        return ""
    if not os.path.isdir(os.path.join(workspace_path, ".git")):
        # Worktree-backed dirs have .git as a file pointer; that's fine.
        if not os.path.isfile(os.path.join(workspace_path, ".git")):
            return ""

    def _run(args: list[str]) -> Optional[str]:
        try:
            out = subprocess.run(
                args, cwd=workspace_path, capture_output=True,
                text=True, timeout=10, check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None
        if out.returncode != 0:
            return None
        return out.stdout

    for base_spec in ("@{upstream}", "origin/main", "main"):
        # First check the base exists (cheap), then diff against it.
        if _run(["git", "rev-parse", "--verify", base_spec]) is None:
            continue
        stat = _run(["git", "diff", "--stat", base_spec, "HEAD"])
        if stat is not None:
            return stat
    return ""


def verify_workspace_diff(
    assignee: Optional[str],
    workspace_kind: Optional[str],
    workspace_path: Optional[str],
    summary: Optional[str],
    *,
    allow_no_code: bool = False,
) -> Optional[WorkspaceDiffViolation]:
    """Reject completions that claim code work but show no diff.

    Skipped (returns None) when:
    - assignee is a review or orchestration role
    - workspace is scratch (no diff target)
    - workspace_path is missing or not a directory
    - summary doesn't claim implementation
    - caller opted out via ``allow_no_code=True``
    """
    if allow_no_code:
        return None
    if not assignee:
        return None
    role = assignee.lower()
    if role in REVIEW_ROLES or role in ORCHESTRATION_ROLES:
        return None
    if (workspace_kind or "scratch") not in {"dir", "worktree"}:
        return None
    if not workspace_path or not os.path.isdir(workspace_path):
        # Wrong / typo'd path is the dispatcher's problem to surface
        # elsewhere — we don't punish the worker for it.
        return None
    if not _summary_claims_implementation(summary or ""):
        return None
    diff_stat = _git_diff_stat_against_base(workspace_path)
    # A real implementation produces SOME change line. We only reject when
    # the diff is empty / whitespace.
    if diff_stat and diff_stat.strip():
        return None
    summary_excerpt = (summary or "").strip().splitlines()[0][:200]
    return WorkspaceDiffViolation(
        assignee=assignee, workspace_path=workspace_path,
        summary_excerpt=summary_excerpt, diff_stat=diff_stat,
    )


# =====================================================================
# Repo-hygiene gate (#28)
# =====================================================================

# Patterns that mark a path as "stray orchestration artifact" rather than
# real source. Matched against the path relative to the repo root, case
# insensitive. Aligned with the agent-dashboard PR #1 audit findings (
# `all prior block evidence files`, `commit-hash.txt`, `triage/v6.4-*`).
_STRAY_PATH_PATTERNS = [
    re.compile(r"(^|/)(evidence|.*-evidence|.*_evidence)(/|\b)", re.IGNORECASE),
    re.compile(r"(^|/)commit-hash(\.[a-z]+)?$", re.IGNORECASE),
    re.compile(r"(^|/)triage/", re.IGNORECASE),
    re.compile(r"(^|/)tmp-[^/]+$", re.IGNORECASE),
    re.compile(r"(^|/)all prior block evidence files$", re.IGNORECASE),
]


@dataclass(frozen=True)
class StrayArtifactViolation:
    workspace_path: str
    stray_paths: tuple[str, ...]

    def message(self) -> str:
        listing = "\n  ".join(self.stray_paths)
        return (
            f"repo-hygiene: workspace {self.workspace_path} contains files "
            f"that look like leftover orchestration artifacts:\n  {listing}\n"
            f"Delete (or .gitignore) them before calling kanban_complete. "
            f"If a stray-looking path is intentional, prefix it with a real "
            f"file extension and add a one-line comment explaining why it's "
            f"in the repo."
        )


def _has_shebang(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(2)
        return head == b"#!"
    except (OSError, IOError):
        return False


def _stray_path_score(repo_root: str, rel_path: str) -> bool:
    """True if ``rel_path`` looks like a stray artifact."""
    norm = rel_path.replace("\\", "/")
    if any(p.search(norm) for p in _STRAY_PATH_PATTERNS):
        return True
    # Tracked file with no extension and no shebang — the "all prior block
    # evidence files" failure mode.
    base = os.path.basename(norm)
    if "." not in base and not _has_shebang(os.path.join(repo_root, rel_path)):
        return True
    return False


def _list_workspace_files(workspace_path: str) -> list[str]:
    """Return the union of `git ls-files` (tracked) and `git ls-files
    --others --exclude-standard` (untracked & not gitignored), as relative
    paths. Empty list on any git error.
    """
    if not workspace_path or not os.path.isdir(workspace_path):
        return []

    def _run(args: list[str]) -> Optional[str]:
        try:
            out = subprocess.run(
                args, cwd=workspace_path, capture_output=True,
                text=True, timeout=10, check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None
        if out.returncode != 0:
            return None
        return out.stdout

    tracked = _run(["git", "ls-files"]) or ""
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard"]) or ""
    paths = set()
    for blob in (tracked, untracked):
        for line in blob.splitlines():
            line = line.strip()
            if line:
                paths.add(line)
    return sorted(paths)


def verify_no_stray_artifacts(
    workspace_kind: Optional[str],
    workspace_path: Optional[str],
    *,
    allow_stray: bool = False,
) -> Optional[StrayArtifactViolation]:
    """Reject completions where the workspace tree contains stray files.

    Skipped (returns None) when:
    - workspace is scratch
    - workspace_path is missing or not a directory
    - caller opted out via ``allow_stray=True``
    """
    if allow_stray:
        return None
    if (workspace_kind or "scratch") not in {"dir", "worktree"}:
        return None
    if not workspace_path or not os.path.isdir(workspace_path):
        return None
    stray = [p for p in _list_workspace_files(workspace_path)
             if _stray_path_score(workspace_path, p)]
    if not stray:
        return None
    return StrayArtifactViolation(
        workspace_path=workspace_path, stray_paths=tuple(stray),
    )


# =====================================================================
# Exception class for the integration in `complete_task`
# =====================================================================

class CompletionGateError(ValueError):
    """Raised by ``complete_task`` when one or more v6.7 gates reject.

    ``violations`` is a list of dataclasses (one per failed gate). Each has
    a ``.message()`` returning a worker-actionable string. Subclass of
    ``ValueError`` so existing tool-error handlers treat this as a
    recoverable user error (same convention as
    :class:`HallucinatedCardsError`).
    """

    def __init__(self, violations: list, completing_task_id: str):
        self.violations = list(violations)
        self.completing_task_id = completing_task_id
        lines = [v.message() for v in self.violations]
        super().__init__(
            "kanban_complete blocked by v6.7 gates:\n- "
            + "\n- ".join(lines)
        )
