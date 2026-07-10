"""
``hermes doctor --upstream`` diagnostic.

Pure READONLY inspection of:

  - upstream reference, tracking, ahead/behind, divergence age;
  - mutual paths between local HEAD and upstream;
  - scope health (UH1, UH2, UH3, UH4, UH5, UH9);
  - safety of a future update operation (UH10).

This module never invokes a mutating Git command. It uses an explicit
allowlist (READONLY_GIT_SUBCOMMANDS) and only ever calls ``git`` via argv
through ``subprocess.run`` (no shell), captures stdout/stderr, and has a
hard timeout. Anything that doesn't match the allowlist raises
:class:`GitCommandForbidden`.

Diagnostic-only, no network, no persistence, no fixtures on disk other
than temporary Git repos that are cleaned up by the caller (typically a
test fixture). ``run_upstream_health`` is the single entry point and
returns a serializable dict suitable for ``render_text``/``render_compact``/
``serialize_json``.

The contract is frozen by
``HERMES_DOCTOR_UPSTREAM_HEALTH_CONTRACT_FINAL_FROZEN`` — see the
docstrings on :func:`run_upstream_health` and :func:`update_safety_check`
for the gating rules.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Iterable, Optional


# --------------------------------------------------------------------------- #
# Git allowlist (READONLY).
# --------------------------------------------------------------------------- #

# Subcommands that may be invoked. Anything outside this set is forbidden
# at runtime — see ``_run_git``. This is the *only* surface through which
# the module touches Git; it never uses ``shell=True``.
READONLY_GIT_SUBCOMMANDS: frozenset[str] = frozenset({
    "rev-parse",
    "rev-list",
    "merge-base",
    "diff",
    "show",
    "symbolic-ref",
    "config",
    "remote",
    "status",
})

# Subcommands explicitly forbidden even if someone extends the allowlist
# by accident (defense in depth).
FORBIDDEN_GIT_SUBCOMMANDS: frozenset[str] = frozenset({
    "fetch",
    "pull",
    "merge",
    "rebase",
    "checkout",
    "switch",
    "reset",
    "stash",
    "push",
    "clean",
    "update-ref",
    "submodule",
    "am",
    "apply",
    "cherry-pick",
    "revert",
    "rm",
})

GIT_COMMAND_TIMEOUT_SECONDS: int = 10


class GitCommandForbidden(RuntimeError):
    """Raised when a Git subcommand outside the allowlist is requested."""


class GitCallError(RuntimeError):
    """Raised when Git returns a non-zero exit or times out."""

    def __init__(self, message: str, *, returncode: int | None = None,
                 stderr: str = "", argv: tuple[str, ...] = ()):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr
        self.argv = argv

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "git_error",
            "message": str(self),
            "returncode": self.returncode,
            "stderr": self.stderr,
            "argv": list(self.argv),
        }


def _run_git(args: Iterable[str], *, cwd: Optional[str] = None,
             timeout: Optional[int] = None) -> str:
    """Run a Git subcommand from the allowlist, return stdout.

    Never raises for normal Git failures — those raise :class:`GitCallError`
    with structured fields. Forbidden subcommands raise
    :class:`GitCommandForbidden`.
    """
    argv = ("git",) + tuple(args)
    if len(argv) < 2:
        raise GitCallError("empty git argv", argv=argv)
    subcommand = argv[1]
    if subcommand in FORBIDDEN_GIT_SUBCOMMANDS:
        raise GitCommandForbidden(
            f"git subcommand '{subcommand}' is forbidden in READONLY mode"
        )
    if subcommand not in READONLY_GIT_SUBCOMMANDS:
        raise GitCommandForbidden(
            f"git subcommand '{subcommand}' not in allowlist: "
            f"{sorted(READONLY_GIT_SUBCOMMANDS)}"
        )
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout or GIT_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as e:
        raise GitCallError(
            f"git {subcommand} timed out after {timeout or GIT_COMMAND_TIMEOUT_SECONDS}s",
            argv=argv,
        ) from e
    except FileNotFoundError as e:
        raise GitCallError("git binary not found", argv=argv) from e
    if completed.returncode != 0:
        raise GitCallError(
            f"git {subcommand} failed (rc={completed.returncode})",
            returncode=completed.returncode,
            stderr=completed.stderr.strip(),
            argv=argv,
        )
    return completed.stdout


def _shell_quote(value: str) -> str:
    """Render a shell-quoted command for diagnostic display only — never
    passed to a shell."""
    return shlex.join([value]) if value else "''"


# --------------------------------------------------------------------------- #
# Enums.
# --------------------------------------------------------------------------- #


class BranchHealth(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    ERROR = "ERROR"


class UpdateSafetyDecision(str, Enum):
    UPDATE_SAFETY_PASS = "UPDATE_SAFETY_PASS"
    UPDATE_SAFETY_BLOCKED = "UPDATE_SAFETY_BLOCKED"


class UpdateBehavior(str, Enum):
    PULL_FF_ONLY = "PULL_FF_ONLY"
    PULL_FF_ONLY_PLUS_RESET_HARD_FALLBACK = "PULL_FF_ONLY_PLUS_RESET_HARD_FALLBACK"
    PULL_REBASE = "PULL_REBASE"
    PULL_MERGE = "PULL_MERGE"
    CHECKOUT_THEN_PULL_FF_ONLY = "CHECKOUT_THEN_PULL_FF_ONLY"
    CHECKOUT_THEN_PULL_FF_ONLY_PLUS_RESET_HARD_FALLBACK = (
        "CHECKOUT_THEN_PULL_FF_ONLY_PLUS_RESET_HARD_FALLBACK"
    )
    UNKNOWN = "UNKNOWN"


# --------------------------------------------------------------------------- #
# Dataclasses — frozen.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class UpdateBehaviorProfile:
    """Orthogonal properties of the real Hermes update behavior."""

    name: UpdateBehavior
    implicit_branch_switch: bool
    autostash: bool
    hard_rollback_on_syntax_failure: bool
    gateway_auto_restart: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "implicit_branch_switch": self.implicit_branch_switch,
            "autostash": self.autostash,
            "hard_rollback_on_syntax_failure": self.hard_rollback_on_syntax_failure,
            "gateway_auto_restart": self.gateway_auto_restart,
        }


# The behavior Hermes *actually* exhibits today, as established by the
# contract freeze. This is the canonical profile passed into
# ``update_safety_check`` when a profile isn't supplied (e.g. from tests
# that need to inject a different one).
CURRENT_UPDATE_BEHAVIOR = UpdateBehaviorProfile(
    name=UpdateBehavior.PULL_FF_ONLY_PLUS_RESET_HARD_FALLBACK,
    implicit_branch_switch=True,
    autostash=True,
    hard_rollback_on_syntax_failure=True,
    gateway_auto_restart=True,
)


@dataclass(frozen=True)
class UpstreamReference:
    resolved: bool
    ref: Optional[str]
    remote: Optional[str]
    branch: Optional[str]
    resolution_chain: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass(frozen=True)
class TrackingInfo:
    has_upstream: bool
    upstream_ref: Optional[str]
    remote: Optional[str]
    merge_ref: Optional[str]
    resolution: str  # "explicit" | "config_fallback" | "none"
    publish_ref: Optional[str] = None
    published_ahead: int = 0
    published_behind: int = 0
    fully_published: bool = False


@dataclass(frozen=True)
class AheadBehind:
    ahead: int
    behind: int


@dataclass(frozen=True)
class DivergenceInfo:
    merge_base_sha: Optional[str]
    merge_base_timestamp: Optional[int]
    head_timestamp: Optional[int]
    oldest_unique_commit_sha: Optional[str]
    oldest_unique_commit_timestamp: Optional[int]
    divergence_age_days: Optional[int]


@dataclass(frozen=True)
class MutualPaths:
    local_paths: list[str]
    upstream_paths: list[str]
    mutual_paths: list[str]
    critical_mutual_paths: list[str]


@dataclass(frozen=True)
class ScopeHealth:
    unique_local_commits: int
    changed_files: int
    insertions: int
    deletions: int


@dataclass(frozen=True)
class BranchHealthReport:
    """Result of :func:`collect_branch_health`."""

    branch: str
    head_sha: str
    head_short: str
    repo_root: str
    health: BranchHealth
    reasons: list[str]
    upstream: UpstreamReference
    tracking: TrackingInfo
    ahead_behind: AheadBehind
    divergence: DivergenceInfo
    mutual: MutualPaths
    scope: ScopeHealth
    raw_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "head_sha": self.head_sha,
            "head_short": self.head_short,
            "repo_root": self.repo_root,
            "health": self.health.value,
            "reasons": list(self.reasons),
            "raw_error": self.raw_error,
            "upstream": {
                "resolved": self.upstream.resolved,
                "ref": self.upstream.ref,
                "remote": self.upstream.remote,
                "branch": self.upstream.branch,
                "resolution_chain": list(self.upstream.resolution_chain),
                "error": self.upstream.error,
            },
            "canonical_upstream": {
                "resolved": self.upstream.resolved,
                "ref": self.upstream.ref,
                "remote": self.upstream.remote,
                "branch": self.upstream.branch,
                "resolution_chain": list(self.upstream.resolution_chain),
                "error": self.upstream.error,
            },
            "tracking": {
                "has_upstream": self.tracking.has_upstream,
                "upstream_ref": self.tracking.upstream_ref,
                "publish_ref": self.tracking.publish_ref,
                "remote": self.tracking.remote,
                "merge_ref": self.tracking.merge_ref,
                "resolution": self.tracking.resolution,
                "published_ahead": self.tracking.published_ahead,
                "published_behind": self.tracking.published_behind,
                "fully_published": self.tracking.fully_published,
            },
            "ahead_behind": {
                "ahead": self.ahead_behind.ahead,
                "behind": self.ahead_behind.behind,
            },
            "divergence": {
                "merge_base_sha": self.divergence.merge_base_sha,
                "merge_base_timestamp": self.divergence.merge_base_timestamp,
                "head_timestamp": self.divergence.head_timestamp,
                "oldest_unique_commit_sha": self.divergence.oldest_unique_commit_sha,
                "oldest_unique_commit_timestamp": (
                    self.divergence.oldest_unique_commit_timestamp
                ),
                "divergence_age_days": self.divergence.divergence_age_days,
            },
            "mutual": {
                "local_paths": list(self.mutual.local_paths),
                "upstream_paths": list(self.mutual.upstream_paths),
                "mutual_paths": list(self.mutual.mutual_paths),
                "critical_mutual_paths": list(self.mutual.critical_mutual_paths),
            },
            "scope": {
                "unique_local_commits": self.scope.unique_local_commits,
                "changed_files": self.scope.changed_files,
                "insertions": self.scope.insertions,
                "deletions": self.scope.deletions,
            },
        }


@dataclass(frozen=True)
class UpdateSafetyReport:
    """Result of :func:`update_safety_check`."""

    decision: UpdateSafetyDecision
    requires_manual_confirmation: bool
    confirmation_reason: Optional[str]
    behavior_name: UpdateBehavior
    behavior_profile: UpdateBehaviorProfile
    reasoning: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "requires_manual_confirmation": self.requires_manual_confirmation,
            "confirmation_reason": self.confirmation_reason,
            "behavior_name": self.behavior_name.value,
            "behavior_profile": self.behavior_profile.to_dict(),
            "reasoning": list(self.reasoning),
        }


@dataclass(frozen=True)
class UpstreamHealthResult:
    branch_health: BranchHealthReport
    update_safety: UpdateSafetyReport
    exit_code: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_health": self.branch_health.to_dict(),
            "update_safety": self.update_safety.to_dict(),
            "exit_code": self.exit_code,
        }


# --------------------------------------------------------------------------- #
# Critical paths (heuristic, conservative). Mutual path on any of these
# surfaces but does not itself cause BLOCKED — only the update_safety
# rules can produce BLOCKED.
# --------------------------------------------------------------------------- #


CRITICAL_PATH_FRAGMENTS: tuple[str, ...] = (
    "hermes_cli/main.py",
    "hermes_cli/config.py",
    "cli.py",
    "run_agent.py",
    "model_tools.py",
    "toolsets.py",
    "hermes_constants.py",
    "hermes_cli/__init__.py",
)


def _classify_critical_paths(mutual: list[str]) -> list[str]:
    out: list[str] = []
    for path in mutual:
        for fragment in CRITICAL_PATH_FRAGMENTS:
            if fragment in path:
                out.append(path)
                break
    return out


# --------------------------------------------------------------------------- #
# Scope-thresholds (UH9, frozen by contract).
# --------------------------------------------------------------------------- #


SCOPE_PASS_MAX_COMMITS = 5
SCOPE_PASS_MAX_FILES = 20
SCOPE_WARN_MAX_COMMITS = 20
SCOPE_WARN_MAX_FILES = 100


def _classify_scope(scope: ScopeHealth) -> BranchHealth:
    if scope.unique_local_commits > SCOPE_WARN_MAX_COMMITS or scope.changed_files > SCOPE_WARN_MAX_FILES:
        return BranchHealth.ERROR
    if (scope.unique_local_commits > SCOPE_PASS_MAX_COMMITS
            or scope.changed_files > SCOPE_PASS_MAX_FILES):
        return BranchHealth.WARN
    return BranchHealth.PASS


# --------------------------------------------------------------------------- #
# Git collection.
# --------------------------------------------------------------------------- #


def _safe_stdout(fn, *args, **kwargs) -> tuple[Optional[str], Optional[str]]:
    """Call ``fn`` (a ``_run_git``-style callable) returning (stdout_or_None, error_or_None)."""
    try:
        return fn(*args, **kwargs), None
    except (GitCallError, GitCommandForbidden) as e:
        return None, str(e)


def _detect_repo_root(cwd: Optional[str] = None) -> Optional[str]:
    out, err = _safe_stdout(_run_git, ("rev-parse", "--show-toplevel"), cwd=cwd)
    if err or out is None:
        return None
    return out.strip()


def _resolve_head(cwd: Optional[str] = None) -> tuple[str, str]:
    head_sha = _run_git(("rev-parse", "HEAD"), cwd=cwd).strip()
    head_short = _run_git(("rev-parse", "--short", "HEAD"), cwd=cwd).strip()
    return head_sha, head_short


def _resolve_branch(cwd: Optional[str] = None) -> str:
    out, err = _safe_stdout(_run_git, ("symbolic-ref", "--short", "HEAD"), cwd=cwd)
    if err or out is None:
        # detached HEAD
        return "HEAD"
    return out.strip() or "HEAD"


def _resolve_upstream_reference(
    *, branch: str, cwd: Optional[str] = None,
    canonical_upstream_ref: Optional[str] = None,
) -> UpstreamReference:
    """Resolve the canonical repository upstream ref.

    Order:
      1) explicit ``canonical_upstream_ref`` argument (tests/internal callers)
      2) ``refs/remotes/origin/HEAD``
      3) ``refs/remotes/origin/main``
      4) ``refs/remotes/origin/master``
      5) structured error

    The branch's ``@{upstream}`` is publish tracking and is intentionally
    resolved separately by :func:`_resolve_tracking`; it must not replace the
    canonical upstream when a feature branch tracks a fork.
    """
    _ = branch  # kept for API symmetry and future branch-specific checks.
    chain: list[str] = []

    if canonical_upstream_ref:
        chain.append("argument")
        out, err = _safe_stdout(
            _run_git,
            ("rev-parse", "--verify", "--quiet", canonical_upstream_ref),
            cwd=cwd,
        )
        if not err and out:
            remote = canonical_upstream_ref.split("/", 1)[0] if "/" in canonical_upstream_ref else None
            branch_name = canonical_upstream_ref.split("/", 1)[1] if "/" in canonical_upstream_ref else canonical_upstream_ref
            return UpstreamReference(
                resolved=True,
                ref=canonical_upstream_ref,
                remote=remote,
                branch=branch_name,
                resolution_chain=chain,
            )
        return UpstreamReference(
            resolved=False,
            ref=None,
            remote=None,
            branch=None,
            resolution_chain=chain,
            error=f"canonical upstream reference not found: {canonical_upstream_ref}",
        )

    # Try known canonical remote-tracking refs.
    for hint in (
        "refs/remotes/origin/HEAD",
        "refs/remotes/origin/main",
        "refs/remotes/origin/master",
    ):
        chain.append(hint)
        if hint.endswith("/HEAD"):
            out, err = _safe_stdout(_run_git, ("symbolic-ref", "--quiet", "--short", hint), cwd=cwd)
            if not err and out:
                ref = out.strip()
                if ref:
                    return UpstreamReference(
                        resolved=True,
                        ref=ref,
                        remote=ref.split("/", 1)[0] if "/" in ref else "origin",
                        branch=ref.split("/", 1)[1] if "/" in ref else ref,
                        resolution_chain=chain,
                    )
            # Local test repos and some clones do not materialize
            # origin/HEAD. Fall through to origin/main/master.
            continue

        out, err = _safe_stdout(_run_git, ("rev-parse", "--verify", "--quiet", hint), cwd=cwd)
        if not err and out:
            ref = hint[len("refs/remotes/"):]
            return UpstreamReference(
                resolved=True,
                ref=ref,
                remote="origin",
                branch=ref.split("/", 1)[1] if "/" in ref else ref,
                resolution_chain=chain,
            )

    return UpstreamReference(
        resolved=False,
        ref=None,
        remote=None,
        branch=None,
        resolution_chain=chain,
        error="canonical upstream reference not found",
    )


def _resolve_tracking(branch: str, cwd: Optional[str] = None) -> TrackingInfo:
    out, err = _safe_stdout(
        _run_git,
        ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"),
        cwd=cwd,
    )
    if not err and out and not out.strip().endswith("@{upstream}"):
        ref = out.strip()
        remote = ref.split("/", 1)[0] if "/" in ref else None
        return TrackingInfo(
            has_upstream=True,
            upstream_ref=ref,
            remote=remote,
            merge_ref=ref,
            resolution="explicit",
        )

    if branch == "HEAD":
        return TrackingInfo(
            has_upstream=False,
            upstream_ref=None,
            remote=None,
            merge_ref=None,
            resolution="none",
        )

    remote_out, remote_err = _safe_stdout(
        _run_git, ("config", "--get", f"branch.{branch}.remote"), cwd=cwd
    )
    merge_out, merge_err = _safe_stdout(
        _run_git, ("config", "--get", f"branch.{branch}.merge"), cwd=cwd
    )
    if remote_out and merge_out and not remote_err and not merge_err:
        remote = remote_out.strip()
        merge_ref = merge_out.strip()
        if merge_ref.startswith("refs/heads/"):
            ref = f"{remote}/{merge_ref[len('refs/heads/'):]}"
        else:
            ref = f"{remote}/{merge_ref}"
        return TrackingInfo(
            has_upstream=True,
            upstream_ref=ref,
            remote=remote,
            merge_ref=merge_ref,
            resolution="config_fallback",
        )

    return TrackingInfo(
        has_upstream=False,
        upstream_ref=None,
        remote=None,
        merge_ref=None,
        resolution="none",
    )


def _with_publish_metrics(tracking: TrackingInfo, *, cwd: Optional[str] = None) -> TrackingInfo:
    """Attach publication metrics computed against ``@{upstream}`` only."""
    publish_ref = tracking.upstream_ref if tracking.has_upstream else None
    published = _ahead_behind(cwd=cwd, upstream_ref=publish_ref)
    return TrackingInfo(
        has_upstream=tracking.has_upstream,
        upstream_ref=tracking.upstream_ref,
        remote=tracking.remote,
        merge_ref=tracking.merge_ref,
        resolution=tracking.resolution,
        publish_ref=publish_ref,
        published_ahead=published.ahead,
        published_behind=published.behind,
        fully_published=tracking.has_upstream and published.ahead == 0,
    )


def _working_tree_clean(cwd: Optional[str]) -> bool:
    out, err = _safe_stdout(_run_git, ("status", "--porcelain"), cwd=cwd)
    return not err and out is not None and out.strip() == ""


def _ahead_behind(cwd: Optional[str], upstream_ref: Optional[str]) -> AheadBehind:
    if not upstream_ref:
        return AheadBehind(ahead=0, behind=0)
    out, err = _safe_stdout(
        _run_git,
        ("rev-list", "--left-right", "--count", f"HEAD...{upstream_ref}"),
        cwd=cwd,
    )
    if err or out is None:
        return AheadBehind(ahead=0, behind=0)
    parts = out.strip().split()
    if len(parts) != 2:
        return AheadBehind(ahead=0, behind=0)
    try:
        return AheadBehind(ahead=int(parts[0]), behind=int(parts[1]))
    except ValueError:
        return AheadBehold_invalid()  # type: ignore[return-value]


def AheadBehold_invalid() -> AheadBehind:  # pragma: no cover - defensive
    return AheadBehind(ahead=0, behind=0)


def _merge_base(cwd: Optional[str], upstream_ref: Optional[str]) -> Optional[str]:
    if not upstream_ref:
        return None
    out, err = _safe_stdout(
        _run_git, ("merge-base", "HEAD", upstream_ref), cwd=cwd
    )
    if err or out is None:
        return None
    return out.strip() or None


def _timestamp_for(sha: Optional[str], cwd: Optional[str]) -> Optional[int]:
    if not sha:
        return None
    out, err = _safe_stdout(
        _run_git,
        ("show", "-s", "--format=%ct", sha),
        cwd=cwd,
    )
    if err or out is None:
        return None
    try:
        return int(out.strip())
    except ValueError:
        return None


def _oldest_unique_commit(
    cwd: Optional[str], merge_base_sha: Optional[str], head_sha: Optional[str]
) -> tuple[Optional[str], Optional[int]]:
    """Oldest commit reachable from HEAD after the canonical merge base."""
    if not merge_base_sha or not head_sha:
        return None, None
    if merge_base_sha == head_sha:
        return None, None
    out, err = _safe_stdout(
        _run_git, ("rev-list", "--reverse", f"{merge_base_sha}..HEAD"), cwd=cwd
    )
    if err or out is None:
        return None, None
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines:
        return None, None
    oldest_sha = lines[0]
    ts = _timestamp_for(oldest_sha, cwd=cwd)
    return oldest_sha, ts


def _divergence_info(cwd: Optional[str], upstream_ref: Optional[str]) -> DivergenceInfo:
    merge_base_sha = _merge_base(cwd, upstream_ref)
    head_out, head_err = _safe_stdout(_run_git, ("rev-parse", "HEAD"), cwd=cwd)
    head_sha = head_out.strip() if not head_err and head_out else None
    head_ts = _timestamp_for("HEAD", cwd=cwd)
    merge_base_ts = _timestamp_for(merge_base_sha, cwd=cwd)
    oldest_sha, oldest_ts = _oldest_unique_commit(cwd, merge_base_sha, head_sha)

    age_days: Optional[int] = None
    if merge_base_sha and head_sha and merge_base_sha == head_sha:
        age_days = 0
    elif head_ts is not None and oldest_ts is not None:
        diff_seconds = max(0, head_ts - oldest_ts)
        age_days = diff_seconds // 86_400

    return DivergenceInfo(
        merge_base_sha=merge_base_sha,
        merge_base_timestamp=merge_base_ts,
        head_timestamp=head_ts,
        oldest_unique_commit_sha=oldest_sha,
        oldest_unique_commit_timestamp=oldest_ts,
        divergence_age_days=age_days,
    )


def _diff_name_only_range(cwd: Optional[str], revision_range: str) -> list[str]:
    out, err = _safe_stdout(
        _run_git, ("diff", "--name-only", revision_range), cwd=cwd
    )
    if err or out is None:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _diff_name_only(cwd: Optional[str], ref1: str, ref2: str) -> list[str]:
    return _diff_name_only_range(cwd, f"{ref1}..{ref2}")


def _shortstat(cwd: Optional[str], ref: Optional[str]) -> tuple[int, int, int]:
    """Return (changed_files, insertions, deletions) for the diff vs ``ref``."""
    if not ref:
        return 0, 0, 0
    out, err = _safe_stdout(
        _run_git, ("diff", "--shortstat", f"{ref}...HEAD"), cwd=cwd
    )
    if err or out is None:
        return 0, 0, 0
    changed = insertions = deletions = 0
    for chunk in out.strip().split(","):
        chunk = chunk.strip()
        if "file" in chunk:
            try:
                changed = int(chunk.split()[0])
            except (ValueError, IndexError):
                pass
        elif "insertion" in chunk:
            try:
                insertions = int(chunk.split()[0])
            except (ValueError, IndexError):
                pass
        elif "deletion" in chunk:
            try:
                deletions = int(chunk.split()[0])
            except (ValueError, IndexError):
                pass
    return changed, insertions, deletions


def _rev_count(cwd: Optional[str], ref1: str, ref2: str) -> int:
    out, err = _safe_stdout(
        _run_git, ("rev-list", "--count", f"{ref1}..{ref2}"), cwd=cwd
    )
    if err or out is None:
        return 0
    try:
        return int(out.strip())
    except ValueError:
        return 0


def _mutual_paths(cwd: Optional[str], upstream_ref: Optional[str],
                  merge_base_sha: Optional[str]) -> MutualPaths:
    local = _diff_name_only(cwd, merge_base_sha or "HEAD", "HEAD") if merge_base_sha else []
    upstream = (
        _diff_name_only(cwd, merge_base_sha or upstream_ref, upstream_ref)
        if (merge_base_sha and upstream_ref)
        else []
    )
    mutual = sorted(set(local) & set(upstream))
    critical = _classify_critical_paths(mutual)
    return MutualPaths(
        local_paths=local,
        upstream_paths=upstream,
        mutual_paths=mutual,
        critical_mutual_paths=critical,
    )


def _scope_health(cwd: Optional[str], upstream_ref: Optional[str]) -> ScopeHealth:
    if not upstream_ref:
        return ScopeHealth(0, 0, 0, 0)
    unique_local = _rev_count(cwd, upstream_ref, "HEAD")
    changed_paths = _diff_name_only_range(cwd, f"{upstream_ref}...HEAD")
    shortstat_changed, ins, dele = _shortstat(cwd, upstream_ref)
    changed = len(changed_paths) if changed_paths else shortstat_changed
    return ScopeHealth(
        unique_local_commits=unique_local,
        changed_files=changed,
        insertions=ins,
        deletions=dele,
    )


# --------------------------------------------------------------------------- #
# BranchHealth classification.
# --------------------------------------------------------------------------- #


def classify_branch_health(
    *,
    upstream: UpstreamReference,
    tracking: TrackingInfo,
    ahead_behind: AheadBehind,
    mutual: MutualPaths,
    scope: ScopeHealth,
) -> BranchHealth:
    """Roll up the BranchHealth verdict from UH1–UH5, UH9.

    Contract:
      - missing upstream reference -> ERROR (UH1)
      - no tracking on a feature branch with unique commits -> ERROR
      - behind > 0 is not by itself a problem (UH3)
      - divergence days > 30 is WARN, never ERROR
      - mutual critical paths is WARN (UH5), never BLOCKED/ERROR
      - scope (UH9) follows SCOPE_*_MAX_* thresholds
    """
    if not upstream.resolved:
        return BranchHealth.ERROR
    if not tracking.has_upstream and scope.unique_local_commits > 0:
        return BranchHealth.ERROR
    if scope.unique_local_commits > SCOPE_WARN_MAX_COMMITS or scope.changed_files > SCOPE_WARN_MAX_FILES:
        return BranchHealth.ERROR
    if (scope.unique_local_commits > SCOPE_PASS_MAX_COMMITS
            or scope.changed_files > SCOPE_PASS_MAX_FILES):
        return BranchHealth.WARN
    if mutual.critical_mutual_paths:
        return BranchHealth.WARN
    return BranchHealth.PASS


# --------------------------------------------------------------------------- #
# Public collectors.
# --------------------------------------------------------------------------- #


def collect_branch_health(
    *, cwd: Optional[str] = None,
    repo_root: Optional[str] = None,
    canonical_upstream_ref: Optional[str] = None,
) -> BranchHealthReport:
    """Read-only collection of BranchHealth components UH1–UH5 + UH9."""
    reasons: list[str] = []
    raw_error: Optional[str] = None

    repo = repo_root or _detect_repo_root(cwd=cwd)
    if not repo:
        raw_error = "not inside a git working tree"
        # Synthesize empty defaults so the typed result still renders.
        empty = BranchHealthReport(
            branch="HEAD",
            head_sha="",
            head_short="",
            repo_root=cwd or "",
            health=BranchHealth.ERROR,
            reasons=["not inside a git working tree"],
            upstream=UpstreamReference(False, None, None, None, error=raw_error),
            tracking=TrackingInfo(False, None, None, None, "none"),
            ahead_behind=AheadBehind(0, 0),
            divergence=DivergenceInfo(None, None, None, None, None, None),
            mutual=MutualPaths([], [], [], []),
            scope=ScopeHealth(0, 0, 0, 0),
            raw_error=raw_error,
        )
        return empty

    try:
        branch = _resolve_branch(cwd=repo)
        head_sha, head_short = _resolve_head(cwd=repo)
        upstream = _resolve_upstream_reference(
            branch=branch,
            cwd=repo,
            canonical_upstream_ref=canonical_upstream_ref,
        )
        tracking = _with_publish_metrics(_resolve_tracking(branch=branch, cwd=repo), cwd=repo)
        # UH1 collect
        if not upstream.resolved:
            reasons.append("UH1: canonical upstream reference unresolved")
        # UH3 ahead/behind only when we have a canonical upstream
        effective_upstream = upstream.ref if upstream.resolved else None
        ab = _ahead_behind(cwd=repo, upstream_ref=effective_upstream)
        # UH4 divergence
        divergence = _divergence_info(cwd=repo, upstream_ref=effective_upstream)
        merge_base_sha = divergence.merge_base_sha
        # UH5 mutual paths
        mutual = _mutual_paths(cwd=repo, upstream_ref=effective_upstream,
                               merge_base_sha=merge_base_sha)
        # UH9 scope
        scope = _scope_health(cwd=repo, upstream_ref=effective_upstream)
        health = classify_branch_health(
            upstream=upstream,
            tracking=tracking,
            ahead_behind=ab,
            mutual=mutual,
            scope=scope,
        )
        # Surface readable reasons that don't change the verdict:
        if not tracking.has_upstream and branch != "HEAD":
            reasons.append("UH2: no upstream tracking for branch")
        if ab.behind > 0:
            reasons.append(f"UH3: local behind upstream by {ab.behind} commit(s)")
        if divergence.divergence_age_days is not None and divergence.divergence_age_days > 30:
            reasons.append(
                f"UH4: divergence age {divergence.divergence_age_days} day(s)"
            )
        if mutual.critical_mutual_paths:
            reasons.append(
                "UH5: critical mutual paths touched "
                f"({len(mutual.critical_mutual_paths)} path(s))"
            )
        if health == BranchHealth.PASS and (
            scope.unique_local_commits > SCOPE_PASS_MAX_COMMITS / 2 or scope.changed_files > SCOPE_PASS_MAX_FILES / 2
        ):
            reasons.append("UH9: scope approaching thresholds")
        return BranchHealthReport(
            health=health,
            reasons=reasons,
            upstream=upstream,
            tracking=tracking,
            ahead_behind=ab,
            divergence=divergence,
            mutual=mutual,
            scope=scope,
            branch=branch,
            head_sha=head_sha,
            head_short=head_short,
            repo_root=repo,
        )
    except (GitCallError, GitCommandForbidden) as e:
        raw_error = str(e)
        return BranchHealthReport(
            branch="HEAD",
            head_sha="",
            head_short="",
            repo_root=repo or (cwd or ""),
            health=BranchHealth.ERROR,
            reasons=[f"git collection failed: {raw_error}"],
            upstream=UpstreamReference(False, None, None, None, error=raw_error),
            tracking=TrackingInfo(False, None, None, None, "none"),
            ahead_behind=AheadBehind(0, 0),
            divergence=DivergenceInfo(None, None, None, None, None, None),
            mutual=MutualPaths([], [], [], []),
            scope=ScopeHealth(0, 0, 0, 0),
            raw_error=raw_error,
        )


# --------------------------------------------------------------------------- #
# Update safety (UH10).
# --------------------------------------------------------------------------- #


def update_safety_check(
    branch_health: BranchHealthReport,
    *,
    behavior: UpdateBehaviorProfile = CURRENT_UPDATE_BEHAVIOR,
    is_published_clean_feature: bool = False,
    is_untracked_target_with_unique_commits: bool = False,
) -> UpdateSafetyReport:
    """Pure decision over ``branch_health`` + behavior profile.

    Rules frozen by the contract:

      A. Same branch, synced/behind-only          -> PASS
      B. Same branch, ahead-only                  -> PASS w/ confirmation
      C. Same branch, diverged + reset fallback   -> BLOCKED
      D. Feature branch distinct from target
         with unique commits                      -> BLOCKED
      E. Published clean feature, ahead=0
         and clean tree                           -> PASS w/ confirmation
      F. No tracking + unique commits             -> BranchHealth.ERROR;
         only BLOCKED when the specific operation
         could abandon or reset work.
      G. Target local diverged + reset fallback   -> BLOCKED

    Mutual paths never directly cause BLOCKED. WARN never changes exit
    code. ``requires_manual_confirmation`` is True whenever a PASS
    result is given for a non-trivial ahead-only or branch-switch case.
    """
    reasoning: list[str] = []
    ahead = branch_health.ahead_behind.ahead
    behind = branch_health.ahead_behind.behind
    behind_only = ahead == 0 and behind > 0
    ahead_only = ahead > 0 and behind == 0
    diverged = ahead > 0 and behind > 0

    # Reset-on-failure blocklist: only when the actual behavior profile
    # would hard-reset.
    reset_in_play = (
        behavior.name in (
            UpdateBehavior.PULL_FF_ONLY_PLUS_RESET_HARD_FALLBACK,
            UpdateBehavior.CHECKOUT_THEN_PULL_FF_ONLY_PLUS_RESET_HARD_FALLBACK,
        )
    )

    # Rule F: no tracking + unique commits -> ERROR already captured by
    # BranchHealth; we still need to BLOCKED because the operation can
    # reset work.
    if (not branch_health.tracking.has_upstream
            and branch_health.scope.unique_local_commits > 0
            and reset_in_play):
        reasoning.append(
            "Rule F: no tracking and reset fallback — operation could "
            "abandon untracked work"
        )
        return UpdateSafetyReport(
            decision=UpdateSafetyDecision.UPDATE_SAFETY_BLOCKED,
            requires_manual_confirmation=False,
            confirmation_reason=None,
            behavior_name=behavior.name,
            behavior_profile=behavior,
            reasoning=reasoning,
        )

    # Rule E: published clean feature, ahead=0.
    if is_published_clean_feature and ahead == 0 and behind == 0:
        reasoning.append("Rule E: published clean feature — pass with confirmation")
        return UpdateSafetyReport(
            decision=UpdateSafetyDecision.UPDATE_SAFETY_PASS,
            requires_manual_confirmation=True,
            confirmation_reason="implicit branch switch on published clean feature",
            behavior_name=behavior.name,
            behavior_profile=behavior,
            reasoning=reasoning,
        )

    # Case A: synced / behind-only.
    if ahead == 0 and behind >= 0 and branch_health.health != BranchHealth.ERROR:
        reasoning.append("Rule A: branch synced or behind-only — safe to pull")
        return UpdateSafetyReport(
            decision=UpdateSafetyDecision.UPDATE_SAFETY_PASS,
            requires_manual_confirmation=False,
            confirmation_reason=None,
            behavior_name=behavior.name,
            behavior_profile=behavior,
            reasoning=reasoning,
        )

    # Case B: ahead-only.
    if ahead_only:
        if branch_health.health == BranchHealth.ERROR:
            reasoning.append("Rule B suppressed by BranchHealth.ERROR")
            return UpdateSafetyReport(
                decision=UpdateSafetyDecision.UPDATE_SAFETY_BLOCKED,
                requires_manual_confirmation=False,
                confirmation_reason=None,
                behavior_name=behavior.name,
                behavior_profile=behavior,
                reasoning=reasoning,
            )
        reasoning.append("Rule B: branch ahead of upstream — pass with confirmation")
        return UpdateSafetyReport(
            decision=UpdateSafetyDecision.UPDATE_SAFETY_PASS,
            requires_manual_confirmation=True,
            confirmation_reason="local commits not yet on upstream; "
            "fast-forward will replay them",
            behavior_name=behavior.name,
            behavior_profile=behavior,
            reasoning=reasoning,
        )

    # Case C: diverged + reset fallback.
    if diverged and reset_in_play:
        reasoning.append(
            "Rule C: diverged histories and reset fallback — could drop work"
        )
        return UpdateSafetyReport(
            decision=UpdateSafetyDecision.UPDATE_SAFETY_BLOCKED,
            requires_manual_confirmation=False,
            confirmation_reason=None,
            behavior_name=behavior.name,
            behavior_profile=behavior,
            reasoning=reasoning,
        )

    # Case D: feature branch distinct from target with unique commits.
    if (branch_health.tracking.has_upstream
            and branch_health.branch != "HEAD"
            and branch_health.scope.unique_local_commits > 0
            and not is_published_clean_feature):
        # Implicit branch switch + reset fallback would lose state — block.
        if behavior.implicit_branch_switch and reset_in_play:
            reasoning.append(
                "Rule D: unique local commits on a feature branch with "
                "implicit branch-switch and reset fallback"
            )
            return UpdateSafetyReport(
                decision=UpdateSafetyDecision.UPDATE_SAFETY_BLOCKED,
                requires_manual_confirmation=False,
                confirmation_reason=None,
                behavior_name=behavior.name,
                behavior_profile=behavior,
                reasoning=reasoning,
            )
        # Otherwise, non-reset branches could still proceed safely.
        reasoning.append("Rule D alt: behavior profile is non-resetting; safe to pull")
        return UpdateSafetyReport(
            decision=UpdateSafetyDecision.UPDATE_SAFETY_PASS,
            requires_manual_confirmation=True,
            confirmation_reason="feature branch will be switched implicitly",
            behavior_name=behavior.name,
            behavior_profile=behavior,
            reasoning=reasoning,
        )

    # Case G: target local diverged + reset fallback.
    if diverged and reset_in_play and behind > 0:
        reasoning.append("Rule G: diverged with reset fallback — BLOCKED")
        return UpdateSafetyReport(
            decision=UpdateSafetyDecision.UPDATE_SAFETY_BLOCKED,
            requires_manual_confirmation=False,
            confirmation_reason=None,
            behavior_name=behavior.name,
            behavior_profile=behavior,
            reasoning=reasoning,
        )

    # Untracked target fallback path.
    if is_untracked_target_with_unique_commits and reset_in_play:
        reasoning.append("Untracked target with reset fallback — BLOCKED")
        return UpdateSafetyReport(
            decision=UpdateSafetyDecision.UPDATE_SAFETY_BLOCKED,
            requires_manual_confirmation=False,
            confirmation_reason=None,
            behavior_name=behavior.name,
            behavior_profile=behavior,
            reasoning=reasoning,
        )

    # Default: pass with no confirmation.
    reasoning.append("No update-safety rule triggered")
    return UpdateSafetyReport(
        decision=UpdateSafetyDecision.UPDATE_SAFETY_PASS,
        requires_manual_confirmation=False,
        confirmation_reason=None,
        behavior_name=behavior.name,
        behavior_profile=behavior,
        reasoning=reasoning,
    )


# --------------------------------------------------------------------------- #
# Exit codes.
# --------------------------------------------------------------------------- #


def aggregate_exit_code(branch_health: BranchHealthReport,
                        safety: UpdateSafetyReport) -> int:
    """Frozen exit-code rule:

        safety == BLOCKED     -> 2
        branch_health.ERROR   -> 1
        otherwise             -> 0

    WARN and ``requires_manual_confirmation`` do NOT change the exit code.
    """
    if safety.decision == UpdateSafetyDecision.UPDATE_SAFETY_BLOCKED:
        return 2
    if branch_health.health == BranchHealth.ERROR:
        return 1
    return 0


# --------------------------------------------------------------------------- #
# Renderers.
# --------------------------------------------------------------------------- #


def render_text(result: UpstreamHealthResult) -> str:
    """Human-readable, ANSI-free text mode output.

    Never echoes a mutating command. Includes summarized evidence.
    """
    bh = result.branch_health
    us = result.update_safety

    lines: list[str] = []
    lines.append("◆ Hermes Upstream Health (READONLY)")
    lines.append("")
    lines.append(f"  branch:    {bh.branch}")
    lines.append(f"  head:      {bh.head_short}")
    if bh.upstream.resolved:
        lines.append(f"  canonical_upstream: {bh.upstream.ref}")
    else:
        lines.append(f"  canonical_upstream: (unresolved — {bh.upstream.error or 'no candidate'})")
    lines.append(
        f"  tracking:  publish_ref={bh.tracking.publish_ref or '(none)'} "
        f"fully_published={'yes' if bh.tracking.fully_published else 'no'} "
        f"published_ahead/behind={bh.tracking.published_ahead}/{bh.tracking.published_behind}"
    )
    lines.append(
        f"  ahead/behind: {bh.ahead_behind.ahead}/{bh.ahead_behind.behind}"
    )
    if bh.divergence.divergence_age_days is not None:
        lines.append(
            f"  divergence age (days): {bh.divergence.divergence_age_days}"
        )
    lines.append(
        f"  scope: {bh.scope.unique_local_commits} commit(s), "
        f"{bh.scope.changed_files} file(s) changed"
    )
    if bh.mutual.mutual_paths:
        lines.append(
            f"  mutual paths: {len(bh.mutual.mutual_paths)} "
            f"(critical: {len(bh.mutual.critical_mutual_paths)})"
        )
    lines.append(f"  branch_health: {bh.health.value}")
    if bh.reasons:
        for r in bh.reasons:
            lines.append(f"    - {r}")
    lines.append(f"  update_safety: {us.decision.value}")
    lines.append(
        f"  requires_manual_confirmation: "
        f"{'yes' if us.requires_manual_confirmation else 'no'}"
    )
    if us.confirmation_reason:
        lines.append(f"  confirmation_reason: {us.confirmation_reason}")
    lines.append(
        f"  behavior: {us.behavior_name.value}"
    )
    for r in us.reasoning:
        lines.append(f"    - {r}")
    lines.append(f"  exit_code: {result.exit_code}")
    return "\n".join(lines) + "\n"


def render_compact(result: UpstreamHealthResult) -> str:
    """Single-line, ANSI-free stable format.

    Order is contract-frozen: health, safety, ahead, behind, confirmation.
    """
    bh = result.branch_health
    us = result.update_safety
    confirm = "confirm" if us.requires_manual_confirmation else "auto"
    return (
        f"upstream-health health={bh.health.value} "
        f"safety={us.decision.value} canonical={bh.upstream.ref or 'unresolved'} "
        f"publish={bh.tracking.publish_ref or 'none'} ahead={bh.ahead_behind.ahead} "
        f"behind={bh.ahead_behind.behind} confirmation={confirm} "
        f"behavior={us.behavior_name.value}"
    )


def serialize_json(result: UpstreamHealthResult) -> str:
    """Single JSON object, no ANSI, no banner, no surrounding text."""
    return json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# Top-level entry point.
# --------------------------------------------------------------------------- #


def run_upstream_health(
    *,
    cwd: Optional[str] = None,
    behavior: UpdateBehaviorProfile = CURRENT_UPDATE_BEHAVIOR,
    is_published_clean_feature: Optional[bool] = None,
    is_untracked_target_with_unique_commits: bool = False,
    canonical_upstream_ref: Optional[str] = None,
) -> UpstreamHealthResult:
    """Single READONLY entry point used by ``hermes doctor --upstream``.

    Steps:
      1) ``collect_branch_health`` (UH1–UH5, UH9) — pure Git inspection.
      2) ``update_safety_check``  (UH10) — pure decision over (1) + profile.
      3) ``aggregate_exit_code``  — frozen rule, see :func:`aggregate_exit_code`.
    """
    bh = collect_branch_health(cwd=cwd, canonical_upstream_ref=canonical_upstream_ref)
    published_clean = (
        bool(
            bh.branch != "HEAD"
            and bh.branch != (bh.upstream.branch or "")
            and bh.tracking.fully_published
            and _working_tree_clean(bh.repo_root)
        )
        if is_published_clean_feature is None
        else is_published_clean_feature
    )
    safety = update_safety_check(
        bh,
        behavior=behavior,
        is_published_clean_feature=published_clean,
        is_untracked_target_with_unique_commits=is_untracked_target_with_unique_commits,
    )
    exit_code = aggregate_exit_code(bh, safety)
    return UpstreamHealthResult(
        branch_health=bh,
        update_safety=safety,
        exit_code=exit_code,
    )


__all__ = [
    "READONLY_GIT_SUBCOMMANDS",
    "FORBIDDEN_GIT_SUBCOMMANDS",
    "GIT_COMMAND_TIMEOUT_SECONDS",
    "GitCallError",
    "GitCommandForbidden",
    "BranchHealth",
    "UpdateSafetyDecision",
    "UpdateBehavior",
    "UpdateBehaviorProfile",
    "CURRENT_UPDATE_BEHAVIOR",
    "UpstreamReference",
    "TrackingInfo",
    "AheadBehind",
    "DivergenceInfo",
    "MutualPaths",
    "ScopeHealth",
    "BranchHealthReport",
    "UpdateSafetyReport",
    "UpstreamHealthResult",
    "collect_branch_health",
    "update_safety_check",
    "aggregate_exit_code",
    "render_text",
    "render_compact",
    "serialize_json",
    "run_upstream_health",
    "classify_branch_health",
    "CRITICAL_PATH_FRAGMENTS",
    "SCOPE_PASS_MAX_COMMITS",
    "SCOPE_PASS_MAX_FILES",
    "SCOPE_WARN_MAX_COMMITS",
    "SCOPE_WARN_MAX_FILES",
]
