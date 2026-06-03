"""Ultragoal cleanup/residue policy helpers.

These helpers deliberately classify cleanup intent before any deletion.  The
controller records the resulting proof as an audit artifact; actual filesystem
removal is restricted to explicit, allowlisted post-merge cleanup paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DELETE_CANDIDATE_KINDS = {"cache", "build_output", "test_output", "temp_log", "scratch"}
PRESERVE_KINDS = {"run_artifact", "evidence", "implementation_worktree", "pr_review_workspace"}
NEVER_TOUCH_KINDS = {"secret", "env", "canonical_checkout", "run_root", "provider_state", "customer_state"}


def _reason_for_preserve(candidate: dict[str, Any]) -> str:
    if candidate.get("dirty"):
        return "dirty_worktree"
    if candidate.get("activeCwd"):
        return "active_cwd"
    if candidate.get("referencedByEvidence"):
        return "referenced_by_evidence"
    if candidate.get("openPrDependency"):
        return "open_pr_dependency"
    if candidate.get("unpushedUserBranch"):
        return "unpushed_user_branch"
    return candidate.get("reason") or "preserved_for_review_or_evidence"


def classify_cleanup_candidates(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Classify cleanup candidates into delete/preserve/never-touch actions.

    The return value is deterministic and safe to persist as read-only cleanup
    proof.  It does not delete anything.
    """
    out: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        path = str(candidate.get("path") or "")
        kind = str(candidate.get("kind") or "")
        if not path:
            continue
        if kind in NEVER_TOUCH_KINDS:
            out[path] = {"action": "never_touch", "reason": kind}
        elif candidate.get("dirty") or candidate.get("activeCwd") or candidate.get("referencedByEvidence") or candidate.get("openPrDependency") or candidate.get("unpushedUserBranch"):
            out[path] = {"action": "preserve", "reason": _reason_for_preserve(candidate)}
        elif kind in DELETE_CANDIDATE_KINDS:
            out[path] = {"action": "delete_candidate", "reason": kind}
        elif kind in PRESERVE_KINDS:
            out[path] = {"action": "preserve", "reason": _reason_for_preserve(candidate)}
        else:
            out[path] = {"action": "preserve", "reason": "unknown_kind_fail_closed"}
    return out


def _inside_allowed_roots(path: str, allowed_roots: list[str | Path] | None) -> bool:
    if not allowed_roots:
        return False
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    for root in allowed_roots:
        try:
            root_resolved = Path(root).resolve()
        except OSError:
            continue
        if resolved == root_resolved or root_resolved in resolved.parents:
            return True
    return False


def post_merge_cleanup_plan(candidates: list[dict[str, Any]], *, allowed_roots: list[str | Path] | None = None) -> dict[str, dict[str, Any]]:
    """Plan post-merge cleanup for registered implementation worktrees.

    Only clean, inactive implementation worktrees after merge/close confirmation
    are eligible for `git worktree remove`.  Active run roots, evidence-bearing
    artifacts, canonical checkouts, and review workspaces fail closed.
    """
    out: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        path = str(candidate.get("path") or "")
        kind = str(candidate.get("kind") or "")
        if not path:
            continue
        if kind in NEVER_TOUCH_KINDS or kind in {"run_root", "evidence", "run_artifact", "canonical_checkout"}:
            out[path] = {"action": "never_touch", "reason": kind}
        elif kind == "implementation_worktree" and candidate.get("mergeClosed") is True and not candidate.get("dirty") and not candidate.get("activeCwd"):
            if _inside_allowed_roots(path, allowed_roots):
                out[path] = {"action": "remove_worktree", "reason": "clean_inactive_after_merge_close"}
            else:
                out[path] = {"action": "preserve", "reason": "outside_allowed_roots"}
        elif candidate.get("dirty") or candidate.get("activeCwd"):
            out[path] = {"action": "preserve", "reason": _reason_for_preserve(candidate)}
        else:
            out[path] = {"action": "preserve", "reason": "not_post_merge_removable"}
    return out


__all__ = ["classify_cleanup_candidates", "post_merge_cleanup_plan"]
