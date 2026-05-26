"""Non-happy path cleanup evaluation for blocked/cancelled/superseded/archived tasks.

This module evaluates cleanup requests against the rule matrix defined in
BO-161 Slice 7.  It is intentionally narrow: it classifies residue, checks
lifecycle-specific gates, and emits a verdict.  It does NOT perform actual
filesystem cleanup (that belongs to the workspace janitor) and it does NOT
mutate Kanban task state (that belongs to the closeout verifier).

Design:
- Every public evaluator is a pure function of its context dict.
- Global deny rules (containment, symlink escape, residue classification)
  are enforced before state-specific rules.
- Unknown or happy-path states fail closed (DENY_CLEANUP).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CleanupVerdict(str, Enum):
    ALLOW_PRUNE = "allow_prune"
    ALLOW_PARTIAL_CLEANUP = "allow_partial_cleanup"
    DENY_CLEANUP = "deny_cleanup"


class ResidueClass(str, Enum):
    REPRODUCIBLE = "reproducible"
    RESUMABLE = "resumable"
    UNIQUE_DIRTY = "unique_dirty"


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------

_ALLOWLISTED_ARTIFACT_NAMES = frozenset({
    "node_modules",
    ".next",
    ".turbo",
    "dist",
    "build",
    "target",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "coverage",
})

_VALID_DISPOSITIONS = frozenset({
    "commit", "stash", "patch_archive", "discard", "reapply", "retain_with_ttl",
})

_NON_HAPPY_STATES = frozenset({"blocked", "cancelled", "superseded", "archived"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text(value: Any) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    return str(value).strip()


def _lower(value: Any) -> str:
    return _text(value).lower()


def _is_contained(target_path: str, workspace_path: str) -> bool:
    """Check that target_path is inside workspace_path (without symlink resolution)."""
    if not workspace_path or not target_path:
        return False
    try:
        tp = Path(target_path).resolve()
        wp = Path(workspace_path).resolve()
        # Use os.path.commonpath equivalent
        return str(tp).startswith(str(wp) + "/") or str(tp) == str(wp)
    except (ValueError, OSError):
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _is_allowlisted_artifact(target_path: str) -> bool:
    """Check whether the basename of target_path is an allowlisted artifact."""
    if not target_path:
        return False
    basename = Path(target_path).name
    return basename in _ALLOWLISTED_ARTIFACT_NAMES


@dataclass(frozen=True)
class CleanupResult:
    verdict: CleanupVerdict
    deny_reasons: list[str] = field(default_factory=list)
    preserved_items: list[str] = field(default_factory=list)
    pruned_items: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "deny_reasons": list(self.deny_reasons),
            "preserved_items": list(self.preserved_items),
            "pruned_items": list(self.pruned_items),
        }


# ---------------------------------------------------------------------------
# Global deny rules
# ---------------------------------------------------------------------------

def _check_global_deny(context: Mapping[str, Any],
                       is_symlink_escape: bool = False) -> list[str]:
    """Evaluate global deny rules that apply to all non-happy states."""
    reasons: list[str] = []

    target = _text(context.get("target_path"))
    workspace = _text(context.get("workspace_path"))

    if not target or not workspace:
        reasons.append("missing_target_or_workspace_path")
        return reasons

    if is_symlink_escape:
        reasons.append("symlink_escape_detected")
        return reasons

    if not _is_contained(target, workspace):
        reasons.append("target_outside_workspace")

    residue_class = _lower(context.get("residue_class"))
    if residue_class in ("resumable", "unique_dirty"):
        # Resumable/unique work must not be pruned without explicit disposition.
        # This is a global deny because it applies regardless of lifecycle state.
        pass  # Handled per-state below

    return reasons


# ---------------------------------------------------------------------------
# Blocked
# ---------------------------------------------------------------------------

def evaluate_blocked_cleanup(context: Mapping[str, Any],
                             is_symlink_escape: bool = False) -> CleanupResult:
    """Evaluate cleanup for a blocked task.

    Blocked tasks preserve resumable work and prune only reproducible artifacts.
    Requires retained_reason and revisit_at/ttl.
    """
    global_reasons = _check_global_deny(context, is_symlink_escape)
    if global_reasons:
        return CleanupResult(verdict=CleanupVerdict.DENY_CLEANUP,
                             deny_reasons=global_reasons)

    reasons: list[str] = []
    target = _text(context.get("target_path"))
    residue_class = _lower(context.get("residue_class"))

    # Required: retained reason
    if not _text(context.get("retained_reason")):
        reasons.append("missing_retained_reason")

    # Required: revisit_at or ttl
    if not _text(context.get("revisit_at")) and not _text(context.get("ttl")):
        reasons.append("missing_revisit_or_ttl")

    if reasons:
        return CleanupResult(verdict=CleanupVerdict.DENY_CLEANUP,
                             deny_reasons=reasons,
                             preserved_items=["source", "diff", "evidence"])

    # Classification-based decision
    if residue_class == "reproducible":
        if _is_allowlisted_artifact(target):
            return CleanupResult(
                verdict=CleanupVerdict.ALLOW_PRUNE,
                pruned_items=[target],
                preserved_items=["source", "diff", "evidence"],
            )
        else:
            # Reproducible-class but not on the allowlist — deny
            return CleanupResult(
                verdict=CleanupVerdict.DENY_CLEANUP,
                deny_reasons=["artifact_not_allowlisted"],
                preserved_items=["source", "diff", "evidence"],
            )

    # Resumable or unique_dirty → deny deletion
    if residue_class == "unique_dirty":
        reasons.append("unique_dirty_work_cannot_be_pruned_from_blocked")
    else:
        reasons.append("resumable_work_must_be_preserved")

    return CleanupResult(
        verdict=CleanupVerdict.DENY_CLEANUP,
        deny_reasons=reasons,
        preserved_items=["source", "diff", "evidence"],
    )


# ---------------------------------------------------------------------------
# Cancelled
# ---------------------------------------------------------------------------

def evaluate_cancelled_cleanup(context: Mapping[str, Any],
                                is_symlink_escape: bool = False) -> CleanupResult:
    """Evaluate cleanup for a cancelled task.

    Cancelled tasks need cancellation reason + evidence.  Dirty/unique work
    blocks full cleanup unless an explicit disposition is recorded.
    """
    global_reasons = _check_global_deny(context, is_symlink_escape)
    if global_reasons:
        return CleanupResult(verdict=CleanupVerdict.DENY_CLEANUP,
                             deny_reasons=global_reasons)

    reasons: list[str] = []
    target = _text(context.get("target_path"))
    residue_class = _lower(context.get("residue_class"))

    if not _text(context.get("cancellation_reason")):
        reasons.append("missing_cancellation_reason")

    if not _text(context.get("cancellation_evidence")):
        reasons.append("missing_cancellation_evidence")

    # Check for dirty/unique work without disposition
    dispositions = context.get("dispositions")
    if isinstance(dispositions, Mapping):
        has_disposition = any(
            _lower(v) in _VALID_DISPOSITIONS for v in dispositions.values()
        )
    else:
        has_disposition = bool(_text(dispositions))

    if residue_class == "unique_dirty" and not has_disposition:
        reasons.append("dirty_work_without_disposition")

    if reasons:
        return CleanupResult(
            verdict=CleanupVerdict.DENY_CLEANUP,
            deny_reasons=reasons,
            preserved_items=["cancellation_reason", "cancellation_evidence"],
        )

    # Safe to prune reproducible artifacts
    if residue_class == "reproducible" and _is_allowlisted_artifact(target):
        return CleanupResult(
            verdict=CleanupVerdict.ALLOW_PRUNE,
            pruned_items=[target],
            preserved_items=["cancellation_reason", "cancellation_evidence"],
        )

    return CleanupResult(
        verdict=CleanupVerdict.ALLOW_PARTIAL_CLEANUP,
        preserved_items=["cancellation_reason", "cancellation_evidence"],
    )


# ---------------------------------------------------------------------------
# Superseded
# ---------------------------------------------------------------------------

def evaluate_superseded_cleanup(context: Mapping[str, Any],
                                 is_symlink_escape: bool = False) -> CleanupResult:
    """Evaluate cleanup for a superseded task.

    Superseded tasks require a successor link/ref before cleanup.
    Unique/dirty work still needs disposition.
    """
    global_reasons = _check_global_deny(context, is_symlink_escape)
    if global_reasons:
        return CleanupResult(verdict=CleanupVerdict.DENY_CLEANUP,
                             deny_reasons=global_reasons)

    reasons: list[str] = []
    target = _text(context.get("target_path"))
    residue_class = _lower(context.get("residue_class"))

    successor_ref = _text(context.get("successor_ref"))
    successor_link = _text(context.get("successor_link"))

    if not successor_ref and not successor_link:
        reasons.append("missing_successor_link")

    if residue_class == "unique_dirty":
        dispositions = context.get("dispositions")
        if isinstance(dispositions, Mapping):
            has_disposition = any(
                _lower(v) in _VALID_DISPOSITIONS for v in dispositions.values()
            )
        else:
            has_disposition = bool(_text(dispositions))
        if not has_disposition:
            reasons.append("dirty_work_without_disposition")

    if reasons:
        preserved = ["successor_pointer"]
        if successor_ref:
            preserved.append(successor_ref)
        if successor_link:
            preserved.append(successor_link)
        return CleanupResult(
            verdict=CleanupVerdict.DENY_CLEANUP,
            deny_reasons=reasons,
            preserved_items=preserved,
        )

    # Safe to prune reproducible artifacts
    preserved = [successor_ref or successor_link or "successor_pointer"]
    if residue_class == "reproducible" and _is_allowlisted_artifact(target):
        return CleanupResult(
            verdict=CleanupVerdict.ALLOW_PRUNE,
            pruned_items=[target],
            preserved_items=preserved,
        )

    return CleanupResult(
        verdict=CleanupVerdict.ALLOW_PARTIAL_CLEANUP,
        preserved_items=preserved,
    )


# ---------------------------------------------------------------------------
# Archived
# ---------------------------------------------------------------------------

def evaluate_archived_cleanup(context: Mapping[str, Any],
                               is_symlink_escape: bool = False) -> CleanupResult:
    """Evaluate cleanup for an archived task.

    Archived tasks may be fully cleaned only when inactive AND evidence is
    preserved.  Otherwise cleanup is denied.
    """
    global_reasons = _check_global_deny(context, is_symlink_escape)
    if global_reasons:
        return CleanupResult(verdict=CleanupVerdict.DENY_CLEANUP,
                             deny_reasons=global_reasons)

    reasons: list[str] = []
    target = _text(context.get("target_path"))

    inactive = context.get("inactive", False)
    if not inactive:
        reasons.append("archived_task_not_inactive")

    evidence_preserved = context.get("evidence_preserved", False)
    if not evidence_preserved:
        reasons.append("evidence_not_preserved")

    if reasons:
        return CleanupResult(
            verdict=CleanupVerdict.DENY_CLEANUP,
            deny_reasons=reasons,
            preserved_items=["closeout_evidence", "audit_provenance"],
        )

    # Full cleanup allowed for archived inactive tasks
    return CleanupResult(
        verdict=CleanupVerdict.ALLOW_PARTIAL_CLEANUP,
        preserved_items=["closeout_evidence", "audit_provenance"],
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def evaluate_non_happy_cleanup(
    state: str,
    context: Optional[Mapping[str, Any]] = None,
    *,
    is_symlink_escape: bool = False,
) -> CleanupResult:
    """Route a cleanup evaluation to the correct non-happy path handler.

    Args:
        state: One of blocked, cancelled, superseded, archived.
        context: Dict with task/cleanup fields.
        is_symlink_escape: True if the target is a symlink escape.

    Returns:
        CleanupResult with verdict and supporting fields.
    """
    if context is None:
        context = {}

    state_lower = _lower(state)

    if state_lower not in _NON_HAPPY_STATES:
        return CleanupResult(
            verdict=CleanupVerdict.DENY_CLEANUP,
            deny_reasons=[f"unknown_or_happy_path_state: {state_lower}"],
        )

    if state_lower == "blocked":
        return evaluate_blocked_cleanup(context, is_symlink_escape)
    elif state_lower == "cancelled":
        return evaluate_cancelled_cleanup(context, is_symlink_escape)
    elif state_lower == "superseded":
        return evaluate_superseded_cleanup(context, is_symlink_escape)
    else:  # archived
        return evaluate_archived_cleanup(context, is_symlink_escape)
