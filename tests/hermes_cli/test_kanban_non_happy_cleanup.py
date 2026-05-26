"""Tests for non-happy path cleanup evaluation (Slice 7).

Covers blocked, cancelled, superseded, and archived lifecycle states with
fail-closed cleanup semantics per the rule matrix.
"""

from __future__ import annotations

import pytest

from hermes_cli.kanban_non_happy_cleanup import (
    CleanupVerdict,
    evaluate_blocked_cleanup,
    evaluate_cancelled_cleanup,
    evaluate_superseded_cleanup,
    evaluate_archived_cleanup,
    evaluate_non_happy_cleanup,
    _is_allowlisted_artifact,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_context(**overrides):
    ctx = {
        "workspace_path": "/tmp/test-workspace",
        "retained_reason": "",
        "revisit_at": "",
        "ttl": "",
        "cancellation_reason": "",
        "cancellation_evidence": "",
        "dispositions": {},
        "successor_ref": "",
        "successor_link": "",
        "inactive": False,
        "evidence_preserved": False,
        "residue_class": "reproducible",
        "target_path": "/tmp/test-workspace/node_modules",
    }
    ctx.update(overrides)
    return ctx


# ---------------------------------------------------------------------------
# Artifact allowlist tests
# ---------------------------------------------------------------------------

class TestArtifactAllowlist:
    def test_allowlisted_artifact_passes(self):
        for name in ("node_modules", ".next", ".turbo", "dist", "build",
                      "target", ".pytest_cache", ".ruff_cache", ".mypy_cache", "coverage"):
            assert _is_allowlisted_artifact(f"/ws/{name}") is True
            assert _is_allowlisted_artifact(f"/ws/sub/{name}") is True
            assert _is_allowlisted_artifact(name) is True

    def test_source_file_fails(self):
        assert _is_allowlisted_artifact("/ws/src/main.py") is False
        assert _is_allowlisted_artifact("/ws/tests/test_x.py") is False
        assert _is_allowlisted_artifact("Pipfile") is False
        assert _is_allowlisted_artifact(".gitignore") is False

    def test_empty_basename_fails(self):
        assert _is_allowlisted_artifact("") is False
        assert _is_allowlisted_artifact("/") is False


# ---------------------------------------------------------------------------
# Blocked
# ---------------------------------------------------------------------------

class TestBlockedCleanup:
    """Blocked path: prune reproducible artifacts, preserve source/diff/evidence."""

    def test_prunes_reproducible_artifacts(self):
        ctx = _base_context(
            retained_reason="waiting for upstream fix",
            revisit_at="2026-06-01",
            residue_class="reproducible",
            target_path="/tmp/test-workspace/node_modules",
        )
        result = evaluate_blocked_cleanup(ctx)
        assert result.verdict == CleanupVerdict.ALLOW_PRUNE

    def test_preserves_source_files(self):
        ctx = _base_context(
            retained_reason="waiting for review",
            revisit_at="2026-06-01",
            residue_class="resumable",
            target_path="/tmp/test-workspace/src/main.py",
        )
        result = evaluate_blocked_cleanup(ctx)
        assert result.verdict == CleanupVerdict.DENY_CLEANUP
        assert any("source" in r.lower() or "resumable" in r.lower() for r in result.deny_reasons)

    def test_missing_retained_reason_denies(self):
        ctx = _base_context(
            revisit_at="2026-06-01",
        )
        result = evaluate_blocked_cleanup(ctx)
        assert result.verdict == CleanupVerdict.DENY_CLEANUP
        assert any("reason" in r.lower() for r in result.deny_reasons)

    def test_missing_revisit_or_ttl_denies(self):
        ctx = _base_context(
            retained_reason="waiting",
        )
        result = evaluate_blocked_cleanup(ctx)
        assert result.verdict == CleanupVerdict.DENY_CLEANUP
        assert any("revisit" in r.lower() or "ttl" in r.lower() for r in result.deny_reasons)


# ---------------------------------------------------------------------------
# Cancelled
# ---------------------------------------------------------------------------

class TestCancelledCleanup:
    """Cancelled path: dirty work blocks full cleanup without disposition."""

    def test_cancelled_with_reason_and_evidence_allows_artifact_prune(self):
        ctx = _base_context(
            cancellation_reason="superseded by BO-200",
            cancellation_evidence="Linked from BO-200",
            residue_class="reproducible",
            dispositions={"/tmp/test-workspace/dirty.py": "discard"},
        )
        result = evaluate_cancelled_cleanup(ctx)
        assert result.verdict in (CleanupVerdict.ALLOW_PRUNE, CleanupVerdict.ALLOW_PARTIAL_CLEANUP)

    def test_cancelled_dirty_work_without_disposition_denies_full(self):
        ctx = _base_context(
            cancellation_reason="no longer needed",
            cancellation_evidence="recorded",
            residue_class="unique_dirty",
        )
        result = evaluate_cancelled_cleanup(ctx)
        assert result.verdict == CleanupVerdict.DENY_CLEANUP
        assert any("disposition" in r.lower() or "dirty" in r.lower() for r in result.deny_reasons)

    def test_cancelled_missing_reason_denies(self):
        ctx = _base_context(
            residue_class="reproducible",
        )
        result = evaluate_cancelled_cleanup(ctx)
        assert result.verdict == CleanupVerdict.DENY_CLEANUP
        assert any("reason" in r.lower() for r in result.deny_reasons)

    def test_cancelled_missing_evidence_denies(self):
        ctx = _base_context(
            cancellation_reason="done",
        )
        result = evaluate_cancelled_cleanup(ctx)
        assert result.verdict == CleanupVerdict.DENY_CLEANUP
        assert any("evidence" in r.lower() for r in result.deny_reasons)

    def test_cancelled_clean_with_disposition_allows(self):
        ctx = _base_context(
            cancellation_reason="replaced",
            cancellation_evidence="commit abc123",
            residue_class="reproducible",
            dispositions={"all": "discard"},
        )
        result = evaluate_cancelled_cleanup(ctx)
        assert result.verdict in (CleanupVerdict.ALLOW_PRUNE, CleanupVerdict.ALLOW_PARTIAL_CLEANUP)


# ---------------------------------------------------------------------------
# Superseded
# ---------------------------------------------------------------------------

class TestSupersededCleanup:
    """Superseded path: successor link required before cleanup."""

    def test_superseded_with_successor_allows(self):
        ctx = _base_context(
            successor_ref="BO-200",
            successor_link="t_abc123",
            residue_class="reproducible",
        )
        result = evaluate_superseded_cleanup(ctx)
        assert result.verdict in (CleanupVerdict.ALLOW_PRUNE, CleanupVerdict.ALLOW_PARTIAL_CLEANUP)

    def test_superseded_missing_successor_denies(self):
        ctx = _base_context(
            residue_class="reproducible",
        )
        result = evaluate_superseded_cleanup(ctx)
        assert result.verdict == CleanupVerdict.DENY_CLEANUP
        assert any("successor" in r.lower() for r in result.deny_reasons)

    def test_superseded_dirty_work_without_disposition_blocks_full(self):
        ctx = _base_context(
            successor_ref="BO-200",
            residue_class="unique_dirty",
        )
        result = evaluate_superseded_cleanup(ctx)
        assert result.verdict == CleanupVerdict.DENY_CLEANUP


# ---------------------------------------------------------------------------
# Archived
# ---------------------------------------------------------------------------

class TestArchivedCleanup:
    """Archived path: full cleanup only if inactive + evidence preserved."""

    def test_archived_inactive_evidence_preserved_allows_full(self):
        ctx = _base_context(
            inactive=True,
            evidence_preserved=True,
            residue_class="reproducible",
        )
        result = evaluate_archived_cleanup(ctx)
        assert result.verdict in (CleanupVerdict.ALLOW_PRUNE, CleanupVerdict.ALLOW_PARTIAL_CLEANUP)

    def test_archived_not_inactive_denies(self):
        ctx = _base_context(
            inactive=False,
            evidence_preserved=True,
            residue_class="reproducible",
        )
        result = evaluate_archived_cleanup(ctx)
        assert result.verdict == CleanupVerdict.DENY_CLEANUP
        assert any("inactive" in r.lower() for r in result.deny_reasons)

    def test_archived_evidence_not_preserved_denies(self):
        ctx = _base_context(
            inactive=True,
            evidence_preserved=False,
            residue_class="reproducible",
        )
        result = evaluate_archived_cleanup(ctx)
        assert result.verdict == CleanupVerdict.DENY_CLEANUP
        assert any("evidence" in r.lower() for r in result.deny_reasons)

    def test_archived_both_missing_denies(self):
        ctx = _base_context(
            inactive=False,
            evidence_preserved=False,
        )
        result = evaluate_archived_cleanup(ctx)
        assert result.verdict == CleanupVerdict.DENY_CLEANUP


# ---------------------------------------------------------------------------
# Global deny rules
# ---------------------------------------------------------------------------

class TestGlobalDenyRules:
    """Global deny rules apply across all lifecycle states."""

    def test_outside_workspace_denies(self):
        ctx = _base_context(
            workspace_path="/tmp/test-workspace",
            target_path="/etc/passwd",
            retained_reason="test",
            revisit_at="2026-06-01",
        )
        result = evaluate_blocked_cleanup(ctx)
        assert result.verdict == CleanupVerdict.DENY_CLEANUP
        assert any("outside" in r.lower() or "contain" in r.lower() for r in result.deny_reasons)

    def test_symlink_target_denies(self):
        result = evaluate_non_happy_cleanup(
            state="blocked",
            context={"workspace_path": "/ws", "target_path": "/ws/link"},
            is_symlink_escape=True,
        )
        assert result.verdict == CleanupVerdict.DENY_CLEANUP
        assert any("symlink" in r.lower() for r in result.deny_reasons)

    def test_source_file_denies_even_if_allowlisted_basename(self):
        ctx = _base_context(
            retained_reason="test",
            revisit_at="2026-06-01",
            residue_class="resumable",
            target_path="/tmp/test-workspace/build/main.o",
        )
        result = evaluate_blocked_cleanup(ctx)
        assert result.verdict == CleanupVerdict.DENY_CLEANUP


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class TestEvaluateNonHappyCleanup:
    def test_routes_blocked(self):
        result = evaluate_non_happy_cleanup(
            state="blocked",
            context={
                "workspace_path": "/ws",
                "target_path": "/ws/node_modules",
                "retained_reason": "waiting",
                "revisit_at": "2026-06-01",
                "residue_class": "reproducible",
            },
        )
        assert result.verdict == CleanupVerdict.ALLOW_PRUNE

    def test_routes_cancelled(self):
        result = evaluate_non_happy_cleanup(
            state="cancelled",
            context={
                "workspace_path": "/ws",
                "target_path": "/ws/node_modules",
                "cancellation_reason": "done",
                "cancellation_evidence": "recorded",
                "residue_class": "reproducible",
            },
        )
        assert result.verdict in (CleanupVerdict.ALLOW_PRUNE, CleanupVerdict.ALLOW_PARTIAL_CLEANUP)

    def test_routes_superseded(self):
        result = evaluate_non_happy_cleanup(
            state="superseded",
            context={
                "workspace_path": "/ws",
                "target_path": "/ws/node_modules",
                "successor_ref": "BO-200",
                "residue_class": "reproducible",
            },
        )
        assert result.verdict in (CleanupVerdict.ALLOW_PRUNE, CleanupVerdict.ALLOW_PARTIAL_CLEANUP)

    def test_routes_archived(self):
        result = evaluate_non_happy_cleanup(
            state="archived",
            context={
                "workspace_path": "/ws",
                "target_path": "/ws/node_modules",
                "inactive": True,
                "evidence_preserved": True,
                "residue_class": "reproducible",
            },
        )
        assert result.verdict in (CleanupVerdict.ALLOW_PRUNE, CleanupVerdict.ALLOW_PARTIAL_CLEANUP)

    def test_unknown_state_denies(self):
        result = evaluate_non_happy_cleanup(
            state="unknown_state",
            context={"workspace_path": "/ws", "target_path": "/ws/x"},
        )
        assert result.verdict == CleanupVerdict.DENY_CLEANUP

    def test_happy_path_states_return_deny(self):
        for state in ("worker_done", "review_ready", "closed"):
            result = evaluate_non_happy_cleanup(state=state, context={})
            assert result.verdict == CleanupVerdict.DENY_CLEANUP


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_context_denies(self):
        result = evaluate_non_happy_cleanup(state="blocked", context={})
        assert result.verdict == CleanupVerdict.DENY_CLEANUP

    def test_none_context_denies(self):
        result = evaluate_non_happy_cleanup(state="blocked", context=None)
        assert result.verdict == CleanupVerdict.DENY_CLEANUP


# ---------------------------------------------------------------------------
# Verdict fields
# ---------------------------------------------------------------------------

class TestVerdictFields:
    def test_allow_prune_has_no_deny_reasons(self):
        ctx = _base_context(
            retained_reason="r",
            revisit_at="2026-06-01",
            residue_class="reproducible",
        )
        result = evaluate_blocked_cleanup(ctx)
        assert result.verdict == CleanupVerdict.ALLOW_PRUNE
        assert result.deny_reasons == []

    def test_deny_has_deny_reasons(self):
        ctx = _base_context()
        result = evaluate_blocked_cleanup(ctx)
        assert result.verdict == CleanupVerdict.DENY_CLEANUP
        assert len(result.deny_reasons) > 0

    def test_all_verdicts_are_valid_enums(self):
        for verdict in CleanupVerdict:
            assert isinstance(verdict.value, str)
