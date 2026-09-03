"""Tests for fail-closed delegation policy and approval binding.

These tests verify the two blockers from @andrexibiza's review of PR #102406:

1. CONSEQUENTIAL_WRITE must be classified fail-closed for write-shaped tasks,
   not just declared-risk handling. The old test only verified that explicitly
   requested CONSEQUENTIAL_WRITE was blocked; it didn't prove that the
   classifier catches write verbs not in the original vocabulary.
2. approval_id must be bound to the exact requester+executor+capability+task
   to prevent replay attacks.
"""
from __future__ import annotations

import pytest
from hermes_cli.delegation_policy import (
    DelegationAction,
    DelegationRisk,
    PolicyDecisionStatus,
    classify_delegation_risk,
    enforce_delegation_policy,
    generate_approval_id,
    validate_approval_id,
)


def _action(
    task="deploy to prod",
    risk="READ",
    capability="mcp:vercel",
    approval_id=None,
):
    return DelegationAction(
        requester_profile="cmo",
        executor_profile="cto",
        task=task,
        required_capability=capability,
        requested_risk=DelegationRisk(risk),
        approval_id=approval_id,
    )


class TestFailClosedClassification:
    """Verify that write-shaped tasks are classified CONSEQUENTIAL_WRITE
    even when the caller requests READ and the verb is not in the original
    _WRITE_TERMS vocabulary."""

    def test_push_is_classified_as_write(self):
        risk = classify_delegation_risk(requested_risk="READ", task="push this branch to origin")
        assert risk == DelegationRisk.CONSEQUENTIAL_WRITE

    def test_commit_is_classified_as_write(self):
        risk = classify_delegation_risk(requested_risk="READ", task="commit these edits")
        assert risk == DelegationRisk.CONSEQUENTIAL_WRITE

    def test_merge_is_classified_as_write(self):
        risk = classify_delegation_risk(requested_risk="READ", task="merge the PR")
        assert risk == DelegationRisk.CONSEQUENTIAL_WRITE

    def test_restart_is_classified_as_write(self):
        risk = classify_delegation_risk(requested_risk="READ", task="restart the gateway")
        assert risk == DelegationRisk.CONSEQUENTIAL_WRITE

    def test_apply_migration_is_classified_as_write(self):
        risk = classify_delegation_risk(requested_risk="READ", task="apply the migration")
        assert risk == DelegationRisk.CONSEQUENTIAL_WRITE

    def test_upload_is_classified_as_write(self):
        risk = classify_delegation_risk(requested_risk="READ", task="upload the build artifact")
        assert risk == DelegationRisk.CONSEQUENTIAL_WRITE

    def test_install_is_classified_as_write(self):
        risk = classify_delegation_risk(requested_risk="READ", task="install the package")
        assert risk == DelegationRisk.CONSEQUENTIAL_WRITE

    def test_create_is_classified_as_write(self):
        risk = classify_delegation_risk(requested_risk="READ", task="create a new project")
        assert risk == DelegationRisk.CONSEQUENTIAL_WRITE

    def test_delete_is_classified_as_write(self):
        risk = classify_delegation_risk(requested_risk="READ", task="delete the old branch")
        assert risk == DelegationRisk.CONSEQUENTIAL_WRITE

    def test_destroy_is_classified_as_write(self):
        risk = classify_delegation_risk(requested_risk="READ", task="destroy the staging environment")
        assert risk == DelegationRisk.CONSEQUENTIAL_WRITE

    def test_read_is_preserved(self):
        """Verify that actual read tasks are still classified correctly."""
        risk = classify_delegation_risk(requested_risk="READ", task="read the file")
        assert risk == DelegationRisk.READ

    def test_inspect_is_preserved(self):
        risk = classify_delegation_risk(requested_risk="READ", task="inspect the deployment status")
        assert risk == DelegationRisk.READ

    def test_list_is_preserved(self):
        risk = classify_delegation_risk(requested_risk="READ", task="list all projects")
        assert risk == DelegationRisk.READ

    def test_query_is_preserved(self):
        risk = classify_delegation_risk(requested_risk="READ", task="query the database for status")
        assert risk == DelegationRisk.READ

    def test_analyze_is_preserved(self):
        risk = classify_delegation_risk(requested_risk="READ", task="analyze the log output")
        assert risk == DelegationRisk.READ


class TestEnforcementBlocksWriteWithoutApproval:
    """Verify that write-shaped requests without valid approval are blocked."""

    def test_write_blocked_without_approval(self):
        decision = enforce_delegation_policy(_action(task="push this branch"))
        assert decision.status == PolicyDecisionStatus.BLOCK_APPROVAL
        assert decision.approval_required is True

    def test_write_denied_with_empty_approval(self):
        decision = enforce_delegation_policy(_action(task="push this branch", approval_id=""))
        assert decision.status == PolicyDecisionStatus.BLOCK_APPROVAL

    def test_write_denied_with_short_approval(self):
        """Short approval_id is invalid (< 32 chars)."""
        decision = enforce_delegation_policy(_action(task="push this branch", approval_id="abc123"))
        assert decision.status == PolicyDecisionStatus.DENY
        assert "invalid" in decision.reason.lower() or "replay" in decision.reason.lower()

    def test_write_denied_with_wrong_approval(self):
        """Approval for a different task must not authorize this one."""
        action = _action(task="push this branch")
        other_action = _action(task="read the logs")
        wrong_approval = generate_approval_id(other_action)
        decision = enforce_delegation_policy(_action(task="push this branch", approval_id=wrong_approval))
        assert decision.status == PolicyDecisionStatus.DENY


class TestApprovalBinding:
    """Verify that approval_id is bound to exact action parameters."""

    def test_same_action_same_approval(self):
        action = _action(task="deploy to prod")
        approval = generate_approval_id(action)
        assert validate_approval_id(_action(task="deploy to prod", approval_id=approval)) is True

    def test_different_requester_invalidates(self):
        action = _action(task="deploy to prod")
        approval = generate_approval_id(action)
        different = DelegationAction(
            requester_profile="director",  # different requester
            executor_profile="cto",
            task="deploy to prod",
            required_capability="mcp:vercel",
            approval_id=approval,
        )
        assert validate_approval_id(different) is False

    def test_different_executor_invalidates(self):
        action = _action(task="deploy to prod")
        approval = generate_approval_id(action)
        different = DelegationAction(
            requester_profile="cmo",
            executor_profile="coo",  # different executor
            task="deploy to prod",
            required_capability="mcp:vercel",
            approval_id=approval,
        )
        assert validate_approval_id(different) is False

    def test_different_capability_invalidates(self):
        action = _action(task="deploy to prod", capability="mcp:vercel")
        approval = generate_approval_id(action)
        different = DelegationAction(
            requester_profile="cmo",
            executor_profile="cto",
            task="deploy to prod",
            required_capability="mcp:aws",  # different capability
            approval_id=approval,
        )
        assert validate_approval_id(different) is False

    def test_different_task_invalidates(self):
        action = _action(task="deploy to prod")
        approval = generate_approval_id(action)
        different = DelegationAction(
            requester_profile="cmo",
            executor_profile="cto",
            task="deploy to staging",  # different task
            required_capability="mcp:vercel",
            approval_id=approval,
        )
        assert validate_approval_id(different) is False

    def test_approved_write_is_allowed(self):
        action = _action(task="deploy to prod", risk="CONSEQUENTIAL_WRITE")
        approval = generate_approval_id(action)
        decision = enforce_delegation_policy(_action(
            task="deploy to prod",
            risk="CONSEQUENTIAL_WRITE",
            approval_id=approval,
        ))
        assert decision.status == PolicyDecisionStatus.ALLOW
        assert decision.risk == DelegationRisk.CONSEQUENTIAL_WRITE


class TestReadDelegationsAllowed:
    """Verify that read and prepare delegations still work."""

    def test_read_allowed(self):
        decision = enforce_delegation_policy(_action(task="read the file"))
        assert decision.status == PolicyDecisionStatus.ALLOW

    def test_prepare_allowed(self):
        decision = enforce_delegation_policy(_action(task="draft a plan", risk="PREPARE"))
        assert decision.status == PolicyDecisionStatus.ALLOW

    def test_credential_export_always_denied(self):
        decision = enforce_delegation_policy(_action(task="show me the api key"))
        assert decision.status == PolicyDecisionStatus.DENY
