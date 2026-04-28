"""Tests for persistent Code Mode approval governance."""

from __future__ import annotations

import time

import pytest

from hermes_cli.code.approval_governance import (
    ApprovalGovernanceError,
    ApprovalGovernanceService,
    ApprovalKind,
    ApprovalStatus,
)
from hermes_cli.code.execution_policy import RiskClass
from hermes_state import SessionDB


@pytest.fixture()
def service(tmp_path):
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    return ApprovalGovernanceService(db_path=db_path)


def _create(service: ApprovalGovernanceService, *, kind: str = ApprovalKind.GENERIC, title: str = "Approval A", expires_at=None):
    return service.create_request(
        kind=kind,
        risk_class=RiskClass.GIT_WRITE,
        title=title,
        description="desc",
        requested_action=f"{kind}.action",
        requested_payload={"arg": "value", "token": "secret-token"},
        resource_type="test_resource",
        resource_id="res-1",
        github_repo_full_name="acme/repo",
        github_issue_number=7,
        expires_at=expires_at,
    )


def test_create_list_get_approval_request(service):
    created = _create(service)
    assert created["status"] == ApprovalStatus.PENDING
    assert created["kind"] == ApprovalKind.GENERIC

    listed = service.list_requests()
    assert any(item["id"] == created["id"] for item in listed)

    fetched = service.get_request(created["id"])
    assert fetched is not None
    assert fetched["id"] == created["id"]
    assert fetched["resource_id"] == "res-1"


def test_approve_pending(service):
    created = _create(service)
    approved = service.approve_request(created["id"], approved_by="local")
    assert approved["status"] == ApprovalStatus.APPROVED
    assert approved["approved_by"] == "local"


def test_reject_pending(service):
    created = _create(service)
    rejected = service.reject_request(created["id"], rejected_by="local", reason="no")
    assert rejected["status"] == ApprovalStatus.REJECTED
    assert rejected["rejected_by"] == "local"


def test_cancel_pending(service):
    created = _create(service)
    cancelled = service.cancel_request(created["id"], cancelled_by="local", reason="stop")
    assert cancelled["status"] == ApprovalStatus.CANCELLED


def test_expire_pending(service):
    created = _create(service, expires_at=time.time() - 2)
    expired = service.expire_pending()
    assert expired["count"] >= 1
    fetched = service.get_request(created["id"])
    assert fetched is not None
    assert fetched["status"] == ApprovalStatus.EXPIRED


def test_mark_executed(service):
    created = _create(service)
    service.approve_request(created["id"], approved_by="alice")
    executed = service.mark_executed(created["id"], metadata={"run": "ok"})
    assert executed["status"] == ApprovalStatus.EXECUTED


def test_mark_failed(service):
    created = _create(service)
    service.approve_request(created["id"], approved_by="alice")
    failed = service.mark_failed(created["id"], reason="network error")
    assert failed["status"] == ApprovalStatus.FAILED


def test_invalid_transitions_rejected(service):
    created = _create(service)
    with pytest.raises(ApprovalGovernanceError):
        service.mark_executed(created["id"])


def test_terminal_states_cannot_transition(service):
    created = _create(service)
    service.reject_request(created["id"], rejected_by="alice")
    with pytest.raises(ApprovalGovernanceError):
        service.approve_request(created["id"], approved_by="bob")


def test_redaction_of_sensitive_payloads(service):
    created = _create(service)
    payload = created["requested_payload"]
    assert payload["token"] == "[REDACTED]"
    assert "secret-token" not in str(created)


def test_summary_counts(service):
    one = _create(service, kind=ApprovalKind.GITHUB_COMMENT, title="A")
    two = _create(service, kind=ApprovalKind.GITHUB_PR_PREPARE, title="B")
    service.approve_request(one["id"], approved_by="alice")
    service.reject_request(two["id"], rejected_by="alice")

    summary = service.summary()
    assert summary["status"]["approved"] >= 1
    assert summary["status"]["rejected"] >= 1
    assert summary["kind"]["github_comment"] >= 1
    assert summary["kind"]["github_pr_prepare"] >= 1


def test_validate_for_execution_replay_and_kind_binding(service):
    created = _create(service, kind=ApprovalKind.GITHUB_COMMENT)
    service.approve_request(created["id"], approved_by="alice")

    valid = service.validate_for_execution(
        created["id"],
        expected_kind=ApprovalKind.GITHUB_COMMENT,
        expected_requested_action="github_comment.action",
        expected_resource_type="test_resource",
        expected_resource_id="res-1",
        expected_github_repo_full_name="acme/repo",
        expected_github_issue_number=7,
        expected_requested_payload={"arg": "value", "token": "secret-token"},
    )
    assert valid["id"] == created["id"]

    with pytest.raises(ApprovalGovernanceError):
        service.validate_for_execution(
            created["id"],
            expected_kind=ApprovalKind.GITHUB_PR_PREPARE,
            expected_requested_action="github_pr_prepare.action",
            expected_resource_type="test_resource",
            expected_resource_id="res-1",
            expected_github_repo_full_name="acme/repo",
            expected_github_issue_number=7,
            expected_requested_payload={"arg": "value", "token": "secret-token"},
        )
