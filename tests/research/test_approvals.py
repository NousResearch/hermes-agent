"""Tests for research.approvals."""

import pytest

from research.approvals import ApprovalStore
from research.state_store import ResearchStateStore


def test_request_creates_pending_record(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Test")
    approvals = ApprovalStore(store)
    record = approvals.request(project.project_id, "start_run", {"model": "llama"}, reason="test")
    assert record["status"] == "pending"
    assert record["action"] == "start_run"
    assert record["approval_id"].startswith("approval_")


def test_approve_changes_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Test")
    approvals = ApprovalStore(store)
    record = approvals.request(project.project_id, "start_run", {})
    approved = approvals.approve(project.project_id, record["approval_id"])
    assert approved["status"] == "approved"
    assert "approved_at" in approved


def test_is_approved_true(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Test")
    approvals = ApprovalStore(store)
    record = approvals.request(project.project_id, "start_run", {})
    approvals.approve(project.project_id, record["approval_id"])
    assert approvals.is_approved(project.project_id, "start_run", record["approval_id"]) is True


def test_is_approved_wrong_action(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Test")
    approvals = ApprovalStore(store)
    record = approvals.request(project.project_id, "start_run", {})
    approvals.approve(project.project_id, record["approval_id"])
    # Wrong action
    assert approvals.is_approved(project.project_id, "stop_run", record["approval_id"]) is False


def test_is_approved_missing_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Test")
    approvals = ApprovalStore(store)
    assert approvals.is_approved(project.project_id, "start_run", None) is False
    assert approvals.is_approved(project.project_id, "start_run", "nonexistent") is False
