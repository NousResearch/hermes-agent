import time
import pytest
from hermes_cli import tunnel_approvals as ta


def test_file_request_then_list_pending(tmp_path):
    p = tmp_path / "hold.jsonl"
    rid = ta.file_request(str(p), user="alice", subdomains=["alice.noit2.com"],
                          reason="demo", requested_until=None)
    assert isinstance(rid, str) and rid
    pending = ta.list_pending(str(p))
    assert len(pending) == 1
    assert pending[0]["id"] == rid
    assert pending[0]["status"] == "pending"
    assert pending[0]["user"] == "alice"


def test_approve_sets_status_and_until(tmp_path):
    p = tmp_path / "hold.jsonl"
    rid = ta.file_request(str(p), user="alice", subdomains=["alice.noit2.com"],
                          reason="demo", requested_until=None)
    rec = ta.approve(str(p), rid, until=99999.0, by="admin1", admin_ids=["admin1"])
    assert rec["status"] == "approved"
    assert rec["approved_until"] == 99999.0
    assert rec["decided_by"] == "admin1"
    assert ta.is_approved(str(p), rid) is True
    assert ta.approved_until(str(p), rid) == 99999.0
    assert ta.list_pending(str(p)) == []


def test_deny_sets_status(tmp_path):
    p = tmp_path / "hold.jsonl"
    rid = ta.file_request(str(p), user="alice", subdomains=["alice.noit2.com"],
                          reason="demo", requested_until=None)
    rec = ta.deny(str(p), rid, reason="too long", by="admin1", admin_ids=["admin1"])
    assert rec["status"] == "denied"
    assert ta.is_approved(str(p), rid) is False


def test_non_admin_cannot_approve(tmp_path):
    p = tmp_path / "hold.jsonl"
    rid = ta.file_request(str(p), user="alice", subdomains=["alice.noit2.com"],
                          reason="demo", requested_until=None)
    with pytest.raises(PermissionError):
        ta.approve(str(p), rid, until=99999.0, by="alice", admin_ids=["admin1"])


def test_unknown_id_raises(tmp_path):
    p = tmp_path / "hold.jsonl"
    with pytest.raises(KeyError):
        ta.approve(str(p), "nope", until=1.0, by="admin1", admin_ids=["admin1"])


def test_double_approve_is_valueerror(tmp_path):
    p = tmp_path / "hold.jsonl"
    rid = ta.file_request(str(p), user="alice", subdomains=["alice.noit2.com"],
                          reason="demo", requested_until=None)
    ta.approve(str(p), rid, until=99999.0, by="admin1", admin_ids=["admin1"])
    with pytest.raises(ValueError):
        ta.approve(str(p), rid, until=99999.0, by="admin1", admin_ids=["admin1"])