"""Native evidence for strict invocation binding (#60): write-once bind, request ordinals, no laundering.

The record lives on the session row (``model_config.invocation_binding``). These tests drive the real
``SessionDB`` against a temp file; the agent-side gate that feeds it is covered in
``tests/agent/test_invocation_binding.py``.
"""

import pytest

from agent.invocation_binding import InvocationBinding, InvocationBindingError, build_record
from hermes_state import SessionDB


def _binding(**overrides):
    fields = dict(
        invocation_id="inv-1", credential_id="cred-a", provider="anthropic", model="claude-sonnet-4-6",
        reasoning_effort="high",
    )
    return InvocationBinding(**{**fields, **overrides})


SNAPSHOT = {
    "provider": "anthropic", "api_mode": "anthropic_messages", "model": "claude-sonnet-4-6",
    "wire_reasoning_effort": "high", "credential_id": "cred-a", "endpoint_host": "api.anthropic.com", "at": 1.0,
}


@pytest.fixture
def db(tmp_path):
    database = SessionDB(db_path=tmp_path / "state.db")
    database.create_session("sess-1", "cli", model="claude-sonnet-4-6")
    yield database
    database.close()


def _refused(code):
    return pytest.raises(InvocationBindingError, match=f"^{code}$")


def test_bind_is_write_once_and_identical_rebind_is_idempotent(db):
    record = build_record(_binding())
    assert db.bind_invocation("sess-1", record) == "bound"
    stored = db.get_invocation_binding("sess-1")
    assert stored["digest"] == _binding().digest and stored["account_verified"] is False

    assert db.bind_invocation("sess-1", build_record(_binding())) == "existing"  # restart / resume
    assert db.get_invocation_binding("sess-1")["bound_at"] == stored["bound_at"]


def test_different_binding_on_a_bound_session_is_a_conflict(db):
    db.bind_invocation("sess-1", build_record(_binding()))
    for other in (_binding(invocation_id="inv-2"), _binding(model="claude-opus-4-6"), _binding(credential_id="cred-b")):
        with _refused("binding_conflict"):
            db.bind_invocation("sess-1", build_record(other))
    assert db.get_invocation_binding("sess-1")["digest"] == _binding().digest


def test_reused_invocation_id_in_an_unrelated_session_is_refused(db):
    db.bind_invocation("sess-1", build_record(_binding()))
    db.create_session("sess-2", "cli", model="claude-sonnet-4-6")
    with _refused("invocation_reused"):
        db.bind_invocation("sess-2", build_record(_binding()))
    assert db.get_invocation_binding("sess-2") is None


def test_compression_continuation_may_rebind_the_same_invocation(db):
    db.bind_invocation("sess-1", build_record(_binding()))
    db.create_session("sess-child", "cli", model="claude-sonnet-4-6", parent_session_id="sess-1")
    assert db.bind_invocation("sess-child", build_record(_binding())) in ("bound", "existing")

    db.create_session("sess-grandchild", "cli", model="claude-sonnet-4-6", parent_session_id="sess-child")
    assert db.bind_invocation("sess-grandchild", build_record(_binding())) in ("bound", "existing")


def test_binding_survives_a_session_row_reupsert(db):
    db.bind_invocation("sess-1", build_record(_binding()))
    db.create_session("sess-1", "cli", model="claude-sonnet-4-6", model_config={"yolo_mode": False})
    assert db.get_invocation_binding("sess-1")["invocation_id"] == "inv-1"


def test_requests_are_numbered_and_settled_without_rewriting_history(db):
    db.bind_invocation("sess-1", build_record(_binding()))
    assert db.record_invocation_request("sess-1", "inv-1", dict(SNAPSHOT)) == 1
    db.settle_invocation_request("sess-1", "inv-1", 1, "responded")
    assert db.record_invocation_request("sess-1", "inv-1", dict(SNAPSHOT)) == 2
    db.settle_invocation_request("sess-1", "inv-1", 2, "failed", "TimeoutError")
    assert db.record_invocation_request("sess-1", "inv-1", dict(SNAPSHOT)) == 3  # never settled

    record = db.get_invocation_binding("sess-1")
    assert (record["request_count"], record["responded_count"], record["failed_count"]) == (3, 1, 1)
    assert record["last_request"]["ordinal"] == 3 and record["last_request"]["state"] == "prepared"

    # a late settle for an older ordinal counts but must not repaint the newest request
    db.settle_invocation_request("sess-1", "inv-1", 1, "responded")
    record = db.get_invocation_binding("sess-1")
    assert record["responded_count"] == 2 and record["last_request"]["state"] == "prepared"


def test_violations_are_appended_and_bounded(db):
    db.bind_invocation("sess-1", build_record(_binding()))
    for index in range(25):
        db.record_invocation_violation("sess-1", "inv-1", f"code_{index}")
    violations = db.get_invocation_binding("sess-1")["violations"]
    assert len(violations) == 20 and violations[-1] == "code_24"


def test_writes_without_a_matching_binding_are_refused(db):
    with _refused("native_binding_missing"):
        db.record_invocation_request("sess-1", "inv-1", dict(SNAPSHOT))
    with _refused("native_session_missing"):
        db.bind_invocation("no-such-session", build_record(_binding()))

    db.bind_invocation("sess-1", build_record(_binding()))
    with _refused("binding_conflict"):
        db.record_invocation_request("sess-1", "inv-other", dict(SNAPSHOT))
    with _refused("native_record_failed"):
        db.settle_invocation_request("sess-1", "inv-1", 1, "bogus")
