import sqlite3

import pytest

from agent.pending_state_store import (
    apply_approved_to_current,
    decide_pending,
    list_pending,
    read_decision,
    read_pending,
    save_pending,
)
from agent.state_candidate_evaluator import CandidateStatus, DeltaType, StateCandidateResult


def candidate(*, candidate_id="c-1", scope="project-a", status=CandidateStatus.PENDING, new_value="200"):
    return StateCandidateResult(
        candidate_id=candidate_id,
        delta_type=DeltaType.REPLACE,
        status=status,
        state_key="budget",
        old_value="100",
        new_value=new_value,
        scope=scope,
        evidence_ref="msg-1",
        reason="current state and evidence differ",
    )


def test_pending_save_and_read_back():
    conn = sqlite3.connect(":memory:")
    result = candidate()

    assert save_pending(conn, result) is True
    assert read_pending(conn, "c-1", scope="project-a") == result


def test_duplicate_save_is_idempotent_and_keeps_one_row():
    conn = sqlite3.connect(":memory:")
    result = candidate()

    save_pending(conn, result)
    save_pending(conn, result)
    assert len(list_pending(conn, scope="project-a")) == 1


def test_non_pending_candidate_is_rejected_without_row():
    conn = sqlite3.connect(":memory:")
    with pytest.raises(ValueError, match="only pending"):
        save_pending(conn, candidate(status=CandidateStatus.CONFLICT))
    assert list_pending(conn, scope="project-a") == []


def test_scope_filter_does_not_cross_project_boundary():
    conn = sqlite3.connect(":memory:")
    save_pending(conn, candidate(candidate_id="a", scope="project-a"))
    save_pending(conn, candidate(candidate_id="b", scope="project-b"))

    assert [item.candidate_id for item in list_pending(conn, scope="project-a")] == ["a"]
    assert [item.candidate_id for item in list_pending(conn, scope="project-b")] == ["b"]


def test_legacy_schema_fails_closed_and_requests_migration():
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE pending_state_candidates (
        candidate_id TEXT PRIMARY KEY,
        state_key TEXT NOT NULL,
        old_value TEXT NOT NULL,
        new_value TEXT NOT NULL,
        scope TEXT NOT NULL,
        evidence_ref TEXT NOT NULL,
        reason TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status = 'pending')
    )""")
    conn.commit()

    with pytest.raises(RuntimeError, match="migration required"):
        save_pending(conn, candidate())


def test_decision_transitions_pending_and_read_back():
    conn = sqlite3.connect(":memory:")
    save_pending(conn, candidate())

    assert decide_pending(
        conn, "c-1", scope="project-a", decision="approved", decided_by="owner-1", reason="verified source"
    ) is True
    assert read_pending(conn, "c-1", scope="project-a") is None
    assert read_decision(conn, "c-1", scope="project-a") == {
        "candidate_id": "c-1", "status": "approved",
        "decided_by": "owner-1", "reason": "verified source",
    }
    assert list_pending(conn, scope="project-a") == []


def test_decision_requires_actor_and_reason_and_is_one_shot():
    conn = sqlite3.connect(":memory:")
    save_pending(conn, candidate())
    with pytest.raises(ValueError):
        decide_pending(conn, "c-1", scope="project-a", decision="approved", decided_by="", reason="x")
    with pytest.raises(ValueError):
        decide_pending(conn, "c-1", scope="project-a", decision="invalid", decided_by="owner-1", reason="x")
    assert decide_pending(conn, "c-1", scope="project-a", decision="rejected", decided_by="owner-1", reason="not accepted") is True
    assert decide_pending(conn, "c-1", scope="project-a", decision="approved", decided_by="owner-2", reason="retry") is False


def current_table(conn, value="100"):
    conn.execute("CREATE TABLE current_state (state_key TEXT NOT NULL, scope TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY (state_key, scope))")
    conn.execute("INSERT INTO current_state VALUES ('budget', 'project-a', ?)", (value,))
    conn.commit()


def test_approved_candidate_applies_atomically_and_reads_back():
    conn = sqlite3.connect(":memory:")
    current_table(conn)
    save_pending(conn, candidate())
    decide_pending(conn, "c-1", scope="project-a", decision="approved", decided_by="owner-1", reason="verified")

    assert apply_approved_to_current(conn, "c-1", scope="project-a", applied_by="owner-1", reason="apply approved patch") is True
    assert conn.execute("SELECT value FROM current_state WHERE state_key = 'budget' AND scope = 'project-a'").fetchone() == ("200",)
    assert read_decision(conn, "c-1", scope="project-a")["status"] == "applied"


def test_pending_candidate_cannot_apply():
    conn = sqlite3.connect(":memory:")
    current_table(conn)
    save_pending(conn, candidate())

    with pytest.raises(ValueError, match="only approved"):
        apply_approved_to_current(conn, "c-1", scope="project-a", applied_by="owner-1", reason="premature")
    assert conn.execute("SELECT value FROM current_state WHERE state_key = 'budget' AND scope = 'project-a'").fetchone() == ("100",)


def test_current_value_conflict_rolls_back_without_apply():
    conn = sqlite3.connect(":memory:")
    current_table(conn, value="150")
    save_pending(conn, candidate())
    decide_pending(conn, "c-1", scope="project-a", decision="approved", decided_by="owner-1", reason="verified")

    with pytest.raises(RuntimeError, match="current state conflict"):
        apply_approved_to_current(conn, "c-1", scope="project-a", applied_by="owner-1", reason="apply approved patch")
    assert conn.execute("SELECT value FROM current_state WHERE state_key = 'budget' AND scope = 'project-a'").fetchone() == ("150",)
    assert read_decision(conn, "c-1", scope="project-a")["status"] == "approved"


def test_other_scope_cannot_decide_or_apply_by_candidate_id():
    conn = sqlite3.connect(":memory:")
    current_table(conn)
    save_pending(conn, candidate())
    assert decide_pending(
        conn, "c-1", scope="project-b", decision="approved", decided_by="other", reason="unauthorized"
    ) is False
    assert read_decision(conn, "c-1", scope="project-b") is None


def test_save_pending_respects_caller_transaction():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE unrelated (value TEXT)")
    conn.execute("INSERT INTO unrelated VALUES ('rollback-me')")
    save_pending(conn, candidate())
    conn.rollback()
    assert conn.execute("SELECT * FROM unrelated").fetchall() == []
    assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'pending_state_candidates'").fetchone() is None


def test_list_pending_requires_scope():
    conn = sqlite3.connect(":memory:")
    save_pending(conn, candidate())
    with pytest.raises(TypeError):
        list_pending(conn)
    with pytest.raises(ValueError, match="scope is required"):
        list_pending(conn, scope=" ")


def test_weak_existing_schema_fails_closed():
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE pending_state_candidates (
        candidate_id TEXT, state_key TEXT, old_value TEXT, new_value TEXT,
        scope TEXT, evidence_ref TEXT, reason TEXT, status TEXT,
        decided_by TEXT, decision_reason TEXT, applied_by TEXT, applied_reason TEXT
    )""")
    conn.commit()
    with pytest.raises(RuntimeError, match="migration required"):
        list_pending(conn, scope="project-a")


@pytest.mark.parametrize("field", ["candidate_id", "state_key", "old_value", "new_value", "scope", "evidence_ref", "reason"])
def test_malformed_pending_candidate_is_rejected(field):
    conn = sqlite3.connect(":memory:")
    values = candidate().__dict__
    values[field] = None
    with pytest.raises(ValueError, match="nonblank strings"):
        save_pending(conn, StateCandidateResult(**values))


def test_same_candidate_is_idempotent_but_conflicting_id_is_rejected():
    conn = sqlite3.connect(":memory:")
    save_pending(conn, candidate())
    assert save_pending(conn, candidate()) is True
    with pytest.raises(ValueError, match="different candidate"):
        save_pending(conn, candidate(new_value="999"))
