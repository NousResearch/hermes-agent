"""Storage-engine tests for the task-ownership controller.

HERMES_HOME is isolated per test by the autouse ``_hermetic_environment``
fixture in tests/conftest.py, so every test gets a fresh
task_ownership.db with no extra setup.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hermes_cli import task_ownership_db as tdb


@pytest.fixture
def conn():
    with tdb.connect_closing() as c:
        yield c


def _touch_state_changed_at(conn, task_id: str, hours_ago: float) -> None:
    when = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    conn.execute(
        "UPDATE tasks SET state_changed_at = ?, updated_at = ? WHERE id = ?",
        (when, when, task_id),
    )
    conn.commit()


# ── basic CRUD ───────────────────────────────────────────────────────────


def test_create_task_starts_new(conn):
    task = tdb.create_task(conn, title="Do the thing", next_action="start it")
    assert task["state"] == "NEW"
    assert task["next_action"] == "start it"
    assert task["retry_count"] == 0
    assert task["max_retries"] == 3

    fetched = tdb.get_task(conn, task["id"])
    assert fetched == task


def test_get_missing_task_raises(conn):
    with pytest.raises(tdb.TaskNotFoundError):
        tdb.get_task(conn, "t_does_not_exist")


def test_list_tasks_excludes_terminal_by_default(conn):
    a = tdb.create_task(conn, title="A")
    b = tdb.create_task(conn, title="B")
    tdb.set_state(conn, b["id"], "WORKING")
    tdb.set_state(conn, b["id"], "CANCELLED")

    active = tdb.list_tasks(conn, include_terminal=False)
    assert [t["id"] for t in active] == [a["id"]]

    everything = tdb.list_tasks(conn, include_terminal=True)
    assert {t["id"] for t in everything} == {a["id"], b["id"]}


# ── state machine ────────────────────────────────────────────────────────


def test_valid_transition_chain(conn):
    task = tdb.create_task(conn, title="Chain")
    task = tdb.set_state(conn, task["id"], "WORKING")
    assert task["state"] == "WORKING"
    task = tdb.set_state(conn, task["id"], "VERIFYING")
    assert task["state"] == "VERIFYING"


def test_invalid_transition_rejected(conn):
    task = tdb.create_task(conn, title="Bad jump")
    with pytest.raises(tdb.InvalidTransitionError):
        tdb.set_state(conn, task["id"], "DONE")  # NEW -> DONE is not reachable
    # State must be unchanged after the rejected attempt.
    assert tdb.get_task(conn, task["id"])["state"] == "NEW"


@pytest.mark.parametrize("terminal", ["DONE", "CANCELLED"])
def test_terminal_states_have_no_outgoing_transitions(terminal):
    assert tdb.TRANSITIONS[terminal] == frozenset()


def test_same_state_transition_is_a_noop(conn):
    task = tdb.create_task(conn, title="Idempotent")
    before = tdb.get_task(conn, task["id"])
    after = tdb.set_state(conn, task["id"], "NEW")
    assert after["updated_at"] == before["updated_at"]


# ── no-false-completion invariant ───────────────────────────────────────


def test_mark_done_without_evidence_is_refused(conn):
    task = tdb.create_task(conn, title="Needs proof")
    tdb.set_state(conn, task["id"], "WORKING")
    tdb.set_state(conn, task["id"], "VERIFYING")
    with pytest.raises(tdb.VerificationRequiredError):
        tdb.mark_done(conn, task["id"])
    assert tdb.get_task(conn, task["id"])["state"] == "VERIFYING"


def test_mark_done_requires_verifying_state_even_with_evidence_field_set(conn):
    """Evidence alone isn't enough — the task must actually be in VERIFYING."""
    task = tdb.create_task(conn, title="Sneaky")
    # Directly poke evidence onto a NEW task without going through the
    # normal record_verification() path (which would itself transition to
    # VERIFYING) to prove mark_done() checks *state*, not just the field.
    conn.execute(
        "UPDATE tasks SET verification_evidence = 'looks fine' WHERE id = ?",
        (task["id"],),
    )
    conn.commit()
    with pytest.raises(tdb.InvalidTransitionError):
        tdb.mark_done(conn, task["id"])


def test_verify_then_done_succeeds(conn):
    task = tdb.create_task(conn, title="Provable")
    tdb.set_state(conn, task["id"], "WORKING")
    tdb.record_verification(conn, task["id"], "row counts match: 100 == 100")
    done = tdb.mark_done(conn, task["id"])
    assert done["state"] == "DONE"
    assert done["verification_evidence"] == "row counts match: 100 == 100"


def test_done_with_inline_evidence_succeeds_without_prior_verify_call(conn):
    task = tdb.create_task(conn, title="Inline proof")
    tdb.set_state(conn, task["id"], "WORKING")
    done = tdb.mark_done(conn, task["id"], evidence="checked output by hand")
    assert done["state"] == "DONE"


def test_outcome_success_never_completes_a_task(conn):
    """A worker reporting success cannot, by itself, produce DONE."""
    task = tdb.create_task(conn, title="Reported done")
    tdb.set_state(conn, task["id"], "WORKING")
    result = tdb.record_outcome(conn, task["id"], result="success", detail="looks done to me")
    assert result["state"] == "WORKING"
    assert result["state"] != "DONE"


def test_events_log_records_verification_before_completion(conn):
    task = tdb.create_task(conn, title="Audited")
    tdb.set_state(conn, task["id"], "WORKING")
    tdb.record_verification(conn, task["id"], "checked")
    tdb.mark_done(conn, task["id"])
    events = tdb.list_events(conn, task["id"])
    kinds = [e["event"] for e in events]
    assert "verification_recorded" in kinds
    assert "completed" in kinds
    assert kinds.index("verification_recorded") < kinds.index("completed")


def test_approval_required_blocks_done_until_approved(conn):
    task = tdb.create_task(conn, title="Needs sign-off", approval_required=True)
    tdb.set_state(conn, task["id"], "WORKING")
    tdb.record_verification(conn, task["id"], "evidence attached")
    with pytest.raises(tdb.ApprovalRequiredError):
        tdb.mark_done(conn, task["id"])

    tdb.approve_task(conn, task["id"], "zalmen")
    done = tdb.mark_done(conn, task["id"])
    assert done["state"] == "DONE"
    assert done["approved_by"] == "zalmen"


# ── duplicate / idempotent receipts ─────────────────────────────────────


def test_duplicate_receipt_is_idempotent_noop(conn):
    task = tdb.create_task(conn, title="Payment task")
    first = tdb.record_receipt(conn, task["id"], receipt_id="ext-123", source="stripe")
    assert first["duplicate"] is False

    second = tdb.record_receipt(conn, task["id"], receipt_id="ext-123", source="stripe")
    assert second["duplicate"] is True
    assert second["id"] == first["id"]

    rows = tdb.list_receipts(conn, task["id"])
    assert len(rows) == 1


def test_different_receipt_ids_both_recorded(conn):
    task = tdb.create_task(conn, title="Multi-receipt")
    tdb.record_receipt(conn, task["id"], receipt_id="a")
    tdb.record_receipt(conn, task["id"], receipt_id="b")
    assert len(tdb.list_receipts(conn, task["id"])) == 2


def test_receipt_requires_existing_task(conn):
    with pytest.raises(tdb.TaskNotFoundError):
        tdb.record_receipt(conn, "t_missing", receipt_id="x")


# ── retries + fallback ──────────────────────────────────────────────────


def test_retry_moves_to_retrying_until_limit(conn):
    task = tdb.create_task(conn, title="Flaky", max_retries=2)
    tdb.set_state(conn, task["id"], "WORKING")

    r1 = tdb.record_outcome(conn, task["id"], result="failure", detail="timeout", retry=True)
    assert r1["state"] == "RETRYING"
    assert r1["retry_count"] == 1

    tdb.set_state(conn, task["id"], "WORKING")
    r2 = tdb.record_outcome(conn, task["id"], result="failure", detail="timeout again", retry=True)
    assert r2["state"] == "RETRYING"
    assert r2["retry_count"] == 2


def test_retry_limit_exceeded_blocks_and_records_fallback(conn):
    task = tdb.create_task(conn, title="Doomed", max_retries=1)
    tdb.set_state(conn, task["id"], "WORKING")
    tdb.record_outcome(conn, task["id"], result="failure", retry=True)
    tdb.set_state(conn, task["id"], "WORKING")

    result = tdb.record_outcome(
        conn, task["id"], result="failure", retry=True, fallback="escalate to on-call"
    )
    assert result["state"] == "BLOCKED"
    assert result["retry_count"] == 2
    assert "retry limit exceeded" in result["blocker"]
    assert result["fallback"] == "escalate to on-call"


def test_outcome_without_retry_flag_does_not_change_state(conn):
    task = tdb.create_task(conn, title="Just logging")
    tdb.set_state(conn, task["id"], "WORKING")
    result = tdb.record_outcome(conn, task["id"], result="failure", detail="noted")
    assert result["state"] == "WORKING"
    assert result["retry_count"] == 0


def test_invalid_outcome_result_rejected(conn):
    task = tdb.create_task(conn, title="Bad result")
    with pytest.raises(ValueError):
        tdb.record_outcome(conn, task["id"], result="maybe")


# ── aging ────────────────────────────────────────────────────────────────


def test_aging_warn_tier_flags_without_state_change(conn):
    task = tdb.create_task(conn, title="Getting old")
    _touch_state_changed_at(conn, task["id"], hours_ago=25)

    flagged = tdb.age_check(conn, enabled=True, warn_hours=24, stale_hours=72)
    assert len(flagged) == 1
    assert flagged[0]["tier"] == "warn"
    assert tdb.get_task(conn, task["id"])["state"] == "NEW"


def test_aging_stale_tier_auto_transitions(conn):
    task = tdb.create_task(conn, title="Really old")
    tdb.set_state(conn, task["id"], "WORKING")
    _touch_state_changed_at(conn, task["id"], hours_ago=73)

    flagged = tdb.age_check(conn, enabled=True, warn_hours=24, stale_hours=72)
    assert len(flagged) == 1
    assert flagged[0]["tier"] == "stale"
    assert tdb.get_task(conn, task["id"])["state"] == "STALE"


def test_aging_does_not_repeat_within_same_window(conn):
    """No notification spam: the same threshold crossing only fires once."""
    task = tdb.create_task(conn, title="Silent")
    _touch_state_changed_at(conn, task["id"], hours_ago=25)

    first = tdb.age_check(conn, enabled=True, warn_hours=24, stale_hours=72)
    assert len(first) == 1

    second = tdb.age_check(conn, enabled=True, warn_hours=24, stale_hours=72)
    assert second == []


def test_aging_dry_run_never_mutates(conn):
    task = tdb.create_task(conn, title="Preview only")
    tdb.set_state(conn, task["id"], "WORKING")
    _touch_state_changed_at(conn, task["id"], hours_ago=73)

    flagged = tdb.age_check(conn, enabled=True, dry_run=True, warn_hours=24, stale_hours=72)
    assert len(flagged) == 1
    assert tdb.get_task(conn, task["id"])["state"] == "WORKING"

    # A live run afterwards still fires — dry-run left no marker behind.
    live = tdb.age_check(conn, enabled=True, dry_run=False, warn_hours=24, stale_hours=72)
    assert len(live) == 1
    assert tdb.get_task(conn, task["id"])["state"] == "STALE"


def test_aging_disabled_flag_never_mutates_even_when_flagged(conn):
    task = tdb.create_task(conn, title="Shadow")
    tdb.set_state(conn, task["id"], "WORKING")
    _touch_state_changed_at(conn, task["id"], hours_ago=73)

    flagged = tdb.age_check(conn, enabled=False, warn_hours=24, stale_hours=72)
    assert len(flagged) == 1  # still computed/returned for introspection
    assert tdb.get_task(conn, task["id"])["state"] == "WORKING"  # but not applied


def test_aging_resets_after_state_change(conn):
    task = tdb.create_task(conn, title="Revived")
    _touch_state_changed_at(conn, task["id"], hours_ago=25)
    tdb.age_check(conn, enabled=True, warn_hours=24, stale_hours=72)

    tdb.set_state(conn, task["id"], "WORKING")  # moves state_changed_at forward
    flagged = tdb.age_check(conn, enabled=True, warn_hours=24, stale_hours=72)
    assert flagged == []  # fresh anchor, not yet 24h old again


def test_aging_ignores_terminal_tasks(conn):
    task = tdb.create_task(conn, title="Finished")
    tdb.set_state(conn, task["id"], "CANCELLED")
    _touch_state_changed_at(conn, task["id"], hours_ago=200)

    flagged = tdb.age_check(conn, enabled=True, warn_hours=24, stale_hours=72)
    assert flagged == []
