import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt


def _claimed_admission(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s", source="test")
    epoch = rt.begin_runtime_epoch(db, instance_id="budget-owner")
    accepted = rt.admit_session_input(
        db,
        epoch=epoch,
        principal_id="human",
        session_id="s",
        request_id="turn",
        payload={"text": "work"},
    )
    claimed = rt.claim_session_input(db, epoch=epoch, session_id="s")
    assert claimed["admission_id"] == accepted["admission_id"]
    return db, epoch, claimed


def test_admission_tool_budget_is_atomic_and_pinned(tmp_path):
    db, epoch, claimed = _claimed_admission(tmp_path)
    args = dict(
        epoch=epoch,
        admission_id=claimed["admission_id"],
        generation=claimed["generation"],
        max_tool_executions=2,
    )
    try:
        first = rt.consume_admission_tool_execution(
            db, **args, tool_name="read_file", tool_call_id="one"
        )
        second = rt.consume_admission_tool_execution(
            db, **args, tool_name="terminal", tool_call_id="two"
        )
        denied = rt.consume_admission_tool_execution(
            db, **args, tool_name="search_files", tool_call_id="three"
        )

        assert first == {
            "allowed": True,
            "used_tool_executions": 1,
            "max_tool_executions": 2,
            "exhausted": False,
            "became_exhausted": False,
        }
        assert second["allowed"] is True
        assert second["used_tool_executions"] == 2
        assert second["became_exhausted"] is True
        assert denied["allowed"] is False
        assert denied["used_tool_executions"] == 2

        row = rt.get_session_admission(db, admission_id=claimed["admission_id"])
        assert row["tool_budget_max"] == 2
        assert row["tool_budget_used"] == 2

        with pytest.raises(rt.RuntimeStoreError, match="admission_conflict"):
            rt.consume_admission_tool_execution(
                db, **(args | {"max_tool_executions": 3}),
                tool_name="terminal", tool_call_id="changed-limit",
            )
        with pytest.raises(rt.RuntimeStoreError, match="stale_generation"):
            rt.consume_admission_tool_execution(
                db, **(args | {"generation": claimed["generation"] + 1}),
                tool_name="terminal", tool_call_id="stale",
            )
    finally:
        db.close()


def test_worker_budget_debit_is_receipt_idempotent(tmp_path):
    db, epoch, claimed = _claimed_admission(tmp_path)
    assignment = dict(
        execution_id="admission-worker:test-budget",
        session_id="s",
        generation=claimed["generation"],
    )
    try:
        rt.register_worker_execution(
            db,
            epoch=epoch,
            **assignment,
            kind="compute",
            adoption_secret="budget-secret",
        )
        payload = {
            "max_tool_executions": 2,
            "tool_name": "read_file",
            "tool_call_id": "call-1",
        }
        args = dict(
            epoch=epoch,
            **assignment,
            sequence=1,
            operation="budget.tool_execution",
            payload=payload,
        )
        first = rt.mutate_worker_execution(db, **args)
        assert first["allowed"] is True
        assert first["used_tool_executions"] == 1

        # Exact transport retry returns the receipt and cannot double debit.
        assert rt.mutate_worker_execution(db, **args) == first
        row = rt.get_session_admission(db, admission_id=claimed["admission_id"])
        assert row["tool_budget_used"] == 1

        with pytest.raises(rt.RuntimeStoreError, match="admission_conflict"):
            rt.mutate_worker_execution(
                db,
                **(args | {"payload": payload | {"tool_call_id": "different"}}),
            )

        second = rt.mutate_worker_execution(
            db,
            **(args | {
                "sequence": 2,
                "payload": payload | {"tool_call_id": "call-2"},
            }),
        )
        denied = rt.mutate_worker_execution(
            db,
            **(args | {
                "sequence": 3,
                "payload": payload | {"tool_call_id": "call-3"},
            }),
        )
        assert second["became_exhausted"] is True
        assert denied["allowed"] is False
        assert denied["used_tool_executions"] == 2

        row = rt.get_session_admission(db, admission_id=claimed["admission_id"])
        assert row["tool_budget_used"] == 2
        assert db._conn.execute(
            "SELECT last_sequence FROM worker_executions WHERE execution_id=?",
            (assignment["execution_id"],),
        ).fetchone()[0] == 3
    finally:
        db.close()
