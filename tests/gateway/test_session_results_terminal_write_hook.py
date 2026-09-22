"""The result owner forwards terminal metadata writes only for first settlement."""
import json

from gateway.session_results import admission_result, finish_result
from hermes_state import SessionDB
from hermes_state_runtime import (
    admit_session_input,
    begin_runtime_epoch,
    claim_session_input,
)


def test_finish_result_forwards_terminal_write_once_and_replay_reads_saved_result(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", source="test")
        epoch = begin_runtime_epoch(db, instance_id="owner")
        admitted = admit_session_input(
            db,
            epoch=epoch,
            principal_id="human",
            session_id="s",
            request_id="finish",
            payload={"text": "input"},
        )
        started = claim_session_input(db, epoch=epoch, session_id="s")
        result = {
            "result": {"final_response": "durable reply", "messages": [], "failed": True},
            "usage": {"output_tokens": 3},
        }
        calls = []
        key = "test.finish_result.terminal_write"

        def terminal_write(conn, row, outcome, callback_result):
            calls.append((conn, row, outcome, callback_result))
            assert conn.in_transaction
            conn.execute(
                "INSERT INTO state_meta(key,value) VALUES(?,?)",
                (key, json.dumps({"admission_id": row["admission_id"], "outcome": outcome})),
            )

        settled, response = finish_result(
            db,
            epoch=epoch,
            row=started,
            response="delivery response",
            outcome="completed",
            result=result,
            _terminal_write=terminal_write,
        )

        assert response == "delivery response"
        assert (settled["status"], settled["outcome"]) == ("terminal", "failed")
        assert len(calls) == 1
        conn, callback_row, callback_outcome, callback_result = calls[0]
        assert conn is db._conn
        assert callback_row == started
        assert callback_outcome == "failed"
        assert callback_result is result
        assert result["result"]["completed"] is False
        assert admission_result(db, admitted["admission_id"]) == result
        saved_metadata = db._read_one("SELECT value FROM state_meta WHERE key=?", (key,))[0]
        revision = db.get_session("s")["runtime_revision"]

        def replay_callback(*args):
            raise AssertionError(f"terminal replay invoked callback: {args!r}")

        replayed, replay_response = finish_result(
            db,
            epoch=epoch,
            row=started,
            response="replacement response",
            outcome="interrupted",
            result={"result": {"final_response": "replacement"}, "usage": {}},
            _terminal_write=replay_callback,
        )

        assert {
            key: replayed[key] for key in ("admission_id", "status", "outcome", "generation")
        } == {
            key: settled[key] for key in ("admission_id", "status", "outcome", "generation")
        }
        assert replay_response == "durable reply"
        assert len(calls) == 1
        assert admission_result(db, admitted["admission_id"]) == result
        assert db._read_one("SELECT value FROM state_meta WHERE key=?", (key,))[0] == saved_metadata
        assert db.get_session("s")["runtime_revision"] == revision
    finally:
        db.close()
