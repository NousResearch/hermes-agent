"""Deterministic KI-007 session/export lifecycle coverage.

This exercises the canonical SessionDB owner through the lifecycle that was
historically reported as producing ``session: null``.  It intentionally uses
two handles for the concurrent close/export case: a close belongs to the
connection being retired, while another reconnecting reader must remain able
to export the durable row.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from hermes_state import SessionDB


def _seed_session(db: SessionDB, session_id: str) -> None:
    db.create_session(
        session_id,
        source="desktop",
        user_id="user-ki007",
        session_key="desktop:task-ki007",
        chat_id="chat-ki007",
        chat_type="direct",
        thread_id="thread-ki007",
        model_config='{"task_id":"task-ki007","browser_task_id":"browser-ki007"}',
    )
    db.append_message(session_id, "user", "Run the deterministic KI-007 export check")
    db.append_message(
        session_id,
        "assistant",
        "The session is durably persisted.",
        tool_calls=[{"id": "call-ki007", "name": "browser_status"}],
    )
    db.append_message(
        session_id,
        "tool",
        '{"success":true,"result_ref":"result://ki007"}',
        tool_name="browser_status",
        tool_call_id="call-ki007",
    )


def test_ki007_session_export_survives_rotation_finalization_and_reconnect(tmp_path):
    db_path = tmp_path / "state.db"
    session_id = "ki007-session"

    db = SessionDB(db_path=db_path)
    _seed_session(db, session_id)
    assert db.get_session(session_id)["id"] == session_id
    assert db.export_session(session_id)["id"] == session_id

    # A normal session rotation/finalization changes lifecycle fields, not the
    # durable identity used by export.
    db.end_session(session_id, "desktop_finalized")
    finalized = db.export_session(session_id)
    assert finalized is not None
    assert finalized["id"] == session_id
    assert finalized["end_reason"] == "desktop_finalized"
    assert len(finalized["messages"]) == 3
    db.close()

    # Reconnect through a fresh canonical handle, as a restarted gateway or
    # desktop backend would.
    reconnected = SessionDB(db_path=db_path)
    assert reconnected.get_session(session_id)["session_key"] == "desktop:task-ki007"
    lineage = reconnected.export_session_lineage(session_id)
    assert lineage is not None
    assert lineage["lineage_session_ids"] == [session_id]
    assert [message["role"] for message in lineage["messages"]] == [
        "user", "assistant", "tool"
    ]
    reconnected.close()


def test_ki007_concurrent_close_and_export_do_not_turn_a_durable_session_null(tmp_path):
    db_path = tmp_path / "state.db"
    session_id = "ki007-concurrent"
    writer = SessionDB(db_path=db_path)
    _seed_session(writer, session_id)

    exporting = SessionDB(db_path=db_path)
    retiring = SessionDB(db_path=db_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        export_future = pool.submit(exporting.export_session, session_id)
        close_future = pool.submit(retiring.close)
        exported = export_future.result()
        close_future.result()

    assert exported is not None
    assert exported["id"] == session_id
    assert len(exported["messages"]) == 3

    # The remaining writer handle and a subsequent reconnect still see the
    # same row; there is no process-local ``session`` object to lose.
    writer.end_session(session_id, "worker_complete")
    assert writer.export_session(session_id)["id"] == session_id
    writer.close()
    reconnected = SessionDB(db_path=db_path)
    assert reconnected.export_session(session_id)["id"] == session_id
    reconnected.close()
