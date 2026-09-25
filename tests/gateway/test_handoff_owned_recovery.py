"""A row a COMPLETED ``/handoff`` owns must stay recoverable even when the source interface's
teardown stamps a non-recoverable ``end_reason`` (``cli_close``) after ownership moved to the
gateway; otherwise the stale-route self-heal mints a fresh empty session and orphans the leg.
Only ``handoff_state='completed'`` widens the contract — ``failed``/``pending`` stay closed.
"""

import time

import pytest

from hermes_state import SessionDB

PEER = dict(
    source="qqbot",
    user_id="D425866BDDB69A1305199BAE0B166AFB",
    session_key="agent:main:qqbot:dm:D425866BDDB69A1305199BAE0B166AFB",
    chat_id="D425866BDDB69A1305199BAE0B166AFB",
    chat_type="dm",
    thread_id=None,
)


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    d.close()


def _mk_handed_off(db, session_id, *, handoff_state):
    db.create_session(
        session_id, PEER["source"], user_id=PEER["user_id"],
        session_key=PEER["session_key"], chat_id=PEER["chat_id"], chat_type=PEER["chat_type"],
    )
    for i in range(3):
        db.append_message(session_id, "user" if i % 2 == 0 else "assistant", f"m{i}")
    with db._lock:
        db._conn.execute(
            "UPDATE sessions SET handoff_state=?, handoff_platform='qqbot', last_activity_at=?, "
            "ended_at=1000.0, end_reason='cli_close' WHERE id=?",
            (handoff_state, time.time(), session_id),
        )
        db._conn.commit()


def test_cli_close_after_completed_handoff_is_recoverable(db):
    """The incident: handoff completed, then the CLI teardown stamps cli_close."""
    _mk_handed_off(db, "handoff-leg", handoff_state="completed")
    assert db.find_latest_gateway_session_for_peer(**PEER)["id"] == "handoff-leg"


def test_failed_handoff_does_not_widen_recovery(db):
    """``handoff_state='failed'`` is not a handoff that owns the row: cli_close stays closed."""
    _mk_handed_off(db, "failed-handoff", handoff_state="failed")
    assert db.find_latest_gateway_session_for_peer(**PEER) is None
