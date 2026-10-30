"""Session job firing: CAS exactly-once, logical reschedule, remaining, cascade."""

import pytest

from hermes_state import SessionDB
from scheduler.engine import SchedulingEngine


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    database.create_session("s", "cli", model="m")
    try:
        yield database
    finally:
        database.close()


def _engine(db, calls):
    return SchedulingEngine(
        register_builtins=False,
        job_loader=lambda sid: db.list_session_jobs(sid, enabled_only=True),
        job_claim=lambda jid, due: db.claim_session_job(jid, expected_due=due),
        job_runner=lambda job, seq: calls.append((job["id"], seq)),
    )


def test_due_job_fires_once_and_advances_by_its_stride(db):
    job_id = db.create_session_job(origin_session_id="s", kind="step", every=2,
                                   prompt="p", next_due_seq=2)
    calls = []
    engine = _engine(db, calls)
    engine.ensure_session("s", step_seq=0)
    engine.step_committed("s", step_index=1)
    assert calls == []
    engine.step_committed("s", step_index=2)
    assert calls == [(job_id, 2)]
    assert db.get_session_job(job_id)["next_due_seq"] == 4
    engine.step_committed("s", step_index=3)
    assert calls == [(job_id, 2)]
    engine.step_committed("s", step_index=4)
    assert calls == [(job_id, 2), (job_id, 4)]


def test_remaining_reaches_zero_and_disables_the_job(db):
    job_id = db.create_session_job(origin_session_id="s", kind="turn", every=1,
                                   prompt="p", next_due_seq=1, remaining=1)
    calls = []
    engine = _engine(db, calls)
    engine.ensure_session("s", turn_seq=0)
    engine.turn_committed("s", turn_seq=1)
    assert calls == [(job_id, 1)]
    job = db.get_session_job(job_id)
    assert (job["enabled"], job["remaining"]) == (0, 0)
    engine.turn_committed("s", turn_seq=2)
    assert calls == [(job_id, 1)]


def test_lost_cas_does_not_dispatch(db):
    job_id = db.create_session_job(origin_session_id="s", kind="step", every=2,
                                   prompt="p", next_due_seq=2)
    calls = []
    engine = SchedulingEngine(
        register_builtins=False,
        job_loader=lambda sid: db.list_session_jobs(sid, enabled_only=True),
        job_claim=lambda jid, due: False,  # another process won the claim
        job_runner=lambda job, seq: calls.append(job["id"]),
    )
    engine.ensure_session("s", step_seq=0)
    engine.step_committed("s", step_index=2)
    assert calls == []
    assert db.get_session_job(job_id)["next_due_seq"] == 2


def test_refresh_picks_up_a_new_job_in_a_live_session(db):
    calls = []
    engine = _engine(db, calls)
    engine.ensure_session("s", step_seq=0)
    job_id = db.create_session_job(origin_session_id="s", kind="step", every=1,
                                   prompt="p", next_due_seq=1)
    engine.refresh_session_jobs("s")
    engine.step_committed("s", step_index=1)
    assert calls == [(job_id, 1)]


def test_session_deletion_cascades_to_jobs_and_keeps_run_history(db):
    job_id = db.create_session_job(origin_session_id="s", kind="step", every=1,
                                   prompt="p", next_due_seq=1)
    db.record_session_job_run(job_id, fired_at_seq=1, status="ok", origin_session_id="s")
    db._execute_write(lambda conn: conn.execute("DELETE FROM sessions WHERE id = 's'"))
    assert db.get_session_job(job_id) is None
    # Run history is an append-only ledger (no FK): the fire happened, keep the receipt.
    assert len(db.session_job_runs(job_id)) == 1


def test_end_session_pauses_its_jobs(db):
    job_id = db.create_session_job(origin_session_id="s", kind="step", every=1,
                                   prompt="p", next_due_seq=1)
    db.end_session("s", "cli_close")
    job = db.get_session_job(job_id)
    assert (job["enabled"], job["paused_reason"]) == (0, "origin session ended")


def test_job_never_advances_its_own_origin_clock(db):
    """The fired run is a different session; the origin clock only moves on real steps."""
    job_id = db.create_session_job(origin_session_id="s", kind="step", every=1,
                                   prompt="p", next_due_seq=1)
    calls = []
    engine = _engine(db, calls)
    engine.ensure_session("s", step_seq=0)
    engine.step_committed("s", step_index=1)
    assert db.get_session("s")["step_seq"] == 0  # engine ticks never write session counters
    assert calls == [(job_id, 1)]
