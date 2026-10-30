"""Durable registration: identical spec keeps cadence, changed spec resets once, admin wins."""

import pytest

from hermes_state import SessionDB
from scheduler.engine import SchedulingEngine
from scheduler.plugin_api import register_scheduled_task, spec_hash
from scheduler.tasks import TaskSpec


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _spec(every=10, phase="post", handler=None):
    return TaskSpec(plugin_id="acme", name="acme.cache-audit", period="step", every=every,
                    phase=phase, handler=handler or (lambda ctx: None))


def _state(db):
    return db.get_task_state("acme", "acme.cache-audit")


def test_first_registration_writes_state_and_activates(db):
    engine = SchedulingEngine(register_builtins=False)
    handle = register_scheduled_task(_spec(), engine=engine, db=db)
    assert handle is not None
    state = _state(db)
    assert (state["period"], state["every"], state["phase"], state["enabled"]) == ("step", 10, "post", 1)
    assert state["spec_hash"] == spec_hash(_spec())
    assert engine.active


def test_identical_spec_keeps_cadence_and_breaker_state(db):
    engine = SchedulingEngine(register_builtins=False)
    register_scheduled_task(_spec(), engine=engine, db=db)
    db._execute_write(lambda conn: conn.execute(
        "UPDATE task_state SET error_streak = 2, run_count = 7, last_status = 'ok'"))
    register_scheduled_task(_spec(), engine=engine, db=db)
    state = _state(db)
    assert (state["error_streak"], state["run_count"], state["last_status"]) == (2, 7, "ok")


def test_changed_spec_resets_once_and_writes_an_audit_row(db):
    engine = SchedulingEngine(register_builtins=False)
    register_scheduled_task(_spec(every=10), engine=engine, db=db)
    db._execute_write(lambda conn: conn.execute("UPDATE task_state SET error_streak = 2, run_count = 7"))
    register_scheduled_task(_spec(every=3, phase="pre"), engine=engine, db=db)
    state = _state(db)
    assert (state["every"], state["phase"], state["error_streak"], state["run_count"]) == (3, "pre", 0, 7)
    [audit] = [run for run in db.task_runs() if run["status"] == "rescheduled"]
    assert "every 10→3" in audit["error"] and "phase post→pre" in audit["error"]


def test_admin_disable_survives_a_spec_change_and_stays_unregistered(db):
    engine = SchedulingEngine(register_builtins=False)
    register_scheduled_task(_spec(), engine=engine, db=db)
    db.set_task_enabled("acme", "acme.cache-audit", enabled=False, disabled_by="admin")
    engine.unregister("acme", "acme.cache-audit")
    assert register_scheduled_task(_spec(every=4), engine=engine, db=db) is None
    state = _state(db)
    assert (state["enabled"], state["disabled_by"], state["every"]) == (0, "admin", 4)


def test_plugin_unload_disable_is_cleared_on_the_next_registration(db):
    engine = SchedulingEngine(register_builtins=False)
    handle = register_scheduled_task(_spec(), engine=engine, db=db)
    assert handle is not None
    handle.release("plugin unloaded")
    assert engine.active is False
    assert _state(db)["disabled_by"] == "plugin unloaded"
    assert register_scheduled_task(_spec(), engine=engine, db=db) is not None
    state = _state(db)
    assert (state["enabled"], state["disabled_by"]) == (1, None)


def test_spec_hash_covers_every_phase_and_period():
    base = spec_hash(_spec())
    assert spec_hash(_spec(every=11)) != base
    assert spec_hash(_spec(phase="pre")) != base
    turn = _spec()
    assert spec_hash(TaskSpec(plugin_id="acme", name="acme.t", period="turn", every=10,
                              phase="post", handler=turn.handler)) != spec_hash(
        TaskSpec(plugin_id="acme", name="acme.t", period="step", every=10, phase="post", handler=turn.handler))


def test_engine_run_sink_records_skipped_and_ok_outcomes(db):
    runs = []
    engine = SchedulingEngine(register_builtins=False, handler_timeout_ms=200,
                              run_sink=lambda spec, ctx, outcome: runs.append((spec.name, outcome.status)))
    engine.register(TaskSpec(plugin_id="acme", name="acme.probe", period="step", every=1,
                             phase="post", handler=lambda ctx: None))
    engine.ensure_session("s")
    engine.step_committed("s", step_index=1)
    assert runs == [("acme.probe", "ok")]
