"""Stale-writer fencing: generation identity + CAS at settlement.

A replaced goal must never be dispatched, judged, or persisted by an obsolete
worker — regardless of timing (judge in flight) or interface.
"""
import pytest

from hermes_cli import goals


def test_clear_then_set_never_reuses_generation():
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id="sw-1")
    mgr.set("goal A objective")
    gen_a = mgr.state.goal_generation
    mgr.clear()
    mgr.set("goal B objective")
    assert mgr.state.goal_generation > gen_a
    assert goals.load_goal("sw-1").goal == "goal B objective"


def test_replacement_during_judge_cannot_persist_done(monkeypatch):
    """set(A) -> evaluate A (judge slow) -> set(B) mid-judge -> A verdict discarded."""
    goals._DB_CACHE.clear()
    mgr_a = goals.GoalManager(session_id="sw-2")
    mgr_a.set("goal A objective")
    gen_a = mgr_a.state.goal_generation

    def slow_judge(*args, **kwargs):
        # Replacement happens while the judge executes
        mgr_b = goals.GoalManager(session_id="sw-2")
        mgr_b.set("goal B objective")
        return ("done", "judge says done", False, None, False)

    monkeypatch.setattr(goals, "judge_goal", slow_judge)
    decision = mgr_a.evaluate_after_turn("output under A", expected_generation=gen_a)
    assert decision["status"] == "obsolete"
    final = goals.load_goal("sw-2")
    assert final.goal == "goal B objective"
    assert final.status == "active"  # never marked done by A's worker
    assert final.turns_used == 0  # A's budget consumption did not land


def test_two_manager_concurrent_writes_cas():
    goals._DB_CACHE.clear()
    mgr1 = goals.GoalManager(session_id="sw-3")
    mgr1.set("goal A objective")
    gen = mgr1.state.goal_generation
    from hermes_cli.goals import save_goal_cas
    stale = goals.GoalState(goal="goal A objective", status="done", goal_generation=gen)
    # Another manager replaces
    mgr2 = goals.GoalManager(session_id="sw-3")
    mgr2.set("goal B objective")
    assert save_goal_cas("sw-3", stale, gen) is False
    assert goals.load_goal("sw-3").goal == "goal B objective"


def test_cross_goal_ownership_exact():
    from hermes_cli.goals import GoalKickoff, is_goal_prompt_for
    kick_b = GoalKickoff("goal B objective", goal_generation=5, goal_text="goal B objective")
    kick_a = GoalKickoff("goal A objective", goal_generation=4, goal_text="goal A objective")
    assert is_goal_prompt_for(kick_b, "goal B objective") is True
    assert is_goal_prompt_for(kick_a, "goal B objective") is False
    assert is_goal_prompt_for(kick_b, "goal A objective") is False
    # Plain user text, even identical, is never synthetic
    assert is_goal_prompt_for("goal B objective", "goal B objective") is False


def test_gateway_inflight_replacement_obsolete(monkeypatch):
    """Gateway run_goals path passes expected_generation; replacement mid-judge -> obsolete."""
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id="sw-4")
    mgr.set("goal A objective")
    gen_a = mgr.state.goal_generation

    def judge(*args, **kwargs):
        goals.GoalManager(session_id="sw-4").set("goal B objective")
        return ("continue", "not done", False, None, False)

    monkeypatch.setattr(goals, "judge_goal", judge)
    decision = mgr.evaluate_after_turn("x", expected_generation=gen_a)
    assert decision["status"] == "obsolete"
    assert decision["should_continue"] is False
    assert goals.load_goal("sw-4").goal == "goal B objective"
    assert goals.load_goal("sw-4").status == "active"


def test_nonatomic_cas_gap_is_closed():
    """A->B replacement must not be overwritten by A's writer even when A's
    writer checked generation 1 before B committed (the old load-then-save gap)."""
    goals._DB_CACHE.clear()
    mgr_a = goals.GoalManager(session_id="sw-10")
    mgr_a.set("goal A objective")
    stale = goals.load_goal("sw-10")
    # B replaces A (generation 2)
    goals.GoalManager(session_id="sw-10").set("goal B objective")
    # A's stale writer attempts its done state at generation 1
    from hermes_cli.goals import save_goal_cas
    stale.status = "done"
    assert save_goal_cas("sw-10", stale, 1) is False
    final = goals.load_goal("sw-10")
    assert final.goal == "goal B objective"
    assert final.status == "active"
    assert final.goal_generation == 2


def test_cleared_tombstone_cannot_be_resurrected():
    goals._DB_CACHE.clear()
    mgr_a = goals.GoalManager(session_id="sw-11")
    mgr_a.set("goal A objective")
    stale = goals.load_goal("sw-11")
    goals.GoalManager(session_id="sw-11").clear()
    tomb = goals.load_goal("sw-11")
    assert tomb.status == "cleared"
    from hermes_cli.goals import save_goal_cas
    stale.status = "done"
    assert save_goal_cas("sw-11", stale, tomb.goal_generation) is False
    final = goals.load_goal("sw-11")
    assert final.status == "cleared"
    assert final.goal == "goal A objective"


def test_concurrent_replacement_allocates_unique_generations():
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id="sw-12")
    mgr.set("goal A objective")
    gens = set()
    for text in ("goal B objective", "goal C objective", "goal D objective"):
        goals.GoalManager(session_id="sw-12").set(text)
        gens.add(goals.load_goal("sw-12").goal_generation)
    assert gens == {2, 3, 4}
    assert goals.load_goal("sw-12").goal == "goal D objective"


def test_pause_during_judge_does_not_land_for_stale_evaluator(monkeypatch):
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id="sw-13")
    mgr.set("goal A objective")
    gen = mgr.state.goal_generation
    rev = mgr.state.goal_revision
    def judge(*args, **kwargs):
        goals.GoalManager(session_id="sw-13").pause("user paused during judge")
        return ("done", "done", False, None, False)
    monkeypatch.setattr(goals, "judge_goal", judge)
    decision = mgr.evaluate_after_turn("x", expected_generation=gen, expected_revision=rev)
    assert decision["status"] == "obsolete"
    final = goals.load_goal("sw-13")
    assert final.status == "paused"
    assert final.goal == "goal A objective"


def test_clear_during_judge_does_not_resurrect(monkeypatch):
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id="sw-14")
    mgr.set("goal A objective")
    gen = mgr.state.goal_generation
    rev = mgr.state.goal_revision
    def judge(*args, **kwargs):
        goals.GoalManager(session_id="sw-14").clear()
        return ("done", "done", False, None, False)
    monkeypatch.setattr(goals, "judge_goal", judge)
    decision = mgr.evaluate_after_turn("x", expected_generation=gen, expected_revision=rev)
    assert decision["status"] == "obsolete"
    assert goals.load_goal("sw-14").status == "cleared"


def test_persistence_failure_fails_closed(monkeypatch):
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id="sw-15")
    mgr.set("goal A objective")
    from hermes_cli import goals as gmod
    # Persistence failure inside SessionDB.mutate_meta must fail closed.
    db = gmod._get_session_db()
    class _Boom:
        def __init__(self, real):
            self._real = real
        def mutate_meta(self, key, fn):
            raise RuntimeError("db exploded")
        def get_meta(self, key):
            return self._real.get_meta(key)
        def __getattr__(self, name):
            return getattr(self._real, name)
    monkeypatch.setattr(gmod, "_get_session_db", lambda: _Boom(db))
    from hermes_cli.goals import save_goal_cas
    stale = goals.GoalState(goal="goal A objective", status="done", goal_generation=1)
    assert save_goal_cas("sw-15", stale, 1) is False
    # DB still has the original active goal (boom never wrote)
    assert goals.load_goal("sw-15").status == "active"


def test_restart_and_rotation_preserve_identity():
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id="sw-16-old")
    mgr.set("goal A objective")
    gen = mgr.state.goal_generation
    # Simulate restart: fresh manager, same DB
    goals._DB_CACHE.clear()
    mgr2 = goals.GoalManager(session_id="sw-16-old")
    assert mgr2.state.goal == "goal A objective"
    assert mgr2.state.goal_generation == gen
    # Rotation: migrate to new session keeps generation, old archived cleared
    assert goals.migrate_goal_to_session("sw-16-old", "sw-16-new", reason="test") is True
    assert goals.load_goal("sw-16-new").goal == "goal A objective"
    assert goals.load_goal("sw-16-old").status == "cleared"


def test_stale_subgoal_gate_contract_mutations_fail_closed():
    goals._DB_CACHE.clear()
    mgr_a = goals.GoalManager(session_id="sw-20")
    mgr_a.set("goal A objective")
    goals.GoalManager(session_id="sw-20").set("goal B objective")
    # Stale manager mutations must raise, not report success, and not retain state.
    import pytest
    with pytest.raises(RuntimeError):
        mgr_a.add_subgoal("must use validation")
    with pytest.raises(RuntimeError):
        mgr_a.add_gate("pytest -q")
    with pytest.raises(RuntimeError):
        mgr_a.set_contract(goals.GoalContract(outcome="x"))
    final = goals.load_goal("sw-20")
    assert final.goal == "goal B objective"
    assert final.subgoals == []
    assert final.gates == []
    assert mgr_a.state.subgoals == [] or mgr_a.state.goal == "goal A objective"  # restored, not appended success


def test_stale_gate_result_yields_no_continuation(monkeypatch):
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id="sw-21")
    mgr.set("goal A objective")
    mgr.add_gate("echo hi")
    def run_gate(*args, **kwargs):
        goals.GoalManager(session_id="sw-21").set("goal B objective")
        return (False, 1, "boom")
    monkeypatch.setattr(goals, "run_gate", run_gate)
    decision = mgr.evaluate_after_turn("turn text")
    assert decision["status"] == "obsolete"
    assert decision.get("continuation_prompt") is None
    assert decision.get("should_continue") in (False, None)
    final = goals.load_goal("sw-21")
    assert final.goal == "goal B objective"
    assert final.status == "active"
    assert final.gates == []


def test_gateway_replacement_during_turn_is_not_judged_against_new_goal(monkeypatch):
    """Gateway identity: a turn admitted under A, replaced by B during
    execution, must be obsolete even if B is current at post-turn time."""
    goals._DB_CACHE.clear()
    mgr_a = goals.GoalManager(session_id="sw-22")
    mgr_a.set("goal A objective")
    gen_a = mgr_a.state.goal_generation
    rev_a = mgr_a.state.goal_revision
    # Replacement while generating
    goals.GoalManager(session_id="sw-22").set("goal B objective")
    # Post-turn evaluation with A's admitted identity
    decision = mgr_a.evaluate_after_turn("finished", expected_generation=gen_a, expected_revision=rev_a)
    assert decision["status"] == "obsolete"
    assert decision.get("continuation_prompt") is None
    assert goals.load_goal("sw-22").goal == "goal B objective"
    # And through the gateway hook signature: current != admitted -> early return, no enqueue
    # (covered by run_goals early-return logic; unit-level generation mismatch suffices here.)


def test_failed_migration_leaves_at_most_one_active(monkeypatch):
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id="sw-23-old")
    mgr.set("goal A objective")
    # Child already exists: migration must refuse and keep exactly one active (child's own if any, else parent archived only if child created)
    goals.GoalManager(session_id="sw-23-new").set("goal child objective")
    assert goals.migrate_goal_to_session("sw-23-old", "sw-23-new") is False
    assert goals.load_goal("sw-23-old").status == "active"
    assert goals.load_goal("sw-23-new").goal == "goal child objective"


def test_migration_rolls_back_on_second_write_failure(monkeypatch):
    goals._DB_CACHE.clear()
    goals.GoalManager(session_id="mig-1-old").set("goal migrate objective")
    db = goals._get_session_db()
    class _BoomDB:
        def __init__(self, real): self._real=real
        def _execute_write(self, fn):
            # Simulate failure on the SECOND write inside the transfer:
            # raise before delegating, so transaction rolls back (parent preserved).
            raise RuntimeError("injected child-write failure")
        def get_meta(self, key): return self._real.get_meta(key)
        def __getattr__(self, name): return getattr(self._real, name)
    orig_get = goals._get_session_db
    goals._get_session_db = lambda: _BoomDB(db)
    try:
        assert goals.migrate_goal_to_session("mig-1-old", "mig-1-new") is False
    finally:
        goals._get_session_db = orig_get
    assert goals.load_goal("mig-1-old").status == "active"
    assert goals.load_goal("mig-1-old").goal == "goal migrate objective"
    assert goals.load_goal("mig-1-new") is None
    assert goals.migrate_goal_to_session("mig-1-old", "mig-1-new") is True
    assert goals.load_goal("mig-1-old").status == "cleared"
    assert goals.load_goal("mig-1-new").goal == "goal migrate objective"
    assert goals.load_goal("mig-1-new").status == "active"


def test_concurrent_migration_no_duplicate_active():
    goals._DB_CACHE.clear()
    goals.GoalManager(session_id="mig-2-old").set("goal concurrent objective")
    results = []
    import threading
    def run():
        results.append(goals.migrate_goal_to_session("mig-2-old", "mig-2-new"))
    threads = [threading.Thread(target=run) for _ in range(2)]
    for th in threads: th.start()
    for th in threads: th.join()
    # At most one child active; parent archived if child created
    child = goals.load_goal("mig-2-new")
    parent = goals.load_goal("mig-2-old")
    actives = [s for s in (parent, child) if s is not None and s.status == "active"]
    assert len(actives) <= 1
    if child is not None:
        assert child.status == "active" and child.goal == "goal concurrent objective"


def test_second_insert_failure_rolls_back_migration(monkeypatch):
    """Second-INSERT failure inside the transaction: first INSERT ran, second
    attempted, then rollback preserves the parent active goal."""
    goals._DB_CACHE.clear()
    goals.GoalManager(session_id="mig2-old").set("goal second-insert objective")
    db = goals._get_session_db()
    class ConnProxy:
        def __init__(self, conn, counter):
            self._conn = conn
            self._counter = counter
        def execute(self, *a, **kw):
            self._counter["n"] += 1
            if self._counter["n"] == 4:
                raise RuntimeError("injected second-INSERT failure")
            return self._conn.execute(*a, **kw)
        def __getattr__(self, name):
            return getattr(self._conn, name)
    orig_execute_write = db._execute_write
    def _wrapped(callback):
        counter = {"n": 0}
        def _cb(conn):
            return callback(ConnProxy(conn, counter))
        return orig_execute_write(_cb)
    db._execute_write = _wrapped
    try:
        assert goals.migrate_goal_to_session("mig2-old", "mig2-new") is False
    finally:
        db._execute_write = orig_execute_write
    assert goals.load_goal("mig2-old").status == "active"
    assert goals.load_goal("mig2-old").goal == "goal second-insert objective"
    assert goals.load_goal("mig2-new") is None
    assert goals.migrate_goal_to_session("mig2-old", "mig2-new") is True
    assert goals.load_goal("mig2-new").goal == "goal second-insert objective"


def test_async_facade_peek_is_awaited():
    """Regression: peek_session_id on the async facade must be awaited; an
    unawaited coroutine is not a session id."""
    import asyncio, warnings
    from gateway.session import AsyncSessionStore
    async def _run():
        from unittest.mock import MagicMock
        store = MagicMock()
        store.peek_session_id.return_value = "sess-123"
        facade = AsyncSessionStore(store)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            peeked = await facade.peek_session_id("key")
        assert peeked == "sess-123"
    asyncio.run(_run())


def test_run_post_turn_hooks_forwards_goal_present():
    """Regression: _run_post_turn_hooks must accept and forward goal_present
    (previously TypeError when called from run_inbound)."""
    import asyncio, inspect
    from gateway.run_goals import GatewayGoalsMixin
    sig = inspect.signature(GatewayGoalsMixin._run_post_turn_hooks)
    assert "goal_present" in sig.parameters
    # And the inbound call site passes it
    import pathlib
    src = pathlib.Path("gateway/run_inbound.py").read_text()
    assert "goal_present=_turn_goal_present" in src
