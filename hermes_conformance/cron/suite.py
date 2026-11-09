"""The cron storage conformance suite: isolation and at-most-once invariants every backend meets.

Parametrized over backends through the ``backend`` fixture (:class:`Backend`). Each test talks to
the port only (``cron_store_port``); the fixture says how to bind each agent's scope and open
its cron store. Agents are named by short keys (``"a"``, ``"b"``); both belong to one tenant.

Covered:

- pinning: every mutating operation of the port (:data:`JOB_WRITES`, :data:`EXECUTION_WRITES`)
  is refused for another agent, another tenant and no pin, and a refusal changes nothing the
  agent can observe through the port (:func:`observe`: its jobs, their records, and its
  execution history); every read (:data:`ALL_READS`) is refused for the same pins; a foreign job
  or attempt reads like a missing one;
- the creating principal: a job records who created it, from the bound scope's actor, and no
  update can re-point it;
- the tick boundary: one agent cannot list, trigger, re-point or fire another agent's jobs, and
  a tick pinned to one agent never returns or fires another agent's due job;
- the fire claim: a due job is claimed once, two racing claimants produce exactly one winner, a
  fresh claim blocks every later claimant for the lease, a claim is heartbeated while the run is
  alive and released by its outcome, the fence holds a claim across a side effect, and a
  completed occurrence never fires twice; the scan dispatches a due one-shot once;
- the execution audit: an attempt is claimed, runs and finishes exactly once (a terminal attempt
  is never rewritten), a handoff is adopted once, the only post-terminal rewrite is a queued
  delivery outcome settling once, missed occurrences are booked, this owner's attempts are
  never swept as abandoned, and the firer may give its attempt up exactly once.

NOT covered. A backend that passes this suite may still violate these, and must not claim them:

- the ownership levels above agent (participant, owner-wide, tenant-wide), and the
  administration right (``Scope.can_administer``): the local backend stores the agent level only
  and its person always administers, so the fixture cannot vary them;
- actor authority (which principal may create, trigger or re-point a job at all);
- cross-process recovery (receipt settlement after a departed owner, the dead-owner sweep of
  another process's attempts): the fixture owns every attempt it makes;
- retention (completed one-shot records, terminal attempts and output files pruning);
- job validation and schedule semantics (what a schedule string means, which knobs combine):
  the suite only stores schedules the backend itself computed;
- transaction-local scope pinning (a pin change while an operation waits for a writer);
- the shape of operation results beyond the fields the tests read.
"""

from __future__ import annotations

import contextvars
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import pytest

from cron_store_port import CronPinRefused, CronStore


def _now() -> datetime:
    return datetime.now().astimezone()


def _past(seconds: float) -> str:
    return (_now() - timedelta(seconds=seconds)).isoformat()


def _future(seconds: float) -> str:
    return (_now() + timedelta(seconds=seconds)).isoformat()


@dataclass
class Backend:
    """How the suite drives one backend.

    - ``open(agent)``: the agent's cron store.
    - ``replica(agent)``: an independent second handle on the same agent's store (another
      replica's connection).
    - ``bind(agent, actor=None)``: bind the agent's scope (its owner, administering). With
      *actor*, bind that named principal as the actor instead — the suite uses it to vary who
      creates a job.
    - ``foreign_tenant(agent)``: bind a scope of ANOTHER tenant that names the same agent id.
    - ``unbound()``: run with no scope pinned. A local backend's pin is then the ambient profile,
      so its ``unbound`` names a third profile; a hosted one binds nothing.
    """

    open: Callable[[str], CronStore]
    replica: Callable[[str], CronStore]
    bind: Callable[..., AbstractContextManager]
    foreign_tenant: Callable[[str], AbstractContextManager]
    unbound: Callable[[], AbstractContextManager]


def _replica_thread(target: Callable[..., Any], *args: Any) -> threading.Thread:
    """An unstarted thread running *target* in a fresh, empty context: another replica's worker,
    which must not inherit this test's scope and binds its own."""
    # health: allow HX012 -- a replica's worker deliberately starts with no scope; each target binds its own
    return threading.Thread(target=contextvars.Context().run, args=(target, *args))


def _refused(call: Callable[[], Any]) -> str:
    """The refusal *call* raises (its message), asserting it is the store's pin refusal."""
    with pytest.raises(CronPinRefused) as refused:
        call()
    return str(refused.value)


def _foreign_pins(backend):
    """Every pin but agent ``a``'s own: another agent, another tenant naming ``a``, no pin."""
    return {"other_agent": lambda: backend.bind("b"), "other_tenant": lambda: backend.foreign_tenant("a"),
            "unbound": backend.unbound}


# --- the world every per-operation test runs against ---------------------------------------------

@dataclass
class World:
    """One agent's world: a recurring job due now (``due``), a paused job (``paused``), a future
    one-shot (``oneshot``), one completed attempt whose delivery outcome is still queued
    (``finished``) and one claimed attempt on ``due`` (``execution``, the newest)."""

    store: CronStore
    agent: str
    due: str
    paused: str
    oneshot: str
    execution: str
    finished: str


def make_world(backend, agent: str = "a") -> World:
    store = backend.open(agent)
    with backend.bind(agent):
        due = store.create_job(f"the due job of {agent}", "2h")["id"]
        store.update_job(due, {"next_run_at": _past(60)})
        paused = store.create_job(f"the paused job of {agent}", "0 9 * * *")["id"]
        store.pause_job(paused, reason="suite")
        oneshot = store.create_job(f"the one-shot of {agent}", _future(2 * 3600))["id"]
        finished = store.create_execution(due, source="suite")["id"]
        store.finish_execution(finished, success=True, delivery_outcome="queued")
        execution = store.create_execution(due, source="suite")["id"]
    return World(store, agent, due, paused, oneshot, execution, finished)


#: What a job's observable projection keeps: the creator, the schedule the backend honours, the
#: lifecycle flags, and the claims. Result fields the suite never reads are dropped.
_JOB_PROJECTION = ("prompt", "created_by", "enabled", "state", "paused_reason", "next_run_at",
                   "last_status", "last_run_at", "repeat", "fire_claim", "run_claim", "pending_slot")


def _job_projection(row: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    return {key: row.get(key) for key in _JOB_PROJECTION}


def _execution_projection(row: dict[str, Any]) -> tuple:
    return (row.get("id"), row.get("job_id"), row.get("status"), row.get("delivery_outcome"),
            row.get("scheduled_instant"), row.get("handoff_pending"),
            row.get("progress_at") is not None, row.get("source"))


def observe(backend, world: World):
    """Everything about the world's agent the port exposes: its jobs (enabled and not), their
    records, and its execution history. A refused operation must leave all of it unchanged."""
    with backend.bind(world.agent):
        store = world.store
        state = {"listed": sorted(job["id"] for job in store.list_jobs(include_disabled=True))}
        for name in ("due", "paused", "oneshot"):
            state[name] = _job_projection(store.get_job(getattr(world, name)))
        state["absent"] = _job_projection(store.get_job("absent"))
        state["executions"] = sorted(_execution_projection(row) for row in store.list_executions(limit=500))
        live = store.live_executions([world.due, world.paused, world.oneshot])
        state["live"] = sorted(live)
        state["live_detail"] = {job: row["status"] for job, row in live.items()}
    return state


#: Every mutating job operation, with arguments that succeed in a :class:`World`.
JOB_WRITES: dict[str, Callable[[CronStore, World], Any]] = {
    "create_job": lambda st, w: st.create_job("a created job", "*/5 * * * *"),
    "update_job": lambda st, w: st.update_job(w.due, {"prompt": "changed"}),
    "pause_job": lambda st, w: st.pause_job(w.due, reason="suite"),
    "resume_job": lambda st, w: st.resume_job(w.paused),
    "trigger_job": lambda st, w: st.trigger_job(w.paused),
    "rearm_oneshot": lambda st, w: st.rearm_oneshot(w.oneshot, _future(3 * 3600)),
    "remove_job": lambda st, w: st.remove_job(w.oneshot),
    "cancel_job_run": lambda st, w: st.cancel_job_run(w.due, reason="suite"),
    "get_due_jobs": lambda st, w: st.get_due_jobs(),
    "claim_dispatch": lambda st, w: st.claim_dispatch(w.oneshot),
    "advance_next_run": lambda st, w: st.advance_next_run(w.due),
    "mark_job_run": lambda st, w: st.mark_job_run(w.due, True),
    "claim_job_for_fire": lambda st, w: st.claim_job_for_fire(w.due),
}

#: Every mutating execution-audit operation, with arguments that succeed in a :class:`World`.
EXECUTION_WRITES: dict[str, Callable[[CronStore, World], Any]] = {
    "create_execution": lambda st, w: st.create_execution(w.due, source="suite"),
    "set_execution_occurrence": lambda st, w: st.set_execution_occurrence(w.execution, _past(60)),
    "mark_execution_handoff_pending": lambda st, w: st.mark_execution_handoff_pending(w.execution),
    "mark_execution_running": lambda st, w: st.mark_execution_running(w.execution),
    "touch_execution_progress": lambda st, w: st.touch_execution_progress(w.execution),
    "finish_execution": lambda st, w: st.finish_execution(w.execution, success=True),
    "record_missed_occurrences": lambda st, w: st.record_missed_occurrences(
        w.due, count=2, first_instant=_past(600), detail="suite"),
    "settle_delivery_outcome": lambda st, w: st.settle_delivery_outcome(w.finished, "delivered"),
    "settle_unstarted_execution": lambda st, w: st.settle_unstarted_execution(
        w.execution, w.due, "suite: never started"),
    "give_up_execution": lambda st, w: st.give_up_execution(w.execution, reason="suite"),
    "recover_interrupted_executions": lambda st, w: st.recover_interrupted_executions(),
    "terminalize_dead_owner": lambda st, w: st.terminalize_dead_owner(w.execution, reason="suite"),
}

ALL_WRITES = {**JOB_WRITES, **EXECUTION_WRITES}

#: Every read of the port, naming rows agent ``a`` holds in a :class:`World`.
ALL_READS: dict[str, Callable[[CronStore, World], Any]] = {
    "get_job": lambda st, w: st.get_job(w.due),
    "resolve_job_ref": lambda st, w: st.resolve_job_ref(w.due),
    "list_jobs": lambda st, w: st.list_jobs(),
    "get_due_jobs": lambda st, w: st.get_due_jobs(),
    "get_execution": lambda st, w: st.get_execution(w.execution),
    "list_executions": lambda st, w: st.list_executions(job_id=w.due),
    "latest_execution": lambda st, w: st.latest_execution(w.due),
    "latest_executions": lambda st, w: st.latest_executions([w.due]),
    "live_executions": lambda st, w: st.live_executions([w.due]),
    "live_inflight_execution": lambda st, w: st.live_inflight_execution(w.due),
    "cancellable_execution": lambda st, w: st.cancellable_execution(w.due),
    "execution_owner_live": lambda st, w: st.execution_owner_live(w.execution, w.due),
}

#: Operations that succeed in a :class:`World` without changing what :func:`observe` reads: a
#: cancel that only asks the live run to stop, and the sweeps that find nothing to act on (every
#: attempt the world made is this owner's and live).
_NO_VISIBLE_CHANGE = frozenset({"cancel_job_run", "recover_interrupted_executions",
                                "terminalize_dead_owner"})


class PinningConformance:
    """Every operation is pinned to its agent. Inherit and supply a ``backend`` fixture."""

    @pytest.mark.parametrize("operation", sorted(ALL_READS))
    def test_every_read_is_refused_for_any_pin_but_its_agents(self, backend, operation):
        world = make_world(backend)
        for bind in _foreign_pins(backend).values():
            with bind():
                _refused(lambda: ALL_READS[operation](world.store, world))
        with backend.bind("a"):
            ALL_READS[operation](world.store, world)

    @pytest.mark.parametrize("operation", sorted(ALL_WRITES))
    @pytest.mark.parametrize("pin", ["other_agent", "other_tenant", "unbound"])
    def test_every_write_is_refused_for_any_pin_but_its_agents_and_changes_nothing(
            self, backend, operation, pin):
        world = make_world(backend)
        before = observe(backend, world)
        with _foreign_pins(backend)[pin]():
            _refused(lambda: ALL_WRITES[operation](world.store, world))
        assert observe(backend, world) == before

    @pytest.mark.parametrize("operation", sorted(ALL_WRITES))
    def test_every_write_succeeds_for_its_own_agent(self, backend, operation):
        """The positive control for the refusal tests: each call is one the agent may make, and
        it changes what the agent observes (the sweeps that find nothing to act on, and a cancel
        that only asks a live run to stop, succeed without a visible change)."""
        world = make_world(backend)
        before = observe(backend, world)
        with backend.bind("a"):
            ALL_WRITES[operation](world.store, world)
        if operation not in _NO_VISIBLE_CHANGE:
            assert observe(backend, world) != before

    def test_with_no_scope_pinned_reads_see_nothing(self, backend):
        world = make_world(backend)
        with backend.unbound():
            _refused(lambda: world.store.get_job(world.due))
            _refused(lambda: world.store.list_jobs())
            _refused(lambda: world.store.get_due_jobs())
            _refused(lambda: world.store.get_execution(world.execution))
            _refused(lambda: world.store.list_executions())

    def test_another_agent_never_sees_the_agents_jobs(self, backend):
        world = make_world(backend)
        theirs = backend.open("b")
        with backend.bind("b"):
            assert theirs.get_job(world.due) is None
            assert theirs.list_jobs(include_disabled=True) == []
            assert theirs.list_executions() == []
            _refused(lambda: world.store.get_job(world.due))
            _refused(lambda: world.store.list_jobs(include_disabled=True))
            _refused(lambda: world.store.list_executions())

    def test_a_foreign_job_is_indistinguishable_from_a_missing_one(self, backend):
        world = make_world(backend)
        with backend.bind("b"):
            assert _refused(lambda: world.store.get_job(world.due)) \
                == _refused(lambda: world.store.get_job("absent"))
            assert _refused(lambda: world.store.update_job(world.due, {"prompt": "x"})) \
                == _refused(lambda: world.store.update_job("absent", {"prompt": "x"}))
            assert _refused(lambda: world.store.remove_job(world.due)) \
                == _refused(lambda: world.store.remove_job("absent"))
            assert _refused(lambda: world.store.get_execution(world.execution)) \
                == _refused(lambda: world.store.get_execution("absent"))

    def test_a_job_records_the_principal_who_created_it(self, backend):
        store = backend.open("a")
        with backend.bind("a", actor="jane"):
            jane = store.create_job("jane's job", "*/5 * * * *")
            assert jane["created_by"] == "jane"
        with backend.bind("a", actor="nadia"):
            nadia = store.create_job("nadia's job", "*/5 * * * *")
            assert nadia["created_by"] == "nadia"
        with backend.bind("a"):
            assert store.get_job(jane["id"])["created_by"] == "jane"
            assert store.get_job(nadia["id"])["created_by"] == "nadia"

    def test_no_update_can_repoint_the_creating_principal(self, backend):
        world = make_world(backend)
        with backend.bind("a"):
            creator = world.store.get_job(world.due)["created_by"]
            with pytest.raises(ValueError):
                world.store.update_job(world.due, {"created_by": "someone-else"})
            assert world.store.get_job(world.due)["created_by"] == creator


class TickConformance:
    """The tick boundary: one agent's tick and hands stay on its own jobs. Inherit and supply a
    ``backend`` fixture."""

    def test_a_tick_pinned_to_one_agent_never_fires_another_agents_due_job(self, backend):
        theirs = backend.open("b")
        with backend.bind("b"):
            due = theirs.create_job("their due job", "2h")["id"]
            theirs.update_job(due, {"next_run_at": _past(60)})
        mine = backend.open("a")
        with backend.bind("a"):
            assert all(job["id"] != due for job in mine.get_due_jobs())
            assert all(job["id"] != due for job in mine.list_jobs(include_disabled=True))
        with backend.bind("a"):
            # Their store under this agent's pin is refused whole: no listing, no scan, no claim.
            _refused(lambda: theirs.get_due_jobs())
            _refused(lambda: theirs.claim_job_for_fire(due))
            _refused(lambda: theirs.claim_dispatch(due))
            _refused(lambda: theirs.mark_job_run(due, True))
        with backend.bind("b"):  # and their due job is still theirs, due and unfired
            assert [job["id"] for job in theirs.get_due_jobs()] == [due]

    def test_one_agent_cannot_list_trigger_or_repoint_another_agents_jobs(self, backend):
        world = make_world(backend, "b")
        before = observe(backend, world)
        with backend.bind("a"):
            _refused(lambda: world.store.list_jobs())
            _refused(lambda: world.store.trigger_job(world.due))
            _refused(lambda: world.store.update_job(world.due, {"prompt": "repointed"}))
            _refused(lambda: world.store.remove_job(world.due))
            _refused(lambda: world.store.pause_job(world.due))
            _refused(lambda: world.store.resume_job(world.due))
        assert observe(backend, world) == before
        with backend.bind("b"):
            assert world.store.get_job(world.due)["prompt"] == "the due job of b"


class FireClaimConformance:
    """The at-most-once fire claim. Inherit and supply a ``backend`` fixture."""

    def test_a_due_job_is_claimed_once_and_every_later_claimant_loses(self, backend):
        world = make_world(backend)
        with backend.bind("a"):
            assert world.store.claim_job_for_fire(world.due) is True
            assert world.store.claim_job_for_fire(world.due) is False
            assert world.store.claim_job_for_fire(world.due, force=True) is False

    def test_two_racing_claimants_produce_exactly_one_winner(self, backend):
        world = make_world(backend)
        second = backend.replica("a")
        start, results = threading.Barrier(2), []

        def claim(store):
            with backend.bind("a"):
                start.wait(10)
                results.append(bool(store.claim_job_for_fire(world.due)))
        threads = [_replica_thread(claim, store) for store in (world.store, second)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert not any(thread.is_alive() for thread in threads)
        assert sorted(results) == [False, True]

    def test_a_claim_is_heartbeated_while_alive_and_released_by_the_runs_outcome(self, backend):
        world = make_world(backend)
        with backend.bind("a"):
            claimed = world.store.claim_job_for_fire(world.due, return_job=True)
            assert isinstance(claimed, dict)
            owner = claimed["fire_claim"]["by"]
            assert world.store.heartbeat_fire_claim(world.due, expected_owner=owner) is True
            assert world.store.heartbeat_fire_claim(world.due, expected_owner="not-the-owner") is False
            assert world.store.mark_job_run(world.due, True, expected_fire_owner=owner) is True
            # The run is over: the occurrence it took is released for the next one.
            assert world.store.claim_job_for_fire(world.due) is True

    def test_a_completion_of_a_claim_nobody_holds_anymore_is_discarded(self, backend):
        world = make_world(backend)
        with backend.bind("a"):
            claimed = world.store.claim_job_for_fire(world.due, return_job=True)
            owner = claimed["fire_claim"]["by"]
            assert world.store.mark_job_run(world.due, True, expected_fire_owner=owner) is True
            # The claim is gone; a late runner speaking for another owner writes nothing.
            assert world.store.mark_job_run(world.due, True, expected_fire_owner="another-owner") is False

    def test_a_fresh_claim_blocks_a_manual_fire(self, backend):
        world = make_world(backend)
        with backend.bind("a"):
            assert world.store.claim_job_for_fire(world.due) is True
            assert world.store.claim_job_for_fire(world.due, manual=True) is False

    def test_a_paused_job_is_refused_unless_the_fire_is_forced(self, backend):
        world = make_world(backend)
        with backend.bind("a"):
            assert world.store.claim_job_for_fire(world.paused) is False
            assert world.store.claim_job_for_fire(world.paused, force=True) is True
            assert world.store.get_job(world.paused)["enabled"] is True  # the forced fire resumed it

    def test_a_completed_occurrence_never_fires_twice(self, backend):
        world = make_world(backend)
        with backend.bind("a"):
            claimed = world.store.claim_job_for_fire(world.due, return_job=True)
            instant = claimed["_scheduled_instant"]
            owner = claimed["fire_claim"]["by"]
            execution = world.store.create_execution(world.due, source="suite",
                                                     scheduled_instant=instant)["id"]
            assert world.store.mark_execution_running(execution) is not None
            assert world.store.mark_job_run(world.due, True, expected_fire_owner=owner) is True
            assert world.store.finish_execution(execution, success=True) is not None
            # A stale re-delivery re-points the job at the occurrence it already ran.
            assert world.store.update_job(world.due, {"next_run_at": instant}) is not None
            assert world.store.claim_job_for_fire(world.due) is False
            assert all(job["id"] != world.due for job in world.store.get_due_jobs())

    def test_the_fence_holds_a_claim_across_a_side_effect(self, backend):
        world = make_world(backend)
        with backend.bind("a"):
            claimed = world.store.claim_job_for_fire(world.due, return_job=True)
            owner = claimed["fire_claim"]["by"]
            with world.store.fire_claim_fence(world.due, expected_owner=owner) as holds:
                assert holds is True
            with world.store.fire_claim_fence(world.due, expected_owner="not-the-owner") as holds:
                assert holds is False

    def test_the_scan_dispatches_a_due_one_shot_once(self, backend):
        store = backend.open("a")
        with backend.bind("a"):
            oneshot = store.create_job("the one-shot", _future(2 * 3600))["id"]
            store.update_job(oneshot, {"next_run_at": _past(60)})
            first = store.get_due_jobs()
            assert [job["id"] for job in first] == [oneshot]
            claim = first[0]["run_claim"]
            assert isinstance(claim, dict) and claim.get("by")
            # The scan claimed it for the duration of the run: no second dispatch.
            assert store.get_due_jobs() == []
            assert store.heartbeat_run_claim(oneshot, expected_owner=claim["by"]) is True
            assert store.heartbeat_run_claim(oneshot, expected_owner="not-the-owner") is False
            assert store.clear_run_claim(oneshot) is True
            assert [job["id"] for job in store.get_due_jobs()] == [oneshot]


class ExecutionConformance:
    """The execution audit. Inherit and supply a ``backend`` fixture."""

    def test_an_attempt_is_claimed_runs_and_finishes_once(self, backend):
        store = backend.open("a")
        with backend.bind("a"):
            job = store.create_job("audited job", "*/5 * * * *")["id"]
            execution = store.create_execution(job, source="suite")["id"]
            assert store.get_execution(execution)["status"] == "claimed"
            running = store.mark_execution_running(execution)
            assert running is not None and running["status"] == "running"
            assert store.mark_execution_running(execution) is None  # exactly once
            finished = store.finish_execution(execution, success=True)
            assert finished is not None and finished["status"] == "completed"
            # A terminal attempt is never rewritten.
            assert store.finish_execution(execution, success=False, error="late") is None
            assert store.get_execution(execution)["status"] == "completed"
            assert store.mark_execution_running(execution) is None

    def test_an_attempt_may_be_handed_to_a_worker_and_adopted_once(self, backend):
        store = backend.open("a")
        with backend.bind("a"):
            job = store.create_job("handed off job", "*/5 * * * *")["id"]
            execution = store.create_execution(job, source="suite")["id"]
            assert store.adopt_claimed_execution(execution) is None  # no handoff pending
            pending = store.mark_execution_handoff_pending(execution)
            assert pending is not None
            adopted = store.adopt_claimed_execution(execution)
            assert adopted is not None and adopted["status"] == "running"
            assert store.adopt_claimed_execution(execution) is None  # the gate admits one winner

    def test_a_terminal_attempt_only_settles_its_queued_delivery_once(self, backend):
        store = backend.open("a")
        with backend.bind("a"):
            job = store.create_job("delivered job", "*/5 * * * *")["id"]
            execution = store.create_execution(job, source="suite")["id"]
            store.finish_execution(execution, success=True, delivery_outcome="queued")
            assert store.settle_delivery_outcome(execution, "delivered") is True
            assert store.settle_delivery_outcome(execution, "replayed") is False
            assert store.get_execution(execution)["delivery_outcome"] == "delivered"

    def test_missed_occurrences_are_booked_as_one_row(self, backend):
        store = backend.open("a")
        with backend.bind("a"):
            job = store.create_job("missed job", "*/5 * * * *")["id"]
            row = store.record_missed_occurrences(
                job, count=3, first_instant=_past(1800), detail="the gateway was down")
            assert row is not None and row["status"] == "missed"
            assert store.record_missed_occurrences(job, count=0, first_instant=None,
                                                    detail="nothing") is None

    def test_this_owners_attempts_are_never_swept_as_abandoned(self, backend):
        world = make_world(backend)
        with backend.bind("a"):
            before = observe(backend, world)
            assert world.store.recover_interrupted_executions() == 0
            assert world.store.execution_owner_live(world.execution, world.due) is True
            assert world.store.terminalize_dead_owner(world.execution, reason="suite") is False
            assert observe(backend, world) == before

    def test_the_firer_may_give_its_attempt_up_exactly_once(self, backend):
        store = backend.open("a")
        with backend.bind("a"):
            job = store.create_job("given up job", "*/5 * * * *")["id"]
            execution = store.create_execution(job, source="suite")["id"]
            assert store.give_up_execution(execution, reason="suite: unknown") is True
            assert store.get_execution(execution)["status"] == "unknown"
            assert store.give_up_execution(execution, reason="suite: again") is False

    def test_an_unstarted_attempt_closes_its_receipt_without_raising(self, backend):
        store = backend.open("a")
        with backend.bind("a"):
            job = store.create_job("unstarted job", "*/5 * * * *")["id"]
            execution = store.create_execution(job, source="suite")["id"]
            store.settle_unstarted_execution(execution, job, "suite: never started")
            assert store.get_execution(execution)["status"] == "failed"

    def test_the_audits_reads_see_the_history_they_own(self, backend):
        world = make_world(backend)
        with backend.bind("a"):
            store = world.store
            assert store.get_execution(world.execution)["status"] == "claimed"
            assert [row["id"] for row in store.list_executions(job_id=world.due)] \
                == [row["id"] for row in store.list_executions(job_id=world.due)]
            assert store.latest_execution(world.due)["id"] in {world.execution, world.finished}
            assert set(store.latest_executions([world.due, world.paused])) == {world.due}
            assert store.live_inflight_execution(world.due)["id"] == world.execution
            assert store.cancellable_execution(world.due)["id"] == world.execution
            assert store.live_executions([world.due, world.paused]).keys() == {world.due}
            assert store.execution_owner_live(world.finished, world.due) is False


class CronConformance(PinningConformance, TickConformance, FireClaimConformance,
                      ExecutionConformance):
    """The whole suite. Inherit and supply a ``backend`` fixture (or inherit the parts
    separately, e.g. one test module each, to spread a slow backend over several runners)."""
