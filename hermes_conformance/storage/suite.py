"""The storage conformance suite: isolation and turn-admission invariants every backend meets.

Parametrized over backends through the ``backend`` fixture (:class:`Backend`). Each test talks to
the ports only (``hermes_state_port``); the fixture says how to bind each agent's scope and open
its storage. Agents are named by short keys (``"a"``, ``"b"``); both belong to one tenant.

Covered:

- pinning: every mutating operation of both ports (:data:`SESSION_WRITES`, :data:`ADMISSION_WRITES`)
  is refused for another agent, another tenant and no pin, and a refusal changes nothing the
  agent can observe through the ports (:func:`observe`: rows, transcripts, titles, admissions,
  generations, and auxiliary usage per task); every read (:data:`ALL_READS`) and the owner lock
  are refused for the same pins; a foreign row reads like a missing one;
- turn fences: every session write a turn makes (:data:`TURN_WRITES`, session creation and the
  upsert of an existing row included) is refused whole inside a claimed turn whose owner epoch or
  generation moved, with no accounting left behind, including a takeover that commits from
  another replica while the write waits for the writer (transaction-local, not a preflight);
- claims: a moved head claims nothing, a refusing gate rolls the claim back, a gate that closes
  while the claim waits for the writer still refuses it (asked inside the claim's transaction,
  not before it), one FIFO per session across producers admitting and claiming through different
  replicas, one writer per session;
- owners: one owner per agent across replicas, release and reacquisition, independent agents
  owned side by side, owner exclusion kept apart from per-session claim exclusion;
- the owner epoch fences every ledger operation (:data:`EPOCH_FENCED`); a lost turn becomes
  ``unknown`` and is never replayed; resolving it resumes the FIFO.

NOT covered. A backend that passes this suite may still violate these, and must not claim them:

- participant visibility (a participant-level record visible to that participant only);
- actor- and owner-level resolution (the level a write lands at for an actor or owner, and an
  owner-wide record visible to every agent of that owner and no other);
- child-session inheritance (a compression or delegation child keeping its parent's pin);
- retention and pruning (deleted and pruned rows leave nothing readable through the ports);
- the hosted isolation assertion (``SessionStorageProvider.assert_isolating`` failing for a role
  that bypasses tenant isolation); only that the method is required is tested, elsewhere;
- transaction-local scope pinning (a pin change while an operation waits for the writer);
- the shape of operation results beyond the fields the tests read.

The fixture cannot vary participant, actor or owner (the local backend stores the agent level
only), so the first two need the fixture extended before any backend can be tested for them.
"""

from __future__ import annotations

import contextvars
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pytest

from hermes_state_errors import SessionTurnLeaseLostError, TurnFenceRequiredError
from hermes_state_port import AdmissionGate, ClaimPreconditions, SessionStorage, StoragePinRefused


@dataclass
class Backend:
    """How the suite drives one backend.

    - ``open(agent)``: the agent's ports.
    - ``replica(agent)``: an independent second handle on the same agent's storage (another
      replica's connection).
    - ``bind(agent)``: bind the agent's scope (a non-turn activity of its owner).
    - ``foreign_tenant(agent)``: bind a scope of ANOTHER tenant that names the same agent id.
    - ``unbound()``: run with no scope pinned. A local backend's pin is then the ambient profile,
      so its ``unbound`` names a third profile; a hosted one binds nothing.
    - ``turn(storage, session_id, claimed)``: bind the claimed turn's fence for the body.
    - ``contend_takeover(agent)``: hold the agent's writer from another replica for the body.
      ``__enter__`` returns a ``threading.Event`` the backend sets once a write is waiting behind
      the hold; on exit another owner's takeover (a new owner epoch) commits before the hold is
      released. Required: without it the contending-takeover test fails.
    - ``hold_writer(agent)``: the same hold, released on exit having changed nothing. Required:
      without it the contended-gate test fails.
    """

    open: Callable[[str], SessionStorage]
    replica: Callable[[str], SessionStorage]
    bind: Callable[[str], AbstractContextManager]
    foreign_tenant: Callable[[str], AbstractContextManager]
    unbound: Callable[[], AbstractContextManager]
    turn: Callable[[SessionStorage, str, dict], AbstractContextManager]
    contend_takeover: Optional[Callable[[str], AbstractContextManager]] = None
    hold_writer: Optional[Callable[[str], AbstractContextManager]] = None


def _elsewhere(fn: Callable[[], Any]) -> Any:
    """Run *fn* in a fresh context: another writer, outside this test's turn."""
    return contextvars.Context().run(fn)


def _replica_thread(target: Callable[..., Any], *args: Any) -> threading.Thread:
    """An unstarted thread running *target* in a fresh, empty context: another replica's worker,
    which must not inherit this test's scope or turn."""
    # health: allow HX012 -- a replica's worker deliberately starts with no scope; each target binds its own
    return threading.Thread(target=contextvars.Context().run, args=(target, *args))


def _refused(call: Callable[[], Any]) -> str:
    """The refusal *call* raises (its message), asserting it is the store's pin refusal."""
    with pytest.raises(StoragePinRefused) as refused:
        call()
    return str(refused.value)


def _session(backend, agent, sid="s1", text=None):
    storage = backend.open(agent)
    with backend.bind(agent):
        storage.sessions.create(sid, "cli")
        if text is not None:
            storage.sessions.append(sid, [{"role": "user", "content": text}])
    return storage


def _owned_claim(backend, agent, sid="s1", requests=("r1",)):
    """A session of *agent* with *requests* admitted under a fresh owner epoch, the head claimed."""
    storage = _session(backend, agent, sid)
    with backend.bind(agent):
        epoch = _begin_epoch(storage.admission, instance_id="owner")
        for request in requests:
            storage.admission.admit(epoch=epoch, principal_id="human", session_id=sid, request_id=request,
                                    payload={"text": request})
        claimed = storage.admission.claim_next(epoch=epoch, session_id=sid) if requests else None
    return storage, epoch, claimed


def _texts(backend, agent, storage, sid="s1"):
    with backend.bind(agent):
        return [m["content"] for m in storage.sessions.read_conversation(sid)]


def _begin_epoch(admission, *, instance_id):
    """``begin_epoch``'s contract says the caller holds the owner lock, so the suite holds it: a
    backend that enforces the precondition passes the setup instead of failing it."""
    owner = admission.acquire_owner()
    try:
        return admission.begin_epoch(instance_id=instance_id)
    finally:
        owner.release()


def _take_over(backend, agent="a"):
    """Another owner takes *agent* over from another replica, outside the caller's context."""
    def run():
        with backend.bind(agent):
            return _begin_epoch(backend.replica(agent).admission, instance_id="successor")
    return _elsewhere(run)


# --- the world every per-operation test runs against ---------------------------------------------

@dataclass
class World:
    """Agent ``a``: session ``s1`` (titled, a two-message transcript, admissions ``r1`` claimed,
    ``r2`` and ``r4`` queued, ``r3`` steered into ``r1``'s turn) and session ``s2`` (empty, with a
    registered worker execution ``w-s2``) and ``idle`` (nothing running on it), all under one owner
    epoch."""

    storage: SessionStorage
    epoch: int
    rows: list
    claimed: dict
    user_row: int
    s2_generation: int


def make_world(backend) -> World:
    storage = _session(backend, "a")
    s, a = storage.sessions, storage.admission
    with backend.bind("a"):
        s.set_title("s1", "original")
        s.append("s1", [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}])
        s.create("s2", "cli")
        s.create("idle", "cli")
        epoch = _begin_epoch(a, instance_id="owner")
        rows = [a.admit(epoch=epoch, principal_id="human", session_id="s1", request_id=r, payload={"text": r})
                for r in ("r1", "r2", "r3", "r4")]
        claimed = a.claim_next(epoch=epoch, session_id="s1")
        a.steer(epoch=epoch, admission_id=rows[2]["admission_id"], generation=claimed["generation"])
        s2_generation = a.generation("s2")
        a.register_worker(epoch=epoch, execution_id="w-s2", session_id="s2", generation=s2_generation,
                          kind="compute", adoption_secret="secret")
        user_row = s.read_conversation("s1", include_row_ids=True)[0]["_row_id"]
    return World(storage, epoch, rows, claimed, user_row, s2_generation)


_OBSERVED_SESSIONS = ("s1", "s2", "s3", "c1", "idle")


def observe(backend, storage):
    """Everything about agent ``a`` the ports expose: its sessions, rows, titles, transcripts,
    usage (the session's counters and its auxiliary usage per task), admissions and generations.
    A refused operation must leave all of it unchanged."""
    s, a = storage.sessions, storage.admission
    with backend.bind("a"):
        assert s.flush_usage(), "queued usage did not finish before observation"
        state = {"listed": sorted(r["id"] for r in s.list_sessions(limit=100))}
        for sid in _OBSERVED_SESSIONS:
            row = s.get(sid)
            state[sid] = None if row is None else {
                "row": row, "title": s.get_title(sid), "generation": a.generation(sid),
                "messages": s.read_conversation(sid, include_row_ids=True),
                "auxiliary_usage": s.read_auxiliary_usage(sid),
                "admissions": a.list(sid, pending_only=False)}
    return state


#: Every mutating SessionStore operation, with arguments that succeed in a :class:`World`.
SESSION_WRITES: dict[str, Callable[[SessionStorage, World], Any]] = {
    "create": lambda st, w: st.sessions.create("s3", "cli"),
    "create_existing": lambda st, w: st.sessions.create("s1", "cli", model="m"),
    "end": lambda st, w: st.sessions.end("s1", "user_exit"),
    "append": lambda st, w: st.sessions.append("s1", [{"role": "user", "content": "x"}]),
    "append_turn": lambda st, w: st.sessions.append_turn("s1", [{"role": "assistant", "content": "x"}]),
    "set_title": lambda st, w: st.sessions.set_title("s1", "changed"),
    "delete": lambda st, w: st.sessions.delete("idle"),
    "replace_messages": lambda st, w: st.sessions.replace_messages("s1", [{"role": "user", "content": "x"}]),
    "rewind_to_message": lambda st, w: st.sessions.rewind_to_message("s1", w.user_row),
    "rewind_user_turn": lambda st, w: st.sessions.rewind_user_turn("s1", -1),
    "set_user_message_content": lambda st, w: st.sessions.set_user_message_content("s1", w.user_row, "edited"),
    "deactivate_message": lambda st, w: st.sessions.deactivate_message("s1", w.user_row),
    "set_message_reaction": lambda st, w: st.sessions.set_message_reaction("s1", w.user_row, "+1"),
    "publish_compression_child": lambda st, w: st.sessions.publish_compression_child(
        parent_session_id="s1", child_session_id="c1", source="cli",
        messages=[{"role": "user", "content": "summary"}], require_compression_lease=False),
    "archive_and_compact": lambda st, w: st.sessions.archive_and_compact(
        "s1", [{"role": "user", "content": "summary"}]),
    "record_usage": lambda st, w: st.sessions.record_usage("s1", queued=False, input_tokens=7),
    "record_usage_queued": lambda st, w: st.sessions.record_usage("s1", input_tokens=7),
    "record_auxiliary_usage": lambda st, w: st.sessions.record_auxiliary_usage(
        "s1", "title_generation", model="m", input_tokens=3),
}

#: Every mutating TurnAdmission operation but ``acquire_owner`` (tested on its own).
ADMISSION_WRITES: dict[str, Callable[[SessionStorage, World], Any]] = {
    "begin_epoch": lambda st, w: st.admission.begin_epoch(instance_id="another"),
    "recover": lambda st, w: st.admission.recover(epoch=w.epoch),
    "admit": lambda st, w: st.admission.admit(epoch=w.epoch, principal_id="human", session_id="s1",
                                              request_id="r5", payload={}),
    "claim_next": lambda st, w: st.admission.claim_next(epoch=w.epoch, session_id="s2"),
    "steer": lambda st, w: st.admission.steer(epoch=w.epoch, admission_id=w.rows[1]["admission_id"],
                                              generation=w.claimed["generation"]),
    "revert_steer": lambda st, w: st.admission.revert_steer(epoch=w.epoch, admission_id=w.rows[2]["admission_id"],
                                                            generation=w.claimed["generation"]),
    "settle": lambda st, w: st.admission.settle(epoch=w.epoch, admission_id=w.claimed["admission_id"],
                                                generation=w.claimed["generation"], outcome="completed"),
    "cancel": lambda st, w: st.admission.cancel(epoch=w.epoch, admission_id=w.rows[3]["admission_id"]),
    "resolve_unknown": lambda st, w: st.admission.resolve_unknown(
        epoch=w.epoch, admission_id=w.claimed["admission_id"], generation=w.claimed["generation"]),
    "mutate_session": lambda st, w: st.admission.mutate_session(
        epoch=w.epoch, principal_id="human", session_id="s2", request_id="m1",
        expected_revision=0, operation="rename", payload={"title": "renamed"}),
    "register_worker": lambda st, w: st.admission.register_worker(
        epoch=w.epoch, execution_id="w-new", session_id="s1", generation=w.claimed["generation"],
        kind="compute", adoption_secret="x"),
    "adopt_worker": lambda st, w: st.admission.adopt_worker(
        epoch=w.epoch, execution_id="w-s2", session_id="s2", generation=w.s2_generation, adoption_secret="secret"),
    "apply_worker_op": lambda st, w: st.admission.apply_worker_op(
        epoch=w.epoch, execution_id="w-s2", session_id="s2", generation=w.s2_generation, sequence=1,
        operation="transcript.append", payload={"messages": [{"role": "user", "content": "w"}]}),
}

ALL_WRITES = {**{f"sessions.{k}": v for k, v in SESSION_WRITES.items()},
              **{f"admission.{k}": v for k, v in ADMISSION_WRITES.items()}}

#: Every read of both ports, naming rows agent ``a`` holds in a :class:`World`.
ALL_READS: dict[str, Callable[[SessionStorage, World], Any]] = {
    "sessions.get": lambda st, w: st.sessions.get("s1"),
    "sessions.get_title": lambda st, w: st.sessions.get_title("s1"),
    "sessions.read_conversation": lambda st, w: st.sessions.read_conversation("s1"),
    "sessions.list_sessions": lambda st, w: st.sessions.list_sessions(),
    "sessions.read_auxiliary_usage": lambda st, w: st.sessions.read_auxiliary_usage("s1"),
    "admission.get": lambda st, w: st.admission.get(w.claimed["admission_id"]),
    "admission.find": lambda st, w: st.admission.find(principal_id="human", session_id="s1", request_id="r1"),
    "admission.list": lambda st, w: st.admission.list("s1"),
    "admission.generation": lambda st, w: st.admission.generation("s1"),
}

#: The session writes a claimed turn makes: all of them, creating a session (a child, a fork)
#: included. A turn that lost its lease lands none of them.
TURN_WRITES = dict(SESSION_WRITES)
#: Turn writes whose refusal is silent: queued accounting is dropped (and logged), never raised
#: into the turn; it still lands nothing.
SILENT_TURN_WRITES = frozenset({"record_usage_queued"})

#: The ledger operations an owner makes under its epoch; a superseded epoch refuses each.
#: ``begin_epoch`` starts an epoch, so it has none to be stale against.
EPOCH_FENCED = {k: v for k, v in ADMISSION_WRITES.items() if k != "begin_epoch"}

#: Operations whose positive control needs state the shared world does not have; each is
#: exercised end to end by its own test below.
_NO_POSITIVE_CONTROL = frozenset({"admission.resolve_unknown"})
#: Operations that succeed in a :class:`World` without changing what :func:`observe` reads (owner
#: epochs, worker rows, an empty claim).
_NO_VISIBLE_CHANGE = frozenset({"admission.begin_epoch", "admission.recover", "admission.adopt_worker",
                                "admission.claim_next", "admission.register_worker"})


class _Gate(AdmissionGate):
    def __init__(self, closed=False):
        self.closed, self.calls = closed, 0

    def require_open(self):
        self.calls += 1
        if self.closed:
            raise PermissionError("admission closed")


def _foreign_pins(backend):
    """Every pin but agent ``a``'s own: another agent, another tenant naming ``a``, no pin."""
    return {"other_agent": lambda: backend.bind("b"), "other_tenant": lambda: backend.foreign_tenant("a"),
            "unbound": backend.unbound}


class PinningConformance:
    """Every operation is pinned to its agent. Inherit and supply a ``backend`` fixture."""

    @pytest.mark.parametrize("operation", sorted(ALL_READS))
    def test_every_read_is_refused_for_any_pin_but_its_agents(self, backend, operation):
        world = make_world(backend)
        for bind in _foreign_pins(backend).values():
            with bind():
                _refused(lambda: ALL_READS[operation](world.storage, world))
        with backend.bind("a"):
            ALL_READS[operation](world.storage, world)

    def test_the_owner_lock_is_refused_for_any_pin_but_its_agents(self, backend):
        storage = _session(backend, "a")
        for bind in _foreign_pins(backend).values():
            with bind():
                _refused(lambda: storage.admission.acquire_owner())
        with backend.bind("a"):  # and no refusal left the lock held
            storage.admission.acquire_owner().release()

    @pytest.mark.parametrize("operation", sorted(ALL_WRITES))
    @pytest.mark.parametrize("pin", ["other_agent", "other_tenant", "unbound"])
    def test_every_write_is_refused_for_any_pin_but_its_agents_and_changes_nothing(self, backend, operation, pin):
        world = make_world(backend)
        before = observe(backend, world.storage)
        binds = {"other_agent": lambda: backend.bind("b"), "other_tenant": lambda: backend.foreign_tenant("a"),
                 "unbound": backend.unbound}
        with binds[pin]():
            if operation == "sessions.append_turn":
                with backend.turn(world.storage, "s1", world.claimed):
                    _refused(lambda: ALL_WRITES[operation](world.storage, world))
            else:
                _refused(lambda: ALL_WRITES[operation](world.storage, world))
        assert observe(backend, world.storage) == before
        with backend.bind("a"):  # nor did it supersede the owner: an empty claim under its epoch still runs
            assert not world.storage.admission.claim_next(epoch=world.epoch, session_id="idle")

    @pytest.mark.parametrize("operation", sorted(set(ALL_WRITES) - _NO_POSITIVE_CONTROL))
    def test_every_write_succeeds_for_its_own_agent(self, backend, operation):
        """The positive control for the refusal tests: each call is one the agent may make, and it
        changes what the agent observes."""
        world = make_world(backend)
        before = observe(backend, world.storage)
        with backend.bind("a"):
            if operation == "sessions.append_turn":
                with backend.turn(world.storage, "s1", world.claimed):
                    ALL_WRITES[operation](world.storage, world)
            else:
                ALL_WRITES[operation](world.storage, world)
        if operation not in _NO_VISIBLE_CHANGE:
            assert observe(backend, world.storage) != before

    def test_with_no_scope_pinned_reads_see_nothing(self, backend):
        storage = _session(backend, "a", text="hello")
        with backend.unbound():
            _refused(lambda: storage.sessions.get("s1"))
            _refused(lambda: storage.sessions.read_conversation("s1"))
            _refused(lambda: storage.sessions.read_auxiliary_usage("s1"))
            _refused(lambda: storage.sessions.list_sessions())
            _refused(lambda: storage.admission.list("s1"))
            _refused(lambda: storage.admission.acquire_owner())

    def test_another_agent_never_sees_the_agents_rows(self, backend):
        mine = _session(backend, "a", text="private")
        theirs = backend.open("b")
        with backend.bind("b"):
            assert theirs.sessions.get("s1") is None
            assert theirs.sessions.list_sessions() == []
            _refused(lambda: mine.sessions.read_conversation("s1"))
            _refused(lambda: mine.admission.list("s1"))
            _refused(lambda: mine.admission.acquire_owner())
        assert _texts(backend, "a", mine) == ["private"]

    def test_a_foreign_row_is_indistinguishable_from_a_missing_one(self, backend):
        mine = _session(backend, "a")
        with backend.bind("b"):
            assert _refused(lambda: mine.sessions.get("s1")) == _refused(lambda: mine.sessions.get("absent"))
            assert (_refused(lambda: mine.sessions.get_title("s1"))
                    == _refused(lambda: mine.sessions.get_title("absent")))
            assert (_refused(lambda: mine.sessions.set_title("s1", "x"))
                    == _refused(lambda: mine.sessions.set_title("absent", "x")))



class FenceConformance:
    """A claimed turn's writes are fenced by its claim. Inherit and supply a ``backend`` fixture."""

    def test_a_turn_path_write_with_no_fence_is_refused(self, backend):
        storage, _epoch, _claimed = _owned_claim(backend, "a")
        with backend.bind("a"), pytest.raises(TurnFenceRequiredError):
            storage.sessions.append_turn("s1", [{"role": "assistant", "content": "unfenced"}])
        assert _texts(backend, "a", storage) == []

    def test_a_turn_holding_its_lease_writes_and_settles(self, backend):
        storage, epoch, claimed = _owned_claim(backend, "a")
        with backend.bind("a"):
            with backend.turn(storage, "s1", claimed):
                assert storage.sessions.append_turn("s1", [{"role": "assistant", "content": "ok"}]) == 1
                storage.sessions.append("s1", [{"role": "assistant", "content": "ordinary"}])
            settled = storage.admission.settle(epoch=epoch, admission_id=claimed["admission_id"],
                                               generation=claimed["generation"], outcome="completed")
        assert settled["status"] == "terminal"
        assert _texts(backend, "a", storage) == ["ok", "ordinary"]

    @pytest.mark.parametrize("operation", sorted(TURN_WRITES))
    @pytest.mark.parametrize("supersede", ["owner_epoch", "generation"])
    def test_every_turn_write_under_a_superseded_lease_is_refused_whole(self, backend, operation, supersede):
        world = make_world(backend)

        def next_claim():  # the owner settles this turn and claims the session's next input
            with backend.bind("a"):
                other = backend.replica("a")
                other.admission.settle(epoch=world.epoch, admission_id=world.claimed["admission_id"],
                                       generation=world.claimed["generation"], outcome="interrupted")
                successor = other.admission.claim_next(epoch=world.epoch, session_id="s1")
                assert successor["generation"] > world.claimed["generation"]
        if supersede == "owner_epoch":
            _take_over(backend)
        else:
            _elsewhere(next_claim)
        before = observe(backend, world.storage)
        with backend.bind("a"), backend.turn(world.storage, "s1", world.claimed):
            if operation in SILENT_TURN_WRITES:
                TURN_WRITES[operation](world.storage, world)
            else:
                with pytest.raises(SessionTurnLeaseLostError):
                    TURN_WRITES[operation](world.storage, world)
        assert observe(backend, world.storage) == before

    @pytest.mark.parametrize("operation", ["append", "append_turn", "set_title", "record_usage"])
    def test_a_takeover_that_commits_while_the_write_waits_still_refuses_it(self, backend, operation):
        """The fence is checked inside the write's own transaction: a takeover that commits after the
        write was issued, while it waits for the writer, refuses it. A preflight check would pass."""
        assert backend.contend_takeover is not None, "the backend fixture must supply contend_takeover"
        world = make_world(backend)
        before = observe(backend, world.storage)
        outcome = {}

        def write():
            try:
                with backend.bind("a"), backend.turn(world.storage, "s1", world.claimed):
                    TURN_WRITES[operation](world.storage, world)
                outcome["written"] = True
            except SessionTurnLeaseLostError as exc:
                outcome["refused"] = exc
        worker = _replica_thread(write)
        with backend.contend_takeover("a") as waiting:
            worker.start()
            assert waiting.wait(10), "the write never waited behind the contending replica"
        worker.join(30)
        assert not worker.is_alive()
        assert "refused" in outcome and "written" not in outcome
        after = observe(backend, world.storage)
        for sid in _OBSERVED_SESSIONS:
            assert (after[sid] or {}).get("messages") == (before[sid] or {}).get("messages")
            assert (after[sid] or {}).get("title") == (before[sid] or {}).get("title")
        assert after["s1"]["row"].get("input_tokens") == before["s1"]["row"].get("input_tokens")



class AdmissionConformance:
    """Claims, FIFOs, owners and recovery. Inherit and supply a ``backend`` fixture."""

    def test_a_claim_whose_validated_head_moved_claims_nothing(self, backend):
        storage, epoch, _none = _owned_claim(backend, "a", requests=())
        with backend.bind("a"):
            rows = [storage.admission.admit(epoch=epoch, principal_id="human", session_id="s1", request_id=r,
                                            payload={}) for r in ("r1", "r2")]
        before = observe(backend, storage)
        with backend.bind("a"):
            for wrong in (rows[1]["admission_id"], None, "not-an-admission"):
                assert storage.admission.claim_next(epoch=epoch, session_id="s1",
                                                    preconditions=ClaimPreconditions(head=wrong)) is False
        assert observe(backend, storage) == before
        with backend.bind("a"):
            claimed = storage.admission.claim_next(epoch=epoch, session_id="s1",
                                                   preconditions=ClaimPreconditions(head=rows[0]["admission_id"]))
        assert claimed["admission_id"] == rows[0]["admission_id"] and claimed["status"] == "started"

    def test_a_refusing_gate_rolls_the_claim_back(self, backend):
        storage, epoch, _none = _owned_claim(backend, "a", requests=())
        with backend.bind("a"):
            row = storage.admission.admit(epoch=epoch, principal_id="human", session_id="s1", request_id="r1",
                                          payload={})
        before = observe(backend, storage)
        closed = _Gate(closed=True)
        with backend.bind("a"), pytest.raises(PermissionError):
            storage.admission.claim_next(epoch=epoch, session_id="s1",
                                         preconditions=ClaimPreconditions(head=row["admission_id"], gate=closed))
        assert closed.calls >= 1
        assert observe(backend, storage) == before
        opened = _Gate()
        with backend.bind("a"):
            claimed = storage.admission.claim_next(epoch=epoch, session_id="s1",
                                                   preconditions=ClaimPreconditions(gate=opened))
        assert opened.calls >= 1 and claimed["admission_id"] == row["admission_id"]

    def test_a_gate_that_closes_while_the_claim_waits_still_refuses_it(self, backend):
        """The gate is asked inside the claim's own transaction: a drain that begins after the
        claim was issued, while it waits for the writer, refuses it. A preflight gate would pass."""
        assert backend.hold_writer is not None, "the backend fixture must supply hold_writer"
        storage, epoch, _none = _owned_claim(backend, "a", requests=())
        with backend.bind("a"):
            row = storage.admission.admit(epoch=epoch, principal_id="human", session_id="s1", request_id="r1",
                                          payload={})
        before = observe(backend, storage)
        gate, outcome = _Gate(), {}

        def claim():
            try:
                with backend.bind("a"):
                    outcome["claimed"] = storage.admission.claim_next(
                        epoch=epoch, session_id="s1",
                        preconditions=ClaimPreconditions(head=row["admission_id"], gate=gate))
            except PermissionError as exc:
                outcome["refused"] = exc
        worker = _replica_thread(claim)
        with backend.hold_writer("a") as waiting:
            worker.start()
            assert waiting.wait(10), "the claim never waited behind the contending replica"
            gate.closed = True  # the owner starts draining while the claim waits
        worker.join(30)
        assert not worker.is_alive()
        assert "claimed" not in outcome and "refused" in outcome
        assert not isinstance(outcome["refused"], StoragePinRefused)
        assert observe(backend, storage) == before
        with backend.bind("a"):
            claimed = storage.admission.claim_next(epoch=epoch, session_id="s1",
                                                   preconditions=ClaimPreconditions(gate=_Gate()))
        assert claimed["admission_id"] == row["admission_id"]

    def test_one_fifo_per_session(self, backend):
        storage, epoch, first = _owned_claim(backend, "a", requests=("r1", "r2", "r3"))
        order = [first["request_id"]]
        with backend.bind("a"):
            assert storage.admission.claim_next(epoch=epoch, session_id="s1") is None  # one at a time
            claimed = first
            while claimed:
                storage.admission.settle(epoch=epoch, admission_id=claimed["admission_id"],
                                         generation=claimed["generation"], outcome="completed")
                claimed = storage.admission.claim_next(epoch=epoch, session_id="s1")
                if claimed:
                    order.append(claimed["request_id"])
        assert order == ["r1", "r2", "r3"]

    def test_one_fifo_across_producers_and_replicas(self, backend):
        """Producers admit through different replicas, interleaved; the session has one order, the
        order admission accepted them in, whichever producer or replica claims."""
        replicas = [_session(backend, "a"), backend.replica("a"), backend.replica("a")]
        producers = ["z-person", "a-cron", "m-api"]
        accepted, claimed_order = [], []
        with backend.bind("a"):
            epoch = _begin_epoch(replicas[0].admission, instance_id="owner")
            for n in range(9):
                row = replicas[n % 3].admission.admit(epoch=epoch, principal_id=producers[(n * 2) % 3],
                                                      session_id="s1", request_id=f"req-{n}", payload={"n": n})
                accepted.append(row["admission_id"])
            listed = [r["admission_id"] for r in replicas[2].admission.list("s1")]
            for n in range(9):
                claimed = replicas[(n + 1) % 3].admission.claim_next(epoch=epoch, session_id="s1")
                claimed_order.append(claimed["admission_id"])
                replicas[n % 3].admission.settle(epoch=epoch, admission_id=claimed["admission_id"],
                                                 generation=claimed["generation"], outcome="completed")
        assert listed == accepted == claimed_order

    def test_one_writer_per_session_across_two_replicas(self, backend):
        first = _session(backend, "a")
        second = backend.replica("a")
        with backend.bind("a"):
            epoch = _begin_epoch(first.admission, instance_id="owner")
            for request in ("r1", "r2"):  # work for both: only the one-writer rule stops the second
                first.admission.admit(epoch=epoch, principal_id="human", session_id="s1", request_id=request,
                                      payload={"text": request})
        start, results = threading.Barrier(2), []

        def claim(storage):
            with backend.bind("a"):
                start.wait(10)
                results.append(storage.admission.claim_next(epoch=epoch, session_id="s1"))
        threads = [_replica_thread(claim, s) for s in (first, second)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert sorted(bool(r) for r in results) == [False, True]

    def test_two_sessions_of_one_agent_run_side_by_side(self, backend):
        first = _session(backend, "a", "s1")
        _session(backend, "a", "s2")
        storages = {"s1": first, "s2": backend.replica("a")}  # each session on its own replica
        claims = {}
        with backend.bind("a"):
            epoch = _begin_epoch(first.admission, instance_id="owner")
            for sid in storages:
                first.admission.admit(epoch=epoch, principal_id="human", session_id=sid, request_id="r",
                                      payload={"text": sid})
                claims[sid] = first.admission.claim_next(epoch=epoch, session_id=sid)
        start, errors = threading.Barrier(2), []

        def run(sid):
            storage, claimed = storages[sid], claims[sid]
            try:
                with backend.bind("a"), backend.turn(storage, sid, claimed):
                    start.wait(10)
                    for n in range(20):
                        storage.sessions.append_turn(sid, [{"role": "assistant", "content": f"{sid}-{n}"}])
            except Exception as exc:  # health: allow BLE001 -- the thread's failure is asserted below
                errors.append(exc)
        threads = [_replica_thread(run, sid) for sid in storages]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        assert errors == []
        for sid in storages:
            assert _texts(backend, "a", first, sid) == [f"{sid}-{n}" for n in range(20)]

    # --- owners

    def test_one_owner_per_agent_across_replicas_released_and_reacquired(self, backend):
        first, second = _session(backend, "a"), backend.replica("a")
        with backend.bind("a"):
            held = first.admission.acquire_owner()
            try:
                with pytest.raises(Exception):
                    second.admission.acquire_owner()
            finally:
                held.release()
            again = second.admission.acquire_owner()  # released: another replica may own it now
            try:
                with pytest.raises(Exception):
                    first.admission.acquire_owner()
            finally:
                again.release()
            again.release()  # idempotent
            third = first.admission.acquire_owner()
            third.release()

    def test_independent_agents_are_owned_side_by_side(self, backend):
        mine, theirs = _session(backend, "a"), _session(backend, "b")
        with backend.bind("a"):
            a_owner = mine.admission.acquire_owner()
        try:
            with backend.bind("b"):
                b_owner = theirs.admission.acquire_owner()
                b_owner.release()
        finally:
            with backend.bind("a"):
                a_owner.release()

    def test_owner_exclusion_is_not_session_claim_exclusion(self, backend):
        """The owner lock does not claim a session and a started claim does not hold the owner
        lock: two sessions claim under one owner, and the lock moves while a claim is started."""
        first = _session(backend, "a", "s1")
        _session(backend, "a", "s2")
        second = backend.replica("a")
        with backend.bind("a"):
            owner = first.admission.acquire_owner()
            try:
                epoch = first.admission.begin_epoch(instance_id="owner")
                for sid in ("s1", "s2"):
                    first.admission.admit(epoch=epoch, principal_id="human", session_id=sid, request_id="r",
                                          payload={})
                assert first.admission.claim_next(epoch=epoch, session_id="s1")
                assert second.admission.claim_next(epoch=epoch, session_id="s2")
            finally:
                owner.release()
            successor = second.admission.acquire_owner()
            try:
                assert [r["status"] for r in second.admission.list("s1", pending_only=False)] == ["started"]
            finally:
                successor.release()

    # --- the owner epoch and crash recovery

    @pytest.mark.parametrize("operation", sorted(EPOCH_FENCED))
    def test_a_superseded_owner_epoch_refuses_every_ledger_operation(self, backend, operation):
        world = make_world(backend)
        if operation == "resolve_unknown":  # a wrong-status refusal must not be what catches it
            with backend.bind("a"):
                successor = backend.replica("a")
                epoch = _begin_epoch(successor.admission, instance_id="recovering")
                assert successor.admission.recover(epoch=epoch) == 1  # the claimed row is unknown now
        _take_over(backend)
        before = observe(backend, world.storage)
        with backend.bind("a"):
            with pytest.raises(Exception) as refused:
                EPOCH_FENCED[operation](world.storage, world)
        assert not isinstance(refused.value, StoragePinRefused)
        assert observe(backend, world.storage) == before

    def test_a_turn_whose_owner_died_becomes_unknown_and_is_never_replayed(self, backend):
        _storage, _epoch, claimed = _owned_claim(backend, "a", requests=("r1", "r2"))
        successor = backend.replica("a")
        with backend.bind("a"):
            epoch = _begin_epoch(successor.admission, instance_id="successor")
            assert successor.admission.recover(epoch=epoch) == 1
            assert successor.admission.get(claimed["admission_id"])["status"] == "unknown"
            try:
                replayed = successor.admission.claim_next(epoch=epoch, session_id="s1")
            except Exception:  # health: allow BLE001 -- a backend may refuse outright; it must not replay
                replayed = None
            assert not replayed
            assert successor.admission.get(claimed["admission_id"])["status"] == "unknown"

    def test_recovery_resumes_the_fifo_after_the_lost_turn_is_resolved(self, backend):
        _storage, _epoch, claimed = _owned_claim(backend, "a", requests=("r1", "r2"))
        successor = backend.replica("a")
        with backend.bind("a"):
            epoch = _begin_epoch(successor.admission, instance_id="successor")
            successor.admission.recover(epoch=epoch)
            resolved = successor.admission.resolve_unknown(epoch=epoch, admission_id=claimed["admission_id"],
                                                           generation=claimed["generation"])
            assert resolved["status"] == "terminal"
            follower = successor.admission.claim_next(epoch=epoch, session_id="s1")
        assert follower and follower["request_id"] == "r2" and follower["owner_epoch"] == epoch

    def test_a_stale_owner_cannot_settle(self, backend):
        storage, epoch, claimed = _owned_claim(backend, "a")
        _take_over(backend)
        with backend.bind("a"):
            with pytest.raises(Exception):  # the backend's stale-owner refusal
                storage.admission.settle(epoch=epoch, admission_id=claimed["admission_id"],
                                         generation=claimed["generation"], outcome="completed")
            assert storage.admission.get(claimed["admission_id"])["status"] == "started"


class StorageConformance(PinningConformance, FenceConformance, AdmissionConformance):
    """The whole suite. Inherit and supply a ``backend`` fixture (or inherit the parts separately,
    e.g. one test module each, to spread a slow backend over several runners)."""
