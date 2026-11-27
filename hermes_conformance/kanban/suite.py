"""The kanban conformance suite: isolation and claim invariants every backend meets.

Parametrized over backends through the ``backend`` fixture (:class:`Backend`). Each test talks
to the port only (``hermes_kanban_port``); the fixture says how to bind each agent's scope and
open its store. Agents are named by short keys (``"a"``, ``"b"``); both belong to one tenant.

Covered:

- pinning: every operation of the store (:data:`READS`, :data:`WRITES`) is refused for a tenant
  the store does not serve, and a refusal changes nothing the agent can observe through the port
  (:func:`observe`: boards, task rows, event counts, comment counts and notify subscriptions);
  a backend declares whether it serves the ambient caller with no scope bound
  (:attr:`Backend.serves_unbound`) and behaves as declared either way;
- isolation: another agent's task is never dispatched, swept, reclaimed or re-assigned from
  here, and is indistinguishable from a missing one; a sweep pinned to one agent never releases
  another agent's expired claim; a worker of one agent cannot leave for a board that exists only
  in another agent's board set (it looks like a board that does not exist); another agent's
  notify targets are invisible — its subscriptions, cursors and events;
- the same-tenant cross-agent matrix: every operation of A's handle run under a bound scope of
  another agent of the same tenant (B's worker) is refused — a read may instead answer exactly
  what a missing row would — and changes nothing A can observe, unless the backend declares the
  one local exception on its own class (:attr:`KanbanStore.LOCAL_SHARED_BOARD`
  ``hermes_kanban_port``: a single-user machine's one board root, shared by every profile,
  which today's dispatcher handoff needs), when the matrix asserts the inverse: B's scope is
  served on A's handle, answering exactly A's own seeded answer — never an empty stand-in —
  as A's own is. A write row's own preparation (the card a link or unlink targets, the block
  an unblock resumes, the board a removal removes) runs under A's scope first, so the refusal
  the matrix pins is the named operation's, never a prerequisite's that refused before it;
- claims: a ready task is claimed by at most one of two replicas, and the loser sees ``None``
  with no trace of a claim; a review claim behaves the same.

NOT covered. A backend that passes this suite may still violate these, and must not claim them:

- today's full per-operation semantics (block routing, review provenance, acceptance contracts,
  event kinds, idempotency keys, attachments, workspaces): the suite pins and isolates the
  operations, it does not re-test the kanban kernel;
- concurrent notify watchers of one subscription across replicas (one cursor claim per event
  range is asserted for a single watcher only);
- retention sweeps (``gc_events``, worker-log pruning) and the hosted isolation assertion
  (``KanbanStorageProvider.assert_isolating``; only that the method is required is tested,
  elsewhere);
- the shape of operation results beyond the fields the tests read.
"""

from __future__ import annotations

import contextvars
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pytest

from hermes_kanban_port import KanbanPinRefused, KanbanStore

#: The board every world agent uses for its ordinary cards.
WORLD_BOARD = "work"
#: The board only the second agent owns; a worker of the first must never reach it.
FOREIGN_BOARD = "private"
#: The chat id each agent's notify subscription uses.
CHAT = "telegram-chat"
#: The board a removal row prepares under the owning agent's own scope before it removes it.
TEMP_BOARD = "temp-board"


@dataclass
class Backend:
    """How the suite drives one backend.

    - ``open(agent)``: the agent's store. Also the only place the fixture may point the process
      environment at that agent's kanban root: a store serves the root it was opened for, never
      the ambient one, and the suite relies on the environment being left at whichever agent
      opened last to catch a backend that re-resolves per call.
    - ``replica(agent)``: an independent second handle on the same agent's storage (another
      replica's connection).
    - ``bind(agent)``: bind the agent's scope (a worker of that agent, of the suite's one
      tenant — both agents belong to it, so binding the second agent is the same-tenant
      cross-agent scope the suite's cross-agent matrix runs under). Must not touch the process
      environment.
    - ``foreign_tenant(agent)``: bind a scope of ANOTHER tenant that names the same agent id.
    - ``unbound()``: run with no scope pinned.
    - ``serves_unbound``: whether the backend serves the ambient caller with no scope bound
      (today's local CLI does; a backend of a shared hosted process never may). The suite tests
      the declared behaviour either way.
    """

    open: Callable[[str], KanbanStore]
    replica: Callable[[str], KanbanStore]
    bind: Callable[[str], AbstractContextManager]
    foreign_tenant: Callable[[str], AbstractContextManager]
    unbound: Callable[[], AbstractContextManager]
    serves_unbound: bool = False


@dataclass
class AgentWorld:
    """One agent's slice of the world, on that agent's own store."""
    agent: str
    store: KanbanStore
    ready_id: str
    claimed_id: str
    review_id: str
    #: the card on the second agent's private board (``None`` for the first agent)
    private_id: Optional[str] = None
    #: the card a later link or unlink targets, created by that row's preparation under the
    #: owning agent's own scope (a named operation is the only thing that runs under a
    #: foreign scope)
    prepared_card_id: Optional[str] = None


@dataclass
class World:
    """The world: both agents, each with its own store. Opening order is first ``a`` then
    ``b`` (the fixture leaves the environment, if any, at ``b``)."""
    agents: dict[str, AgentWorld] = field(default_factory=dict)

    def __getitem__(self, agent: str) -> AgentWorld:
        return self.agents[agent]


def make_world(backend: Backend) -> World:
    """Both agents' boards: a ready card, a running (claimed) card and a review card on the
    shared board slug, a comment and a notify subscription on the ready card; the second agent
    also owns a board the first must never see."""
    world = World()
    for agent in ("a", "b"):
        store = backend.open(agent)
        with backend.bind(agent):
            store.create_board(WORLD_BOARD)
            ready_id = store.create_task(title=f"{agent} ready card", board=WORLD_BOARD)
            claimed_id = store.create_task(title=f"{agent} running card", board=WORLD_BOARD)
            store.claim_task(claimed_id, board=WORLD_BOARD, claimer=f"{agent}-worker")
            review_id = store.create_task(title=f"{agent} review card", board=WORLD_BOARD)
            store.request_review(review_id, reviewer=f"{agent}-reviewer", board=WORLD_BOARD)
            store.add_comment(ready_id, "user", f"note for {agent}", board=WORLD_BOARD)
            store.add_notify_sub(task_id=ready_id, platform="telegram", chat_id=f"{CHAT}-{agent}",
                                 board=WORLD_BOARD)
        world.agents[agent] = AgentWorld(agent, store, ready_id, claimed_id, review_id)
    with backend.bind("b"):
        b_world = world.agents["b"]
        b_world.store.create_board(FOREIGN_BOARD)
        b_world.private_id = b_world.store.create_task(title="b private card", board=FOREIGN_BOARD)
    return world


def observe(store: KanbanStore, w: AgentWorld) -> dict[str, Any]:
    """Everything the agent can observe through the port about its own slice."""
    return {
        "boards": sorted(m["slug"] for m in store.list_boards()),
        "tasks": sorted((t.id, t.status) for t in
                        store.list_tasks(board=WORLD_BOARD, include_archived=True)),
        "events": len(store.list_events(w.ready_id, board=WORLD_BOARD)),
        "comments": len(store.list_comments(w.ready_id, board=WORLD_BOARD)),
        "subs": sorted((s["task_id"], s["platform"], s["chat_id"]) for s in
                       store.list_notify_subs(board=WORLD_BOARD)),
    }


def _wait_until(predicate: Callable[[], Any], timeout: float = 15.0) -> bool:
    """Poll until *predicate* holds (a claim's TTL running out is a wall-clock wait)."""
    deadline = time.monotonic() + timeout
    while True:
        result = predicate()
        if result or time.monotonic() > deadline:
            return bool(result)
        time.sleep(0.05)


def _refused(call: Callable[[], Any]) -> str:
    with pytest.raises(KanbanPinRefused) as refused:
        call()
    return str(refused.value)


def _replica_thread(target: Callable[..., Any], *args: Any) -> threading.Thread:
    """An unstarted thread running *target* in a fresh, empty context: another replica's
    worker, which must not inherit this test's scope."""
    # health: allow HX012 -- a replica's worker deliberately starts with no scope; each target binds its own
    return threading.Thread(target=contextvars.Context().run, args=(target, *args))


def _prepare_card(title: str) -> Callable[[KanbanStore, AgentWorld], None]:
    """A row's preparation: the card a later link or unlink targets, created under the owning
    agent's own scope so the named operation is the only thing that runs under a foreign one."""
    def prepare(store: KanbanStore, w: AgentWorld) -> None:
        w.prepared_card_id = store.create_task(title=title, board=WORLD_BOARD)
    return prepare


def _prepared(w: AgentWorld) -> str:
    """The target card a prepared row operates on (created by its preparation)."""
    assert w.prepared_card_id is not None, "the row's preparation did not run"
    return w.prepared_card_id


READS: list[tuple[str, Callable[[KanbanStore, AgentWorld], Any]]] = [
    ("get_task", lambda s, w: s.get_task(w.ready_id, board=WORLD_BOARD)),
    ("list_tasks", lambda s, w: s.list_tasks(board=WORLD_BOARD)),
    ("list_events", lambda s, w: s.list_events(w.ready_id, board=WORLD_BOARD)),
    ("list_comments", lambda s, w: s.list_comments(w.ready_id, board=WORLD_BOARD)),
    ("list_boards", lambda s, w: s.list_boards()),
    ("list_notify_subs", lambda s, w: s.list_notify_subs(board=WORLD_BOARD)),
    ("latest_run", lambda s, w: s.latest_run(w.claimed_id, board=WORLD_BOARD)),
    ("list_runs", lambda s, w: s.list_runs(w.claimed_id, board=WORLD_BOARD)),
    ("unseen_events_for_sub",
     lambda s, w: s.unseen_events_for_sub(task_id=w.ready_id, platform="telegram",
                                           chat_id=f"{CHAT}-{w.agent}", board=WORLD_BOARD)),
]

WRITES: list[tuple[str, Callable[[KanbanStore, AgentWorld], Any], Callable[[Any], bool]]] = [
    ("create_task", lambda s, w: s.create_task(title="fresh card", board=WORLD_BOARD),
     lambda r: isinstance(r, str) and r != ""),
    ("edit_task", lambda s, w: s.edit_task(w.ready_id, title="renamed", board=WORLD_BOARD),
     lambda r: r is True),
    ("assign_task", lambda s, w: s.assign_task(w.ready_id, "profile-2", board=WORLD_BOARD),
     lambda r: r is True),
    ("add_comment", lambda s, w: s.add_comment(w.ready_id, "user", "another note", board=WORLD_BOARD),
     lambda r: isinstance(r, int) and r > 0),
    ("link_tasks",
     lambda s, w: s.link_tasks(w.ready_id, _prepared(w), board=WORLD_BOARD),
     lambda r: r is True),
    ("unlink_tasks",
     lambda s, w: s.unlink_tasks(w.ready_id, _prepared(w), board=WORLD_BOARD),
     lambda r: r is False),  # no edge existed: the op ran and answered, it did not reach for another agent
    ("request_review", lambda s, w: s.request_review(w.ready_id, board=WORLD_BOARD), lambda r: r is True),
    ("complete_task", lambda s, w: s.complete_task(w.ready_id, result="finished", board=WORLD_BOARD),
     lambda r: r is True),
    ("block_task", lambda s, w: s.block_task(w.ready_id, reason="waiting", board=WORLD_BOARD),
     lambda r: r is True),
    ("unblock_task",
     lambda s, w: s.unblock_task(w.ready_id, board=WORLD_BOARD),
     lambda r: r is True),
    ("archive_task", lambda s, w: s.archive_task(w.ready_id, board=WORLD_BOARD), lambda r: r is True),
    ("delete_task", lambda s, w: s.delete_task(w.ready_id, board=WORLD_BOARD), lambda r: r is True),
    ("claim_task", lambda s, w: s.claim_task(w.ready_id, board=WORLD_BOARD, claimer="claimer-a"),
     lambda r: r is not None),
    ("claim_review_task",
     lambda s, w: s.claim_review_task(w.review_id, board=WORLD_BOARD, claimer="reviewer-a"),
     lambda r: r is not None),
    ("heartbeat_claim",
     lambda s, w: s.heartbeat_claim(w.claimed_id, board=WORLD_BOARD, claimer=f"{w.agent}-worker"),
     lambda r: r is True),
    ("release_stale_claims", lambda s, w: s.release_stale_claims(board=WORLD_BOARD),
     lambda r: isinstance(r, int)),
    ("reclaim_task", lambda s, w: s.reclaim_task(w.claimed_id, board=WORLD_BOARD), lambda r: r is True),
    ("reassign_task", lambda s, w: s.reassign_task(w.ready_id, "profile-2", board=WORLD_BOARD),
     lambda r: r is True),
    ("recompute_ready", lambda s, w: s.recompute_ready(board=WORLD_BOARD),
     lambda r: isinstance(r, int)),
    ("add_notify_sub",
     lambda s, w: s.add_notify_sub(task_id=w.ready_id, platform="telegram", chat_id="chat-fresh",
                                   board=WORLD_BOARD),
     lambda r: r is None),
    ("remove_notify_sub",
     lambda s, w: s.remove_notify_sub(task_id=w.ready_id, platform="telegram",
                                       chat_id=f"{CHAT}-{w.agent}", board=WORLD_BOARD),
     lambda r: r is True),
    ("advance_notify_cursor",
     lambda s, w: s.advance_notify_cursor(task_id=w.ready_id, platform="telegram",
                                           chat_id=f"{CHAT}-{w.agent}", new_cursor=1,
                                           board=WORLD_BOARD),
     lambda r: r is None),
    ("claim_unseen_events_for_sub",
     lambda s, w: s.claim_unseen_events_for_sub(task_id=w.ready_id, platform="telegram",
                                                chat_id=f"{CHAT}-{w.agent}", board=WORLD_BOARD),
     lambda r: isinstance(r, tuple) and len(r) == 3),
    ("rewind_notify_cursor",
     lambda s, w: s.rewind_notify_cursor(task_id=w.ready_id, platform="telegram",
                                          chat_id=f"{CHAT}-{w.agent}", claimed_cursor=1,
                                          old_cursor=0, board=WORLD_BOARD),
     lambda r: isinstance(r, bool)),
    ("create_board", lambda s, w: s.create_board("fresh-board"), lambda r: isinstance(r, dict)),
    ("remove_board", lambda s, w: s.remove_board(TEMP_BOARD), lambda r: isinstance(r, dict)),
]

#: The positive control: every operation, read and write, once for its own agent.
OPERATIONS = [(name, call) for name, call, _ok in WRITES] + READS

#: What each read answers for a row the bound scope may not see: exactly what a missing one
#: gives (``get_task`` → ``None``, a listing → empty, an unseen-events probe → nothing new).
READ_MISSING: dict[str, Callable[[Any], bool]] = {
    "get_task": lambda r: r is None,
    "list_tasks": lambda r: r == [],
    "list_events": lambda r: r == [],
    "list_comments": lambda r: r == [],
    "list_boards": lambda r: r == [],
    "list_notify_subs": lambda r: r == [],
    "latest_run": lambda r: r is None,
    "list_runs": lambda r: r == [],
    "unseen_events_for_sub": lambda r: r == (0, []),
}

#: What the same read must answer for the seeded world it is served: the seeded rows
#: themselves, never the missing-row stand-in — an empty answer hides a populated board as
#: easily as a refusal the agent never sees (``READ_MISSING``), and the inverse matrix
#: compares the other agent's answer against the owning agent's own, so the owning answer
#: itself is asserted seeded here.
SEEDED_READ: dict[str, Callable[[Any, AgentWorld], bool]] = {
    "get_task": lambda r, w: r is not None and r.id == w.ready_id,
    "list_tasks": lambda r, w: {t.id for t in r} >= {w.ready_id, w.claimed_id, w.review_id},
    "list_events": lambda r, w: len(r) > 0,
    "list_comments": lambda r, w: [c.body for c in r] == [f"note for {w.agent}"],
    "list_boards": lambda r, w: {"default", WORLD_BOARD} <= {m["slug"] for m in r},
    "list_notify_subs": lambda r, w: (w.ready_id, "telegram", f"{CHAT}-{w.agent}") in
        {(s["task_id"], s["platform"], s["chat_id"]) for s in r},
    "latest_run": lambda r, w: r is not None and r.task_id == w.claimed_id,
    "list_runs": lambda r, w: len(r) > 0,
    "unseen_events_for_sub": lambda r, w: r != (0, []),
}

#: What a write row needs in place before its named operation runs, prepared under the owning
#: agent's own scope (never the foreign scope the operation itself runs under): a mutating
#: prerequisite inside the operation would refuse there first, and the row would pass without
#: ever invoking the operation it names. A row absent here needs nothing in place.
PREPARE_WRITE: dict[str, Callable[[KanbanStore, AgentWorld], Any]] = {
    "link_tasks": _prepare_card("child card"),
    "unlink_tasks": _prepare_card("unlink target"),
    "unblock_task": lambda s, w: s.block_task(w.ready_id, reason="blocked to unblock",
                                              board=WORLD_BOARD),
    "remove_board": lambda s, w: s.create_board(TEMP_BOARD),
}


def _shared_board_root(store: KanbanStore) -> bool:
    """Whether the backend declares the one local exception on its own class
    (``KanbanStore.LOCAL_SHARED_BOARD``): a single-user machine's one board root, shared by
    every profile. Never a fixture option — a backend that wants the inverse matrix says so
    where its own code lives, and ``resolve_kanban_storage`` refuses it in hosted mode."""
    return bool(getattr(type(store), "LOCAL_SHARED_BOARD", False))


class PinningConformance:
    """The store serves only the pins it declares, and a refusal changes nothing."""

    @pytest.fixture
    def world(self, backend):
        return make_world(backend)

    @pytest.mark.parametrize("name,operation", READS, ids=[name for name, _ in READS])
    def test_every_read_is_refused_for_a_tenant_the_store_does_not_serve(self, backend, world, name, operation):
        w = world["a"]
        with backend.foreign_tenant("a"):
            _refused(lambda: operation(w.store, w))

    @pytest.mark.parametrize("name,operation", [(n, c) for n, c, _ in WRITES], ids=[n for n, _, _ in WRITES])
    def test_every_write_is_refused_for_a_foreign_tenant_and_changes_nothing(
            self, backend, world, name, operation):
        w = world["a"]
        with backend.bind("a"):  # the row's preparation, under the agent's own scope
            prepare = PREPARE_WRITE.get(name)
            if prepare is not None:
                prepare(w.store, w)
        before = observe(w.store, w)
        with backend.foreign_tenant("a"):
            _refused(lambda: operation(w.store, w))
        assert observe(w.store, w) == before

    def test_unpinned_calls_behave_as_the_backend_declares(self, backend, world):
        w = world["a"]
        with backend.unbound():
            if backend.serves_unbound:
                # The ambient caller of a single-user machine is served, as today's CLI is.
                assert w.store.get_task(w.ready_id, board=WORLD_BOARD) is not None
                fresh = w.store.create_task(title="ambient card", board=WORLD_BOARD)
                assert w.store.get_task(fresh, board=WORLD_BOARD) is not None
            else:
                # A backend of a shared process serves nobody without a scope.
                _refused(lambda: w.store.get_task(w.ready_id, board=WORLD_BOARD))
                _refused(lambda: w.store.create_task(title="no pin", board=WORLD_BOARD))

    @pytest.mark.parametrize("name,operation", OPERATIONS, ids=[name for name, _ in OPERATIONS])
    def test_every_operation_succeeds_for_its_own_agent(self, backend, world, name, operation):
        """The positive control: with its own agent bound, every operation of the store runs.
        The mutant harness runs exactly this test (plus the world setup) against each broken
        backend first: a backend that cannot serve its own agent is a broken harness, not a
        hole the suite gets credit for finding."""
        w = world["a"]
        store = backend.open(w.agent)  # the environment, if any, back at this agent's root
        with backend.bind(w.agent):
            prepare = PREPARE_WRITE.get(name)
            if prepare is not None:
                prepare(store, w)
            result = operation(store, w)
        checks = {name: ok for name, _c, ok in WRITES}
        assert (checks.get(name, lambda r: r is not None))(result)


class IsolationConformance:
    """Another agent's work is invisible and unreachable, and a claim is one claimant's."""

    @pytest.fixture
    def world(self, backend):
        return make_world(backend)

    def test_b_task_is_never_dispatched_into_a(self, backend, world):
        a, b = world["a"], world["b"]
        with backend.bind("a"):
            assert a.store.claim_task(b.ready_id, board=WORLD_BOARD) is None
            assert a.store.claim_review_task(b.review_id, board=WORLD_BOARD) is None
            assert a.store.heartbeat_claim(b.claimed_id, board=WORLD_BOARD) is False
            assert a.store.reclaim_task(b.claimed_id, board=WORLD_BOARD) is False
            assert a.store.reassign_task(b.claimed_id, "profile-a", board=WORLD_BOARD) is False
        with backend.bind("b"):
            # Nothing of b's moved: the ready card is still ready and still b's to claim.
            assert b.store.get_task(b.ready_id, board=WORLD_BOARD).status == "ready"
            assert b.store.claim_task(b.ready_id, board=WORLD_BOARD, claimer="b-worker") is not None

    def test_a_foreign_task_is_indistinguishable_from_a_missing_one(self, backend, world):
        a, b = world["a"], world["b"]
        with backend.bind("a"):
            foreign = a.store.get_task(b.ready_id, board=WORLD_BOARD)
            missing = a.store.get_task("t_nonexistent", board=WORLD_BOARD)
            assert foreign is None and missing is None
            assert a.store.list_events(b.ready_id, board=WORLD_BOARD) == \
                a.store.list_events("t_nonexistent", board=WORLD_BOARD)
            assert a.store.list_comments(b.ready_id, board=WORLD_BOARD) == \
                a.store.list_comments("t_nonexistent", board=WORLD_BOARD)
            assert a.store.claim_task(b.ready_id, board=WORLD_BOARD) is None
            assert a.store.latest_run(b.ready_id, board=WORLD_BOARD) is None

    def test_a_sweep_pinned_to_a_never_releases_bs_expired_claim(self, backend, world):
        a, b = world["a"], world["b"]
        with backend.bind("b"):
            b.store.claim_task(b.ready_id, board=WORLD_BOARD, ttl_seconds=1, claimer="b-slow-worker")
        expires = b.store.get_task(b.ready_id, board=WORLD_BOARD).claim_expires
        # The sweep reads whole seconds: the claim must be strictly in the past by its clock.
        assert _wait_until(lambda: int(time.time()) > (b.store.get_task(
            b.ready_id, board=WORLD_BOARD).claim_expires or 0)), f"claim did not expire by {expires}"
        with backend.bind("a"):
            assert a.store.release_stale_claims(board=WORLD_BOARD) == 0
            assert a.store.recompute_ready(board=WORLD_BOARD) == 0
        with backend.bind("b"):
            assert b.store.get_task(b.ready_id, board=WORLD_BOARD).status == "running"
            assert b.store.release_stale_claims(board=WORLD_BOARD) == 1

    def test_a_worker_cannot_leave_for_a_board_that_is_not_its_agents(self, backend, world):
        a, b = world["a"], world["b"]
        with backend.bind("a"):
            boards = sorted(m["slug"] for m in a.store.list_boards())
            assert WORLD_BOARD in boards and FOREIGN_BOARD not in boards
            # b's private board behaves exactly like one that was never created: every way in
            # answers with today's "no such board" refusal, never with b's rows.
            for slug in (FOREIGN_BOARD, "board-never-created"):
                with pytest.raises(ValueError):
                    a.store.get_task(b.private_id, board=slug)
                with pytest.raises(ValueError):
                    a.store.claim_task(b.private_id, board=slug)
                with pytest.raises(ValueError):
                    a.store.create_task(title="escape attempt", board=slug)

    def test_b_notify_targets_are_invisible_to_a(self, backend, world):
        a, b = world["a"], world["b"]
        with backend.bind("b"):
            b.store.add_notify_sub(task_id=b.claimed_id, platform="telegram", chat_id="b-secret-chat",
                                   board=WORLD_BOARD)
        with backend.bind("a"):
            subs = a.store.list_notify_subs(board=WORLD_BOARD)
            assert all(s["chat_id"] != "b-secret-chat" for s in subs)
            # b's subscription, named outright, answers like a subscription that does not exist.
            assert a.store.list_notify_subs(b.ready_id, board=WORLD_BOARD) == []
            assert a.store.unseen_events_for_sub(task_id=b.ready_id, platform="telegram",
                                                 chat_id=f"{CHAT}-b", board=WORLD_BOARD) == (0, [])
            assert a.store.claim_unseen_events_for_sub(task_id=b.ready_id, platform="telegram",
                                                       chat_id=f"{CHAT}-b", board=WORLD_BOARD) == (0, 0, [])
            assert a.store.remove_notify_sub(task_id=b.ready_id, platform="telegram",
                                              chat_id=f"{CHAT}-b", board=WORLD_BOARD) is False

    def test_at_most_one_replica_claims_a_ready_task(self, backend, world):
        a = world["a"]
        replica = backend.replica("a")
        winners: list = []
        barrier = threading.Barrier(2)

        def claim(store):
            with backend.bind("a"):
                barrier.wait(timeout=20)
                won = store.claim_task(a.ready_id, board=WORLD_BOARD, claimer=f"replica-{store is replica}")
                if won is not None:
                    winners.append(store)

        threads = [_replica_thread(claim, a.store), _replica_thread(claim, replica)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert not any(thread.is_alive() for thread in threads)
        assert len(winners) == 1, "two replicas both claimed the same ready task"


class CrossAgentConformance:
    """A's handle under a bound same-tenant scope of another agent (B's worker).

    The gap between "the tenant is mine" and "the agent is mine": a backend whose pin checks
    the tenant alone would pass the foreign-tenant matrix and still let B's worker read, claim
    and rewrite A's rows through A's handle. Here that backend fails: every operation of A's
    handle under B's scope is refused — a read may instead answer exactly what a missing row
    would — and changes nothing A can observe, unless the backend declares the one local
    exception on its own class (:func:`_shared_board_root`), when the matrix asserts the
    inverse: B's scope is served on A's handle, as today's dispatcher handoff needs.
    """

    @pytest.fixture
    def world(self, backend):
        return make_world(backend)

    @pytest.mark.parametrize("name,operation", READS, ids=[name for name, _ in READS])
    def test_every_read_is_refused_or_missing_under_a_same_tenant_other_agents_scope(
            self, backend, world, name, operation):
        """The cross-agent read: B's worker holding A's handle never sees A's rows — the
        operation is refused, or answers exactly what a missing row would — unless the backend
        declares its one board root shared, when B's scope reads A's rows like A's own:
        exactly A's own answer, and never the missing-row stand-in an empty-handed backend
        would hide behind."""
        a = world["a"]
        if _shared_board_root(a.store):
            with backend.bind("a"):
                own = operation(a.store, a)
            assert SEEDED_READ[name](own, a), (
                f"{name} answered A's own read without the world's seeded rows")
            with backend.bind("b"):
                under_b = operation(a.store, a)
            assert under_b == own, (
                f"{name} under a same-tenant other agent's scope did not answer A's own read")
            return
        with backend.bind("b"):
            try:
                result = operation(a.store, a)
            except KanbanPinRefused:
                return
            assert READ_MISSING[name](result), (
                f"{name} under a same-tenant other agent's scope answered with A's rows")

    @pytest.mark.parametrize("name,operation", [(n, c) for n, c, _ in WRITES],
                             ids=[n for n, _, _ in WRITES])
    def test_every_write_is_refused_under_a_same_tenant_other_agents_scope_and_changes_nothing(
            self, backend, world, name, operation):
        """The cross-agent write: unless the backend declares its one board root shared (when
        B's scope writes A's rows like A's own), the operation is refused and nothing A can
        observe changes — no row moves, no event lands, no subscription or cursor does. The
        row's own preparation runs under A's scope first, so the refusal the test pins is the
        named operation's, never a prerequisite's that fired before it; A observes its world
        unchanged afterwards, the preparation included."""
        a = world["a"]
        shared = _shared_board_root(a.store)
        with backend.bind("a"):
            prepare = PREPARE_WRITE.get(name)
            if prepare is not None:
                prepare(a.store, a)
            before = observe(a.store, a)
        with backend.bind("b"):
            if shared:
                result = operation(a.store, a)
            else:
                _refused(lambda: operation(a.store, a))
        if shared:
            checks = {name: ok for name, _c, ok in WRITES}
            assert checks[name](result)
        else:
            with backend.bind("a"):
                assert observe(a.store, a) == before


class KanbanConformance(PinningConformance, IsolationConformance, CrossAgentConformance):
    """The whole suite: a backend plugin subclasses this and supplies the ``backend`` fixture."""
